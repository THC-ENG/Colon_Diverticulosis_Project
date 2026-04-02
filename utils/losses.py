import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * (1 - pt).pow(self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class FocalWithDiceLoss(nn.Module):
    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float | None = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = (1.0 - focal_weight) if dice_weight is None else dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return total_loss, focal_loss.detach(), dice_loss.detach()


class DualTaskLoss(nn.Module):
    def __init__(
        self,
        seg_focal_weight: float = 0.7,
        seg_dice_weight: float = 0.3,
        boundary_weight: float = 0.3,
        boundary_bce_weight: float = 0.7,
        boundary_dice_weight: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.seg_focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.seg_dice = DiceLoss()
        self.boundary_bce = nn.BCEWithLogitsLoss()
        self.boundary_dice = DiceLoss()

        self.seg_focal_weight = seg_focal_weight
        self.seg_dice_weight = seg_dice_weight
        self.boundary_weight = boundary_weight
        self.boundary_bce_weight = boundary_bce_weight
        self.boundary_dice_weight = boundary_dice_weight

    def forward(
        self,
        seg_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        boundary_targets: torch.Tensor,
        boundary_weight_override: float | None = None,
    ):
        seg_focal_loss = self.seg_focal(seg_logits, seg_targets)
        seg_dice_loss = self.seg_dice(seg_logits, seg_targets)
        seg_total = self.seg_focal_weight * seg_focal_loss + self.seg_dice_weight * seg_dice_loss

        boundary_bce_loss = self.boundary_bce(boundary_logits, boundary_targets)
        boundary_dice_loss = self.boundary_dice(boundary_logits, boundary_targets)
        boundary_total = (
            self.boundary_bce_weight * boundary_bce_loss
            + self.boundary_dice_weight * boundary_dice_loss
        )

        boundary_weight = self.boundary_weight if boundary_weight_override is None else boundary_weight_override
        total_loss = seg_total + boundary_weight * boundary_total

        stats = {
            "seg_focal": seg_focal_loss.detach(),
            "seg_dice": seg_dice_loss.detach(),
            "seg_total": seg_total.detach(),
            "boundary_bce": boundary_bce_loss.detach(),
            "boundary_dice": boundary_dice_loss.detach(),
            "boundary_total": boundary_total.detach(),
            "boundary_weight": torch.tensor(boundary_weight, device=seg_logits.device),
        }
        return total_loss, stats


def _reduce_per_sample(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1).mean(dim=1)


def _focal_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(targets > 0.5, alpha, 1.0 - alpha)
    return _reduce_per_sample(alpha_t * (1.0 - pt).pow(gamma) * bce)


def _dice_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits).reshape(logits.shape[0], -1)
    t = targets.reshape(targets.shape[0], -1)
    inter = (probs * t).sum(dim=1)
    denom = probs.sum(dim=1) + t.sum(dim=1)
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)


def _sobel_edge_map(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError(f"Expected tensor shape [B,1,H,W], got {tuple(x.shape)}")
    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(1)
    ky = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(1)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return torch.clamp(edge, 0.0, 1.0)


def _safe_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(weights.sum(), min=1e-6)
    return (values * weights).sum() / denom


class StudentCompositeLoss(nn.Module):
    def __init__(
        self,
        lambda_sup: float = 1.0,
        lambda_pseudo: float = 0.4,
        lambda_edge: float = 0.2,
        lambda_distill: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        seg_focal_weight: float = 0.7,
        seg_dice_weight: float = 0.3,
        distill_temperature: float = 2.0,
        schedule_milestones: tuple[float, float, float] = (0.2, 0.6, 0.8),
    ):
        super().__init__()
        self.lambda_sup = float(lambda_sup)
        self.lambda_pseudo = float(lambda_pseudo)
        self.lambda_edge = float(lambda_edge)
        self.lambda_distill = float(lambda_distill)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.seg_focal_weight = float(seg_focal_weight)
        self.seg_dice_weight = float(seg_dice_weight)
        self.distill_temperature = float(distill_temperature)
        self.m1, self.m2, self.m3 = schedule_milestones

    def _weights_for_ratio(self, epoch_ratio: float) -> tuple[float, float, float, float]:
        r = float(max(0.0, min(1.0, epoch_ratio)))
        if r < self.m1:
            return self.lambda_sup, 0.0, 0.0, 0.0
        if r < self.m2:
            t = (r - self.m1) / max(1e-6, self.m2 - self.m1)
            return self.lambda_sup, self.lambda_pseudo * t, self.lambda_edge * t, 0.0
        if r < self.m3:
            return self.lambda_sup, self.lambda_pseudo, self.lambda_edge, 0.0
        t = (r - self.m3) / max(1e-6, 1.0 - self.m3)
        return self.lambda_sup, self.lambda_pseudo, self.lambda_edge, self.lambda_distill * t

    def forward(
        self,
        seg_logits: torch.Tensor,
        masks: torch.Tensor,
        is_labeled: torch.Tensor,
        is_pseudo: torch.Tensor,
        pseudo_weight: torch.Tensor,
        distill_soft: torch.Tensor,
        distill_edge: torch.Tensor,
        has_distill_soft: torch.Tensor,
        has_distill_edge: torch.Tensor,
        epoch_ratio: float,
    ) -> tuple[torch.Tensor, dict]:
        lw_sup, lw_pseudo, lw_edge, lw_distill = self._weights_for_ratio(epoch_ratio)

        focal_ps = _focal_per_sample(
            seg_logits,
            masks,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        dice_ps = _dice_per_sample(seg_logits, masks)
        seg_ps = self.seg_focal_weight * focal_ps + self.seg_dice_weight * dice_ps

        sup_sel = (is_labeled > 0.5) & (is_pseudo < 0.5)
        pseudo_sel = is_pseudo > 0.5
        pseudo_w = torch.clamp(pseudo_weight, min=0.0)

        l_sup = torch.zeros((), dtype=seg_logits.dtype, device=seg_logits.device)
        if sup_sel.any():
            l_sup = seg_ps[sup_sel].mean()

        l_pseudo = torch.zeros((), dtype=seg_logits.dtype, device=seg_logits.device)
        if pseudo_sel.any():
            l_pseudo = _safe_weighted_mean(seg_ps[pseudo_sel], torch.clamp(pseudo_w[pseudo_sel], min=1e-6))

        pred_prob = torch.sigmoid(seg_logits)
        pred_edge = _sobel_edge_map(pred_prob)
        target_edge = _sobel_edge_map(masks)
        edge_ps = _reduce_per_sample(F.l1_loss(pred_edge, target_edge, reduction="none"))

        edge_sel = (is_labeled > 0.5) | (is_pseudo > 0.5)
        edge_w = torch.where(is_pseudo > 0.5, torch.clamp(pseudo_w, min=1e-6), torch.ones_like(pseudo_w))
        l_edge = torch.zeros((), dtype=seg_logits.dtype, device=seg_logits.device)
        if edge_sel.any():
            l_edge = _safe_weighted_mean(edge_ps[edge_sel], edge_w[edge_sel])

        l_soft = torch.zeros((), dtype=seg_logits.dtype, device=seg_logits.device)
        soft_sel = has_distill_soft > 0.5
        if soft_sel.any():
            t = max(1e-6, self.distill_temperature)
            s_prob = torch.sigmoid(seg_logits / t)
            teacher_soft = torch.clamp(distill_soft, min=1e-4, max=1.0 - 1e-4)
            soft_bce = -(teacher_soft * torch.log(torch.clamp(s_prob, min=1e-6)) +
                         (1.0 - teacher_soft) * torch.log(torch.clamp(1.0 - s_prob, min=1e-6)))
            soft_ps = _reduce_per_sample(soft_bce)
            l_soft = soft_ps[soft_sel].mean()

        l_edge_distill = torch.zeros((), dtype=seg_logits.dtype, device=seg_logits.device)
        edge_distill_sel = has_distill_edge > 0.5
        if edge_distill_sel.any():
            edge_distill_ps = _reduce_per_sample(F.l1_loss(pred_edge, distill_edge, reduction="none"))
            l_edge_distill = edge_distill_ps[edge_distill_sel].mean()

        l_distill = l_soft + l_edge_distill
        total = lw_sup * l_sup + lw_pseudo * l_pseudo + lw_edge * l_edge + lw_distill * l_distill

        stats = {
            "loss_sup": l_sup.detach(),
            "loss_pseudo": l_pseudo.detach(),
            "loss_edge": l_edge.detach(),
            "loss_distill_soft": l_soft.detach(),
            "loss_distill_edge": l_edge_distill.detach(),
            "loss_distill": l_distill.detach(),
            "lambda_sup": torch.tensor(lw_sup, device=seg_logits.device),
            "lambda_pseudo": torch.tensor(lw_pseudo, device=seg_logits.device),
            "lambda_edge": torch.tensor(lw_edge, device=seg_logits.device),
            "lambda_distill": torch.tensor(lw_distill, device=seg_logits.device),
            "seg_focal": focal_ps.mean().detach(),
            "seg_dice": dice_ps.mean().detach(),
        }
        return total, stats
