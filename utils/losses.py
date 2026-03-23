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
