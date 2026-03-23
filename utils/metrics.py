import cv2
import numpy as np
import torch


def _binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float()


def dice_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    preds = _binarize(logits, threshold=threshold).reshape(logits.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    return torch.where(
        union == 0,
        torch.ones_like(union),
        (2.0 * inter + eps) / (union + eps),
    )


def iou_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    preds = _binarize(logits, threshold=threshold).reshape(logits.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    return torch.where(
        union == 0,
        torch.ones_like(union),
        (inter + eps) / (union + eps),
    )


def dice_coeff(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return dice_per_sample(logits, targets, threshold=0.5, eps=eps).mean()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return iou_per_sample(logits, targets, threshold=0.5, eps=eps).mean()


def mask_to_boundary(mask_bin: np.ndarray, radius: int = 1) -> np.ndarray:
    mask_bin = (mask_bin > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return np.zeros_like(mask_bin, dtype=np.uint8)

    radius = max(1, int(radius))
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    eroded = cv2.erode(mask_bin, kernel, iterations=1)
    return ((dilated - eroded) > 0).astype(np.uint8)


def boundary_f1_from_masks(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    boundary_radius: int = 1,
    eps: float = 1e-6,
) -> float:
    pred_boundary = mask_to_boundary(pred_mask, radius=boundary_radius)
    gt_boundary = mask_to_boundary(gt_mask, radius=boundary_radius)

    pred_sum = float(pred_boundary.sum())
    gt_sum = float(gt_boundary.sum())

    if pred_sum == 0.0 and gt_sum == 0.0:
        return 1.0
    if pred_sum == 0.0 or gt_sum == 0.0:
        return 0.0

    kernel_size = 2 * boundary_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    gt_dil = cv2.dilate(gt_boundary, kernel, iterations=1)
    pred_dil = cv2.dilate(pred_boundary, kernel, iterations=1)

    precision = float((pred_boundary & gt_dil).sum()) / (pred_sum + eps)
    recall = float((gt_boundary & pred_dil).sum()) / (gt_sum + eps)
    return float((2.0 * precision * recall) / (precision + recall + eps))


def _surface_points(mask_bin: np.ndarray) -> np.ndarray:
    boundary = mask_to_boundary(mask_bin, radius=1)
    points = np.argwhere(boundary > 0)
    if points.size == 0:
        points = np.argwhere(mask_bin > 0)
    return points.astype(np.float32)


def _pairwise_min_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    diffs = src[:, None, :] - dst[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    return dists.min(axis=1)


def hd95_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0

    fallback = float(np.hypot(pred.shape[0], pred.shape[1]))
    if pred.sum() == 0 or gt.sum() == 0:
        return fallback

    pred_points = _surface_points(pred)
    gt_points = _surface_points(gt)
    if pred_points.size == 0 or gt_points.size == 0:
        return fallback

    d_pred_to_gt = _pairwise_min_distances(pred_points, gt_points)
    d_gt_to_pred = _pairwise_min_distances(gt_points, pred_points)
    return float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))
