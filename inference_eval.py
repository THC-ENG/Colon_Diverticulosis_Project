import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.res_swin_unet import ResSwinUNet
from utils.augmentations import ValAugmentor
from utils.dataset import ColonDataset
from utils.metrics import dice_coeff, iou_score


def save_comparison_grid(
    image_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    pred_prob: torch.Tensor,
    path: Path,
    pred_threshold: float = 0.5,
):
    img_np = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gt_np = (gt_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)

    pred_np = (pred_prob.squeeze().cpu().numpy() > pred_threshold).astype(np.uint8) * 255

    overlay = img_bgr.copy()
    overlay[gt_np == 255] = [0, 255, 0]
    overlay[pred_np == 255] = [0, 0, 255]
    overlay[(gt_np == 255) & (pred_np == 255)] = [0, 255, 255]

    grid = np.hstack((img_bgr, gt_bgr, overlay))
    cv2.imwrite(str(path), grid)


def _load_checkpoint(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _morph_boundary_from_mask(mask_bin: np.ndarray, radius: int = 1) -> np.ndarray:
    if mask_bin.sum() == 0:
        return np.zeros_like(mask_bin, dtype=np.float32)

    radius = max(1, int(radius))
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(mask_bin, kernel, iterations=1)
    ero = cv2.erode(mask_bin, kernel, iterations=1)
    return ((dil - ero) > 0).astype(np.float32)


def _boundary_from_seg_prob(seg_prob: torch.Tensor, threshold: float, radius: int) -> torch.Tensor:
    seg_np = (seg_prob.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    boundary_np = _morph_boundary_from_mask(seg_np, radius=radius)
    return torch.from_numpy(boundary_np).to(seg_prob.device)


def _remap_legacy_head_keys(state_dict: dict) -> tuple[dict, bool]:
    if not isinstance(state_dict, dict):
        return state_dict, False

    remapped = dict(state_dict)
    changed = False

    if "head.weight" in remapped and "seg_head.weight" not in remapped:
        remapped["seg_head.weight"] = remapped.pop("head.weight")
        changed = True
    if "head.bias" in remapped and "seg_head.bias" not in remapped:
        remapped["seg_head.bias"] = remapped.pop("head.bias")
        changed = True

    return remapped, changed


def _load_model_state(model: torch.nn.Module, state_dict: dict):
    state_dict, remapped = _remap_legacy_head_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    seg_missing = [k for k in missing if k.startswith("seg_head.")]
    if seg_missing:
        raise RuntimeError(
            "Checkpoint is missing segmentation head weights. "
            f"Missing keys: {seg_missing}"
        )

    boundary_head_ready = True
    if model.use_boundary:
        boundary_missing = [k for k in missing if k.startswith("boundary_head.")]
        boundary_head_ready = len(boundary_missing) == 0
        if not boundary_head_ready:
            print(
                "[checkpoint warning] boundary_head weights are missing. "
                "Boundary map will fallback to morphology boundary from predicted segmentation."
            )

    if remapped:
        print("[checkpoint] mapped legacy keys head.* -> seg_head.*")
    if missing or unexpected:
        print(f"[checkpoint] missing_keys={missing}")
        print(f"[checkpoint] unexpected_keys={unexpected}")

    return boundary_head_ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="data/processed_images/images/test")
    parser.add_argument("--mask-dir", type=str, default="data/processed_images/masks/test")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="results/preds_comparison")
    parser.add_argument("--report-path", type=str, default="results/metrics_report.json")
    parser.add_argument("--pred-threshold", type=float, default=0.5)

    parser.add_argument("--use-boundary", action="store_true")
    parser.add_argument("--save-boundary", action="store_true")
    parser.add_argument("--boundary-dir", type=str, default="results/boundary_maps")
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--boundary-save-mode", type=str, choices=["binary", "heatmap"], default="binary")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(Path(args.report_path).parent, exist_ok=True)
    if args.save_boundary:
        os.makedirs(args.boundary_dir, exist_ok=True)

    dataset = ColonDataset(
        args.image_dir,
        args.mask_dir,
        transform=ValAugmentor((args.img_size, args.img_size)),
        use_boundary=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ckpt = _load_checkpoint(args.checkpoint, args.device)
    ckpt_use_boundary = bool(ckpt.get("use_boundary", False)) if isinstance(ckpt, dict) else False
    effective_use_boundary = args.use_boundary or ckpt_use_boundary

    model = ResSwinUNet(num_classes=1, use_boundary=effective_use_boundary).to(args.device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    boundary_head_ready = _load_model_state(model, state_dict)
    model.eval()

    print(f"[eval mode] {'dual_task_boundary' if effective_use_boundary else 'baseline'}")

    dices, ious = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            ids = batch["id"]

            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                seg_logits, boundary_logits = outputs
            else:
                seg_logits = outputs
                boundary_logits = None

            d = dice_coeff(seg_logits, masks).item()
            i = iou_score(seg_logits, masks).item()
            dices.append(d)
            ious.append(i)

            seg_probs = torch.sigmoid(seg_logits)

            if boundary_logits is not None and boundary_head_ready:
                boundary_probs = torch.sigmoid(boundary_logits)
            else:
                boundary_list = [
                    _boundary_from_seg_prob(seg_probs[j], args.pred_threshold, args.boundary_radius)
                    for j in range(seg_probs.size(0))
                ]
                boundary_probs = torch.stack(boundary_list, dim=0).unsqueeze(1)

            for j, sample_id in enumerate(ids):
                save_comparison_grid(
                    image_tensor=images[j],
                    gt_tensor=masks[j],
                    pred_prob=seg_probs[j],
                    path=Path(args.save_dir) / f"{sample_id}_eval.png",
                    pred_threshold=args.pred_threshold,
                )

                if args.save_boundary:
                    boundary_np = boundary_probs[j].squeeze().cpu().numpy()
                    if args.boundary_save_mode == "binary":
                        boundary_map = (boundary_np > args.boundary_threshold).astype(np.uint8) * 255
                    else:
                        boundary_map = (np.clip(boundary_np, 0.0, 1.0) * 255.0).astype(np.uint8)
                    cv2.imwrite(str(Path(args.boundary_dir) / f"{sample_id}_boundary.png"), boundary_map)

    report = {
        "num_samples": len(dataset),
        "dice_mean": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
    }

    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Evaluation Report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[Visualizations saved to] {args.save_dir}")
    if args.save_boundary:
        print(f"[Boundary maps saved to] {args.boundary_dir}")
    print(f"[Report saved to] {args.report_path}")


if __name__ == "__main__":
    main()
