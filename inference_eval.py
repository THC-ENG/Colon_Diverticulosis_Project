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
from utils.data_protocol import load_protocol_samples, summarize_samples, validate_protocol_samples
from utils.dataset import ColonDataset, ProtocolSegDataset
from utils.metrics import (
    boundary_f1_from_masks,
    dice_per_sample,
    hd95_from_masks,
    iou_per_sample,
    mask_to_boundary,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _parse_model_outputs(outputs):
    parsed = {
        "seg": None,
        "boundary": None,
        "aux": [],
    }

    if isinstance(outputs, dict):
        parsed["seg"] = outputs.get("seg")
        parsed["boundary"] = outputs.get("boundary")
        parsed["aux"] = outputs.get("aux") or []
        return parsed

    if isinstance(outputs, (tuple, list)):
        if len(outputs) >= 1:
            parsed["seg"] = outputs[0]
        if len(outputs) >= 2:
            parsed["boundary"] = outputs[1]
        if len(outputs) >= 3:
            parsed["aux"] = outputs[2] if outputs[2] is not None else []
        return parsed

    parsed["seg"] = outputs
    return parsed


def _parse_threshold_range(text: str) -> list[float]:
    items = [v.strip() for v in text.split(",") if v.strip()]
    if len(items) != 3:
        raise ValueError("--threshold-range must be in format: start,end,step")

    start, end, step = map(float, items)
    if step <= 0:
        raise ValueError("threshold step must be > 0")
    if end < start:
        raise ValueError("threshold end must be >= start")

    values = []
    cur = start
    while cur <= end + 1e-8:
        values.append(float(round(cur, 4)))
        cur += step
    return values


def _to_mask_np(mask_tensor: torch.Tensor) -> np.ndarray:
    return (mask_tensor.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8)


def save_comparison_grid(
    image_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    pred_prob: torch.Tensor,
    path: Path,
    pred_threshold: float = 0.5,
):
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)
    img_bgr = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    gt_np = (gt_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)

    pred_np = (pred_prob.squeeze().cpu().numpy() > pred_threshold).astype(np.uint8) * 255

    overlay = img_bgr.copy()
    overlay[gt_np == 255] = [0, 255, 0]
    overlay[pred_np == 255] = [0, 0, 255]
    overlay[(gt_np == 255) & (pred_np == 255)] = [0, 255, 255]

    grid = np.hstack((img_bgr, gt_bgr, overlay))
    cv2.imwrite(str(path), grid)


def save_boundary_probability_heatmap(
    boundary_prob: torch.Tensor,
    path: Path,
    upscale: int = 4,
):
    prob = boundary_prob.squeeze().detach().cpu().numpy().astype(np.float32)
    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)

    lo, hi = np.percentile(prob, (1.0, 99.0))
    if float(hi - lo) < 1e-8:
        lo = float(prob.min())
        hi = float(prob.max())

    if float(hi - lo) < 1e-8:
        norm_u8 = np.zeros_like(prob, dtype=np.uint8)
    else:
        norm = np.clip((prob - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        norm_u8 = (norm * 255.0).astype(np.uint8)

    heatmap = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)

    if upscale > 1:
        h, w = heatmap.shape[:2]
        heatmap = cv2.resize(
            heatmap,
            (w * upscale, h * upscale),
            interpolation=cv2.INTER_CUBIC,
        )

    cv2.imwrite(str(path), heatmap)


def _load_checkpoint(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        t = value.strip().lower()
        if t in {"1", "true", "yes", "y", "on"}:
            return True
        if t in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def _model_kwargs_from_checkpoint(ckpt: dict | object) -> dict:
    if not isinstance(ckpt, dict):
        return {}

    model_kwargs = ckpt.get("model_kwargs", {})
    if isinstance(model_kwargs, dict) and model_kwargs:
        return dict(model_kwargs)

    # Backward compatibility: older train checkpoints only save `args`.
    args = ckpt.get("args", {})
    if not isinstance(args, dict):
        return {}

    inferred = {
        "num_classes": int(args.get("num_classes", 1)),
        "use_boundary": _as_bool(args.get("use_boundary", False), False),
        "norm_type": str(args.get("norm_type", "bn")),
        "deep_supervision": _as_bool(args.get("deep_supervision", False), False),
        "window_size": int(args.get("window_size", 8)),
        "use_shift_mask": _as_bool(args.get("use_shift_mask", True), True),
        "use_rel_pos_bias": _as_bool(args.get("use_rel_pos_bias", True), True),
        "pad_to_window": _as_bool(args.get("pad_to_window", True), True),
        "use_wavelet_bottleneck": _as_bool(args.get("use_wavelet_bottleneck", False), False),
    }
    return inferred


def _boundary_from_seg_prob(seg_prob: torch.Tensor, threshold: float, radius: int) -> torch.Tensor:
    seg_np = (seg_prob.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    boundary_np = mask_to_boundary(seg_np, radius=radius).astype(np.float32)
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


def _build_loader(image_dir: str, mask_dir: str, img_size: int, batch_size: int, num_workers: int):
    dataset = ColonDataset(
        image_dir,
        mask_dir,
        transform=ValAugmentor((img_size, img_size)),
        use_boundary=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader


def _build_loader_from_protocol(
    data_manifest: str,
    data_root: str,
    manifest_mode: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
):
    samples = load_protocol_samples(
        data_manifest=data_manifest or None,
        data_root=data_root or None,
        manifest_mode=manifest_mode,
    )
    validate_protocol_samples(samples)
    print(f"[eval protocol] {json.dumps(summarize_samples(samples), ensure_ascii=False)}")

    test_rows = [
        s
        for s in samples
        if (s.subset == "external" and s.split in {"test", ""})
    ]
    if not test_rows:
        test_rows = [s for s in samples if s.split == "test" and s.mask_path]
    if not test_rows:
        raise RuntimeError("No test rows found in protocol manifest.")

    dataset = ProtocolSegDataset(
        test_rows,
        transform=ValAugmentor((img_size, img_size)),
        use_boundary=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader


def _search_best_threshold(model, loader, device: str, thresholds: list[float]) -> float:
    scores = {th: [] for th in thresholds}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Threshold Search"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            parsed = _parse_model_outputs(model(images))
            seg_logits = parsed["seg"]
            for th in thresholds:
                d = dice_per_sample(seg_logits, masks, threshold=th)
                scores[th].extend(d.cpu().tolist())

    best_th = max(thresholds, key=lambda t: float(np.mean(scores[t])) if scores[t] else -1.0)
    return float(best_th)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=str, default="")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument(
        "--manifest-mode",
        type=str,
        default="prefer",
        choices=["prefer", "only", "off"],
    )
    parser.add_argument("--image-dir", type=str, default="data/processed_images/images/test")
    parser.add_argument("--mask-dir", type=str, default="data/processed_images/masks/test")
    parser.add_argument("--val-image-dir", type=str, default="")
    parser.add_argument("--val-mask-dir", type=str, default="")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="results/preds_comparison")
    parser.add_argument("--report-path", type=str, default="results/metrics_report.json")
    parser.add_argument("--per-sample-report", type=str, default="")
    parser.add_argument("--pred-threshold", type=float, default=0.5)

    parser.add_argument("--threshold-search", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--threshold-range", type=str, default="0.30,0.70,0.02")
    parser.add_argument("--report-boundary-metrics", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--use-boundary", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-boundary", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--boundary-dir", type=str, default="results/boundary_maps")
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--boundary-heatmap-scale", type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(Path(args.report_path).parent, exist_ok=True)
    if args.save_boundary:
        os.makedirs(args.boundary_dir, exist_ok=True)

    use_protocol = bool(args.data_manifest or args.data_root) and args.manifest_mode != "off"
    if use_protocol:
        test_dataset, test_loader = _build_loader_from_protocol(
            data_manifest=args.data_manifest,
            data_root=args.data_root,
            manifest_mode=args.manifest_mode,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        test_dataset, test_loader = _build_loader(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    ckpt = _load_checkpoint(args.checkpoint, args.device)
    ckpt_use_boundary = bool(ckpt.get("use_boundary", False)) if isinstance(ckpt, dict) else False
    model_kwargs = _model_kwargs_from_checkpoint(ckpt)

    effective_use_boundary = args.use_boundary or ckpt_use_boundary or bool(model_kwargs.get("use_boundary", False))
    model_kwargs = dict(model_kwargs)
    model_kwargs["num_classes"] = int(model_kwargs.get("num_classes", 1))
    model_kwargs["use_boundary"] = effective_use_boundary
    print(f"[eval model kwargs] {json.dumps(model_kwargs, ensure_ascii=False)}")

    model = ResSwinUNet(**model_kwargs).to(args.device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    boundary_head_ready = _load_model_state(model, state_dict)
    model.eval()

    print(f"[eval mode] {'dual_task_boundary' if effective_use_boundary else 'baseline'}")

    threshold_used = float(args.pred_threshold)
    threshold_source = "fixed"
    if args.threshold_search:
        thresholds = _parse_threshold_range(args.threshold_range)
        if args.val_image_dir and args.val_mask_dir:
            _, search_loader = _build_loader(
                image_dir=args.val_image_dir,
                mask_dir=args.val_mask_dir,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            threshold_source = "val"
        else:
            search_loader = test_loader
            threshold_source = "test_fallback"
            print("[threshold search warning] val dirs not provided, falling back to test set search.")

        threshold_used = _search_best_threshold(model, search_loader, args.device, thresholds)
        print(f"[threshold search] best threshold={threshold_used:.4f} source={threshold_source}")

    dice_values = []
    iou_values = []
    boundary_f1_values = []
    hd95_values = []
    sample_rows = []
    grouped = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            ids = batch["id"]
            sources = batch.get("source", [""] * len(ids))
            subsets = batch.get("subset", [""] * len(ids))
            round_ids = batch.get("round_id", None)
            if torch.is_tensor(round_ids):
                round_ids = [int(v) for v in round_ids.detach().cpu().tolist()]
            elif round_ids is None:
                round_ids = [0] * len(ids)

            parsed = _parse_model_outputs(model(images))
            seg_logits = parsed["seg"]
            boundary_logits = parsed["boundary"]

            d = dice_per_sample(seg_logits, masks, threshold=threshold_used)
            i = iou_per_sample(seg_logits, masks, threshold=threshold_used)
            dice_values.extend(d.cpu().tolist())
            iou_values.extend(i.cpu().tolist())

            seg_probs = torch.sigmoid(seg_logits)

            if boundary_logits is not None and boundary_head_ready:
                boundary_probs = torch.sigmoid(boundary_logits)
            else:
                boundary_list = [
                    _boundary_from_seg_prob(seg_probs[j], threshold_used, args.boundary_radius)
                    for j in range(seg_probs.size(0))
                ]
                boundary_probs = torch.stack(boundary_list, dim=0).unsqueeze(1)

            for j, sample_id in enumerate(ids):
                save_comparison_grid(
                    image_tensor=images[j],
                    gt_tensor=masks[j],
                    pred_prob=seg_probs[j],
                    path=Path(args.save_dir) / f"{sample_id}_eval.png",
                    pred_threshold=threshold_used,
                )

                pred_mask_np = (seg_probs[j].squeeze().cpu().numpy() > threshold_used).astype(np.uint8)
                gt_mask_np = _to_mask_np(masks[j])
                row = {
                    "id": sample_id,
                    "dice": float(d[j].item()),
                    "iou": float(i[j].item()),
                    "source": str(sources[j]) if j < len(sources) else "",
                    "subset": str(subsets[j]) if j < len(subsets) else "",
                    "round_id": int(round_ids[j]) if j < len(round_ids) else 0,
                }

                if args.report_boundary_metrics:
                    bf1 = boundary_f1_from_masks(pred_mask_np, gt_mask_np, boundary_radius=args.boundary_radius)
                    h95 = hd95_from_masks(pred_mask_np, gt_mask_np)
                    boundary_f1_values.append(bf1)
                    hd95_values.append(h95)
                    row["boundary_f1"] = float(bf1)
                    row["hd95"] = float(h95)

                prob = torch.clamp(seg_probs[j], min=1e-6, max=1.0 - 1e-6)
                entropy = -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob)).mean()
                row["uncertainty_entropy"] = float(entropy.item())
                sample_rows.append(row)

                gk = f"{row['source']}|{row['subset']}|r{row['round_id']}"
                if gk not in grouped:
                    grouped[gk] = {"dice": [], "iou": [], "boundary_f1": [], "hd95": []}
                grouped[gk]["dice"].append(row["dice"])
                grouped[gk]["iou"].append(row["iou"])
                if "boundary_f1" in row:
                    grouped[gk]["boundary_f1"].append(row["boundary_f1"])
                if "hd95" in row:
                    grouped[gk]["hd95"].append(row["hd95"])

                if args.save_boundary:
                    save_boundary_probability_heatmap(
                        boundary_prob=boundary_probs[j],
                        path=Path(args.boundary_dir) / f"{sample_id}_boundary.png",
                        upscale=max(1, args.boundary_heatmap_scale),
                    )

    report = {
        "num_samples": len(test_dataset),
        "threshold_used": float(threshold_used),
        "threshold_source": threshold_source,
        "dice_mean": float(np.mean(dice_values)) if dice_values else 0.0,
        "dice_std": float(np.std(dice_values)) if dice_values else 0.0,
        "iou_mean": float(np.mean(iou_values)) if iou_values else 0.0,
        "iou_std": float(np.std(iou_values)) if iou_values else 0.0,
    }

    if args.report_boundary_metrics:
        report["boundary_f1_mean"] = float(np.mean(boundary_f1_values)) if boundary_f1_values else 0.0
        report["boundary_f1_std"] = float(np.std(boundary_f1_values)) if boundary_f1_values else 0.0
        report["hd95_mean"] = float(np.mean(hd95_values)) if hd95_values else 0.0
        report["hd95_std"] = float(np.std(hd95_values)) if hd95_values else 0.0

    grouped_report = {}
    for key, vals in grouped.items():
        grouped_report[key] = {
            "n": len(vals["dice"]),
            "dice_mean": float(np.mean(vals["dice"])) if vals["dice"] else 0.0,
            "iou_mean": float(np.mean(vals["iou"])) if vals["iou"] else 0.0,
            "boundary_f1_mean": float(np.mean(vals["boundary_f1"])) if vals["boundary_f1"] else 0.0,
            "hd95_mean": float(np.mean(vals["hd95"])) if vals["hd95"] else 0.0,
        }
    if grouped_report:
        report["grouped_metrics"] = grouped_report

    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.per_sample_report:
        per_sample_path = Path(args.per_sample_report)
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        with open(per_sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_rows, f, indent=2, ensure_ascii=False)
        print(f"[Per-sample report saved to] {per_sample_path}")

    print("\n=== Evaluation Report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[Visualizations saved to] {args.save_dir}")
    if args.save_boundary:
        print(f"[Boundary maps saved to] {args.boundary_dir}")
    print(f"[Report saved to] {args.report_path}")


if __name__ == "__main__":
    main()
