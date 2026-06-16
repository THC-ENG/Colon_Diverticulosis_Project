from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def _to_bin_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 127).astype(np.uint8)


def _dice_iou_from_masks(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    pred = pred.astype(np.uint8).reshape(-1)
    gt = gt.astype(np.uint8).reshape(-1)
    inter = float((pred * gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())

    union_dice = pred_sum + gt_sum
    if union_dice == 0.0:
        dice = 1.0
    else:
        dice = float((2.0 * inter + eps) / (union_dice + eps))

    union_iou = pred_sum + gt_sum - inter
    if union_iou == 0.0:
        iou = 1.0
    else:
        iou = float((inter + eps) / (union_iou + eps))

    return dice, iou


def _mask_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> dict:
    pred_flat = pred.astype(np.uint8).reshape(-1)
    gt_flat = gt.astype(np.uint8).reshape(-1)

    tp = float((pred_flat * gt_flat).sum())
    pred_sum = float(pred_flat.sum())
    gt_sum = float(gt_flat.sum())
    fp = pred_sum - tp
    fn = gt_sum - tp

    dice, iou = _dice_iou_from_masks(pred, gt, eps=eps)
    precision = 1.0 if pred_sum == 0.0 and gt_sum == 0.0 else float((tp + eps) / (tp + fp + eps))
    recall = 1.0 if pred_sum == 0.0 and gt_sum == 0.0 else float((tp + eps) / (tp + fn + eps))
    mae = float(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32))))

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "mae": mae,
    }


def _mask_to_boundary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    radius = max(1, int(radius))
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return ((dilated - eroded) > 0).astype(np.uint8)


def _boundary_f1(pred: np.ndarray, gt: np.ndarray, boundary_radius: int = 1, eps: float = 1e-6) -> float:
    pred_boundary = _mask_to_boundary(pred, radius=boundary_radius)
    gt_boundary = _mask_to_boundary(gt, radius=boundary_radius)

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


def _hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if int(pred.sum()) == 0 and int(gt.sum()) == 0:
        return 0.0

    fallback = float(np.hypot(pred.shape[0], pred.shape[1]))
    if int(pred.sum()) == 0 or int(gt.sum()) == 0:
        return fallback

    pred_boundary = _mask_to_boundary(pred, radius=1)
    gt_boundary = _mask_to_boundary(gt, radius=1)
    if int(pred_boundary.sum()) == 0 or int(gt_boundary.sum()) == 0:
        return fallback

    pred_inv = (1 - pred_boundary).astype(np.uint8)
    gt_inv = (1 - gt_boundary).astype(np.uint8)
    dist_to_pred = cv2.distanceTransform(pred_inv, cv2.DIST_L2, 5)
    dist_to_gt = cv2.distanceTransform(gt_inv, cv2.DIST_L2, 5)

    pred_to_gt = dist_to_gt[pred_boundary > 0]
    gt_to_pred = dist_to_pred[gt_boundary > 0]
    if pred_to_gt.size == 0 or gt_to_pred.size == 0:
        return fallback
    return float(max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95)))


def _resolve_pred_path(pred_root: Path, pred_template: str, sample_id: str, source: str) -> Path:
    rel = pred_template.format(id=sample_id, source=source)
    path = pred_root / rel
    if path.exists():
        return path

    stem = path.with_suffix("")
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = Path(str(stem) + ext)
        if candidate.exists():
            return candidate
    return path


def _load_external_rows(manifest_path: Path) -> list[dict]:
    rows = []
    with manifest_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {str(k).strip(): v for k, v in raw_row.items()}
            subset = str(row.get("subset", "")).strip()
            split = str(row.get("split", "")).strip()
            if subset != "external":
                continue
            if split and split != "test":
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError("No external test rows found in manifest.")
    return rows


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


METRIC_NAMES = ("dice", "iou", "precision", "recall", "mae", "boundary_f1", "hd95")


def main():
    parser = argparse.ArgumentParser(description="Evaluate external predictions by manifest protocol.")
    parser.add_argument("--manifest", type=str, default="data/joint_polyp_v1/manifest/samples_v1.csv")
    parser.add_argument("--pred-root", type=str, required=True, help="Prediction root directory.")
    parser.add_argument(
        "--pred-template",
        type=str,
        default="{source}/{id}.png",
        help="Relative template from pred-root. Supports {source} and {id}.",
    )
    parser.add_argument("--report-path", type=str, required=True)
    parser.add_argument("--per-sample-json", type=str, default="")
    parser.add_argument("--per-sample-csv", type=str, default="")
    parser.add_argument("--resize-pred-to-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--allow-missing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--eval-size",
        type=int,
        default=0,
        help="If >0, resize prediction and GT to eval-size x eval-size before metrics.",
    )
    args = parser.parse_args()
    if np is None:
        raise ModuleNotFoundError(
            "NumPy is required for mask evaluation. Install with: pip install numpy"
        )
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required for mask evaluation. Install with: pip install opencv-python"
        )

    manifest_path = Path(args.manifest)
    pred_root = Path(args.pred_root)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_external_rows(manifest_path)
    per_sample = []
    grouped = defaultdict(lambda: {metric: [] for metric in METRIC_NAMES} | {"n_resized": 0})
    missing = []

    for row in rows:
        sample_id = str(row["id"])
        source = str(row.get("source", ""))
        gt_path = Path(str(row["mask_path"]))
        pred_path = _resolve_pred_path(pred_root, args.pred_template, sample_id, source)

        if not gt_path.exists():
            raise FileNotFoundError(f"GT mask not found: {gt_path}")
        if not pred_path.exists():
            missing.append({"id": sample_id, "source": source, "pred_path": str(pred_path)})
            continue

        gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        if gt is None:
            raise RuntimeError(f"Failed to read GT mask: {gt_path}")
        if pred is None:
            raise RuntimeError(f"Failed to read prediction mask: {pred_path}")

        gt_bin = _to_bin_mask(gt)
        pred_bin = _to_bin_mask(pred)

        eval_size = int(args.eval_size)
        if eval_size > 0:
            gt_bin = cv2.resize(gt_bin, (eval_size, eval_size), interpolation=cv2.INTER_NEAREST)
            pred_bin = cv2.resize(pred_bin, (eval_size, eval_size), interpolation=cv2.INTER_NEAREST)
            gt_bin = (gt_bin > 0).astype(np.uint8)
            pred_bin = (pred_bin > 0).astype(np.uint8)

        resized = False
        if pred_bin.shape != gt_bin.shape:
            if not args.resize_pred_to_gt:
                raise ValueError(
                    f"Shape mismatch for {sample_id}: pred={pred_bin.shape}, gt={gt_bin.shape}. "
                    "Set --resize-pred-to-gt to enable nearest-neighbor resize."
                )
            pred_bin = cv2.resize(
                pred_bin,
                (gt_bin.shape[1], gt_bin.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            pred_bin = (pred_bin > 0).astype(np.uint8)
            resized = True

        metrics = _mask_metrics(pred_bin, gt_bin)
        metrics["boundary_f1"] = _boundary_f1(
            pred_bin,
            gt_bin,
            boundary_radius=max(1, int(args.boundary_radius)),
        )
        metrics["hd95"] = _hd95(pred_bin, gt_bin)

        per_sample.append(
            {
                "id": sample_id,
                "source": source,
                **{metric: float(metrics[metric]) for metric in METRIC_NAMES},
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
                "resized_pred": bool(resized),
            }
        )
        for metric in METRIC_NAMES:
            grouped[source][metric].append(float(metrics[metric]))
        if resized:
            grouped[source]["n_resized"] += 1

    if not per_sample:
        raise RuntimeError("No valid predictions were evaluated. Check pred-root/pred-template.")
    if missing and not args.allow_missing:
        first = missing[0]
        raise RuntimeError(
            f"Missing {len(missing)} predictions. First missing: "
            f"id={first['id']} source={first['source']} pred_path={first['pred_path']}. "
            "Use --allow-missing only for debugging partial outputs."
        )

    overall = {
        metric: _summarize([r[metric] for r in per_sample])
        for metric in METRIC_NAMES
    }

    grouped_report = {}
    for source, payload in sorted(grouped.items(), key=lambda x: x[0]):
        grouped_report[source] = {
            "n": len(payload["dice"]),
            "n_resized_pred": int(payload["n_resized"]),
        }
        for metric in METRIC_NAMES:
            summary = _summarize(payload[metric])
            grouped_report[source][f"{metric}_mean"] = summary["mean"]
            grouped_report[source][f"{metric}_std"] = summary["std"]

    report = {
        "manifest": str(manifest_path),
        "pred_root": str(pred_root),
        "pred_template": args.pred_template,
        "eval_size": int(args.eval_size),
        "num_expected_external_samples": len(rows),
        "num_evaluated_samples": len(per_sample),
        "num_missing_predictions": len(missing),
        "boundary_radius": int(args.boundary_radius),
        "grouped_metrics": grouped_report,
        "missing_predictions": missing[:100],
    }
    for metric in METRIC_NAMES:
        report[f"{metric}_mean"] = overall[metric]["mean"]
        report[f"{metric}_std"] = overall[metric]["std"]

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.per_sample_json:
        per_sample_json = Path(args.per_sample_json)
        per_sample_json.parent.mkdir(parents=True, exist_ok=True)
        with per_sample_json.open("w", encoding="utf-8") as f:
            json.dump(per_sample, f, indent=2, ensure_ascii=False)

    if args.per_sample_csv:
        per_sample_csv = Path(args.per_sample_csv)
        per_sample_csv.parent.mkdir(parents=True, exist_ok=True)
        with per_sample_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "source",
                    *METRIC_NAMES,
                    "gt_path",
                    "pred_path",
                    "resized_pred",
                ],
            )
            writer.writeheader()
            writer.writerows(per_sample)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
