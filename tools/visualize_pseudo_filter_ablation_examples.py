import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


STAGES = [
    (
        "Unfiltered/weak-filtered",
        "results/pseudo_filter_ablation/unfiltered_student_r1r2/preds",
        "results/pseudo_filter_ablation/unfiltered_student_r1r2/per_sample.csv",
    ),
    (
        "Strict filtered + SDF",
        "results/external_benchmark/ours_full_pipeline/preds",
        "results/external_benchmark/ours_full_pipeline/per_sample.csv",
    ),
]

SOURCES = ["CVC-300", "CVC-ColonDB", "ETIS", "PolypGen"]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def _read_manifest(path: Path, repo_root: Path) -> dict[str, dict]:
    manifest = {}
    for row in _read_csv(path):
        if row.get("subset") != "external" or not row.get("mask_path"):
            continue
        row = dict(row)
        row["image_path"] = str(_resolve_path(row["image_path"], repo_root))
        row["mask_path"] = str(_resolve_path(row["mask_path"], repo_root))
        manifest[row["id"]] = row
    return manifest


def _read_boxes(path: Path) -> dict[str, dict]:
    boxes = {}
    for row in _read_csv(path):
        try:
            boxes[row["id"]] = {
                "x0": float(row["x0"]),
                "y0": float(row["y0"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
            }
        except (KeyError, ValueError):
            continue
    return boxes


def _read_metrics(repo_root: Path) -> dict[str, dict[str, dict]]:
    metrics = {}
    for name, _pred_root, csv_path in STAGES:
        metrics[name] = {row["id"]: row for row in _read_csv(repo_root / csv_path)}
    return metrics


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape[:2] != (size[1], size[0]):
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def _panel(
    image_rgb: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    box: dict,
    title: str,
    dice: float,
    iou: float,
    panel_size: int,
) -> np.ndarray:
    h0, w0 = image_rgb.shape[:2]
    scale = min(panel_size / max(h0, w0), 1.0)
    w = int(round(w0 * scale))
    h = int(round(h0 * scale))
    image = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)
    pred = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    gt = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = image.copy()
    cyan = np.array([0, 194, 255], dtype=np.uint8)
    overlay[pred > 0] = (0.46 * overlay[pred > 0] + 0.54 * cyan).astype(np.uint8)

    canvas_h = panel_size + 58
    canvas = np.full((canvas_h, panel_size, 3), 246, dtype=np.uint8)
    y0 = 46
    x0 = (panel_size - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = overlay

    contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas[y0:y0 + h, x0:x0 + w], contours, -1, (76, 220, 96), 2)

    bx0 = int(round(box["x0"] * scale)) + x0
    by0 = int(round(box["y0"] * scale)) + y0
    bx1 = int(round(box["x1"] * scale)) + x0
    by1 = int(round(box["y1"] * scale)) + y0
    cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (255, 210, 42), 2)

    cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Dice {dice:.3f}  IoU {iou:.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (35, 35, 35), 1, cv2.LINE_AA)
    return canvas


def _choose_samples(manifest: dict[str, dict], metrics: dict[str, dict[str, dict]]) -> dict[str, str]:
    chosen = {}
    base_name = STAGES[0][0]
    final_name = STAGES[-1][0]
    for source in SOURCES:
        candidates = []
        for sample_id, row in manifest.items():
            if row["source"] != source or sample_id not in metrics[base_name] or sample_id not in metrics[final_name]:
                continue
            gt = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            area = float((gt > 127).sum()) / float(gt.shape[0] * gt.shape[1])
            base_dice = float(metrics[base_name][sample_id]["dice"])
            final_dice = float(metrics[final_name][sample_id]["dice"])
            final_iou = float(metrics[final_name][sample_id]["iou"])
            if 0.006 <= area <= 0.22 and final_dice >= 0.72 and final_dice > base_dice:
                candidates.append((final_dice - base_dice, final_dice, final_iou, sample_id))
        if not candidates:
            for sample_id, row in manifest.items():
                if row["source"] != source or sample_id not in metrics[base_name] or sample_id not in metrics[final_name]:
                    continue
                final_dice = float(metrics[final_name][sample_id]["dice"])
                base_dice = float(metrics[base_name][sample_id]["dice"])
                candidates.append((final_dice - base_dice, final_dice, float(metrics[final_name][sample_id]["iou"]), sample_id))
        chosen[source] = sorted(candidates, reverse=True)[0][3]
    return chosen


def _source_grid(source: str, sample_id: str, manifest: dict[str, dict], boxes: dict[str, dict], metrics: dict[str, dict[str, dict]], repo_root: Path, panel_size: int) -> np.ndarray:
    row = manifest[sample_id]
    image_bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(row["image_path"])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    gt_mask = _load_mask(Path(row["mask_path"]), (w, h))
    box = boxes[sample_id]

    panels = []
    for stage_name, pred_root, _csv_path in STAGES:
        pred_path = repo_root / pred_root / source / f"{sample_id}.png"
        pred_mask = _load_mask(pred_path, (w, h))
        metric = metrics[stage_name][sample_id]
        panels.append(_panel(image_rgb, pred_mask, gt_mask, box, stage_name, float(metric["dice"]), float(metric["iou"]), panel_size))

    gap = 12
    header_h = 58
    grid_w = len(panels) * panel_size + (len(panels) - 1) * gap
    grid_h = panels[0].shape[0] + header_h
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    title = f"{source} | {sample_id}"
    cv2.putText(grid, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)

    x = 0
    for panel in panels:
        grid[header_h:header_h + panel.shape[0], x:x + panel_size] = panel
        x += panel_size + gap
    return grid


def _stack(grids: list[np.ndarray], out_path: Path):
    width = max(grid.shape[1] for grid in grids)
    gap = 18
    height = sum(grid.shape[0] for grid in grids) + gap * (len(grids) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y = 0
    for grid in grids:
        canvas[y:y + grid.shape[0], :grid.shape[1]] = grid
        y += grid.shape[0] + gap
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _build_parser():
    p = argparse.ArgumentParser(description="Visualize pseudo-label filtering ablation examples.")
    p.add_argument("--manifest", type=str, default="data/joint_polyp_v1/manifest/samples_v1.csv")
    p.add_argument("--boxes", type=str, default="results/external_benchmark/auto_boxes_external/boxes.csv")
    p.add_argument("--out-dir", type=str, default="results/pseudo_filter_ablation/visual_examples_filtered_better")
    p.add_argument("--panel-size", type=int, default=280)
    p.add_argument("--sample-json", type=str, default="")
    return p


def main():
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = _read_manifest(repo_root / args.manifest, repo_root)
    boxes = _read_boxes(repo_root / args.boxes)
    metrics = _read_metrics(repo_root)

    if args.sample_json:
        with open(args.sample_json, "r", encoding="utf-8") as f:
            chosen = json.load(f)
    else:
        chosen = _choose_samples(manifest, metrics)

    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    grids = []
    selected_rows = []
    for source in SOURCES:
        sample_id = chosen[source]
        grid = _source_grid(source, sample_id, manifest, boxes, metrics, repo_root, args.panel_size)
        grids.append(grid)
        cv2.imwrite(str(out_dir / f"{source}_pseudo_filter_ablation.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        selected_rows.append({"source": source, "id": sample_id})

    _stack(grids, out_dir / "pseudo_filter_ablation_four_sources.png")
    with (out_dir / "selected_samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "id"])
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"[saved] {out_dir / 'pseudo_filter_ablation_four_sources.png'}")
    for row in selected_rows:
        print(f"[selected] {row['source']} {row['id']}")


if __name__ == "__main__":
    main()
