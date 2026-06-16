import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


MODEL_SPECS = [
    ("Ours", "results/external_benchmark/ours_full_pipeline/preds", "results/external_benchmark/ours_full_pipeline/per_sample.csv"),
    ("U-Net", "results/external_benchmark/unet/preds", "results/external_benchmark/unet/per_sample.csv"),
    ("U-Net++", "results/external_benchmark/unetpp/preds", "results/external_benchmark/unetpp/per_sample.csv"),
    ("PraNet", "results/external_benchmark/pranet/preds", "results/external_benchmark/pranet/per_sample.csv"),
    ("TransUNet", "results/external_benchmark/transunet/preds", "results/external_benchmark/transunet/per_sample.csv"),
    ("SANet", "results/external_benchmark/sanet/preds", "results/external_benchmark/sanet/per_sample.csv"),
    ("SAM-ViT-B", "results/external_benchmark/sam_vit_b_box/preds", "results/external_benchmark/sam_vit_b_box/per_sample.csv"),
    ("MobileSAM", "results/external_benchmark/mobile_sam_box/preds", "results/external_benchmark/mobile_sam_box/per_sample.csv"),
]

SOURCES = ["CVC-300", "CVC-ColonDB", "ETIS", "PolypGen"]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_path(path_text: str, repo_root: Path) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else repo_root / p


def _read_manifest(path: Path, repo_root: Path) -> dict[str, dict]:
    rows = _read_csv(path)
    out = {}
    for row in rows:
        if row.get("subset") == "external" and row.get("mask_path"):
            row = dict(row)
            row["image_path"] = str(_resolve_path(row["image_path"], repo_root))
            row["mask_path"] = str(_resolve_path(row["mask_path"], repo_root))
            out[row["id"]] = row
    return out


def _read_boxes(path: Path) -> dict[str, dict]:
    rows = _read_csv(path)
    out = {}
    for row in rows:
        try:
            out[row["id"]] = {
                "x0": float(row["x0"]),
                "y0": float(row["y0"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
            }
        except (KeyError, ValueError):
            continue
    return out


def _read_metrics(specs: list[tuple[str, str, str]], repo_root: Path) -> dict[str, dict[str, dict]]:
    metrics = {}
    for model_name, _pred_root, metric_path in specs:
        rows = _read_csv(repo_root / metric_path)
        metrics[model_name] = {row["id"]: row for row in rows}
    return metrics


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape[:2] != (size[1], size[0]):
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def _overlay_panel(
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
    color = np.array([0, 194, 255], dtype=np.uint8)
    overlay[pred > 0] = (0.48 * overlay[pred > 0] + 0.52 * color).astype(np.uint8)

    canvas_h = panel_size + 58
    canvas = np.full((canvas_h, panel_size, 3), 245, dtype=np.uint8)
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
    cv2.putText(
        canvas,
        f"Dice {dice:.3f}  IoU {iou:.3f}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _choose_samples(metrics: dict[str, dict[str, dict]], manifest: dict[str, dict]) -> dict[str, str]:
    chosen = {}
    model_names = list(metrics.keys())
    for source in SOURCES:
        candidates = []
        for sample_id, row in manifest.items():
            if row.get("source") != source:
                continue
            dices = []
            for model_name in model_names:
                metric_row = metrics[model_name].get(sample_id)
                if not metric_row:
                    dices = []
                    break
                dices.append(float(metric_row["dice"]))
            if dices:
                candidates.append((sample_id, float(np.mean(dices))))
        if not candidates:
            raise RuntimeError(f"No common candidate found for source={source}")
        values = np.array([v for _sid, v in candidates], dtype=np.float32)
        median = float(np.median(values))
        chosen[source] = min(candidates, key=lambda item: abs(item[1] - median))[0]
    return chosen


def _make_source_grid(
    source: str,
    sample_id: str,
    manifest: dict[str, dict],
    boxes: dict[str, dict],
    metrics: dict[str, dict[str, dict]],
    specs: list[tuple[str, str, str]],
    repo_root: Path,
    out_path: Path,
    panel_size: int,
) -> np.ndarray:
    row = manifest[sample_id]
    image_bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(row["image_path"])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    gt_mask = _load_mask(Path(row["mask_path"]), (w, h))
    box = boxes[sample_id]

    panels = []
    for model_name, pred_root, _metric_path in specs:
        pred_path = repo_root / pred_root / source / f"{sample_id}.png"
        pred_mask = _load_mask(pred_path, (w, h))
        metric_row = metrics[model_name][sample_id]
        panels.append(
            _overlay_panel(
                image_rgb=image_rgb,
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                box=box,
                title=model_name,
                dice=float(metric_row["dice"]),
                iou=float(metric_row["iou"]),
                panel_size=panel_size,
            )
        )

    gap = 10
    header_h = 58
    grid_w = len(panels) * panel_size + (len(panels) - 1) * gap
    grid_h = panels[0].shape[0] + header_h
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    title = f"{source} | {sample_id} | cyan=prediction, green=GT contour, yellow=box"
    cv2.putText(grid, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    x = 0
    for panel in panels:
        grid[header_h:header_h + panel.shape[0], x:x + panel_size] = panel
        x += panel_size + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return grid


def _stack_grids(grids: list[np.ndarray], out_path: Path):
    width = max(g.shape[1] for g in grids)
    gap = 18
    height = sum(g.shape[0] for g in grids) + gap * (len(grids) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y = 0
    for grid in grids:
        canvas[y:y + grid.shape[0], :grid.shape[1]] = grid
        y += grid.shape[0] + gap
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _build_parser():
    p = argparse.ArgumentParser(description="Visualize the same external samples across eight models.")
    p.add_argument("--manifest", type=str, default="data/joint_polyp_v1/manifest/samples_v1.csv")
    p.add_argument("--boxes", type=str, default="results/external_benchmark/auto_boxes_external/boxes.csv")
    p.add_argument("--out-dir", type=str, default="results/external_benchmark/visual_examples")
    p.add_argument("--panel-size", type=int, default=220)
    p.add_argument("--sample-json", type=str, default="")
    return p


def main():
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = _read_manifest(repo_root / args.manifest, repo_root)
    boxes = _read_boxes(repo_root / args.boxes)
    metrics = _read_metrics(MODEL_SPECS, repo_root)

    if args.sample_json:
        with Path(args.sample_json).open("r", encoding="utf-8") as f:
            chosen = json.load(f)
    else:
        chosen = _choose_samples(metrics, manifest)

    out_dir = repo_root / args.out_dir
    grids = []
    selected_rows = []
    for source in SOURCES:
        sample_id = chosen[source]
        if sample_id not in boxes:
            raise RuntimeError(f"Missing auto box for sample_id={sample_id}")
        grid = _make_source_grid(
            source=source,
            sample_id=sample_id,
            manifest=manifest,
            boxes=boxes,
            metrics=metrics,
            specs=MODEL_SPECS,
            repo_root=repo_root,
            out_path=out_dir / f"{source}_eight_model_overlay.png",
            panel_size=args.panel_size,
        )
        grids.append(grid)
        selected_rows.append({"source": source, "id": sample_id})

    _stack_grids(grids, out_dir / "external_four_sources_eight_models.png")
    with (out_dir / "selected_samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "id"])
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"[saved] {out_dir / 'external_four_sources_eight_models.png'}")
    for row in selected_rows:
        print(f"[selected] {row['source']} {row['id']}")


if __name__ == "__main__":
    main()
