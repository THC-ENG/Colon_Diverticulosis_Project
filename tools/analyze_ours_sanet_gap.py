import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ["dice", "iou", "precision", "recall", "boundary_f1", "hd95"]


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row, key: str, default=0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _mask_stats(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {"gt_area_ratio": None, "gt_components": None}
    binary = (mask > 127).astype(np.uint8)
    area_ratio = float(binary.mean())
    num_labels, _ = cv2.connectedComponents(binary, connectivity=8)
    return {
        "gt_area_ratio": area_ratio,
        "gt_components": int(max(0, num_labels - 1)),
    }


def _area_bin(area_ratio):
    if area_ratio is None:
        return "unknown"
    if area_ratio < 0.01:
        return "tiny_<1%"
    if area_ratio < 0.03:
        return "small_1-3%"
    if area_ratio < 0.08:
        return "medium_3-8%"
    if area_ratio < 0.18:
        return "large_8-18%"
    return "xlarge_>=18%"


def _summarize(rows, group_key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    out = []
    for group, items in sorted(groups.items()):
        rec = {"group": group, "n": len(items)}
        for m in METRICS:
            key = f"delta_{m}"
            vals = [float(r[key]) for r in items if r.get(key) not in {"", None}]
            if vals:
                rec[f"{key}_mean"] = float(np.mean(vals))
                rec[f"{key}_median"] = float(np.median(vals))
        for m in ["ours_dice", "sanet_dice", "ours_recall", "sanet_recall", "ours_precision", "sanet_precision"]:
            vals = [float(r[m]) for r in items if r.get(m) not in {"", None}]
            if vals:
                rec[f"{m}_mean"] = float(np.mean(vals))
        out.append(rec)
    return out


def _write_csv(path: Path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _overlay(image, mask, color, alpha=0.42):
    out = image.copy()
    if mask is None:
        return out
    binary = mask > 127
    color_arr = np.array(color, dtype=np.uint8)
    out[binary] = (out[binary] * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)
    return out


def _load_rgb(path: Path, fallback_shape=(256, 256, 3)):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros(fallback_shape, dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path, size=None):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        if size is None:
            return None
        return np.zeros(size, dtype=np.uint8)
    if size is not None and mask.shape[:2] != size:
        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def _make_panel(row, out_path: Path):
    image = _load_rgb(Path(row["image_path"]))
    h, w = image.shape[:2]
    gt = _load_mask(Path(row["gt_path"]), (h, w))
    ours = _load_mask(Path(row["ours_pred_path"]), (h, w))
    sanet = _load_mask(Path(row["sanet_pred_path"]), (h, w))

    gt_overlay = _overlay(image, gt, (22, 163, 74))
    ours_overlay = _overlay(image, ours, (37, 99, 235))
    sanet_overlay = _overlay(image, sanet, (249, 115, 22))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=130)
    axes[0].imshow(image)
    axes[0].set_title(f"{row['id']} | {row['source']}")
    axes[1].imshow(gt_overlay)
    axes[1].set_title("GT green")
    axes[2].imshow(ours_overlay)
    axes[2].set_title(f"Ours Dice {float(row['ours_dice']):.3f}")
    axes[3].imshow(sanet_overlay)
    axes[3].set_title(f"SANet Dice {float(row['sanet_dice']):.3f}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"Delta Dice Ours-SANet={float(row['delta_dice']):+.3f} | "
        f"Delta Recall={float(row['delta_recall']):+.3f} | "
        f"Area={float(row['gt_area_ratio']):.3%}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_group_bars(rows, value_key, title, out_path: Path):
    if not rows:
        return
    labels = [r["group"] for r in rows]
    values = [float(r.get(value_key, 0.0)) for r in rows]
    colors = ["#0f766e" if v >= 0 else "#dc2626" for v in values]
    plt.figure(figsize=(9, 4.8), dpi=150)
    bars = plt.bar(labels, values, color=colors)
    plt.axhline(0, color="#334155", linewidth=0.8)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(value_key)
    plt.title(title)
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:+.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze per-sample gap between Ours and SANet.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ours-per-sample", required=True)
    parser.add_argument("--sanet-per-sample", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--panel-k", type=int, default=24)
    args = parser.parse_args()

    manifest_rows = _read_csv(Path(args.manifest))
    manifest_by_id = {
        (r.get("source", ""), r.get("id", "")): r
        for r in manifest_rows
        if r.get("subset") == "external" and r.get("split") in {"", "test"}
    }
    ours_rows = _read_csv(Path(args.ours_per_sample))
    sanet_rows = _read_csv(Path(args.sanet_per_sample))
    sanet_by_id = {(r["source"], r["id"]): r for r in sanet_rows}

    out_root = Path(args.output_root)
    panel_root = out_root / "panels"
    fig_root = out_root / "figures"
    panel_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for ours in ours_rows:
        key = (ours["source"], ours["id"])
        sanet = sanet_by_id.get(key)
        manifest = manifest_by_id.get(key, {})
        if not sanet:
            continue
        gt_path = Path(ours.get("gt_path", manifest.get("mask_path", "")))
        stats = _mask_stats(gt_path)
        row = {
            "id": ours["id"],
            "source": ours["source"],
            "image_path": manifest.get("image_path", ""),
            "gt_path": ours.get("gt_path", manifest.get("mask_path", "")),
            "ours_pred_path": ours.get("pred_path", ""),
            "sanet_pred_path": sanet.get("pred_path", ""),
            **stats,
        }
        row["area_bin"] = _area_bin(row["gt_area_ratio"])
        for m in METRICS:
            o = _as_float(ours, m)
            s = _as_float(sanet, m)
            row[f"ours_{m}"] = o
            row[f"sanet_{m}"] = s
            row[f"delta_{m}"] = o - s
        # Quick failure taxonomy.
        if row["delta_recall"] < -0.15 and row["delta_precision"] >= -0.05:
            row["gap_type"] = "ours_under_segments"
        elif row["delta_precision"] < -0.15 and row["delta_recall"] >= -0.05:
            row["gap_type"] = "ours_over_segments"
        elif row["delta_boundary_f1"] < -0.12:
            row["gap_type"] = "ours_boundary_worse"
        elif row["delta_dice"] < -0.15:
            row["gap_type"] = "ours_global_worse"
        elif row["delta_dice"] > 0.15:
            row["gap_type"] = "ours_global_better"
        else:
            row["gap_type"] = "similar_or_mixed"
        rows.append(row)

    sanet_wins = sorted(rows, key=lambda r: r["delta_dice"])
    ours_wins = sorted(rows, key=lambda r: r["delta_dice"], reverse=True)
    source_summary = _summarize(rows, "source")
    area_summary = _summarize(rows, "area_bin")
    type_summary = _summarize(rows, "gap_type")

    _write_csv(out_root / "ours_vs_sanet_per_sample_gap.csv", rows)
    _write_csv(out_root / "sanet_wins_top.csv", sanet_wins[: args.top_k])
    _write_csv(out_root / "ours_wins_top.csv", ours_wins[: args.top_k])
    _write_csv(out_root / "gap_by_source.csv", source_summary)
    _write_csv(out_root / "gap_by_area_bin.csv", area_summary)
    _write_csv(out_root / "gap_by_type.csv", type_summary)

    _plot_group_bars(
        source_summary,
        "delta_dice_mean",
        "Mean Dice Gap by Source (Ours - SANet)",
        fig_root / "delta_dice_by_source.png",
    )
    _plot_group_bars(
        area_summary,
        "delta_dice_mean",
        "Mean Dice Gap by Lesion Area (Ours - SANet)",
        fig_root / "delta_dice_by_area.png",
    )
    _plot_group_bars(
        type_summary,
        "delta_dice_mean",
        "Mean Dice Gap by Failure Type (Ours - SANet)",
        fig_root / "delta_dice_by_gap_type.png",
    )

    panel_items = [(r, "sanet_win") for r in sanet_wins[: args.panel_k]]
    panel_items += [(r, "ours_win") for r in ours_wins[: args.panel_k]]
    panel_paths = []
    for idx, (row, prefix) in enumerate(panel_items, start=1):
        out_path = panel_root / f"{prefix}_{idx:03d}_{row['source']}_{row['id']}.jpg"
        _make_panel(row, out_path)
        panel_paths.append(out_path)

    html_lines = [
        "<html><head><meta charset='utf-8'><title>Ours vs SANet Gap</title></head><body>",
        "<h1>Ours vs SANet Gap Panels</h1>",
        "<p>Green=GT, Blue=Ours, Orange=SANet. Delta is Ours - SANet.</p>",
    ]
    for path in panel_paths:
        rel = path.relative_to(out_root).as_posix()
        html_lines.append(f"<div style='margin:18px 0'><img src='{rel}' style='max-width:100%'></div>")
    html_lines.append("</body></html>")
    (out_root / "gap_panel_gallery.html").write_text("\n".join(html_lines), encoding="utf-8")

    payload = {
        "num_samples": len(rows),
        "mean_delta_dice": float(np.mean([r["delta_dice"] for r in rows])) if rows else None,
        "mean_delta_iou": float(np.mean([r["delta_iou"] for r in rows])) if rows else None,
        "mean_delta_boundary_f1": float(np.mean([r["delta_boundary_f1"] for r in rows])) if rows else None,
        "mean_delta_hd95": float(np.mean([r["delta_hd95"] for r in rows])) if rows else None,
        "source_summary": source_summary,
        "area_summary": area_summary,
        "type_summary": type_summary,
        "top_sanet_wins": sanet_wins[:10],
        "top_ours_wins": ours_wins[:10],
    }
    (out_root / "gap_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:4000])
    print(f"[saved] {out_root}")


if __name__ == "__main__":
    main()
