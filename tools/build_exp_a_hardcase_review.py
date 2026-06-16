import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Build Exp A hard-case review candidates from Ours/SANet reference/MedSAM pseudo disagreement. "
            "SANet is used only for mining/reference, not as a training label."
        )
    )
    p.add_argument("--target-rows-csv", type=str, required=True)
    p.add_argument("--medsam-quality-csv", type=str, required=True)
    p.add_argument("--medsam-mask-dir", type=str, required=True)
    p.add_argument("--ours-mask-dir", type=str, required=True)
    p.add_argument("--sanet-mask-dir", type=str, required=True)
    p.add_argument("--out-root", type=str, required=True)
    p.add_argument("--top-k", type=int, default=300)
    p.add_argument("--panel-size", type=int, default=256)
    p.add_argument("--min-medsam-quality", type=float, default=0.45)
    p.add_argument("--max-auto-accept-quality", type=float, default=0.72)
    return p


def _read_csv(path: str) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _safe_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_mask(path: Path, size: int) -> np.ndarray | None:
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def _load_image(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.putText(image, "image not found", (12, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def _area(mask: np.ndarray | None) -> float:
    if mask is None:
        return 0.0
    return float((mask > 0).mean())


def _component_count(mask: np.ndarray | None) -> int:
    if mask is None or mask.sum() == 0:
        return 0
    n, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(max(0, n - 1))


def _boundary_complexity(mask: np.ndarray | None) -> float:
    if mask is None or mask.sum() == 0:
        return 0.0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    area = float(mask.sum())
    return float(perimeter / math.sqrt(max(area, 1.0)))


def _overlay(image: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.copy()
    if mask is None:
        return out
    color_arr = np.zeros_like(out)
    color_arr[:, :] = color
    m = mask.astype(bool)
    blended = out.astype(np.float32) * (1.0 - alpha) + color_arr.astype(np.float32) * alpha
    out[m] = np.clip(blended[m], 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1)
    return out


def _label_panel(panel: np.ndarray, label: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, label[:42], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _make_panel(image_path: Path, medsam, ours, sanet, row: dict, out_path: Path, size: int):
    image = _load_image(image_path, size)
    med = _label_panel(_overlay(image, medsam, (0, 0, 255)), "MedSAM pseudo")
    our = _label_panel(_overlay(image, ours, (0, 220, 0)), "Ours current")
    san = _label_panel(_overlay(image, sanet, (255, 80, 0)), "SANet reference")

    mix = image.copy()
    if medsam is not None:
        mix = _overlay(mix, medsam, (0, 0, 255), 0.28)
    if ours is not None:
        mix = _overlay(mix, ours, (0, 220, 0), 0.28)
    if sanet is not None:
        mix = _overlay(mix, sanet, (255, 80, 0), 0.28)
    mix_label = (
        f"score={_safe_float(row.get('priority_score')):.3f} "
        f"q={_safe_float(row.get('medsam_quality')):.3f} "
        f"om={_safe_float(row.get('iou_ours_medsam')):.2f} "
        f"os={_safe_float(row.get('iou_ours_sanet')):.2f}"
    )
    mix = _label_panel(mix, mix_label)

    raw = _label_panel(image, f"{row['id']} | {row.get('source', '')}")
    panel = np.concatenate([raw, med, our, san, mix], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def _score_case(medsam, ours, sanet, quality_row: dict) -> dict:
    iou_om = _mask_iou(ours, medsam)
    iou_os = _mask_iou(ours, sanet)
    iou_ms = _mask_iou(medsam, sanet)
    area_m = _area(medsam)
    area_o = _area(ours)
    area_s = _area(sanet)
    area_values = [area_m, area_o, area_s]
    area_spread = max(area_values) - min(area_values)
    quality = _safe_float(quality_row.get("quality"), 0.0)
    consistency = _safe_float(quality_row.get("consistency_iou"), 0.0)
    components = max(_component_count(medsam), _component_count(ours), _component_count(sanet))
    complexity = max(_boundary_complexity(medsam), _boundary_complexity(ours), _boundary_complexity(sanet))

    disagreement = 1.0 - ((iou_om + iou_os + iou_ms) / 3.0)
    quality_band = 1.0 if 0.45 <= quality <= 0.72 else 0.35
    area_extreme = 1.0 if min(area_values) < 0.004 or max(area_values) > 0.22 else 0.0
    under_hint = 1.0 if area_o < 0.35 * max(area_m, area_s, 1e-6) and max(area_m, area_s) > 0.015 else 0.0
    over_hint = 1.0 if area_o > 2.2 * max(min(area_m, area_s), 1e-6) and area_o > 0.04 else 0.0
    medsam_good_disagree = 1.0 if quality >= 0.60 and iou_om < 0.55 else 0.0

    priority = (
        0.40 * disagreement
        + 0.16 * area_spread
        + 0.12 * (1.0 - consistency)
        + 0.10 * medsam_good_disagree
        + 0.08 * under_hint
        + 0.06 * over_hint
        + 0.05 * min(complexity / 25.0, 1.0)
        + 0.03 * min(max(components - 1, 0) / 4.0, 1.0)
    )
    priority *= quality_band

    return {
        "priority_score": priority,
        "iou_ours_medsam": iou_om,
        "iou_ours_sanet": iou_os,
        "iou_medsam_sanet": iou_ms,
        "area_medsam": area_m,
        "area_ours": area_o,
        "area_sanet": area_s,
        "area_spread": area_spread,
        "component_max": components,
        "boundary_complexity_max": complexity,
        "medsam_quality": quality,
        "medsam_consistency_iou": consistency,
        "under_hint": under_hint,
        "over_hint": over_hint,
    }


def _write_html(rows: list[dict], out_path: Path, rel_panel_dir: str):
    cards = []
    for row in rows:
        panel_name = html.escape(Path(row["panel_path"]).name)
        cards.append(
            "<div class='card'>"
            f"<h3>{html.escape(row['id'])} | score={float(row['priority_score']):.3f}</h3>"
            f"<img src='{html.escape(rel_panel_dir)}/{panel_name}' />"
            "<p>"
            f"MedSAM q={float(row['medsam_quality']):.3f}, "
            f"IoU(Ours,MedSAM)={float(row['iou_ours_medsam']):.3f}, "
            f"IoU(Ours,SANet)={float(row['iou_ours_sanet']):.3f}"
            "</p></div>"
        )
    page = """<!doctype html>
<html><head><meta charset="utf-8"><title>Exp A Hardcase Review</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f7f4ee;color:#1d1a16}
.hint{max-width:1100px;line-height:1.45}
.card{background:white;border:1px solid #ddd3c2;border-radius:12px;padding:14px;margin:18px 0;box-shadow:0 8px 24px rgba(50,35,15,.08)}
.card img{max-width:100%;height:auto;border-radius:8px}
h1{margin-bottom:6px}
h3{margin:0 0 10px}
code{background:#eee4d4;padding:2px 5px;border-radius:4px}
</style></head><body>
<h1>Exp A Hardcase Review</h1>
<div class="hint">
<p>颜色约定：MedSAM=红，Ours=绿，SANet reference=蓝。SANet 只用于挑样和对比，不作为训练标签。</p>
<p>建议决策：<code>accept_medsam</code>、<code>accept_ours</code>、<code>manual_mask</code>、<code>override_box_then_medsam</code>、<code>reject</code>。</p>
</div>
"""
    page += "\n".join(cards)
    page += "\n</body></html>\n"
    out_path.write_text(page, encoding="utf-8")


def main():
    args = _build_parser().parse_args()
    out_root = Path(args.out_root)
    panels_dir = out_root / "panels"
    out_root.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    target_rows = {row["id"]: row for row in _read_csv(args.target_rows_csv)}
    quality_rows = {row["id"]: row for row in _read_csv(args.medsam_quality_csv)}
    medsam_dir = Path(args.medsam_mask_dir)
    ours_dir = Path(args.ours_mask_dir)
    sanet_dir = Path(args.sanet_mask_dir)

    candidates = []
    for sample_id, target in target_rows.items():
        qrow = quality_rows.get(sample_id, {})
        medsam = _load_mask(medsam_dir / f"{sample_id}.png", args.panel_size)
        ours = _load_mask(ours_dir / f"{sample_id}.png", args.panel_size)
        sanet = _load_mask(sanet_dir / f"{sample_id}.png", args.panel_size)
        if medsam is None and ours is None and sanet is None:
            continue

        metrics = _score_case(medsam, ours, sanet, qrow)
        metrics.update(
            {
                "id": sample_id,
                "image_path": target.get("image_path", ""),
                "source": target.get("source", ""),
                "center": target.get("center", ""),
                "medsam_mask_path": str(medsam_dir / f"{sample_id}.png"),
                "ours_mask_path": str(ours_dir / f"{sample_id}.png"),
                "sanet_mask_path": str(sanet_dir / f"{sample_id}.png"),
                "prompt_box": qrow.get("prompt_box", ""),
                "prompt_source": qrow.get("prompt_source", ""),
                "medsam_hard_mask_path": qrow.get("hard_mask_path", ""),
                "medsam_soft_path": qrow.get("soft_path", ""),
                "medsam_edge_path": qrow.get("edge_path", ""),
            }
        )
        candidates.append(metrics)

    candidates.sort(key=lambda r: (float(r["priority_score"]), float(r["medsam_quality"])), reverse=True)
    top_rows = candidates[: max(1, int(args.top_k))]

    for rank, row in enumerate(top_rows, start=1):
        row["rank"] = rank
        row["panel_path"] = str(panels_dir / f"{rank:04d}_{row['id']}.jpg")
        _make_panel(
            Path(row["image_path"]),
            _load_mask(Path(row["medsam_mask_path"]), args.panel_size),
            _load_mask(Path(row["ours_mask_path"]), args.panel_size),
            _load_mask(Path(row["sanet_mask_path"]), args.panel_size),
            row,
            Path(row["panel_path"]),
            args.panel_size,
        )

    score_fields = [
        "rank",
        "id",
        "image_path",
        "source",
        "center",
        "priority_score",
        "medsam_quality",
        "medsam_consistency_iou",
        "iou_ours_medsam",
        "iou_ours_sanet",
        "iou_medsam_sanet",
        "area_medsam",
        "area_ours",
        "area_sanet",
        "area_spread",
        "component_max",
        "boundary_complexity_max",
        "under_hint",
        "over_hint",
        "prompt_box",
        "prompt_source",
        "medsam_hard_mask_path",
        "medsam_soft_path",
        "medsam_edge_path",
        "ours_mask_path",
        "sanet_mask_path",
        "panel_path",
    ]
    with (out_root / "hardcase_candidates_all.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=score_fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(candidates, start=1):
            row_out = dict(row)
            row_out.setdefault("rank", i)
            row_out.setdefault("panel_path", "")
            writer.writerow(row_out)

    review_fields = score_fields + [
        "decision",
        "chosen_label",
        "manual_mask_path",
        "manual_box_xyxy",
        "notes",
    ]
    with (out_root / "hardcase_review_template.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=review_fields, extrasaction="ignore")
        writer.writeheader()
        for row in top_rows:
            row_out = dict(row)
            row_out.update(
                {
                    "decision": "",
                    "chosen_label": "",
                    "manual_mask_path": "",
                    "manual_box_xyxy": "",
                    "notes": "",
                }
            )
            writer.writerow(row_out)

    _write_html(top_rows, out_root / "hardcase_review_gallery.html", "panels")
    meta = {
        "target_rows_csv": args.target_rows_csv,
        "medsam_quality_csv": args.medsam_quality_csv,
        "medsam_mask_dir": args.medsam_mask_dir,
        "ours_mask_dir": args.ours_mask_dir,
        "sanet_mask_dir": args.sanet_mask_dir,
        "candidate_n": len(candidates),
        "review_n": len(top_rows),
        "note": "SANet is used only as a reference for hard-case mining.",
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[candidates] all={len(candidates)} review={len(top_rows)}")
    print(f"[review csv] {out_root / 'hardcase_review_template.csv'}")
    print(f"[gallery] {out_root / 'hardcase_review_gallery.html'}")


if __name__ == "__main__":
    main()
