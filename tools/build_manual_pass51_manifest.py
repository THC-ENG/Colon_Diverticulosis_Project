import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


PASS_DECISIONS = {"pass", "keep", "accept", "approved", "yes", "y", "1", "ok"}


BASE_COLUMNS = [
    "id",
    "image_path",
    "mask_path",
    "subset",
    "split",
    "source",
    "center",
    "is_labeled",
    "is_pseudo",
    "pseudo_weight",
    "round_id",
    "exclude_from_tuning",
    "soft_path",
    "edge_path",
    "tier",
]


def _read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve(path_text: str, anchor: Path) -> Path:
    p = Path(str(path_text or "").strip())
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    for cand in (anchor.parent / p, anchor.parent.parent / p, Path.cwd() / p):
        if cand.exists():
            return cand.resolve()
    return (Path.cwd() / p).resolve()


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    fg = (mask_u8 > 0).astype(np.uint8)
    if int(fg.sum()) == 0:
        return fg
    h, w = fg.shape[:2]
    flood = fg.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = flood == 0
    out = fg.copy()
    out[holes] = 1
    return out


def _keep_largest_components(mask_u8: np.ndarray, keep_max: int) -> np.ndarray:
    fg = (mask_u8 > 0).astype(np.uint8)
    if keep_max <= 0 or int(fg.sum()) == 0:
        return fg
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num <= 1:
        return fg
    labs = list(range(1, num))
    labs = sorted(labs, key=lambda lab: int(stats[lab, cv2.CC_STAT_AREA]), reverse=True)
    out = np.zeros_like(fg)
    for lab in labs[: int(keep_max)]:
        out[labels == lab] = 1
    return out


def _postprocess_mask(
    mask_u8: np.ndarray,
    close_iters: int,
    open_iters: int,
    smooth_blur: int,
    smooth_threshold: float,
    keep_max_components: int,
) -> np.ndarray:
    fg = (mask_u8 > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if close_iters > 0:
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))
    fg = _fill_holes(fg)
    if open_iters > 0:
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=int(open_iters))
    fg = _keep_largest_components(fg, keep_max=int(keep_max_components))
    fg = _fill_holes(fg)
    if smooth_blur > 0:
        k = int(smooth_blur)
        if k % 2 == 0:
            k += 1
        prob = cv2.GaussianBlur(fg.astype(np.float32), (k, k), 0)
        fg = (prob >= float(smooth_threshold)).astype(np.uint8)
        fg = _fill_holes(fg)
    return fg.astype(np.uint8) * 255


def _edge_from_mask(mask_u8: np.ndarray) -> np.ndarray:
    fg = (mask_u8 > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(fg, cv2.MORPH_GRADIENT, kernel, iterations=1).astype(np.float32)


def _soft_from_mask(mask_u8: np.ndarray, blur: int) -> np.ndarray:
    fg = (mask_u8 > 0).astype(np.float32)
    k = max(3, int(blur))
    if k % 2 == 0:
        k += 1
    soft = cv2.GaussianBlur(fg, (k, k), 0)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def _overlay(image_bgr: np.ndarray, mask_u8: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image_bgr.copy().astype(np.float32)
    m = mask_u8 > 0
    if np.any(m):
        out[m] = 0.62 * out[m] + 0.38 * np.array(color, dtype=np.float32)
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1)
    return out


def _panel(image_bgr: np.ndarray, before: np.ndarray, after: np.ndarray, sample_id: str) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    cols = [
        image_bgr,
        _overlay(image_bgr, before, (0, 0, 255)),
        _overlay(image_bgr, after, (0, 255, 0)),
    ]
    out = np.concatenate(cols, axis=1)
    for i, label in enumerate(("image", "before", "after")):
        cv2.putText(out, label, (i * w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, sample_id, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess manually approved pseudo masks and append them to a student manifest.")
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-pass-manifest", required=True)
    parser.add_argument("--output-student-manifest", required=True)
    parser.add_argument("--pseudo-weight", type=float, default=0.22)
    parser.add_argument("--subset", default="pseudo_round3_manual51")
    parser.add_argument("--round-id", type=int, default=3)
    parser.add_argument("--tier", default="mid")
    parser.add_argument("--close-iters", type=int, default=2)
    parser.add_argument("--open-iters", type=int, default=1)
    parser.add_argument("--smooth-blur", type=int, default=5)
    parser.add_argument("--smooth-threshold", type=float, default=0.45)
    parser.add_argument("--soft-blur", type=int, default=9)
    parser.add_argument("--keep-max-components", type=int, default=1)
    args = parser.parse_args()

    review_path = Path(args.review_csv)
    base_path = Path(args.base_manifest)
    out_root = Path(args.output_root)
    hard_dir = out_root / "hard_masks"
    soft_dir = out_root / "soft_probs"
    edge_dir = out_root / "edge_probs"
    panel_dir = out_root / "panels_before_after"
    for d in (hard_dir, soft_dir, edge_dir, panel_dir):
        d.mkdir(parents=True, exist_ok=True)

    review_rows, review_fields = _read_csv(review_path)
    base_rows, base_fields = _read_csv(base_path)
    pass_rows = [
        r
        for r in review_rows
        if str(r.get("decision", "")).strip().lower() in PASS_DECISIONS
        and str(r.get("id", "")).strip()
    ]

    out_pass_rows = []
    changed_pixels = []
    for row in pass_rows:
        sid = str(row.get("id", "")).strip()
        image_path = _resolve(str(row.get("image_path", "")).strip(), review_path)
        mask_path = _resolve(str(row.get("hard_mask_path", "")).strip(), review_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Cannot read image for {sid}: {image_path}")
        if mask is None:
            raise RuntimeError(f"Cannot read mask for {sid}: {mask_path}")
        h, w = image.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        post = _postprocess_mask(
            mask,
            close_iters=int(args.close_iters),
            open_iters=int(args.open_iters),
            smooth_blur=int(args.smooth_blur),
            smooth_threshold=float(args.smooth_threshold),
            keep_max_components=int(args.keep_max_components),
        )
        before_bin = mask > 127
        after_bin = post > 127
        denom = float(max(1, before_bin.size))
        changed = float(np.logical_xor(before_bin, after_bin).sum()) / denom
        changed_pixels.append(changed)

        out_hard = hard_dir / f"{sid}.png"
        out_soft = soft_dir / f"{sid}.npy"
        out_edge = edge_dir / f"{sid}.npy"
        out_panel = panel_dir / f"{sid}.jpg"
        cv2.imwrite(str(out_hard), post)
        np.save(str(out_soft), _soft_from_mask(post, blur=int(args.soft_blur)))
        np.save(str(out_edge), _edge_from_mask(post))
        cv2.imwrite(str(out_panel), _panel(image, mask, post, sid))

        out = dict(row)
        out.update(
            {
                "mask_path": str(out_hard),
                "hard_mask_path": str(out_hard),
                "soft_path": str(out_soft),
                "edge_path": str(out_edge),
                "panel_path": str(out_panel),
                "subset": str(args.subset),
                "split": "pseudo_train",
                "is_labeled": "0",
                "is_pseudo": "1",
                "pseudo_weight": f"{float(args.pseudo_weight):.6f}",
                "round_id": str(int(args.round_id)),
                "exclude_from_tuning": "0",
                "tier": str(args.tier),
                "postprocess_changed_ratio": f"{changed:.8f}",
                "postprocess_note": "fill_holes_close_open_smooth_keep_largest",
            }
        )
        out_pass_rows.append(out)

    pass_fields = []
    for col in BASE_COLUMNS + review_fields + [
        "hard_mask_path",
        "panel_path",
        "postprocess_changed_ratio",
        "postprocess_note",
    ]:
        if col not in pass_fields:
            pass_fields.append(col)
    _write_csv(Path(args.output_pass_manifest), out_pass_rows, pass_fields)

    student_fields = []
    for col in BASE_COLUMNS + base_fields + pass_fields:
        if col not in student_fields:
            student_fields.append(col)
    pass_by_id = {str(r.get("id", "")).strip(): r for r in out_pass_rows}
    updated_base_count = 0
    student_rows = []
    for row in base_rows:
        sid = str(row.get("id", "")).strip()
        if sid in pass_by_id:
            merged = dict(row)
            # Keep original L_small rows untouched; replace only pseudo rows that were manually re-approved.
            if str(row.get("is_pseudo", "")).strip() == "1":
                merged.update(pass_by_id[sid])
                updated_base_count += 1
            student_rows.append(merged)
        else:
            student_rows.append(row)
    base_ids = {str(r.get("id", "")).strip() for r in base_rows}
    new_pass_rows = [r for r in out_pass_rows if str(r.get("id", "")).strip() not in base_ids]
    student_rows.extend(new_pass_rows)
    _write_csv(Path(args.output_student_manifest), student_rows, student_fields)

    summary = {
        "review_csv": str(review_path),
        "base_manifest": str(base_path),
        "num_review_rows": len(review_rows),
        "num_pass_rows_total": len(out_pass_rows),
        "num_pass_rows_updated_existing_base": int(updated_base_count),
        "num_pass_rows_added_new": len(new_pass_rows),
        "output_pass_manifest": str(args.output_pass_manifest),
        "output_student_manifest": str(args.output_student_manifest),
        "output_root": str(out_root),
        "pseudo_weight": float(args.pseudo_weight),
        "by_source": dict(Counter(str(r.get("source", "")) for r in out_pass_rows)),
        "changed_ratio_mean": float(np.mean(changed_pixels)) if changed_pixels else 0.0,
        "changed_ratio_max": float(np.max(changed_pixels)) if changed_pixels else 0.0,
    }
    summary_path = Path(args.output_student_manifest).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
