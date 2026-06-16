import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def _read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _resolve(path_text: str, anchor_path: Path) -> Path:
    p = Path(str(path_text or "").strip())
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    cands = [
        anchor_path.parent / p,
        anchor_path.parent.parent / p,
        Path.cwd() / p,
    ]
    for c in cands:
        if c.exists():
            return c.resolve()
    return cands[0].resolve()


def _ensure_odd(k: int) -> int:
    kk = int(max(1, k))
    return kk if kk % 2 == 1 else kk + 1


def _edge_from_prob(prob: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = np.clip(edge, 0.0, np.percentile(edge, 99.0) + 1e-6)
    edge = edge / (edge.max() + 1e-6)
    return edge.astype(np.float32)


def _edge_quality(mask_u8: np.ndarray, edge: np.ndarray) -> float:
    fg = (mask_u8 > 127).astype(np.uint8)
    area = int(fg.sum())
    if area <= 0:
        return 0.0
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = float(sum(cv2.arcLength(c, True) for c in contours)) + 1e-6
    shape_score = float(np.clip((4.0 * math.pi * float(area)) / (perim * perim), 0.0, 1.0))
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        largest = float(np.max(stats[1:, cv2.CC_STAT_AREA]))
    else:
        largest = float(area)
    comp_score = float(np.clip(largest / float(max(1, area)), 0.0, 1.0))
    edge_strength = float(np.clip(edge[fg > 0].mean() * 4.0, 0.0, 1.0))
    return float(np.clip(0.5 * edge_strength + 0.3 * comp_score + 0.2 * shape_score, 0.0, 1.0))


def _smooth_contours(mask_bin: np.ndarray, eps_ratio: float) -> np.ndarray:
    fg = (mask_bin > 0).astype(np.uint8)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(fg, dtype=np.uint8)
    for c in contours:
        if c is None or len(c) < 3:
            continue
        peri = float(cv2.arcLength(c, True))
        eps = max(1e-3, float(eps_ratio) * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        cv2.drawContours(out, [approx], -1, 1, thickness=-1)
    if int(out.sum()) <= 0:
        return fg
    return out


def _smooth_signed_distance(
    mask_bin: np.ndarray,
    sigma: float,
    preserve_area: bool,
) -> np.ndarray:
    fg = (mask_bin > 0).astype(np.uint8)
    area = int(fg.sum())
    if area <= 0 or float(sigma) <= 1e-6:
        return fg

    inside = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5)
    sdf = inside - outside
    sdf_sm = cv2.GaussianBlur(sdf, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))

    if preserve_area:
        flat = sdf_sm.reshape(-1)
        keep = int(np.clip(area, 1, flat.size))
        threshold = float(np.partition(flat, flat.size - keep)[flat.size - keep])
        out = (sdf_sm >= threshold).astype(np.uint8)
    else:
        out = (sdf_sm >= 0.0).astype(np.uint8)

    if int(out.sum()) <= 0:
        return fg
    return out


def _filter_components(
    mask_bin: np.ndarray,
    min_component_area_ratio: float,
    keep_max_components: int,
) -> tuple[np.ndarray, float, int]:
    fg = (mask_bin > 0).astype(np.uint8)
    h, w = fg.shape[:2]
    total = float(max(1, h * w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return fg, 0.0, 0

    comps = []
    for i in range(1, int(n)):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comps.append(
            {
                "label": i,
                "area": area,
                "area_ratio": float(area) / total,
            }
        )
    if not comps:
        return np.zeros_like(fg, dtype=np.uint8), 0.0, 0

    valid = [c for c in comps if c["area_ratio"] >= float(min_component_area_ratio)]
    if not valid:
        valid = sorted(comps, key=lambda x: x["area"], reverse=True)[:1]
    else:
        valid = sorted(valid, key=lambda x: x["area"], reverse=True)
        if int(keep_max_components) > 0:
            valid = valid[: int(keep_max_components)]

    out = np.zeros_like(fg, dtype=np.uint8)
    keep_labels = {int(c["label"]) for c in valid}
    keep_areas = []
    for lab in keep_labels:
        m = labels == lab
        out[m] = 1
        keep_areas.append(int(m.sum()))
    area_total = float(max(1, int(out.sum())))
    largest_cc_ratio = float(max(keep_areas) / area_total) if keep_areas else 0.0
    return out, float(np.clip(largest_cc_ratio, 0.0, 1.0)), int(len(keep_areas))


def _boundary_grad_and_compactness(mask_bin: np.ndarray, edge: np.ndarray) -> tuple[float, float]:
    fg = (mask_bin > 0).astype(np.uint8)
    if int(fg.sum()) <= 0:
        return 0.0, 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dil = cv2.dilate(fg, kernel, iterations=1)
    ero = cv2.erode(fg, kernel, iterations=1)
    band = (dil - ero) > 0
    boundary_grad = float(edge[band].mean()) if int(band.sum()) > 0 else 0.0

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = float(sum(cv2.arcLength(c, True) for c in contours)) + 1e-6
    area = float(fg.sum())
    compactness = float(np.clip((4.0 * math.pi * area) / (perim * perim), 0.0, 1.0))
    return float(np.clip(boundary_grad, 0.0, 1.0)), compactness


def _tier_and_scale(boundary_quality: float) -> tuple[str, float]:
    bq = float(np.clip(boundary_quality, 0.0, 1.0))
    if bq >= 0.72:
        return "high", 1.0
    if bq >= 0.58:
        return "mid", 0.6
    return "low", 0.25


def main():
    parser = argparse.ArgumentParser(description="Postprocess round1 pseudo labels and export quality_post-aware artifacts.")
    parser.add_argument("--quality-csv", type=str, required=True)
    parser.add_argument("--candidates-manifest", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--output-quality-csv", type=str, default="")
    parser.add_argument("--output-candidates-manifest", type=str, default="")
    parser.add_argument("--soft-sigma", type=float, default=1.0)
    parser.add_argument("--hard-threshold", type=float, default=0.5)
    parser.add_argument("--contour-eps-ratio", type=float, default=0.003)
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--min-component-area-ratio", type=float, default=0.0006)
    parser.add_argument("--keep-max-components", type=int, default=2)
    parser.add_argument("--soft-hard-blend", type=float, default=0.30)
    parser.add_argument("--ignore-soft-path", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--boundary-smooth-mode", type=str, default="poly", choices=["poly", "sdf", "none"])
    parser.add_argument("--sdf-sigma", type=float, default=1.8)
    parser.add_argument("--sdf-preserve-area", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    quality_csv = Path(args.quality_csv)
    candidates_manifest = Path(args.candidates_manifest)
    if not quality_csv.exists():
        raise FileNotFoundError(f"quality csv not found: {quality_csv}")
    if not candidates_manifest.exists():
        raise FileNotFoundError(f"candidates manifest not found: {candidates_manifest}")

    quality_rows, quality_fields = _read_csv(quality_csv)
    cand_rows, cand_fields = _read_csv(candidates_manifest)
    if not quality_rows:
        raise RuntimeError(f"Empty quality csv: {quality_csv}")
    if not cand_rows:
        raise RuntimeError(f"Empty candidates manifest: {candidates_manifest}")

    if int(args.max_samples) > 0:
        quality_rows = quality_rows[: int(args.max_samples)]

    default_out_root = quality_csv.parent.parent / "postprocessed"
    out_root = Path(args.output_root) if str(args.output_root).strip() else default_out_root
    out_quality_csv = (
        Path(args.output_quality_csv)
        if str(args.output_quality_csv).strip()
        else out_root / "pseudo_quality_post.csv"
    )
    out_cand_manifest = (
        Path(args.output_candidates_manifest)
        if str(args.output_candidates_manifest).strip()
        else out_root / "pseudo_candidates_manifest_post.csv"
    )

    hard_dir = out_root / "hard_masks"
    soft_dir = out_root / "soft_probs"
    edge_dir = out_root / "edge_probs"
    hard_dir.mkdir(parents=True, exist_ok=True)
    soft_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)

    open_k = _ensure_odd(int(args.open_kernel))
    close_k = _ensure_odd(int(args.close_kernel))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    quality_out_rows = []
    updated_by_id = {}

    for row in quality_rows:
        pid = str(row.get("id", "")).strip()
        if not pid:
            continue

        hard_path = _resolve(str(row.get("hard_mask_path", "")).strip(), quality_csv)
        soft_path_text = "" if args.ignore_soft_path else str(row.get("soft_path", "")).strip()
        soft_path = _resolve(soft_path_text, quality_csv) if soft_path_text else Path("")

        hard_img = cv2.imread(str(hard_path), cv2.IMREAD_GRAYSCALE) if hard_path.exists() else None
        if hard_img is None:
            raise RuntimeError(f"Cannot read hard mask for id={pid}: {hard_path}")
        h, w = hard_img.shape[:2]

        if soft_path_text and soft_path.exists():
            soft = np.load(str(soft_path)).astype(np.float32)
            if soft.shape != (h, w):
                soft = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)
            soft = np.clip(soft, 0.0, 1.0)
        else:
            soft = (hard_img > 127).astype(np.float32)

        if float(args.soft_sigma) > 1e-6:
            soft_sm = cv2.GaussianBlur(soft, (0, 0), sigmaX=float(args.soft_sigma), sigmaY=float(args.soft_sigma))
        else:
            soft_sm = soft
        soft_sm = np.clip(soft_sm, 0.0, 1.0).astype(np.float32)

        hard_bin = (soft_sm >= float(args.hard_threshold)).astype(np.uint8)
        if args.boundary_smooth_mode == "poly":
            hard_bin = _smooth_contours(hard_bin, eps_ratio=float(args.contour_eps_ratio))
        elif args.boundary_smooth_mode == "sdf":
            hard_bin = _smooth_signed_distance(
                hard_bin,
                sigma=float(args.sdf_sigma),
                preserve_area=bool(args.sdf_preserve_area),
            )
        hard_bin = cv2.morphologyEx(hard_bin, cv2.MORPH_OPEN, kernel_open, iterations=1)
        hard_bin = cv2.morphologyEx(hard_bin, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        hard_bin, largest_cc_ratio, num_components = _filter_components(
            hard_bin,
            min_component_area_ratio=float(args.min_component_area_ratio),
            keep_max_components=int(args.keep_max_components),
        )

        blend = float(np.clip(float(args.soft_hard_blend), 0.0, 1.0))
        soft_out = np.clip((1.0 - blend) * soft_sm + blend * hard_bin.astype(np.float32), 0.0, 1.0).astype(np.float32)
        edge_out = _edge_from_prob(soft_out)
        hard_u8 = (hard_bin > 0).astype(np.uint8) * 255

        edge_q = _edge_quality(hard_u8, edge_out)
        boundary_grad, compactness = _boundary_grad_and_compactness(hard_bin, edge_out)
        continuity = float(np.clip(largest_cc_ratio, 0.0, 1.0))
        boundary_quality = float(
            np.clip(
                0.45 * float(boundary_grad) + 0.35 * float(compactness) + 0.20 * float(continuity),
                0.0,
                1.0,
            )
        )
        quality_pre = _to_float(row.get("quality", 0.0), default=0.0)
        quality_post = float(np.clip(0.55 * quality_pre + 0.30 * boundary_quality + 0.15 * edge_q, 0.0, 1.0))
        tier, tier_scale = _tier_and_scale(boundary_quality)
        pseudo_weight_final = float(np.clip(quality_post * float(tier_scale), 0.0, 1.0))
        area_ratio = float(hard_bin.sum()) / float(max(1, h * w))

        out_hard = hard_dir / f"{pid}.png"
        out_soft = soft_dir / f"{pid}.npy"
        out_edge = edge_dir / f"{pid}.npy"
        cv2.imwrite(str(out_hard), hard_u8)
        np.save(out_soft, soft_out)
        np.save(out_edge, edge_out)

        qrow = dict(row)
        qrow["hard_mask_path"] = str(out_hard)
        qrow["soft_path"] = str(out_soft)
        qrow["edge_path"] = str(out_edge)
        qrow["quality_pre"] = float(quality_pre)
        qrow["quality_post"] = float(quality_post)
        qrow["quality"] = float(quality_post)
        qrow["edge_quality"] = float(edge_q)
        qrow["area_ratio"] = float(area_ratio)
        qrow["largest_cc_ratio"] = float(largest_cc_ratio)
        qrow["num_components"] = int(num_components)
        qrow["boundary_grad"] = float(boundary_grad)
        qrow["compactness"] = float(compactness)
        qrow["continuity"] = float(continuity)
        qrow["boundary_quality"] = float(boundary_quality)
        qrow["tier"] = str(tier)
        qrow["tier_scale"] = float(tier_scale)
        qrow["pseudo_weight_raw"] = float(quality_post)
        qrow["pseudo_weight_final"] = float(pseudo_weight_final)
        quality_out_rows.append(qrow)

        updated_by_id[pid] = {
            "mask_path": str(out_hard),
            "soft_path": str(out_soft),
            "edge_path": str(out_edge),
            "tier": str(tier),
            "pseudo_weight_raw": float(quality_post),
            "pseudo_weight_final": float(pseudo_weight_final),
            "pseudo_weight": float(pseudo_weight_final),
            "boundary_quality": float(boundary_quality),
            "quality_post": float(quality_post),
        }

    quality_extra_fields = [
        "quality_pre",
        "quality_post",
        "boundary_grad",
        "compactness",
        "continuity",
        "boundary_quality",
        "tier",
        "tier_scale",
        "pseudo_weight_raw",
        "pseudo_weight_final",
    ]
    quality_fieldnames = list(quality_fields)
    for k in ["hard_mask_path", "soft_path", "edge_path", "quality", "edge_quality", "area_ratio", "largest_cc_ratio", "num_components"]:
        if k not in quality_fieldnames:
            quality_fieldnames.append(k)
    for k in quality_extra_fields:
        if k not in quality_fieldnames:
            quality_fieldnames.append(k)
    _write_csv(out_quality_csv, quality_out_rows, quality_fieldnames)

    updated_cand_rows = []
    for r in cand_rows:
        pid = str(r.get("id", "")).strip()
        out = dict(r)
        if pid in updated_by_id:
            pack = updated_by_id[pid]
            out["mask_path"] = pack["mask_path"]
            out["soft_path"] = pack["soft_path"]
            out["edge_path"] = pack["edge_path"]
            out["tier"] = pack["tier"]
            out["pseudo_weight_raw"] = pack["pseudo_weight_raw"]
            out["pseudo_weight_final"] = pack["pseudo_weight_final"]
            out["pseudo_weight"] = pack["pseudo_weight"]
            out["boundary_quality"] = pack["boundary_quality"]
            out["quality_post"] = pack["quality_post"]
        updated_cand_rows.append(out)

    cand_fieldnames = list(cand_fields)
    for k in ["tier", "pseudo_weight_raw", "pseudo_weight_final", "boundary_quality", "quality_post", "soft_path", "edge_path"]:
        if k not in cand_fieldnames:
            cand_fieldnames.append(k)
    _write_csv(out_cand_manifest, updated_cand_rows, cand_fieldnames)

    bq_vals = [float(r.get("boundary_quality", 0.0)) for r in quality_out_rows]
    qpre_vals = [float(r.get("quality_pre", 0.0)) for r in quality_out_rows]
    qpost_vals = [float(r.get("quality_post", 0.0)) for r in quality_out_rows]
    summary = {
        "quality_csv": str(quality_csv),
        "candidates_manifest": str(candidates_manifest),
        "output_root": str(out_root),
        "output_quality_csv": str(out_quality_csv),
        "output_candidates_manifest": str(out_cand_manifest),
        "num_rows": int(len(quality_out_rows)),
        "boundary_quality_mean": float(np.mean(bq_vals)) if bq_vals else 0.0,
        "quality_pre_mean": float(np.mean(qpre_vals)) if qpre_vals else 0.0,
        "quality_post_mean": float(np.mean(qpost_vals)) if qpost_vals else 0.0,
        "soft_sigma": float(args.soft_sigma),
        "hard_threshold": float(args.hard_threshold),
        "contour_eps_ratio": float(args.contour_eps_ratio),
        "open_kernel": int(open_k),
        "close_kernel": int(close_k),
        "min_component_area_ratio": float(args.min_component_area_ratio),
        "keep_max_components": int(args.keep_max_components),
        "ignore_soft_path": bool(args.ignore_soft_path),
        "boundary_smooth_mode": str(args.boundary_smooth_mode),
        "sdf_sigma": float(args.sdf_sigma),
        "sdf_preserve_area": bool(args.sdf_preserve_area),
    }
    summary_path = out_root / "postprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
