import argparse
import csv
import json
import math
from html import escape
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


def _resolve(path_text: str, base: Path | None = None) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    if base is not None:
        p2 = base / p
        if p2.exists():
            return p2
    return (Path.cwd() / p).resolve()


def _load_manifest_rows(manifest_path: str, subset_filter: set[str]) -> list[dict]:
    out: list[dict] = []
    mpath = Path(manifest_path)
    with open(mpath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subset = str(row.get("subset", "")).strip()
            if subset_filter and subset not in subset_filter:
                continue
            img_text = str(row.get("image_path", "")).strip()
            if not img_text:
                continue
            img = _resolve(img_text, mpath.parent)
            if not img.exists():
                continue
            pid = str(row.get("id", "")).strip() or img.stem
            out.append(
                {
                    "id": pid,
                    "image_path": str(img),
                    "subset": subset,
                    "source": str(row.get("source", "")).strip(),
                    "center": str(row.get("center", "")).strip(),
                }
            )
    return out


def _load_dir_rows(image_dir: str) -> list[dict]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    out = []
    for p in sorted([x for x in Path(image_dir).glob("*") if x.suffix.lower() in exts]):
        out.append({"id": p.stem, "image_path": str(p), "subset": "U_large", "source": "", "center": ""})
    return out


def _load_id_filter(path: str) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x) for x in data}
        if isinstance(data, dict) and "ids" in data and isinstance(data["ids"], list):
            return {str(x) for x in data["ids"]}
        return set()
    ids = set()
    with open(p, "r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if text:
                ids.add(text.split(",")[0])
    return ids


def _load_proposals(path: str) -> dict[str, list[float]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        if isinstance(v, list) and len(v) == 4:
            out[str(k)] = [float(x) for x in v]
    return out


def _edge_from_prob(prob: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = np.clip(edge, 0.0, np.percentile(edge, 99.0) + 1e-6)
    edge = edge / (edge.max() + 1e-6)
    return edge.astype(np.float32)


def _edge_quality(mask: np.ndarray, edge: np.ndarray) -> float:
    fg = (mask > 0).astype(np.uint8)
    if fg.sum() == 0:
        return 0.0

    num_cc, _ = cv2.connectedComponents(fg)
    comp_score = 1.0 / (1.0 + max(0, num_cc - 2))

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = float(sum(cv2.arcLength(c, True) for c in contours)) + 1e-6
    area = float(fg.sum())
    shape_score = np.clip((area / perim) / 6.0, 0.0, 1.0)

    edge_strength = float(np.clip(edge.mean() * 4.0, 0.0, 1.0))
    return float(np.clip(0.5 * edge_strength + 0.3 * comp_score + 0.2 * shape_score, 0.0, 1.0))


def _mask_area_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float((mask > 0).mean())


def _area_prior(area_ratio: float, min_ratio: float, target_ratio: float, max_ratio: float) -> float:
    a = float(area_ratio)
    lo = float(min_ratio)
    mid = float(target_ratio)
    hi = float(max_ratio)
    if hi <= lo:
        return 0.0
    if a < lo or a > hi:
        return 0.0
    if mid <= lo:
        return 1.0 - max(0.0, min(1.0, (a - lo) / (hi - lo + 1e-6)))
    if mid >= hi:
        return max(0.0, min(1.0, (a - lo) / (hi - lo + 1e-6)))
    if a <= mid:
        return max(0.0, min(1.0, (a - lo) / (mid - lo + 1e-6)))
    return max(0.0, min(1.0, (hi - a) / (hi - mid + 1e-6)))


def _center_prior(mask: np.ndarray) -> float:
    fg = np.where(mask > 0)
    if len(fg[0]) == 0:
        return 0.0
    cy = float(np.mean(fg[0]))
    cx = float(np.mean(fg[1]))
    h, w = mask.shape[:2]
    my = 0.5 * (h - 1)
    mx = 0.5 * (w - 1)
    dist = math.sqrt((cy - my) ** 2 + (cx - mx) ** 2)
    max_dist = math.sqrt(my * my + mx * mx) + 1e-6
    return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 0.0:
        return 1.0
    return float(inter / max(1e-6, union))


def _box_binary_mask(h: int, w: int, box_xyxy: list[float]) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = [int(round(float(v))) for v in box_xyxy]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if x1 < x0 or y1 < y0:
        return out
    out[y0 : y1 + 1, x0 : x1 + 1] = 1
    return out


def _mask_geometry_metrics(mask_bin_u8: np.ndarray, box_xyxy: list[float], reflection_map: np.ndarray | None) -> dict:
    fg = mask_bin_u8 > 0
    area = int(fg.sum())
    h, w = mask_bin_u8.shape[:2]
    if area <= 0:
        return {
            "box_in_ratio": 0.0,
            "spill_ratio": 1.0,
            "reflection_overlap": 0.0,
            "largest_cc_ratio": 0.0,
            "num_components": 0,
        }

    box_mask = _box_binary_mask(h=h, w=w, box_xyxy=box_xyxy) > 0
    in_box = int(np.logical_and(fg, box_mask).sum())
    box_in_ratio = float(in_box) / float(max(1, area))
    spill_ratio = 1.0 - box_in_ratio

    reflection_overlap = 0.0
    if reflection_map is not None and reflection_map.shape == fg.shape:
        reflection_overlap = float(np.logical_and(fg, reflection_map > 0).sum()) / float(max(1, area))

    fg_u8 = fg.astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    num_comp = max(0, int(num_labels) - 1)
    largest_cc_area = 0
    if num_comp > 0:
        largest_cc_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    largest_cc_ratio = float(largest_cc_area) / float(max(1, area))
    return {
        "box_in_ratio": float(np.clip(box_in_ratio, 0.0, 1.0)),
        "spill_ratio": float(np.clip(spill_ratio, 0.0, 1.0)),
        "reflection_overlap": float(np.clip(reflection_overlap, 0.0, 1.0)),
        "largest_cc_ratio": float(np.clip(largest_cc_ratio, 0.0, 1.0)),
        "num_components": int(num_comp),
    }


def _postprocess_mask(
    mask_bin_u8: np.ndarray,
    box_xyxy: list[float],
    reflection_map: np.ndarray | None,
    min_component_area_ratio: float,
    min_inbox_ratio: float,
    max_reflection_overlap: float,
    keep_max_components: int,
) -> np.ndarray:
    fg = (mask_bin_u8 > 0).astype(np.uint8)
    h, w = fg.shape[:2]
    total = float(max(1, h * w))
    if int(fg.sum()) == 0:
        return fg * 255

    box_mask = _box_binary_mask(h=h, w=w, box_xyxy=box_xyxy) > 0
    refl = None
    if reflection_map is not None and reflection_map.shape == fg.shape:
        refl = reflection_map > 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    comps = []
    for lab in range(1, int(num_labels)):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp = labels == lab
        area_ratio = float(area) / total
        in_box = float(np.logical_and(comp, box_mask).sum()) / float(max(1, area))
        refl_overlap = 0.0
        if refl is not None:
            refl_overlap = float(np.logical_and(comp, refl).sum()) / float(max(1, area))
        score = float(area) * (0.5 + 0.5 * in_box) * (1.0 - 0.5 * refl_overlap)
        comps.append(
            {
                "label": int(lab),
                "area": area,
                "area_ratio": float(area_ratio),
                "in_box": float(in_box),
                "refl_overlap": float(refl_overlap),
                "score": float(score),
            }
        )

    if not comps:
        return fg * 255

    min_comp = float(max(0.0, min_component_area_ratio))
    min_inb = float(np.clip(min_inbox_ratio, 0.0, 1.0))
    max_refl = float(np.clip(max_reflection_overlap, 0.0, 1.0))
    valid = [
        c
        for c in comps
        if c["area_ratio"] >= min_comp and c["in_box"] >= min_inb and c["refl_overlap"] <= max_refl
    ]
    if not valid:
        valid = sorted(comps, key=lambda x: (x["score"], x["area"]), reverse=True)[:1]
    else:
        valid = sorted(valid, key=lambda x: (x["score"], x["area"]), reverse=True)
        if int(keep_max_components) > 0:
            valid = valid[: int(keep_max_components)]

    out = np.zeros_like(fg, dtype=np.uint8)
    keep_labs = {int(c["label"]) for c in valid}
    for lab in keep_labs:
        out[labels == lab] = 1
    return out * 255


def _candidate_consistency(best_pack: dict, candidate_pool: list[dict], top_k: int) -> tuple[float, float]:
    if not candidate_pool:
        return 1.0, 1.0
    ranked = sorted(candidate_pool, key=lambda x: float(x.get("quality", 0.0)), reverse=True)
    if len(ranked) == 1:
        return 1.0, 1.0
    best = ranked[0]
    comp = ranked[1 : max(2, int(top_k))]
    ious = [_mask_iou(best["hard"], p["hard"]) for p in comp if "hard" in p]
    consistency = float(np.mean(ious)) if ious else 1.0
    gap = float(best.get("quality", 0.0)) - float(ranked[1].get("quality", 0.0))
    return float(np.clip(consistency, 0.0, 1.0)), float(max(0.0, gap))


def _reflection_map(image_bgr: np.ndarray, v_thresh: int, s_thresh: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    m = ((val >= int(v_thresh)) & (sat <= int(s_thresh))).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return m


def _box_from_mask(mask_bin: np.ndarray, expand_ratio: float, w: int, h: int, min_side: float = 8.0) -> list[float]:
    fg = mask_bin > 0
    if int(fg.sum()) == 0:
        return [0.0, 0.0, float(max(0, w - 1)), float(max(0, h - 1))]
    ys, xs = np.where(fg)
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max())
    y1 = float(ys.max())
    bw = max(1.0, x1 - x0 + 1.0)
    bh = max(1.0, y1 - y0 + 1.0)
    ex = float(max(0.0, expand_ratio)) * bw
    ey = float(max(0.0, expand_ratio)) * bh
    x0 = max(0.0, x0 - ex)
    y0 = max(0.0, y0 - ey)
    x1 = min(float(max(0, w - 1)), x1 + ex)
    y1 = min(float(max(0, h - 1)), y1 + ey)

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    if (x1 - x0 + 1.0) < float(min_side):
        half = 0.5 * float(min_side)
        x0 = max(0.0, cx - half)
        x1 = min(float(max(0, w - 1)), cx + half)
    if (y1 - y0 + 1.0) < float(min_side):
        half = 0.5 * float(min_side)
        y0 = max(0.0, cy - half)
        y1 = min(float(max(0, h - 1)), cy + half)
    return [float(x0), float(y0), float(x1), float(y1)]


def _topk_points_from_mask(prob: np.ndarray, cand: np.ndarray, k: int, descending: bool) -> list[tuple[float, float]]:
    if k <= 0:
        return []
    ys, xs = np.where(cand)
    if len(xs) == 0:
        return []
    vals = prob[ys, xs]
    order = np.argsort(vals)
    if descending:
        order = order[::-1]
    out = []
    used = set()
    for idx in order:
        x = float(xs[idx])
        y = float(ys[idx])
        key = (int(round(x)), int(round(y)))
        if key in used:
            continue
        used.add(key)
        out.append((x, y))
        if len(out) >= int(k):
            break
    return out


def _build_two_pass_points(
    soft_prob: np.ndarray,
    hard_mask: np.ndarray,
    image_bgr: np.ndarray,
    box_xyxy: list[float],
    pos_points: int,
    neg_points: int,
    use_reflection_neg: bool,
    reflection_v_threshold: int,
    reflection_s_threshold: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    h, w = hard_mask.shape[:2]
    fg = hard_mask > 0
    if int(fg.sum()) == 0:
        return None, None

    points: list[tuple[float, float]] = []
    labels: list[int] = []
    used = set()

    ys, xs = np.where(fg)
    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    ckey = (int(round(cx)), int(round(cy)))
    used.add(ckey)
    points.append((cx, cy))
    labels.append(1)

    extra_pos = max(0, int(pos_points) - 1)
    for x, y in _topk_points_from_mask(soft_prob, fg, k=extra_pos * 3, descending=True):
        key = (int(round(x)), int(round(y)))
        if key in used:
            continue
        used.add(key)
        points.append((x, y))
        labels.append(1)
        if len(labels) >= int(pos_points):
            break

    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    roi = np.zeros((h, w), dtype=bool)
    roi[y0 : y1 + 1, x0 : x1 + 1] = True
    neg_cand = roi & (~fg)

    if use_reflection_neg:
        refl = _reflection_map(
            image_bgr=image_bgr,
            v_thresh=reflection_v_threshold,
            s_thresh=reflection_s_threshold,
            enabled=True,
        )
        neg_reflect = neg_cand & (refl > 0)
        neg_pts = _topk_points_from_mask(soft_prob, neg_reflect, k=int(neg_points), descending=True)
    else:
        neg_pts = []
    if len(neg_pts) < int(neg_points):
        remain = int(neg_points) - len(neg_pts)
        neg_more = _topk_points_from_mask(soft_prob, neg_cand, k=remain * 3, descending=True)
        for p in neg_more:
            if p not in neg_pts:
                neg_pts.append(p)
            if len(neg_pts) >= int(neg_points):
                break

    for x, y in neg_pts[: int(neg_points)]:
        key = (int(round(x)), int(round(y)))
        if key in used:
            continue
        used.add(key)
        points.append((x, y))
        labels.append(0)

    if not points:
        return None, None
    point_coords = np.array(points, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int32)
    return point_coords, point_labels


def _parse_scales(text: str) -> list[float]:
    vals = []
    for part in str(text).split(","):
        t = part.strip()
        if not t:
            continue
        try:
            v = float(t)
        except ValueError:
            continue
        if 0.1 <= v <= 1.0:
            vals.append(v)
    if not vals:
        vals = [1.0, 0.85, 0.7, 0.55]
    if 1.0 not in vals:
        vals = [1.0] + vals
    seen = set()
    out = []
    for v in vals:
        key = round(v, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _parse_centers(text: str) -> list[float]:
    vals = []
    for part in str(text).split(","):
        t = part.strip()
        if not t:
            continue
        try:
            v = float(t)
        except ValueError:
            continue
        if 0.05 <= v <= 0.95:
            vals.append(v)
    if not vals:
        vals = [0.3, 0.5, 0.7]
    seen = set()
    out = []
    for v in vals:
        key = round(v, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _parse_offsets(text: str) -> list[float]:
    vals = []
    for part in str(text).split(","):
        t = part.strip()
        if not t:
            continue
        try:
            v = float(t)
        except ValueError:
            continue
        if -0.5 <= v <= 0.5:
            vals.append(v)
    if not vals:
        vals = [0.0]
    seen = set()
    out = []
    for v in vals:
        key = round(v, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _box_from_center_scale(w: int, h: int, scale: float, center_x: float, center_y: float) -> list[float]:
    cx = float(center_x) * float(max(1, w - 1))
    cy = float(center_y) * float(max(1, h - 1))
    bw = max(8.0, (w - 1) * float(scale))
    bh = max(8.0, (h - 1) * float(scale))
    x0 = max(0.0, cx - 0.5 * bw)
    y0 = max(0.0, cy - 0.5 * bh)
    x1 = min(float(w - 1), cx + 0.5 * bw)
    y1 = min(float(h - 1), cy + 0.5 * bh)
    return [x0, y0, x1, y1]


def _normalize01(x: np.ndarray) -> np.ndarray:
    arr = x.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, 2.0))
    hi = float(np.percentile(arr, 98.0))
    if hi <= lo + 1e-6:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi <= lo + 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def _content_attention_map(
    image_bgr: np.ndarray,
    reflection_v_threshold: int,
    reflection_s_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    b = image_bgr[:, :, 0].astype(np.float32)
    g = image_bgr[:, :, 1].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    lap = np.abs(cv2.Laplacian(gray_eq, cv2.CV_32F, ksize=3)).astype(np.float32)

    red_dom = np.clip(r - 0.62 * g - 0.28 * b, 0.0, 255.0)
    a_pos = np.clip(a - 128.0, 0.0, 127.0)

    score = (
        0.36 * _normalize01(a_pos)
        + 0.27 * _normalize01(red_dom)
        + 0.19 * _normalize01(sat)
        + 0.18 * _normalize01(lap)
    )
    score = cv2.GaussianBlur(score, (0, 0), sigmaX=3.0, sigmaY=3.0)

    refl = _reflection_map(
        image_bgr=image_bgr,
        v_thresh=int(reflection_v_threshold),
        s_thresh=int(reflection_s_threshold),
        enabled=True,
    )
    score = np.clip(score * (1.0 - 0.72 * (refl > 0).astype(np.float32)), 0.0, 1.0)
    return score.astype(np.float32), refl.astype(np.uint8)


def _content_aware_boxes(
    image_bgr: np.ndarray,
    min_area_ratio: float,
    target_area_ratio: float,
    max_area_ratio: float,
    max_candidates: int,
    reflection_v_threshold: int,
    reflection_s_threshold: int,
) -> list[list[float]]:
    h, w = image_bgr.shape[:2]
    total = float(max(1, h * w))
    attn, refl = _content_attention_map(
        image_bgr=image_bgr,
        reflection_v_threshold=reflection_v_threshold,
        reflection_s_threshold=reflection_s_threshold,
    )
    q_list = [0.96, 0.94, 0.92, 0.90, 0.88, 0.85]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    scored: list[tuple[float, list[float]]] = []
    lo_area = float(max(1e-6, min_area_ratio * 0.30))
    hi_area = float(min(1.0, max_area_ratio * 1.40))

    for q in q_list:
        thr = float(np.quantile(attn, q))
        fg = (attn >= thr).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        for i in range(1, int(num_labels)):
            x, y, bw, bh, area = stats[i].tolist()
            area_ratio = float(area) / total
            if area_ratio < lo_area or area_ratio > hi_area:
                continue
            m = labels == i
            if not np.any(m):
                continue
            mean_attn = float(attn[m].mean())
            refl_overlap = float((refl[m] > 0).mean())
            compactness = float(area) / float(max(1, bw * bh))
            cx = float(x + 0.5 * (bw - 1))
            cy = float(y + 0.5 * (bh - 1))
            center_dist = math.sqrt((cx - 0.5 * (w - 1)) ** 2 + (cy - 0.5 * (h - 1)) ** 2)
            max_dist = math.sqrt((0.5 * (w - 1)) ** 2 + (0.5 * (h - 1)) ** 2) + 1e-6
            center_score = float(np.clip(1.0 - center_dist / max_dist, 0.0, 1.0))

            bbox_area_ratio = float(max(1, bw * bh)) / total
            area_p = _area_prior(
                bbox_area_ratio,
                min_ratio=max(1e-6, min_area_ratio * 0.60),
                target_ratio=max(1e-6, target_area_ratio * 1.10),
                max_ratio=min(1.0, max_area_ratio * 1.50),
            )
            score = (
                0.46 * mean_attn
                + 0.20 * float(np.clip(compactness, 0.0, 1.0))
                + 0.18 * float(area_p)
                + 0.12 * center_score
                - 0.28 * refl_overlap
            )

            margin = max(0.04, 0.12 + 0.16 * max(0.0, 0.55 - mean_attn))
            dx = float(bw) * margin
            dy = float(bh) * margin
            x0 = max(0.0, float(x) - dx)
            y0 = max(0.0, float(y) - dy)
            x1 = min(float(w - 1), float(x + bw - 1) + dx)
            y1 = min(float(h - 1), float(y + bh - 1) + dy)
            box = [x0, y0, x1, y1]
            scored.append((float(score), box))

    scored.sort(key=lambda x: float(x[0]), reverse=True)
    dedup: dict[tuple[float, float, float, float], list[float]] = {}
    for _, box in scored:
        key = tuple(round(float(v), 1) for v in box)
        if key in dedup:
            continue
        dedup[key] = [float(v) for v in box]
        if max_candidates > 0 and len(dedup) >= int(max_candidates):
            break
    out = list(dedup.values())
    if not out:
        out = [[0.0, 0.0, float(w - 1), float(h - 1)]]
    return out


def _build_auto_boxes(
    w: int,
    h: int,
    scales: list[float],
    centers: list[float],
    mode: str,
    max_candidates: int,
    image_bgr: np.ndarray | None = None,
    min_area_ratio: float = 0.002,
    target_area_ratio: float = 0.08,
    max_area_ratio: float = 0.35,
    reflection_v_threshold: int = 245,
    reflection_s_threshold: int = 35,
) -> list[list[float]]:
    mode = str(mode).strip().lower()
    if mode == "single_box":
        return [[0.0, 0.0, float(w - 1), float(h - 1)]]
    boxes = []
    if mode == "multi_box":
        for s in scales:
            boxes.append(_box_from_center_scale(w, h, s, center_x=0.5, center_y=0.5))
    elif mode == "grid_multi_box":
        for s in scales:
            for cy in centers:
                for cx in centers:
                    boxes.append(_box_from_center_scale(w, h, s, center_x=cx, center_y=cy))
    elif mode == "content_multi_box":
        if image_bgr is None:
            boxes.append([0.0, 0.0, float(w - 1), float(h - 1)])
        else:
            boxes.extend(
                _content_aware_boxes(
                    image_bgr=image_bgr,
                    min_area_ratio=min_area_ratio,
                    target_area_ratio=target_area_ratio,
                    max_area_ratio=max_area_ratio,
                    max_candidates=max_candidates,
                    reflection_v_threshold=reflection_v_threshold,
                    reflection_s_threshold=reflection_s_threshold,
                )
            )
    elif mode == "hybrid_multi_box":
        if image_bgr is not None:
            boxes.extend(
                _content_aware_boxes(
                    image_bgr=image_bgr,
                    min_area_ratio=min_area_ratio,
                    target_area_ratio=target_area_ratio,
                    max_area_ratio=max_area_ratio,
                    max_candidates=max(8, int(max_candidates)),
                    reflection_v_threshold=reflection_v_threshold,
                    reflection_s_threshold=reflection_s_threshold,
                )
            )
        for s in scales:
            boxes.append(_box_from_center_scale(w, h, s, center_x=0.5, center_y=0.5))
        for s in scales[: max(1, min(3, len(scales)))]:
            for cy in centers:
                for cx in centers:
                    if cx == 0.5 and cy == 0.5:
                        continue
                    boxes.append(_box_from_center_scale(w, h, s, center_x=cx, center_y=cy))
    else:
        raise ValueError(f"Unsupported auto proposal mode: {mode}")

    dedup = {}
    for b in boxes:
        key = tuple(round(float(v), 2) for v in b)
        dedup[key] = [float(v) for v in b]
    out = list(dedup.values())
    if max_candidates > 0:
        out = out[: int(max_candidates)]
    if not out:
        out = [[0.0, 0.0, float(w - 1), float(h - 1)]]
    if [0.0, 0.0, float(w - 1), float(h - 1)] not in out:
        out.append([0.0, 0.0, float(w - 1), float(h - 1)])
    if max_candidates > 0:
        out = out[: int(max_candidates)]
    return out


def _clip_box(box: list[float], w: int, h: int) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = float(max(0.0, min(float(w - 1), x0)))
    y0 = float(max(0.0, min(float(h - 1), y0)))
    x1 = float(max(0.0, min(float(w - 1), x1)))
    y1 = float(max(0.0, min(float(h - 1), y1)))
    if x1 <= x0:
        x1 = float(min(float(w - 1), x0 + 1.0))
    if y1 <= y0:
        y1 = float(min(float(h - 1), y0 + 1.0))
    return [x0, y0, x1, y1]


def _build_augmented_proposal_boxes(
    preset_box: list[float],
    w: int,
    h: int,
    scales: list[float],
    shifts: list[float],
    max_boxes: int,
) -> list[list[float]]:
    base = _clip_box(preset_box, w=w, h=h)
    bx0, by0, bx1, by1 = base
    bw = float(max(2.0, bx1 - bx0 + 1.0))
    bh = float(max(2.0, by1 - by0 + 1.0))
    cx0 = 0.5 * (bx0 + bx1)
    cy0 = 0.5 * (by0 + by1)

    boxes = [base]
    for s in scales:
        sc = float(max(0.4, min(1.8, s)))
        bw_s = bw * sc
        bh_s = bh * sc
        for dy in shifts:
            for dx in shifts:
                cx = cx0 + float(dx) * bw
                cy = cy0 + float(dy) * bh
                x0 = cx - 0.5 * bw_s
                y0 = cy - 0.5 * bh_s
                x1 = cx + 0.5 * bw_s
                y1 = cy + 0.5 * bh_s
                boxes.append(_clip_box([x0, y0, x1, y1], w=w, h=h))

    dedup = {}
    for b in boxes:
        key = tuple(round(float(v), 1) for v in b)
        dedup[key] = [float(v) for v in b]
    out = list(dedup.values())
    if max_boxes > 0:
        out = out[: int(max_boxes)]
    return out


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        dev = base.weight.device
        dt = base.weight.dtype
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features, device=dev, dtype=dt))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = (x @ self.lora_a.t()) @ self.lora_b.t()
        return base_out + self.scale * delta


def inject_lora(module: nn.Module, target_keywords: list[str], rank: int, alpha: int) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(k in name.lower() for k in target_keywords):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            replaced += 1
        else:
            replaced += inject_lora(child, target_keywords, rank, alpha)
    return replaced


def _overlay_mask(image_bgr: np.ndarray, mask_bin: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    image_f = image_bgr.astype(np.float32)
    color = np.zeros_like(image_f)
    color[..., 2] = 255.0
    mask3 = (mask_bin > 0)[..., None]
    blended = image_f * (1.0 - alpha) + color * alpha
    out = np.where(mask3, blended, image_f)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _build_panel(image_bgr: np.ndarray, hard_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask_gray = hard_mask if hard_mask.ndim == 2 else hard_mask.squeeze()
    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    overlay = _overlay_mask(image_bgr, mask_gray, alpha=alpha)
    panel = np.concatenate([image_bgr, mask_bgr, overlay], axis=1)
    h = panel.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "image", (8, min(h - 8, 24)), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        "mask",
        (image_bgr.shape[1] + 8, min(h - 8, 24)),
        font,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "overlay",
        (image_bgr.shape[1] * 2 + 8, min(h - 8, 24)),
        font,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _path_to_uri(path_text: str) -> str:
    if not path_text:
        return ""
    try:
        return Path(path_text).resolve().as_uri()
    except Exception:
        return path_text


def _write_gallery(rows: list[dict], output_html: Path):
    output_html.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda x: float(x.get("quality", 0.0)), reverse=True)
    cards = []
    for row in sorted_rows:
        panel_path = str(row.get("panel_path", "")).strip()
        hard_path = str(row.get("hard_mask_path", "")).strip()
        image_path = str(row.get("image_path", "")).strip()
        preview_uri = _path_to_uri(panel_path if panel_path else hard_path)
        hard_uri = _path_to_uri(hard_path)
        image_uri = _path_to_uri(image_path)
        pid = escape(str(row.get("id", "")))
        conf = float(row.get("conf", 0.0))
        edge_q = float(row.get("edge_quality", 0.0))
        area_ratio = float(row.get("area_ratio", 0.0))
        area_prior = float(row.get("area_prior", 0.0))
        quality = float(row.get("quality", 0.0))
        cards.append(
            (
                "<div class='card'>"
                f"<div class='id'>{pid}</div>"
                f"<img src='{preview_uri}' loading='lazy' alt='{pid}' />"
                f"<div class='meta'>conf={conf:.4f} edge={edge_q:.4f} area={area_ratio:.4f} area_prior={area_prior:.4f} quality={quality:.4f}</div>"
                f"<div class='links'><a href='{image_uri}'>image</a> | <a href='{hard_uri}'>mask</a></div>"
                "</div>"
            )
        )

    html = (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<title>Pseudo Mask Gallery</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:16px;background:#f6f8fb;color:#111;}"
        "h1{margin:0 0 8px 0;font-size:22px;}"
        "p{margin:0 0 16px 0;color:#555;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px;}"
        ".card{background:#fff;border:1px solid #dde3ea;border-radius:8px;padding:10px;box-shadow:0 1px 2px rgba(0,0,0,.05);}"
        ".id{font-weight:700;font-size:13px;margin-bottom:8px;word-break:break-all;}"
        "img{width:100%;height:auto;border-radius:6px;border:1px solid #e5e9ef;background:#000;}"
        ".meta{margin-top:8px;font-size:12px;color:#333;}"
        ".links{margin-top:6px;font-size:12px;}"
        "a{color:#0b5ed7;text-decoration:none;}a:hover{text-decoration:underline;}"
        "</style></head><body>"
        f"<h1>Pseudo Mask Gallery</h1><p>Total samples: {len(sorted_rows)}</p>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>"
    )
    output_html.write_text(html, encoding="utf-8")


def _safe_torch_load(path: str, map_location: str | torch.device | None = None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo labels with MedSAM teacher.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Base MedSAM checkpoint.")
    parser.add_argument("--model-type", type=str, default="vit_b")
    parser.add_argument("--lora-checkpoint", type=str, default="", help="LoRA-adapted teacher checkpoint.")
    parser.add_argument("--data-manifest", type=str, default="")
    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--subset-filter", type=str, default="U_large")
    parser.add_argument("--id-filter", type=str, default="", help="Optional txt/json list of ids to process.")
    parser.add_argument("--proposal-json", type=str, default="", help="Optional id->box mapping.")
    parser.add_argument(
        "--proposal-mix-mode",
        type=str,
        default="replace",
        choices=["replace", "augment", "augment_plus_auto"],
        help="How to use preset proposal boxes when --proposal-json is provided.",
    )
    parser.add_argument(
        "--proposal-jitter-scales",
        type=str,
        default="1.0,0.9,1.15",
        help="Scale jitters around preset proposal box.",
    )
    parser.add_argument(
        "--proposal-jitter-shifts",
        type=str,
        default="0.0,-0.08,0.08",
        help="Center shifts around preset proposal box (fraction of preset box size).",
    )
    parser.add_argument(
        "--proposal-jitter-max-boxes",
        type=int,
        default=27,
        help="Max augmented candidates generated from one preset proposal.",
    )
    parser.add_argument(
        "--append-auto-candidates",
        type=str,
        default="off",
        choices=["off", "all", "polypgen"],
        help="When preset proposals are used, optionally append auto grid candidates.",
    )
    parser.add_argument(
        "--append-auto-max-candidates",
        type=int,
        default=12,
        help="Max appended auto candidates when --append-auto-candidates is enabled.",
    )
    parser.add_argument("--fallback-full-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--auto-proposal-mode",
        type=str,
        default="multi_box",
        choices=["single_box", "multi_box", "grid_multi_box", "content_multi_box", "hybrid_multi_box"],
    )
    parser.add_argument("--candidate-box-scales", type=str, default="1.0,0.85,0.7,0.55")
    parser.add_argument("--candidate-box-centers", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--max-candidate-boxes", type=int, default=40)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.002)
    parser.add_argument("--target-mask-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.35)
    parser.add_argument("--score-weight-conf", type=float, default=0.45)
    parser.add_argument("--score-weight-edge", type=float, default=0.25)
    parser.add_argument("--score-weight-area-prior", type=float, default=0.20)
    parser.add_argument("--score-weight-center-prior", type=float, default=0.10)
    parser.add_argument(
        "--score-bias-preset",
        type=float,
        default=0.0,
        help="Additive quality bias for preset/preset_aug prompt source.",
    )
    parser.add_argument(
        "--score-bias-auto",
        type=float,
        default=0.0,
        help="Additive quality bias for auto prompt source.",
    )
    parser.add_argument(
        "--score-bias-auto-polypgen",
        type=float,
        default=-999.0,
        help="If > -900, override auto prompt bias when source is PolypGen.",
    )
    parser.add_argument(
        "--score-weight-center-prior-polypgen",
        type=float,
        default=-1.0,
        help="If >=0, override center-prior weight for PolypGen source.",
    )
    parser.add_argument("--two-pass-refine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--two-pass-min-first-quality", type=float, default=0.56)
    parser.add_argument("--two-pass-min-first-area-ratio", type=float, default=0.003)
    parser.add_argument("--two-pass-min-gain", type=float, default=0.0)
    parser.add_argument("--two-pass-score-bonus", type=float, default=0.01)
    parser.add_argument("--two-pass-box-expand-ratio", type=float, default=0.12)
    parser.add_argument("--two-pass-pos-points", type=int, default=2)
    parser.add_argument("--two-pass-neg-points", type=int, default=2)
    parser.add_argument("--two-pass-use-reflection-neg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--two-pass-reflection-v-threshold", type=int, default=245)
    parser.add_argument("--two-pass-reflection-s-threshold", type=int, default=35)
    parser.add_argument("--quality-use-reflection-penalty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-reflection-v-threshold", type=int, default=245)
    parser.add_argument("--quality-reflection-s-threshold", type=int, default=35)
    parser.add_argument("--quality-penalty-spill-weight", type=float, default=0.18)
    parser.add_argument("--quality-penalty-reflection-weight", type=float, default=0.12)
    parser.add_argument("--quality-penalty-fragment-weight", type=float, default=0.08)
    parser.add_argument("--quality-penalty-consistency-weight", type=float, default=0.18)
    parser.add_argument("--consistency-topk", type=int, default=5)
    parser.add_argument("--consistency-min-iou", type=float, default=0.55)
    parser.add_argument("--postprocess-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-min-component-area-ratio", type=float, default=0.0005)
    parser.add_argument("--post-min-inbox-ratio", type=float, default=0.55)
    parser.add_argument("--post-max-reflection-overlap", type=float, default=0.35)
    parser.add_argument("--post-keep-max-components", type=int, default=2)
    parser.add_argument("--write-candidate-scores", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--round-id", type=int, default=1)
    parser.add_argument("--output-root", type=str, default="runs/flywheel/round1/pseudo")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-exist", action="store_true")
    parser.add_argument("--save-panels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--write-gallery", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_root = Path(args.output_root)
    hard_dir = out_root / "hard_masks"
    soft_dir = out_root / "soft_probs"
    edge_dir = out_root / "edge_probs"
    panel_dir = out_root / "mask_panels"
    out_root.mkdir(parents=True, exist_ok=True)
    hard_dir.mkdir(parents=True, exist_ok=True)
    soft_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)
    if args.save_panels:
        panel_dir.mkdir(parents=True, exist_ok=True)

    subset_filter = {s.strip() for s in args.subset_filter.split(",") if s.strip()}
    if args.data_manifest:
        rows = _load_manifest_rows(args.data_manifest, subset_filter)
    elif args.image_dir:
        rows = _load_dir_rows(args.image_dir)
    else:
        raise ValueError("Provide --data-manifest or --image-dir")

    id_filter = _load_id_filter(args.id_filter)
    if id_filter:
        rows = [r for r in rows if r["id"] in id_filter]
    if not rows:
        raise RuntimeError("No images to process.")

    proposals = _load_proposals(args.proposal_json)
    candidate_scales = _parse_scales(args.candidate_box_scales)
    candidate_centers = _parse_centers(args.candidate_box_centers)
    proposal_jitter_scales = _parse_scales(args.proposal_jitter_scales)
    proposal_jitter_shifts = _parse_offsets(args.proposal_jitter_shifts)

    model = sam_model_registry[args.model_type](checkpoint=None).to(args.device)
    base_state = _safe_torch_load(args.checkpoint, map_location=args.device)
    if isinstance(base_state, dict) and "state_dict" in base_state and isinstance(base_state["state_dict"], dict):
        base_state = base_state["state_dict"]
    missing0, unexpected0 = model.load_state_dict(base_state, strict=False)
    print(f"[base checkpoint] missing={len(missing0)} unexpected={len(unexpected0)} from {args.checkpoint}")
    if args.lora_checkpoint:
        ckpt = _safe_torch_load(args.lora_checkpoint, map_location=args.device)
        state = ckpt.get("sam_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict) and any(".lora_a" in k for k in state.keys()):
            rank = int(ckpt.get("lora_rank", 8)) if isinstance(ckpt, dict) else 8
            alpha = int(ckpt.get("lora_alpha", 16)) if isinstance(ckpt, dict) else 16
            replaced = inject_lora(
                model.image_encoder,
                target_keywords=["qkv", "proj", "q_proj", "k_proj", "v_proj"],
                rank=rank,
                alpha=alpha,
            )
            print(f"[teacher lora] injected layers={replaced}, rank={rank}, alpha={alpha}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[teacher load] missing={len(missing)} unexpected={len(unexpected)} from {args.lora_checkpoint}")
    predictor = SamPredictor(model)

    quality_rows = []
    manifest_rows = []
    candidate_rows = []
    for row in tqdm(rows, desc="Pseudo Labeling"):
        pid = row["id"]
        out_hard = hard_dir / f"{pid}.png"
        out_soft = soft_dir / f"{pid}.npy"
        out_edge = edge_dir / f"{pid}.npy"
        out_panel = panel_dir / f"{pid}.jpg"
        core_ready = out_hard.exists() and out_soft.exists() and out_edge.exists()
        panel_ready = (not args.save_panels) or out_panel.exists()
        if args.skip_exist and core_ready and panel_ready:
            continue

        image_bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        predictor.set_image(image_rgb)
        quality_reflection_map = _reflection_map(
            image_bgr=image_bgr,
            v_thresh=int(args.quality_reflection_v_threshold),
            s_thresh=int(args.quality_reflection_s_threshold),
            enabled=bool(args.quality_use_reflection_penalty) or bool(args.postprocess_mask),
        )
        source_text = str(row.get("source", "")).strip().lower()

        preset_box = proposals.get(pid)
        auto_boxes = _build_auto_boxes(
            w=w,
            h=h,
            scales=candidate_scales,
            centers=candidate_centers,
            mode=args.auto_proposal_mode,
            max_candidates=args.max_candidate_boxes,
            image_bgr=image_bgr,
            min_area_ratio=float(args.min_mask_area_ratio),
            target_area_ratio=float(args.target_mask_area_ratio),
            max_area_ratio=float(args.max_mask_area_ratio),
            reflection_v_threshold=int(args.quality_reflection_v_threshold),
            reflection_s_threshold=int(args.quality_reflection_s_threshold),
        )
        candidate_entries = []
        if preset_box is not None:
            mix_mode = str(args.proposal_mix_mode).strip().lower()
            if mix_mode == "replace":
                candidate_entries = [{"box": _clip_box(preset_box, w=w, h=h), "source": "preset"}]
            elif mix_mode == "augment":
                aug = _build_augmented_proposal_boxes(
                    preset_box=preset_box,
                    w=w,
                    h=h,
                    scales=proposal_jitter_scales,
                    shifts=proposal_jitter_shifts,
                    max_boxes=int(args.proposal_jitter_max_boxes),
                )
                candidate_entries = [{"box": b, "source": "preset_aug"} for b in aug]
            elif mix_mode == "augment_plus_auto":
                aug = _build_augmented_proposal_boxes(
                    preset_box=preset_box,
                    w=w,
                    h=h,
                    scales=proposal_jitter_scales,
                    shifts=proposal_jitter_shifts,
                    max_boxes=int(args.proposal_jitter_max_boxes),
                )
                candidate_entries = [{"box": b, "source": "preset_aug"} for b in aug]
                candidate_entries.extend({"box": b, "source": "auto"} for b in auto_boxes)
            else:
                raise ValueError(f"Unsupported --proposal-mix-mode: {args.proposal_mix_mode}")
        elif args.fallback_full_image:
            candidate_entries = [{"box": b, "source": "auto"} for b in auto_boxes]
        else:
            candidate_entries = []

        append_mode = str(args.append_auto_candidates).strip().lower()
        append_enabled = False
        if append_mode == "all":
            append_enabled = True
        elif append_mode == "polypgen" and "polypgen" in source_text:
            append_enabled = True
        if preset_box is not None and append_enabled:
            auto_extra = auto_boxes
            if int(args.append_auto_max_candidates) > 0:
                auto_extra = auto_extra[: int(args.append_auto_max_candidates)]
            candidate_entries.extend({"box": b, "source": "auto_extra"} for b in auto_extra)

        dedup_entries = {}
        for ent in candidate_entries:
            b = ent["box"]
            key = tuple(round(float(v), 1) for v in b)
            dedup_entries[key] = {"box": [float(v) for v in b], "source": str(ent.get("source", "auto"))}
        candidate_entries = list(dedup_entries.values())
        if args.max_candidate_boxes > 0 and len(candidate_entries) > int(args.max_candidate_boxes):
            candidate_entries = candidate_entries[: int(args.max_candidate_boxes)]
        if not candidate_entries:
            continue

        all_packs = []
        valid_packs = []
        center_w = float(args.score_weight_center_prior)
        poly_center_w = float(args.score_weight_center_prior_polypgen)
        if "polypgen" in source_text and poly_center_w >= 0.0:
            center_w = poly_center_w

        def _compose_quality(
            conf_k: float,
            edge_q_k: float,
            area_prior_k: float,
            center_prior_k: float,
            prompt_source_used: str,
            geom_metrics: dict,
        ) -> float:
            score_k = float(
                float(args.score_weight_conf) * float(conf_k)
                + float(args.score_weight_edge) * float(edge_q_k)
                + float(args.score_weight_area_prior) * float(area_prior_k)
                + float(center_w) * float(center_prior_k)
            )
            src_text = str(prompt_source_used).strip().lower()
            if src_text.startswith("preset"):
                score_k += float(args.score_bias_preset)
            elif src_text.startswith("auto"):
                auto_bias = float(args.score_bias_auto)
                if "polypgen" in source_text and float(args.score_bias_auto_polypgen) > -900.0:
                    auto_bias = float(args.score_bias_auto_polypgen)
                score_k += auto_bias
            score_k -= float(args.quality_penalty_spill_weight) * float(geom_metrics.get("spill_ratio", 0.0))
            score_k -= float(args.quality_penalty_reflection_weight) * float(geom_metrics.get("reflection_overlap", 0.0))
            score_k -= float(args.quality_penalty_fragment_weight) * (
                1.0 - float(geom_metrics.get("largest_cc_ratio", 0.0))
            )
            return float(score_k)

        def _build_pack(mask_k, score_k_raw, logit_k, box_used, prompt_source_used, refine_stage, prompt_points_n):
            hard_k = (mask_k > 0).astype(np.uint8) * 255
            lowres_logit = logit_k.astype(np.float32)
            lowres_prob = 1.0 / (1.0 + np.exp(-lowres_logit))
            soft_k = cv2.resize(lowres_prob, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            edge_k = _edge_from_prob(soft_k)
            conf_k = float(score_k_raw)
            edge_q_k = _edge_quality(hard_k, edge_k)
            area_ratio_k = _mask_area_ratio(hard_k)
            area_prior_k = _area_prior(
                area_ratio_k,
                min_ratio=args.min_mask_area_ratio,
                target_ratio=args.target_mask_area_ratio,
                max_ratio=args.max_mask_area_ratio,
            )
            center_prior_k = _center_prior(hard_k)
            geom_k = _mask_geometry_metrics(
                mask_bin_u8=hard_k,
                box_xyxy=box_used,
                reflection_map=quality_reflection_map,
            )
            score_k = _compose_quality(
                conf_k=conf_k,
                edge_q_k=edge_q_k,
                area_prior_k=area_prior_k,
                center_prior_k=center_prior_k,
                prompt_source_used=prompt_source_used,
                geom_metrics=geom_k,
            )
            return {
                "hard": hard_k,
                "soft": soft_k,
                "edge": edge_k,
                "conf": conf_k,
                "edge_quality": edge_q_k,
                "area_ratio": area_ratio_k,
                "area_prior": area_prior_k,
                "center_prior": center_prior_k,
                "quality": float(score_k),
                "prompt_box": [float(x) for x in box_used],
                "prompt_source": str(prompt_source_used),
                "refine_stage": str(refine_stage),
                "prompt_points": int(prompt_points_n),
                "box_in_ratio": float(geom_k.get("box_in_ratio", 0.0)),
                "spill_ratio": float(geom_k.get("spill_ratio", 1.0)),
                "reflection_overlap": float(geom_k.get("reflection_overlap", 0.0)),
                "largest_cc_ratio": float(geom_k.get("largest_cc_ratio", 0.0)),
                "num_components": int(geom_k.get("num_components", 0)),
            }

        for entry in candidate_entries:
            box = entry["box"]
            prompt_source = str(entry.get("source", "auto"))
            masks, scores, logits = predictor.predict(
                box=np.array(box, dtype=np.float32)[None, :],
                point_coords=None,
                point_labels=None,
                multimask_output=True,
                return_logits=True,
            )
            for k in range(int(len(scores))):
                pack = _build_pack(
                    mask_k=masks[k],
                    score_k_raw=scores[k],
                    logit_k=logits[k],
                    box_used=box,
                    prompt_source_used=prompt_source,
                    refine_stage="pass1",
                    prompt_points_n=0,
                )
                all_packs.append(pack)
                if args.min_mask_area_ratio <= float(pack["area_ratio"]) <= args.max_mask_area_ratio:
                    valid_packs.append(pack)
        candidate_pool = valid_packs if valid_packs else all_packs
        if not candidate_pool:
            continue
        best_pack = max(candidate_pool, key=lambda x: float(x["quality"]))

        if bool(args.two_pass_refine):
            first_quality = float(best_pack.get("quality", 0.0))
            first_area = float(best_pack.get("area_ratio", 0.0))
            if first_quality >= float(args.two_pass_min_first_quality) and first_area >= float(args.two_pass_min_first_area_ratio):
                refined_box = _box_from_mask(
                    mask_bin=best_pack["hard"],
                    expand_ratio=float(args.two_pass_box_expand_ratio),
                    w=w,
                    h=h,
                    min_side=max(8.0, float(min(h, w)) * 0.04),
                )
                second_packs = []
                second_valid = []

                masks2a, scores2a, logits2a = predictor.predict(
                    box=np.array(refined_box, dtype=np.float32)[None, :],
                    point_coords=None,
                    point_labels=None,
                    multimask_output=True,
                    return_logits=True,
                )
                prompt_source2a = f"{str(best_pack.get('prompt_source', 'auto'))}_p2box"
                for k in range(int(len(scores2a))):
                    pack2a = _build_pack(
                        mask_k=masks2a[k],
                        score_k_raw=scores2a[k],
                        logit_k=logits2a[k],
                        box_used=refined_box,
                        prompt_source_used=prompt_source2a,
                        refine_stage="pass2_box",
                        prompt_points_n=0,
                    )
                    pack2a["quality"] = float(pack2a["quality"]) + float(args.two_pass_score_bonus)
                    all_packs.append(pack2a)
                    second_packs.append(pack2a)
                    if args.min_mask_area_ratio <= float(pack2a["area_ratio"]) <= args.max_mask_area_ratio:
                        second_valid.append(pack2a)

                point_coords, point_labels = _build_two_pass_points(
                    soft_prob=best_pack["soft"],
                    hard_mask=best_pack["hard"],
                    image_bgr=image_bgr,
                    box_xyxy=refined_box,
                    pos_points=int(args.two_pass_pos_points),
                    neg_points=int(args.two_pass_neg_points),
                    use_reflection_neg=bool(args.two_pass_use_reflection_neg),
                    reflection_v_threshold=int(args.two_pass_reflection_v_threshold),
                    reflection_s_threshold=int(args.two_pass_reflection_s_threshold),
                )
                if point_coords is not None and point_labels is not None and len(point_labels) > 0:
                    masks2, scores2, logits2 = predictor.predict(
                        box=np.array(refined_box, dtype=np.float32)[None, :],
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                        return_logits=True,
                    )
                    prompt_source2 = f"{str(best_pack.get('prompt_source', 'auto'))}_p2pts"
                    for k in range(int(len(scores2))):
                        pack2 = _build_pack(
                            mask_k=masks2[k],
                            score_k_raw=scores2[k],
                            logit_k=logits2[k],
                            box_used=refined_box,
                            prompt_source_used=prompt_source2,
                            refine_stage="pass2",
                            prompt_points_n=int(len(point_labels)),
                        )
                        pack2["quality"] = float(pack2["quality"]) + float(args.two_pass_score_bonus)
                        all_packs.append(pack2)
                        second_packs.append(pack2)
                        if args.min_mask_area_ratio <= float(pack2["area_ratio"]) <= args.max_mask_area_ratio:
                            second_valid.append(pack2)
                second_pool = second_valid if second_valid else second_packs
                if second_pool:
                    best_pack2 = max(second_pool, key=lambda x: float(x["quality"]))
                    if float(best_pack2["quality"]) >= float(best_pack["quality"]) + float(args.two_pass_min_gain):
                        best_pack = best_pack2
                    else:
                        # structural tie-break: prefer pass2 if area is closer to target and edge quality is not worse.
                        target_area = float(args.target_mask_area_ratio)
                        old_dist = abs(float(best_pack["area_ratio"]) - target_area)
                        new_dist = abs(float(best_pack2["area_ratio"]) - target_area)
                        edge_old = float(best_pack["edge_quality"])
                        edge_new = float(best_pack2["edge_quality"])
                        if (new_dist + 1e-6) < (old_dist * 0.85 + 1e-6) and edge_new >= (edge_old - 0.02):
                            best_pack = best_pack2

        consistency_pool = [
            p
            for p in all_packs
            if args.min_mask_area_ratio <= float(p.get("area_ratio", 0.0)) <= args.max_mask_area_ratio
        ]
        if not consistency_pool:
            consistency_pool = list(all_packs)
        consistency_iou, quality_gap = _candidate_consistency(
            best_pack=best_pack,
            candidate_pool=consistency_pool,
            top_k=int(args.consistency_topk),
        )
        consistency_penalty = 0.0
        if float(args.consistency_min_iou) > 0.0:
            consistency_penalty = max(0.0, float(args.consistency_min_iou) - float(consistency_iou))
        best_pack["consistency_iou"] = float(consistency_iou)
        best_pack["quality_gap"] = float(quality_gap)
        best_pack["quality"] = float(best_pack.get("quality", 0.0)) - float(args.quality_penalty_consistency_weight) * float(
            consistency_penalty
        )

        if bool(args.postprocess_mask):
            post_hard = _postprocess_mask(
                mask_bin_u8=best_pack["hard"],
                box_xyxy=best_pack["prompt_box"],
                reflection_map=quality_reflection_map,
                min_component_area_ratio=float(args.post_min_component_area_ratio),
                min_inbox_ratio=float(args.post_min_inbox_ratio),
                max_reflection_overlap=float(args.post_max_reflection_overlap),
                keep_max_components=int(args.post_keep_max_components),
            )
            if post_hard.shape == best_pack["hard"].shape and not np.array_equal(post_hard, best_pack["hard"]):
                best_pack["hard"] = post_hard
                best_pack["edge_quality"] = _edge_quality(best_pack["hard"], best_pack["edge"])
                best_pack["area_ratio"] = _mask_area_ratio(best_pack["hard"])
                best_pack["area_prior"] = _area_prior(
                    best_pack["area_ratio"],
                    min_ratio=args.min_mask_area_ratio,
                    target_ratio=args.target_mask_area_ratio,
                    max_ratio=args.max_mask_area_ratio,
                )
                best_pack["center_prior"] = _center_prior(best_pack["hard"])
                geom_post = _mask_geometry_metrics(
                    mask_bin_u8=best_pack["hard"],
                    box_xyxy=best_pack["prompt_box"],
                    reflection_map=quality_reflection_map,
                )
                best_pack["box_in_ratio"] = float(geom_post.get("box_in_ratio", 0.0))
                best_pack["spill_ratio"] = float(geom_post.get("spill_ratio", 1.0))
                best_pack["reflection_overlap"] = float(geom_post.get("reflection_overlap", 0.0))
                best_pack["largest_cc_ratio"] = float(geom_post.get("largest_cc_ratio", 0.0))
                best_pack["num_components"] = int(geom_post.get("num_components", 0))
                best_pack["quality"] = _compose_quality(
                    conf_k=float(best_pack["conf"]),
                    edge_q_k=float(best_pack["edge_quality"]),
                    area_prior_k=float(best_pack["area_prior"]),
                    center_prior_k=float(best_pack["center_prior"]),
                    prompt_source_used=str(best_pack.get("prompt_source", "auto")),
                    geom_metrics=geom_post,
                ) - float(args.quality_penalty_consistency_weight) * float(consistency_penalty)

        hard = best_pack["hard"]
        soft = best_pack["soft"]
        edge = best_pack["edge"]
        conf = float(best_pack["conf"])
        edge_q = float(best_pack["edge_quality"])
        area_ratio = float(best_pack["area_ratio"])
        area_prior = float(best_pack["area_prior"])
        center_prior = float(best_pack["center_prior"])
        quality = float(best_pack["quality"])
        prompt_box = best_pack["prompt_box"]
        prompt_source = str(best_pack.get("prompt_source", "auto"))
        refine_stage = str(best_pack.get("refine_stage", "pass1"))
        prompt_points_n = int(best_pack.get("prompt_points", 0))
        box_in_ratio = float(best_pack.get("box_in_ratio", 0.0))
        spill_ratio = float(best_pack.get("spill_ratio", 1.0))
        reflection_overlap = float(best_pack.get("reflection_overlap", 0.0))
        largest_cc_ratio = float(best_pack.get("largest_cc_ratio", 0.0))
        num_components = int(best_pack.get("num_components", 0))

        if args.write_candidate_scores:
            ranked = sorted(all_packs, key=lambda x: float(x["quality"]), reverse=True)
            for rank, pack in enumerate(ranked[: min(10, len(ranked))], start=1):
                candidate_rows.append(
                    {
                        "id": pid,
                        "rank": int(rank),
                        "quality": float(pack["quality"]),
                        "conf": float(pack["conf"]),
                        "edge_quality": float(pack["edge_quality"]),
                        "area_ratio": float(pack["area_ratio"]),
                        "area_prior": float(pack["area_prior"]),
                        "center_prior": float(pack["center_prior"]),
                        "prompt_source": str(pack.get("prompt_source", "auto")),
                        "prompt_box": json.dumps(pack["prompt_box"], ensure_ascii=False),
                        "refine_stage": str(pack.get("refine_stage", "pass1")),
                        "prompt_points": int(pack.get("prompt_points", 0)),
                        "box_in_ratio": float(pack.get("box_in_ratio", 0.0)),
                        "spill_ratio": float(pack.get("spill_ratio", 1.0)),
                        "reflection_overlap": float(pack.get("reflection_overlap", 0.0)),
                        "largest_cc_ratio": float(pack.get("largest_cc_ratio", 0.0)),
                        "num_components": int(pack.get("num_components", 0)),
                        "selected": int(rank == 1),
                        "source": row.get("source", ""),
                        "subset": row.get("subset", "U_large"),
                    }
                )

        cv2.imwrite(str(out_hard), hard)
        np.save(out_soft, soft)
        np.save(out_edge, edge)
        panel_path = ""
        if args.save_panels:
            panel = _build_panel(image_bgr, hard, alpha=args.overlay_alpha)
            cv2.imwrite(str(out_panel), panel)
            panel_path = str(out_panel)

        quality_rows.append(
            {
                "id": pid,
                "image_path": row["image_path"],
                "hard_mask_path": str(out_hard),
                "panel_path": panel_path,
                "soft_path": str(out_soft),
                "edge_path": str(out_edge),
                "prompt_box": json.dumps(prompt_box, ensure_ascii=False),
                "prompt_source": prompt_source,
                "refine_stage": refine_stage,
                "prompt_points": int(prompt_points_n),
                "subset": row.get("subset", "U_large"),
                "source": row.get("source", ""),
                "center": row.get("center", ""),
                "conf": conf,
                "edge_quality": edge_q,
                "area_ratio": area_ratio,
                "area_prior": area_prior,
                "center_prior": center_prior,
                "box_in_ratio": box_in_ratio,
                "spill_ratio": spill_ratio,
                "reflection_overlap": reflection_overlap,
                "largest_cc_ratio": largest_cc_ratio,
                "num_components": int(num_components),
                "consistency_iou": float(best_pack.get("consistency_iou", 1.0)),
                "quality_gap": float(best_pack.get("quality_gap", 1.0)),
                "quality": quality,
                "round_id": int(args.round_id),
            }
        )
        manifest_rows.append(
            {
                "id": pid,
                "image_path": row["image_path"],
                "mask_path": str(out_hard),
                "subset": f"pseudo_round{int(args.round_id)}",
                "split": "train",
                "source": row.get("source", ""),
                "center": row.get("center", ""),
                "is_labeled": 1,
                "is_pseudo": 1,
                "pseudo_weight": quality,
                "round_id": int(args.round_id),
                "exclude_from_tuning": 0,
                "soft_path": str(out_soft),
                "edge_path": str(out_edge),
            }
        )

    quality_csv = out_root / "pseudo_quality.csv"
    with open(quality_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "image_path",
            "hard_mask_path",
            "panel_path",
            "soft_path",
            "edge_path",
            "prompt_box",
            "prompt_source",
            "refine_stage",
            "prompt_points",
            "subset",
            "source",
            "center",
            "conf",
            "edge_quality",
            "area_ratio",
            "area_prior",
            "center_prior",
            "box_in_ratio",
            "spill_ratio",
            "reflection_overlap",
            "largest_cc_ratio",
            "num_components",
            "consistency_iou",
            "quality_gap",
            "quality",
            "round_id",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(quality_rows)

    manifest_csv = out_root / "pseudo_candidates_manifest.csv"
    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
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
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)

    candidate_csv = out_root / "candidate_scores.csv"
    if args.write_candidate_scores and candidate_rows:
        with open(candidate_csv, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "id",
                "rank",
                "selected",
                "quality",
                "conf",
                "edge_quality",
                "area_ratio",
                "area_prior",
                "center_prior",
                "box_in_ratio",
                "spill_ratio",
                "reflection_overlap",
                "largest_cc_ratio",
                "num_components",
                "prompt_source",
                "prompt_box",
                "refine_stage",
                "prompt_points",
                "source",
                "subset",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(candidate_rows)

    gallery_html = out_root / "mask_gallery.html"
    if args.write_gallery:
        _write_gallery(quality_rows, gallery_html)

    print(
        json.dumps(
            {
                "num_processed": len(quality_rows),
                "quality_csv": str(quality_csv),
                "pseudo_candidates_manifest": str(manifest_csv),
                "hard_masks_dir": str(hard_dir),
                "mask_panels_dir": str(panel_dir) if args.save_panels else "",
                "mask_gallery_html": str(gallery_html) if args.write_gallery else "",
                "candidate_scores_csv": str(candidate_csv) if args.write_candidate_scores and candidate_rows else "",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
