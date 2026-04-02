import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models import ResSwinUNet


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _safe_torch_load(path: str, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _parse_model_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs.get("seg")
    if isinstance(outputs, (tuple, list)):
        return outputs[0] if len(outputs) > 0 else None
    return outputs


def _remap_legacy_head_keys(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    remapped = dict(state_dict)
    if "head.weight" in remapped and "seg_head.weight" not in remapped:
        remapped["seg_head.weight"] = remapped.pop("head.weight")
    if "head.bias" in remapped and "seg_head.bias" not in remapped:
        remapped["seg_head.bias"] = remapped.pop("head.bias")
    return remapped


def _resolve(path_text: str, base: Path | None = None) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    if base is not None:
        p2 = (base / p).resolve()
        if p2.exists():
            return p2
        p3 = (base.parent / p).resolve()
        if p3.exists():
            return p3
    return (Path.cwd() / p).resolve()


def _load_manifest_rows(manifest_path: str, subset_filter: set[str]) -> list[dict]:
    out = []
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
            t = line.strip()
            if t:
                ids.add(t.split(",")[0])
    return ids


def _prepare_tensor(image_bgr: np.ndarray, img_size: int) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image_f = image_rgb.astype(np.float32) / 255.0
    image_f = (image_f - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image_f).permute(2, 0, 1).unsqueeze(0)


def _center_box(w: int, h: int, scale: float) -> list[float]:
    s = float(max(0.1, min(1.0, scale)))
    bw = max(8.0, (w - 1) * s)
    bh = max(8.0, (h - 1) * s)
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    x0 = max(0.0, cx - 0.5 * bw)
    y0 = max(0.0, cy - 0.5 * bh)
    x1 = min(float(w - 1), cx + 0.5 * bw)
    y1 = min(float(h - 1), cy + 0.5 * bh)
    return [x0, y0, x1, y1]


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


def _reflection_map(image_bgr: np.ndarray, v_thresh: int, s_thresh: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    m = ((val >= int(v_thresh)) & (sat <= int(s_thresh))).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return m


def _candidate_components(
    prob: np.ndarray,
    threshold: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[np.ndarray, list[dict]]:
    h, w = prob.shape
    total = float(max(1, h * w))

    mask = (prob >= float(threshold)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps: list[dict] = []
    for i in range(1, int(num_labels)):
        x, y, bw, bh, area = stats[i].tolist()
        area_ratio = float(area) / total
        if area_ratio < max(1e-6, min_area_ratio * 0.20):
            continue
        if area_ratio > min(1.0, max_area_ratio * 2.0):
            continue
        x0 = int(x)
        y0 = int(y)
        x1 = int(x + bw - 1)
        y1 = int(y + bh - 1)
        bbox_area = float(max(1, bw * bh))
        compactness = float(area) / bbox_area
        m = labels == i
        mean_prob = float(prob[m].mean()) if np.any(m) else 0.0
        comps.append(
            {
                "id": int(i),
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "w": int(bw),
                "h": int(bh),
                "area_ratio": area_ratio,
                "compactness": compactness,
                "mean_prob": mean_prob,
                "mask": m,
            }
        )
    return mask, comps


def _estimate_box_from_prob(
    prob: np.ndarray,
    image_bgr: np.ndarray,
    pred_threshold: float,
    min_area_ratio: float,
    target_area_ratio: float,
    max_area_ratio: float,
    bbox_margin: float,
    fallback_scale: float,
    min_box_side_ratio: float,
    threshold_delta: float,
    suppress_reflection: bool,
    reflection_v_threshold: int,
    reflection_s_threshold: int,
    max_reflection_overlap: float,
) -> tuple[list[float], np.ndarray, dict]:
    h, w = prob.shape
    total = float(max(1, h * w))

    dyn_q = float(np.clip(1.0 - target_area_ratio, 0.5, 0.995))
    dyn_thr = float(np.quantile(prob, dyn_q))
    base_thr = float(np.clip(pred_threshold, 0.05, 0.95))
    thrs = [
        base_thr,
        float(np.clip(base_thr - threshold_delta, 0.05, 0.95)),
        float(np.clip(base_thr + threshold_delta, 0.05, 0.95)),
        float(np.clip(dyn_thr, 0.05, 0.95)),
    ]
    thrs = sorted({round(t, 4) for t in thrs})

    refl = _reflection_map(
        image_bgr=image_bgr,
        v_thresh=reflection_v_threshold,
        s_thresh=reflection_s_threshold,
        enabled=suppress_reflection,
    )

    candidates = []
    for thr in thrs:
        _, comps = _candidate_components(
            prob=prob,
            threshold=thr,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        for c in comps:
            cx = 0.5 * (c["x0"] + c["x1"])
            cy = 0.5 * (c["y0"] + c["y1"])
            center_dist = np.sqrt((cx - 0.5 * (w - 1)) ** 2 + (cy - 0.5 * (h - 1)) ** 2)
            center_score = float(np.clip(1.0 - center_dist / (np.sqrt((0.5 * (w - 1)) ** 2 + (0.5 * (h - 1)) ** 2) + 1e-6), 0.0, 1.0))

            area_p = _area_prior(c["area_ratio"], min_area_ratio, target_area_ratio, max_area_ratio)
            refl_overlap = float(refl[c["mask"]].mean()) if np.any(c["mask"]) else 0.0

            score = (
                0.46 * float(c["mean_prob"])
                + 0.22 * float(area_p)
                + 0.16 * float(np.clip(c["compactness"], 0.0, 1.0))
                + 0.10 * center_score
                - 0.28 * refl_overlap
            )

            c_out = dict(c)
            c_out["threshold"] = float(thr)
            c_out["area_prior"] = float(area_p)
            c_out["center_score"] = float(center_score)
            c_out["reflection_overlap"] = float(refl_overlap)
            c_out["score"] = float(score)
            candidates.append(c_out)

    if not candidates:
        box = _center_box(w, h, scale=fallback_scale)
        out_mask = (prob >= max(0.35, base_thr)).astype(np.uint8)
        info = {
            "mean_prob": float(prob.mean()),
            "area_ratio": float(out_mask.mean()),
            "bbox_area_ratio": float((box[2] - box[0]) * (box[3] - box[1]) / total),
            "threshold": float(base_thr),
            "is_fallback": 1,
            "reflection_overlap": 0.0,
            "component_score": 0.0,
        }
        return box, out_mask, info

    max_refl = float(np.clip(max_reflection_overlap, 0.0, 1.0))
    low_refl_candidates = [c for c in candidates if float(c.get("reflection_overlap", 0.0)) <= max_refl]
    candidate_pool = low_refl_candidates if low_refl_candidates else candidates
    best = max(candidate_pool, key=lambda x: float(x["score"]))

    x0 = float(best["x0"])
    y0 = float(best["y0"])
    x1 = float(best["x1"])
    y1 = float(best["y1"])
    bw = float(max(1.0, x1 - x0 + 1.0))
    bh = float(max(1.0, y1 - y0 + 1.0))

    conf = float(best["mean_prob"])
    extra = max(0.0, 0.18 * (0.60 - conf))
    margin = float(max(0.0, bbox_margin + extra))

    dx = bw * margin
    dy = bh * margin
    x0 = max(0.0, x0 - dx)
    y0 = max(0.0, y0 - dy)
    x1 = min(float(w - 1), x1 + dx)
    y1 = min(float(h - 1), y1 + dy)

    min_side = float(max(4.0, min(h, w) * float(max(0.02, min_box_side_ratio))))
    cur_w = x1 - x0 + 1.0
    cur_h = y1 - y0 + 1.0
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    if cur_w < min_side:
        half = 0.5 * min_side
        x0 = max(0.0, cx - half)
        x1 = min(float(w - 1), cx + half)
    if cur_h < min_side:
        half = 0.5 * min_side
        y0 = max(0.0, cy - half)
        y1 = min(float(h - 1), cy + half)

    box = [float(x0), float(y0), float(x1), float(y1)]
    mask_best = best["mask"].astype(np.uint8)
    bbox_area_ratio = float(max(1e-6, (x1 - x0 + 1.0) * (y1 - y0 + 1.0)) / total)
    info = {
        "mean_prob": float(best["mean_prob"]),
        "area_ratio": float(best["area_ratio"]),
        "bbox_area_ratio": bbox_area_ratio,
        "threshold": float(best["threshold"]),
        "is_fallback": 0,
        "reflection_overlap": float(best["reflection_overlap"]),
        "component_score": float(best["score"]),
    }
    return box, mask_best, info


def _draw_box_on_image(image_bgr: np.ndarray, box: list[float], mask: np.ndarray | None = None) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))

    if mask is not None:
        m = (mask > 0).astype(np.uint8)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        contour_mask = (m > 0)
        out[contour_mask] = (0.65 * out[contour_mask] + 0.35 * np.array([0, 0, 255], dtype=np.float32)).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 1)

    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
    return out


def _make_preview(
    image_bgr: np.ndarray,
    prob: np.ndarray,
    mask: np.ndarray,
    box: list[float],
    mode: str,
) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "image_box":
        return _draw_box_on_image(image_bgr=image_bgr, box=box, mask=None)
    if mode == "image_mask_box":
        return _draw_box_on_image(image_bgr=image_bgr, box=box, mask=mask)

    h, w = image_bgr.shape[:2]
    heat = cv2.applyColorMap(np.clip(prob * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    left = image_bgr.copy()
    right = _draw_box_on_image(image_bgr=image_bgr, box=box, mask=mask)

    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    cv2.rectangle(heat, (x0, y0), (x1, y1), (0, 255, 255), 2)

    panel = np.concatenate([left, heat, right], axis=1)
    cv2.putText(panel, "image", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "prob", (w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "image+box", (2 * w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _build_model_from_checkpoint(ckpt_path: str, device: str, img_size_override: int):
    ckpt = _safe_torch_load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")

    state_dict = ckpt.get("model", ckpt)
    args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    train_mode = str(args.get("mode", "")).strip()
    img_size = int(img_size_override) if int(img_size_override) > 0 else int(args.get("img_size", 256))
    model_kwargs = {
        "num_classes": 1,
        "use_boundary": False,
        "norm_type": args.get("norm_type", "gn"),
        "deep_supervision": bool(args.get("deep_supervision", True)),
        "window_size": int(args.get("window_size", 8)),
        "use_shift_mask": bool(args.get("use_shift_mask", True)),
        "use_rel_pos_bias": bool(args.get("use_rel_pos_bias", True)),
        "pad_to_window": bool(args.get("pad_to_window", True)),
        "use_wavelet_bottleneck": bool(args.get("use_wavelet_bottleneck", True)),
    }

    model = ResSwinUNet(**model_kwargs).to(device)
    state_dict = _remap_legacy_head_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    seg_missing = [k for k in missing if k.startswith("seg_head.")]
    if seg_missing:
        raise RuntimeError(f"Missing seg head weights in checkpoint: {seg_missing}")
    print(
        f"[box-model] checkpoint={ckpt_path} mode={train_mode or 'unknown'} img_size={img_size} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    model.eval()
    return model, img_size, train_mode


def main():
    parser = argparse.ArgumentParser(description="Generate automatic box prompts from L_small-supervised localizer model.")
    parser.add_argument("--data-manifest", type=str, required=True)
    parser.add_argument("--subset-filter", type=str, default="U_large")
    parser.add_argument("--id-filter", type=str, default="")
    parser.add_argument("--localizer-checkpoint", type=str, default="")
    parser.add_argument("--student-checkpoint", type=str, default="")
    parser.add_argument("--required-train-mode", type=str, default="supervised_only")
    parser.add_argument("--output-json", type=str, default="runs/flywheel/round1/auto_proposals.json")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--preview-dir", type=str, default="")
    parser.add_argument("--max-preview", type=int, default=200)
    parser.add_argument("--preview-mode", type=str, default="image_box", choices=["image_box", "image_mask_box", "panel_heatmap"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img-size", type=int, default=0)
    parser.add_argument("--pred-threshold", type=float, default=0.45)
    parser.add_argument("--threshold-delta", type=float, default=0.08)
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    parser.add_argument("--target-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--bbox-margin", type=float, default=0.15)
    parser.add_argument("--fallback-scale", type=float, default=0.75)
    parser.add_argument("--min-box-side-ratio", type=float, default=0.06)
    parser.add_argument("--suppress-reflection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reflection-v-threshold", type=int, default=245)
    parser.add_argument("--reflection-s-threshold", type=int, default=35)
    parser.add_argument("--max-reflection-overlap", type=float, default=0.18)
    args = parser.parse_args()

    subset_filter = {s.strip() for s in str(args.subset_filter).split(",") if s.strip()}
    rows = _load_manifest_rows(args.data_manifest, subset_filter=subset_filter)
    id_filter = _load_id_filter(args.id_filter)
    if id_filter:
        rows = [r for r in rows if r["id"] in id_filter]
    if not rows:
        raise RuntimeError("No samples found for auto box prompt generation.")

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.output_csv) if args.output_csv else out_json.with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    preview_dir = Path(args.preview_dir) if args.preview_dir else out_json.parent / "proposal_previews"
    if args.max_preview > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(args.localizer_checkpoint or args.student_checkpoint).strip()
    if not ckpt_path:
        raise ValueError("Please provide --localizer-checkpoint (preferred) or --student-checkpoint")

    model, img_size, train_mode = _build_model_from_checkpoint(
        ckpt_path=ckpt_path,
        device=args.device,
        img_size_override=args.img_size,
    )
    req_mode = str(args.required_train_mode).strip().lower()
    tm = str(train_mode).strip().lower()
    if req_mode and req_mode != "off" and tm != req_mode:
        raise RuntimeError(
            "Checkpoint train mode mismatch. Use L_small supervised checkpoint or set --required-train-mode off."
        )

    proposal_map: dict[str, list[float]] = {}
    rows_out: list[dict] = []
    fallback_count = 0
    preview_count = 0

    with torch.no_grad():
        for row in tqdm(rows, desc="Auto Box Prompt"):
            pid = row["id"]
            image_bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue

            h, w = image_bgr.shape[:2]
            x = _prepare_tensor(image_bgr, img_size=img_size).to(args.device)
            seg_logits = _parse_model_outputs(model(x))
            if seg_logits is None:
                continue
            prob_small = torch.sigmoid(seg_logits)[0, 0].detach().float().cpu().numpy()
            prob = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            prob = np.clip(prob, 0.0, 1.0)

            box, mask, info = _estimate_box_from_prob(
                prob=prob,
                image_bgr=image_bgr,
                pred_threshold=args.pred_threshold,
                min_area_ratio=args.min_area_ratio,
                target_area_ratio=args.target_area_ratio,
                max_area_ratio=args.max_area_ratio,
                bbox_margin=args.bbox_margin,
                fallback_scale=args.fallback_scale,
                min_box_side_ratio=args.min_box_side_ratio,
                threshold_delta=args.threshold_delta,
                suppress_reflection=bool(args.suppress_reflection),
                reflection_v_threshold=args.reflection_v_threshold,
                reflection_s_threshold=args.reflection_s_threshold,
                max_reflection_overlap=args.max_reflection_overlap,
            )
            proposal_map[pid] = [float(v) for v in box]
            fallback_count += int(info["is_fallback"])

            if args.max_preview > 0 and preview_count < int(args.max_preview):
                preview = _make_preview(
                    image_bgr=image_bgr,
                    prob=prob,
                    mask=mask,
                    box=box,
                    mode=args.preview_mode,
                )
                cv2.imwrite(str(preview_dir / f"{pid}.jpg"), preview)
                preview_count += 1

            rows_out.append(
                {
                    "id": pid,
                    "image_path": row["image_path"],
                    "x0": float(box[0]),
                    "y0": float(box[1]),
                    "x1": float(box[2]),
                    "y1": float(box[3]),
                    "mean_prob": float(info["mean_prob"]),
                    "area_ratio": float(info["area_ratio"]),
                    "bbox_area_ratio": float(info["bbox_area_ratio"]),
                    "threshold": float(info["threshold"]),
                    "reflection_overlap": float(info["reflection_overlap"]),
                    "component_score": float(info["component_score"]),
                    "is_fallback": int(info["is_fallback"]),
                    "subset": row.get("subset", ""),
                    "source": row.get("source", ""),
                    "center": row.get("center", ""),
                }
            )

    out_json.write_text(json.dumps(proposal_map, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "image_path",
            "x0",
            "y0",
            "x1",
            "y1",
            "mean_prob",
            "area_ratio",
            "bbox_area_ratio",
            "threshold",
            "reflection_overlap",
            "component_score",
            "is_fallback",
            "subset",
            "source",
            "center",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    area_vals = [float(r["area_ratio"]) for r in rows_out]
    bbox_vals = [float(r["bbox_area_ratio"]) for r in rows_out]
    refl_vals = [float(r["reflection_overlap"]) for r in rows_out]
    summary = {
        "num_samples": len(rows_out),
        "num_fallback": int(fallback_count),
        "fallback_ratio": float(fallback_count) / float(max(1, len(rows_out))),
        "avg_area_ratio": float(np.mean(area_vals)) if area_vals else 0.0,
        "avg_bbox_area_ratio": float(np.mean(bbox_vals)) if bbox_vals else 0.0,
        "avg_reflection_overlap": float(np.mean(refl_vals)) if refl_vals else 0.0,
        "output_json": str(out_json),
        "output_csv": str(out_csv),
        "preview_dir": str(preview_dir) if args.max_preview > 0 else "",
        "preview_mode": str(args.preview_mode),
    }
    out_summary = out_json.with_name(out_json.stem + "_summary.json")
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


