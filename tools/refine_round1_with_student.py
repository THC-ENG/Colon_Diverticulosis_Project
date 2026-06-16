import argparse
import csv
import json
import math
import sys
from pathlib import Path

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


def _to_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _resolve(path_text: str, manifest_path: Path) -> Path:
    p = Path(str(path_text or "").strip())
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    cands = [(manifest_path.parent / p), (manifest_path.parent.parent / p), (Path.cwd() / p)]
    for c in cands:
        if c.exists():
            return c.resolve()
    return cands[0].resolve()


def _safe_torch_load(path: str, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _remap_legacy_head_keys(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    remapped = dict(state_dict)
    if "head.weight" in remapped and "seg_head.weight" not in remapped:
        remapped["seg_head.weight"] = remapped.pop("head.weight")
    if "head.bias" in remapped and "seg_head.bias" not in remapped:
        remapped["seg_head.bias"] = remapped.pop("head.bias")
    return remapped


def _build_model_from_checkpoint(ckpt_path: str, device: str, img_size_override: int, required_train_mode: str):
    ckpt = _safe_torch_load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")
    state_dict = ckpt.get("model", ckpt)
    args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    train_mode = str(args.get("mode", "")).strip().lower()
    req_mode = str(required_train_mode).strip().lower()
    if req_mode and req_mode != "off" and train_mode != req_mode:
        raise RuntimeError(f"Checkpoint train mode mismatch: expected={req_mode}, actual={train_mode or 'unknown'}.")
    img_size = int(img_size_override) if int(img_size_override) > 0 else int(args.get("img_size", 256))
    model = ResSwinUNet(
        num_classes=1,
        use_boundary=False,
        norm_type=args.get("norm_type", "gn"),
        deep_supervision=bool(args.get("deep_supervision", True)),
        window_size=int(args.get("window_size", 8)),
        use_shift_mask=bool(args.get("use_shift_mask", True)),
        use_rel_pos_bias=bool(args.get("use_rel_pos_bias", True)),
        pad_to_window=bool(args.get("pad_to_window", True)),
        use_wavelet_bottleneck=bool(args.get("use_wavelet_bottleneck", True)),
    ).to(device)
    state_dict = _remap_legacy_head_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    seg_missing = [k for k in missing if k.startswith("seg_head.")]
    if seg_missing:
        raise RuntimeError(f"Missing seg head weights in checkpoint: {seg_missing}")
    print(
        f"[refine-model] checkpoint={ckpt_path} mode={train_mode or 'unknown'} img_size={img_size} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    model.eval()
    return model, img_size


def _parse_model_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs.get("seg")
    if isinstance(outputs, (tuple, list)):
        return outputs[0] if len(outputs) > 0 else None
    return outputs


def _prepare_tensor(image_bgr: np.ndarray, img_size: int) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image_f = image_rgb.astype(np.float32) / 255.0
    image_f = (image_f - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image_f).permute(2, 0, 1).unsqueeze(0)


def _load_float_map(path: Path, h: int, w: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path)).astype(np.float32)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"Cannot read map file: {path}")
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        arr = raw.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.shape != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _load_mask_bin(mask_path: Path, h: int, w: int) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read mask file: {mask_path}")
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def _sobel_edge_map(x: np.ndarray) -> np.ndarray:
    arr = x.astype(np.float32)
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    e = np.sqrt(gx * gx + gy * gy)
    p99 = float(np.percentile(e, 99.0))
    e = np.clip(e, 0.0, p99 + 1e-6)
    den = float(e.max()) + 1e-6
    return np.clip(e / den, 0.0, 1.0).astype(np.float32)


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


def _component_filter(mask_bin: np.ndarray, min_area_ratio: float, keep_max_components: int) -> tuple[np.ndarray, float, int]:
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
        comps.append({"label": i, "area": area, "area_ratio": float(area) / total})
    if not comps:
        return np.zeros_like(fg), 0.0, 0
    valid = [c for c in comps if c["area_ratio"] >= float(min_area_ratio)]
    if not valid:
        valid = sorted(comps, key=lambda x: x["area"], reverse=True)[:1]
    else:
        valid = sorted(valid, key=lambda x: x["area"], reverse=True)
        if int(keep_max_components) > 0:
            valid = valid[: int(keep_max_components)]
    out = np.zeros_like(fg, dtype=np.uint8)
    kept_areas = []
    for c in valid:
        m = labels == int(c["label"])
        out[m] = 1
        kept_areas.append(int(m.sum()))
    area_total = float(max(1, int(out.sum())))
    largest_cc_ratio = float(max(kept_areas) / area_total) if kept_areas else 0.0
    return out, float(np.clip(largest_cc_ratio, 0.0, 1.0)), int(len(kept_areas))


def _boundary_quality(mask_bin: np.ndarray, edge: np.ndarray) -> tuple[float, float, float, float]:
    fg = (mask_bin > 0).astype(np.uint8)
    if int(fg.sum()) <= 0:
        return 0.0, 0.0, 0.0, 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dil = cv2.dilate(fg, kernel, iterations=1)
    ero = cv2.erode(fg, kernel, iterations=1)
    band = (dil - ero) > 0
    boundary_grad = float(edge[band].mean()) if int(band.sum()) > 0 else 0.0
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = float(sum(cv2.arcLength(c, True) for c in contours)) + 1e-6
    area = float(fg.sum())
    compactness = float(np.clip((4.0 * math.pi * area) / (perim * perim), 0.0, 1.0))
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    continuity = 0.0
    if n > 1:
        largest = float(np.max(stats[1:, cv2.CC_STAT_AREA]))
        continuity = float(np.clip(largest / float(max(1.0, area)), 0.0, 1.0))
    bq = float(np.clip(0.45 * boundary_grad + 0.35 * compactness + 0.20 * continuity, 0.0, 1.0))
    return float(boundary_grad), float(compactness), float(continuity), float(bq)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    if union <= 0.0:
        return 1.0
    return float(inter / max(1e-6, union))


def _tier_and_scale(boundary_quality: float) -> tuple[str, float]:
    bq = float(np.clip(boundary_quality, 0.0, 1.0))
    if bq >= 0.72:
        return "high", 1.0
    if bq >= 0.58:
        return "mid", 0.6
    return "low", 0.25


def main():
    parser = argparse.ArgumentParser(description="Conservative local high-confidence refinement for round1 pseudo labels.")
    parser.add_argument("--input-manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--required-train-mode", type=str, default="off")
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--output-report-csv", type=str, default="")
    parser.add_argument("--output-summary-json", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img-size", type=int, default=0)
    parser.add_argument("--pred-threshold", type=float, default=0.5)
    parser.add_argument("--high-conf-threshold", type=float, default=0.78)
    parser.add_argument("--max-disagree", type=float, default=0.25)
    parser.add_argument("--soft-sigma", type=float, default=0.8)
    parser.add_argument("--min-boundary-quality-gain", type=float, default=0.015)
    parser.add_argument("--min-iou-keep", type=float, default=0.85)
    parser.add_argument("--min-component-area-ratio", type=float, default=0.0006)
    parser.add_argument("--keep-max-components", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    manifest_path = Path(args.input_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {manifest_path}")
    rows, fieldnames = _read_csv(manifest_path)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_path}")
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]

    out_manifest = Path(args.output_manifest)
    report_csv = (
        Path(args.output_report_csv)
        if str(args.output_report_csv).strip()
        else out_manifest.with_name(out_manifest.stem + "_report.csv")
    )
    summary_json = (
        Path(args.output_summary_json)
        if str(args.output_summary_json).strip()
        else out_manifest.with_name(out_manifest.stem + "_summary.json")
    )
    asset_root = out_manifest.parent / "refined_assets"
    hard_dir = asset_root / "hard_masks"
    soft_dir = asset_root / "soft_probs"
    edge_dir = asset_root / "edge_probs"
    hard_dir.mkdir(parents=True, exist_ok=True)
    soft_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)

    model, img_size = _build_model_from_checkpoint(
        ckpt_path=str(args.checkpoint),
        device=str(args.device),
        img_size_override=int(args.img_size),
        required_train_mode=str(args.required_train_mode),
    )

    out_rows = []
    report_rows = []
    with torch.no_grad():
        for row in tqdm(rows, desc="Round1 local refine"):
            pid = str(row.get("id", "")).strip()
            out = dict(row)

            if int(_to_float(row.get("is_pseudo", 1), default=1)) != 1:
                out_rows.append(out)
                report_rows.append(
                    {
                        "id": pid,
                        "accepted": 0,
                        "reason": "non_pseudo",
                        "refine_pixels": 0,
                        "iou_old_new": 1.0,
                        "boundary_quality_old": 0.0,
                        "boundary_quality_new": 0.0,
                        "boundary_quality_gain": 0.0,
                    }
                )
                continue

            image_path = _resolve(str(row.get("image_path", "")).strip(), manifest_path)
            mask_path = _resolve(str(row.get("mask_path", "")).strip(), manifest_path)
            soft_text = str(row.get("soft_path", "")).strip()
            edge_text = str(row.get("edge_path", "")).strip()
            soft_path = _resolve(soft_text, manifest_path) if soft_text else Path("")
            edge_path = _resolve(edge_text, manifest_path) if edge_text else Path("")

            if not image_path.exists():
                raise RuntimeError(f"Missing image for id={pid}: {image_path}")
            if not mask_path.exists():
                raise RuntimeError(f"Missing mask for id={pid}: {mask_path}")

            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Cannot read image for id={pid}: {image_path}")
            h, w = image_bgr.shape[:2]

            teacher_mask = _load_mask_bin(mask_path, h=h, w=w)
            if soft_path and soft_path.exists():
                teacher_soft = _load_float_map(soft_path, h=h, w=w)
            else:
                teacher_soft = teacher_mask.astype(np.float32)

            if edge_path and edge_path.exists():
                teacher_edge = _load_float_map(edge_path, h=h, w=w)
            else:
                teacher_edge = _sobel_edge_map(teacher_soft)

            _, _, _, bq_old = _boundary_quality(teacher_mask, teacher_edge)

            x = _prepare_tensor(image_bgr, img_size=int(img_size)).to(args.device)
            seg_logits = _parse_model_outputs(model(x))
            if seg_logits is None:
                raise RuntimeError(f"Model returned empty seg logits for id={pid}")
            prob_small = torch.sigmoid(seg_logits)[0, 0].detach().float().cpu().numpy()
            student_prob = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            student_prob = np.clip(student_prob, 0.0, 1.0)

            refine_mask = (
                (student_prob >= float(args.high_conf_threshold))
                & (np.abs(student_prob - teacher_soft) <= float(args.max_disagree))
            )
            refine_n = int(refine_mask.sum())
            if refine_n <= 0:
                out_rows.append(out)
                report_rows.append(
                    {
                        "id": pid,
                        "accepted": 0,
                        "reason": "no_refine_pixels",
                        "refine_pixels": refine_n,
                        "iou_old_new": 1.0,
                        "boundary_quality_old": float(bq_old),
                        "boundary_quality_new": float(bq_old),
                        "boundary_quality_gain": 0.0,
                    }
                )
                continue

            refined_soft = teacher_soft.copy()
            refined_soft[refine_mask] = 0.65 * teacher_soft[refine_mask] + 0.35 * student_prob[refine_mask]
            if float(args.soft_sigma) > 1e-6:
                refined_soft = cv2.GaussianBlur(
                    refined_soft,
                    (0, 0),
                    sigmaX=float(args.soft_sigma),
                    sigmaY=float(args.soft_sigma),
                )
            refined_soft = np.clip(refined_soft, 0.0, 1.0).astype(np.float32)

            refined_hard = (refined_soft >= float(args.pred_threshold)).astype(np.uint8)
            refined_hard, largest_cc_ratio, num_components = _component_filter(
                refined_hard,
                min_area_ratio=float(args.min_component_area_ratio),
                keep_max_components=int(args.keep_max_components),
            )
            refined_edge = _sobel_edge_map(refined_soft)
            _, _, _, bq_new = _boundary_quality(refined_hard, refined_edge)
            iou_old_new = _mask_iou(teacher_mask, refined_hard)
            bq_gain = float(bq_new - bq_old)

            accept = bool(bq_gain >= float(args.min_boundary_quality_gain) and iou_old_new >= float(args.min_iou_keep))
            reason = "accepted" if accept else "rejected_by_gate"

            if accept:
                out_hard = hard_dir / f"{pid}.png"
                out_soft = soft_dir / f"{pid}.npy"
                out_edge = edge_dir / f"{pid}.npy"
                cv2.imwrite(str(out_hard), (refined_hard > 0).astype(np.uint8) * 255)
                np.save(out_soft, refined_soft)
                np.save(out_edge, refined_edge)

                edge_q = _edge_quality((refined_hard > 0).astype(np.uint8) * 255, refined_edge)
                tier, tier_scale = _tier_and_scale(float(bq_new))
                quality_pre = _to_float(
                    row.get(
                        "pseudo_weight_raw",
                        row.get("pseudo_weight_final", row.get("pseudo_weight", 0.0)),
                    ),
                    default=0.0,
                )
                quality_post = float(np.clip(0.55 * quality_pre + 0.30 * float(bq_new) + 0.15 * float(edge_q), 0.0, 1.0))
                pseudo_weight_final = float(np.clip(quality_post * float(tier_scale), 0.0, 1.0))
                out["mask_path"] = str(out_hard)
                out["soft_path"] = str(out_soft)
                out["edge_path"] = str(out_edge)
                out["pseudo_weight"] = float(pseudo_weight_final)
                out["pseudo_weight_raw"] = float(quality_post)
                out["pseudo_weight_final"] = float(pseudo_weight_final)
                out["tier"] = str(tier)

            out_rows.append(out)
            report_rows.append(
                {
                    "id": pid,
                    "accepted": int(accept),
                    "reason": reason,
                    "refine_pixels": int(refine_n),
                    "iou_old_new": float(iou_old_new),
                    "boundary_quality_old": float(bq_old),
                    "boundary_quality_new": float(bq_new),
                    "boundary_quality_gain": float(bq_gain),
                    "largest_cc_ratio_new": float(largest_cc_ratio),
                    "num_components_new": int(num_components),
                }
            )

    out_fields = list(fieldnames)
    for k in ["pseudo_weight_raw", "pseudo_weight_final", "tier", "soft_path", "edge_path", "mask_path", "pseudo_weight"]:
        if k not in out_fields:
            out_fields.append(k)
    _write_csv(out_manifest, out_rows, fieldnames=out_fields)

    report_fields = [
        "id",
        "accepted",
        "reason",
        "refine_pixels",
        "iou_old_new",
        "boundary_quality_old",
        "boundary_quality_new",
        "boundary_quality_gain",
        "largest_cc_ratio_new",
        "num_components_new",
    ]
    _write_csv(report_csv, report_rows, fieldnames=report_fields)

    accepted = [r for r in report_rows if int(_to_int(r.get("accepted", 0))) == 1]
    gains = [float(r.get("boundary_quality_gain", 0.0)) for r in report_rows]
    summary = {
        "input_manifest": str(manifest_path),
        "checkpoint": str(args.checkpoint),
        "output_manifest": str(out_manifest),
        "output_report_csv": str(report_csv),
        "num_input": int(len(rows)),
        "num_accepted": int(len(accepted)),
        "accept_ratio": float(len(accepted)) / float(max(1, len(rows))),
        "boundary_quality_gain_mean": float(np.mean(gains)) if gains else 0.0,
        "high_conf_threshold": float(args.high_conf_threshold),
        "max_disagree": float(args.max_disagree),
        "min_boundary_quality_gain": float(args.min_boundary_quality_gain),
        "min_iou_keep": float(args.min_iou_keep),
        "soft_sigma": float(args.soft_sigma),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
