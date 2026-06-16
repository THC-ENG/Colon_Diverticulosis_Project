import argparse
import csv
import json
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


def _parse_model_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs.get("seg")
    if isinstance(outputs, (tuple, list)):
        return outputs[0] if len(outputs) > 0 else None
    return outputs


def _build_model_from_checkpoint(ckpt_path: str, device: str, img_size_override: int, required_train_mode: str):
    ckpt = _safe_torch_load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")
    state_dict = ckpt.get("model", ckpt)
    args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    train_mode = str(args.get("mode", "")).strip().lower()

    req_mode = str(required_train_mode).strip().lower()
    if req_mode and req_mode != "off" and train_mode != req_mode:
        raise RuntimeError(
            f"Checkpoint train mode mismatch: expected={req_mode}, actual={train_mode or 'unknown'}."
        )

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
        f"[bridge-model] checkpoint={ckpt_path} mode={train_mode or 'unknown'} img_size={img_size} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    model.eval()
    return model, img_size


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


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    if union <= 0.0:
        return 1.0
    return float(inter / max(1e-6, union))


def _safe_weighted_bridge_quality(
    teacher_quality: float,
    student_mean_prob: float,
    student_teacher_iou: float,
    student_edge_agree: float,
    w_teacher: float,
    w_conf: float,
    w_iou: float,
    w_edge: float,
) -> float:
    ww = np.array([w_teacher, w_conf, w_iou, w_edge], dtype=np.float32)
    vals = np.array(
        [
            np.clip(teacher_quality, 0.0, 1.0),
            np.clip(student_mean_prob, 0.0, 1.0),
            np.clip(student_teacher_iou, 0.0, 1.0),
            np.clip(student_edge_agree, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    s = float(ww.sum())
    if s <= 1e-8:
        ww = np.array([0.35, 0.30, 0.25, 0.10], dtype=np.float32)
        s = float(ww.sum())
    score = float((vals * ww).sum() / s)
    return float(np.clip(score, 0.0, 1.0))


def main():
    parser = argparse.ArgumentParser(description="Score round1 pseudo samples with student bridge metrics.")
    parser.add_argument("--input-manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--required-train-mode", type=str, default="off")
    parser.add_argument("--output-score-csv", type=str, default="")
    parser.add_argument("--output-selected-manifest", type=str, required=True)
    parser.add_argument("--output-summary-json", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img-size", type=int, default=0)
    parser.add_argument("--pred-threshold", type=float, default=0.5)
    parser.add_argument("--min-teacher-quality", type=float, default=0.68)
    parser.add_argument("--min-conf", type=float, default=0.55)
    parser.add_argument("--min-iou", type=float, default=0.45)
    parser.add_argument("--keep-ratio", type=float, default=0.60)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--bridge-weight-teacher", type=float, default=0.35)
    parser.add_argument("--bridge-weight-conf", type=float, default=0.30)
    parser.add_argument("--bridge-weight-iou", type=float, default=0.25)
    parser.add_argument("--bridge-weight-edge", type=float, default=0.10)
    parser.add_argument("--require-soft-edge", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    manifest_path = Path(args.input_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {manifest_path}")
    rows, fieldnames = _read_csv(manifest_path)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_path}")

    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]

    model, img_size = _build_model_from_checkpoint(
        ckpt_path=str(args.checkpoint),
        device=str(args.device),
        img_size_override=int(args.img_size),
        required_train_mode=str(args.required_train_mode),
    )

    score_rows = []
    with torch.no_grad():
        for row in tqdm(rows, desc="Bridge scoring"):
            pid = str(row.get("id", "")).strip()
            image_path = _resolve(str(row.get("image_path", "")).strip(), manifest_path)
            mask_path = _resolve(str(row.get("mask_path", "")).strip(), manifest_path)
            soft_path = _resolve(str(row.get("soft_path", "")).strip(), manifest_path) if str(row.get("soft_path", "")).strip() else Path("")
            edge_path = _resolve(str(row.get("edge_path", "")).strip(), manifest_path) if str(row.get("edge_path", "")).strip() else Path("")

            if not image_path.exists():
                raise RuntimeError(f"Missing image for id={pid}: {image_path}")
            if not mask_path.exists():
                raise RuntimeError(f"Missing mask for id={pid}: {mask_path}")

            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Cannot read image for id={pid}: {image_path}")
            h, w = image_bgr.shape[:2]

            x = _prepare_tensor(image_bgr=image_bgr, img_size=int(img_size)).to(args.device)
            seg_logits = _parse_model_outputs(model(x))
            if seg_logits is None:
                raise RuntimeError(f"Model returned empty segmentation logits for id={pid}")
            prob_small = torch.sigmoid(seg_logits)[0, 0].detach().float().cpu().numpy()
            prob = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            prob = np.clip(prob, 0.0, 1.0)

            teacher_mask = _load_mask_bin(mask_path=mask_path, h=h, w=w)
            student_mask = (prob >= float(args.pred_threshold)).astype(np.uint8)

            if int(student_mask.sum()) > 0:
                student_mean_prob = float(prob[student_mask > 0].mean())
            else:
                student_mean_prob = float(prob.mean())

            student_teacher_iou = _mask_iou(student_mask, teacher_mask)
            student_edge = _sobel_edge_map(prob)

            if edge_path and edge_path.exists():
                teacher_edge = _load_float_map(edge_path, h=h, w=w)
            elif soft_path and soft_path.exists():
                teacher_soft = _load_float_map(soft_path, h=h, w=w)
                teacher_edge = _sobel_edge_map(teacher_soft)
            else:
                teacher_edge = _sobel_edge_map(teacher_mask.astype(np.float32))

            student_edge_agree = float(np.clip(1.0 - np.mean(np.abs(student_edge - teacher_edge)), 0.0, 1.0))

            teacher_quality = _to_float(
                row.get(
                    "pseudo_weight_raw",
                    row.get("pseudo_weight_final", row.get("pseudo_weight", 0.0)),
                ),
                default=0.0,
            )
            bridge_quality = _safe_weighted_bridge_quality(
                teacher_quality=teacher_quality,
                student_mean_prob=student_mean_prob,
                student_teacher_iou=student_teacher_iou,
                student_edge_agree=student_edge_agree,
                w_teacher=float(args.bridge_weight_teacher),
                w_conf=float(args.bridge_weight_conf),
                w_iou=float(args.bridge_weight_iou),
                w_edge=float(args.bridge_weight_edge),
            )

            out = dict(row)
            out["teacher_quality"] = float(np.clip(teacher_quality, 0.0, 1.0))
            out["student_mean_prob"] = float(np.clip(student_mean_prob, 0.0, 1.0))
            out["student_teacher_iou"] = float(np.clip(student_teacher_iou, 0.0, 1.0))
            out["student_edge_agree"] = float(np.clip(student_edge_agree, 0.0, 1.0))
            out["bridge_quality"] = float(np.clip(bridge_quality, 0.0, 1.0))
            score_rows.append(out)

    eligible = [
        r
        for r in score_rows
        if float(r["teacher_quality"]) >= float(args.min_teacher_quality)
        and float(r["student_mean_prob"]) >= float(args.min_conf)
        and float(r["student_teacher_iou"]) >= float(args.min_iou)
    ]

    keep_ratio = float(np.clip(float(args.keep_ratio), 0.0, 1.0))
    if eligible:
        eligible_sorted = sorted(eligible, key=lambda r: float(r.get("bridge_quality", 0.0)), reverse=True)
        keep_n = int(round(float(len(eligible_sorted)) * keep_ratio))
        keep_n = max(1, min(len(eligible_sorted), keep_n))
        kept_rows = eligible_sorted[:keep_n]
    else:
        all_sorted = sorted(score_rows, key=lambda r: float(r.get("bridge_quality", 0.0)), reverse=True)
        keep_n = int(round(float(len(all_sorted)) * keep_ratio)) if all_sorted else 0
        keep_n = max(1, min(len(all_sorted), keep_n)) if all_sorted else 0
        kept_rows = all_sorted[:keep_n]

    kept_ids = {str(r.get("id", "")).strip() for r in kept_rows}
    for r in score_rows:
        r["keep_for_refresh"] = int(str(r.get("id", "")).strip() in kept_ids)

    selected_rows = []
    for r in score_rows:
        pid = str(r.get("id", "")).strip()
        if pid not in kept_ids:
            continue

        soft_text = str(r.get("soft_path", "")).strip()
        edge_text = str(r.get("edge_path", "")).strip()
        soft_ok = bool(soft_text) and _resolve(soft_text, manifest_path).exists()
        edge_ok = bool(edge_text) and _resolve(edge_text, manifest_path).exists()
        if bool(args.require_soft_edge) and (not soft_ok or not edge_ok):
            raise RuntimeError(
                f"Bridge selected row missing soft/edge artifact for id={pid} (soft_ok={soft_ok}, edge_ok={edge_ok})."
            )

        out = dict(r)
        out["pseudo_weight"] = float(r.get("bridge_quality", 0.0))
        out["pseudo_weight_final"] = float(r.get("bridge_quality", 0.0))
        out.pop("keep_for_refresh", None)
        selected_rows.append(out)

    extra_fields = [
        "teacher_quality",
        "student_mean_prob",
        "student_teacher_iou",
        "student_edge_agree",
        "bridge_quality",
    ]
    score_fields = list(fieldnames)
    for k in extra_fields + ["keep_for_refresh"]:
        if k not in score_fields:
            score_fields.append(k)

    selected_fields = list(fieldnames)
    for k in extra_fields:
        if k not in selected_fields:
            selected_fields.append(k)
    if "pseudo_weight_final" not in selected_fields:
        selected_fields.append("pseudo_weight_final")

    output_score_csv = (
        Path(args.output_score_csv)
        if str(args.output_score_csv).strip()
        else Path(args.output_selected_manifest).with_name("bridge_scores.csv")
    )
    output_selected_manifest = Path(args.output_selected_manifest)
    output_summary_json = (
        Path(args.output_summary_json)
        if str(args.output_summary_json).strip()
        else output_selected_manifest.with_name(output_selected_manifest.stem + "_summary.json")
    )

    _write_csv(output_score_csv, score_rows, fieldnames=score_fields)
    _write_csv(output_selected_manifest, selected_rows, fieldnames=selected_fields)

    summary = {
        "input_manifest": str(manifest_path),
        "checkpoint": str(args.checkpoint),
        "num_input": int(len(score_rows)),
        "num_eligible": int(len(eligible)),
        "num_kept": int(len(selected_rows)),
        "keep_ratio": float(keep_ratio),
        "min_teacher_quality": float(args.min_teacher_quality),
        "min_conf": float(args.min_conf),
        "min_iou": float(args.min_iou),
        "output_score_csv": str(output_score_csv),
        "output_selected_manifest": str(output_selected_manifest),
        "require_soft_edge": bool(args.require_soft_edge),
    }
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
