import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medsam_tools.medsam_binding import load_sam_components
from utils.data_protocol import load_protocol_samples, summarize_samples, validate_protocol_samples


def _load_boxes(path: str) -> dict[str, np.ndarray]:
    boxes = {}
    p = Path(path)
    if p.suffix.lower() == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        for k, v in payload.items():
            if isinstance(v, list) and len(v) == 4:
                boxes[str(k)] = np.array(v, dtype=np.float32)
        return boxes

    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("id", "")).strip()
            if not sid:
                continue
            boxes[sid] = np.array(
                [float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])],
                dtype=np.float32,
            )
    return boxes


def _select_external(samples):
    rows = [
        s for s in samples
        if s.subset == "external"
        and s.mask_path
        and s.split in {"", "test"}
    ]
    if not rows:
        raise RuntimeError("No external test rows found.")
    return rows


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    out = box.astype(np.float32).copy()
    out[0] = np.clip(out[0], 0, max(0, width - 1))
    out[2] = np.clip(out[2], 0, max(0, width - 1))
    out[1] = np.clip(out[1], 0, max(0, height - 1))
    out[3] = np.clip(out[3], 0, max(0, height - 1))
    if out[2] <= out[0]:
        out[2] = min(float(width - 1), out[0] + 1.0)
    if out[3] <= out[1]:
        out[3] = min(float(height - 1), out[1] + 1.0)
    return out


def _draw_preview(image_bgr: np.ndarray, mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay[mask > 0] = (0, 0, 255)
    out = cv2.addWeighted(image_bgr, 0.72, overlay, 0.28, 0)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
    return out


def _build_parser():
    p = argparse.ArgumentParser(description="Predict external masks with SAM-family box prompts.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model-type", type=str, default="vit_b")
    p.add_argument("--backend", type=str, default="medsam", choices=["medsam", "mobile_sam"])
    p.add_argument("--mobile-sam-root", type=str, default="external_models/MobileSAM")
    p.add_argument("--boxes", type=str, required=True, help="Box prompt CSV/JSON from generate_box_prompts.py")
    p.add_argument("--data-manifest", type=str, default="data/joint_polyp_v1/manifest/samples_v1.csv")
    p.add_argument("--data-root", type=str, default="data/joint_polyp_v1")
    p.add_argument("--manifest-mode", type=str, default="prefer", choices=["prefer", "only", "off"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pred-root", type=str, required=True)
    p.add_argument("--report-path", type=str, default="")
    p.add_argument("--per-sample-csv", type=str, default="")
    p.add_argument("--preview-dir", type=str, default="")
    p.add_argument("--max-preview", type=int, default=20)
    p.add_argument("--eval-size", type=int, default=256)
    p.add_argument("--skip-eval", action=argparse.BooleanOptionalAction, default=False)
    return p


def _load_sam_backend(args):
    if args.backend == "medsam":
        SamPredictor, sam_model_registry, backend_info = load_sam_components(caller="predict_sam_box_external.py")
        print(
            "[sam backend] "
            f"backend=medsam medsam={backend_info['medsam_version']} "
            f"segment_anything_file={backend_info['segment_anything_file']}"
        )
        return SamPredictor, sam_model_registry

    mobile_root = Path(args.mobile_sam_root).resolve()
    if not mobile_root.exists():
        raise FileNotFoundError(f"MobileSAM root not found: {mobile_root}")
    if str(mobile_root) not in sys.path:
        sys.path.insert(0, str(mobile_root))
    import mobile_sam  # noqa: PLC0415

    print(f"[sam backend] backend=mobile_sam root={mobile_root} file={mobile_sam.__file__}")
    return mobile_sam.SamPredictor, mobile_sam.sam_model_registry


def main():
    args = _build_parser().parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    boxes = _load_boxes(args.boxes)
    if not boxes:
        raise RuntimeError(f"No boxes loaded from: {args.boxes}")

    samples = load_protocol_samples(args.data_manifest, args.data_root, args.manifest_mode)
    validate_protocol_samples(samples)
    rows = _select_external(samples)
    print(f"[protocol] {json.dumps(summarize_samples(samples), ensure_ascii=False)}")
    print(f"[external] n={len(rows)} boxes={len(boxes)}")

    SamPredictor, sam_model_registry = _load_sam_backend(args)
    if args.model_type not in sam_model_registry:
        raise KeyError(f"model_type={args.model_type} not found in sam_model_registry")

    model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    model.eval()
    predictor = SamPredictor(model)

    pred_root = Path(args.pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(args.preview_dir) if args.preview_dir else pred_root.parent / "previews"
    if int(args.max_preview) > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    missing_boxes = []
    preview_count = 0
    for row in tqdm(rows, desc="SAM Box Predict"):
        if row.id not in boxes:
            missing_boxes.append(row.id)
            continue

        image_bgr = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {row.image_path}")
        h, w = image_bgr.shape[:2]
        box = _clip_box(boxes[row.id], width=w, height=h)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            box=box[None, :],
            point_coords=None,
            point_labels=None,
            multimask_output=False,
        )
        mask_u8 = (masks[0] > 0).astype(np.uint8) * 255

        out_dir = pred_root / row.source
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{row.id}.png"), mask_u8)

        if preview_count < int(args.max_preview):
            preview = _draw_preview(image_bgr=image_bgr, mask=mask_u8, box=box)
            cv2.imwrite(str(preview_dir / f"{row.id}.jpg"), preview)
            preview_count += 1

    if missing_boxes:
        raise RuntimeError(f"Missing {len(missing_boxes)} boxes. First missing: {missing_boxes[0]}")

    if not args.skip_eval:
        report_path = args.report_path or str(pred_root.parent / "report.json")
        per_sample_csv = args.per_sample_csv or str(pred_root.parent / "per_sample.csv")
        eval_cmd = [
            sys.executable,
            "tools/eval_pred_masks_external.py",
            "--manifest",
            args.data_manifest,
            "--pred-root",
            str(pred_root),
            "--pred-template",
            "{source}/{id}.png",
            "--report-path",
            report_path,
            "--per-sample-csv",
            per_sample_csv,
            "--eval-size",
            str(args.eval_size),
        ]
        print("[eval]", " ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True)

    print(f"[predictions saved] {pred_root}")


if __name__ == "__main__":
    main()
