from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import albumentations as album
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_external_rows(manifest_path: Path) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("subset", "")).strip() != "external":
                continue
            split = str(row.get("split", "")).strip()
            if split and split != "test":
                continue
            if not str(row.get("mask_path", "")).strip():
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No external test rows found in {manifest_path}")
    return rows


class ExternalImageDataset(Dataset):
    def __init__(self, rows: list[dict], width: int, height: int, preprocessing_fn):
        self.rows = list(rows)
        self.width = int(width)
        self.height = int(height)
        self.preprocessing = album.Compose(
            [
                album.Lambda(image=preprocessing_fn),
                album.Lambda(image=lambda x, **kwargs: x.transpose(2, 0, 1).astype("float32")),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = self.preprocessing(image=image)["image"]
        return {
            "id": str(row["id"]),
            "source": str(row.get("source", "")),
            "image": torch.from_numpy(image),
        }


def _load_model(path: Path, device: str) -> torch.nn.Module:
    model = torch.load(str(path), map_location=device, weights_only=False)
    if hasattr(model, "encoder"):
        encoder = model.encoder
        if not hasattr(encoder, "_drop_connect_rate") and hasattr(encoder, "_global_params"):
            encoder._drop_connect_rate = getattr(encoder._global_params, "drop_connect_rate", 0.0)
        if not hasattr(encoder, "_out_indexes") and hasattr(encoder, "_stage_idxs"):
            encoder._out_indexes = [int(i) - 1 for i in encoder._stage_idxs]
    return model.to(device).eval()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict external masks with the HuggingFace DeepLabV3+ polyp checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/public_weights/hf_deeplabv3plus_polyp/best_model.pth",
    )
    parser.add_argument("--manifest", type=str, default="data/joint_polyp_v1/manifest/samples_v1.csv")
    parser.add_argument("--pred-root", type=str, default="results/public_weight_baselines/hf_deeplabv3plus_polyp/preds")
    parser.add_argument("--report-path", type=str, default="results/public_weight_baselines/hf_deeplabv3plus_polyp/report.json")
    parser.add_argument("--per-sample-csv", type=str, default="results/public_weight_baselines/hf_deeplabv3plus_polyp/per_sample.csv")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--encoder-name", type=str, default="efficientnet-b4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-eval", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    rows = _read_external_rows(manifest_path)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, "imagenet")
    model = _load_model(checkpoint, args.device)

    dataset = ExternalImageDataset(rows, width=args.width, height=args.height, preprocessing_fn=preprocessing_fn)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=str(args.device).startswith("cuda"),
    )

    pred_root = Path(args.pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict HF DeepLabV3+"):
            images = batch["image"].to(args.device)
            probs = model(images).detach().cpu().numpy()
            ids = list(batch["id"])
            sources = list(batch["source"])
            polyp_probs = probs[:, 1]
            for i, sample_id in enumerate(ids):
                source = str(sources[i])
                out_dir = pred_root / source
                out_dir.mkdir(parents=True, exist_ok=True)
                mask = (polyp_probs[i] >= float(args.threshold)).astype(np.uint8) * 255
                cv2.imwrite(str(out_dir / f"{sample_id}.png"), mask)

    meta = {
        "model": "HuggingFace DeepLabV3+ polyp",
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path),
        "external_samples": len(rows),
        "width": int(args.width),
        "height": int(args.height),
        "eval_size": int(args.eval_size),
        "threshold": float(args.threshold),
        "encoder_name": str(args.encoder_name),
    }
    pred_root.parent.mkdir(parents=True, exist_ok=True)
    (pred_root.parent / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not args.skip_eval:
        eval_cmd = [
            sys.executable,
            "tools/eval_pred_masks_external.py",
            "--manifest",
            str(manifest_path),
            "--pred-root",
            str(pred_root),
            "--pred-template",
            "{source}/{id}.png",
            "--report-path",
            str(Path(args.report_path)),
            "--per-sample-csv",
            str(Path(args.per_sample_csv)),
        ]
        if int(args.eval_size) > 0:
            eval_cmd.extend(["--eval-size", str(args.eval_size)])
        print("[eval]", " ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True, cwd=str(REPO_ROOT))

    print(f"[predictions saved] {pred_root}")


if __name__ == "__main__":
    main()
