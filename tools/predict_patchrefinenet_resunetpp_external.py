from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHREFINENET_ROOT = REPO_ROOT / "external_models" / "PatchRefineNet"
PATCHREFINENET_SEG_MODELS = PATCHREFINENET_ROOT / "src" / "kvasir" / "seg-models"

if str(PATCHREFINENET_SEG_MODELS) not in sys.path:
    sys.path.insert(0, str(PATCHREFINENET_SEG_MODELS))

if "torchsummary" not in sys.modules:
    torchsummary_stub = types.ModuleType("torchsummary")
    torchsummary_stub.summary = lambda *args, **kwargs: None
    sys.modules["torchsummary"] = torchsummary_stub

try:
    from networks.resunetplusplus import ResUnetPlusPlus
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PatchRefineNet ResUNet++ code not found. Clone "
        "https://github.com/savinay95n/PatchRefineNet.git into external_models/PatchRefineNet first."
    ) from exc


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
    def __init__(self, rows: list[dict], image_size: int):
        self.rows = list(rows)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32).transpose(2, 0, 1) / 255.0
        return {
            "id": str(row["id"]),
            "source": str(row.get("source", "")),
            "image": torch.from_numpy(image),
        }


def _torch_load(path: Path, device: str):
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict external masks with PatchRefineNet's public Kvasir ResUNet++ checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to resunetplusplus_weights.th from PatchRefineNet's Kvasir model bundle.",
    )
    parser.add_argument("--manifest", type=str, default="data/joint_polyp_v1/manifest/samples_v1.csv")
    parser.add_argument("--pred-root", type=str, default="results/public_weight_baselines/resunetpp_patchrefinenet/preds")
    parser.add_argument("--report-path", type=str, default="results/public_weight_baselines/resunetpp_patchrefinenet/report.json")
    parser.add_argument("--per-sample-csv", type=str, default="results/public_weight_baselines/resunetpp_patchrefinenet/per_sample.csv")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-eval", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Expected PatchRefineNet Kvasir file "
            "`resunetplusplus_weights.th`."
        )

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    rows = _read_external_rows(manifest_path)
    model = ResUnetPlusPlus().to(args.device)
    model.load_state_dict(_torch_load(checkpoint, args.device), strict=True)
    model.eval()

    dataset = ExternalImageDataset(rows, image_size=args.image_size)
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
        for batch in tqdm(loader, desc="Predict PatchRefineNet ResUNet++"):
            images = batch["image"].to(args.device)
            probs = model(images).detach().cpu().numpy()
            ids = list(batch["id"])
            sources = list(batch["source"])
            for i, sample_id in enumerate(ids):
                source = str(sources[i])
                out_dir = pred_root / source
                out_dir.mkdir(parents=True, exist_ok=True)
                mask = (probs[i, 0] >= float(args.threshold)).astype(np.uint8) * 255
                cv2.imwrite(str(out_dir / f"{sample_id}.png"), mask)

    meta = {
        "model": "PatchRefineNet ResUNet++",
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path),
        "external_samples": len(rows),
        "image_size": int(args.image_size),
        "threshold": float(args.threshold),
    }
    pred_root.parent.mkdir(parents=True, exist_ok=True)
    (pred_root.parent / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not args.skip_eval:
        report_path = Path(args.report_path)
        per_sample_csv = Path(args.per_sample_csv)
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
            str(report_path),
            "--per-sample-csv",
            str(per_sample_csv),
            "--eval-size",
            str(args.image_size),
        ]
        print("[eval]", " ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True, cwd=str(REPO_ROOT))

    print(f"[predictions saved] {pred_root}")


if __name__ == "__main__":
    main()
