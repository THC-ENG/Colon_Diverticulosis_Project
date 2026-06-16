import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_baseline_model
from utils.augmentations import ValAugmentor
from utils.data_protocol import load_protocol_samples, summarize_samples, validate_protocol_samples
from utils.dataset import ProtocolSegDataset


def _load_checkpoint(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


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


def _build_parser():
    p = argparse.ArgumentParser(description="Predict external masks with a local baseline checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, default="")
    p.add_argument("--base-channels", type=int, default=0)
    p.add_argument("--norm-type", type=str, default="")
    p.add_argument("--data-manifest", type=str, default="data/joint_polyp_v1/manifest/samples_v1.csv")
    p.add_argument("--data-root", type=str, default="data/joint_polyp_v1")
    p.add_argument("--manifest-mode", type=str, default="prefer", choices=["prefer", "only", "off"])
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pred-threshold", type=float, default=0.5)
    p.add_argument("--pred-root", type=str, required=True)
    p.add_argument("--report-path", type=str, default="")
    p.add_argument("--per-sample-csv", type=str, default="")
    p.add_argument("--skip-eval", action=argparse.BooleanOptionalAction, default=False)
    return p


def _final_output(outputs):
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs


def main():
    args = _build_parser().parse_args()
    ckpt = _load_checkpoint(args.checkpoint, args.device)

    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model_name = args.model or ckpt.get("model_name", "") or ckpt_args.get("model", "")
    if not model_name:
        raise RuntimeError("Model name missing. Pass --model or use a checkpoint from train_baseline_seg.py.")

    model_kwargs = dict(ckpt.get("model_kwargs", {}) if isinstance(ckpt, dict) else {})
    if args.base_channels > 0:
        model_kwargs["base_channels"] = int(args.base_channels)
    if args.norm_type:
        model_kwargs["norm_type"] = args.norm_type
    model_kwargs.setdefault("in_channels", 3)
    model_kwargs.setdefault("num_classes", 1)
    model_kwargs.setdefault("base_channels", int(ckpt_args.get("base_channels", 32)))
    model_kwargs.setdefault("norm_type", ckpt_args.get("norm_type", "gn"))

    model = build_baseline_model(model_name, **model_kwargs).to(args.device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    samples = load_protocol_samples(args.data_manifest, args.data_root, args.manifest_mode)
    validate_protocol_samples(samples)
    external_rows = _select_external(samples)
    print(f"[protocol] {json.dumps(summarize_samples(samples), ensure_ascii=False)}")
    print(f"[external] n={len(external_rows)}")

    dataset = ProtocolSegDataset(
        external_rows,
        transform=ValAugmentor((args.img_size, args.img_size)),
        mask_threshold=127,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    pred_root = Path(args.pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict"):
            images = batch["image"].to(args.device)
            ids = list(batch["id"])
            sources = list(batch["source"])
            logits = _final_output(model(images))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            for i, sample_id in enumerate(ids):
                source = str(sources[i])
                out_dir = pred_root / source
                out_dir.mkdir(parents=True, exist_ok=True)
                mask = (probs[i, 0] >= float(args.pred_threshold)).astype(np.uint8) * 255
                cv2.imwrite(str(out_dir / f"{sample_id}.png"), mask)

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
            str(args.img_size),
        ]
        print("[eval]", " ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True)

    print(f"[predictions saved] {pred_root}")


if __name__ == "__main__":
    main()
