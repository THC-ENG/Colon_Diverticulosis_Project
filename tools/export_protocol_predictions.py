import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_eval import (  # noqa: E402
    _load_checkpoint as _load_student_checkpoint,
    _load_model_state,
    _model_kwargs_from_checkpoint,
    _parse_model_outputs,
)
from models import build_baseline_model  # noqa: E402
from models.res_swin_unet import ResSwinUNet  # noqa: E402
from utils.augmentations import ValAugmentor  # noqa: E402
from utils.data_protocol import load_protocol_samples, summarize_samples, validate_protocol_samples  # noqa: E402
from utils.dataset import ProtocolSegDataset  # noqa: E402


def _build_parser():
    p = argparse.ArgumentParser(
        description="Export prediction masks/probabilities for a protocol subset such as U_large."
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model-type", type=str, required=True, choices=["ours", "baseline"])
    p.add_argument("--model", type=str, default="", help="Required for baseline if checkpoint metadata lacks model_name.")
    p.add_argument("--base-channels", type=int, default=0)
    p.add_argument("--norm-type", type=str, default="")
    p.add_argument("--data-manifest", type=str, default="data/joint_polyp_v1/manifest/samples_v1.csv")
    p.add_argument("--data-root", type=str, default="data/joint_polyp_v1")
    p.add_argument("--manifest-mode", type=str, default="prefer", choices=["prefer", "only", "off"])
    p.add_argument("--subset", type=str, default="U_large")
    p.add_argument("--split", type=str, default="", help="Optional exact split filter, e.g. unlabeled/test.")
    p.add_argument("--ids-file", type=str, default="", help="Optional one-id-per-line filter.")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pred-threshold", type=float, default=0.5)
    p.add_argument("--pred-root", type=str, required=True)
    p.add_argument("--save-probs", action=argparse.BooleanOptionalAction, default=True)
    return p


def _torch_load(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_ids(path: str) -> set[str] | None:
    if not path:
        return None
    ids = {
        line.strip().lstrip("\ufeff")
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip().lstrip("\ufeff") and not line.strip().lstrip("\ufeff").startswith("#")
    }
    if not ids:
        raise RuntimeError(f"No ids found in {path}")
    return ids


def _select_rows(samples, subset: str, split: str, ids: set[str] | None):
    subset_norm = subset.strip()
    split_norm = split.strip()
    rows = []
    for sample in samples:
        if subset_norm and sample.subset != subset_norm:
            continue
        if split_norm and sample.split != split_norm:
            continue
        if ids is not None and sample.id not in ids:
            continue
        rows.append(sample)
    if not rows:
        raise RuntimeError(
            f"No rows selected for subset={subset_norm!r}, split={split_norm!r}, "
            f"ids={0 if ids is None else len(ids)}."
        )
    return rows


def _final_output(outputs):
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs


def _build_model(args, ckpt):
    if args.model_type == "ours":
        model_kwargs = _model_kwargs_from_checkpoint(ckpt)
        model_kwargs.setdefault("num_classes", 1)
        model = ResSwinUNet(**model_kwargs).to(args.device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        _load_model_state(model, state_dict)
        return model, {"model_type": "ours", "model_kwargs": model_kwargs}

    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model_name = args.model or ckpt.get("model_name", "") or ckpt_args.get("model", "")
    if not model_name:
        raise RuntimeError("Baseline model name missing. Pass --model.")

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
    return model, {"model_type": "baseline", "model": model_name, "model_kwargs": model_kwargs}


def main():
    args = _build_parser().parse_args()
    pred_root = Path(args.pred_root)
    hard_dir = pred_root / "hard_masks"
    prob_dir = pred_root / "soft_probs"
    hard_dir.mkdir(parents=True, exist_ok=True)
    if args.save_probs:
        prob_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _load_student_checkpoint(args.checkpoint, args.device) if args.model_type == "ours" else _torch_load(args.checkpoint, args.device)
    model, model_meta = _build_model(args, ckpt)
    model.eval()

    samples = load_protocol_samples(args.data_manifest, args.data_root, args.manifest_mode)
    validate_protocol_samples(samples)
    ids = _load_ids(args.ids_file)
    rows = _select_rows(samples, args.subset, args.split, ids)
    print(f"[protocol] {json.dumps(summarize_samples(samples), ensure_ascii=False)}")
    print(f"[selected] n={len(rows)} subset={args.subset} split={args.split or '*'}")

    dataset = ProtocolSegDataset(rows, transform=ValAugmentor((args.img_size, args.img_size)), mask_threshold=127)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )

    meta_rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predict {args.model_type}"):
            images = batch["image"].to(args.device)
            sample_ids = list(batch["id"])
            outputs = model(images)
            logits = _parse_model_outputs(outputs)["seg"] if args.model_type == "ours" else _final_output(outputs)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            for i, sample_id in enumerate(sample_ids):
                prob = probs[i, 0].astype(np.float32)
                mask = (prob >= float(args.pred_threshold)).astype(np.uint8) * 255
                hard_path = hard_dir / f"{sample_id}.png"
                cv2.imwrite(str(hard_path), mask)
                prob_path = ""
                if args.save_probs:
                    prob_path = str(prob_dir / f"{sample_id}.npy")
                    np.save(prob_path, prob)
                meta_rows.append({"id": sample_id, "hard_mask_path": str(hard_path), "soft_path": prob_path})

    with (pred_root / "selected_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "image_path", "mask_path", "subset", "split", "source", "center"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.id,
                    "image_path": row.image_path,
                    "mask_path": row.mask_path,
                    "subset": row.subset,
                    "split": row.split,
                    "source": row.source,
                    "center": row.center,
                }
            )

    with (pred_root / "prediction_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "hard_mask_path", "soft_path"])
        writer.writeheader()
        writer.writerows(meta_rows)

    meta = {
        "checkpoint": args.checkpoint,
        "subset": args.subset,
        "split": args.split,
        "ids_file": args.ids_file,
        "img_size": args.img_size,
        "threshold": args.pred_threshold,
        "selected_n": len(rows),
        **model_meta,
    }
    (pred_root / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[predictions saved] {pred_root}")


if __name__ == "__main__":
    main()
