import argparse
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
    _load_checkpoint,
    _load_model_state,
    _model_kwargs_from_checkpoint,
    _parse_model_outputs,
)
from models.res_swin_unet import ResSwinUNet  # noqa: E402
from utils.augmentations import ValAugmentor  # noqa: E402
from utils.data_protocol import load_protocol_samples, summarize_samples, validate_protocol_samples  # noqa: E402
from utils.dataset import ProtocolSegDataset  # noqa: E402


def _build_parser():
    p = argparse.ArgumentParser(description="Export external prediction masks for the final student model.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-manifest", type=str, default="data/joint_polyp_v1/manifest/samples_v1.csv")
    p.add_argument("--data-root", type=str, default="data/joint_polyp_v1")
    p.add_argument("--manifest-mode", type=str, default="prefer", choices=["prefer", "only", "off"])
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pred-threshold", type=float, default=0.5)
    p.add_argument("--pred-root", type=str, required=True)
    return p


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


def main():
    args = _build_parser().parse_args()
    ckpt = _load_checkpoint(args.checkpoint, args.device)
    model_kwargs = _model_kwargs_from_checkpoint(ckpt)
    model_kwargs.setdefault("num_classes", 1)

    model = ResSwinUNet(**model_kwargs).to(args.device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    _load_model_state(model, state_dict)
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )

    pred_root = Path(args.pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict Ours"):
            images = batch["image"].to(args.device)
            ids = list(batch["id"])
            sources = list(batch["source"])
            outputs = _parse_model_outputs(model(images))
            seg_logits = outputs["seg"]
            probs = torch.sigmoid(seg_logits).detach().cpu().numpy()

            for i, sample_id in enumerate(ids):
                source = str(sources[i])
                out_dir = pred_root / source
                out_dir.mkdir(parents=True, exist_ok=True)
                mask = (probs[i, 0] >= float(args.pred_threshold)).astype(np.uint8) * 255
                cv2.imwrite(str(out_dir / f"{sample_id}.png"), mask)

    print(f"[predictions saved] {pred_root}")


if __name__ == "__main__":
    main()
