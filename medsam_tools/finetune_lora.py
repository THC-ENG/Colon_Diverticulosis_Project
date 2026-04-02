import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from segment_anything import sam_model_registry


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        # Keep LoRA params on the same device/dtype as the wrapped linear layer.
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


def inject_lora(module: nn.Module, target_keywords: List[str], rank: int, alpha: int) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(k in name.lower() for k in target_keywords):
            lora = LoRALinear(child, rank=rank, alpha=alpha)
            lora = lora.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, lora)
            replaced += 1
        else:
            replaced += inject_lora(child, target_keywords, rank, alpha)
    return replaced


def _load_manifest_pairs(
    manifest_path: str,
    subset_filter: set[str] | None,
    split_filter: set[str] | None,
) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subset = str(row.get("subset", "")).strip()
            split = str(row.get("split", "")).strip()
            mask_path = str(row.get("mask_path", "")).strip()
            if subset_filter is not None and subset not in subset_filter:
                continue
            if split_filter is not None and split not in split_filter:
                continue
            if not mask_path:
                continue

            img_path = str(row.get("image_path", "")).strip()
            if not img_path:
                continue
            pid = str(row.get("id", "")).strip() or Path(img_path).stem

            img = Path(img_path if Path(img_path).is_absolute() else Path.cwd() / img_path)
            msk = Path(mask_path if Path(mask_path).is_absolute() else Path.cwd() / mask_path)
            if img.exists() and msk.exists():
                pairs.append((pid, str(img), str(msk)))
    if not pairs:
        raise RuntimeError(f"No valid image/mask pairs loaded from manifest: {manifest_path}")
    return pairs


class MedSamTuneDataset(Dataset):
    def __init__(
        self,
        image_dir: str | None = None,
        mask_dir: str | None = None,
        image_size: int = 1024,
        pairs: List[Tuple[str, str, str]] | None = None,
        mask_threshold: int = 127,
    ):
        self.image_size = image_size
        self.mask_threshold = int(mask_threshold)
        self.items: List[Tuple[str, Path, Path]] = []

        if pairs is not None:
            self.items = [(pid, Path(img), Path(msk)) for pid, img, msk in pairs]
        else:
            if image_dir is None or mask_dir is None:
                raise ValueError("image_dir/mask_dir are required when pairs is None")
            image_dir_p = Path(image_dir)
            mask_dir_p = Path(mask_dir)
            image_map = {p.stem: p for p in image_dir_p.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]}
            mask_map = {p.stem: p for p in mask_dir_p.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]}
            keys = sorted(set(image_map.keys()) & set(mask_map.keys()))
            self.items = [(k, image_map[k], mask_map[k]) for k in keys]

        if not self.items:
            raise ValueError("No image/mask pairs found for MedSAM fine-tuning.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        key, image_path, mask_path = self.items[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > self.mask_threshold).astype(np.uint8)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        yx = np.where(mask > 0)
        if len(yx[0]) == 0:
            box = np.array([0, 0, self.image_size - 1, self.image_size - 1], dtype=np.float32)
        else:
            y_min, y_max = yx[0].min(), yx[0].max()
            x_min, x_max = yx[1].min(), yx[1].max()
            jitter = random.randint(0, max(2, self.image_size // 50))
            box = np.array(
                [
                    max(0, x_min - jitter),
                    max(0, y_min - jitter),
                    min(self.image_size - 1, x_max + jitter),
                    min(self.image_size - 1, y_max + jitter),
                ],
                dtype=np.float32,
            )

        image_t = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        box_t = torch.from_numpy(box).float()
        return {"image": image_t, "mask": mask_t, "box": box_t, "id": key}


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (probs * target).sum(dim=1)
    dice = (2.0 * inter + smooth) / (probs.sum(dim=1) + target.sum(dim=1) + smooth)
    return 1.0 - dice.mean()


def _safe_torch_load(path: str, map_location: str | torch.device | None = None):
    # Prefer weights_only=True to avoid pickle execution and future warning noise.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _forward_lowres_logits_with_grad(sam: nn.Module, image_chw: torch.Tensor, box_xyxy: torch.Tensor) -> torch.Tensor:
    # image_chw: [3,H,W] in [0,255], box_xyxy: [4]
    input_image = sam.preprocess(image_chw)
    image_embeddings = sam.image_encoder(input_image.unsqueeze(0))

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        boxes=box_xyxy.unsqueeze(0),
        masks=None,
    )
    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to medsam_vit_b.pth")
    parser.add_argument("--model-type", type=str, default="vit_b")
    parser.add_argument("--image-dir", type=str, default="data/processed_images/images")
    parser.add_argument("--mask-dir", type=str, default="data/processed_images/masks")
    parser.add_argument("--data-manifest", type=str, default="")
    parser.add_argument("--subset-filter", type=str, default="L_small")
    parser.add_argument("--split-filter", type=str, default="train,val")
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--init-lora-checkpoint", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="checkpoints/medsam_lora_best.pth")
    parser.add_argument("--train-mask-decoder", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(Path(args.save_path).parent, exist_ok=True)

    sam = sam_model_registry[args.model_type](checkpoint=None).to(args.device)
    base_state = _safe_torch_load(args.checkpoint, map_location=args.device)
    if isinstance(base_state, dict) and "state_dict" in base_state and isinstance(base_state["state_dict"], dict):
        base_state = base_state["state_dict"]
    missing0, unexpected0 = sam.load_state_dict(base_state, strict=False)
    print(f"[base checkpoint] missing={len(missing0)} unexpected={len(unexpected0)} from {args.checkpoint}")
    for p in sam.parameters():
        p.requires_grad = False

    replaced = inject_lora(
        sam.image_encoder,
        target_keywords=["qkv", "proj", "q_proj", "k_proj", "v_proj"],
        rank=args.rank,
        alpha=args.alpha,
    )
    if args.train_mask_decoder:
        for p in sam.mask_decoder.parameters():
            p.requires_grad = True

    if args.init_lora_checkpoint:
        ckpt = _safe_torch_load(args.init_lora_checkpoint, map_location=args.device)
        state = ckpt.get("sam_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = sam.load_state_dict(state, strict=False)
        print(f"[resume] missing={len(missing)} unexpected={len(unexpected)} from {args.init_lora_checkpoint}")

    trainable = [p for p in sam.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found. Check LoRA injection keywords.")
    print(f"[LoRA] injected layers: {replaced}, trainable params: {sum(p.numel() for p in trainable)}")

    if args.data_manifest:
        subsets = {s.strip() for s in args.subset_filter.split(",") if s.strip()} if args.subset_filter else None
        splits = {s.strip() for s in args.split_filter.split(",") if s.strip()} if args.split_filter else None
        pairs = _load_manifest_pairs(args.data_manifest, subset_filter=subsets, split_filter=splits)
        dataset = MedSamTuneDataset(
            image_size=args.image_size,
            pairs=pairs,
            mask_threshold=args.mask_threshold,
        )
    else:
        dataset = MedSamTuneDataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            image_size=args.image_size,
            mask_threshold=args.mask_threshold,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    sam.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            # SAM preprocess expects image values in [0,255], so keep raw scale here.
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            boxes = batch["box"].to(args.device)
            batch_size = images.size(0)

            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0

            for i in range(batch_size):
                logits = _forward_lowres_logits_with_grad(
                    sam=sam,
                    image_chw=images[i],
                    box_xyxy=boxes[i],
                )

                target_low = F.interpolate(masks[i].unsqueeze(0), size=logits.shape[-2:], mode="nearest")
                loss = bce(logits, target_low) + dice_loss_from_logits(logits, target_low)
                loss_total = loss_total + loss

            loss_total = loss_total / batch_size
            loss_total.backward()
            optimizer.step()

            epoch_loss += loss_total.item() * batch_size
            pbar.set_postfix(loss=f"{loss_total.item():.4f}")

        epoch_loss /= len(dataset)
        print(f"epoch {epoch}/{args.epochs} | loss={epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "sam_state_dict": sam.state_dict(),
                    "best_loss": best_loss,
                    "epoch": epoch,
                    "lora_rank": args.rank,
                    "lora_alpha": args.alpha,
                },
                args.save_path,
            )
            print(f"[checkpoint] saved -> {args.save_path}")


if __name__ == "__main__":
    main()
