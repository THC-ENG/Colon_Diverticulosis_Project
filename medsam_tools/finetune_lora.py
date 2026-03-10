import argparse
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
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
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
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            replaced += 1
        else:
            replaced += inject_lora(child, target_keywords, rank, alpha)
    return replaced


class MedSamTuneDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, image_size: int = 1024):
        self.image_size = image_size
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        image_map = {p.stem: p for p in image_dir.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]}
        mask_map = {p.stem: p for p in mask_dir.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]}
        self.keys = sorted(set(image_map.keys()) & set(mask_map.keys()))
        self.image_map = image_map
        self.mask_map = mask_map
        if not self.keys:
            raise ValueError("No overlapping image/mask pairs found for MedSAM fine-tuning.")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        image = cv2.imread(str(self.image_map[key]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_map[key]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

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

        image_t = torch.from_numpy(image).permute(2, 0, 1).float()  # 0..255
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to medsam_vit_b.pth")
    parser.add_argument("--model-type", type=str, default="vit_b")
    parser.add_argument("--image-dir", type=str, default="data/processed_images/images")
    parser.add_argument("--mask-dir", type=str, default="data/processed_images/masks")
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

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)

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

    trainable = [p for p in sam.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found. Check LoRA injection keywords.")

    print(f"[LoRA] injected layers: {replaced}, trainable params: {sum(p.numel() for p in trainable)}")

    dataset = MedSamTuneDataset(args.image_dir, args.mask_dir, image_size=args.image_size)
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
            images = batch["image"].to(args.device) / 255.0
            masks = batch["mask"].to(args.device)
            boxes = batch["box"].to(args.device)
            batch_size = images.size(0)

            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0

            for i in range(batch_size):
                outputs = sam(
                    [
                        {
                            "image": images[i],
                            "boxes": boxes[i].unsqueeze(0),
                            "original_size": (args.image_size, args.image_size),
                        }
                    ],
                    multimask_output=False,
                )[0]

                if "low_res_logits" in outputs:
                    logits = outputs["low_res_logits"]  # 1,1,256,256
                else:
                    pred = outputs["masks"].float()
                    logits = torch.logit(pred.clamp(1e-4, 1 - 1e-4))

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
