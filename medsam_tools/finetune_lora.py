import argparse
import csv
import os
import random
import re
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


def _match_any(text: str, keywords: List[str]) -> bool:
    t = str(text).strip().lower()
    return any(str(k).strip().lower() in t for k in keywords if str(k).strip())


def inject_lora(
    module: nn.Module,
    target_keywords: List[str],
    rank: int,
    alpha: int,
    prefix: str = "",
    exclude_keywords: List[str] | None = None,
) -> int:
    replaced = 0
    excludes = exclude_keywords or []
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        full_name_l = full_name.lower()
        if (
            isinstance(child, nn.Linear)
            and _match_any(full_name_l, target_keywords)
            and not _match_any(full_name_l, excludes)
        ):
            lora = LoRALinear(child, rank=rank, alpha=alpha)
            lora = lora.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, lora)
            replaced += 1
        else:
            replaced += inject_lora(
                child,
                target_keywords=target_keywords,
                rank=rank,
                alpha=alpha,
                prefix=full_name,
                exclude_keywords=excludes,
            )
    return replaced


def _encoder_block_index(param_name: str) -> int:
    m = re.search(r"image_encoder\.blocks\.(\d+)\.", str(param_name))
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _collect_encoder_blocks(sam: nn.Module) -> int:
    mx = -1
    for n, _ in sam.named_parameters():
        b = _encoder_block_index(n)
        if b > mx:
            mx = b
    return int(mx + 1) if mx >= 0 else 0


def _set_stage_trainable(
    sam: nn.Module,
    stage_idx: int,
    encoder_tail_start_block: int,
    train_mask_decoder_full: bool,
) -> tuple[int, int]:
    total = 0
    enabled = 0
    for name, p in sam.named_parameters():
        lname = str(name).lower()
        is_lora = ".lora_a" in lname or ".lora_b" in lname
        keep = False
        if is_lora:
            total += int(p.numel())
            if lname.startswith("mask_decoder."):
                keep = True
            elif lname.startswith("image_encoder."):
                if int(stage_idx) >= 3:
                    keep = True
                elif int(stage_idx) == 2:
                    block_idx = _encoder_block_index(lname)
                    keep = block_idx >= int(encoder_tail_start_block) and block_idx >= 0
                else:
                    keep = False
        elif train_mask_decoder_full and lname.startswith("mask_decoder."):
            keep = True
            total += int(p.numel())
        p.requires_grad = bool(keep)
        if keep:
            enabled += int(p.numel())
    return enabled, total


def _parse_stage_epochs(text: str, total_epochs: int) -> List[int]:
    raw = [x.strip() for x in str(text or "").split(",") if x.strip()]
    vals: List[int] = []
    for x in raw:
        try:
            vals.append(max(0, int(x)))
        except Exception:
            pass
    if len(vals) >= 3 and sum(vals[:3]) > 0:
        return vals[:3]
    base = max(1, int(total_epochs) // 3)
    rem = max(0, int(total_epochs) - base * 3)
    out = [base, base, base]
    for i in range(rem):
        out[i % 3] += 1
    return out


def _parse_stage_lrs(text: str, base_lr: float) -> List[float]:
    raw = [x.strip() for x in str(text or "").split(",") if x.strip()]
    vals: List[float] = []
    for x in raw:
        try:
            vals.append(float(x))
        except Exception:
            pass
    if len(vals) >= 3:
        return vals[:3]
    return [float(base_lr), float(base_lr) * 0.5, float(base_lr) * 0.1]


def _stage_for_epoch(epoch: int, stage_epochs: List[int]) -> int:
    c1 = int(stage_epochs[0])
    c2 = int(stage_epochs[0]) + int(stage_epochs[1])
    if int(epoch) <= c1:
        return 1
    if int(epoch) <= c2:
        return 2
    return 3


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
        enable_augment: bool = False,
        augment_prob: float = 0.0,
        box_jitter_scale: float = 0.0,
        box_jitter_shift: float = 0.0,
        box_full_image_prob: float = 0.0,
        num_pos_points: int = 1,
        num_neg_points: int = 1,
        point_jitter_frac: float = 0.03,
    ):
        self.image_size = image_size
        self.mask_threshold = int(mask_threshold)
        self.enable_augment = bool(enable_augment)
        self.augment_prob = float(max(0.0, min(1.0, augment_prob)))
        self.box_jitter_scale = float(max(0.0, box_jitter_scale))
        self.box_jitter_shift = float(max(0.0, box_jitter_shift))
        self.box_full_image_prob = float(max(0.0, min(1.0, box_full_image_prob)))
        self.num_pos_points = int(max(1, num_pos_points))
        self.num_neg_points = int(max(1, num_neg_points))
        self.point_jitter_frac = float(max(0.0, point_jitter_frac))
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

    def _augment_pair(self, image_rgb: np.ndarray, mask_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img = image_rgb
        msk = mask_u8.astype(np.uint8)

        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            msk = cv2.flip(msk, 1)
        if random.random() < 0.25:
            img = cv2.flip(img, 0)
            msk = cv2.flip(msk, 0)

        h, w = img.shape[:2]
        if random.random() < 0.55:
            scale = 1.0 + random.uniform(-0.08, 0.12)
            tx = random.uniform(-0.05, 0.05) * float(w)
            ty = random.uniform(-0.05, 0.05) * float(h)
            m = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            msk = cv2.warpAffine(msk, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if random.random() < 0.75:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10.0, 10.0)) % 180.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.75, 1.30), 0.0, 255.0)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.80, 1.20), 0.0, 255.0)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if random.random() < 0.6:
            gamma = random.uniform(0.75, 1.35)
            img_f = np.power(np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0), gamma)
            img = np.clip(img_f * 255.0, 0.0, 255.0).astype(np.uint8)

        if random.random() < 0.30:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

        if random.random() < 0.18:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), sigmaX=0.0)

        return img, (msk > 0).astype(np.uint8)

    def _jitter_box(self, box: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        if random.random() < self.box_full_image_prob:
            return np.array([0.0, 0.0, float(self.image_size - 1), float(self.image_size - 1)], dtype=np.float32)

        bw = max(2.0, x1 - x0 + 1.0)
        bh = max(2.0, y1 - y0 + 1.0)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)

        if self.box_jitter_scale > 0.0:
            scale = 1.0 + random.uniform(-self.box_jitter_scale, self.box_jitter_scale)
            bw *= max(0.45, scale)
            bh *= max(0.45, scale)
            if random.random() < 0.30:
                loose = 1.0 + random.uniform(0.05, min(0.35, self.box_jitter_scale + 0.05))
                bw *= loose
                bh *= loose
        if self.box_jitter_shift > 0.0:
            cx += random.uniform(-self.box_jitter_shift, self.box_jitter_shift) * bw
            cy += random.uniform(-self.box_jitter_shift, self.box_jitter_shift) * bh

        min_side = max(6.0, float(self.image_size) * 0.03)
        bw = max(min_side, bw)
        bh = max(min_side, bh)
        x0n = cx - 0.5 * bw
        y0n = cy - 0.5 * bh
        x1n = cx + 0.5 * bw
        y1n = cy + 0.5 * bh

        x0n = max(0.0, min(float(self.image_size - 1), x0n))
        y0n = max(0.0, min(float(self.image_size - 1), y0n))
        x1n = max(0.0, min(float(self.image_size - 1), x1n))
        y1n = max(0.0, min(float(self.image_size - 1), y1n))
        if x1n <= x0n:
            x1n = min(float(self.image_size - 1), x0n + 1.0)
        if y1n <= y0n:
            y1n = min(float(self.image_size - 1), y0n + 1.0)
        return np.array([x0n, y0n, x1n, y1n], dtype=np.float32)

    def __len__(self):
        return len(self.items)

    def _sample_point(self, mask: np.ndarray, positive: bool) -> np.ndarray:
        h, w = mask.shape[:2]
        if positive:
            ys, xs = np.where(mask > 0)
        else:
            ys, xs = np.where(mask == 0)
        if len(xs) == 0:
            x = 0.5 * float(max(1, w - 1))
            y = 0.5 * float(max(1, h - 1))
        else:
            j = random.randint(0, len(xs) - 1)
            x = float(xs[j])
            y = float(ys[j])

        if self.point_jitter_frac > 0.0:
            jx = random.uniform(-self.point_jitter_frac, self.point_jitter_frac) * float(max(1, w - 1))
            jy = random.uniform(-self.point_jitter_frac, self.point_jitter_frac) * float(max(1, h - 1))
            x += jx
            y += jy
        x = max(0.0, min(float(max(0, w - 1)), x))
        y = max(0.0, min(float(max(0, h - 1)), y))
        return np.array([x, y], dtype=np.float32)

    def __getitem__(self, idx: int):
        key, image_path, mask_path = self.items[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > self.mask_threshold).astype(np.uint8)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.enable_augment and random.random() < self.augment_prob:
            image, mask = self._augment_pair(image, mask)

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
            box = self._jitter_box(box)

        image_t = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        box_t = torch.from_numpy(box).float()
        points = []
        labels = []
        for _ in range(self.num_pos_points):
            points.append(self._sample_point(mask, positive=True))
            labels.append(1)
        for _ in range(self.num_neg_points):
            points.append(self._sample_point(mask, positive=False))
            labels.append(0)
        point_coords_t = torch.from_numpy(np.stack(points, axis=0)).float()
        point_labels_t = torch.tensor(labels, dtype=torch.long)
        return {
            "image": image_t,
            "mask": mask_t,
            "box": box_t,
            "point_coords": point_coords_t,
            "point_labels": point_labels_t,
            "id": key,
        }


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (probs * target).sum(dim=1)
    dice = (2.0 * inter + smooth) / (probs.sum(dim=1) + target.sum(dim=1) + smooth)
    return 1.0 - dice.mean()


def focal_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.clamp(min=eps, max=1.0 - eps)
    target = target.float()
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = -alpha_t * torch.pow(1.0 - p_t, gamma) * torch.log(p_t)
    return loss.mean()


def _morph_gradient(x: torch.Tensor, radius: int) -> torch.Tensor:
    r = int(max(0, radius))
    if r <= 0:
        return torch.zeros_like(x)
    k = 2 * r + 1
    maxp = F.max_pool2d(x, kernel_size=k, stride=1, padding=r)
    minp = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=r)
    return (maxp - minp).clamp(0.0, 1.0)


def boundary_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, radius: int = 1) -> torch.Tensor:
    probs = torch.sigmoid(logits).float()
    tgt = target.float().clamp(0.0, 1.0)
    pred_edge = _morph_gradient(probs, radius=radius)
    tgt_edge = _morph_gradient(tgt, radius=radius)
    return F.l1_loss(pred_edge, tgt_edge)


def _safe_torch_load(path: str, map_location: str | torch.device | None = None):
    # Prefer weights_only=True to avoid pickle execution and future warning noise.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _forward_lowres_logits_with_grad(
    sam: nn.Module,
    image_chw: torch.Tensor,
    box_xyxy: torch.Tensor | None,
    point_coords: torch.Tensor | None,
    point_labels: torch.Tensor | None,
) -> torch.Tensor:
    # image_chw: [3,H,W] in [0,255]
    input_image = sam.preprocess(image_chw)
    image_embeddings = sam.image_encoder(input_image.unsqueeze(0))

    points_in = None
    if point_coords is not None and point_labels is not None:
        points_in = (point_coords.unsqueeze(0), point_labels.unsqueeze(0))
    boxes_in = box_xyxy.unsqueeze(0) if box_xyxy is not None else None
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=points_in,
        boxes=boxes_in,
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


def _select_prompt_inputs(
    prompt_mode: str,
    box_xyxy: torch.Tensor,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    mix_both_prob: float,
    mix_box_only_prob: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, str]:
    mode = str(prompt_mode).strip().lower()
    if mode == "box_only":
        return box_xyxy, None, None, "box"
    if mode == "point_only":
        return None, point_coords, point_labels, "point"

    r = random.random()
    p_both = float(max(0.0, min(1.0, mix_both_prob)))
    p_box = float(max(0.0, min(1.0, mix_box_only_prob)))
    if r < p_both:
        return box_xyxy, point_coords, point_labels, "box+point"
    if r < p_both + p_box:
        return box_xyxy, None, None, "box"
    return None, point_coords, point_labels, "point"


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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--encoder-rank", type=int, default=-1)
    parser.add_argument("--encoder-alpha", type=int, default=-1)
    parser.add_argument("--decoder-rank", type=int, default=-1)
    parser.add_argument("--decoder-alpha", type=int, default=-1)
    parser.add_argument(
        "--lora-target-keywords",
        type=str,
        default="qkv,proj,q_proj,k_proj,v_proj,out_proj,to_q,to_k,to_v",
    )
    parser.add_argument("--decoder-lora-target-keywords", type=str, default="")
    parser.add_argument("--stage-epochs", type=str, default="10,10,10")
    parser.add_argument("--stage-lrs", type=str, default="")
    parser.add_argument("--unfreeze-encoder-tail-fraction", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="checkpoints/medsam_lora_best.pth")
    parser.add_argument("--train-mask-decoder", action="store_true")
    parser.add_argument("--enable-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment-prob", type=float, default=0.85)
    parser.add_argument("--box-jitter-scale", type=float, default=0.30)
    parser.add_argument("--box-jitter-shift", type=float, default=0.25)
    parser.add_argument("--box-full-image-prob", type=float, default=0.03)
    parser.add_argument("--num-pos-points", type=int, default=1)
    parser.add_argument("--num-neg-points", type=int, default=1)
    parser.add_argument("--point-jitter-frac", type=float, default=0.03)
    parser.add_argument("--prompt-mode", type=str, default="box_point_mix", choices=["box_only", "point_only", "box_point_mix"])
    parser.add_argument("--prompt-mix-both-prob", type=float, default=0.60)
    parser.add_argument("--prompt-mix-box-only-prob", type=float, default=0.20)
    parser.add_argument("--loss-dice", type=float, default=0.50)
    parser.add_argument("--loss-focal", type=float, default=0.30)
    parser.add_argument("--loss-boundary", type=float, default=0.20)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--boundary-radius", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(Path(args.save_path).parent, exist_ok=True)

    enc_rank = int(args.encoder_rank) if int(args.encoder_rank) > 0 else int(args.rank)
    enc_alpha = int(args.encoder_alpha) if int(args.encoder_alpha) > 0 else int(args.alpha)
    dec_rank = int(args.decoder_rank) if int(args.decoder_rank) > 0 else int(args.rank)
    dec_alpha = int(args.decoder_alpha) if int(args.decoder_alpha) > 0 else int(args.alpha)

    stage_epochs = _parse_stage_epochs(args.stage_epochs, total_epochs=int(args.epochs))
    total_epochs = int(sum(stage_epochs))
    if total_epochs != int(args.epochs):
        print(f"[stage epochs] override epochs={int(args.epochs)} -> {total_epochs} from stage split {stage_epochs}")
    args.epochs = int(total_epochs)
    stage_lrs = _parse_stage_lrs(args.stage_lrs, base_lr=float(args.lr))

    lora_keywords = [x.strip() for x in str(args.lora_target_keywords).split(",") if x.strip()]
    dec_keywords = (
        [x.strip() for x in str(args.decoder_lora_target_keywords).split(",") if x.strip()]
        if str(args.decoder_lora_target_keywords).strip()
        else lora_keywords
    )
    if not lora_keywords:
        raise ValueError("No valid --lora-target-keywords were provided.")

    loss_w_dice = float(args.loss_dice)
    loss_w_focal = float(args.loss_focal)
    loss_w_boundary = float(args.loss_boundary)
    if (loss_w_dice + loss_w_focal + loss_w_boundary) <= 0.0:
        loss_w_dice, loss_w_focal, loss_w_boundary = 1.0, 0.0, 0.0
        print("[loss weights] all non-positive, fallback to Dice-only.")

    sam = sam_model_registry[args.model_type](checkpoint=None).to(args.device)
    base_state = _safe_torch_load(args.checkpoint, map_location=args.device)
    if isinstance(base_state, dict) and "state_dict" in base_state and isinstance(base_state["state_dict"], dict):
        base_state = base_state["state_dict"]
    missing0, unexpected0 = sam.load_state_dict(base_state, strict=False)
    print(f"[base checkpoint] missing={len(missing0)} unexpected={len(unexpected0)} from {args.checkpoint}")
    for p in sam.parameters():
        p.requires_grad = False

    replaced_encoder = inject_lora(
        sam.image_encoder,
        target_keywords=lora_keywords,
        rank=enc_rank,
        alpha=enc_alpha,
        prefix="image_encoder",
    )
    replaced_decoder = inject_lora(
        sam.mask_decoder,
        target_keywords=dec_keywords,
        rank=dec_rank,
        alpha=dec_alpha,
        prefix="mask_decoder",
    )

    if args.init_lora_checkpoint:
        ckpt = _safe_torch_load(args.init_lora_checkpoint, map_location=args.device)
        state = ckpt.get("sam_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = sam.load_state_dict(state, strict=False)
        print(f"[resume] missing={len(missing)} unexpected={len(unexpected)} from {args.init_lora_checkpoint}")

    potential_trainable = []
    seen = set()
    for n, p in sam.named_parameters():
        ln = str(n).lower()
        is_lora = ".lora_a" in ln or ".lora_b" in ln
        if is_lora or (bool(args.train_mask_decoder) and ln.startswith("mask_decoder.")):
            if id(p) not in seen:
                potential_trainable.append(p)
                seen.add(id(p))
    if not potential_trainable:
        raise RuntimeError("No trainable parameters found after LoRA injection.")

    n_encoder_blocks = _collect_encoder_blocks(sam)
    if n_encoder_blocks > 0:
        tail_start = int((1.0 - float(max(0.0, min(1.0, args.unfreeze_encoder_tail_fraction)))) * n_encoder_blocks)
        tail_start = max(0, min(n_encoder_blocks - 1, tail_start))
    else:
        tail_start = 0
    enabled_stage1, total_adapt = _set_stage_trainable(
        sam=sam,
        stage_idx=1,
        encoder_tail_start_block=tail_start,
        train_mask_decoder_full=bool(args.train_mask_decoder),
    )
    print(
        f"[LoRA] encoder_layers={replaced_encoder}, decoder_layers={replaced_decoder}, "
        f"adapt_params_stage1={enabled_stage1}/{total_adapt}"
    )
    print(
        "[LoRA train setup] "
        f"augment={bool(args.enable_augment)} prob={float(args.augment_prob):.2f} "
        f"box_jitter_scale={float(args.box_jitter_scale):.2f} "
        f"box_jitter_shift={float(args.box_jitter_shift):.2f} "
        f"box_full_prob={float(args.box_full_image_prob):.2f} "
        f"prompt={str(args.prompt_mode)} stage_epochs={stage_epochs} stage_lrs={stage_lrs}"
    )

    if args.data_manifest:
        subsets = {s.strip() for s in args.subset_filter.split(",") if s.strip()} if args.subset_filter else None
        splits = {s.strip() for s in args.split_filter.split(",") if s.strip()} if args.split_filter else None
        pairs = _load_manifest_pairs(args.data_manifest, subset_filter=subsets, split_filter=splits)
        dataset = MedSamTuneDataset(
            image_size=args.image_size,
            pairs=pairs,
            mask_threshold=args.mask_threshold,
            enable_augment=bool(args.enable_augment),
            augment_prob=float(args.augment_prob),
            box_jitter_scale=float(args.box_jitter_scale),
            box_jitter_shift=float(args.box_jitter_shift),
            box_full_image_prob=float(args.box_full_image_prob),
            num_pos_points=int(args.num_pos_points),
            num_neg_points=int(args.num_neg_points),
            point_jitter_frac=float(args.point_jitter_frac),
        )
    else:
        dataset = MedSamTuneDataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            image_size=args.image_size,
            mask_threshold=args.mask_threshold,
            enable_augment=bool(args.enable_augment),
            augment_prob=float(args.augment_prob),
            box_jitter_scale=float(args.box_jitter_scale),
            box_jitter_shift=float(args.box_jitter_shift),
            box_full_image_prob=float(args.box_full_image_prob),
            num_pos_points=int(args.num_pos_points),
            num_neg_points=int(args.num_neg_points),
            point_jitter_frac=float(args.point_jitter_frac),
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(potential_trainable, lr=float(stage_lrs[0]), weight_decay=float(args.weight_decay))

    best_loss = float("inf")
    sam.train()
    last_stage = -1
    last_lr = -1.0
    for epoch in range(1, args.epochs + 1):
        stage_idx = _stage_for_epoch(epoch=epoch, stage_epochs=stage_epochs)
        stage_lr = float(stage_lrs[max(0, min(2, stage_idx - 1))])
        if stage_idx != last_stage:
            enabled_now, total_now = _set_stage_trainable(
                sam=sam,
                stage_idx=stage_idx,
                encoder_tail_start_block=tail_start,
                train_mask_decoder_full=bool(args.train_mask_decoder),
            )
            print(
                f"[stage] epoch={epoch}/{args.epochs} stage={stage_idx} "
                f"trainable_params={enabled_now}/{total_now} tail_start_block={tail_start}"
            )
            last_stage = stage_idx
        if abs(stage_lr - last_lr) > 1e-12:
            for g in optimizer.param_groups:
                g["lr"] = stage_lr
            last_lr = stage_lr
            print(f"[stage] set lr={stage_lr:.6g}")

        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_focal = 0.0
        epoch_boundary = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            # SAM preprocess expects image values in [0,255], so keep raw scale here.
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            boxes = batch["box"].to(args.device)
            point_coords_all = batch["point_coords"].to(args.device)
            point_labels_all = batch["point_labels"].to(args.device)
            batch_size = images.size(0)

            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            loss_d = 0.0
            loss_f = 0.0
            loss_b = 0.0

            for i in range(batch_size):
                box_in, p_coords_in, p_labels_in, _ = _select_prompt_inputs(
                    prompt_mode=str(args.prompt_mode),
                    box_xyxy=boxes[i],
                    point_coords=point_coords_all[i],
                    point_labels=point_labels_all[i],
                    mix_both_prob=float(args.prompt_mix_both_prob),
                    mix_box_only_prob=float(args.prompt_mix_box_only_prob),
                )
                logits = _forward_lowres_logits_with_grad(
                    sam=sam,
                    image_chw=images[i],
                    box_xyxy=box_in,
                    point_coords=p_coords_in,
                    point_labels=p_labels_in,
                )

                target_low = F.interpolate(masks[i].unsqueeze(0), size=logits.shape[-2:], mode="nearest")
                dice_l = dice_loss_from_logits(logits, target_low)
                focal_l = focal_loss_from_logits(
                    logits,
                    target_low,
                    alpha=float(args.focal_alpha),
                    gamma=float(args.focal_gamma),
                )
                boundary_l = boundary_loss_from_logits(
                    logits,
                    target_low,
                    radius=int(args.boundary_radius),
                )
                loss = (
                    float(loss_w_dice) * dice_l
                    + float(loss_w_focal) * focal_l
                    + float(loss_w_boundary) * boundary_l
                )
                loss_total = loss_total + loss
                loss_d = loss_d + float(dice_l.detach().item())
                loss_f = loss_f + float(focal_l.detach().item())
                loss_b = loss_b + float(boundary_l.detach().item())

            loss_total = loss_total / batch_size
            loss_total.backward()
            optimizer.step()

            epoch_loss += loss_total.item() * batch_size
            epoch_dice += loss_d
            epoch_focal += loss_f
            epoch_boundary += loss_b
            pbar.set_postfix(loss=f"{loss_total.item():.4f}", stage=str(stage_idx))

        epoch_loss /= len(dataset)
        denom = max(1, len(dataset))
        print(
            f"epoch {epoch}/{args.epochs} | stage={stage_idx} lr={stage_lr:.6g} "
            f"loss={epoch_loss:.4f} dice={epoch_dice/denom:.4f} "
            f"focal={epoch_focal/denom:.4f} boundary={epoch_boundary/denom:.4f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "sam_state_dict": sam.state_dict(),
                    "best_loss": best_loss,
                    "epoch": epoch,
                    "lora_rank": enc_rank,
                    "lora_alpha": enc_alpha,
                    "encoder_lora_rank": enc_rank,
                    "encoder_lora_alpha": enc_alpha,
                    "decoder_lora_rank": dec_rank,
                    "decoder_lora_alpha": dec_alpha,
                    "stage_epochs": stage_epochs,
                    "stage_lrs": stage_lrs,
                    "prompt_mode": str(args.prompt_mode),
                    "loss_weights": {
                        "dice": float(loss_w_dice),
                        "focal": float(loss_w_focal),
                        "boundary": float(loss_w_boundary),
                    },
                },
                args.save_path,
            )
            print(f"[checkpoint] saved -> {args.save_path}")


if __name__ == "__main__":
    main()
