import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import build_baseline_model
from utils.augmentations import TrainAugmentor, ValAugmentor
from utils.data_protocol import load_protocol_samples, summarize_samples, validate_protocol_samples
from utils.dataset import ProtocolSegDataset
from utils.metrics import dice_per_sample, iou_per_sample


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_yaml(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _build_parser():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args()
    cfg = _load_yaml(pre_args.config)

    p = argparse.ArgumentParser(description="Train local supervised segmentation baselines.")
    p.add_argument("--config", type=str, default=pre_args.config)
    p.add_argument("--model", type=str, default=cfg.get("model", "unet"), choices=["unet", "unetpp", "pranet", "transunet", "sanet"])
    p.add_argument("--base-channels", type=int, default=cfg.get("base_channels", 32))
    p.add_argument("--norm-type", type=str, default=cfg.get("norm_type", "gn"), choices=["bn", "gn"])

    p.add_argument("--data-manifest", type=str, default=cfg.get("data_manifest", "runs/flywheel_full_manual_bridge_1500/student_manifest_final_sdf_smooth.csv"))
    p.add_argument("--data-root", type=str, default=cfg.get("data_root", "data/joint_polyp_v1"))
    p.add_argument("--manifest-mode", type=str, default=cfg.get("manifest_mode", "only"), choices=["prefer", "only", "off"])
    p.add_argument("--mask-threshold", type=int, default=cfg.get("mask_threshold", 127))

    p.add_argument("--epochs", type=int, default=cfg.get("epochs", 100))
    p.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 8))
    p.add_argument("--lr", type=float, default=cfg.get("lr", 1e-4))
    p.add_argument("--weight-decay", type=float, default=cfg.get("weight_decay", 1e-4))
    p.add_argument("--warmup-epochs", type=int, default=cfg.get("warmup_epochs", 5))
    p.add_argument("--grad-clip-norm", type=float, default=cfg.get("grad_clip_norm", 1.0))
    p.add_argument("--img-size", type=int, default=cfg.get("img_size", 256))
    p.add_argument("--num-workers", type=int, default=cfg.get("num_workers", 2))
    p.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    p.add_argument("--device", type=str, default=cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg.get("amp", True))
    p.add_argument("--save-path", type=str, default=cfg.get("save_path", "checkpoints/baselines/unet_final.pth"))
    p.add_argument("--init-checkpoint", type=str, default=cfg.get("init_checkpoint", ""))
    p.add_argument("--run-name", type=str, default=cfg.get("run_name", "baseline_unet"))

    p.add_argument("--seg-focal-weight", type=float, default=cfg.get("seg_focal_weight", 0.7))
    p.add_argument("--seg-dice-weight", type=float, default=cfg.get("seg_dice_weight", 0.3))
    p.add_argument("--focal-alpha", type=float, default=cfg.get("focal_alpha", 0.25))
    p.add_argument("--focal-gamma", type=float, default=cfg.get("focal_gamma", 2.0))
    p.add_argument("--use-pseudo-weight", action=argparse.BooleanOptionalAction, default=cfg.get("use_pseudo_weight", True))
    p.add_argument("--pseudo-weight-min", type=float, default=cfg.get("pseudo_weight_min", 0.05))
    return p


def _build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    warmup = max(0, int(warmup_epochs))

    def lr_lambda(e: int):
        if warmup > 0 and e < warmup:
            return float(e + 1) / float(max(1, warmup))
        if epochs <= warmup:
            return 1.0
        p = float(e - warmup + 1) / float(max(1, epochs - warmup))
        p = min(max(p, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _select_rows(samples):
    train_rows = [
        s for s in samples
        if s.exclude_from_tuning == 0
        and s.mask_path
        and s.split != "val"
        and (s.is_labeled == 1 or s.is_pseudo == 1)
    ]
    val_rows = [
        s for s in samples
        if s.exclude_from_tuning == 0
        and s.mask_path
        and s.split == "val"
        and s.is_labeled == 1
        and s.is_pseudo == 0
    ]
    if not train_rows or not val_rows:
        raise RuntimeError(f"Need non-empty train/val rows, got train={len(train_rows)} val={len(val_rows)}")
    return train_rows, val_rows


def _sample_weights(batch, args, device: str) -> torch.Tensor:
    is_pseudo = batch["is_pseudo"].to(device) > 0.5
    pseudo_weight = batch["pseudo_weight"].to(device)
    weights = torch.ones_like(pseudo_weight)
    if bool(args.use_pseudo_weight):
        weights = torch.where(
            is_pseudo,
            torch.clamp(pseudo_weight, min=float(args.pseudo_weight_min)),
            weights,
        )
    return weights


def _weighted_focal_dice_loss(logits, targets, sample_weights, args):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(targets > 0.5, float(args.focal_alpha), 1.0 - float(args.focal_alpha))
    focal = alpha_t * (1.0 - pt).pow(float(args.focal_gamma)) * bce
    focal_ps = focal.view(focal.shape[0], -1).mean(dim=1)

    p = probs.view(probs.shape[0], -1)
    t = targets.view(targets.shape[0], -1)
    inter = (p * t).sum(dim=1)
    dice_ps = 1.0 - (2.0 * inter + 1.0) / (p.sum(dim=1) + t.sum(dim=1) + 1.0)

    seg_ps = float(args.seg_focal_weight) * focal_ps + float(args.seg_dice_weight) * dice_ps
    weights = torch.clamp(sample_weights, min=1e-6)
    return (seg_ps * weights).sum() / weights.sum(), focal_ps.mean().detach(), dice_ps.mean().detach()


def _as_output_list(outputs):
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    return [outputs]


def _final_output(outputs):
    return _as_output_list(outputs)[-1]


def _multi_output_loss(outputs, masks, weights, args):
    losses = []
    focal_losses = []
    dice_losses = []
    for logits in _as_output_list(outputs):
        loss, focal_loss, dice_loss = _weighted_focal_dice_loss(logits, masks, weights, args)
        losses.append(loss)
        focal_losses.append(focal_loss)
        dice_losses.append(dice_loss)
    return torch.stack(losses).mean(), torch.stack(focal_losses).mean(), torch.stack(dice_losses).mean()


def _run_epoch(model, loader, optimizer, scaler, args, train: bool):
    model.train(train)
    total = 0.0
    focal_total = 0.0
    dice_loss_total = 0.0
    dice_total = 0.0
    iou_total = 0.0
    n = 0
    amp_enabled = bool(args.amp and str(args.device).startswith("cuda"))

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(args.device)
        masks = batch["mask"].to(args.device)
        weights = _sample_weights(batch, args, args.device)
        bs = int(images.size(0))

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(images)
                logits = _final_output(outputs)
                loss, focal_loss, dice_loss = _multi_output_loss(outputs, masks, weights, args)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if args.grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                    optimizer.step()

        total += float(loss.detach().item()) * bs
        focal_total += float(focal_loss.item()) * bs
        dice_loss_total += float(dice_loss.item()) * bs
        dice_total += float(dice_per_sample(logits.detach(), masks, threshold=0.5).sum().item())
        iou_total += float(iou_per_sample(logits.detach(), masks, threshold=0.5).sum().item())
        n += bs

    n = max(1, n)
    return {
        "loss": total / n,
        "focal_loss": focal_total / n,
        "dice_loss": dice_loss_total / n,
        "dice": dice_total / n,
        "iou": iou_total / n,
    }


def main():
    args = _build_parser().parse_args()
    set_seed(int(args.seed))

    samples = load_protocol_samples(args.data_manifest, args.data_root, args.manifest_mode)
    validate_protocol_samples(samples)
    train_rows, val_rows = _select_rows(samples)
    print(f"[protocol] {json.dumps(summarize_samples(samples), ensure_ascii=False)}")
    print(f"[split] train={len(train_rows)} val={len(val_rows)}")

    train_ds = ProtocolSegDataset(
        train_rows,
        transform=TrainAugmentor((args.img_size, args.img_size), mask_morph_prob=0.0),
        mask_threshold=args.mask_threshold,
    )
    val_ds = ProtocolSegDataset(
        val_rows,
        transform=ValAugmentor((args.img_size, args.img_size)),
        mask_threshold=args.mask_threshold,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model_kwargs = {
        "in_channels": 3,
        "num_classes": 1,
        "base_channels": int(args.base_channels),
        "norm_type": args.norm_type,
    }
    if str(args.model).lower() == "transunet":
        model_kwargs["img_size"] = int(args.img_size)
    model = build_baseline_model(args.model, **model_kwargs).to(args.device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=args.device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"[init_checkpoint] {args.init_checkpoint} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and str(args.device).startswith("cuda")))

    os.makedirs(Path(args.save_path).parent, exist_ok=True)
    writer_dir = Path("runs") / str(args.run_name)
    writer_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(writer_dir))

    best = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        train_stats = _run_epoch(model, train_loader, optimizer, scaler, args, train=True)
        val_stats = _run_epoch(model, val_loader, optimizer, scaler, args, train=False)
        scheduler.step()

        writer.add_scalar("Loss/train", train_stats["loss"], epoch)
        writer.add_scalar("Loss/val", val_stats["loss"], epoch)
        writer.add_scalar("Metric/train_dice", train_stats["dice"], epoch)
        writer.add_scalar("Metric/val_dice", val_stats["dice"], epoch)
        writer.add_scalar("Metric/val_iou", val_stats["iou"], epoch)

        if val_stats["dice"] > best:
            best = float(val_stats["dice"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "model_kwargs": model_kwargs,
                    "best_val_dice": best,
                    "epoch": epoch,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"[checkpoint] epoch={epoch} best_val_dice={best:.4f} -> {args.save_path}")

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_stats['loss']:.4f} train_dice={train_stats['dice']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_dice={val_stats['dice']:.4f} val_iou={val_stats['iou']:.4f}"
        )

    writer.close()


if __name__ == "__main__":
    main()
