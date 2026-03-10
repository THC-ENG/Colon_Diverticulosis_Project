import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import ResSwinUNet
from utils.augmentations import TrainAugmentor, ValAugmentor
from utils.dataset import ColonDataset
from utils.losses import DualTaskLoss
from utils.metrics import dice_coeff, iou_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    tracked = {
        "total": 0.0,
        "seg_focal": 0.0,
        "seg_dice": 0.0,
        "seg_total": 0.0,
        "boundary_bce": 0.0,
        "boundary_dice": 0.0,
        "boundary_total": 0.0,
        "dice": 0.0,
        "iou": 0.0,
    }

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        boundaries = batch["boundary"].to(device)

        with torch.set_grad_enabled(train):
            seg_logits, boundary_logits = model(images)
            total_loss, stats = criterion(seg_logits, boundary_logits, masks, boundaries)
            if train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

        d = dice_coeff(seg_logits.detach(), masks)
        i = iou_score(seg_logits.detach(), masks)

        bs = images.size(0)
        tracked["total"] += total_loss.item() * bs
        tracked["seg_focal"] += stats["seg_focal"].item() * bs
        tracked["seg_dice"] += stats["seg_dice"].item() * bs
        tracked["seg_total"] += stats["seg_total"].item() * bs
        tracked["boundary_bce"] += stats["boundary_bce"].item() * bs
        tracked["boundary_dice"] += stats["boundary_dice"].item() * bs
        tracked["boundary_total"] += stats["boundary_total"].item() * bs
        tracked["dice"] += d.item() * bs
        tracked["iou"] += i.item() * bs

        pbar.set_description(
            f"{'train' if train else 'val'} "
            f"loss={total_loss.item():.4f} "
            f"dice={d.item():.4f} "
            f"bnd={stats['boundary_total'].item():.4f}"
        )

    n = len(loader.dataset)
    return {k: v / n for k, v in tracked.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--mask-dir", type=str, default="")
    parser.add_argument("--train-image-dir", type=str, default="data/processed_images/images/train")
    parser.add_argument("--train-mask-dir", type=str, default="data/processed_images/masks/train")
    parser.add_argument("--val-image-dir", type=str, default="data/processed_images/images/val")
    parser.add_argument("--val-mask-dir", type=str, default="data/processed_images/masks/val")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="checkpoints/best_model.pth")

    parser.add_argument("--seg-focal-weight", type=float, default=0.7)
    parser.add_argument("--seg-dice-weight", type=float, default=0.3)
    parser.add_argument("--boundary-weight", type=float, default=0.3)
    parser.add_argument("--boundary-bce-weight", type=float, default=0.7)
    parser.add_argument("--boundary-dice-weight", type=float, default=0.3)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(Path(args.save_path).parent, exist_ok=True)

    train_dataset = ColonDataset(
        args.train_image_dir,
        args.train_mask_dir,
        transform=TrainAugmentor((args.img_size, args.img_size)),
        boundary_radius=args.boundary_radius,
    )

    val_dataset = ColonDataset(
        args.val_image_dir,
        args.val_mask_dir,
        transform=ValAugmentor((args.img_size, args.img_size)),
        boundary_radius=args.boundary_radius,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ResSwinUNet(num_classes=1).to(args.device)
    criterion = DualTaskLoss(
        seg_focal_weight=args.seg_focal_weight,
        seg_dice_weight=args.seg_dice_weight,
        boundary_weight=args.boundary_weight,
        boundary_bce_weight=args.boundary_bce_weight,
        boundary_dice_weight=args.boundary_dice_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter("runs/exp")

    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, args.device, train=True)
        va = run_epoch(model, val_loader, criterion, optimizer, args.device, train=False)

        scheduler.step()

        writer.add_scalar("Loss/train_total", tr["total"], epoch)
        writer.add_scalar("Loss/train_focal", tr["seg_focal"], epoch)
        writer.add_scalar("Loss/train_dice", tr["seg_dice"], epoch)
        writer.add_scalar("Loss/train_seg_total", tr["seg_total"], epoch)
        writer.add_scalar("Loss/train_boundary_bce", tr["boundary_bce"], epoch)
        writer.add_scalar("Loss/train_boundary_dice", tr["boundary_dice"], epoch)
        writer.add_scalar("Loss/train_boundary_total", tr["boundary_total"], epoch)

        writer.add_scalar("Loss/val_total", va["total"], epoch)
        writer.add_scalar("Loss/val_focal", va["seg_focal"], epoch)
        writer.add_scalar("Loss/val_dice", va["seg_dice"], epoch)
        writer.add_scalar("Loss/val_seg_total", va["seg_total"], epoch)
        writer.add_scalar("Loss/val_boundary_bce", va["boundary_bce"], epoch)
        writer.add_scalar("Loss/val_boundary_dice", va["boundary_dice"], epoch)
        writer.add_scalar("Loss/val_boundary_total", va["boundary_total"], epoch)

        writer.add_scalar("Metric/train_dice", tr["dice"], epoch)
        writer.add_scalar("Metric/train_iou", tr["iou"], epoch)
        writer.add_scalar("Metric/val_dice", va["dice"], epoch)
        writer.add_scalar("Metric/val_iou", va["iou"], epoch)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={tr['total']:.4f} dice={tr['dice']:.4f} bnd={tr['boundary_total']:.4f} | "
            f"val loss={va['total']:.4f} dice={va['dice']:.4f} bnd={va['boundary_total']:.4f}"
        )

        if va["dice"] > best_dice:
            best_dice = va["dice"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_val_dice": best_dice,
                    "epoch": epoch,
                },
                args.save_path,
            )
            print(f"[checkpoint] saved -> {args.save_path} (best_dice={best_dice:.4f})")

    writer.close()


if __name__ == "__main__":
    main()
