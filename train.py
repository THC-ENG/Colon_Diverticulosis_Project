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
from utils.losses import DualTaskLoss, FocalWithDiceLoss
from utils.metrics import dice_coeff, iou_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_dataset_dirs(args):
    if args.dataset_root:
        root = Path(args.dataset_root)
        train_image_dir = root / "images" / "train"
        train_mask_dir = root / "masks" / "train"
        val_image_dir = root / "images" / "val"
        val_mask_dir = root / "masks" / "val"
    else:
        train_image_dir = Path(args.train_image_dir)
        train_mask_dir = Path(args.train_mask_dir)
        val_image_dir = Path(args.val_image_dir)
        val_mask_dir = Path(args.val_mask_dir)

    return str(train_image_dir), str(train_mask_dir), str(val_image_dir), str(val_mask_dir)


def _resolve_save_path(args) -> str:
    default_path = "checkpoints/best_model.pth"
    if args.save_path != default_path:
        return args.save_path
    if args.use_boundary:
        return "checkpoints/best_model_boundary.pth"
    return "checkpoints/best_model_baseline.pth"


def _resolve_writer_dir(args) -> str:
    if args.run_name:
        return str(Path("runs") / args.run_name)
    return str(Path("runs") / ("boundary" if args.use_boundary else "baseline"))


def run_epoch(model, loader, criterion, optimizer, device, train: bool, use_boundary: bool):
    model.train(train)
    tracked = {
        "total": 0.0,
        "seg_focal": 0.0,
        "seg_dice": 0.0,
        "dice": 0.0,
        "iou": 0.0,
    }
    if use_boundary:
        tracked.update(
            {
                "boundary_bce": 0.0,
                "boundary_dice": 0.0,
                "boundary_total": 0.0,
            }
        )

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)

            if use_boundary:
                seg_logits, boundary_logits = outputs
                boundaries = batch["boundary"].to(device)
                total_loss, stats = criterion(seg_logits, boundary_logits, masks, boundaries)
            else:
                seg_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                total_loss, seg_focal, seg_dice = criterion(seg_logits, masks)
                stats = {
                    "seg_focal": seg_focal,
                    "seg_dice": seg_dice,
                }

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
        tracked["dice"] += d.item() * bs
        tracked["iou"] += i.item() * bs

        if use_boundary:
            tracked["boundary_bce"] += stats["boundary_bce"].item() * bs
            tracked["boundary_dice"] += stats["boundary_dice"].item() * bs
            tracked["boundary_total"] += stats["boundary_total"].item() * bs

        desc = (
            f"{'train' if train else 'val'} "
            f"loss={total_loss.item():.4f} "
            f"dice={d.item():.4f}"
        )
        if use_boundary:
            desc += f" bnd={stats['boundary_total'].item():.4f}"
        pbar.set_description(desc)

    n = len(loader.dataset)
    return {k: v / n for k, v in tracked.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--run-name", type=str, default="")

    parser.add_argument("--use-boundary", action="store_true")
    parser.add_argument("--seg-focal-weight", type=float, default=0.7)
    parser.add_argument("--seg-dice-weight", type=float, default=0.3)
    parser.add_argument("--boundary-weight", type=float, default=0.3)
    parser.add_argument("--boundary-bce-weight", type=float, default=1.0)
    parser.add_argument("--boundary-dice-weight", type=float, default=0.0)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    args = parser.parse_args()

    train_image_dir, train_mask_dir, val_image_dir, val_mask_dir = _resolve_dataset_dirs(args)
    save_path = _resolve_save_path(args)
    writer_dir = _resolve_writer_dir(args)

    set_seed(args.seed)
    os.makedirs(Path(save_path).parent, exist_ok=True)
    os.makedirs(writer_dir, exist_ok=True)

    train_dataset = ColonDataset(
        train_image_dir,
        train_mask_dir,
        transform=TrainAugmentor((args.img_size, args.img_size)),
        use_boundary=args.use_boundary,
        boundary_radius=args.boundary_radius,
    )
    val_dataset = ColonDataset(
        val_image_dir,
        val_mask_dir,
        transform=ValAugmentor((args.img_size, args.img_size)),
        use_boundary=args.use_boundary,
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

    model = ResSwinUNet(num_classes=1, use_boundary=args.use_boundary).to(args.device)

    if args.use_boundary:
        criterion = DualTaskLoss(
            seg_focal_weight=args.seg_focal_weight,
            seg_dice_weight=args.seg_dice_weight,
            boundary_weight=args.boundary_weight,
            boundary_bce_weight=args.boundary_bce_weight,
            boundary_dice_weight=args.boundary_dice_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
        )
    else:
        criterion = FocalWithDiceLoss(
            focal_weight=args.seg_focal_weight,
            dice_weight=args.seg_dice_weight,
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(writer_dir)
    best_dice = -1.0

    mode_name = "dual_task_boundary" if args.use_boundary else "baseline"
    print(f"[mode] {mode_name}")
    print(f"[train images] {train_image_dir}")
    print(f"[train masks]  {train_mask_dir}")
    print(f"[val images]   {val_image_dir}")
    print(f"[val masks]    {val_mask_dir}")
    print(f"[log dir]      {writer_dir}")
    print(f"[save path]    {save_path}")

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, args.device, train=True, use_boundary=args.use_boundary)
        va = run_epoch(model, val_loader, criterion, optimizer, args.device, train=False, use_boundary=args.use_boundary)

        scheduler.step()

        writer.add_scalar("Loss/train_total", tr["total"], epoch)
        writer.add_scalar("Loss/train_focal", tr["seg_focal"], epoch)
        writer.add_scalar("Loss/train_dice", tr["seg_dice"], epoch)

        writer.add_scalar("Loss/val_total", va["total"], epoch)
        writer.add_scalar("Loss/val_focal", va["seg_focal"], epoch)
        writer.add_scalar("Loss/val_dice", va["seg_dice"], epoch)

        if args.use_boundary:
            writer.add_scalar("Loss/train_boundary_bce", tr["boundary_bce"], epoch)
            writer.add_scalar("Loss/train_boundary_dice", tr["boundary_dice"], epoch)
            writer.add_scalar("Loss/train_boundary_total", tr["boundary_total"], epoch)

            writer.add_scalar("Loss/val_boundary_bce", va["boundary_bce"], epoch)
            writer.add_scalar("Loss/val_boundary_dice", va["boundary_dice"], epoch)
            writer.add_scalar("Loss/val_boundary_total", va["boundary_total"], epoch)

        writer.add_scalar("Metric/train_dice", tr["dice"], epoch)
        writer.add_scalar("Metric/train_iou", tr["iou"], epoch)
        writer.add_scalar("Metric/val_dice", va["dice"], epoch)
        writer.add_scalar("Metric/val_iou", va["iou"], epoch)

        if args.use_boundary:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss={tr['total']:.4f} dice={tr['dice']:.4f} bnd={tr['boundary_total']:.4f} | "
                f"val loss={va['total']:.4f} dice={va['dice']:.4f} bnd={va['boundary_total']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss={tr['total']:.4f} dice={tr['dice']:.4f} | "
                f"val loss={va['total']:.4f} dice={va['dice']:.4f}"
            )

        if va["dice"] > best_dice:
            best_dice = va["dice"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_val_dice": best_dice,
                    "epoch": epoch,
                    "use_boundary": args.use_boundary,
                },
                save_path,
            )
            print(f"[checkpoint] saved -> {save_path} (best_dice={best_dice:.4f})")

    writer.close()


if __name__ == "__main__":
    main()
