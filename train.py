import argparse
import json
import math
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
from utils.metrics import dice_per_sample, iou_per_sample


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_yaml_config(config_path: str) -> dict:
    path = Path(config_path)
    if not config_path or not path.exists():
        return {}

    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "PyYAML is required for --config support. Install with: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must map keys to values: {config_path}")
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="configs/train_res_swin_unet.yaml")
    pre_args, _ = pre_parser.parse_known_args()
    cfg = _load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=pre_args.config)

    parser.add_argument("--dataset-root", type=str, default=cfg.get("dataset_root", ""))
    parser.add_argument("--image-dir", type=str, default=cfg.get("image_dir", ""))
    parser.add_argument("--mask-dir", type=str, default=cfg.get("mask_dir", ""))
    parser.add_argument("--train-image-dir", type=str, default=cfg.get("train_image_dir", "data/processed_images/images/train"))
    parser.add_argument("--train-mask-dir", type=str, default=cfg.get("train_mask_dir", "data/processed_images/masks/train"))
    parser.add_argument("--val-image-dir", type=str, default=cfg.get("val_image_dir", "data/processed_images/images/val"))
    parser.add_argument("--val-mask-dir", type=str, default=cfg.get("val_mask_dir", "data/processed_images/masks/val"))
    parser.add_argument("--all-image-dir", type=str, default=cfg.get("all_image_dir", ""))
    parser.add_argument("--all-mask-dir", type=str, default=cfg.get("all_mask_dir", ""))
    parser.add_argument("--split-json", type=str, default=cfg.get("split_json", ""))
    parser.add_argument("--fold-index", type=int, default=cfg.get("fold_index", 0))

    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 100))
    parser.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 4))
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 1e-4))
    parser.add_argument("--encoder-lr-mult", type=float, default=cfg.get("encoder_lr_mult", 0.1))
    parser.add_argument("--num-workers", type=int, default=cfg.get("num_workers", 2))
    parser.add_argument("--img-size", type=int, default=cfg.get("img_size", 256))
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    parser.add_argument("--device", type=str, default=cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--save-path", type=str, default=cfg.get("save_path", "checkpoints/best_model.pth"))
    parser.add_argument("--init-checkpoint", type=str, default=cfg.get("init_checkpoint", ""))
    parser.add_argument("--run-name", type=str, default=cfg.get("run_name", ""))

    parser.add_argument("--weight-decay", type=float, default=cfg.get("weight_decay", 1e-4))
    parser.add_argument("--warmup-epochs", type=int, default=cfg.get("warmup_epochs", 5))
    parser.add_argument("--grad-clip-norm", type=float, default=cfg.get("grad_clip_norm", 1.0))

    parser.add_argument("--use-boundary", action=argparse.BooleanOptionalAction, default=cfg.get("use_boundary", False))
    parser.add_argument("--boundary-warmup-ratio", type=float, default=cfg.get("boundary_warmup_ratio", 0.3))

    parser.add_argument("--seg-focal-weight", type=float, default=cfg.get("seg_focal_weight", 0.7))
    parser.add_argument("--seg-dice-weight", type=float, default=cfg.get("seg_dice_weight", 0.3))
    parser.add_argument("--boundary-weight", type=float, default=cfg.get("boundary_weight", 0.3))
    parser.add_argument("--boundary-bce-weight", type=float, default=cfg.get("boundary_bce_weight", 1.0))
    parser.add_argument("--boundary-dice-weight", type=float, default=cfg.get("boundary_dice_weight", 0.0))
    parser.add_argument("--boundary-radius", type=int, default=cfg.get("boundary_radius", 1))
    parser.add_argument("--focal-alpha", type=float, default=cfg.get("focal_alpha", 0.25))
    parser.add_argument("--focal-gamma", type=float, default=cfg.get("focal_gamma", 2.0))

    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg.get("amp", True))
    parser.add_argument("--norm-type", type=str, default=cfg.get("norm_type", "bn"), choices=["bn", "gn"])
    parser.add_argument("--deep-supervision", action=argparse.BooleanOptionalAction, default=cfg.get("deep_supervision", False))
    parser.add_argument("--ds-weight-d2", type=float, default=cfg.get("ds_weight_d2", 0.4))
    parser.add_argument("--ds-weight-d3", type=float, default=cfg.get("ds_weight_d3", 0.2))
    parser.add_argument("--ds-weight-d4", type=float, default=cfg.get("ds_weight_d4", 0.1))

    parser.add_argument("--window-size", type=int, default=cfg.get("window_size", 8))
    parser.add_argument("--use-shift-mask", action=argparse.BooleanOptionalAction, default=cfg.get("use_shift_mask", True))
    parser.add_argument("--use-rel-pos-bias", action=argparse.BooleanOptionalAction, default=cfg.get("use_rel_pos_bias", True))
    parser.add_argument("--pad-to-window", action=argparse.BooleanOptionalAction, default=cfg.get("pad_to_window", True))

    return parser


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


def _resolve_fold_split(args):
    if not args.split_json:
        return None, None, None, None

    split_path = Path(args.split_json)
    if not split_path.exists():
        raise FileNotFoundError(f"split_json file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    folds = payload.get("folds", [])
    if not folds:
        raise ValueError(f"split_json has no folds: {split_path}")
    if args.fold_index < 0 or args.fold_index >= len(folds):
        raise IndexError(f"fold_index out of range: {args.fold_index}, total folds: {len(folds)}")

    fold = folds[args.fold_index]
    train_ids = set(fold.get("train_ids", []))
    val_ids = set(fold.get("val_ids", []))
    if not train_ids or not val_ids:
        raise ValueError(f"Invalid fold content in {split_path}, fold={args.fold_index}")

    if args.all_image_dir and args.all_mask_dir:
        image_dir = args.all_image_dir
        mask_dir = args.all_mask_dir
    elif args.dataset_root:
        image_dir = str(Path(args.dataset_root) / "images" / "all")
        mask_dir = str(Path(args.dataset_root) / "masks" / "all")
    else:
        image_dir = args.train_image_dir
        mask_dir = args.train_mask_dir

    return image_dir, mask_dir, train_ids, val_ids


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


def _parse_model_outputs(outputs):
    parsed = {
        "seg": None,
        "boundary": None,
        "aux": [],
    }

    if isinstance(outputs, dict):
        parsed["seg"] = outputs.get("seg")
        parsed["boundary"] = outputs.get("boundary")
        parsed["aux"] = outputs.get("aux") or []
        return parsed

    if isinstance(outputs, (tuple, list)):
        if len(outputs) >= 1:
            parsed["seg"] = outputs[0]
        if len(outputs) >= 2:
            parsed["boundary"] = outputs[1]
        if len(outputs) >= 3:
            parsed["aux"] = outputs[2] if outputs[2] is not None else []
        return parsed

    parsed["seg"] = outputs
    return parsed


def _boundary_weight_for_epoch(
    epoch: int,
    total_epochs: int,
    target_weight: float,
    warmup_ratio: float,
) -> float:
    if warmup_ratio <= 0.0 or target_weight <= 0.0:
        return target_weight

    warmup_epochs = max(1, int(total_epochs * warmup_ratio))
    if epoch >= warmup_epochs:
        return target_weight
    return target_weight * (float(epoch + 1) / float(warmup_epochs))


def _build_optimizer(model, args):
    encoder_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": args.lr * args.encoder_lr_mult})

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def _build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    warmup_epochs = max(0, int(warmup_epochs))

    def lr_lambda(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, warmup_epochs))

        if epochs <= warmup_epochs:
            return 1.0

        progress = float(epoch_idx - warmup_epochs + 1) / float(max(1, epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_epoch(
    model,
    loader,
    criterion,
    aux_criterion,
    optimizer,
    scaler,
    device,
    train: bool,
    use_boundary: bool,
    deep_supervision: bool,
    ds_weights,
    grad_clip_norm: float,
    epoch_idx: int,
    total_epochs: int,
    boundary_warmup_ratio: float,
):
    model.train(train)

    tracked = {
        "total": 0.0,
        "seg_focal": 0.0,
        "seg_dice": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "aux_total": 0.0,
        "boundary_weight": 0.0,
    }
    if use_boundary:
        tracked.update(
            {
                "boundary_bce": 0.0,
                "boundary_dice": 0.0,
                "boundary_total": 0.0,
            }
        )

    sample_count = 0
    amp_enabled = scaler.is_enabled() if scaler is not None else False

    boundary_weight_override = None
    if use_boundary:
        target_boundary_weight = float(getattr(criterion, "boundary_weight", 0.0))
        if train:
            boundary_weight_override = _boundary_weight_for_epoch(
                epoch=epoch_idx,
                total_epochs=total_epochs,
                target_weight=target_boundary_weight,
                warmup_ratio=boundary_warmup_ratio,
            )
        else:
            boundary_weight_override = target_boundary_weight

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        bs = images.size(0)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(images)
                parsed = _parse_model_outputs(outputs)
                seg_logits = parsed["seg"]
                boundary_logits = parsed["boundary"]
                aux_logits = parsed["aux"]

                if seg_logits is None:
                    raise RuntimeError("Model output does not contain segmentation logits.")

                if use_boundary:
                    if boundary_logits is None:
                        raise RuntimeError("Boundary mode is enabled but model did not return boundary logits.")
                    boundaries = batch["boundary"].to(device)
                    total_loss, stats = criterion(
                        seg_logits,
                        boundary_logits,
                        masks,
                        boundaries,
                        boundary_weight_override=boundary_weight_override,
                    )
                else:
                    total_loss, seg_focal, seg_dice = criterion(seg_logits, masks)
                    stats = {
                        "seg_focal": seg_focal,
                        "seg_dice": seg_dice,
                    }

                aux_total = torch.zeros((), device=images.device)
                if deep_supervision and aux_criterion is not None and aux_logits:
                    for weight, aux_logit in zip(ds_weights, aux_logits):
                        if weight <= 0:
                            continue
                        aux_loss, _, _ = aux_criterion(aux_logit, masks)
                        aux_total = aux_total + float(weight) * aux_loss

                total_loss = total_loss + aux_total

            if train:
                optimizer.zero_grad(set_to_none=True)

                if amp_enabled:
                    scaler.scale(total_loss).backward()
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()

        d = dice_per_sample(seg_logits.detach(), masks, threshold=0.5)
        i = iou_per_sample(seg_logits.detach(), masks, threshold=0.5)

        tracked["total"] += total_loss.detach().item() * bs
        tracked["seg_focal"] += stats["seg_focal"].item() * bs
        tracked["seg_dice"] += stats["seg_dice"].item() * bs
        tracked["dice"] += d.sum().item()
        tracked["iou"] += i.sum().item()
        tracked["aux_total"] += aux_total.detach().item() * bs

        if use_boundary:
            tracked["boundary_bce"] += stats["boundary_bce"].item() * bs
            tracked["boundary_dice"] += stats["boundary_dice"].item() * bs
            tracked["boundary_total"] += stats["boundary_total"].item() * bs
            tracked["boundary_weight"] += stats["boundary_weight"].item() * bs

        sample_count += bs

        desc = (
            f"{'train' if train else 'val'} "
            f"loss={total_loss.item():.4f} "
            f"dice={d.mean().item():.4f}"
        )
        if use_boundary:
            desc += f" bnd={stats['boundary_total'].item():.4f}"
        pbar.set_description(desc)

    n = max(1, sample_count)
    out = {k: v / n for k, v in tracked.items()}
    if not use_boundary:
        out["boundary_weight"] = 0.0
    return out


def main():
    parser = _build_parser()
    args = parser.parse_args()

    train_image_dir, train_mask_dir, val_image_dir, val_mask_dir = _resolve_dataset_dirs(args)
    fold_image_dir, fold_mask_dir, fold_train_ids, fold_val_ids = _resolve_fold_split(args)

    use_fold_split = fold_image_dir is not None
    if use_fold_split:
        train_image_dir = fold_image_dir
        train_mask_dir = fold_mask_dir
        val_image_dir = fold_image_dir
        val_mask_dir = fold_mask_dir

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
        include_ids=fold_train_ids if use_fold_split else None,
    )
    val_dataset = ColonDataset(
        val_image_dir,
        val_mask_dir,
        transform=ValAugmentor((args.img_size, args.img_size)),
        use_boundary=args.use_boundary,
        boundary_radius=args.boundary_radius,
        include_ids=fold_val_ids if use_fold_split else None,
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

    model_kwargs = {
        "num_classes": 1,
        "use_boundary": args.use_boundary,
        "norm_type": args.norm_type,
        "deep_supervision": args.deep_supervision,
        "window_size": args.window_size,
        "use_shift_mask": args.use_shift_mask,
        "use_rel_pos_bias": args.use_rel_pos_bias,
        "pad_to_window": args.pad_to_window,
    }
    model = ResSwinUNet(**model_kwargs).to(args.device)

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=args.device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[init checkpoint] loaded from {args.init_checkpoint}")
        if missing:
            print(f"[init checkpoint] missing_keys={missing}")
        if unexpected:
            print(f"[init checkpoint] unexpected_keys={unexpected}")

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

    aux_criterion = None
    if args.deep_supervision:
        aux_criterion = FocalWithDiceLoss(
            focal_weight=args.seg_focal_weight,
            dice_weight=args.seg_dice_weight,
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
        )

    optimizer = _build_optimizer(model, args)
    scheduler = _build_scheduler(optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    amp_enabled = bool(args.amp and str(args.device).startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    writer = SummaryWriter(writer_dir)
    best_dice = -1.0

    mode_name = "dual_task_boundary" if args.use_boundary else "baseline"
    print(f"[mode] {mode_name}")
    print(f"[train images] {train_image_dir}")
    print(f"[train masks]  {train_mask_dir}")
    print(f"[val images]   {val_image_dir}")
    print(f"[val masks]    {val_mask_dir}")
    if use_fold_split:
        print(f"[kfold]       split={args.split_json}, fold={args.fold_index}, train_n={len(train_dataset)}, val_n={len(val_dataset)}")
    print(f"[log dir]      {writer_dir}")
    print(f"[save path]    {save_path}")
    print(f"[amp]          {amp_enabled}")

    ds_weights = [args.ds_weight_d2, args.ds_weight_d3, args.ds_weight_d4]

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            aux_criterion=aux_criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=args.device,
            train=True,
            use_boundary=args.use_boundary,
            deep_supervision=args.deep_supervision,
            ds_weights=ds_weights,
            grad_clip_norm=args.grad_clip_norm,
            epoch_idx=epoch - 1,
            total_epochs=args.epochs,
            boundary_warmup_ratio=args.boundary_warmup_ratio,
        )

        va = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            aux_criterion=aux_criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=args.device,
            train=False,
            use_boundary=args.use_boundary,
            deep_supervision=args.deep_supervision,
            ds_weights=ds_weights,
            grad_clip_norm=args.grad_clip_norm,
            epoch_idx=epoch - 1,
            total_epochs=args.epochs,
            boundary_warmup_ratio=args.boundary_warmup_ratio,
        )

        scheduler.step()

        writer.add_scalar("LR/main", optimizer.param_groups[0]["lr"], epoch)
        if len(optimizer.param_groups) > 1:
            writer.add_scalar("LR/encoder", optimizer.param_groups[1]["lr"], epoch)

        writer.add_scalar("Loss/train_total", tr["total"], epoch)
        writer.add_scalar("Loss/train_focal", tr["seg_focal"], epoch)
        writer.add_scalar("Loss/train_dice", tr["seg_dice"], epoch)
        writer.add_scalar("Loss/train_aux", tr["aux_total"], epoch)

        writer.add_scalar("Loss/val_total", va["total"], epoch)
        writer.add_scalar("Loss/val_focal", va["seg_focal"], epoch)
        writer.add_scalar("Loss/val_dice", va["seg_dice"], epoch)
        writer.add_scalar("Loss/val_aux", va["aux_total"], epoch)

        if args.use_boundary:
            writer.add_scalar("Loss/train_boundary_bce", tr["boundary_bce"], epoch)
            writer.add_scalar("Loss/train_boundary_dice", tr["boundary_dice"], epoch)
            writer.add_scalar("Loss/train_boundary_total", tr["boundary_total"], epoch)
            writer.add_scalar("Loss/train_boundary_weight", tr["boundary_weight"], epoch)

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
                f"train loss={tr['total']:.4f} dice={tr['dice']:.4f} bnd={tr['boundary_total']:.4f} bw={tr['boundary_weight']:.4f} | "
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
                    "model_kwargs": model_kwargs,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"[checkpoint] saved -> {save_path} (best_dice={best_dice:.4f})")

    writer.close()


if __name__ == "__main__":
    main()
