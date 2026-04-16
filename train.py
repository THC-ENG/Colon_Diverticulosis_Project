import argparse
import json
import math
import os
import random
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import ResSwinUNet
from utils.augmentations import DistillTrainAugmentor, TrainAugmentor, ValAugmentor
from utils.data_protocol import load_protocol_samples, summarize_samples, validate_protocol_samples
from utils.dataset import ColonDataset, ProtocolSegDataset, split_lsmall_train_val
from utils.losses import DualTaskLoss, FocalWithDiceLoss, StudentCompositeLoss
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
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be mapping: {config_path}")
    return cfg


def _parse_schedule(text: str) -> tuple[float, float, float]:
    vals = [v.strip() for v in str(text).split(",") if v.strip()]
    if len(vals) != 3:
        raise ValueError("--schedule-milestones must be like 0.2,0.6,0.8")
    m1, m2, m3 = map(float, vals)
    if not (0.0 <= m1 < m2 < m3 < 1.0):
        raise ValueError("Need 0<=m1<m2<m3<1")
    return m1, m2, m3


def _build_parser():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="configs/train_res_swin_unet.yaml")
    pre_args, _ = pre.parse_known_args()
    cfg = _load_yaml_config(pre_args.config)

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=pre_args.config)
    p.add_argument("--mode", type=str, default=cfg.get("mode", "supervised_only"), choices=["supervised_only", "student_with_pseudo_distill", "full_flywheel"])

    p.add_argument("--data-manifest", type=str, default=cfg.get("data_manifest", ""))
    p.add_argument("--data-root", type=str, default=cfg.get("data_root", ""))
    p.add_argument("--manifest-mode", type=str, default=cfg.get("manifest_mode", "prefer"), choices=["prefer", "only", "off"])
    p.add_argument("--mask-threshold", type=int, default=cfg.get("mask_threshold", 127))
    p.add_argument("--val-ratio", type=float, default=cfg.get("val_ratio", 0.2))
    p.add_argument("--train-pseudo-max-ratio", type=float, default=cfg.get("train_pseudo_max_ratio", -1.0))
    p.add_argument("--train-pseudo-max-count", type=int, default=cfg.get("train_pseudo_max_count", 0))
    p.add_argument("--use-domain-aware-sampler", action=argparse.BooleanOptionalAction, default=cfg.get("use_domain_aware_sampler", False))
    p.add_argument("--source-balance-power", type=float, default=cfg.get("source_balance_power", 0.5))
    p.add_argument("--polypgen-source-boost", type=float, default=cfg.get("polypgen_source_boost", 1.6))
    p.add_argument("--labeled-sample-factor", type=float, default=cfg.get("labeled_sample_factor", 1.0))
    p.add_argument("--pseudo-sample-factor", type=float, default=cfg.get("pseudo_sample_factor", 0.7))
    p.add_argument("--tier-mid-sample-factor", type=float, default=cfg.get("tier_mid_sample_factor", 0.6))
    p.add_argument("--tier-high-sample-factor", type=float, default=cfg.get("tier_high_sample_factor", 1.0))

    p.add_argument("--dataset-root", type=str, default=cfg.get("dataset_root", ""))
    p.add_argument("--train-image-dir", type=str, default=cfg.get("train_image_dir", "data/processed_images/images/train"))
    p.add_argument("--train-mask-dir", type=str, default=cfg.get("train_mask_dir", "data/processed_images/masks/train"))
    p.add_argument("--val-image-dir", type=str, default=cfg.get("val_image_dir", "data/processed_images/images/val"))
    p.add_argument("--val-mask-dir", type=str, default=cfg.get("val_mask_dir", "data/processed_images/masks/val"))
    p.add_argument("--all-image-dir", type=str, default=cfg.get("all_image_dir", ""))
    p.add_argument("--all-mask-dir", type=str, default=cfg.get("all_mask_dir", ""))
    p.add_argument("--split-json", type=str, default=cfg.get("split_json", ""))
    p.add_argument("--fold-index", type=int, default=cfg.get("fold_index", 0))

    p.add_argument("--epochs", type=int, default=cfg.get("epochs", 100))
    p.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 4))
    p.add_argument("--lr", type=float, default=cfg.get("lr", 1e-4))
    p.add_argument("--encoder-lr-mult", type=float, default=cfg.get("encoder_lr_mult", 0.1))
    p.add_argument("--weight-decay", type=float, default=cfg.get("weight_decay", 1e-4))
    p.add_argument("--warmup-epochs", type=int, default=cfg.get("warmup_epochs", 5))
    p.add_argument("--grad-clip-norm", type=float, default=cfg.get("grad_clip_norm", 1.0))
    p.add_argument("--img-size", type=int, default=cfg.get("img_size", 256))
    p.add_argument("--num-workers", type=int, default=cfg.get("num_workers", 2))
    p.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    p.add_argument("--device", type=str, default=cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg.get("amp", True))
    p.add_argument("--save-path", type=str, default=cfg.get("save_path", "checkpoints/best_model.pth"))
    p.add_argument("--init-checkpoint", type=str, default=cfg.get("init_checkpoint", ""))
    p.add_argument("--run-name", type=str, default=cfg.get("run_name", ""))
    p.add_argument("--polypgen-aug-prob", type=float, default=cfg.get("polypgen_aug_prob", 0.7))
    p.add_argument("--polypgen-h-shift-max", type=int, default=cfg.get("polypgen_h_shift_max", 12))
    p.add_argument("--polypgen-sat-scale-min", type=float, default=cfg.get("polypgen_sat_scale_min", 0.70))
    p.add_argument("--polypgen-sat-scale-max", type=float, default=cfg.get("polypgen_sat_scale_max", 1.40))
    p.add_argument("--polypgen-val-scale-min", type=float, default=cfg.get("polypgen_val_scale_min", 0.75))
    p.add_argument("--polypgen-val-scale-max", type=float, default=cfg.get("polypgen_val_scale_max", 1.30))
    p.add_argument("--polypgen-gamma-min", type=float, default=cfg.get("polypgen_gamma_min", 0.75))
    p.add_argument("--polypgen-gamma-max", type=float, default=cfg.get("polypgen_gamma_max", 1.35))
    p.add_argument("--polypgen-clahe-prob", type=float, default=cfg.get("polypgen_clahe_prob", 0.60))

    p.add_argument("--norm-type", type=str, default=cfg.get("norm_type", "gn"), choices=["bn", "gn"])
    p.add_argument("--deep-supervision", action=argparse.BooleanOptionalAction, default=cfg.get("deep_supervision", True))
    p.add_argument("--ds-weight-d2", type=float, default=cfg.get("ds_weight_d2", 0.4))
    p.add_argument("--ds-weight-d3", type=float, default=cfg.get("ds_weight_d3", 0.2))
    p.add_argument("--ds-weight-d4", type=float, default=cfg.get("ds_weight_d4", 0.1))
    p.add_argument("--window-size", type=int, default=cfg.get("window_size", 8))
    p.add_argument("--use-shift-mask", action=argparse.BooleanOptionalAction, default=cfg.get("use_shift_mask", True))
    p.add_argument("--use-rel-pos-bias", action=argparse.BooleanOptionalAction, default=cfg.get("use_rel_pos_bias", True))
    p.add_argument("--pad-to-window", action=argparse.BooleanOptionalAction, default=cfg.get("pad_to_window", True))
    p.add_argument("--use-wavelet-bottleneck", action=argparse.BooleanOptionalAction, default=cfg.get("use_wavelet_bottleneck", True))

    p.add_argument("--use-boundary", action=argparse.BooleanOptionalAction, default=cfg.get("use_boundary", False))
    p.add_argument("--use-boundary-head", action=argparse.BooleanOptionalAction, default=cfg.get("use_boundary_head", cfg.get("use_boundary", False)))
    p.add_argument("--boundary-radius", type=int, default=cfg.get("boundary_radius", 1))
    p.add_argument("--boundary-weight", type=float, default=cfg.get("boundary_weight", 0.3))
    p.add_argument("--boundary-bce-weight", type=float, default=cfg.get("boundary_bce_weight", 1.0))
    p.add_argument("--boundary-dice-weight", type=float, default=cfg.get("boundary_dice_weight", 0.0))

    p.add_argument("--seg-focal-weight", type=float, default=cfg.get("seg_focal_weight", 0.7))
    p.add_argument("--seg-dice-weight", type=float, default=cfg.get("seg_dice_weight", 0.3))
    p.add_argument("--focal-alpha", type=float, default=cfg.get("focal_alpha", 0.25))
    p.add_argument("--focal-gamma", type=float, default=cfg.get("focal_gamma", 2.0))
    p.add_argument("--use-edge-consistency", action=argparse.BooleanOptionalAction, default=cfg.get("use_edge_consistency", True))
    p.add_argument("--use-distill", action=argparse.BooleanOptionalAction, default=cfg.get("use_distill", True))
    p.add_argument("--distill-signals", type=str, default=cfg.get("distill_signals", "softmask,edge"))
    p.add_argument("--distill-temperature", type=float, default=cfg.get("distill_temperature", 2.0))
    p.add_argument("--lambda-sup", type=float, default=cfg.get("lambda_sup", 1.0))
    p.add_argument("--lambda-pseudo", type=float, default=cfg.get("lambda_pseudo", 0.4))
    p.add_argument("--lambda-edge", type=float, default=cfg.get("lambda_edge", 0.2))
    p.add_argument("--lambda-distill", type=float, default=cfg.get("lambda_distill", 0.2))
    p.add_argument("--schedule-milestones", type=str, default=cfg.get("schedule_milestones", "0.2,0.6,0.8"))

    p.add_argument("--base-sam-checkpoint", type=str, default=cfg.get("base_sam_checkpoint", ""))
    p.add_argument("--flywheel-rounds", type=int, default=cfg.get("flywheel_rounds", 2))
    p.add_argument("--round1-keep-quantile", type=float, default=cfg.get("round1_keep_quantile", 0.35))
    p.add_argument("--round2-keep-quantile", type=float, default=cfg.get("round2_keep_quantile", 0.15))
    p.add_argument("--quality-score", type=str, default=cfg.get("quality_score", "0.6*conf+0.4*edge_quality"))
    p.add_argument("--teacher-refresh-between-rounds", action=argparse.BooleanOptionalAction, default=cfg.get("teacher_refresh_between_rounds", True))
    p.add_argument("--flywheel-script", type=str, default=cfg.get("flywheel_script", "tools/run_full_flywheel.py"))
    return p


def _parse_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs.get("seg"), outputs.get("boundary"), outputs.get("aux") or []
    if isinstance(outputs, (tuple, list)):
        seg = outputs[0] if len(outputs) > 0 else None
        bnd = outputs[1] if len(outputs) > 1 else None
        aux = outputs[2] if len(outputs) > 2 and outputs[2] is not None else []
        return seg, bnd, aux
    return outputs, None, []


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


def _resolve_fold_split(args):
    if not args.split_json:
        return None, None, None, None

    split_path = Path(args.split_json)
    if not split_path.exists():
        raise FileNotFoundError(f"split_json not found: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    folds = payload.get("folds", [])
    if not folds:
        raise ValueError(f"No folds in split json: {split_path}")
    if args.fold_index < 0 or args.fold_index >= len(folds):
        raise ValueError(f"fold-index out of range: {args.fold_index}")

    fold = folds[args.fold_index]
    train_ids = set(fold.get("train_ids", []))
    val_ids = set(fold.get("val_ids", []))
    if not train_ids or not val_ids:
        raise ValueError("split_json has empty train_ids/val_ids.")

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


def _is_polypgen_source(text: str) -> bool:
    return "polypgen" in str(text or "").strip().lower()


def _build_protocol_sampler(train_rows, args):
    if not bool(args.use_domain_aware_sampler):
        return None, {}
    if not train_rows:
        return None, {}

    source_key = [str(s.source).strip() or "unknown" for s in train_rows]
    source_counter = Counter(source_key)
    sp = float(max(0.0, args.source_balance_power))
    pg_boost = float(max(0.0, args.polypgen_source_boost))
    labeled_factor = float(max(0.0, args.labeled_sample_factor))
    pseudo_factor = float(max(0.0, args.pseudo_sample_factor))
    tier_mid = float(max(0.0, args.tier_mid_sample_factor))
    tier_high = float(max(0.0, args.tier_high_sample_factor))

    weights = []
    for s, src in zip(train_rows, source_key):
        inv = 1.0 / float(max(1, source_counter[src]))
        source_factor = inv ** sp
        if _is_polypgen_source(src):
            source_factor *= pg_boost

        label_factor = pseudo_factor if int(s.is_pseudo) == 1 else labeled_factor
        if int(s.is_pseudo) == 1:
            tier = str(getattr(s, "tier", "") or "").strip().lower()
            if tier == "mid":
                tier_factor = tier_mid
            elif tier == "low":
                tier_factor = 0.0
            else:
                tier_factor = tier_high
        else:
            tier_factor = 1.0
        weights.append(float(max(0.0, source_factor * label_factor * tier_factor)))

    if not any(w > 0.0 for w in weights):
        return None, {
            "enabled": True,
            "fallback_reason": "all_weights_zero",
        }

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    stats = {
        "enabled": True,
        "num_rows": len(train_rows),
        "num_sources": len(source_counter),
        "source_counts": dict(source_counter),
        "weight_min": float(min(weights)),
        "weight_max": float(max(weights)),
        "weight_mean": float(sum(weights) / float(max(1, len(weights)))),
    }
    return sampler, stats


def _build_data(args):
    use_protocol = bool(args.data_manifest or args.data_root) and args.manifest_mode != "off"
    if use_protocol:
        samples = load_protocol_samples(args.data_manifest or None, args.data_root or None, args.manifest_mode)
        validate_protocol_samples(samples)
        print(f"[protocol] {json.dumps(summarize_samples(samples), ensure_ascii=False)}")
        train_rows, val_rows = split_lsmall_train_val(
            samples,
            val_ratio=args.val_ratio,
            seed=args.seed,
            include_pseudo=args.mode == "student_with_pseudo_distill",
            pseudo_max_ratio=args.train_pseudo_max_ratio,
            pseudo_max_count=args.train_pseudo_max_count,
        )
        sat_range = (float(args.polypgen_sat_scale_min), float(args.polypgen_sat_scale_max))
        val_range = (float(args.polypgen_val_scale_min), float(args.polypgen_val_scale_max))
        gamma_range = (float(args.polypgen_gamma_min), float(args.polypgen_gamma_max))
        if args.mode == "student_with_pseudo_distill" and args.use_distill:
            train_tf = DistillTrainAugmentor(
                (args.img_size, args.img_size),
                polypgen_aug_prob=float(args.polypgen_aug_prob),
                polypgen_h_shift_max=int(args.polypgen_h_shift_max),
                polypgen_sat_scale_range=sat_range,
                polypgen_val_scale_range=val_range,
                polypgen_gamma_range=gamma_range,
                polypgen_clahe_prob=float(args.polypgen_clahe_prob),
            )
        else:
            train_tf = TrainAugmentor(
                (args.img_size, args.img_size),
                polypgen_aug_prob=float(args.polypgen_aug_prob),
                polypgen_h_shift_max=int(args.polypgen_h_shift_max),
                polypgen_sat_scale_range=sat_range,
                polypgen_val_scale_range=val_range,
                polypgen_gamma_range=gamma_range,
                polypgen_clahe_prob=float(args.polypgen_clahe_prob),
            )
        val_tf = ValAugmentor((args.img_size, args.img_size))
        n_lsmall = sum(1 for s in train_rows if s.subset == "L_small" and s.is_pseudo == 0)
        n_pseudo = sum(1 for s in train_rows if s.is_pseudo == 1)
        pseudo_ratio = float(n_pseudo) / float(max(1, n_lsmall + n_pseudo))
        print(
            "[protocol split] "
            f"train_lsmall={n_lsmall} train_pseudo={n_pseudo} "
            f"pseudo_ratio={pseudo_ratio:.3f} val={len(val_rows)}"
        )
        train_ds = ProtocolSegDataset(train_rows, transform=train_tf, mask_threshold=args.mask_threshold)
        val_ds = ProtocolSegDataset(val_rows, transform=val_tf, mask_threshold=args.mask_threshold)
        train_sampler, sampler_stats = _build_protocol_sampler(train_rows, args)
        return use_protocol, train_ds, val_ds, train_sampler, sampler_stats

    fold_img, fold_msk, train_ids, val_ids = _resolve_fold_split(args)
    if fold_img is not None:
        tr_img, tr_msk, va_img, va_msk = fold_img, fold_msk, fold_img, fold_msk
    elif args.dataset_root:
        root = Path(args.dataset_root)
        tr_img = root / "images" / "train"
        tr_msk = root / "masks" / "train"
        va_img = root / "images" / "val"
        va_msk = root / "masks" / "val"
    else:
        tr_img, tr_msk, va_img, va_msk = args.train_image_dir, args.train_mask_dir, args.val_image_dir, args.val_mask_dir
        train_ids, val_ids = None, None

    train_ds = ColonDataset(
        tr_img,
        tr_msk,
        transform=TrainAugmentor(
            (args.img_size, args.img_size),
            polypgen_aug_prob=float(args.polypgen_aug_prob),
            polypgen_h_shift_max=int(args.polypgen_h_shift_max),
            polypgen_sat_scale_range=(float(args.polypgen_sat_scale_min), float(args.polypgen_sat_scale_max)),
            polypgen_val_scale_range=(float(args.polypgen_val_scale_min), float(args.polypgen_val_scale_max)),
            polypgen_gamma_range=(float(args.polypgen_gamma_min), float(args.polypgen_gamma_max)),
            polypgen_clahe_prob=float(args.polypgen_clahe_prob),
        ),
        use_boundary=args.use_boundary,
        boundary_radius=args.boundary_radius,
        include_ids=train_ids,
    )
    val_ds = ColonDataset(
        va_img,
        va_msk,
        transform=ValAugmentor((args.img_size, args.img_size)),
        use_boundary=args.use_boundary,
        boundary_radius=args.boundary_radius,
        include_ids=val_ids,
    )
    return use_protocol, train_ds, val_ds, None, {}


def _run_epoch(model, loader, criterion, aux_criterion, optimizer, scaler, args, train: bool, epoch_idx: int):
    model.train(train)
    amp_enabled = scaler.is_enabled() if scaler is not None else False
    t = {"total": 0.0, "dice": 0.0, "iou": 0.0, "aux_total": 0.0}
    for k in ["loss_sup", "loss_pseudo", "loss_edge", "loss_distill", "loss_distill_soft", "loss_distill_edge", "lambda_sup", "lambda_pseudo", "lambda_edge", "lambda_distill", "seg_focal", "seg_dice", "seg_focal_raw", "seg_dice_raw"]:
        t[k] = 0.0
    n = 0
    ratio = float(epoch_idx) / float(max(1, args.epochs - 1))

    for batch in tqdm(loader, leave=False):
        imgs = batch["image"].to(args.device)
        masks = batch["mask"].to(args.device)
        bs = imgs.size(0)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                seg_logits, b_logits, aux_logits = _parse_outputs(model(imgs))
                if isinstance(criterion, StudentCompositeLoss):
                    signals = {s.strip().lower() for s in str(args.distill_signals).split(",") if s.strip()}
                    has_soft = batch["has_distill_soft"].to(args.device) if "softmask" in signals else torch.zeros_like(batch["has_distill_soft"]).to(args.device)
                    has_edge = batch["has_distill_edge"].to(args.device) if "edge" in signals else torch.zeros_like(batch["has_distill_edge"]).to(args.device)
                    loss, stats = criterion(
                        seg_logits=seg_logits,
                        masks=masks,
                        is_labeled=batch["is_labeled"].to(args.device),
                        is_pseudo=batch["is_pseudo"].to(args.device),
                        pseudo_weight=batch["pseudo_weight"].to(args.device),
                        distill_soft=batch["distill_soft"].to(args.device),
                        distill_edge=batch["distill_edge"].to(args.device),
                        has_distill_soft=has_soft,
                        has_distill_edge=has_edge,
                        epoch_ratio=ratio,
                    )
                elif isinstance(criterion, DualTaskLoss):
                    boundaries = batch["boundary"].to(args.device)
                    loss, stats = criterion(seg_logits, b_logits, masks, boundaries)
                    stats["loss_sup"] = stats["seg_total"]
                    stats["loss_pseudo"] = torch.zeros_like(stats["loss_sup"])
                    stats["loss_edge"] = stats["boundary_total"]
                    stats["loss_distill"] = torch.zeros_like(stats["loss_sup"])
                    stats["loss_distill_soft"] = torch.zeros_like(stats["loss_sup"])
                    stats["loss_distill_edge"] = torch.zeros_like(stats["loss_sup"])
                    stats["lambda_sup"] = torch.tensor(1.0, device=args.device)
                    stats["lambda_pseudo"] = torch.tensor(0.0, device=args.device)
                    stats["lambda_edge"] = torch.tensor(1.0, device=args.device)
                    stats["lambda_distill"] = torch.tensor(0.0, device=args.device)
                    stats["seg_focal"] = stats["seg_focal"]
                    stats["seg_dice"] = stats["seg_dice"]
                else:
                    loss, focal, dice = criterion(seg_logits, masks)
                    stats = {
                        "loss_sup": loss.detach(),
                        "loss_pseudo": torch.zeros((), device=args.device),
                        "loss_edge": torch.zeros((), device=args.device),
                        "loss_distill": torch.zeros((), device=args.device),
                        "loss_distill_soft": torch.zeros((), device=args.device),
                        "loss_distill_edge": torch.zeros((), device=args.device),
                        "lambda_sup": torch.tensor(1.0, device=args.device),
                        "lambda_pseudo": torch.tensor(0.0, device=args.device),
                        "lambda_edge": torch.tensor(0.0, device=args.device),
                        "lambda_distill": torch.tensor(0.0, device=args.device),
                        "seg_focal": focal.detach(),
                        "seg_dice": dice.detach(),
                    }

                aux_total = torch.zeros((), device=args.device)
                if args.deep_supervision and aux_criterion is not None and aux_logits:
                    for w, aux_logit in zip([args.ds_weight_d2, args.ds_weight_d3, args.ds_weight_d4], aux_logits):
                        if w <= 0:
                            continue
                        aux_loss, _, _ = aux_criterion(aux_logit, masks)
                        aux_total = aux_total + float(w) * aux_loss
                loss = loss + aux_total

            if train:
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
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

        t["total"] += loss.detach().item() * bs
        t["aux_total"] += aux_total.detach().item() * bs
        t["dice"] += dice_per_sample(seg_logits.detach(), masks, threshold=0.5).sum().item()
        t["iou"] += iou_per_sample(seg_logits.detach(), masks, threshold=0.5).sum().item()
        for k in ["loss_sup", "loss_pseudo", "loss_edge", "loss_distill", "loss_distill_soft", "loss_distill_edge", "lambda_sup", "lambda_pseudo", "lambda_edge", "lambda_distill", "seg_focal", "seg_dice"]:
            if k in stats:
                t[k] += stats[k].detach().item() * bs
        n += bs
    n = max(1, n)
    return {k: v / n for k, v in t.items()}


def _run_full_flywheel(args):
    cmd = [
        sys.executable,
        args.flywheel_script,
        "--data-manifest", args.data_manifest,
        "--data-root", args.data_root,
        "--base-sam-checkpoint", args.base_sam_checkpoint,
        "--flywheel-rounds", str(args.flywheel_rounds),
        "--round1-keep-quantile", str(args.round1_keep_quantile),
        "--round2-keep-quantile", str(args.round2_keep_quantile),
        "--quality-score", args.quality_score,
        "--teacher-refresh-between-rounds", str(bool(args.teacher_refresh_between_rounds)).lower(),
        "--student-config", args.config,
    ]
    print("[full_flywheel]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main():
    args = _build_parser().parse_args()
    args.use_boundary = bool(args.use_boundary or args.use_boundary_head)
    if args.mode == "full_flywheel":
        if not args.data_manifest or not args.base_sam_checkpoint:
            raise ValueError("full_flywheel requires --data-manifest and --base-sam-checkpoint")
        _run_full_flywheel(args)
        return

    set_seed(args.seed)
    os.makedirs(Path(args.save_path).parent, exist_ok=True)
    writer_dir = str(Path("runs") / (args.run_name if args.run_name else args.mode))
    os.makedirs(writer_dir, exist_ok=True)

    use_protocol, train_ds, val_ds, train_sampler, sampler_stats = _build_data(args)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = ResSwinUNet(
        num_classes=1,
        use_boundary=args.use_boundary if not use_protocol else False,
        norm_type=args.norm_type,
        deep_supervision=args.deep_supervision,
        window_size=args.window_size,
        use_shift_mask=args.use_shift_mask,
        use_rel_pos_bias=args.use_rel_pos_bias,
        pad_to_window=args.pad_to_window,
        use_wavelet_bottleneck=args.use_wavelet_bottleneck,
    ).to(args.device)

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=args.device)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)

    if use_protocol:
        criterion = StudentCompositeLoss(
            lambda_sup=args.lambda_sup,
            lambda_pseudo=args.lambda_pseudo,
            lambda_edge=args.lambda_edge if args.use_edge_consistency else 0.0,
            lambda_distill=args.lambda_distill if args.use_distill else 0.0,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            seg_focal_weight=args.seg_focal_weight,
            seg_dice_weight=args.seg_dice_weight,
            distill_temperature=args.distill_temperature,
            schedule_milestones=_parse_schedule(args.schedule_milestones),
        )
    else:
        criterion = DualTaskLoss(
            seg_focal_weight=args.seg_focal_weight,
            seg_dice_weight=args.seg_dice_weight,
            boundary_weight=args.boundary_weight,
            boundary_bce_weight=args.boundary_bce_weight,
            boundary_dice_weight=args.boundary_dice_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
        ) if args.use_boundary else FocalWithDiceLoss(
            focal_weight=args.seg_focal_weight,
            dice_weight=args.seg_dice_weight,
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
        )
    aux_criterion = FocalWithDiceLoss(
        focal_weight=args.seg_focal_weight,
        dice_weight=args.seg_dice_weight,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
    ) if args.deep_supervision else None

    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": args.lr})
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr * args.encoder_lr_mult})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = _build_scheduler(opt, args.epochs, args.warmup_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and str(args.device).startswith("cuda")))
    writer = SummaryWriter(writer_dir)

    best = -1.0
    print(f"[mode] {args.mode} | [protocol] {use_protocol} | train={len(train_ds)} val={len(val_ds)}")
    if sampler_stats:
        print(f"[domain sampler] {json.dumps(sampler_stats, ensure_ascii=False)}")
    for epoch in range(1, args.epochs + 1):
        tr = _run_epoch(model, train_loader, criterion, aux_criterion, opt, scaler, args, True, epoch - 1)
        va = _run_epoch(model, val_loader, criterion, aux_criterion, opt, scaler, args, False, epoch - 1)
        sched.step()
        writer.add_scalar("Loss/train_total", tr["total"], epoch)
        writer.add_scalar("Loss/val_total", va["total"], epoch)
        writer.add_scalar("Metric/train_dice", tr["dice"], epoch)
        writer.add_scalar("Metric/val_dice", va["dice"], epoch)
        for k in ["loss_sup", "loss_pseudo", "loss_edge", "loss_distill", "lambda_sup", "lambda_pseudo", "lambda_edge", "lambda_distill", "aux_total"]:
            writer.add_scalar(f"Loss/train_{k}", tr[k], epoch)
            writer.add_scalar(f"Loss/val_{k}", va[k], epoch)
        if va["dice"] > best:
            best = va["dice"]
            torch.save({"model": model.state_dict(), "best_val_dice": best, "epoch": epoch, "args": vars(args)}, args.save_path)
            print(f"[checkpoint] epoch={epoch} best={best:.4f} -> {args.save_path}")
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['total']:.4f}/{tr['dice']:.4f} val={va['total']:.4f}/{va['dice']:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
