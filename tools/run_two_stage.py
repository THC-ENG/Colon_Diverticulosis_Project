import argparse
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Two-stage training runner: pretrain then finetune.")
    parser.add_argument("--config", type=str, default="configs/train_res_swin_unet.yaml")
    parser.add_argument("--pretrain-dataset-root", type=str, required=True)
    parser.add_argument("--finetune-dataset-root", type=str, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--finetune-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-args", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-two-stage",
        action="store_true",
        help="Explicitly allow running two-stage training. Keep disabled during expert-only feasibility stage.",
    )
    args = parser.parse_args()

    if not args.allow_two_stage:
        print(
            "[skip] Two-stage training is disabled by default in the current workflow.\n"
            "Use expert-only single-stage training first.\n"
            "If you really want to run this script, add: --allow-two-stage"
        )
        return

    extra_args = shlex.split(args.extra_args)

    pretrain_ckpt = "checkpoints/stage1_pretrain.pth"
    cmd_pretrain = [
        sys.executable,
        "train.py",
        "--config",
        args.config,
        "--dataset-root",
        args.pretrain_dataset_root,
        "--epochs",
        str(args.pretrain_epochs),
        "--seed",
        str(args.seed),
        "--save-path",
        pretrain_ckpt,
        "--run-name",
        "stage1_pretrain",
    ] + extra_args

    cmd_finetune = [
        sys.executable,
        "train.py",
        "--config",
        args.config,
        "--dataset-root",
        args.finetune_dataset_root,
        "--epochs",
        str(args.finetune_epochs),
        "--seed",
        str(args.seed),
        "--init-checkpoint",
        pretrain_ckpt,
        "--save-path",
        "checkpoints/stage2_finetune.pth",
        "--run-name",
        "stage2_finetune",
    ] + extra_args

    print("[stage1]", " ".join(shlex.quote(c) for c in cmd_pretrain))
    print("[stage2]", " ".join(shlex.quote(c) for c in cmd_finetune))

    if not args.dry_run:
        subprocess.run(cmd_pretrain, check=True)
        subprocess.run(cmd_finetune, check=True)


if __name__ == "__main__":
    main()
