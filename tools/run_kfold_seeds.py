import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _parse_seeds(text: str) -> list[int]:
    seeds = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


def main():
    parser = argparse.ArgumentParser(description="Run train.py across folds and seeds.")
    parser.add_argument("--config", type=str, default="configs/train_res_swin_unet.yaml")
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--all-image-dir", type=str, default="")
    parser.add_argument("--all-mask-dir", type=str, default="")
    parser.add_argument("--extra-args", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    extra_args = shlex.split(args.extra_args)

    for seed in seeds:
        for fold_idx in range(args.num_folds):
            run_name = f"cv_fold{fold_idx}_seed{seed}"
            save_path = str(Path("checkpoints") / f"{run_name}.pth")

            cmd = [
                sys.executable,
                "train.py",
                "--config",
                args.config,
                "--seed",
                str(seed),
                "--split-json",
                args.split_json,
                "--fold-index",
                str(fold_idx),
                "--run-name",
                run_name,
                "--save-path",
                save_path,
            ]

            if args.all_image_dir:
                cmd.extend(["--all-image-dir", args.all_image_dir])
            if args.all_mask_dir:
                cmd.extend(["--all-mask-dir", args.all_mask_dir])

            cmd.extend(extra_args)

            print("[run]", " ".join(shlex.quote(c) for c in cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
