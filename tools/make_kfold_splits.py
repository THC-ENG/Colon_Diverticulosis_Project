import argparse
import json
import random
from pathlib import Path


def _list_ids(image_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p.stem for p in image_dir.glob("*") if p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser(description="Create K-fold split JSON from image IDs.")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/splits/kfold_5_seed42.json")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    ids = _list_ids(image_dir)
    if len(ids) < args.num_folds:
        raise ValueError(f"Not enough samples ({len(ids)}) for num_folds={args.num_folds}")

    rng = random.Random(args.seed)
    rng.shuffle(ids)

    fold_sizes = [len(ids) // args.num_folds] * args.num_folds
    for i in range(len(ids) % args.num_folds):
        fold_sizes[i] += 1

    folds = []
    offset = 0
    for fold_idx, fold_size in enumerate(fold_sizes):
        val_ids = ids[offset: offset + fold_size]
        train_ids = ids[:offset] + ids[offset + fold_size:]
        folds.append(
            {
                "fold_index": fold_idx,
                "train_ids": train_ids,
                "val_ids": val_ids,
            }
        )
        offset += fold_size

    payload = {
        "num_folds": args.num_folds,
        "seed": args.seed,
        "image_dir": str(image_dir),
        "num_samples": len(ids),
        "folds": folds,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps({"output": str(output_path), "num_samples": len(ids), "num_folds": args.num_folds}, indent=2))


if __name__ == "__main__":
    main()
