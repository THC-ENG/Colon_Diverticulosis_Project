import argparse
import csv
import json
import random
from pathlib import Path


def _read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _pick_val_ids(lsmall_ids: list[str], seed: int, val_ratio: float, val_count: int) -> set[str]:
    ids = list(lsmall_ids)
    if not ids:
        return set()

    rnd = random.Random(int(seed))
    rnd.shuffle(ids)

    if int(val_count) > 0:
        n_val = int(val_count)
    else:
        n_val = int(round(float(val_ratio) * float(len(ids))))
    if len(ids) > 1:
        n_val = max(1, min(len(ids) - 1, n_val))
    else:
        n_val = 0
    return set(ids[:n_val])


def main():
    parser = argparse.ArgumentParser(description="Freeze explicit train/val split for L_small rows in manifest.")
    parser.add_argument("--input-manifest", type=str, required=True)
    parser.add_argument("--output-manifest", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--fixed-val-id-list", type=str, default="")
    parser.add_argument("--output-id-list", type=str, default="")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    in_path = Path(args.input_manifest)
    if not in_path.exists():
        raise FileNotFoundError(f"Manifest not found: {in_path}")
    out_path = Path(args.output_manifest) if str(args.output_manifest).strip() else in_path

    rows, fieldnames = _read_csv(in_path)
    if not rows:
        raise RuntimeError(f"Empty manifest: {in_path}")
    if "split" not in fieldnames:
        fieldnames.append("split")

    lsmall_rows = [r for r in rows if str(r.get("subset", "")).strip() == "L_small"]
    if not lsmall_rows:
        raise RuntimeError("No L_small rows found in manifest.")

    lsmall_ids = []
    for r in lsmall_rows:
        sid = str(r.get("id", "")).strip()
        if sid:
            lsmall_ids.append(sid)
    if not lsmall_ids:
        raise RuntimeError("L_small rows have no usable IDs.")

    unique_lsmall_ids = sorted(set(lsmall_ids))
    fixed_val_ids_path = Path(args.fixed_val_id_list) if str(args.fixed_val_id_list).strip() else None
    if fixed_val_ids_path is not None and fixed_val_ids_path.exists():
        fixed_text = fixed_val_ids_path.read_text(encoding="utf-8")
        fixed_ids = {line.strip() for line in fixed_text.splitlines() if line.strip()}
        val_ids = {sid for sid in unique_lsmall_ids if sid in fixed_ids}
        if not val_ids:
            raise RuntimeError(
                f"Fixed val id list is present but no IDs match current L_small set: {fixed_val_ids_path}"
            )
    else:
        val_ids = _pick_val_ids(
            lsmall_ids=unique_lsmall_ids,
            seed=int(args.seed),
            val_ratio=float(args.val_ratio),
            val_count=int(args.val_count),
        )

    for r in rows:
        if str(r.get("subset", "")).strip() != "L_small":
            continue
        sid = str(r.get("id", "")).strip()
        r["split"] = "val" if sid in val_ids else "train"

    if not bool(args.dry_run):
        _write_csv(out_path, rows, fieldnames=fieldnames)

    if str(args.output_id_list).strip():
        val_id_path = Path(args.output_id_list)
    elif fixed_val_ids_path is not None:
        val_id_path = fixed_val_ids_path
    else:
        val_id_path = out_path.with_name("lsmall_val_ids.txt")
    val_id_path.parent.mkdir(parents=True, exist_ok=True)
    val_id_path.write_text("\n".join(sorted(val_ids)), encoding="utf-8")

    summary = {
        "input_manifest": str(in_path),
        "output_manifest": str(out_path),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "val_count_arg": int(args.val_count),
        "num_lsmall": int(len(set(lsmall_ids))),
        "num_val": int(len(val_ids)),
        "num_train": int(len(set(lsmall_ids)) - len(val_ids)),
        "val_ids_path": str(val_id_path),
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
