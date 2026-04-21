import argparse
import csv
import json
import random
from pathlib import Path


def _read_csv(path: str) -> tuple[list[dict], list[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _largest_remainder_alloc(group_sizes: dict[str, int], total_pick: int) -> dict[str, int]:
    if total_pick <= 0 or not group_sizes:
        return {k: 0 for k in group_sizes}
    total_avail = sum(max(0, int(v)) for v in group_sizes.values())
    if total_avail <= 0:
        return {k: 0 for k in group_sizes}
    target = min(int(total_pick), int(total_avail))

    base = {}
    frac = []
    used = 0
    for k, n in group_sizes.items():
        n = max(0, int(n))
        exact = float(target) * float(n) / float(total_avail)
        b = min(n, int(exact))
        base[k] = b
        used += b
        frac.append((k, exact - float(b)))

    remain = target - used
    frac.sort(key=lambda x: x[1], reverse=True)
    i = 0
    while remain > 0 and frac:
        k = frac[i % len(frac)][0]
        if base[k] < int(group_sizes[k]):
            base[k] += 1
            remain -= 1
        i += 1
        if i > len(frac) * max(2, target + 1):
            break
    return base


def main():
    parser = argparse.ArgumentParser(description="Create adapted manifest with L_adapt_polypgen subset sampled from external/PolypGen.")
    parser.add_argument("--input-manifest", type=str, required=True)
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--output-id-list", type=str, default="")
    parser.add_argument("--output-summary-json", type=str, default="")
    parser.add_argument("--from-subset", type=str, default="external")
    parser.add_argument("--source-filter", type=str, default="PolypGen")
    parser.add_argument("--new-subset", type=str, default="L_adapt_polypgen")
    parser.add_argument("--num-samples", type=int, default=160)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, fieldnames = _read_csv(args.input_manifest)
    if not rows:
        raise RuntimeError(f"Empty manifest: {args.input_manifest}")
    if not fieldnames:
        fieldnames = list(rows[0].keys())

    from_subset = str(args.from_subset).strip()
    src_filter = str(args.source_filter).strip().lower()
    new_subset = str(args.new_subset).strip()
    if not new_subset:
        raise ValueError("--new-subset cannot be empty.")

    candidates = []
    for i, r in enumerate(rows):
        subset = str(r.get("subset", "")).strip()
        src = str(r.get("source", "")).strip()
        mask_path = str(r.get("mask_path", "")).strip()
        if subset != from_subset:
            continue
        if src_filter and src_filter not in src.lower():
            continue
        if not mask_path:
            continue
        candidates.append((i, r))
    if not candidates:
        raise RuntimeError(f"No candidate rows found for subset={from_subset}, source contains '{src_filter}'.")

    by_center: dict[str, list[tuple[int, dict]]] = {}
    for item in candidates:
        center = str(item[1].get("center", "")).strip() or "unknown"
        by_center.setdefault(center, []).append(item)

    rnd = random.Random(int(args.seed))
    for group in by_center.values():
        rnd.shuffle(group)

    alloc = _largest_remainder_alloc({k: len(v) for k, v in by_center.items()}, int(args.num_samples))
    selected_items: list[tuple[int, dict]] = []
    for center, group in by_center.items():
        take = min(len(group), int(alloc.get(center, 0)))
        selected_items.extend(group[:take])

    if len(selected_items) < int(args.num_samples):
        selected_set = {idx for idx, _ in selected_items}
        remain_pool = [x for x in candidates if x[0] not in selected_set]
        rnd.shuffle(remain_pool)
        need = int(args.num_samples) - len(selected_items)
        selected_items.extend(remain_pool[: max(0, need)])

    if len(selected_items) > int(args.num_samples):
        rnd.shuffle(selected_items)
        selected_items = selected_items[: int(args.num_samples)]

    if not selected_items:
        raise RuntimeError("No rows were selected; check filters.")

    selected_by_center: dict[str, list[tuple[int, dict]]] = {}
    for idx, row in selected_items:
        center = str(row.get("center", "")).strip() or "unknown"
        selected_by_center.setdefault(center, []).append((idx, row))
    for group in selected_by_center.values():
        rnd.shuffle(group)

    total_pick = len(selected_items)
    train_target = max(1, min(total_pick - 1, int(round(float(total_pick) * float(args.train_ratio))))) if total_pick > 1 else 1
    train_alloc = _largest_remainder_alloc({k: len(v) for k, v in selected_by_center.items()}, train_target)
    train_ids = set()
    for center, group in selected_by_center.items():
        take = min(len(group), int(train_alloc.get(center, 0)))
        for idx, _ in group[:take]:
            train_ids.add(idx)

    selected_ids = sorted(idx for idx, _ in selected_items)
    for idx in selected_ids:
        row = rows[idx]
        row["subset"] = new_subset
        row["split"] = "train" if idx in train_ids else "val"
        if "is_labeled" in row and str(row.get("is_labeled", "")).strip() == "":
            row["is_labeled"] = "1"

    out_ids = Path(args.output_id_list) if str(args.output_id_list).strip() else Path(args.output_manifest).with_name(f"{new_subset}_ids.txt")
    out_ids.parent.mkdir(parents=True, exist_ok=True)
    id_values = [str(rows[i].get("id", "")).strip() for i in selected_ids if str(rows[i].get("id", "")).strip()]
    out_ids.write_text("\n".join(id_values), encoding="utf-8")

    _write_csv(args.output_manifest, rows, fieldnames=fieldnames)

    summary = {
        "input_manifest": str(args.input_manifest),
        "output_manifest": str(args.output_manifest),
        "new_subset": new_subset,
        "source_filter": str(args.source_filter),
        "from_subset": from_subset,
        "num_candidates": int(len(candidates)),
        "num_selected": int(len(selected_ids)),
        "num_train": int(len(train_ids)),
        "num_val": int(len(selected_ids) - len(train_ids)),
        "seed": int(args.seed),
        "output_id_list": str(out_ids),
    }
    out_summary = Path(args.output_summary_json) if str(args.output_summary_json).strip() else Path(args.output_manifest).with_suffix(".summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
