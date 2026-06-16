import argparse
import csv
import json
from collections import Counter
from pathlib import Path


BASE_COLUMNS = [
    "id",
    "image_path",
    "mask_path",
    "subset",
    "split",
    "source",
    "center",
    "is_labeled",
    "is_pseudo",
    "pseudo_weight",
    "round_id",
    "exclude_from_tuning",
    "soft_path",
    "edge_path",
    "tier",
]


def _read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _quality(row: dict) -> float:
    for key in (
        "bridge_quality",
        "quality_post",
        "teacher_quality",
        "pseudo_weight_final",
        "pseudo_weight_raw",
        "pseudo_weight",
    ):
        text = str(row.get(key, "")).strip()
        if text:
            return max(0.0, min(1.0, _to_float(text)))
    return 0.0


def _rank_bucket(idx: int, n: int) -> str:
    if n <= 0:
        return "low"
    pct = (idx + 1) / float(n)
    if pct <= 0.30:
        return "high"
    if pct <= 0.80:
        return "mid"
    return "low"


def _weight_for_bucket(bucket: str, high: float, mid: float, low: float) -> float:
    if bucket == "high":
        return float(high)
    if bucket == "mid":
        return float(mid)
    return float(low)


def _allocate_source_quotas(rows: list[dict], max_extra: int) -> dict[str, int]:
    if max_extra <= 0 or max_extra >= len(rows):
        return dict(Counter(str(r.get("source", "")).strip() or "unknown" for r in rows))
    counts = Counter(str(r.get("source", "")).strip() or "unknown" for r in rows)
    raw = {src: max_extra * count / float(len(rows)) for src, count in counts.items()}
    quotas = {src: min(counts[src], int(raw[src])) for src in counts}
    remaining = max_extra - sum(quotas.values())
    order = sorted(counts, key=lambda s: (raw[s] - int(raw[s]), counts[s]), reverse=True)
    while remaining > 0:
        changed = False
        for src in order:
            if remaining <= 0:
                break
            if quotas[src] < counts[src]:
                quotas[src] += 1
                remaining -= 1
                changed = True
        if not changed:
            break
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a follow-up student manifest by adding low-weight pseudo samples from an existing pseudo pool."
    )
    parser.add_argument("--base-final-manifest", required=True)
    parser.add_argument("--all-pseudo-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--max-extra", type=int, default=420)
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument("--extra-high-weight", type=float, default=0.06)
    parser.add_argument("--extra-mid-weight", type=float, default=0.035)
    parser.add_argument("--extra-low-weight", type=float, default=0.015)
    parser.add_argument("--extra-tier", default="mid")
    args = parser.parse_args()

    base_path = Path(args.base_final_manifest)
    all_path = Path(args.all_pseudo_manifest)
    out_path = Path(args.output_manifest)
    summary_path = Path(args.output_summary) if str(args.output_summary).strip() else out_path.with_suffix(".summary.json")

    base_rows, base_fields = _read_csv(base_path)
    all_rows, all_fields = _read_csv(all_path)

    base_ids = {str(r.get("id", "")).strip() for r in base_rows if str(r.get("id", "")).strip()}
    pool = [
        r
        for r in all_rows
        if str(r.get("is_pseudo", "")).strip() == "1"
        and str(r.get("id", "")).strip()
        and str(r.get("id", "")).strip() not in base_ids
        and str(r.get("exclude_from_tuning", "0")).strip() not in {"1", "true", "True"}
        and str(r.get("mask_path", "")).strip()
        and _quality(r) >= float(args.min_quality)
    ]

    by_source: dict[str, list[dict]] = {}
    for row in pool:
        src = str(row.get("source", "")).strip() or "unknown"
        by_source.setdefault(src, []).append(row)
    for src in by_source:
        by_source[src] = sorted(by_source[src], key=_quality, reverse=True)

    quotas = _allocate_source_quotas(pool, int(args.max_extra))
    selected_extra: list[dict] = []
    for src, quota in quotas.items():
        selected_extra.extend(by_source.get(src, [])[: int(quota)])
    selected_extra = sorted(selected_extra, key=lambda r: (str(r.get("source", "")), -_quality(r), str(r.get("id", ""))))

    adjusted_extra: list[dict] = []
    by_src_selected: dict[str, list[dict]] = {}
    for row in selected_extra:
        by_src_selected.setdefault(str(row.get("source", "")).strip() or "unknown", []).append(row)

    for src, src_rows in by_src_selected.items():
        src_sorted = sorted(src_rows, key=_quality, reverse=True)
        for idx, row in enumerate(src_sorted):
            bucket = _rank_bucket(idx, len(src_sorted))
            out = dict(row)
            out["split"] = "pseudo_train"
            out["is_labeled"] = "0"
            out["is_pseudo"] = "1"
            out["exclude_from_tuning"] = "0"
            out["tier"] = str(args.extra_tier)
            out["pseudo_weight"] = f"{_weight_for_bucket(bucket, args.extra_high_weight, args.extra_mid_weight, args.extra_low_weight):.6f}"
            out["low_weight_source_rank_bucket"] = bucket
            out["low_weight_quality"] = f"{_quality(row):.6f}"
            out["low_weight_reason"] = "existing_pseudo_low_weight_expansion"
            adjusted_extra.append(out)

    out_rows = list(base_rows) + sorted(adjusted_extra, key=lambda r: (str(r.get("round_id", "")), str(r.get("source", "")), str(r.get("id", ""))))

    fieldnames = []
    for col in BASE_COLUMNS + base_fields + all_fields + [
        "low_weight_source_rank_bucket",
        "low_weight_quality",
        "low_weight_reason",
    ]:
        if col not in fieldnames:
            fieldnames.append(col)

    _write_csv(out_path, out_rows, fieldnames)

    summary = {
        "base_final_manifest": str(base_path),
        "all_pseudo_manifest": str(all_path),
        "output_manifest": str(out_path),
        "num_base_rows": len(base_rows),
        "num_base_pseudo": sum(1 for r in base_rows if str(r.get("is_pseudo", "")).strip() == "1"),
        "num_pool_after_filters": len(pool),
        "max_extra": int(args.max_extra),
        "num_extra_selected": len(adjusted_extra),
        "num_output_rows": len(out_rows),
        "extra_by_source": dict(Counter(str(r.get("source", "")).strip() or "unknown" for r in adjusted_extra)),
        "extra_by_round": dict(Counter(str(r.get("round_id", "")).strip() for r in adjusted_extra)),
        "extra_weight_counts": dict(Counter(str(r.get("pseudo_weight", "")).strip() for r in adjusted_extra)),
        "extra_rank_buckets": dict(Counter(str(r.get("low_weight_source_rank_bucket", "")).strip() for r in adjusted_extra)),
        "output_subset_counts": dict(Counter(str(r.get("subset", "")).strip() for r in out_rows)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
