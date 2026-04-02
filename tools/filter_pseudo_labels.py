import argparse
import csv
import json
import math
import re
from pathlib import Path


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _score(
    expr: str,
    conf: float,
    edge_quality: float,
    quality: float,
    area_ratio: float,
    area_prior: float,
    center_prior: float,
) -> float:
    safe = re.fullmatch(r"[0-9a-zA-Z_+\-*/().\s]+", expr or "")
    if safe is None:
        raise ValueError(f"Unsafe quality expression: {expr}")
    return float(
        eval(
            expr,
            {"__builtins__": {}},
            {
                "conf": conf,
                "edge_quality": edge_quality,
                "quality": quality,
                "area_ratio": area_ratio,
                "area_prior": area_prior,
                "center_prior": center_prior,
            },
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Filter pseudo labels by quantile quality score.")
    parser.add_argument("--quality-csv", type=str, required=True)
    parser.add_argument("--keep-quantile", type=float, required=True)
    parser.add_argument("--quality-score", type=str, default="0.45*conf+0.25*edge_quality+0.20*area_prior+0.10*center_prior")
    parser.add_argument("--round-id", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="runs/flywheel/round1/filter")
    parser.add_argument("--pseudo-candidates-manifest", type=str, default="")
    parser.add_argument("--base-manifest", type=str, default="")
    args = parser.parse_args()

    rows = _read_csv(args.quality_csv)
    if not rows:
        raise RuntimeError(f"Empty quality csv: {args.quality_csv}")

    scored = []
    for r in rows:
        conf = float(r.get("conf", 0.0))
        edge_q = float(r.get("edge_quality", 0.0))
        quality = float(r.get("quality", 0.0))
        area_ratio = float(r.get("area_ratio", 0.0))
        area_prior = float(r.get("area_prior", 0.0))
        center_prior = float(r.get("center_prior", 0.0))
        q = _score(
            args.quality_score,
            conf=conf,
            edge_quality=edge_q,
            quality=quality,
            area_ratio=area_ratio,
            area_prior=area_prior,
            center_prior=center_prior,
        )
        out = dict(r)
        out["score"] = q
        scored.append(out)

    scored.sort(key=lambda x: float(x["score"]), reverse=True)
    keep_n = max(1, int(math.ceil(len(scored) * float(args.keep_quantile))))
    selected = scored[:keep_n]
    rejected = scored[keep_n:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = list(selected[0].keys()) if selected else list(scored[0].keys())
    selected_csv = out_dir / "selected_quality.csv"
    rejected_csv = out_dir / "rejected_quality.csv"
    _write_csv(str(selected_csv), selected, fieldnames=fieldnames)
    _write_csv(str(rejected_csv), rejected, fieldnames=fieldnames)

    selected_ids = {r["id"] for r in selected}
    (out_dir / "selected_ids.txt").write_text("\n".join(sorted(selected_ids)), encoding="utf-8")

    if args.pseudo_candidates_manifest:
        cand_rows = _read_csv(args.pseudo_candidates_manifest)
        cand_map = {r["id"]: r for r in cand_rows}
        selected_manifest_rows = []
        for sid in sorted(selected_ids):
            if sid in cand_map:
                row = dict(cand_map[sid])
                row["round_id"] = int(args.round_id)
                row["subset"] = f"pseudo_round{int(args.round_id)}"
                row["split"] = "train"
                selected_manifest_rows.append(row)
        mf_fields = [
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
        ]
        selected_manifest = out_dir / "selected_manifest.csv"
        _write_csv(str(selected_manifest), selected_manifest_rows, fieldnames=mf_fields)

    if args.base_manifest:
        base_rows = _read_csv(args.base_manifest)
        remaining_rows = []
        for r in base_rows:
            if str(r.get("subset", "")).strip() != "U_large":
                continue
            if str(r.get("id", "")).strip() in selected_ids:
                continue
            remaining_rows.append(r)
        rem_fields = base_rows[0].keys() if base_rows else []
        if rem_fields:
            _write_csv(str(out_dir / "remaining_u_large_manifest.csv"), remaining_rows, fieldnames=list(rem_fields))

    summary = {
        "round_id": int(args.round_id),
        "num_total": len(scored),
        "num_selected": len(selected),
        "num_rejected": len(rejected),
        "keep_quantile": float(args.keep_quantile),
        "quality_score": args.quality_score,
        "selected_csv": str(selected_csv),
        "rejected_csv": str(rejected_csv),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
