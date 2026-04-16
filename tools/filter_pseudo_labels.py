import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


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


def _score(expr: str, row: dict) -> float:
    safe = re.fullmatch(r"[0-9a-zA-Z_+\-*/().\s]+", expr or "")
    if safe is None:
        raise ValueError(f"Unsafe quality expression: {expr}")
    env: dict[str, float] = {}
    for k, v in row.items():
        key = str(k or "").strip()
        if not key:
            continue
        if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", key) is None:
            continue
        try:
            env[key] = float(v)
        except Exception:
            continue
    # Keep backward compatibility for expressions that use legacy keys.
    env.setdefault("conf", _to_float(row.get("conf", 0.0)))
    env.setdefault("edge_quality", _to_float(row.get("edge_quality", 0.0)))
    env.setdefault("quality", _to_float(row.get("quality", 0.0)))
    env.setdefault("area_ratio", _to_float(row.get("area_ratio", 0.0)))
    env.setdefault("area_prior", _to_float(row.get("area_prior", 0.0)))
    env.setdefault("center_prior", _to_float(row.get("center_prior", 0.0)))
    env.setdefault("consistency_iou", _to_float(row.get("consistency_iou", 0.0)))
    env.setdefault("spill_ratio", _to_float(row.get("spill_ratio", 0.0)))
    env.setdefault("reflection_overlap", _to_float(row.get("reflection_overlap", 0.0)))
    env.setdefault("largest_cc_ratio", _to_float(row.get("largest_cc_ratio", 0.0)))
    return float(
        eval(
            expr,
            {"__builtins__": {}},
            env,
        )
    )


def _load_calibration(path: str) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    feats = data.get("feature_names", [])
    w = data.get("weights", [])
    b = data.get("bias", 0.0)
    if not isinstance(feats, list) or not isinstance(w, list):
        return None
    if len(feats) != len(w):
        return None
    return {"feature_names": [str(x) for x in feats], "weights": [float(x) for x in w], "bias": float(b)}


def _predict_expected_dice(row: dict, calib: dict | None) -> float:
    if not calib:
        return -1.0
    features = calib["feature_names"]
    weights = calib["weights"]
    bias = float(calib.get("bias", 0.0))
    source = str(row.get("source", "")).strip().lower()
    y = float(bias)
    for name, w in zip(features, weights):
        if str(name) == "is_polypgen":
            x = 1.0 if "polypgen" in source else 0.0
        else:
            try:
                x = float(row.get(name, 0.0))
            except Exception:
                x = 0.0
        y += float(w) * float(x)
    return float(max(0.0, min(1.0, y)))


def _is_polypgen_source(text: str) -> bool:
    return "polypgen" in str(text or "").strip().lower()


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    q = float(max(0.0, min(1.0, q)))
    return float(np.quantile(np.array(values, dtype=np.float32), q))


def _score_percentile(values_sorted: np.ndarray, score: float) -> float:
    if values_sorted.size == 0:
        return 0.0
    rank = int(np.searchsorted(values_sorted, float(score), side="right"))
    return float(rank) / float(values_sorted.size)


def _legacy_global_quantile(
    scored: list[dict],
    keep_quantile: float,
) -> tuple[list[dict], list[dict]]:
    scored.sort(key=lambda x: float(x["score"]), reverse=True)
    keep_n = max(1, int(math.ceil(len(scored) * float(keep_quantile))))
    selected = scored[:keep_n]
    rejected = scored[keep_n:]
    return selected, rejected


def _tiered_source_quantile(
    scored: list[dict],
    tier_low_q: float,
    tier_high_q: float,
    polypgen_tier_low_q: float,
    polypgen_tier_high_q: float,
    mid_weight_scale: float,
    high_weight_scale: float,
    polypgen_mid_weight_scale: float,
    polypgen_high_weight_scale: float,
) -> tuple[list[dict], list[dict], dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        src = str(row.get("source", "")).strip()
        by_source[src].append(row)

    selected: list[dict] = []
    rejected: list[dict] = []
    source_stats: dict[str, dict] = {}

    for src, rows in by_source.items():
        src_is_poly = _is_polypgen_source(src)
        q_low = float(polypgen_tier_low_q if src_is_poly else tier_low_q)
        q_high = float(polypgen_tier_high_q if src_is_poly else tier_high_q)
        q_low = max(0.0, min(1.0, q_low))
        q_high = max(q_low, min(1.0, q_high))

        scores = [float(r["score"]) for r in rows]
        low_th = _quantile(scores, q_low)
        high_th = _quantile(scores, q_high)
        sorted_scores = np.sort(np.array(scores, dtype=np.float32))

        cnt_low = 0
        cnt_mid = 0
        cnt_high = 0

        for r in rows:
            score = float(r["score"])
            raw_weight = float(r.get("pseudo_weight_raw", score))
            if score <= low_th:
                tier = "low"
                scale = 0.0
                cnt_low += 1
            elif score <= high_th:
                tier = "mid"
                scale = float(polypgen_mid_weight_scale if src_is_poly else mid_weight_scale)
                cnt_mid += 1
            else:
                tier = "high"
                scale = float(polypgen_high_weight_scale if src_is_poly else high_weight_scale)
                cnt_high += 1

            out = dict(r)
            out["tier"] = tier
            out["source_q_low"] = float(q_low)
            out["source_q_high"] = float(q_high)
            out["source_score_low_threshold"] = float(low_th)
            out["source_score_high_threshold"] = float(high_th)
            out["score_source_percentile"] = _score_percentile(sorted_scores, score)
            out["pseudo_weight_raw"] = float(raw_weight)
            out["pseudo_weight_final"] = float(max(0.0, raw_weight * scale))
            out["tier_scale"] = float(scale)

            if tier == "low":
                rejected.append(out)
            else:
                selected.append(out)

        source_stats[src] = {
            "num_rows": len(rows),
            "q_low": float(q_low),
            "q_high": float(q_high),
            "score_low_threshold": float(low_th),
            "score_high_threshold": float(high_th),
            "num_low": int(cnt_low),
            "num_mid": int(cnt_mid),
            "num_high": int(cnt_high),
            "selected_frac": float((cnt_mid + cnt_high) / float(max(1, len(rows)))),
            "is_polypgen": bool(src_is_poly),
        }

    if not selected and scored:
        best = max(scored, key=lambda x: float(x["score"]))
        fallback = dict(best)
        fallback["tier"] = "high"
        fallback["source_q_low"] = 0.0
        fallback["source_q_high"] = 1.0
        fallback["source_score_low_threshold"] = float(best["score"])
        fallback["source_score_high_threshold"] = float(best["score"])
        fallback["score_source_percentile"] = 1.0
        fallback["pseudo_weight_raw"] = float(best.get("pseudo_weight_raw", best["score"]))
        fallback["pseudo_weight_final"] = float(fallback["pseudo_weight_raw"])
        fallback["tier_scale"] = 1.0
        selected = [fallback]
        rejected = [dict(r) for r in scored if r["id"] != fallback["id"]]

    selected.sort(key=lambda x: float(x["score"]), reverse=True)
    rejected.sort(key=lambda x: float(x["score"]), reverse=True)
    return selected, rejected, source_stats


def main():
    parser = argparse.ArgumentParser(description="Filter pseudo labels by quantile quality score.")
    parser.add_argument("--quality-csv", type=str, required=True)
    parser.add_argument("--keep-quantile", type=float, required=True)
    parser.add_argument("--quality-score", type=str, default="0.45*conf+0.25*edge_quality+0.20*area_prior+0.10*center_prior")
    parser.add_argument("--round-id", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="runs/flywheel/round1/filter")
    parser.add_argument("--pseudo-candidates-manifest", type=str, default="")
    parser.add_argument("--base-manifest", type=str, default="")

    parser.add_argument("--tiered-pseudo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tier-low-q", type=float, default=0.45)
    parser.add_argument("--tier-high-q", type=float, default=0.80)
    parser.add_argument("--polypgen-tier-low-q", type=float, default=0.55)
    parser.add_argument("--polypgen-tier-high-q", type=float, default=0.88)
    parser.add_argument("--mid-weight-scale", type=float, default=0.30)
    parser.add_argument("--high-weight-scale", type=float, default=1.00)
    parser.add_argument("--polypgen-mid-weight-scale", type=float, default=0.20)
    parser.add_argument("--polypgen-high-weight-scale", type=float, default=0.90)
    parser.add_argument("--calibration-json", type=str, default="")
    parser.add_argument("--expected-dice-min", type=float, default=-1.0)
    parser.add_argument("--polypgen-expected-dice-min", type=float, default=-1.0)
    parser.add_argument("--expected-dice-mid", type=float, default=-1.0)
    parser.add_argument("--expected-dice-mid-scale", type=float, default=0.65)
    args = parser.parse_args()

    rows = _read_csv(args.quality_csv)
    if not rows:
        raise RuntimeError(f"Empty quality csv: {args.quality_csv}")
    calib = _load_calibration(args.calibration_json)

    scored = []
    for r in rows:
        q = _score(args.quality_score, r)
        out = dict(r)
        out["score"] = q
        out["tier"] = ""
        out["source_q_low"] = ""
        out["source_q_high"] = ""
        out["source_score_low_threshold"] = ""
        out["source_score_high_threshold"] = ""
        out["score_source_percentile"] = ""
        out["pseudo_weight_raw"] = float(r.get("quality", q))
        out["pseudo_weight_final"] = float(r.get("quality", q))
        out["tier_scale"] = 1.0
        out["expected_dice"] = _predict_expected_dice(out, calib)
        out["calib_reject_min"] = 0
        out["calib_mid_scaled"] = 0
        scored.append(out)

    calib_rejected: list[dict] = []
    scored_after_calib: list[dict] = []
    for r in scored:
        if calib is None:
            scored_after_calib.append(r)
            continue
        ed = float(r.get("expected_dice", -1.0))
        if ed < 0.0:
            scored_after_calib.append(r)
            continue
        min_th = float(args.expected_dice_min)
        if _is_polypgen_source(r.get("source", "")) and float(args.polypgen_expected_dice_min) >= 0.0:
            min_th = float(args.polypgen_expected_dice_min)
        if min_th >= 0.0 and ed < min_th:
            rr = dict(r)
            rr["tier"] = "calib_low"
            rr["pseudo_weight_final"] = 0.0
            rr["tier_scale"] = 0.0
            rr["calib_reject_min"] = 1
            calib_rejected.append(rr)
        else:
            scored_after_calib.append(r)

    source_stats: dict[str, dict] = {}
    if not scored_after_calib:
        selected = []
        rejected = list(calib_rejected)
    elif args.tiered_pseudo:
        selected, rejected, source_stats = _tiered_source_quantile(
            scored=scored_after_calib,
            tier_low_q=args.tier_low_q,
            tier_high_q=args.tier_high_q,
            polypgen_tier_low_q=args.polypgen_tier_low_q,
            polypgen_tier_high_q=args.polypgen_tier_high_q,
            mid_weight_scale=args.mid_weight_scale,
            high_weight_scale=args.high_weight_scale,
            polypgen_mid_weight_scale=args.polypgen_mid_weight_scale,
            polypgen_high_weight_scale=args.polypgen_high_weight_scale,
        )
    else:
        selected, rejected = _legacy_global_quantile(scored_after_calib, args.keep_quantile)

    if calib is not None and float(args.expected_dice_mid) >= 0.0:
        mid_th = float(args.expected_dice_mid)
        mid_scale = float(max(0.0, args.expected_dice_mid_scale))
        for r in selected:
            ed = float(r.get("expected_dice", -1.0))
            if 0.0 <= ed < mid_th:
                base_w = float(r.get("pseudo_weight_final", r.get("pseudo_weight_raw", r.get("score", 0.0))))
                r["pseudo_weight_final"] = float(base_w * mid_scale)
                r["calib_mid_scaled"] = 1

    rejected.extend(calib_rejected)
    rejected.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_for_fields = selected if selected else (rejected if rejected else scored)
    fieldnames = list(rows_for_fields[0].keys()) if rows_for_fields else []
    selected_csv = out_dir / "selected_quality.csv"
    rejected_csv = out_dir / "rejected_quality.csv"
    _write_csv(str(selected_csv), selected, fieldnames=fieldnames)
    _write_csv(str(rejected_csv), rejected, fieldnames=fieldnames)

    selected_ids = {r["id"] for r in selected}
    (out_dir / "selected_ids.txt").write_text("\n".join(sorted(selected_ids)), encoding="utf-8")

    if args.pseudo_candidates_manifest:
        cand_rows = _read_csv(args.pseudo_candidates_manifest)
        cand_map = {r["id"]: r for r in cand_rows}
        selected_map = {r["id"]: r for r in selected}
        selected_manifest_rows = []
        for sid in sorted(selected_ids):
            if sid in cand_map:
                row = dict(cand_map[sid])
                srow = selected_map[sid]
                row["round_id"] = int(args.round_id)
                row["subset"] = f"pseudo_round{int(args.round_id)}"
                row["split"] = "train"
                row["pseudo_weight"] = float(srow.get("pseudo_weight_final", row.get("pseudo_weight", 0.0)))
                row["tier"] = str(srow.get("tier", "")).strip()
                row["score_source_percentile"] = srow.get("score_source_percentile", "")
                row["pseudo_weight_raw"] = srow.get("pseudo_weight_raw", row.get("pseudo_weight", 0.0))
                row["pseudo_weight_final"] = srow.get("pseudo_weight_final", row.get("pseudo_weight", 0.0))
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
            "tier",
            "score_source_percentile",
            "pseudo_weight_raw",
            "pseudo_weight_final",
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
        "num_calib_rejected": len(calib_rejected),
        "calibration_json": str(args.calibration_json),
        "expected_dice_min": float(args.expected_dice_min),
        "polypgen_expected_dice_min": float(args.polypgen_expected_dice_min),
        "expected_dice_mid": float(args.expected_dice_mid),
        "expected_dice_mid_scale": float(args.expected_dice_mid_scale),
        "keep_quantile": float(args.keep_quantile),
        "quality_score": args.quality_score,
        "tiered_pseudo": bool(args.tiered_pseudo),
        "tier_low_q": float(args.tier_low_q),
        "tier_high_q": float(args.tier_high_q),
        "polypgen_tier_low_q": float(args.polypgen_tier_low_q),
        "polypgen_tier_high_q": float(args.polypgen_tier_high_q),
        "mid_weight_scale": float(args.mid_weight_scale),
        "high_weight_scale": float(args.high_weight_scale),
        "polypgen_mid_weight_scale": float(args.polypgen_mid_weight_scale),
        "polypgen_high_weight_scale": float(args.polypgen_high_weight_scale),
        "source_stats": source_stats,
        "selected_csv": str(selected_csv),
        "rejected_csv": str(rejected_csv),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
