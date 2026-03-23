import argparse
import glob
import json
from pathlib import Path

import numpy as np


def _parse_group(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Invalid group format: {text}. Expected label=glob")
    label, pattern = text.split("=", 1)
    label = label.strip()
    pattern = pattern.strip()
    if not label or not pattern:
        raise ValueError(f"Invalid group format: {text}. Expected label=glob")
    return label, pattern


def _load_metric_values(pattern: str, metric: str) -> list[float]:
    values = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if metric not in payload:
            raise KeyError(f"Metric '{metric}' not found in {path}")
        values.append(float(payload[metric]))
    return values


def _paired_significance(values_a: list[float], values_b: list[float]) -> dict:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return {"enabled": False, "reason": "need paired samples with n>=2"}

    out = {
        "enabled": True,
    }
    try:
        from scipy.stats import ttest_rel, wilcoxon

        t_stat, t_p = ttest_rel(values_b, values_a)
        out["paired_t_test"] = {
            "statistic": float(t_stat),
            "p_value": float(t_p),
        }

        w_stat, w_p = wilcoxon(values_b, values_a)
        out["wilcoxon"] = {
            "statistic": float(w_stat),
            "p_value": float(w_p),
        }
    except Exception as e:
        out["enabled"] = False
        out["reason"] = f"scipy unavailable or failed: {e}"
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate seed-level metric reports.")
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="Group in format label=glob_path (repeatable)",
    )
    parser.add_argument("--metric", type=str, default="dice_mean")
    parser.add_argument("--compare-to", type=str, default="", help="Baseline label for paired significance")
    parser.add_argument("--output", type=str, default="results/ablation_summary.json")
    args = parser.parse_args()

    groups = [_parse_group(item) for item in args.group]
    summary = {
        "metric": args.metric,
        "groups": {},
        "significance": {},
    }

    raw_values = {}
    for label, pattern in groups:
        vals = _load_metric_values(pattern, args.metric)
        if not vals:
            raise RuntimeError(f"No files matched for group '{label}' with pattern: {pattern}")

        arr = np.array(vals, dtype=np.float64)
        raw_values[label] = vals
        summary["groups"][label] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "values": [float(v) for v in vals],
            "pattern": pattern,
        }

    if args.compare_to:
        if args.compare_to not in raw_values:
            raise KeyError(f"compare-to label '{args.compare_to}' not found")

        baseline_vals = raw_values[args.compare_to]
        for label, vals in raw_values.items():
            if label == args.compare_to:
                continue
            summary["significance"][f"{label}_vs_{args.compare_to}"] = _paired_significance(
                baseline_vals,
                vals,
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    main()
