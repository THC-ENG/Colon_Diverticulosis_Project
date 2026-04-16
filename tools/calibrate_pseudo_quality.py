import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _build_features(row: dict, names: list[str]) -> list[float]:
    out = []
    source = str(row.get("source", "")).strip().lower()
    for name in names:
        key = str(name).strip()
        if key == "is_polypgen":
            out.append(1.0 if "polypgen" in source else 0.0)
        else:
            out.append(_to_float(row.get(key, 0.0), default=0.0))
    return out


def main():
    parser = argparse.ArgumentParser(description="Fit expected-Dice calibration from QC pseudo quality rows.")
    parser.add_argument("--quality-csv", type=str, required=True, help="Pseudo quality CSV (e.g. lora_qc/pseudo_val/pseudo_quality.csv)")
    parser.add_argument("--per-sample-csv", type=str, required=True, help="QC per-sample CSV with dice column")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument(
        "--feature-names",
        type=str,
        default="conf,edge_quality,area_ratio,area_prior,center_prior,quality,is_polypgen",
    )
    args = parser.parse_args()

    q_rows = _read_csv(args.quality_csv)
    p_rows = _read_csv(args.per_sample_csv)
    if not q_rows:
        raise RuntimeError(f"Empty quality csv: {args.quality_csv}")
    if not p_rows:
        raise RuntimeError(f"Empty per-sample csv: {args.per_sample_csv}")

    q_map = {str(r.get("id", "")).strip(): r for r in q_rows if str(r.get("id", "")).strip()}
    pairs = []
    for r in p_rows:
        pid = str(r.get("id", "")).strip()
        if not pid or pid not in q_map:
            continue
        dice = _to_float(r.get("dice", 0.0), default=0.0)
        pairs.append((q_map[pid], float(np.clip(dice, 0.0, 1.0))))

    if len(pairs) < int(args.min_samples):
        raise RuntimeError(f"Too few matched samples for calibration: {len(pairs)} < {int(args.min_samples)}")

    features = [x.strip() for x in str(args.feature_names).split(",") if x.strip()]
    if not features:
        raise RuntimeError("No feature names provided.")

    x_list = []
    y_list = []
    for row, dice in pairs:
        feats = _build_features(row, features)
        x_list.append([1.0] + feats)
        y_list.append(float(dice))

    X = np.array(x_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    n, d = X.shape

    alpha = float(max(0.0, args.ridge_alpha))
    reg = alpha * np.eye(d, dtype=np.float64)
    reg[0, 0] = 0.0
    w = np.linalg.solve(X.T @ X + reg, X.T @ y)

    y_pred = np.clip(X @ w, 0.0, 1.0)
    mae = float(np.mean(np.abs(y_pred - y)))
    rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    y_mean = float(np.mean(y))
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / max(1e-12, ss_tot)

    out = {
        "feature_names": features,
        "weights": [float(v) for v in w[1:].tolist()],
        "bias": float(w[0]),
        "ridge_alpha": float(alpha),
        "num_samples": int(n),
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "dice_mean": float(y_mean),
            "pred_mean": float(np.mean(y_pred)),
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
