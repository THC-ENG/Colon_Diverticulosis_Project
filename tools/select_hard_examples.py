import argparse
import json
from pathlib import Path


def _score_row(row: dict, strategy: str) -> float:
    dice = float(row.get("dice", 0.0))
    boundary_f1 = float(row.get("boundary_f1", 0.0))
    hd95 = float(row.get("hd95", 0.0))
    uncertainty = float(row.get("uncertainty_entropy", 0.0))

    if strategy == "low_dice":
        return 1.0 - dice
    if strategy == "boundary":
        return (1.0 - boundary_f1) + 0.1 * hd95
    if strategy == "uncertainty":
        return uncertainty
    if strategy == "composite":
        return 0.6 * (1.0 - dice) + 0.2 * (1.0 - boundary_f1) + 0.1 * uncertainty + 0.1 * hd95
    raise ValueError(f"Unsupported strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(description="Select hard samples for active relabeling.")
    parser.add_argument("--per-sample-report", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--strategy",
        type=str,
        default="composite",
        choices=["low_dice", "boundary", "uncertainty", "composite"],
    )
    parser.add_argument("--output", type=str, default="results/hard_examples.json")
    args = parser.parse_args()

    with open(args.per_sample_report, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError("per-sample report must be a JSON list.")

    scored = []
    for row in rows:
        score = _score_row(row, strategy=args.strategy)
        out = dict(row)
        out["hardness_score"] = float(score)
        scored.append(out)

    scored.sort(key=lambda x: x["hardness_score"], reverse=True)
    selected = scored[: max(1, int(args.top_k))]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print(json.dumps({"strategy": args.strategy, "top_k": len(selected), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
