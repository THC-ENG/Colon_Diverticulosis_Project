import argparse
import csv
import json
from pathlib import Path


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Select uncertain auto-box prompts for optional manual review.")
    parser.add_argument("--proposal-csv", type=str, required=True)
    parser.add_argument("--mean-prob-threshold", type=float, default=0.50)
    parser.add_argument("--bbox-area-min", type=float, default=0.01)
    parser.add_argument("--bbox-area-max", type=float, default=0.45)
    parser.add_argument("--max-review", type=int, default=300)
    parser.add_argument("--output-txt", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    rows = _read_csv(args.proposal_csv)
    if not rows:
        raise RuntimeError(f"Empty proposal csv: {args.proposal_csv}")

    flagged = []
    for r in rows:
        pid = str(r.get("id", "")).strip()
        if not pid:
            continue
        mean_prob = float(r.get("mean_prob", 0.0))
        bbox_area_ratio = float(r.get("bbox_area_ratio", 0.0))
        is_fallback = int(float(r.get("is_fallback", 0)))
        bad = (
            is_fallback == 1
            or mean_prob < float(args.mean_prob_threshold)
            or bbox_area_ratio < float(args.bbox_area_min)
            or bbox_area_ratio > float(args.bbox_area_max)
        )
        if bad:
            flagged.append(
                {
                    "id": pid,
                    "mean_prob": mean_prob,
                    "bbox_area_ratio": bbox_area_ratio,
                    "is_fallback": is_fallback,
                }
            )

    flagged.sort(key=lambda x: (x["is_fallback"], -x["mean_prob"]), reverse=True)
    flagged = flagged[: max(1, int(args.max_review))]
    ids = [x["id"] for x in flagged]

    out_txt = Path(args.output_txt) if args.output_txt else Path(args.proposal_csv).with_name("review_ids.txt")
    out_json = Path(args.output_json) if args.output_json else Path(args.proposal_csv).with_name("review_ids.json")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_txt.write_text("\n".join(ids), encoding="utf-8")
    out_json.write_text(json.dumps({"ids": ids, "rows": flagged}, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "num_total": len(rows),
        "num_flagged": len(flagged),
        "output_txt": str(out_txt),
        "output_json": str(out_json),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
