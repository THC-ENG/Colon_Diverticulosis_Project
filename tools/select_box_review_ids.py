import argparse
import csv
import json
import math
from pathlib import Path

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _center_bias_from_box(row: dict) -> float:
    if cv2 is None:
        return 0.0
    image_path = str(row.get("image_path", "")).strip()
    if not image_path:
        return 0.0
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    if h <= 1 or w <= 1:
        return 0.0
    x0 = _to_float(row.get("x0", 0.0))
    y0 = _to_float(row.get("y0", 0.0))
    x1 = _to_float(row.get("x1", 0.0))
    y1 = _to_float(row.get("y1", 0.0))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    dx = (cx / float(max(1, w - 1))) - 0.5
    dy = (cy / float(max(1, h - 1))) - 0.5
    dist = math.sqrt(dx * dx + dy * dy)
    return float(dist / (math.sqrt(0.5 * 0.5 + 0.5 * 0.5) + 1e-6))


def main():
    parser = argparse.ArgumentParser(description="Select uncertain auto-box prompts for optional manual review.")
    parser.add_argument("--proposal-csv", type=str, required=True)
    parser.add_argument("--mean-prob-threshold", type=float, default=0.50)
    parser.add_argument("--bbox-area-min", type=float, default=0.01)
    parser.add_argument("--bbox-area-max", type=float, default=0.45)
    parser.add_argument("--high-conf-threshold", type=float, default=0.72)
    parser.add_argument("--tiny-box-area-max", type=float, default=0.01)
    parser.add_argument("--reflection-high-threshold", type=float, default=0.10)
    parser.add_argument("--aspect-ratio-max", type=float, default=4.5)
    parser.add_argument("--center-bias-threshold", type=float, default=0.62)
    parser.add_argument("--polypgen-min-quota", type=int, default=70)
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
        mean_prob = _to_float(r.get("mean_prob", 0.0))
        bbox_area_ratio = _to_float(r.get("bbox_area_ratio", 0.0))
        reflection_overlap = _to_float(r.get("reflection_overlap", 0.0))
        x0 = _to_float(r.get("x0", 0.0))
        y0 = _to_float(r.get("y0", 0.0))
        x1 = _to_float(r.get("x1", 0.0))
        y1 = _to_float(r.get("y1", 0.0))
        bw = max(1.0, abs(x1 - x0) + 1.0)
        bh = max(1.0, abs(y1 - y0) + 1.0)
        aspect = max(bw / bh, bh / bw)
        center_bias = _center_bias_from_box(r)
        is_fallback = int(_to_float(r.get("is_fallback", 0.0)))

        basic_bad = (
            is_fallback == 1
            or mean_prob < float(args.mean_prob_threshold)
            or bbox_area_ratio < float(args.bbox_area_min)
            or bbox_area_ratio > float(args.bbox_area_max)
        )
        high_conf_anomaly = (
            mean_prob >= float(args.high_conf_threshold)
            and (
                bbox_area_ratio <= float(args.tiny_box_area_max)
                or reflection_overlap >= float(args.reflection_high_threshold)
                or aspect >= float(args.aspect_ratio_max)
                or center_bias >= float(args.center_bias_threshold)
            )
        )
        if not (basic_bad or high_conf_anomaly):
            continue

        reasons = []
        if is_fallback == 1:
            reasons.append("fallback")
        if mean_prob < float(args.mean_prob_threshold):
            reasons.append("low_conf")
        if bbox_area_ratio < float(args.bbox_area_min):
            reasons.append("small_area")
        if bbox_area_ratio > float(args.bbox_area_max):
            reasons.append("large_area")
        if mean_prob >= float(args.high_conf_threshold) and bbox_area_ratio <= float(args.tiny_box_area_max):
            reasons.append("highconf_tiny")
        if mean_prob >= float(args.high_conf_threshold) and reflection_overlap >= float(args.reflection_high_threshold):
            reasons.append("highconf_reflection")
        if mean_prob >= float(args.high_conf_threshold) and aspect >= float(args.aspect_ratio_max):
            reasons.append("highconf_aspect")
        if mean_prob >= float(args.high_conf_threshold) and center_bias >= float(args.center_bias_threshold):
            reasons.append("highconf_center_bias")

        score = 0.0
        score += 10.0 * float(is_fallback)
        score += max(0.0, float(args.mean_prob_threshold) - mean_prob) * 6.0
        score += max(0.0, bbox_area_ratio - float(args.bbox_area_max)) * 5.0
        score += max(0.0, float(args.bbox_area_min) - bbox_area_ratio) * 5.0
        score += max(0.0, reflection_overlap - float(args.reflection_high_threshold)) * 3.0
        if mean_prob >= float(args.high_conf_threshold):
            score += 0.8
        if "polypgen" in str(r.get("source", "")).lower():
            score += 0.2

        flagged.append(
            {
                "id": pid,
                "source": str(r.get("source", "")),
                "mean_prob": mean_prob,
                "bbox_area_ratio": bbox_area_ratio,
                "reflection_overlap": reflection_overlap,
                "aspect_ratio": float(aspect),
                "center_bias": float(center_bias),
                "is_fallback": is_fallback,
                "score": float(score),
                "reasons": ",".join(reasons),
            }
        )

    flagged.sort(key=lambda x: (-float(x["score"]), int(x["is_fallback"]), -float(x["mean_prob"])))
    max_review = max(1, int(args.max_review))
    poly_quota = max(0, int(args.polypgen_min_quota))

    selected = []
    selected_ids = set()
    if poly_quota > 0:
        for row in flagged:
            if len(selected) >= max_review:
                break
            src = str(row.get("source", "")).lower()
            if "polypgen" not in src:
                continue
            selected.append(row)
            selected_ids.add(str(row["id"]))
            if len(selected) >= poly_quota:
                break
    for row in flagged:
        if len(selected) >= max_review:
            break
        rid = str(row["id"])
        if rid in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(rid)

    ids = [str(x["id"]) for x in selected]
    out_txt = Path(args.output_txt) if args.output_txt else Path(args.proposal_csv).with_name("review_ids.txt")
    out_json = Path(args.output_json) if args.output_json else Path(args.proposal_csv).with_name("review_ids.json")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_txt.write_text("\n".join(ids), encoding="utf-8")
    out_json.write_text(json.dumps({"ids": ids, "rows": selected}, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "num_total": len(rows),
        "num_flagged": len(flagged),
        "num_selected": len(selected),
        "num_selected_polypgen": sum(1 for x in selected if "polypgen" in str(x.get("source", "")).lower()),
        "polypgen_min_quota": int(poly_quota),
        "output_txt": str(out_txt),
        "output_json": str(out_json),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
