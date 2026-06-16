import argparse
import csv
import json
from pathlib import Path


def _parse_entry(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Invalid entry: {text}. Expected Model=path/to/report.json")
    model, path = text.split("=", 1)
    model = model.strip()
    report = Path(path.strip())
    if not model or not str(report):
        raise ValueError(f"Invalid entry: {text}")
    return model, report


def _extract_source_from_group_key(key: str) -> str:
    # Compatible with inference_eval grouped key format: "source|subset|rX"
    return key.split("|", 1)[0].strip()


def _collect_source_metrics(report: dict, metrics: list[str]) -> dict:
    grouped = report.get("grouped_metrics", {}) or {}
    bucket = {}

    for k, v in grouped.items():
        source = _extract_source_from_group_key(str(k))
        n = int(v.get("n", 0))

        if source not in bucket:
            bucket[source] = {"n": 0}
            for metric in metrics:
                bucket[source][f"{metric}_weighted_sum"] = 0.0
        bucket[source]["n"] += n
        for metric in metrics:
            bucket[source][f"{metric}_weighted_sum"] += float(v.get(f"{metric}_mean", 0.0)) * n

    out = {}
    for source, s in bucket.items():
        if s["n"] <= 0:
            continue
        out[source] = {"n": int(s["n"])}
        for metric in metrics:
            out[source][f"{metric}_mean"] = float(s[f"{metric}_weighted_sum"] / s["n"])
    return out


def _metric_label(metric: str) -> str:
    if metric == "iou":
        return "IoU"
    if metric in {"hd95", "mae"}:
        return metric.upper()
    return metric.replace("_", " ").title()


def _to_markdown(rows: list[dict], sources: list[str], metrics: list[str]) -> str:
    header = ["Model"]
    for metric in metrics:
        header.append(f"Overall {_metric_label(metric)}")
    for source in sources:
        for metric in metrics:
            header.append(f"{source} {_metric_label(metric)}")

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in rows:
        line = [row["model"]]
        for metric in metrics:
            line.append(f"{row.get(f'overall_{metric}', 0.0):.4f}")
        for source in sources:
            for metric in metrics:
                line.append(f"{row.get(f'{source}_{metric}', 0.0):.4f}")
        lines.append("| " + " | ".join(line) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Build external benchmark table from report JSON files.")
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Entry in format Model=path/to/report.json (repeatable)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="dice,iou",
        help="Comma-separated metrics to include.",
    )
    parser.add_argument("--output-csv", type=str, default="results/external_benchmark/benchmark_table.csv")
    parser.add_argument("--output-md", type=str, default="results/external_benchmark/benchmark_table.md")
    args = parser.parse_args()

    entries = [_parse_entry(x) for x in args.entry]
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    if not metrics:
        raise ValueError("--metrics must include at least one metric")
    parsed_rows = []
    all_sources = set()

    for model, report_path in entries:
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found: {report_path}")

        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        source_metrics = _collect_source_metrics(report, metrics)
        all_sources.update(source_metrics.keys())

        row = {"model": model}
        for metric in metrics:
            row[f"overall_{metric}"] = float(report.get(f"{metric}_mean", 0.0))
        for source, vals in source_metrics.items():
            for metric in metrics:
                row[f"{source}_{metric}"] = float(vals[f"{metric}_mean"])
            row[f"{source}_n"] = int(vals["n"])
        parsed_rows.append(row)

    sources = sorted(all_sources)
    csv_fields = ["model"]
    for metric in metrics:
        csv_fields.append(f"overall_{metric}")
    for source in sources:
        for metric in metrics:
            csv_fields.append(f"{source}_{metric}")
        csv_fields.append(f"{source}_n")

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in parsed_rows:
            writer.writerow(row)

    md = _to_markdown(parsed_rows, sources, metrics)
    with output_md.open("w", encoding="utf-8") as f:
        f.write(md)

    print(f"[saved] {output_csv}")
    print(f"[saved] {output_md}")
    print("\n" + md)


if __name__ == "__main__":
    main()
