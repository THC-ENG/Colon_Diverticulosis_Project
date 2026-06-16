import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["CVC-300", "CVC-ColonDB", "ETIS", "PolypGen"]
METRICS = [
    ("Dice", "dice_mean"),
    ("IoU", "iou_mean"),
    ("Precision", "precision_mean"),
    ("Recall", "recall_mean"),
    ("MAE", "mae_mean"),
    ("Boundary-F1", "boundary_f1_mean"),
    ("HD95", "hd95_mean"),
]

EXPERIMENT_GROUPS = [
    {
        "title": "SANet Data Flywheel Ablation",
        "out_dir": "results/sanet_data_ablation",
        "stem": "sanet_data_ablation",
        "reference": "SANet L_small only",
        "experiments": [
            ("SANet L_small only", "results/sanet_data_ablation/lsmall_only/report.json"),
            ("SANet L_small + R1 + R2", "results/external_benchmark/sanet/report.json"),
        ],
    },
    {
        "title": "Student Pseudo-label Filtering Ablation",
        "out_dir": "results/pseudo_filter_ablation",
        "stem": "pseudo_filter_ablation",
        "reference": "Student unfiltered/weak-filtered R1+R2",
        "experiments": [
            ("Student unfiltered/weak-filtered R1+R2", "results/pseudo_filter_ablation/unfiltered_student_r1r2/report.json"),
            ("Student strict filtered + SDF smooth", "results/external_benchmark/ours_full_pipeline/report.json"),
        ],
    },
]


def _load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _group(report: dict, dataset: str) -> dict:
    grouped = report["grouped_metrics"]
    if dataset in grouped:
        return grouped[dataset]
    for key, value in grouped.items():
        if key == dataset or key.startswith(dataset + "|"):
            return value
    raise KeyError(dataset)


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            })


def _write_md(path: Path, rows: list[dict], fields: list[str]):
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_overall(rows: list[dict], title: str, out_path: Path):
    metrics = ["Dice", "IoU", "Precision", "Recall", "MAE", "Boundary-F1"]
    labels = [row["Experiment"] for row in rows]
    x = np.arange(len(metrics))
    width = 0.34
    colors = ["#8FB8DE", "#EE8F76"]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    for idx, row in enumerate(rows):
        values = [row[m] for m in metrics]
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=labels[idx], color=colors[idx])

    ax.set_title(f"{title}: Overall Metrics")
    ax.set_ylabel("Mean score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_dataset_dice_iou(rows: list[dict], labels: list[str], title: str, out_path: Path):
    colors = ["#8FB8DE", "#EE8F76"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=180, sharex=True)
    x = np.arange(len(DATASETS))
    width = 0.34

    for ax, metric in zip(axes, ["Dice", "IoU"]):
        for idx, label in enumerate(labels):
            values = [
                next(r[metric] for r in rows if r["Experiment"] == label and r["Dataset"] == ds)
                for ds in DATASETS
            ]
            ax.bar(x + (idx - 0.5) * width, values, width=width, label=label, color=colors[idx])
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_title(f"{title}: External Dataset Dice/IoU")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(DATASETS)
    axes[0].legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _build_group(config: dict):
    out_dir = Path(config["out_dir"])
    reports = [(name, _load_report(path)) for name, path in config["experiments"]]

    overall_rows = []
    for name, report in reports:
        row = {"Experiment": name, "N": report.get("num_evaluated_samples", report.get("num_samples", ""))}
        for label, key in METRICS:
            row[label] = float(report[key])
        overall_rows.append(row)

    by_dataset_rows = []
    for name, report in reports:
        for dataset in DATASETS:
            group = _group(report, dataset)
            row = {"Experiment": name, "Dataset": dataset, "N": group.get("n", "")}
            for label, key in METRICS:
                row[label] = float(group[key])
            by_dataset_rows.append(row)

    ref_row = next(row for row in overall_rows if row["Experiment"] == config["reference"])
    delta_rows = []
    for row in overall_rows:
        if row["Experiment"] == config["reference"]:
            continue
        delta = {"Experiment": row["Experiment"], "Reference": config["reference"]}
        for label, _key in METRICS:
            delta[f"Delta {label}"] = row[label] - ref_row[label]
        delta_rows.append(delta)

    overall_fields = ["Experiment", "N"] + [label for label, _key in METRICS]
    by_dataset_fields = ["Experiment", "Dataset", "N"] + [label for label, _key in METRICS]
    delta_fields = ["Experiment", "Reference"] + [f"Delta {label}" for label, _key in METRICS]
    stem = config["stem"]

    _write_csv(out_dir / f"{stem}_overall.csv", overall_rows, overall_fields)
    _write_md(out_dir / f"{stem}_overall.md", overall_rows, overall_fields)
    _write_csv(out_dir / f"{stem}_by_dataset.csv", by_dataset_rows, by_dataset_fields)
    _write_md(out_dir / f"{stem}_by_dataset.md", by_dataset_rows, by_dataset_fields)
    _write_csv(out_dir / f"{stem}_delta.csv", delta_rows, delta_fields)
    _write_md(out_dir / f"{stem}_delta.md", delta_rows, delta_fields)
    _plot_overall(overall_rows, config["title"], out_dir / f"{stem}_overall_metrics.png")
    _plot_dataset_dice_iou(
        by_dataset_rows,
        [name for name, _path in config["experiments"]],
        config["title"],
        out_dir / f"{stem}_dataset_dice_iou.png",
    )

    print(f"[saved] {out_dir / f'{stem}_overall.md'}")
    print(f"[saved] {out_dir / f'{stem}_by_dataset.md'}")
    print(f"[saved] {out_dir / f'{stem}_delta.md'}")
    print(f"[saved] {out_dir / f'{stem}_overall_metrics.png'}")
    print(f"[saved] {out_dir / f'{stem}_dataset_dice_iou.png'}")


def main():
    for config in EXPERIMENT_GROUPS:
        _build_group(config)


if __name__ == "__main__":
    main()
