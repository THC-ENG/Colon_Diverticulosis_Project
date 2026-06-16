import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ENTRIES = [
    ("Ours", "results/external_benchmark/ours_full_pipeline/report.json"),
    ("U-Net", "results/external_benchmark/unet/report.json"),
    ("U-Net++", "results/external_benchmark/unetpp/report.json"),
    ("PraNet", "results/external_benchmark/pranet/report.json"),
    ("TransUNet", "results/external_benchmark/transunet/report.json"),
    ("SANet", "results/external_benchmark/sanet/report.json"),
    ("SAM-ViT-B(box)", "results/external_benchmark/sam_vit_b_box/report.json"),
    ("MobileSAM(box)", "results/external_benchmark/mobile_sam_box/report.json"),
]

DATASETS = ["CVC-300", "CVC-ColonDB", "ETIS", "PolypGen"]


def _load(path: str) -> dict:
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


def _plot_overall(reports: list[tuple[str, dict]], out_path: Path):
    labels = [name for name, _report in reports]
    dice = [report["dice_mean"] for _name, report in reports]
    iou = [report["iou_mean"] for _name, report in reports]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=180)
    ax.bar(x - width / 2, dice, width, label="Dice", color="#7BC6A4")
    ax.bar(x + width / 2, iou, width, label="IoU", color="#8FB8DE")
    ax.set_title("External Benchmark: Overall Dice / IoU")
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Mean score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for i, (d, j) in enumerate(zip(dice, iou)):
        ax.text(i - width / 2, d + 0.012, f"{d:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, j + 0.012, f"{j:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_heatmap(reports: list[tuple[str, dict]], metric_key: str, title: str, out_path: Path):
    labels = [name for name, _report in reports]
    data = np.array([
        [_group(report, dataset)[metric_key] for dataset in DATASETS]
        for _name, report in reports
    ], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8.5, 6.4), dpi=180)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value > 0.62 else "black"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    out_dir = Path("results/external_benchmark")
    reports = [(name, _load(path)) for name, path in ENTRIES]
    _plot_overall(reports, out_dir / "external_benchmark_overall_dice_iou.png")
    _plot_heatmap(reports, "dice_mean", "External Benchmark: Dice by Dataset", out_dir / "external_benchmark_dataset_dice_heatmap.png")
    _plot_heatmap(reports, "iou_mean", "External Benchmark: IoU by Dataset", out_dir / "external_benchmark_dataset_iou_heatmap.png")
    print(f"[saved] {out_dir / 'external_benchmark_overall_dice_iou.png'}")
    print(f"[saved] {out_dir / 'external_benchmark_dataset_dice_heatmap.png'}")
    print(f"[saved] {out_dir / 'external_benchmark_dataset_iou_heatmap.png'}")


if __name__ == "__main__":
    main()
