import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT_DIR / '.deps'
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))


def _import_deps():
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required. Install with: python -m pip install matplotlib"
        ) from e

    try:
        from tensorboard.backend.event_processing import event_accumulator as ea
    except Exception as e:
        raise RuntimeError(
            "tensorboard is required. Install with: python -m pip install tensorboard"
        ) from e
    return plt, ea


def _dedupe_by_step(events):
    entries = [(int(e.step), float(e.wall_time), float(e.value)) for e in events]
    entries.sort(key=lambda x: (x[0], x[1]))
    latest = {}
    for step, _wall_time, value in entries:
        latest[step] = value
    steps = sorted(latest.keys())
    values = [latest[s] for s in steps]
    return steps, values


def _ema(values, alpha):
    if alpha <= 0.0:
        return list(values)
    out = []
    prev = None
    for v in values:
        if prev is None:
            prev = float(v)
        else:
            prev = alpha * float(v) + (1.0 - alpha) * prev
        out.append(prev)
    return out


def _load_scalar_series(run_dir: Path, tag: str, ea):
    acc = ea.EventAccumulator(str(run_dir), size_guidance={ea.SCALARS: 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if tag not in tags:
        return [], [], tags
    events = acc.Scalars(tag)
    steps, values = _dedupe_by_step(events)
    return steps, values, tags


def _write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Plot four loss components (sup/pseudo/edge/distill) from TensorBoard logs."
    )
    parser.add_argument("--run-dir", type=str, default="runs/student_flywheel")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--csv-output", type=str, default="")
    parser.add_argument("--smoothing", type=float, default=0.0, help="EMA alpha in [0,1). 0 means no smoothing.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-val", action="store_true", help="Hide validation curves.")
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    if not (0.0 <= float(args.smoothing) < 1.0):
        raise ValueError("--smoothing must be in [0, 1).")

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    out_path = Path(args.output) if args.output else Path("results/loss_curves") / f"{run_dir.name}_four_losses.png"
    csv_path = Path(args.csv_output) if args.csv_output else out_path.with_suffix(".csv")

    plt, ea = _import_deps()

    components = [
        ("sup", "loss_sup", "L_sup"),
        ("pseudo", "loss_pseudo", "L_pseudo"),
        ("edge", "loss_edge", "L_edge"),
        ("distill", "loss_distill", "L_distill"),
    ]

    all_tags = []
    series = {}
    for key, tag_suffix, _label in components:
        tr_tag = f"Loss/train_{tag_suffix}"
        va_tag = f"Loss/val_{tag_suffix}"
        tr_steps, tr_vals, tags = _load_scalar_series(run_dir, tr_tag, ea)
        va_steps, va_vals, _ = _load_scalar_series(run_dir, va_tag, ea)
        all_tags = tags
        series[key] = {
            "train_steps": tr_steps,
            "train_vals": _ema(tr_vals, float(args.smoothing)),
            "val_steps": va_steps,
            "val_vals": _ema(va_vals, float(args.smoothing)),
            "train_tag": tr_tag,
            "val_tag": va_tag,
        }

    if not any(len(series[k]["train_steps"]) > 0 or len(series[k]["val_steps"]) > 0 for k, _, _ in components):
        raise RuntimeError(
            "No matching scalar tags found. Expected tags like "
            "'Loss/train_loss_sup'. Available tags:\n" + "\n".join(sorted(all_tags))
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.flatten()
    for i, (key, _suffix, label) in enumerate(components):
        ax = axes[i]
        tr_steps = series[key]["train_steps"]
        tr_vals = series[key]["train_vals"]
        va_steps = series[key]["val_steps"]
        va_vals = series[key]["val_vals"]

        if tr_steps:
            ax.plot(tr_steps, tr_vals, label="train", linewidth=2.0)
        if (not args.no_val) and va_steps:
            ax.plot(va_steps, va_vals, label="val", linewidth=2.0, linestyle="--")

        if not tr_steps and (args.no_val or not va_steps):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend()

    title = args.title.strip() if args.title else f"Loss Components: {run_dir.name}"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)

    all_steps = sorted(
        {
            s
            for key, _, _ in components
            for s in (series[key]["train_steps"] + series[key]["val_steps"])
        }
    )
    rows = []
    for step in all_steps:
        row = {"epoch": step}
        for key, suffix, _label in components:
            tr_map = dict(zip(series[key]["train_steps"], series[key]["train_vals"]))
            va_map = dict(zip(series[key]["val_steps"], series[key]["val_vals"]))
            row[f"train_{suffix}"] = tr_map.get(step, "")
            row[f"val_{suffix}"] = va_map.get(step, "")
        rows.append(row)
    columns = ["epoch"] + [f"train_{c[1]}" for c in components] + [f"val_{c[1]}" for c in components]
    _write_csv(csv_path, rows, columns)

    print(f"[saved figure] {out_path}")
    print(f"[saved csv] {csv_path}")


if __name__ == "__main__":
    main()

