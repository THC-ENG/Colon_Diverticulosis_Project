import argparse
import shutil
from pathlib import Path


def _resolve_under_workspace(path_text: str, workspace: Path) -> Path:
    p = Path(path_text)
    rp = p.resolve() if p.is_absolute() else (workspace / p).resolve()
    if rp == workspace or workspace in rp.parents:
        return rp
    raise RuntimeError(f"Refusing to operate outside workspace: {rp}")


def _remove_path(target: Path):
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def main():
    parser = argparse.ArgumentParser(description="Safely cleanup failed flywheel artifacts.")
    parser.add_argument(
        "--targets",
        type=str,
        default="runs/flywheel,runs/smoke_rescue",
        help="Comma-separated files/dirs to cleanup.",
    )
    parser.add_argument(
        "--with-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also remove known failed checkpoints (teacher/student flywheel).",
    )
    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Actually delete targets. Default is dry-run.",
    )
    args = parser.parse_args()

    workspace = Path.cwd().resolve()
    targets = [t.strip() for t in str(args.targets).split(",") if t.strip()]
    if args.with_checkpoints:
        targets.extend(
            [
                "checkpoints/teacher_r0.pth",
                "checkpoints/teacher_r1.pth",
                "checkpoints/student_flywheel_best.pth",
                "checkpoints/student_smoke_rescue_best.pth",
                "checkpoints/student_smoke_baselinecfg_best.pth",
            ]
        )

    resolved = []
    for t in targets:
        rp = _resolve_under_workspace(t, workspace=workspace)
        resolved.append((t, rp, rp.exists()))

    print("[cleanup] mode=execute" if args.execute else "[cleanup] mode=dry-run")
    for raw, rp, ok in resolved:
        state = "exists" if ok else "missing"
        print(f"  - {raw} -> {rp} [{state}]")

    if not args.execute:
        print("[cleanup] Dry-run only. Add --execute to perform deletion.")
        return

    removed = 0
    for _, rp, ok in resolved:
        if not ok:
            continue
        _remove_path(rp)
        removed += 1
    print(f"[cleanup] removed={removed}")


if __name__ == "__main__":
    main()
