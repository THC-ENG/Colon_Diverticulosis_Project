import argparse
import csv
import json
import shlex
import subprocess
import sys
from html import escape
from pathlib import Path


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _run(cmd: list[str]):
    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _path_to_uri(path_text: str) -> str:
    if not path_text:
        return ""
    try:
        return Path(path_text).resolve().as_uri()
    except Exception:
        return path_text


def _build_flywheel_gallery(round_quality_csvs: list[tuple[int, str]], output_html: str) -> str:
    rows = []
    for round_id, quality_csv in round_quality_csvs:
        p = Path(quality_csv)
        if not p.exists():
            continue
        for row in _read_csv(str(p)):
            item = dict(row)
            item["round_id"] = int(round_id)
            rows.append(item)

    if not rows:
        return ""

    rows.sort(key=lambda x: (int(x.get("round_id", 0)), -float(x.get("quality", 0.0))))
    cards = []
    for row in rows:
        rid = int(row.get("round_id", 0))
        pid = escape(str(row.get("id", "")))
        conf = float(row.get("conf", 0.0))
        edge_q = float(row.get("edge_quality", 0.0))
        quality = float(row.get("quality", 0.0))
        panel_path = str(row.get("panel_path", "")).strip()
        mask_path = str(row.get("hard_mask_path", "")).strip()
        image_path = str(row.get("image_path", "")).strip()
        preview_uri = _path_to_uri(panel_path if panel_path else mask_path)
        image_uri = _path_to_uri(image_path)
        mask_uri = _path_to_uri(mask_path)
        cards.append(
            (
                "<div class='card'>"
                f"<div class='hdr'>round {rid} | {pid}</div>"
                f"<img src='{preview_uri}' loading='lazy' alt='{pid}'/>"
                f"<div class='meta'>conf={conf:.4f} edge={edge_q:.4f} quality={quality:.4f}</div>"
                f"<div class='links'><a href='{image_uri}'>image</a> | <a href='{mask_uri}'>mask</a></div>"
                "</div>"
            )
        )

    html = (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<title>Flywheel Pseudo Masks (All Rounds)</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:16px;background:#f6f8fb;color:#111;}"
        "h1{margin:0 0 8px 0;font-size:22px;}"
        "p{margin:0 0 16px 0;color:#555;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px;}"
        ".card{background:#fff;border:1px solid #dde3ea;border-radius:8px;padding:10px;box-shadow:0 1px 2px rgba(0,0,0,.05);}"
        ".hdr{font-weight:700;font-size:13px;margin-bottom:8px;word-break:break-all;}"
        "img{width:100%;height:auto;border-radius:6px;border:1px solid #e5e9ef;background:#000;}"
        ".meta{margin-top:8px;font-size:12px;color:#333;}"
        ".links{margin-top:6px;font-size:12px;}"
        "a{color:#0b5ed7;text-decoration:none;}a:hover{text-decoration:underline;}"
        "</style></head><body>"
        f"<h1>Flywheel Pseudo Masks (All Rounds)</h1><p>Total samples: {len(rows)}</p>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>"
    )
    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


def _merge_teacher_manifest(base_manifest: str, selected_manifest: str, output: str):
    base_rows = _read_csv(base_manifest)
    sel_rows = _read_csv(selected_manifest)
    keep = [r for r in base_rows if str(r.get("subset", "")).strip() == "L_small"]
    merged = keep + sel_rows
    fields = [
        "id",
        "image_path",
        "mask_path",
        "subset",
        "split",
        "source",
        "center",
        "is_labeled",
        "is_pseudo",
        "pseudo_weight",
        "round_id",
        "exclude_from_tuning",
        "soft_path",
        "edge_path",
    ]
    _write_csv(output, merged, fields)


def _build_student_manifest(base_manifest: str, selected_manifests: list[str], output: str):
    base_rows = _read_csv(base_manifest)
    keep_base = [r for r in base_rows if str(r.get("subset", "")).strip() in {"L_small", "external"}]
    merged = keep_base
    for p in selected_manifests:
        if p and Path(p).exists():
            merged.extend(_read_csv(p))
    fields = [
        "id",
        "image_path",
        "mask_path",
        "subset",
        "split",
        "source",
        "center",
        "is_labeled",
        "is_pseudo",
        "pseudo_weight",
        "round_id",
        "exclude_from_tuning",
        "soft_path",
        "edge_path",
    ]
    _write_csv(output, merged, fields)


def _round_quality_guard(
    quality_csv: str,
    large_mask_threshold: float,
    max_large_mask_frac: float,
    enabled: bool,
):
    if not enabled:
        return
    p = Path(quality_csv)
    if not p.exists():
        return
    rows = _read_csv(str(p))
    if not rows:
        return
    if "area_ratio" not in rows[0]:
        print("[round quality guard] area_ratio column not found, skip guard for compatibility.")
        return
    area_vals = []
    for r in rows:
        try:
            area_vals.append(float(r.get("area_ratio", 0.0)))
        except Exception:
            area_vals.append(0.0)
    n = len(area_vals)
    large = sum(1 for a in area_vals if a > float(large_mask_threshold))
    frac = float(large) / float(max(1, n))
    mean_area = float(sum(area_vals)) / float(max(1, n))
    print(
        f"[round quality guard] n={n} mean_area={mean_area:.4f} "
        f"large(>{large_mask_threshold})={large} frac={frac:.4f}"
    )
    if frac > float(max_large_mask_frac):
        raise RuntimeError(
            "Round quality guard failed: too many overly-large masks. "
            "Please inspect round1 mask gallery and tighten pseudo settings."
        )


def main():
    parser = argparse.ArgumentParser(description="Run full teacher->pseudo->student flywheel.")
    parser.add_argument("--data-manifest", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--base-sam-checkpoint", type=str, required=True)
    parser.add_argument("--student-config", type=str, default="configs/student_joint_training.yaml")
    parser.add_argument("--flywheel-rounds", type=int, default=2)
    parser.add_argument("--round1-keep-quantile", type=float, default=0.35)
    parser.add_argument("--round2-keep-quantile", type=float, default=0.15)
    parser.add_argument("--quality-score", type=str, default="0.45*conf+0.25*edge_quality+0.20*area_prior+0.10*center_prior")
    parser.add_argument("--round1-proposal-json", type=str, default="")
    parser.add_argument("--round2-proposal-json", type=str, default="")
    parser.add_argument("--pseudo-auto-proposal-mode", type=str, default="multi_box", choices=["single_box", "multi_box", "grid_multi_box"])
    parser.add_argument("--pseudo-candidate-box-scales", type=str, default="1.0,0.85,0.7,0.55")
    parser.add_argument("--pseudo-candidate-box-centers", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--pseudo-max-candidate-boxes", type=int, default=40)
    parser.add_argument("--pseudo-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--pseudo-target-area-ratio", type=float, default=0.08)
    parser.add_argument("--pseudo-max-area-ratio", type=float, default=0.35)
    parser.add_argument("--enable-round1-quality-guard", type=str, default="true")
    parser.add_argument("--large-mask-threshold", type=float, default=0.40)
    parser.add_argument("--max-large-mask-frac", type=float, default=0.35)
    parser.add_argument("--teacher-refresh-between-rounds", type=str, default="true")
    parser.add_argument("--lora-num-workers", type=int, default=0)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--python-exec", type=str, default=sys.executable)
    args = parser.parse_args()
    if int(args.flywheel_rounds) != 2:
        raise ValueError("Current implementation supports exactly 2 flywheel rounds.")

    teacher_refresh = str(args.teacher_refresh_between_rounds).strip().lower() in {"1", "true", "yes", "y"}
    round1_guard = str(args.enable_round1_quality_guard).strip().lower() in {"1", "true", "yes", "y"}
    work = Path("runs/flywheel")
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    teacher_r0 = Path("checkpoints/teacher_r0.pth")
    teacher_r1 = Path("checkpoints/teacher_r1.pth")

    round1 = work / "round1"
    round2 = work / "round2"
    round1.mkdir(parents=True, exist_ok=True)
    round2.mkdir(parents=True, exist_ok=True)

    _run([
        args.python_exec,
        "medsam_tools/finetune_lora.py",
        "--checkpoint", args.base_sam_checkpoint,
        "--data-manifest", args.data_manifest,
        "--subset-filter", "L_small",
        "--split-filter", "train,val",
        "--num-workers", str(args.lora_num_workers),
        "--save-path", str(teacher_r0),
    ])

    round1_pseudo_cmd = [
        args.python_exec,
        "medsam_tools/generate_pseudo_labels.py",
        "--checkpoint", args.base_sam_checkpoint,
        "--lora-checkpoint", str(teacher_r0),
        "--data-manifest", args.data_manifest,
        "--subset-filter", "U_large",
        "--round-id", "1",
        "--output-root", str(round1 / "pseudo"),
        "--auto-proposal-mode", args.pseudo_auto_proposal_mode,
        "--candidate-box-scales", args.pseudo_candidate_box_scales,
        "--candidate-box-centers", args.pseudo_candidate_box_centers,
        "--max-candidate-boxes", str(args.pseudo_max_candidate_boxes),
        "--min-mask-area-ratio", str(args.pseudo_min_area_ratio),
        "--target-mask-area-ratio", str(args.pseudo_target_area_ratio),
        "--max-mask-area-ratio", str(args.pseudo_max_area_ratio),
    ]
    if args.round1_proposal_json:
        round1_pseudo_cmd.extend(["--proposal-json", args.round1_proposal_json])
    _run(round1_pseudo_cmd)
    _round_quality_guard(
        quality_csv=str(round1 / "pseudo" / "pseudo_quality.csv"),
        large_mask_threshold=float(args.large_mask_threshold),
        max_large_mask_frac=float(args.max_large_mask_frac),
        enabled=round1_guard,
    )
    _run([
        args.python_exec,
        "tools/filter_pseudo_labels.py",
        "--quality-csv", str(round1 / "pseudo" / "pseudo_quality.csv"),
        "--keep-quantile", str(args.round1_keep_quantile),
        "--quality-score", args.quality_score,
        "--round-id", "1",
        "--output-dir", str(round1 / "filter"),
        "--pseudo-candidates-manifest", str(round1 / "pseudo" / "pseudo_candidates_manifest.csv"),
        "--base-manifest", args.data_manifest,
    ])
    round1_selected_manifest = round1 / "filter" / "selected_manifest.csv"
    round1_remaining_manifest = round1 / "filter" / "remaining_u_large_manifest.csv"

    teacher_for_round2 = teacher_r0
    if teacher_refresh and round1_selected_manifest.exists():
        teacher_train_manifest = round1 / "teacher_round1_manifest.csv"
        _merge_teacher_manifest(args.data_manifest, str(round1_selected_manifest), str(teacher_train_manifest))
        _run([
            args.python_exec,
            "medsam_tools/finetune_lora.py",
            "--checkpoint", args.base_sam_checkpoint,
            "--data-manifest", str(teacher_train_manifest),
            "--subset-filter", "L_small,pseudo_round1",
            "--split-filter", "train,val",
            "--num-workers", str(args.lora_num_workers),
            "--init-lora-checkpoint", str(teacher_r0),
            "--save-path", str(teacher_r1),
        ])
        teacher_for_round2 = teacher_r1

    manifest_for_round2 = str(round1_remaining_manifest) if round1_remaining_manifest.exists() else args.data_manifest
    round2_pseudo_cmd = [
        args.python_exec,
        "medsam_tools/generate_pseudo_labels.py",
        "--checkpoint", args.base_sam_checkpoint,
        "--lora-checkpoint", str(teacher_for_round2),
        "--data-manifest", manifest_for_round2,
        "--subset-filter", "U_large",
        "--round-id", "2",
        "--output-root", str(round2 / "pseudo"),
        "--auto-proposal-mode", args.pseudo_auto_proposal_mode,
        "--candidate-box-scales", args.pseudo_candidate_box_scales,
        "--candidate-box-centers", args.pseudo_candidate_box_centers,
        "--max-candidate-boxes", str(args.pseudo_max_candidate_boxes),
        "--min-mask-area-ratio", str(args.pseudo_min_area_ratio),
        "--target-mask-area-ratio", str(args.pseudo_target_area_ratio),
        "--max-mask-area-ratio", str(args.pseudo_max_area_ratio),
    ]
    if args.round2_proposal_json:
        round2_pseudo_cmd.extend(["--proposal-json", args.round2_proposal_json])
    _run(round2_pseudo_cmd)
    _run([
        args.python_exec,
        "tools/filter_pseudo_labels.py",
        "--quality-csv", str(round2 / "pseudo" / "pseudo_quality.csv"),
        "--keep-quantile", str(args.round2_keep_quantile),
        "--quality-score", args.quality_score,
        "--round-id", "2",
        "--output-dir", str(round2 / "filter"),
        "--pseudo-candidates-manifest", str(round2 / "pseudo" / "pseudo_candidates_manifest.csv"),
        "--base-manifest", manifest_for_round2,
    ])
    round2_selected_manifest = round2 / "filter" / "selected_manifest.csv"

    flywheel_gallery = _build_flywheel_gallery(
        round_quality_csvs=[
            (1, str(round1 / "pseudo" / "pseudo_quality.csv")),
            (2, str(round2 / "pseudo" / "pseudo_quality.csv")),
        ],
        output_html=str(work / "all_pseudo_masks_gallery.html"),
    )

    student_manifest = work / "student_manifest.csv"
    _build_student_manifest(
        base_manifest=args.data_manifest,
        selected_manifests=[str(round1_selected_manifest), str(round2_selected_manifest)],
        output=str(student_manifest),
    )

    _run([
        args.python_exec,
        "train.py",
        "--config", args.student_config,
        "--mode", "student_with_pseudo_distill",
        "--data-manifest", str(student_manifest),
        "--manifest-mode", "only",
        "--num-workers", str(args.train_num_workers),
    ])

    summary = {
        "teacher_r0": str(teacher_r0),
        "teacher_r1": str(teacher_r1) if teacher_refresh else str(teacher_r0),
        "round1_selected_manifest": str(round1_selected_manifest),
        "round2_selected_manifest": str(round2_selected_manifest),
        "round1_hard_masks_dir": str(round1 / "pseudo" / "hard_masks"),
        "round2_hard_masks_dir": str(round2 / "pseudo" / "hard_masks"),
        "flywheel_mask_gallery_html": flywheel_gallery,
        "student_manifest": str(student_manifest),
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()



















