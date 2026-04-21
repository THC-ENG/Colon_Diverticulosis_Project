import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _run_if_outputs_missing(step_name: str, done_paths: list[Path], cmd: list[str]):
    if done_paths and all(Path(p).exists() for p in done_paths):
        print(f"[skip] {step_name} (outputs exist)")
        return
    _run(cmd)


def _read_checkpoint_epoch(checkpoint_path: Path) -> int | None:
    p = Path(checkpoint_path)
    if not p.exists():
        return None
    try:
        import torch  # local import to avoid hard dependency for non-training flows
        ckpt = torch.load(str(p), map_location="cpu")
    except Exception as exc:
        print(f"[checkpoint inspect] failed to read epoch from {p}: {exc}")
        return None
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        try:
            return int(ckpt.get("epoch", 0))
        except Exception:
            return None
    return None


def _is_lora_checkpoint_complete(checkpoint_path: Path, expected_epochs: int) -> bool:
    p = Path(checkpoint_path)
    if not p.exists():
        return False
    ep = _read_checkpoint_epoch(p)
    if ep is None:
        # Keep backward compatibility for checkpoints without epoch metadata.
        print(f"[checkpoint inspect] no epoch metadata in {p}, treat as complete for compatibility.")
        return True
    ok = int(ep) >= int(expected_epochs)
    if not ok:
        print(
            f"[checkpoint inspect] incomplete checkpoint: {p} "
            f"(epoch={ep} < expected={int(expected_epochs)}), will rerun this stage."
        )
    return ok


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
        conf = _to_float(row.get("conf", 0.0), default=0.0)
        edge_q = _to_float(row.get("edge_quality", 0.0), default=0.0)
        quality = _to_float(row.get("quality", 0.0), default=0.0)
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


def _merge_teacher_manifest(base_manifest: str, selected_manifest: str, output: str, keep_subsets: set[str] | None = None):
    base_rows = _read_csv(base_manifest)
    sel_rows = _read_csv(selected_manifest)
    keep_set = keep_subsets if keep_subsets is not None else {"L_small"}
    keep = [r for r in base_rows if str(r.get("subset", "")).strip() in keep_set]
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
        "tier",
        "score_source_percentile",
        "pseudo_weight_raw",
        "pseudo_weight_final",
    ]
    _write_csv(output, merged, fields)


def _build_student_manifest(base_manifest: str, selected_manifests: list[str], output: str):
    base_rows = _read_csv(base_manifest)
    keep_base = [r for r in base_rows if str(r.get("subset", "")).strip() in {"L_small", "L_adapt_polypgen", "external"}]
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
        "tier",
        "score_source_percentile",
        "pseudo_weight_raw",
        "pseudo_weight_final",
    ]
    _write_csv(output, merged, fields)


def _manual_pass_ids(mask_review_csv: str) -> set[str]:
    p = Path(mask_review_csv)
    if not p.exists():
        return set()
    ids: set[str] = set()
    for r in _read_csv(str(p)):
        pid = str(r.get("id", "")).strip()
        if not pid:
            continue
        decision = str(r.get("decision", "")).strip().lower()
        if not decision:
            decision = "pass"
        if decision in {"pass", "keep", "accept", "approved"}:
            ids.add(pid)
    return ids


def _filter_manifest_by_ids(input_manifest: str, keep_ids: set[str], output_manifest: str) -> int:
    rows = _read_csv(input_manifest)
    if not rows:
        _write_csv(output_manifest, [], [])
        return 0
    fields = list(rows[0].keys())
    out = [r for r in rows if str(r.get("id", "")).strip() in keep_ids]
    _write_csv(output_manifest, out, fields)
    return int(len(out))


def _review_non_reject_ids(review_csv: str) -> set[str]:
    p = Path(review_csv)
    if not p.exists():
        return set()
    ids: set[str] = set()
    for r in _read_csv(str(p)):
        pid = str(r.get("id", "")).strip()
        if not pid:
            continue
        decision = str(r.get("decision", "")).strip().lower()
        if not decision:
            continue
        if decision in {"reject", "drop", "discard", "bad"}:
            continue
        ids.add(pid)
    return ids


def _augment_manifest_with_ids(base_manifest: str, source_manifest: str, add_ids: set[str], output_manifest: str) -> tuple[int, int]:
    base_rows = _read_csv(base_manifest) if Path(base_manifest).exists() else []
    src_rows = _read_csv(source_manifest) if Path(source_manifest).exists() else []
    if not base_rows and not src_rows:
        _write_csv(output_manifest, [], [])
        return 0, 0

    fields = list(base_rows[0].keys()) if base_rows else list(src_rows[0].keys())
    out = list(base_rows)
    seen = {str(r.get("id", "")).strip() for r in out}
    added = 0
    for r in src_rows:
        pid = str(r.get("id", "")).strip()
        if not pid or pid in seen:
            continue
        if pid not in add_ids:
            continue
        out.append(r)
        seen.add(pid)
        added += 1
    _write_csv(output_manifest, out, fields)
    return int(len(out)), int(added)


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


def _pseudo_artifact_guard(quality_csv: str, hard_masks_dir: str, enabled: bool = True):
    if not enabled:
        return
    qpath = Path(quality_csv)
    mdir = Path(hard_masks_dir)
    if not qpath.exists() or not mdir.exists():
        return
    rows = _read_csv(str(qpath))
    n_quality = int(len(rows))
    n_masks = int(sum(1 for p in mdir.glob("*.png") if p.is_file()))
    print(f"[pseudo artifact guard] quality_rows={n_quality} hard_masks={n_masks}")
    if n_quality != n_masks:
        raise RuntimeError(
            "Pseudo artifact guard failed: pseudo_quality row count mismatches hard mask files. "
            "This usually indicates stale masks or partial resume outputs. "
            "Please rerun pseudo generation to rebuild metadata and gallery."
        )


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _bool_text(v, default: bool = False) -> bool:
    t = str(v or "").strip().lower()
    if t in {"1", "true", "yes", "y", "on"}:
        return True
    if t in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _resolve_manual_csv(path_text: str, fallback_path: Path) -> Path:
    if str(path_text or "").strip():
        return Path(path_text)
    return fallback_path


def _pick_stratified(rows: list[dict], target: int, preferred_ids: list[str]) -> list[dict]:
    target = int(max(0, target))
    if target <= 0:
        return []
    if not rows:
        return []

    by_id = {}
    ordered = []
    for r in rows:
        pid = str(r.get("id", "")).strip()
        if not pid or pid in by_id:
            continue
        by_id[pid] = r
        ordered.append(r)

    selected = []
    selected_ids = set()
    for pid in preferred_ids:
        if len(selected) >= target:
            break
        if pid in by_id and pid not in selected_ids:
            selected.append(by_id[pid])
            selected_ids.add(pid)

    remain = [r for r in ordered if str(r.get("id", "")).strip() not in selected_ids]
    groups = {}
    for r in remain:
        src = str(r.get("source", "")).strip() or "unknown"
        groups.setdefault(src, []).append(r)

    while len(selected) < target:
        non_empty = [(k, v) for k, v in groups.items() if v]
        if not non_empty:
            break
        non_empty.sort(key=lambda kv: len(kv[1]), reverse=True)
        for _, lst in non_empty:
            if len(selected) >= target:
                break
            selected.append(lst.pop(0))
    return selected


def _require_manual_csv(manual_csv: Path, template_csv: Path, stage_desc: str):
    if manual_csv.exists():
        return
    raise RuntimeError(
        f"Missing manual review file for {stage_desc}: {manual_csv}\n"
        f"Template prepared at: {template_csv}\n"
        "Please fill the CSV and rerun."
    )


def _check_round_qa_gate(summary_path: Path, pass_rate_min: float, polypgen_pass_rate_min: float, boundary_bad_max: float, round_id: int):
    if not summary_path.exists():
        raise RuntimeError(f"Round{round_id} QA summary not found: {summary_path}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    n_total = int(data.get("num_manual_reviewed_total", 0))
    overall_pass_rate = float(data.get("overall_pass_rate", 0.0))
    polypgen_pass_rate = float(data.get("polypgen_pass_rate", 1.0))
    boundary_bad_ratio = float(data.get("boundary_bad_ratio", 0.0))
    if n_total <= 0:
        raise RuntimeError(f"Round{round_id} QA gate failed: no manual reviews recorded.")
    if overall_pass_rate < float(pass_rate_min):
        raise RuntimeError(
            f"Round{round_id} QA gate failed: overall_pass_rate={overall_pass_rate:.4f} < {pass_rate_min:.4f}"
        )
    if polypgen_pass_rate < float(polypgen_pass_rate_min):
        raise RuntimeError(
            f"Round{round_id} QA gate failed: polypgen_pass_rate={polypgen_pass_rate:.4f} < {polypgen_pass_rate_min:.4f}"
        )
    if boundary_bad_ratio > float(boundary_bad_max):
        raise RuntimeError(
            f"Round{round_id} QA gate failed: boundary_bad_ratio={boundary_bad_ratio:.4f} > {boundary_bad_max:.4f}"
        )


def _bbox_from_mask(mask_path: str) -> list[float]:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV (cv2) and numpy are required for LoRA QC but not installed in the current environment.")
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read mask for box proposal: {mask_path}")
    fg = m > 127
    h, w = m.shape[:2]
    if int(fg.sum()) == 0:
        return [0.0, 0.0, float(max(0, w - 1)), float(max(0, h - 1))]
    ys, xs = np.where(fg)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _resolve_path(raw_path: str, manifest_path: str) -> str:
    raw = str(raw_path or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p.resolve())
    m = Path(manifest_path)
    cands = [(m.parent / p), (m.parent.parent / p), (Path.cwd() / p)]
    for c in cands:
        if c.exists():
            return str(c.resolve())
    return str(cands[0].resolve())


def _dice_from_masks(pred, gt) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = float((pred & gt).sum())
    denom = float(pred.sum() + gt.sum())
    if denom == 0.0:
        return 1.0
    return float((2.0 * inter) / max(1e-6, denom))


def _run_lora_qc(
    python_exec: str,
    work: Path,
    data_manifest: str,
    base_sam_checkpoint: str,
    teacher_checkpoint: str,
    dice_min: float,
    bf1_min: float,
    subset_filter: str,
    polypgen_dice_min: float,
    polypgen_bf1_min: float,
    worst_k: int,
):
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV (cv2) and numpy are required for --skip-lora-qc=false but are not installed.")
    try:
        from utils.metrics import boundary_f1_from_masks
    except Exception as e:
        raise RuntimeError(f"LoRA QC requires utils.metrics dependencies (cv2/numpy/torch): {e}") from e

    qc_root = work / "lora_qc"
    qc_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_csv(data_manifest)
    qc_subsets = {s.strip() for s in str(subset_filter or "").split(",") if s.strip()}
    if not qc_subsets:
        qc_subsets = {"L_small"}
    val_rows = [
        r
        for r in manifest_rows
        if str(r.get("subset", "")).strip() in qc_subsets
        and str(r.get("split", "")).strip() == "val"
        and str(r.get("mask_path", "")).strip()
    ]
    if not val_rows:
        val_rows = [
            r
            for r in manifest_rows
            if str(r.get("subset", "")).strip() in qc_subsets
            and str(r.get("mask_path", "")).strip()
        ]
    if not val_rows:
        raise RuntimeError(f"LoRA QC failed: no val rows found for subsets={sorted(qc_subsets)} in manifest.")

    id_filter = qc_root / "qc_val_ids.txt"
    proposal_json = qc_root / "qc_val_gt_boxes.json"
    id_to_gt = {}
    proposals = {}
    for r in val_rows:
        pid = str(r.get("id", "")).strip()
        mpath = _resolve_path(str(r.get("mask_path", "")).strip(), data_manifest)
        if not pid or not mpath or not Path(mpath).exists():
            continue
        id_to_gt[pid] = mpath
        proposals[pid] = _bbox_from_mask(mpath)
    if not id_to_gt:
        raise RuntimeError("LoRA QC failed: no valid val IDs with masks.")
    id_filter.write_text("\n".join(sorted(id_to_gt.keys())), encoding="utf-8")
    proposal_json.write_text(json.dumps(proposals, indent=2, ensure_ascii=False), encoding="utf-8")

    pseudo_root = qc_root / "pseudo_val"
    _run(
        [
            python_exec,
            "medsam_tools/generate_pseudo_labels.py",
            "--checkpoint",
            base_sam_checkpoint,
            "--lora-checkpoint",
            teacher_checkpoint,
            "--data-manifest",
            data_manifest,
            "--subset-filter",
            ",".join(sorted(qc_subsets)),
            "--id-filter",
            str(id_filter),
            "--proposal-json",
            str(proposal_json),
            "--no-fallback-full-image",
            "--round-id",
            "0",
            "--output-root",
            str(pseudo_root),
        ]
    )

    quality_rows = _read_csv(str(pseudo_root / "pseudo_quality.csv"))
    if not quality_rows:
        raise RuntimeError("LoRA QC failed: pseudo_quality.csv is empty.")

    per_sample = []
    for r in quality_rows:
        pid = str(r.get("id", "")).strip()
        pred_path = str(r.get("hard_mask_path", "")).strip()
        gt_path = id_to_gt.get(pid, "")
        if not pid or not pred_path or not gt_path:
            continue
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue
        pred_bin = (pred > 127).astype(np.uint8)
        gt_bin = (gt > 127).astype(np.uint8)
        if pred_bin.shape != gt_bin.shape:
            pred_bin = cv2.resize(pred_bin, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred_bin = (pred_bin > 0).astype(np.uint8)
        dice = _dice_from_masks(pred_bin, gt_bin)
        bf1 = float(boundary_f1_from_masks(pred_bin, gt_bin, boundary_radius=1))
        per_sample.append(
            {
                "id": pid,
                "dice": dice,
                "boundary_f1": bf1,
                "source": str(r.get("source", "")),
                "panel_path": str(r.get("panel_path", "")),
            }
        )
    if not per_sample:
        raise RuntimeError("LoRA QC failed: no sample metrics were computed.")

    per_sample.sort(key=lambda x: x["dice"])
    dice_mean = float(np.mean([x["dice"] for x in per_sample]))
    bf1_mean = float(np.mean([x["boundary_f1"] for x in per_sample]))
    source_summary = {}
    for row in per_sample:
        src = str(row.get("source", "")).strip() or "unknown"
        source_summary.setdefault(src, {"dice": [], "bf1": []})
        source_summary[src]["dice"].append(float(row["dice"]))
        source_summary[src]["bf1"].append(float(row["boundary_f1"]))
    by_source = {}
    for src, v in source_summary.items():
        by_source[src] = {
            "count": int(len(v["dice"])),
            "dice_mean": float(np.mean(v["dice"])) if v["dice"] else 0.0,
            "boundary_f1_mean": float(np.mean(v["bf1"])) if v["bf1"] else 0.0,
        }
    polypgen_rows = [x for x in per_sample if "polypgen" in str(x.get("source", "")).lower()]
    polypgen_dice_mean = float(np.mean([x["dice"] for x in polypgen_rows])) if polypgen_rows else 0.0
    polypgen_bf1_mean = float(np.mean([x["boundary_f1"] for x in polypgen_rows])) if polypgen_rows else 0.0
    _write_csv(
        str(qc_root / "per_sample.csv"),
        per_sample,
        fieldnames=["id", "dice", "boundary_f1", "source", "panel_path"],
    )
    worst_dir = qc_root / "worst_cases"
    worst_dir.mkdir(parents=True, exist_ok=True)
    for row in per_sample[: max(1, int(worst_k))]:
        panel = str(row.get("panel_path", "")).strip()
        pid = str(row.get("id", "sample"))
        if panel and Path(panel).exists():
            shutil.copy2(panel, worst_dir / f"{pid}_panel.jpg")

    summary = {
        "num_samples": len(per_sample),
        "dice_mean": dice_mean,
        "boundary_f1_mean": bf1_mean,
        "subset_filter": ",".join(sorted(qc_subsets)),
        "dice_min_gate": float(dice_min),
        "boundary_f1_min_gate": float(bf1_min),
        "polypgen_samples": int(len(polypgen_rows)),
        "polypgen_dice_mean": float(polypgen_dice_mean),
        "polypgen_boundary_f1_mean": float(polypgen_bf1_mean),
        "polypgen_dice_min_gate": float(polypgen_dice_min),
        "polypgen_boundary_f1_min_gate": float(polypgen_bf1_min),
        "by_source": by_source,
        "per_sample_csv": str(qc_root / "per_sample.csv"),
        "worst_cases_dir": str(worst_dir),
    }
    (qc_root / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[lora_qc]", json.dumps(summary, ensure_ascii=False))
    if dice_mean < float(dice_min) or bf1_mean < float(bf1_min):
        raise RuntimeError(
            "LoRA QC gate failed: "
            f"dice_mean={dice_mean:.4f} (min={dice_min:.4f}), "
            f"boundary_f1_mean={bf1_mean:.4f} (min={bf1_min:.4f})"
        )
    if polypgen_rows and (
        polypgen_dice_mean < float(polypgen_dice_min) or polypgen_bf1_mean < float(polypgen_bf1_min)
    ):
        raise RuntimeError(
            "LoRA QC PolypGen gate failed: "
            f"dice_mean={polypgen_dice_mean:.4f} (min={polypgen_dice_min:.4f}), "
            f"boundary_f1_mean={polypgen_bf1_mean:.4f} (min={polypgen_bf1_min:.4f})"
        )


def _reuse_or_validate_lora_qc(
    metrics_path: Path,
    dice_min: float,
    bf1_min: float,
    polypgen_dice_min: float,
    polypgen_bf1_min: float,
):
    if not metrics_path.exists():
        return False
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    dice_mean = float(data.get("dice_mean", 0.0))
    bf1_mean = float(data.get("boundary_f1_mean", 0.0))
    if dice_mean < float(dice_min) or bf1_mean < float(bf1_min):
        raise RuntimeError(
            "Existing LoRA QC metrics fail gate: "
            f"dice_mean={dice_mean:.4f} (min={dice_min:.4f}), "
            f"boundary_f1_mean={bf1_mean:.4f} (min={bf1_min:.4f}). "
            "Remove old outputs or retrain teacher."
        )
    poly_n = int(data.get("polypgen_samples", 0))
    poly_d = float(data.get("polypgen_dice_mean", 0.0))
    poly_b = float(data.get("polypgen_boundary_f1_mean", 0.0))
    if poly_n > 0 and (poly_d < float(polypgen_dice_min) or poly_b < float(polypgen_bf1_min)):
        raise RuntimeError(
            "Existing LoRA QC PolypGen metrics fail gate: "
            f"dice_mean={poly_d:.4f} (min={polypgen_dice_min:.4f}), "
            f"boundary_f1_mean={poly_b:.4f} (min={polypgen_bf1_min:.4f}). "
            "Remove old outputs or retrain teacher."
        )
    print(f"[skip] lora_qc (reuse {metrics_path})")
    return True


def _prepare_box_review_template(
    proposal_csv: Path,
    uncertain_json: Path,
    out_template: Path,
    target_count: int,
):
    rows = _read_csv(str(proposal_csv))
    if not rows:
        raise RuntimeError(f"No proposal rows found: {proposal_csv}")
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -int(_to_float(r.get("is_fallback", 0), default=0)),
            _to_float(r.get("mean_prob", 0.0), default=0.0),
            -abs(_to_float(r.get("bbox_area_ratio", 0.0), default=0.0) - 0.12),
        ),
    )
    preferred_ids = []
    if uncertain_json.exists():
        try:
            data = json.loads(uncertain_json.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("ids", []), list):
                preferred_ids = [str(x) for x in data.get("ids", [])]
        except Exception:
            preferred_ids = []
    picked = _pick_stratified(rows_sorted, target_count, preferred_ids)
    template_rows = []
    for r in picked:
        template_rows.append(
            {
                "id": str(r.get("id", "")).strip(),
                "decision": "",
                "x0": str(r.get("x0", "")),
                "y0": str(r.get("y0", "")),
                "x1": str(r.get("x1", "")),
                "y1": str(r.get("y1", "")),
                "reason": "",
                "source": str(r.get("source", "")),
                "mean_prob": str(r.get("mean_prob", "")),
                "bbox_area_ratio": str(r.get("bbox_area_ratio", "")),
                "is_fallback": str(r.get("is_fallback", "")),
                "image_path": str(r.get("image_path", "")),
            }
        )
    _write_csv(
        str(out_template),
        template_rows,
        fieldnames=[
            "id",
            "decision",
            "x0",
            "y0",
            "x1",
            "y1",
            "reason",
            "source",
            "mean_prob",
            "bbox_area_ratio",
            "is_fallback",
            "image_path",
        ],
    )


def _prepare_mask_review_template(
    selected_quality_csv: Path,
    out_template: Path,
    target_count: int,
    hard_fraction: float = 0.50,
):
    rows = _read_csv(str(selected_quality_csv))
    if not rows:
        raise RuntimeError(f"No selected quality rows found: {selected_quality_csv}")

    target_count = int(max(0, target_count))
    if target_count <= 0:
        _write_csv(
            str(out_template),
            rows=[],
            fieldnames=[
                "id",
                "decision",
                "reason",
                "review_bucket",
                "source",
                "quality",
                "edge_quality",
                "area_ratio",
                "panel_path",
                "image_path",
            ],
        )
        return

    rows_low_sorted = sorted(
        rows,
        key=lambda r: (
            _to_float(r.get("quality", 0.0), default=0.0),
            _to_float(r.get("edge_quality", 0.0), default=0.0),
            -abs(_to_float(r.get("area_ratio", 0.0), default=0.0) - 0.08),
        ),
    )
    hf = float(max(0.0, min(1.0, hard_fraction)))
    hard_target = int(round(float(target_count) * hf))
    hard_target = int(max(0, min(target_count, hard_target)))
    rep_target = int(max(0, target_count - hard_target))

    hard_picked = _pick_stratified(rows_low_sorted, hard_target, preferred_ids=[])
    hard_ids = {str(r.get("id", "")).strip() for r in hard_picked if str(r.get("id", "")).strip()}

    source_quality_vals: dict[str, list[float]] = {}
    for r in rows:
        src = str(r.get("source", "")).strip() or "unknown"
        source_quality_vals.setdefault(src, []).append(_to_float(r.get("quality", 0.0), default=0.0))

    source_median: dict[str, float] = {}
    for src, vals in source_quality_vals.items():
        arr = sorted(float(v) for v in vals)
        if not arr:
            source_median[src] = 0.0
            continue
        n = len(arr)
        if n % 2 == 1:
            source_median[src] = float(arr[n // 2])
        else:
            source_median[src] = 0.5 * float(arr[n // 2 - 1] + arr[n // 2])

    remaining = [r for r in rows if str(r.get("id", "")).strip() not in hard_ids]
    remaining_sorted = sorted(
        remaining,
        key=lambda r: (
            abs(
                _to_float(r.get("quality", 0.0), default=0.0)
                - source_median.get(str(r.get("source", "")).strip() or "unknown", 0.0)
            ),
            -_to_float(r.get("quality", 0.0), default=0.0),
        ),
    )
    rep_picked = _pick_stratified(remaining_sorted, rep_target, preferred_ids=[])

    bucket_map: dict[str, str] = {}
    for r in hard_picked:
        pid = str(r.get("id", "")).strip()
        if pid:
            bucket_map[pid] = "hard"
    for r in rep_picked:
        pid = str(r.get("id", "")).strip()
        if pid and pid not in bucket_map:
            bucket_map[pid] = "representative"

    picked = hard_picked + rep_picked
    if len(picked) < target_count:
        picked_ids = {str(r.get("id", "")).strip() for r in picked if str(r.get("id", "")).strip()}
        fallback_rows = [r for r in rows_low_sorted if str(r.get("id", "")).strip() not in picked_ids]
        need = int(target_count - len(picked))
        extras = _pick_stratified(fallback_rows, need, preferred_ids=[])
        for r in extras:
            pid = str(r.get("id", "")).strip()
            if pid and pid not in bucket_map:
                bucket_map[pid] = "hard"
        picked.extend(extras)

    template_rows = []
    for r in picked:
        pid = str(r.get("id", "")).strip()
        template_rows.append(
            {
                "id": pid,
                "decision": "",
                "reason": "",
                "review_bucket": bucket_map.get(pid, "hard"),
                "source": str(r.get("source", "")),
                "quality": str(r.get("quality", "")),
                "edge_quality": str(r.get("edge_quality", "")),
                "area_ratio": str(r.get("area_ratio", "")),
                "panel_path": str(r.get("panel_path", "")),
                "image_path": str(r.get("image_path", "")),
            }
        )
    _write_csv(
        str(out_template),
        template_rows,
        fieldnames=[
            "id",
            "decision",
            "reason",
            "review_bucket",
            "source",
            "quality",
            "edge_quality",
            "area_ratio",
            "panel_path",
            "image_path",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Run full teacher->pseudo->student flywheel.")
    parser.add_argument("--data-manifest", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--base-sam-checkpoint", type=str, required=True)
    parser.add_argument("--student-config", type=str, default="configs/student_joint_training.yaml")
    parser.add_argument("--run-root", type=str, default="runs/flywheel_clean_v1")
    parser.add_argument("--flywheel-rounds", type=int, default=2)
    parser.add_argument("--round1-keep-quantile", type=float, default=0.35)
    parser.add_argument("--round2-keep-quantile", type=float, default=0.15)
    parser.add_argument(
        "--quality-score",
        type=str,
        default="0.35*conf+0.22*edge_quality+0.16*area_prior+0.07*center_prior+0.20*consistency_iou-0.10*spill_ratio-0.08*reflection_overlap",
    )
    parser.add_argument("--tiered-pseudo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tier-low-q", type=float, default=0.45)
    parser.add_argument("--tier-high-q", type=float, default=0.80)
    parser.add_argument("--polypgen-tier-low-q", type=float, default=0.55)
    parser.add_argument("--polypgen-tier-high-q", type=float, default=0.88)
    parser.add_argument("--mid-weight-scale", type=float, default=0.30)
    parser.add_argument("--high-weight-scale", type=float, default=1.00)
    parser.add_argument("--polypgen-mid-weight-scale", type=float, default=0.20)
    parser.add_argument("--polypgen-high-weight-scale", type=float, default=0.90)
    parser.add_argument("--round1-proposal-json", type=str, default="")
    parser.add_argument("--round2-proposal-json", type=str, default="")
    parser.add_argument(
        "--pseudo-auto-proposal-mode",
        type=str,
        default="hybrid_multi_box",
        choices=["single_box", "multi_box", "grid_multi_box", "content_multi_box", "hybrid_multi_box"],
    )
    parser.add_argument("--pseudo-candidate-box-scales", type=str, default="1.0,0.85,0.7,0.55")
    parser.add_argument("--pseudo-candidate-box-centers", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--pseudo-max-candidate-boxes", type=int, default=40)
    parser.add_argument(
        "--pseudo-proposal-mix-mode",
        type=str,
        default="augment_plus_auto",
        choices=["replace", "augment", "augment_plus_auto"],
    )
    parser.add_argument("--pseudo-proposal-jitter-scales", type=str, default="1.0,0.9,1.15")
    parser.add_argument("--pseudo-proposal-jitter-shifts", type=str, default="0.0,-0.08,0.08")
    parser.add_argument("--pseudo-proposal-jitter-max-boxes", type=int, default=27)
    parser.add_argument(
        "--pseudo-append-auto-candidates",
        type=str,
        default="off",
        choices=["off", "all", "polypgen"],
    )
    parser.add_argument("--pseudo-append-auto-max-candidates", type=int, default=12)
    parser.add_argument("--pseudo-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--pseudo-target-area-ratio", type=float, default=0.08)
    parser.add_argument("--pseudo-max-area-ratio", type=float, default=0.35)
    parser.add_argument("--pseudo-score-weight-conf", type=float, default=0.45)
    parser.add_argument("--pseudo-score-weight-edge", type=float, default=0.25)
    parser.add_argument("--pseudo-score-weight-area-prior", type=float, default=0.20)
    parser.add_argument("--pseudo-score-weight-center-prior", type=float, default=0.10)
    parser.add_argument("--pseudo-score-weight-center-prior-polypgen", type=float, default=0.04)
    parser.add_argument("--pseudo-score-bias-preset", type=float, default=0.0)
    parser.add_argument("--pseudo-score-bias-auto", type=float, default=0.0)
    parser.add_argument("--pseudo-score-bias-auto-polypgen", type=float, default=-999.0)
    parser.add_argument("--pseudo-two-pass-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pseudo-two-pass-min-first-quality", type=float, default=0.56)
    parser.add_argument("--pseudo-two-pass-min-first-area-ratio", type=float, default=0.003)
    parser.add_argument("--pseudo-two-pass-min-gain", type=float, default=0.0)
    parser.add_argument("--pseudo-two-pass-score-bonus", type=float, default=0.01)
    parser.add_argument("--pseudo-two-pass-box-expand-ratio", type=float, default=0.12)
    parser.add_argument("--pseudo-two-pass-pos-points", type=int, default=2)
    parser.add_argument("--pseudo-two-pass-neg-points", type=int, default=2)
    parser.add_argument("--pseudo-two-pass-use-reflection-neg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pseudo-two-pass-reflection-v-threshold", type=int, default=245)
    parser.add_argument("--pseudo-two-pass-reflection-s-threshold", type=int, default=35)
    parser.add_argument("--pseudo-quality-use-reflection-penalty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pseudo-quality-reflection-v-threshold", type=int, default=245)
    parser.add_argument("--pseudo-quality-reflection-s-threshold", type=int, default=35)
    parser.add_argument("--pseudo-quality-penalty-spill-weight", type=float, default=0.18)
    parser.add_argument("--pseudo-quality-penalty-reflection-weight", type=float, default=0.12)
    parser.add_argument("--pseudo-quality-penalty-fragment-weight", type=float, default=0.08)
    parser.add_argument("--pseudo-quality-penalty-consistency-weight", type=float, default=0.18)
    parser.add_argument("--pseudo-consistency-topk", type=int, default=5)
    parser.add_argument("--pseudo-consistency-min-iou", type=float, default=0.55)
    parser.add_argument("--pseudo-postprocess-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pseudo-post-min-component-area-ratio", type=float, default=0.0005)
    parser.add_argument("--pseudo-post-min-inbox-ratio", type=float, default=0.55)
    parser.add_argument("--pseudo-post-max-reflection-overlap", type=float, default=0.35)
    parser.add_argument("--pseudo-post-keep-max-components", type=int, default=2)
    parser.add_argument(
        "--manual-strict-proposals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When manual review is enabled, force pseudo generation to prefer cleaned proposal boxes.",
    )
    parser.add_argument("--manual-strict-append-auto-polypgen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-round1-quality-guard", type=str, default="true")
    parser.add_argument("--large-mask-threshold", type=float, default=0.40)
    parser.add_argument("--max-large-mask-frac", type=float, default=0.35)
    parser.add_argument("--teacher-refresh-between-rounds", type=str, default="true")
    parser.add_argument("--teacher-subset-filter", type=str, default="L_small,L_adapt_polypgen")
    parser.add_argument("--teacher-refresh-subset-filter", type=str, default="L_small,L_adapt_polypgen,pseudo_round1")
    parser.add_argument("--manual-review-per-round", type=int, default=220)
    parser.add_argument("--manual-box-review-count", type=int, default=120)
    parser.add_argument("--manual-mask-review-count", type=int, default=100)
    parser.add_argument(
        "--manual-mask-hard-fraction",
        type=float,
        default=0.50,
        help="Fraction of mask review samples taken from low-quality tail; remainder is representative by source.",
    )
    parser.add_argument("--manual-box-review-csv-round1", type=str, default="")
    parser.add_argument("--manual-box-review-csv-round2", type=str, default="")
    parser.add_argument("--manual-mask-review-csv-round1", type=str, default="")
    parser.add_argument("--manual-mask-review-csv-round2", type=str, default="")
    parser.add_argument(
        "--manual-pass-only-for-refresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only manual pass mask IDs for teacher refresh manifest when manual mask review is enabled.",
    )
    parser.add_argument(
        "--manual-pass-only-for-student",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only manual pass mask IDs from each round for final student manifest (small but cleaner pseudo set).",
    )
    parser.add_argument("--qa-gate-pass-rate", type=float, default=0.75)
    parser.add_argument("--qa-gate-polypgen-pass-rate", type=float, default=0.70)
    parser.add_argument("--qa-gate-boundary-bad-max", type=float, default=0.15)
    parser.add_argument("--skip-lora-qc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-qc-dice-min", type=float, default=0.70)
    parser.add_argument("--lora-qc-bf1-min", type=float, default=0.28)
    parser.add_argument("--lora-qc-subset-filter", type=str, default="L_small,L_adapt_polypgen")
    parser.add_argument("--lora-qc-polypgen-dice-min", type=float, default=0.68)
    parser.add_argument("--lora-qc-polypgen-bf1-min", type=float, default=0.26)
    parser.add_argument("--lora-qc-worst-k", type=int, default=40)
    parser.add_argument("--enable-quality-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibration-ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--calibration-expected-dice-min", type=float, default=0.80)
    parser.add_argument("--calibration-polypgen-expected-dice-min", type=float, default=0.78)
    parser.add_argument("--calibration-expected-dice-mid", type=float, default=0.88)
    parser.add_argument("--calibration-expected-dice-mid-scale", type=float, default=0.55)
    parser.add_argument("--box-localizer-checkpoint", type=str, default="checkpoints/student_flywheel_tiered_pilot_best.pth")
    parser.add_argument("--box-required-train-mode", type=str, default="off")
    parser.add_argument("--box-max-preview", type=int, default=200)
    parser.add_argument("--box-preview-mode", type=str, default="image_mask_box", choices=["image_box", "image_mask_box", "panel_heatmap"])
    parser.add_argument("--box-review-polypgen-min", type=int, default=70)
    parser.add_argument("--box-review-high-conf-threshold", type=float, default=0.72)
    parser.add_argument("--box-review-tiny-area-max", type=float, default=0.01)
    parser.add_argument("--box-review-reflection-high", type=float, default=0.10)
    parser.add_argument("--box-review-aspect-max", type=float, default=4.5)
    parser.add_argument("--box-review-center-bias-min", type=float, default=0.62)
    parser.add_argument("--lora-num-workers", type=int, default=0)
    parser.add_argument("--lora-enable-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-augment-prob", type=float, default=0.85)
    parser.add_argument("--lora-box-jitter-scale", type=float, default=0.30)
    parser.add_argument("--lora-box-jitter-shift", type=float, default=0.25)
    parser.add_argument("--lora-box-full-image-prob", type=float, default=0.03)
    parser.add_argument("--lora-epochs", type=int, default=30)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--lora-weight-decay", type=float, default=1e-4)
    parser.add_argument("--lora-encoder-rank", type=int, default=16)
    parser.add_argument("--lora-encoder-alpha", type=int, default=16)
    parser.add_argument("--lora-decoder-rank", type=int, default=8)
    parser.add_argument("--lora-decoder-alpha", type=int, default=8)
    parser.add_argument("--lora-stage-epochs", type=str, default="10,10,10")
    parser.add_argument("--lora-stage-lrs", type=str, default="")
    parser.add_argument("--lora-unfreeze-encoder-tail-fraction", type=float, default=0.35)
    parser.add_argument("--lora-loss-dice", type=float, default=0.50)
    parser.add_argument("--lora-loss-focal", type=float, default=0.30)
    parser.add_argument("--lora-loss-boundary", type=float, default=0.20)
    parser.add_argument("--lora-focal-alpha", type=float, default=0.25)
    parser.add_argument("--lora-focal-gamma", type=float, default=2.0)
    parser.add_argument("--lora-boundary-radius", type=int, default=1)
    parser.add_argument("--lora-prompt-mode", type=str, default="box_point_mix", choices=["box_only", "point_only", "box_point_mix"])
    parser.add_argument("--lora-prompt-mix-both-prob", type=float, default=0.60)
    parser.add_argument("--lora-prompt-mix-box-only-prob", type=float, default=0.20)
    parser.add_argument("--lora-num-pos-points", type=int, default=1)
    parser.add_argument("--lora-num-neg-points", type=int, default=1)
    parser.add_argument("--lora-point-jitter-frac", type=float, default=0.03)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--python-exec", type=str, default=sys.executable)
    args = parser.parse_args()
    if int(args.flywheel_rounds) != 2:
        raise ValueError("Current implementation supports exactly 2 flywheel rounds.")

    teacher_refresh = _bool_text(args.teacher_refresh_between_rounds, default=True)
    round1_guard = _bool_text(args.enable_round1_quality_guard, default=True)
    manual_enabled = int(args.manual_review_per_round) > 0
    box_count = int(max(0, args.manual_box_review_count))
    mask_count = int(max(0, args.manual_mask_review_count))
    if manual_enabled:
        total_review_budget = box_count + mask_count
        if total_review_budget <= 0:
            raise ValueError(
                "Manual review enabled but both --manual-box-review-count and --manual-mask-review-count are 0."
            )
        if total_review_budget > int(args.manual_review_per_round):
            raise ValueError(
                "Manual review budget overflow: "
                f"box({box_count}) + mask({mask_count}) > manual_review_per_round({args.manual_review_per_round})"
            )
        print(
            "[manual review] enabled "
            f"(per_round={int(args.manual_review_per_round)}, box={box_count}, mask={mask_count})"
        )
    else:
        print("[manual review] disabled (automatic flow only)")

    effective_mix_mode = str(args.pseudo_proposal_mix_mode)
    effective_auto_proposal_mode = str(args.pseudo_auto_proposal_mode)
    effective_fallback_full_image = True
    effective_score_bias_preset = float(args.pseudo_score_bias_preset)
    effective_score_bias_auto = float(args.pseudo_score_bias_auto)
    effective_score_bias_auto_polypgen = float(args.pseudo_score_bias_auto_polypgen)
    effective_append_auto_candidates = str(args.pseudo_append_auto_candidates)
    effective_append_auto_max_candidates = int(args.pseudo_append_auto_max_candidates)
    if manual_enabled and bool(args.manual_strict_proposals):
        # In manual-cleaning mode, avoid letting auto boxes override reviewed proposals.
        effective_mix_mode = "augment"
        effective_fallback_full_image = False
        if effective_auto_proposal_mode in {"multi_box", "grid_multi_box"}:
            effective_auto_proposal_mode = "hybrid_multi_box"
        if effective_score_bias_preset == 0.0 and effective_score_bias_auto == 0.0:
            # Keep source-bias neutral by default under manual-strict mode.
            effective_score_bias_preset = 0.0
            effective_score_bias_auto = 0.0
        if bool(args.manual_strict_append_auto_polypgen):
            # Localizer boxes can be over-tight on domain-shifted samples; always keep a richer auto candidate pool.
            effective_append_auto_candidates = "all"
            effective_append_auto_max_candidates = max(24, min(48, int(args.pseudo_append_auto_max_candidates)))
            if effective_score_bias_auto_polypgen <= -900.0:
                effective_score_bias_auto_polypgen = 0.05
        print(
            "[manual strict] enabled "
            f"(mix_mode={effective_mix_mode}, auto_mode={effective_auto_proposal_mode}, "
            f"fallback_full_image={effective_fallback_full_image}, "
            f"bias_preset={effective_score_bias_preset:+.3f}, bias_auto={effective_score_bias_auto:+.3f}, "
            f"append_auto={effective_append_auto_candidates}, append_auto_max={effective_append_auto_max_candidates})"
        )

    work = Path(args.run_root)
    work.mkdir(parents=True, exist_ok=True)
    ckpt_root = work / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    teacher_r0 = ckpt_root / "teacher_r0.pth"
    teacher_r1 = ckpt_root / "teacher_r1.pth"

    round1 = work / "round1"
    round2 = work / "round2"
    round1.mkdir(parents=True, exist_ok=True)
    round2.mkdir(parents=True, exist_ok=True)
    lora_common_args = [
        "--epochs", str(args.lora_epochs),
        "--lr", str(args.lora_lr),
        "--weight-decay", str(args.lora_weight_decay),
        "--encoder-rank", str(args.lora_encoder_rank),
        "--encoder-alpha", str(args.lora_encoder_alpha),
        "--decoder-rank", str(args.lora_decoder_rank),
        "--decoder-alpha", str(args.lora_decoder_alpha),
        "--stage-epochs", str(args.lora_stage_epochs),
        "--stage-lrs", str(args.lora_stage_lrs),
        "--unfreeze-encoder-tail-fraction", str(args.lora_unfreeze_encoder_tail_fraction),
        "--loss-dice", str(args.lora_loss_dice),
        "--loss-focal", str(args.lora_loss_focal),
        "--loss-boundary", str(args.lora_loss_boundary),
        "--focal-alpha", str(args.lora_focal_alpha),
        "--focal-gamma", str(args.lora_focal_gamma),
        "--boundary-radius", str(args.lora_boundary_radius),
        "--prompt-mode", str(args.lora_prompt_mode),
        "--prompt-mix-both-prob", str(args.lora_prompt_mix_both_prob),
        "--prompt-mix-box-only-prob", str(args.lora_prompt_mix_box_only_prob),
        "--num-pos-points", str(args.lora_num_pos_points),
        "--num-neg-points", str(args.lora_num_neg_points),
        "--point-jitter-frac", str(args.lora_point_jitter_frac),
        "--num-workers", str(args.lora_num_workers),
        "--augment-prob", str(args.lora_augment_prob),
        "--box-jitter-scale", str(args.lora_box_jitter_scale),
        "--box-jitter-shift", str(args.lora_box_jitter_shift),
        "--box-full-image-prob", str(args.lora_box_full_image_prob),
        "--enable-augment" if bool(args.lora_enable_augment) else "--no-enable-augment",
    ]

    teacher_r0_done_paths = [teacher_r0] if _is_lora_checkpoint_complete(teacher_r0, int(args.lora_epochs)) else []
    _run_if_outputs_missing(
        step_name="teacher_r0_lora",
        done_paths=teacher_r0_done_paths,
        cmd=[
            args.python_exec,
            "medsam_tools/finetune_lora.py",
            "--checkpoint", args.base_sam_checkpoint,
            "--data-manifest", args.data_manifest,
            "--subset-filter", str(args.teacher_subset_filter),
            "--split-filter", "train,val",
            *lora_common_args,
            "--save-path", str(teacher_r0),
        ],
    )

    lora_qc_metrics = ""
    if not bool(args.skip_lora_qc):
        lora_qc_metrics_path = work / "lora_qc" / "metrics.json"
        reused = _reuse_or_validate_lora_qc(
            metrics_path=lora_qc_metrics_path,
            dice_min=float(args.lora_qc_dice_min),
            bf1_min=float(args.lora_qc_bf1_min),
            polypgen_dice_min=float(args.lora_qc_polypgen_dice_min),
            polypgen_bf1_min=float(args.lora_qc_polypgen_bf1_min),
        )
        if not reused:
            _run_lora_qc(
                python_exec=args.python_exec,
                work=work,
                data_manifest=args.data_manifest,
                base_sam_checkpoint=args.base_sam_checkpoint,
                teacher_checkpoint=str(teacher_r0),
                dice_min=float(args.lora_qc_dice_min),
                bf1_min=float(args.lora_qc_bf1_min),
                subset_filter=str(args.lora_qc_subset_filter),
                polypgen_dice_min=float(args.lora_qc_polypgen_dice_min),
                polypgen_bf1_min=float(args.lora_qc_polypgen_bf1_min),
                worst_k=int(args.lora_qc_worst_k),
            )
        lora_qc_metrics = str(lora_qc_metrics_path)

    calibration_json = ""
    if bool(args.enable_quality_calibration):
        qc_root = work / "lora_qc"
        qc_quality_csv = qc_root / "pseudo_val" / "pseudo_quality.csv"
        qc_per_sample_csv = qc_root / "per_sample.csv"
        cal_json = qc_root / "quality_calibration.json"
        if qc_quality_csv.exists() and qc_per_sample_csv.exists():
            _run_if_outputs_missing(
                step_name="quality_calibration_fit",
                done_paths=[cal_json],
                cmd=[
                    args.python_exec,
                    "tools/calibrate_pseudo_quality.py",
                    "--quality-csv",
                    str(qc_quality_csv),
                    "--per-sample-csv",
                    str(qc_per_sample_csv),
                    "--ridge-alpha",
                    str(args.calibration_ridge_alpha),
                    "--output-json",
                    str(cal_json),
                ],
            )
            if cal_json.exists():
                calibration_json = str(cal_json)
                print(f"[quality calibration] using {calibration_json}")
        else:
            print("[quality calibration] skipped (missing lora_qc pseudo_val/per_sample artifacts)")

    round1_manual_dir = round1 / "manual"
    round1_manual_dir.mkdir(parents=True, exist_ok=True)
    round1_filter_dir = round1 / ("filter_tiered" if bool(args.tiered_pseudo) else "filter")
    round2_filter_dir = round2 / ("filter_tiered" if bool(args.tiered_pseudo) else "filter")

    round1_proposal_json_for_pseudo = str(args.round1_proposal_json or "")
    round1_id_filter_for_pseudo = ""
    round1_box_template = round1_manual_dir / "box_review_template.csv"
    round1_mask_template = round1_manual_dir / "mask_review_template.csv"
    round1_box_csv = _resolve_manual_csv(args.manual_box_review_csv_round1, round1_manual_dir / "box_review.csv")
    round1_mask_csv = _resolve_manual_csv(args.manual_mask_review_csv_round1, round1_manual_dir / "mask_review.csv")
    round1_mask_summary = round1_manual_dir / "qa_summary.json"

    if manual_enabled and box_count > 0:
        localizer_ckpt = Path(args.box_localizer_checkpoint)
        if not localizer_ckpt.exists():
            raise RuntimeError(f"Missing --box-localizer-checkpoint: {localizer_ckpt}")
        round1_auto_prop_json = round1_manual_dir / "auto_proposals.json"
        round1_auto_prop_csv = round1_manual_dir / "auto_proposals.csv"
        round1_uncertain_json = round1_manual_dir / "uncertain_box_ids.json"
        _run_if_outputs_missing(
            step_name="round1_generate_box_prompts",
            done_paths=[round1_auto_prop_json, round1_auto_prop_csv],
            cmd=[
                args.python_exec,
                "tools/generate_box_prompts.py",
                "--data-manifest", args.data_manifest,
                "--subset-filter", "U_large",
                "--localizer-checkpoint", str(localizer_ckpt),
                "--required-train-mode", str(args.box_required_train_mode),
                "--output-json", str(round1_auto_prop_json),
                "--output-csv", str(round1_auto_prop_csv),
                "--preview-dir", str(round1_manual_dir / "proposal_previews"),
                "--max-preview", str(max(0, int(args.box_max_preview))),
                "--preview-mode", str(args.box_preview_mode),
            ],
        )
        _run_if_outputs_missing(
            step_name="round1_select_box_review_ids",
            done_paths=[round1_uncertain_json, round1_manual_dir / "uncertain_box_ids.txt"],
            cmd=[
                args.python_exec,
                "tools/select_box_review_ids.py",
                "--proposal-csv", str(round1_auto_prop_csv),
                "--max-review", str(max(box_count, int(args.manual_review_per_round))),
                "--polypgen-min-quota", str(max(0, int(args.box_review_polypgen_min))),
                "--high-conf-threshold", str(args.box_review_high_conf_threshold),
                "--tiny-box-area-max", str(args.box_review_tiny_area_max),
                "--reflection-high-threshold", str(args.box_review_reflection_high),
                "--aspect-ratio-max", str(args.box_review_aspect_max),
                "--center-bias-threshold", str(args.box_review_center_bias_min),
                "--output-json", str(round1_uncertain_json),
                "--output-txt", str(round1_manual_dir / "uncertain_box_ids.txt"),
            ],
        )
        _prepare_box_review_template(
            proposal_csv=round1_auto_prop_csv,
            uncertain_json=round1_uncertain_json,
            out_template=round1_box_template,
            target_count=box_count,
        )
        print(f"[round1] box review template ready: {round1_box_template}")
        _require_manual_csv(round1_box_csv, round1_box_template, "round1 box review")
        round1_cleaned_prop_json = round1_manual_dir / "cleaned_proposals.json"
        round1_id_filter_txt = round1_manual_dir / "accepted_ids.txt"
        _run([
            args.python_exec,
            "tools/apply_manual_review.py",
            "--proposal-json", str(round1_auto_prop_json),
            "--proposal-csv", str(round1_auto_prop_csv),
            "--box-review-csv", str(round1_box_csv),
            "--output-proposal-json", str(round1_cleaned_prop_json),
            "--output-id-filter", str(round1_id_filter_txt),
            "--output-qa-summary", str(round1_manual_dir / "box_review_summary.json"),
            "--round-id", "1",
        ])
        round1_proposal_json_for_pseudo = str(round1_cleaned_prop_json)
        round1_id_filter_for_pseudo = str(round1_id_filter_txt)

    round1_pseudo_cmd = [
        args.python_exec,
        "medsam_tools/generate_pseudo_labels.py",
        "--checkpoint", args.base_sam_checkpoint,
        "--lora-checkpoint", str(teacher_r0),
        "--data-manifest", args.data_manifest,
        "--subset-filter", "U_large",
        "--round-id", "1",
        "--output-root", str(round1 / "pseudo"),
        "--auto-proposal-mode", str(effective_auto_proposal_mode),
        "--candidate-box-scales", args.pseudo_candidate_box_scales,
        "--candidate-box-centers", args.pseudo_candidate_box_centers,
        "--max-candidate-boxes", str(args.pseudo_max_candidate_boxes),
        "--proposal-mix-mode", str(effective_mix_mode),
        "--proposal-jitter-scales", str(args.pseudo_proposal_jitter_scales),
        "--proposal-jitter-shifts", str(args.pseudo_proposal_jitter_shifts),
        "--proposal-jitter-max-boxes", str(args.pseudo_proposal_jitter_max_boxes),
        "--append-auto-candidates", str(effective_append_auto_candidates),
        "--append-auto-max-candidates", str(effective_append_auto_max_candidates),
        "--min-mask-area-ratio", str(args.pseudo_min_area_ratio),
        "--target-mask-area-ratio", str(args.pseudo_target_area_ratio),
        "--max-mask-area-ratio", str(args.pseudo_max_area_ratio),
        "--score-weight-conf", str(args.pseudo_score_weight_conf),
        "--score-weight-edge", str(args.pseudo_score_weight_edge),
        "--score-weight-area-prior", str(args.pseudo_score_weight_area_prior),
        "--score-weight-center-prior", str(args.pseudo_score_weight_center_prior),
        "--score-weight-center-prior-polypgen", str(args.pseudo_score_weight_center_prior_polypgen),
        "--score-bias-preset", str(effective_score_bias_preset),
        "--score-bias-auto", str(effective_score_bias_auto),
        "--score-bias-auto-polypgen", str(effective_score_bias_auto_polypgen),
        "--two-pass-min-first-quality", str(args.pseudo_two_pass_min_first_quality),
        "--two-pass-min-first-area-ratio", str(args.pseudo_two_pass_min_first_area_ratio),
        "--two-pass-min-gain", str(args.pseudo_two_pass_min_gain),
        "--two-pass-score-bonus", str(args.pseudo_two_pass_score_bonus),
        "--two-pass-box-expand-ratio", str(args.pseudo_two_pass_box_expand_ratio),
        "--two-pass-pos-points", str(args.pseudo_two_pass_pos_points),
        "--two-pass-neg-points", str(args.pseudo_two_pass_neg_points),
        "--two-pass-reflection-v-threshold", str(args.pseudo_two_pass_reflection_v_threshold),
        "--two-pass-reflection-s-threshold", str(args.pseudo_two_pass_reflection_s_threshold),
        "--quality-reflection-v-threshold", str(args.pseudo_quality_reflection_v_threshold),
        "--quality-reflection-s-threshold", str(args.pseudo_quality_reflection_s_threshold),
        "--quality-penalty-spill-weight", str(args.pseudo_quality_penalty_spill_weight),
        "--quality-penalty-reflection-weight", str(args.pseudo_quality_penalty_reflection_weight),
        "--quality-penalty-fragment-weight", str(args.pseudo_quality_penalty_fragment_weight),
        "--quality-penalty-consistency-weight", str(args.pseudo_quality_penalty_consistency_weight),
        "--consistency-topk", str(args.pseudo_consistency_topk),
        "--consistency-min-iou", str(args.pseudo_consistency_min_iou),
        "--post-min-component-area-ratio", str(args.pseudo_post_min_component_area_ratio),
        "--post-min-inbox-ratio", str(args.pseudo_post_min_inbox_ratio),
        "--post-max-reflection-overlap", str(args.pseudo_post_max_reflection_overlap),
        "--post-keep-max-components", str(args.pseudo_post_keep_max_components),
        "--write-candidate-scores",
    ]
    round1_pseudo_cmd.append("--fallback-full-image" if effective_fallback_full_image else "--no-fallback-full-image")
    round1_pseudo_cmd.append("--two-pass-refine" if bool(args.pseudo_two_pass_refine) else "--no-two-pass-refine")
    round1_pseudo_cmd.append(
        "--two-pass-use-reflection-neg" if bool(args.pseudo_two_pass_use_reflection_neg) else "--no-two-pass-use-reflection-neg"
    )
    round1_pseudo_cmd.append(
        "--quality-use-reflection-penalty"
        if bool(args.pseudo_quality_use_reflection_penalty)
        else "--no-quality-use-reflection-penalty"
    )
    round1_pseudo_cmd.append("--postprocess-mask" if bool(args.pseudo_postprocess_mask) else "--no-postprocess-mask")
    if round1_proposal_json_for_pseudo:
        round1_pseudo_cmd.extend(["--proposal-json", str(round1_proposal_json_for_pseudo)])
    if round1_id_filter_for_pseudo:
        round1_pseudo_cmd.extend(["--id-filter", str(round1_id_filter_for_pseudo)])
    _run_if_outputs_missing(
        step_name="round1_pseudo_generation",
        done_paths=[round1 / "pseudo" / "pseudo_quality.csv", round1 / "pseudo" / "pseudo_candidates_manifest.csv"],
        cmd=round1_pseudo_cmd,
    )
    _pseudo_artifact_guard(
        quality_csv=str(round1 / "pseudo" / "pseudo_quality.csv"),
        hard_masks_dir=str(round1 / "pseudo" / "hard_masks"),
        enabled=True,
    )
    _round_quality_guard(
        quality_csv=str(round1 / "pseudo" / "pseudo_quality.csv"),
        large_mask_threshold=float(args.large_mask_threshold),
        max_large_mask_frac=float(args.max_large_mask_frac),
        enabled=round1_guard,
    )
    _run_if_outputs_missing(
        step_name="round1_pseudo_filtering",
        done_paths=[round1_filter_dir / "selected_quality.csv", round1_filter_dir / "selected_manifest.csv"],
        cmd=[
            args.python_exec,
            "tools/filter_pseudo_labels.py",
            "--quality-csv", str(round1 / "pseudo" / "pseudo_quality.csv"),
            "--keep-quantile", str(args.round1_keep_quantile),
            "--quality-score", args.quality_score,
            "--round-id", "1",
            "--output-dir", str(round1_filter_dir),
            "--pseudo-candidates-manifest", str(round1 / "pseudo" / "pseudo_candidates_manifest.csv"),
            "--base-manifest", args.data_manifest,
            "--calibration-json", str(calibration_json),
            "--expected-dice-min", str(args.calibration_expected_dice_min),
            "--polypgen-expected-dice-min", str(args.calibration_polypgen_expected_dice_min),
            "--expected-dice-mid", str(args.calibration_expected_dice_mid),
            "--expected-dice-mid-scale", str(args.calibration_expected_dice_mid_scale),
            *(
                [
                    "--tiered-pseudo",
                    "--tier-low-q", str(args.tier_low_q),
                    "--tier-high-q", str(args.tier_high_q),
                    "--polypgen-tier-low-q", str(args.polypgen_tier_low_q),
                    "--polypgen-tier-high-q", str(args.polypgen_tier_high_q),
                    "--mid-weight-scale", str(args.mid_weight_scale),
                    "--high-weight-scale", str(args.high_weight_scale),
                    "--polypgen-mid-weight-scale", str(args.polypgen_mid_weight_scale),
                    "--polypgen-high-weight-scale", str(args.polypgen_high_weight_scale),
                ]
                if bool(args.tiered_pseudo)
                else []
            ),
        ],
    )
    round1_selected_manifest = round1_filter_dir / "selected_manifest.csv"
    round1_selected_for_refresh = round1_selected_manifest
    round1_selected_for_student = round1_selected_manifest
    round1_remaining_manifest = round1_filter_dir / "remaining_u_large_manifest.csv"

    if manual_enabled and mask_count > 0:
        _prepare_mask_review_template(
            selected_quality_csv=round1_filter_dir / "selected_quality.csv",
            out_template=round1_mask_template,
            target_count=mask_count,
            hard_fraction=float(args.manual_mask_hard_fraction),
        )
        print(f"[round1] mask review template ready: {round1_mask_template}")
        _require_manual_csv(round1_mask_csv, round1_mask_template, "round1 mask review")
        round1_selected_clean = round1_filter_dir / "selected_manifest_cleaned.csv"
        _run([
            args.python_exec,
            "tools/apply_manual_review.py",
            "--proposal-json", str(round1_proposal_json_for_pseudo) if round1_proposal_json_for_pseudo else str(round1_manual_dir / "auto_proposals.json"),
            "--proposal-csv", str(round1_manual_dir / "auto_proposals.csv"),
            "--box-review-csv", str(round1_box_csv if round1_box_csv.exists() else ""),
            "--mask-review-csv", str(round1_mask_csv),
            "--selected-manifest", str(round1_selected_manifest),
            "--quality-csv", str(round1_filter_dir / "selected_quality.csv"),
            "--output-proposal-json", str(round1_manual_dir / "cleaned_proposals_after_mask.json"),
            "--output-selected-manifest", str(round1_selected_clean),
            "--output-qa-summary", str(round1_mask_summary),
            "--round-id", "1",
        ])
        round1_selected_manifest = round1_selected_clean
        _check_round_qa_gate(
            summary_path=round1_mask_summary,
            pass_rate_min=float(args.qa_gate_pass_rate),
            polypgen_pass_rate_min=float(args.qa_gate_polypgen_pass_rate),
            boundary_bad_max=float(args.qa_gate_boundary_bad_max),
            round_id=1,
        )
        if bool(args.manual_pass_only_for_refresh) or bool(args.manual_pass_only_for_student):
            pass_ids_r1 = _manual_pass_ids(str(round1_mask_csv))
            reviewed_manifest_r1 = round1_filter_dir / "selected_manifest_manual_pass.csv"
            kept_n = _filter_manifest_by_ids(str(round1_selected_manifest), pass_ids_r1, str(reviewed_manifest_r1))
            print(
                "[round1 manual pass filter] "
                f"manual_pass_ids={len(pass_ids_r1)} kept_in_selected={kept_n} "
                f"manifest={reviewed_manifest_r1}"
            )
            if kept_n > 0:
                if bool(args.manual_pass_only_for_refresh):
                    round1_selected_for_refresh = reviewed_manifest_r1
                if bool(args.manual_pass_only_for_student):
                    round1_selected_for_student = reviewed_manifest_r1

    # Optional fastlane: if a reviewed tight-good CSV is present, augment round1 refresh manifest.
    tight_review_csv = round1_manual_dir / "tight_good_seed300_review.csv"
    if teacher_refresh and tight_review_csv.exists() and round1_selected_for_refresh.exists():
        tight_keep_ids = _review_non_reject_ids(str(tight_review_csv))
        if tight_keep_ids:
            tight_aug_manifest = round1_filter_dir / "selected_manifest_refresh_plus_tight.csv"
            total_n, added_n = _augment_manifest_with_ids(
                base_manifest=str(round1_selected_for_refresh),
                source_manifest=str(round1 / "pseudo" / "pseudo_candidates_manifest.csv"),
                add_ids=tight_keep_ids,
                output_manifest=str(tight_aug_manifest),
            )
            if added_n > 0:
                round1_selected_for_refresh = tight_aug_manifest
            print(
                "[round1 tight fastlane] "
                f"review_non_reject={len(tight_keep_ids)} added_to_refresh={added_n} "
                f"refresh_manifest_rows={total_n} manifest={tight_aug_manifest}"
            )

    teacher_for_round2 = teacher_r0
    if teacher_refresh and round1_selected_for_refresh.exists():
        teacher_train_manifest = round1 / "teacher_round1_manifest.csv"
        refresh_keep_subsets = {
            s.strip()
            for s in str(args.teacher_refresh_subset_filter).split(",")
            if s.strip() and not s.strip().lower().startswith("pseudo_")
        }
        if not refresh_keep_subsets:
            refresh_keep_subsets = {"L_small"}
        _merge_teacher_manifest(
            args.data_manifest,
            str(round1_selected_for_refresh),
            str(teacher_train_manifest),
            keep_subsets=refresh_keep_subsets,
        )
        teacher_r1_done_paths = [teacher_r1] if _is_lora_checkpoint_complete(teacher_r1, int(args.lora_epochs)) else []
        _run_if_outputs_missing(
            step_name="teacher_r1_refresh",
            done_paths=teacher_r1_done_paths,
            cmd=[
                args.python_exec,
                "medsam_tools/finetune_lora.py",
                "--checkpoint", args.base_sam_checkpoint,
                "--data-manifest", str(teacher_train_manifest),
                "--subset-filter", str(args.teacher_refresh_subset_filter),
                "--split-filter", "train,val",
                *lora_common_args,
                "--init-lora-checkpoint", str(teacher_r0),
                "--save-path", str(teacher_r1),
            ],
        )
        teacher_for_round2 = teacher_r1

    manifest_for_round2 = str(round1_remaining_manifest) if round1_remaining_manifest.exists() else args.data_manifest
    round2_manual_dir = round2 / "manual"
    round2_manual_dir.mkdir(parents=True, exist_ok=True)
    round2_box_template = round2_manual_dir / "box_review_template.csv"
    round2_mask_template = round2_manual_dir / "mask_review_template.csv"
    round2_box_csv = _resolve_manual_csv(args.manual_box_review_csv_round2, round2_manual_dir / "box_review.csv")
    round2_mask_csv = _resolve_manual_csv(args.manual_mask_review_csv_round2, round2_manual_dir / "mask_review.csv")
    round2_mask_summary = round2_manual_dir / "qa_summary.json"
    round2_proposal_json_for_pseudo = str(args.round2_proposal_json or "")
    round2_id_filter_for_pseudo = ""

    if manual_enabled and box_count > 0:
        localizer_ckpt = Path(args.box_localizer_checkpoint)
        if not localizer_ckpt.exists():
            raise RuntimeError(f"Missing --box-localizer-checkpoint: {localizer_ckpt}")
        round2_auto_prop_json = round2_manual_dir / "auto_proposals.json"
        round2_auto_prop_csv = round2_manual_dir / "auto_proposals.csv"
        round2_uncertain_json = round2_manual_dir / "uncertain_box_ids.json"
        _run_if_outputs_missing(
            step_name="round2_generate_box_prompts",
            done_paths=[round2_auto_prop_json, round2_auto_prop_csv],
            cmd=[
                args.python_exec,
                "tools/generate_box_prompts.py",
                "--data-manifest", manifest_for_round2,
                "--subset-filter", "U_large",
                "--localizer-checkpoint", str(localizer_ckpt),
                "--required-train-mode", str(args.box_required_train_mode),
                "--output-json", str(round2_auto_prop_json),
                "--output-csv", str(round2_auto_prop_csv),
                "--preview-dir", str(round2_manual_dir / "proposal_previews"),
                "--max-preview", str(max(0, int(args.box_max_preview))),
                "--preview-mode", str(args.box_preview_mode),
            ],
        )
        _run_if_outputs_missing(
            step_name="round2_select_box_review_ids",
            done_paths=[round2_uncertain_json, round2_manual_dir / "uncertain_box_ids.txt"],
            cmd=[
                args.python_exec,
                "tools/select_box_review_ids.py",
                "--proposal-csv", str(round2_auto_prop_csv),
                "--max-review", str(max(box_count, int(args.manual_review_per_round))),
                "--polypgen-min-quota", str(max(0, int(args.box_review_polypgen_min))),
                "--high-conf-threshold", str(args.box_review_high_conf_threshold),
                "--tiny-box-area-max", str(args.box_review_tiny_area_max),
                "--reflection-high-threshold", str(args.box_review_reflection_high),
                "--aspect-ratio-max", str(args.box_review_aspect_max),
                "--center-bias-threshold", str(args.box_review_center_bias_min),
                "--output-json", str(round2_uncertain_json),
                "--output-txt", str(round2_manual_dir / "uncertain_box_ids.txt"),
            ],
        )
        _prepare_box_review_template(
            proposal_csv=round2_auto_prop_csv,
            uncertain_json=round2_uncertain_json,
            out_template=round2_box_template,
            target_count=box_count,
        )
        print(f"[round2] box review template ready: {round2_box_template}")
        _require_manual_csv(round2_box_csv, round2_box_template, "round2 box review")
        round2_cleaned_prop_json = round2_manual_dir / "cleaned_proposals.json"
        round2_id_filter_txt = round2_manual_dir / "accepted_ids.txt"
        _run([
            args.python_exec,
            "tools/apply_manual_review.py",
            "--proposal-json", str(round2_auto_prop_json),
            "--proposal-csv", str(round2_auto_prop_csv),
            "--box-review-csv", str(round2_box_csv),
            "--output-proposal-json", str(round2_cleaned_prop_json),
            "--output-id-filter", str(round2_id_filter_txt),
            "--output-qa-summary", str(round2_manual_dir / "box_review_summary.json"),
            "--round-id", "2",
        ])
        round2_proposal_json_for_pseudo = str(round2_cleaned_prop_json)
        round2_id_filter_for_pseudo = str(round2_id_filter_txt)

    round2_pseudo_cmd = [
        args.python_exec,
        "medsam_tools/generate_pseudo_labels.py",
        "--checkpoint", args.base_sam_checkpoint,
        "--lora-checkpoint", str(teacher_for_round2),
        "--data-manifest", manifest_for_round2,
        "--subset-filter", "U_large",
        "--round-id", "2",
        "--output-root", str(round2 / "pseudo"),
        "--auto-proposal-mode", str(effective_auto_proposal_mode),
        "--candidate-box-scales", args.pseudo_candidate_box_scales,
        "--candidate-box-centers", args.pseudo_candidate_box_centers,
        "--max-candidate-boxes", str(args.pseudo_max_candidate_boxes),
        "--proposal-mix-mode", str(effective_mix_mode),
        "--proposal-jitter-scales", str(args.pseudo_proposal_jitter_scales),
        "--proposal-jitter-shifts", str(args.pseudo_proposal_jitter_shifts),
        "--proposal-jitter-max-boxes", str(args.pseudo_proposal_jitter_max_boxes),
        "--append-auto-candidates", str(effective_append_auto_candidates),
        "--append-auto-max-candidates", str(effective_append_auto_max_candidates),
        "--min-mask-area-ratio", str(args.pseudo_min_area_ratio),
        "--target-mask-area-ratio", str(args.pseudo_target_area_ratio),
        "--max-mask-area-ratio", str(args.pseudo_max_area_ratio),
        "--score-weight-conf", str(args.pseudo_score_weight_conf),
        "--score-weight-edge", str(args.pseudo_score_weight_edge),
        "--score-weight-area-prior", str(args.pseudo_score_weight_area_prior),
        "--score-weight-center-prior", str(args.pseudo_score_weight_center_prior),
        "--score-weight-center-prior-polypgen", str(args.pseudo_score_weight_center_prior_polypgen),
        "--score-bias-preset", str(effective_score_bias_preset),
        "--score-bias-auto", str(effective_score_bias_auto),
        "--score-bias-auto-polypgen", str(effective_score_bias_auto_polypgen),
        "--two-pass-min-first-quality", str(args.pseudo_two_pass_min_first_quality),
        "--two-pass-min-first-area-ratio", str(args.pseudo_two_pass_min_first_area_ratio),
        "--two-pass-min-gain", str(args.pseudo_two_pass_min_gain),
        "--two-pass-score-bonus", str(args.pseudo_two_pass_score_bonus),
        "--two-pass-box-expand-ratio", str(args.pseudo_two_pass_box_expand_ratio),
        "--two-pass-pos-points", str(args.pseudo_two_pass_pos_points),
        "--two-pass-neg-points", str(args.pseudo_two_pass_neg_points),
        "--two-pass-reflection-v-threshold", str(args.pseudo_two_pass_reflection_v_threshold),
        "--two-pass-reflection-s-threshold", str(args.pseudo_two_pass_reflection_s_threshold),
        "--quality-reflection-v-threshold", str(args.pseudo_quality_reflection_v_threshold),
        "--quality-reflection-s-threshold", str(args.pseudo_quality_reflection_s_threshold),
        "--quality-penalty-spill-weight", str(args.pseudo_quality_penalty_spill_weight),
        "--quality-penalty-reflection-weight", str(args.pseudo_quality_penalty_reflection_weight),
        "--quality-penalty-fragment-weight", str(args.pseudo_quality_penalty_fragment_weight),
        "--quality-penalty-consistency-weight", str(args.pseudo_quality_penalty_consistency_weight),
        "--consistency-topk", str(args.pseudo_consistency_topk),
        "--consistency-min-iou", str(args.pseudo_consistency_min_iou),
        "--post-min-component-area-ratio", str(args.pseudo_post_min_component_area_ratio),
        "--post-min-inbox-ratio", str(args.pseudo_post_min_inbox_ratio),
        "--post-max-reflection-overlap", str(args.pseudo_post_max_reflection_overlap),
        "--post-keep-max-components", str(args.pseudo_post_keep_max_components),
        "--write-candidate-scores",
    ]
    round2_pseudo_cmd.append("--fallback-full-image" if effective_fallback_full_image else "--no-fallback-full-image")
    round2_pseudo_cmd.append("--two-pass-refine" if bool(args.pseudo_two_pass_refine) else "--no-two-pass-refine")
    round2_pseudo_cmd.append(
        "--two-pass-use-reflection-neg" if bool(args.pseudo_two_pass_use_reflection_neg) else "--no-two-pass-use-reflection-neg"
    )
    round2_pseudo_cmd.append(
        "--quality-use-reflection-penalty"
        if bool(args.pseudo_quality_use_reflection_penalty)
        else "--no-quality-use-reflection-penalty"
    )
    round2_pseudo_cmd.append("--postprocess-mask" if bool(args.pseudo_postprocess_mask) else "--no-postprocess-mask")
    if round2_proposal_json_for_pseudo:
        round2_pseudo_cmd.extend(["--proposal-json", str(round2_proposal_json_for_pseudo)])
    if round2_id_filter_for_pseudo:
        round2_pseudo_cmd.extend(["--id-filter", str(round2_id_filter_for_pseudo)])
    _run_if_outputs_missing(
        step_name="round2_pseudo_generation",
        done_paths=[round2 / "pseudo" / "pseudo_quality.csv", round2 / "pseudo" / "pseudo_candidates_manifest.csv"],
        cmd=round2_pseudo_cmd,
    )
    _pseudo_artifact_guard(
        quality_csv=str(round2 / "pseudo" / "pseudo_quality.csv"),
        hard_masks_dir=str(round2 / "pseudo" / "hard_masks"),
        enabled=True,
    )
    _run_if_outputs_missing(
        step_name="round2_pseudo_filtering",
        done_paths=[round2_filter_dir / "selected_quality.csv", round2_filter_dir / "selected_manifest.csv"],
        cmd=[
            args.python_exec,
            "tools/filter_pseudo_labels.py",
            "--quality-csv", str(round2 / "pseudo" / "pseudo_quality.csv"),
            "--keep-quantile", str(args.round2_keep_quantile),
            "--quality-score", args.quality_score,
            "--round-id", "2",
            "--output-dir", str(round2_filter_dir),
            "--pseudo-candidates-manifest", str(round2 / "pseudo" / "pseudo_candidates_manifest.csv"),
            "--base-manifest", manifest_for_round2,
            "--calibration-json", str(calibration_json),
            "--expected-dice-min", str(args.calibration_expected_dice_min),
            "--polypgen-expected-dice-min", str(args.calibration_polypgen_expected_dice_min),
            "--expected-dice-mid", str(args.calibration_expected_dice_mid),
            "--expected-dice-mid-scale", str(args.calibration_expected_dice_mid_scale),
            *(
                [
                    "--tiered-pseudo",
                    "--tier-low-q", str(args.tier_low_q),
                    "--tier-high-q", str(args.tier_high_q),
                    "--polypgen-tier-low-q", str(args.polypgen_tier_low_q),
                    "--polypgen-tier-high-q", str(args.polypgen_tier_high_q),
                    "--mid-weight-scale", str(args.mid_weight_scale),
                    "--high-weight-scale", str(args.high_weight_scale),
                    "--polypgen-mid-weight-scale", str(args.polypgen_mid_weight_scale),
                    "--polypgen-high-weight-scale", str(args.polypgen_high_weight_scale),
                ]
                if bool(args.tiered_pseudo)
                else []
            ),
        ],
    )
    round2_selected_manifest = round2_filter_dir / "selected_manifest.csv"
    round2_selected_for_student = round2_selected_manifest

    if manual_enabled and mask_count > 0:
        _prepare_mask_review_template(
            selected_quality_csv=round2_filter_dir / "selected_quality.csv",
            out_template=round2_mask_template,
            target_count=mask_count,
            hard_fraction=float(args.manual_mask_hard_fraction),
        )
        print(f"[round2] mask review template ready: {round2_mask_template}")
        _require_manual_csv(round2_mask_csv, round2_mask_template, "round2 mask review")
        round2_selected_clean = round2_filter_dir / "selected_manifest_cleaned.csv"
        _run([
            args.python_exec,
            "tools/apply_manual_review.py",
            "--proposal-json", str(round2_proposal_json_for_pseudo) if round2_proposal_json_for_pseudo else str(round2_manual_dir / "auto_proposals.json"),
            "--proposal-csv", str(round2_manual_dir / "auto_proposals.csv"),
            "--box-review-csv", str(round2_box_csv if round2_box_csv.exists() else ""),
            "--mask-review-csv", str(round2_mask_csv),
            "--selected-manifest", str(round2_selected_manifest),
            "--quality-csv", str(round2_filter_dir / "selected_quality.csv"),
            "--output-proposal-json", str(round2_manual_dir / "cleaned_proposals_after_mask.json"),
            "--output-selected-manifest", str(round2_selected_clean),
            "--output-qa-summary", str(round2_mask_summary),
            "--round-id", "2",
        ])
        round2_selected_manifest = round2_selected_clean
        _check_round_qa_gate(
            summary_path=round2_mask_summary,
            pass_rate_min=float(args.qa_gate_pass_rate),
            polypgen_pass_rate_min=float(args.qa_gate_polypgen_pass_rate),
            boundary_bad_max=float(args.qa_gate_boundary_bad_max),
            round_id=2,
        )
        if bool(args.manual_pass_only_for_student):
            pass_ids_r2 = _manual_pass_ids(str(round2_mask_csv))
            reviewed_manifest_r2 = round2_filter_dir / "selected_manifest_manual_pass.csv"
            kept_n_r2 = _filter_manifest_by_ids(str(round2_selected_manifest), pass_ids_r2, str(reviewed_manifest_r2))
            print(
                "[round2 manual pass filter] "
                f"manual_pass_ids={len(pass_ids_r2)} kept_in_selected={kept_n_r2} "
                f"manifest={reviewed_manifest_r2}"
            )
            if kept_n_r2 > 0:
                round2_selected_for_student = reviewed_manifest_r2

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
        selected_manifests=[str(round1_selected_for_student), str(round2_selected_for_student)],
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
        "run_root": str(work),
        "teacher_r0": str(teacher_r0),
        "teacher_r1": str(teacher_r1) if teacher_refresh else str(teacher_r0),
        "lora_qc_metrics": lora_qc_metrics,
        "quality_calibration_json": str(calibration_json),
        "enable_quality_calibration": bool(args.enable_quality_calibration),
        "calibration_expected_dice_min": float(args.calibration_expected_dice_min),
        "calibration_polypgen_expected_dice_min": float(args.calibration_polypgen_expected_dice_min),
        "calibration_expected_dice_mid": float(args.calibration_expected_dice_mid),
        "calibration_expected_dice_mid_scale": float(args.calibration_expected_dice_mid_scale),
        "manual_review_enabled": bool(manual_enabled),
        "manual_review_per_round": int(args.manual_review_per_round),
        "manual_box_review_count": int(box_count),
        "manual_mask_review_count": int(mask_count),
        "manual_mask_hard_fraction": float(args.manual_mask_hard_fraction),
        "manual_pass_only_for_refresh": bool(args.manual_pass_only_for_refresh),
        "manual_pass_only_for_student": bool(args.manual_pass_only_for_student),
        "qa_gate_pass_rate": float(args.qa_gate_pass_rate),
        "qa_gate_polypgen_pass_rate": float(args.qa_gate_polypgen_pass_rate),
        "qa_gate_boundary_bad_max": float(args.qa_gate_boundary_bad_max),
        "tiered_pseudo": bool(args.tiered_pseudo),
        "manual_strict_proposals": bool(args.manual_strict_proposals),
        "pseudo_auto_proposal_mode": str(args.pseudo_auto_proposal_mode),
        "pseudo_auto_proposal_mode_effective": str(effective_auto_proposal_mode),
        "pseudo_proposal_mix_mode": str(args.pseudo_proposal_mix_mode),
        "pseudo_proposal_mix_mode_effective": str(effective_mix_mode),
        "teacher_subset_filter": str(args.teacher_subset_filter),
        "teacher_refresh_subset_filter": str(args.teacher_refresh_subset_filter),
        "lora_qc_subset_filter": str(args.lora_qc_subset_filter),
        "lora_qc_polypgen_dice_min": float(args.lora_qc_polypgen_dice_min),
        "lora_qc_polypgen_bf1_min": float(args.lora_qc_polypgen_bf1_min),
        "lora_enable_augment": bool(args.lora_enable_augment),
        "lora_epochs": int(args.lora_epochs),
        "lora_lr": float(args.lora_lr),
        "lora_weight_decay": float(args.lora_weight_decay),
        "lora_encoder_rank": int(args.lora_encoder_rank),
        "lora_encoder_alpha": int(args.lora_encoder_alpha),
        "lora_decoder_rank": int(args.lora_decoder_rank),
        "lora_decoder_alpha": int(args.lora_decoder_alpha),
        "lora_stage_epochs": str(args.lora_stage_epochs),
        "lora_stage_lrs": str(args.lora_stage_lrs),
        "lora_prompt_mode": str(args.lora_prompt_mode),
        "lora_loss_dice": float(args.lora_loss_dice),
        "lora_loss_focal": float(args.lora_loss_focal),
        "lora_loss_boundary": float(args.lora_loss_boundary),
        "lora_augment_prob": float(args.lora_augment_prob),
        "lora_box_jitter_scale": float(args.lora_box_jitter_scale),
        "lora_box_jitter_shift": float(args.lora_box_jitter_shift),
        "lora_box_full_image_prob": float(args.lora_box_full_image_prob),
        "box_localizer_checkpoint": str(args.box_localizer_checkpoint),
        "box_required_train_mode": str(args.box_required_train_mode),
        "box_review_polypgen_min": int(args.box_review_polypgen_min),
        "pseudo_proposal_jitter_scales": str(args.pseudo_proposal_jitter_scales),
        "pseudo_proposal_jitter_shifts": str(args.pseudo_proposal_jitter_shifts),
        "pseudo_score_weight_center_prior_polypgen": float(args.pseudo_score_weight_center_prior_polypgen),
        "pseudo_score_bias_preset_effective": float(effective_score_bias_preset),
        "pseudo_score_bias_auto_effective": float(effective_score_bias_auto),
        "pseudo_two_pass_refine": bool(args.pseudo_two_pass_refine),
        "pseudo_two_pass_min_first_quality": float(args.pseudo_two_pass_min_first_quality),
        "pseudo_two_pass_min_first_area_ratio": float(args.pseudo_two_pass_min_first_area_ratio),
        "pseudo_two_pass_min_gain": float(args.pseudo_two_pass_min_gain),
        "pseudo_two_pass_score_bonus": float(args.pseudo_two_pass_score_bonus),
        "pseudo_two_pass_box_expand_ratio": float(args.pseudo_two_pass_box_expand_ratio),
        "pseudo_two_pass_pos_points": int(args.pseudo_two_pass_pos_points),
        "pseudo_two_pass_neg_points": int(args.pseudo_two_pass_neg_points),
        "pseudo_two_pass_use_reflection_neg": bool(args.pseudo_two_pass_use_reflection_neg),
        "pseudo_fallback_full_image_effective": bool(effective_fallback_full_image),
        "mid_weight_scale": float(args.mid_weight_scale),
        "polypgen_mid_weight_scale": float(args.polypgen_mid_weight_scale),
        "round1_selected_manifest": str(round1_selected_manifest),
        "round1_selected_for_refresh": str(round1_selected_for_refresh),
        "round1_selected_for_student": str(round1_selected_for_student),
        "round2_selected_manifest": str(round2_selected_manifest),
        "round2_selected_for_student": str(round2_selected_for_student),
        "round1_hard_masks_dir": str(round1 / "pseudo" / "hard_masks"),
        "round2_hard_masks_dir": str(round2 / "pseudo" / "hard_masks"),
        "round1_manual_box_template": str(round1_box_template),
        "round1_manual_mask_template": str(round1_mask_template),
        "round2_manual_box_template": str(round2_box_template),
        "round2_manual_mask_template": str(round2_mask_template),
        "flywheel_mask_gallery_html": flywheel_gallery,
        "student_manifest": str(student_manifest),
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

