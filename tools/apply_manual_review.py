import argparse
import csv
import json
from pathlib import Path


def _read_csv(path: str) -> tuple[list[dict], list[str]]:
    p = Path(path)
    if not p.exists():
        return [], []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _read_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: str, payload: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _to_decision(text: str, default: str) -> str:
    t = str(text or "").strip().lower()
    return t if t else str(default).strip().lower()


def _is_reject(decision: str) -> bool:
    return str(decision).strip().lower() in {"reject", "drop", "discard", "bad"}


def _is_override(decision: str) -> bool:
    return str(decision).strip().lower() in {"override_box", "override", "edit_box", "fix_box"}


def _is_keep(decision: str) -> bool:
    return str(decision).strip().lower() in {"keep_auto", "keep", "pass", "ok"}


def _parse_box(row: dict) -> list[float]:
    keys = ["x0", "y0", "x1", "y1"]
    vals = []
    for k in keys:
        v = str(row.get(k, "")).strip()
        if v == "":
            raise ValueError(f"Missing {k} for override_box.")
        vals.append(float(v))
    return vals


def _source_map_from_rows(rows: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in rows:
        pid = str(r.get("id", "")).strip()
        if not pid:
            continue
        src = str(r.get("source", "")).strip()
        if src and pid not in out:
            out[pid] = src
    return out


def _contains_boundary_keyword(reason_text: str, keywords: list[str]) -> bool:
    text = str(reason_text or "").strip().lower()
    if not text:
        return False
    for k in keywords:
        kk = str(k).strip().lower()
        if kk and kk in text:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Apply manual box/mask review decisions and produce cleaned artifacts.")
    parser.add_argument("--proposal-json", type=str, required=True)
    parser.add_argument("--proposal-csv", type=str, default="")
    parser.add_argument("--box-review-csv", type=str, default="")
    parser.add_argument("--mask-review-csv", type=str, default="")
    parser.add_argument("--selected-manifest", type=str, default="")
    parser.add_argument("--quality-csv", type=str, default="")
    parser.add_argument("--output-proposal-json", type=str, required=True)
    parser.add_argument("--output-id-filter", type=str, default="")
    parser.add_argument("--output-selected-manifest", type=str, default="")
    parser.add_argument("--output-qa-summary", type=str, required=True)
    parser.add_argument("--default-box-decision", type=str, default="keep_auto")
    parser.add_argument("--default-mask-decision", type=str, default="pass")
    parser.add_argument(
        "--boundary-reason-keywords",
        type=str,
        default="boundary,edge,border,jagged,overseg,underseg,expand,shrink,\u952f\u9f7f,\u5916\u6269,\u5185\u7f29",
    )
    parser.add_argument("--boundary-edge-quality-threshold", type=float, default=0.40)
    parser.add_argument("--round-id", type=int, default=0)
    args = parser.parse_args()

    proposal_map = _read_json(args.proposal_json)
    proposals: dict[str, list[float]] = {}
    for k, v in proposal_map.items():
        if isinstance(v, list) and len(v) == 4:
            try:
                proposals[str(k)] = [float(x) for x in v]
            except Exception:
                continue

    proposal_rows, _ = _read_csv(args.proposal_csv) if args.proposal_csv else ([], [])
    selected_rows, selected_fields = _read_csv(args.selected_manifest) if args.selected_manifest else ([], [])
    quality_rows, _ = _read_csv(args.quality_csv) if args.quality_csv else ([], [])
    quality_edge = {}
    for r in quality_rows:
        pid = str(r.get("id", "")).strip()
        if not pid:
            continue
        try:
            quality_edge[pid] = float(r.get("edge_quality", 0.0))
        except Exception:
            quality_edge[pid] = 0.0

    source_map = {}
    source_map.update(_source_map_from_rows(proposal_rows))
    source_map.update(_source_map_from_rows(selected_rows))
    source_map.update(_source_map_from_rows(quality_rows))

    box_rows, _ = _read_csv(args.box_review_csv) if args.box_review_csv else ([], [])
    mask_rows, _ = _read_csv(args.mask_review_csv) if args.mask_review_csv else ([], [])

    box_reviewed = []
    mask_reviewed = []
    reject_ids: set[str] = set()
    override_ids: set[str] = set()
    unknown_ids: list[str] = []
    box_errors: list[str] = []
    boundary_keywords = [x.strip() for x in str(args.boundary_reason_keywords).split(",") if x.strip()]

    for r in box_rows:
        pid = str(r.get("id", "")).strip()
        if not pid:
            continue
        decision = _to_decision(r.get("decision", ""), args.default_box_decision)
        reason = str(r.get("reason", "")).strip()

        if _is_reject(decision):
            reject_ids.add(pid)
        elif _is_override(decision):
            try:
                box = _parse_box(r)
                proposals[pid] = box
                override_ids.add(pid)
            except Exception as e:
                box_errors.append(f"id={pid}: {e}")
                reject_ids.add(pid)
        elif _is_keep(decision):
            pass
        else:
            box_errors.append(f"id={pid}: unsupported box decision={decision}")
            reject_ids.add(pid)

        if pid not in proposals and not _is_reject(decision):
            unknown_ids.append(pid)

        box_reviewed.append(
            {
                "id": pid,
                "source": source_map.get(pid, ""),
                "decision": decision,
                "reason": reason,
                "is_reject": int(_is_reject(decision)),
            }
        )

    for r in mask_rows:
        pid = str(r.get("id", "")).strip()
        if not pid:
            continue
        decision = _to_decision(r.get("decision", ""), args.default_mask_decision)
        reason = str(r.get("reason", "")).strip()

        if _is_reject(decision):
            reject_ids.add(pid)

        boundary_bad = False
        if _is_reject(decision):
            if _contains_boundary_keyword(reason, boundary_keywords):
                boundary_bad = True
            elif pid in quality_edge and float(quality_edge[pid]) < float(args.boundary_edge_quality_threshold):
                boundary_bad = True

        mask_reviewed.append(
            {
                "id": pid,
                "source": source_map.get(pid, ""),
                "decision": decision,
                "reason": reason,
                "is_reject": int(_is_reject(decision)),
                "boundary_bad": int(boundary_bad),
            }
        )

    cleaned_proposals = {k: v for k, v in proposals.items() if k not in reject_ids}
    _write_json(args.output_proposal_json, cleaned_proposals)

    if args.output_id_filter:
        out_ids = Path(args.output_id_filter)
        out_ids.parent.mkdir(parents=True, exist_ok=True)
        out_ids.write_text("\n".join(sorted(cleaned_proposals.keys())), encoding="utf-8")

    cleaned_selected_rows = []
    selected_before = len(selected_rows)
    if selected_rows:
        cleaned_selected_rows = [r for r in selected_rows if str(r.get("id", "")).strip() not in reject_ids]
        if args.output_selected_manifest:
            if selected_fields:
                _write_csv(args.output_selected_manifest, cleaned_selected_rows, selected_fields)
            else:
                fields = list(cleaned_selected_rows[0].keys()) if cleaned_selected_rows else []
                _write_csv(args.output_selected_manifest, cleaned_selected_rows, fields)

    reviewed_all = box_reviewed + mask_reviewed
    pass_all = sum(1 for r in reviewed_all if int(r["is_reject"]) == 0)
    reviewed_all_n = len(reviewed_all)
    overall_pass_rate = float(pass_all) / float(max(1, reviewed_all_n))

    poly_all = [r for r in reviewed_all if "polypgen" in str(r.get("source", "")).lower()]
    poly_pass = sum(1 for r in poly_all if int(r["is_reject"]) == 0)
    polypgen_pass_rate = float(poly_pass) / float(max(1, len(poly_all))) if poly_all else 1.0

    boundary_bad_n = sum(int(r["boundary_bad"]) for r in mask_reviewed)
    boundary_bad_ratio = float(boundary_bad_n) / float(max(1, len(mask_reviewed))) if mask_reviewed else 0.0

    qa_summary = {
        "round_id": int(args.round_id),
        "num_box_reviewed": len(box_reviewed),
        "num_mask_reviewed": len(mask_reviewed),
        "num_manual_reviewed_total": reviewed_all_n,
        "num_manual_pass_total": pass_all,
        "overall_pass_rate": overall_pass_rate,
        "num_polypgen_reviewed": len(poly_all),
        "num_polypgen_pass": poly_pass,
        "polypgen_pass_rate": polypgen_pass_rate,
        "num_boundary_bad": int(boundary_bad_n),
        "boundary_bad_ratio": boundary_bad_ratio,
        "num_reject_ids": len(reject_ids),
        "num_override_box": len(override_ids),
        "num_unknown_ids": len(unknown_ids),
        "unknown_ids": sorted(set(unknown_ids)),
        "box_errors": box_errors,
        "selected_manifest_before": int(selected_before),
        "selected_manifest_after": int(len(cleaned_selected_rows)),
        "output_proposal_json": str(args.output_proposal_json),
        "output_id_filter": str(args.output_id_filter) if args.output_id_filter else "",
        "output_selected_manifest": str(args.output_selected_manifest) if args.output_selected_manifest else "",
        "reject_ids": sorted(reject_ids),
    }
    _write_json(args.output_qa_summary, qa_summary)
    print(json.dumps(qa_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
