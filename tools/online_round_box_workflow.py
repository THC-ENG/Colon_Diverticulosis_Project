import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


LEDGER_FIELDS = [
    "id",
    "image_path",
    "source",
    "center",
    "status",
    "round_tag",
    "batch_id",
    "decision",
    "reason",
    "x0",
    "y0",
    "x1",
    "y1",
    "updated_at",
]

EVENT_FIELDS = [
    "time",
    "id",
    "round_tag",
    "batch_id",
    "prev_status",
    "new_status",
    "decision",
    "reason",
    "x0",
    "y0",
    "x1",
    "y1",
]

TEMPLATE_FIELDS = [
    "id",
    "image_path",
    "source",
    "center",
    "decision",
    "x0",
    "y0",
    "x1",
    "y1",
    "reason",
    "batch_id",
    "round",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _fmt_float(v: float) -> str:
    return f"{float(v):.2f}"


def _normalize_decision(text: str) -> str:
    return str(text or "").strip().lower()


def _is_reject(decision: str) -> bool:
    return _normalize_decision(decision) in {"reject", "drop", "discard", "bad", "r"}


def _is_accept(decision: str) -> bool:
    return _normalize_decision(decision) in {
        "override_box",
        "override",
        "edit_box",
        "fix_box",
        "keep_auto",
        "keep",
        "pass",
        "ok",
        "o",
        "k",
    }


def _parse_box(row: dict) -> tuple[float, float, float, float] | None:
    x0 = _to_float(row.get("x0", "nan"), default=float("nan"))
    y0 = _to_float(row.get("y0", "nan"), default=float("nan"))
    x1 = _to_float(row.get("x1", "nan"), default=float("nan"))
    y1 = _to_float(row.get("y1", "nan"), default=float("nan"))
    vals = [x0, y0, x1, y1]
    if any(str(v) == "nan" for v in vals):
        return None
    if not (x1 > x0 and y1 > y0):
        return None
    return float(x0), float(y0), float(x1), float(y1)


def _largest_remainder_alloc(group_sizes: dict[str, int], n_total: int) -> dict[str, int]:
    if n_total <= 0:
        return {k: 0 for k in group_sizes}
    total = int(sum(max(0, v) for v in group_sizes.values()))
    if total <= 0:
        return {k: 0 for k in group_sizes}
    base = {}
    rema = []
    used = 0
    for k, v in group_sizes.items():
        frac = float(n_total) * float(max(0, v)) / float(total)
        b = int(frac)
        base[k] = b
        used += b
        rema.append((frac - b, k))
    remain = int(max(0, n_total - used))
    rema.sort(key=lambda x: x[0], reverse=True)
    for i in range(remain):
        _, k = rema[i % len(rema)]
        base[k] += 1
    return base


def _stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    n = int(max(0, n))
    if n <= 0 or not rows:
        return []
    if n >= len(rows):
        out = list(rows)
        random.Random(int(seed)).shuffle(out)
        return out

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        src = str(r.get("source", "")).strip() or "unknown_source"
        center = str(r.get("center", "")).strip() or "unknown_center"
        key = f"{src}||{center}"
        groups[key].append(r)

    rng = random.Random(int(seed))
    for g in groups.values():
        rng.shuffle(g)

    alloc = _largest_remainder_alloc({k: len(v) for k, v in groups.items()}, n)
    picked: list[dict] = []
    used_ids: set[str] = set()
    for key, k in sorted(alloc.items(), key=lambda x: x[0]):
        if k <= 0:
            continue
        for r in groups[key][: int(min(k, len(groups[key])))]:
            pid = str(r.get("id", "")).strip()
            if pid and pid not in used_ids:
                picked.append(r)
                used_ids.add(pid)

    if len(picked) < n:
        remain = [r for r in rows if str(r.get("id", "")).strip() not in used_ids]
        rng.shuffle(remain)
        need = int(n - len(picked))
        picked.extend(remain[: max(0, need)])

    rng.shuffle(picked)
    return picked[:n]


def _status_counts(rows: list[dict]) -> dict[str, int]:
    c = Counter(str(r.get("status", "")).strip() for r in rows)
    return dict(sorted(c.items(), key=lambda x: x[0]))


def _read_ledger(path: Path) -> list[dict]:
    rows, _ = _read_csv(path)
    for r in rows:
        for k in LEDGER_FIELDS:
            r.setdefault(k, "")
    return rows


def _write_ledger(path: Path, rows: list[dict]):
    _write_csv(path, rows, LEDGER_FIELDS)


def _read_manifest_u_large(manifest_path: Path) -> list[dict]:
    rows, _ = _read_csv(manifest_path)
    out = []
    seen = set()
    for r in rows:
        if str(r.get("subset", "")).strip() != "U_large":
            continue
        pid = str(r.get("id", "")).strip()
        if not pid or pid in seen:
            continue
        img = str(r.get("image_path", "")).strip()
        if not img:
            continue
        out.append(
            {
                "id": pid,
                "image_path": img,
                "source": str(r.get("source", "")).strip(),
                "center": str(r.get("center", "")).strip(),
            }
        )
        seen.add(pid)
    return out


def cmd_init(args):
    manifest_path = Path(args.manifest)
    ledger_path = Path(args.ledger)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if ledger_path.exists() and not bool(args.force):
        raise RuntimeError(f"Ledger already exists: {ledger_path}. Use --force to overwrite.")

    base = _read_manifest_u_large(manifest_path)
    now = _now_iso()
    rows = []
    for r in base:
        row = {
            "id": r["id"],
            "image_path": r["image_path"],
            "source": r["source"],
            "center": r["center"],
            "status": "pool",
            "round_tag": "",
            "batch_id": "",
            "decision": "",
            "reason": "",
            "x0": "",
            "y0": "",
            "x1": "",
            "y1": "",
            "updated_at": now,
        }
        rows.append(row)
    _write_ledger(ledger_path, rows)
    print(
        json.dumps(
            {
                "ledger": str(ledger_path),
                "num_u_large": int(len(rows)),
                "status_counts": _status_counts(rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def cmd_sample(args):
    ledger_path = Path(args.ledger)
    template_csv = Path(args.template_csv)
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found: {ledger_path}")
    if template_csv.exists() and not bool(args.overwrite):
        raise RuntimeError(f"Template exists: {template_csv}. Use --overwrite to replace.")

    rows = _read_ledger(ledger_path)
    round_tag = f"r{int(args.round)}"
    batch_size = int(max(0, args.batch_size))
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    if bool(args.reset_sampled):
        now = _now_iso()
        for r in rows:
            if str(r.get("status", "")).strip() == f"sampled_{round_tag}":
                r["status"] = "pool"
                r["updated_at"] = now

    candidates = [r for r in rows if str(r.get("status", "")).strip() == "pool"]
    if not candidates:
        raise RuntimeError("No pool candidates left to sample.")

    if batch_size > len(candidates):
        batch_size = int(len(candidates))

    picked = _stratified_sample(candidates, n=batch_size, seed=int(args.seed))
    if not picked:
        raise RuntimeError("Sampling failed: no samples picked.")

    batch_id = str(args.batch_id).strip() or f"{round_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    picked_ids = {str(r.get("id", "")).strip() for r in picked}
    now = _now_iso()
    for r in rows:
        pid = str(r.get("id", "")).strip()
        if pid in picked_ids:
            r["status"] = f"sampled_{round_tag}"
            r["round_tag"] = round_tag
            r["batch_id"] = batch_id
            r["updated_at"] = now

    _write_ledger(ledger_path, rows)

    out_rows = []
    for r in picked:
        out_rows.append(
            {
                "id": str(r.get("id", "")).strip(),
                "image_path": str(r.get("image_path", "")).strip(),
                "source": str(r.get("source", "")).strip(),
                "center": str(r.get("center", "")).strip(),
                "decision": "",
                "x0": "",
                "y0": "",
                "x1": "",
                "y1": "",
                "reason": "",
                "batch_id": batch_id,
                "round": round_tag,
            }
        )
    _write_csv(template_csv, out_rows, TEMPLATE_FIELDS)

    grp = Counter(
        (str(r.get("source", "")).strip() or "unknown_source", str(r.get("center", "")).strip() or "unknown_center")
        for r in out_rows
    )
    print(
        json.dumps(
            {
                "round": round_tag,
                "batch_id": batch_id,
                "num_sampled": int(len(out_rows)),
                "template_csv": str(template_csv),
                "ledger": str(ledger_path),
                "sampled_groups_top20": [
                    {"source": k[0], "center": k[1], "n": int(v)}
                    for k, v in sorted(grp.items(), key=lambda x: x[1], reverse=True)[:20]
                ],
                "status_counts": _status_counts(rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def cmd_ingest(args):
    ledger_path = Path(args.ledger)
    reviewed_csv = Path(args.reviewed_csv)
    events_csv = Path(args.events_csv)
    ids_txt = Path(args.round_ids_txt)
    boxes_json = Path(args.round_boxes_json)
    reject_txt = Path(args.reject_ids_txt)
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found: {ledger_path}")
    if not reviewed_csv.exists():
        raise FileNotFoundError(f"Reviewed CSV not found: {reviewed_csv}")

    rows = _read_ledger(ledger_path)
    by_id = {str(r.get("id", "")).strip(): r for r in rows}
    reviewed_rows, _ = _read_csv(reviewed_csv)
    round_tag = f"r{int(args.round)}"
    now = _now_iso()

    events = []
    stats = Counter()
    for rr in reviewed_rows:
        pid = str(rr.get("id", "")).strip()
        if not pid:
            continue
        base = by_id.get(pid)
        if base is None:
            stats["missing_in_ledger"] += 1
            continue
        decision = _normalize_decision(rr.get("decision", ""))
        if not decision:
            stats["no_decision"] += 1
            continue

        prev_status = str(base.get("status", "")).strip()
        reason = str(rr.get("reason", "")).strip()
        event = {
            "time": now,
            "id": pid,
            "round_tag": round_tag,
            "batch_id": str(rr.get("batch_id", "")).strip() or str(base.get("batch_id", "")).strip(),
            "prev_status": prev_status,
            "new_status": prev_status,
            "decision": decision,
            "reason": reason,
            "x0": "",
            "y0": "",
            "x1": "",
            "y1": "",
        }

        if _is_reject(decision):
            new_status = f"rejected_{round_tag}"
            base["status"] = new_status
            base["round_tag"] = round_tag
            base["decision"] = decision
            base["reason"] = reason
            base["x0"] = ""
            base["y0"] = ""
            base["x1"] = ""
            base["y1"] = ""
            base["updated_at"] = now
            event["new_status"] = new_status
            stats["rejected"] += 1
        elif _is_accept(decision):
            box = _parse_box(rr)
            if box is None:
                new_status = f"rejected_{round_tag}"
                base["status"] = new_status
                base["round_tag"] = round_tag
                base["decision"] = decision
                base["reason"] = reason or "invalid_box"
                base["x0"] = ""
                base["y0"] = ""
                base["x1"] = ""
                base["y1"] = ""
                base["updated_at"] = now
                event["new_status"] = new_status
                event["reason"] = base["reason"]
                stats["invalid_box_rejected"] += 1
            else:
                x0, y0, x1, y1 = box
                new_status = f"boxed_{round_tag}"
                base["status"] = new_status
                base["round_tag"] = round_tag
                base["decision"] = decision
                base["reason"] = reason
                base["x0"] = _fmt_float(x0)
                base["y0"] = _fmt_float(y0)
                base["x1"] = _fmt_float(x1)
                base["y1"] = _fmt_float(y1)
                base["updated_at"] = now
                event["new_status"] = new_status
                event["x0"] = base["x0"]
                event["y0"] = base["y0"]
                event["x1"] = base["x1"]
                event["y1"] = base["y1"]
                stats["boxed"] += 1
        else:
            stats["unsupported_decision"] += 1
            continue

        events.append(event)

    _write_ledger(ledger_path, rows)
    _append_csv(events_csv, events, EVENT_FIELDS)

    boxed_rows = [r for r in rows if str(r.get("status", "")).strip() == f"boxed_{round_tag}"]
    boxed_ids = sorted(str(r.get("id", "")).strip() for r in boxed_rows if str(r.get("id", "")).strip())
    ids_txt.parent.mkdir(parents=True, exist_ok=True)
    ids_txt.write_text("\n".join(boxed_ids), encoding="utf-8")

    box_map = {}
    for r in boxed_rows:
        pid = str(r.get("id", "")).strip()
        box = _parse_box(r)
        if not pid or box is None:
            continue
        box_map[pid] = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    boxes_json.parent.mkdir(parents=True, exist_ok=True)
    boxes_json.write_text(json.dumps(box_map, indent=2, ensure_ascii=False), encoding="utf-8")

    reject_ids = sorted(
        str(r.get("id", "")).strip()
        for r in rows
        if str(r.get("status", "")).strip().startswith("rejected_") and str(r.get("id", "")).strip()
    )
    reject_txt.parent.mkdir(parents=True, exist_ok=True)
    reject_txt.write_text("\n".join(reject_ids), encoding="utf-8")

    print(
        json.dumps(
            {
                "round": round_tag,
                "reviewed_csv": str(reviewed_csv),
                "events_written": int(len(events)),
                "stats": dict(stats),
                "boxed_total_round": int(len(boxed_ids)),
                "rejected_total_global": int(len(reject_ids)),
                "round_ids_txt": str(ids_txt),
                "round_boxes_json": str(boxes_json),
                "reject_ids_txt": str(reject_txt),
                "status_counts": _status_counts(rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def cmd_report(args):
    ledger_path = Path(args.ledger)
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found: {ledger_path}")
    rows = _read_ledger(ledger_path)

    counts = _status_counts(rows)
    boxed_r1 = sum(1 for r in rows if str(r.get("status", "")).strip() == "boxed_r1")
    boxed_r2 = sum(1 for r in rows if str(r.get("status", "")).strip() == "boxed_r2")
    rejected = sum(1 for r in rows if str(r.get("status", "")).strip().startswith("rejected_"))
    sampled_r1 = sum(1 for r in rows if str(r.get("status", "")).strip() == "sampled_r1")
    sampled_r2 = sum(1 for r in rows if str(r.get("status", "")).strip() == "sampled_r2")
    pool = sum(1 for r in rows if str(r.get("status", "")).strip() == "pool")

    out = {
        "ledger": str(ledger_path),
        "num_total": int(len(rows)),
        "status_counts": counts,
        "boxed_r1": int(boxed_r1),
        "boxed_r2": int(boxed_r2),
        "rejected_global": int(rejected),
        "sampled_pending_r1": int(sampled_r1),
        "sampled_pending_r2": int(sampled_r2),
        "pool_remaining": int(pool),
    }
    if int(args.target_per_round) > 0:
        target = int(args.target_per_round)
        out["target_per_round"] = target
        out["gap_r1"] = int(max(0, target - boxed_r1))
        out["gap_r2"] = int(max(0, target - boxed_r2))
    print(json.dumps(out, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online U_large sampling + box review ledger workflow.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize U_large ledger from manifest.")
    p_init.add_argument("--manifest", type=str, required=True)
    p_init.add_argument("--ledger", type=str, required=True)
    p_init.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    p_init.set_defaults(func=cmd_init)

    p_sample = sub.add_parser("sample", help="Sample next stratified batch from pool and export review template.")
    p_sample.add_argument("--ledger", type=str, required=True)
    p_sample.add_argument("--round", type=int, required=True, choices=[1, 2])
    p_sample.add_argument("--batch-size", type=int, default=100)
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument("--batch-id", type=str, default="")
    p_sample.add_argument("--template-csv", type=str, required=True)
    p_sample.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    p_sample.add_argument("--reset-sampled", action=argparse.BooleanOptionalAction, default=False)
    p_sample.set_defaults(func=cmd_sample)

    p_ingest = sub.add_parser("ingest", help="Ingest reviewed CSV, update ledger, export round ids/boxes/reject list.")
    p_ingest.add_argument("--ledger", type=str, required=True)
    p_ingest.add_argument("--round", type=int, required=True, choices=[1, 2])
    p_ingest.add_argument("--reviewed-csv", type=str, required=True)
    p_ingest.add_argument("--events-csv", type=str, required=True)
    p_ingest.add_argument("--round-ids-txt", type=str, required=True)
    p_ingest.add_argument("--round-boxes-json", type=str, required=True)
    p_ingest.add_argument("--reject-ids-txt", type=str, required=True)
    p_ingest.set_defaults(func=cmd_ingest)

    p_report = sub.add_parser("report", help="Show current progress from ledger.")
    p_report.add_argument("--ledger", type=str, required=True)
    p_report.add_argument("--target-per-round", type=int, default=500)
    p_report.set_defaults(func=cmd_report)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

