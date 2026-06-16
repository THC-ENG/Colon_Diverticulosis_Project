import argparse
import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


KEEP_DECISIONS = {"pass", "keep", "accept", "approved", "yes", "y", "1", "ok"}
MAYBE_DECISIONS = {"maybe", "weak", "low", "low_weight", "m", "2"}
REJECT_DECISIONS = {"reject", "drop", "discard", "bad", "no", "n", "0"}


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


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _normalize_decision(text: str) -> str:
    t = str(text or "").strip().lower()
    if not t:
        return ""
    if t in KEEP_DECISIONS:
        return "pass"
    if t in MAYBE_DECISIONS:
        return "maybe"
    if t in REJECT_DECISIONS:
        return "reject"
    return t


def _resolve_path(path_text: str, anchor: Path) -> Path:
    p = Path(str(path_text or "").strip())
    if not str(p):
        return Path("")
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    cands = [anchor.parent / p, anchor.parent.parent / p, Path.cwd() / p]
    for c in cands:
        if c.exists():
            return c.resolve()
    return cands[0].resolve()


def _find_panel_for_id(panel_dir: Path, sid: str) -> Path:
    for ext in (".jpg", ".png", ".jpeg", ".webp", ".bmp"):
        p = panel_dir / f"{sid}{ext}"
        if p.exists():
            return p
    return Path("")


def _iter_rows_by_id(rows: Iterable[dict]) -> dict[str, dict]:
    out = {}
    for r in rows:
        sid = str(r.get("id", "")).strip()
        if sid and sid not in out:
            out[sid] = r
    return out


def cmd_init_template(args):
    quality_csv = Path(args.quality_csv)
    manifest_csv = Path(args.candidates_manifest)
    output_csv = Path(args.output_csv)
    panel_dir = Path(args.panel_dir) if str(args.panel_dir).strip() else Path("")

    if not quality_csv.exists():
        raise FileNotFoundError(f"quality csv not found: {quality_csv}")
    if not manifest_csv.exists():
        raise FileNotFoundError(f"candidates manifest not found: {manifest_csv}")
    if panel_dir and not panel_dir.exists():
        raise FileNotFoundError(f"panel dir not found: {panel_dir}")

    quality_rows, _ = _read_csv(quality_csv)
    manifest_rows, _ = _read_csv(manifest_csv)
    manifest_by_id = _iter_rows_by_id(manifest_rows)

    rows = []
    for q in quality_rows:
        sid = str(q.get("id", "")).strip()
        if not sid:
            continue
        m = manifest_by_id.get(sid, {})
        panel_path = str(q.get("panel_path", "")).strip()
        if panel_dir:
            found = _find_panel_for_id(panel_dir, sid)
            if found:
                panel_path = str(found)
        row = {
            "id": sid,
            "decision": "",
            "reason": "",
            "source": str(q.get("source", m.get("source", ""))).strip(),
            "quality_post": str(q.get("quality_post", "")),
            "quality": str(q.get("quality", "")),
            "boundary_quality": str(q.get("boundary_quality", "")),
            "edge_quality": str(q.get("edge_quality", "")),
            "consistency_iou": str(q.get("consistency_iou", "")),
            "pseudo_weight_final": str(m.get("pseudo_weight_final", q.get("pseudo_weight_final", ""))),
            "tier": str(m.get("tier", q.get("tier", ""))).strip(),
            "panel_path": panel_path,
            "image_path": str(m.get("image_path", q.get("image_path", ""))).strip(),
            "hard_mask_path": str(m.get("mask_path", q.get("hard_mask_path", ""))).strip(),
            "soft_path": str(m.get("soft_path", q.get("soft_path", ""))).strip(),
            "edge_path": str(m.get("edge_path", q.get("edge_path", ""))).strip(),
        }
        rows.append(row)

    if output_csv.exists() and not bool(args.overwrite):
        old_rows, _ = _read_csv(output_csv)
        old_by_id = _iter_rows_by_id(old_rows)
        for r in rows:
            old = old_by_id.get(str(r.get("id", "")).strip())
            if old is None:
                continue
            r["decision"] = str(old.get("decision", "")).strip()
            r["reason"] = str(old.get("reason", "")).strip()

    if bool(args.sort_by_score):
        rows = sorted(rows, key=lambda x: _to_float(x.get("quality_post", x.get("quality", 0.0)), 0.0))

    fields = [
        "id",
        "decision",
        "reason",
        "source",
        "quality_post",
        "quality",
        "boundary_quality",
        "edge_quality",
        "consistency_iou",
        "pseudo_weight_final",
        "tier",
        "panel_path",
        "image_path",
        "hard_mask_path",
        "soft_path",
        "edge_path",
    ]
    _write_csv(output_csv, rows, fields)
    print(
        {
            "output_csv": str(output_csv),
            "rows": len(rows),
            "with_existing_decision": sum(1 for r in rows if str(r.get("decision", "")).strip()),
        }
    )


def _overlay_mask(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    if image_bgr.shape[:2] != mask_u8.shape[:2]:
        mask_u8 = cv2.resize(mask_u8, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = (mask_u8 > 127).astype(np.uint8)
    out = image_bgr.copy()
    color = np.zeros_like(out)
    color[:, :, 2] = 255
    alpha = 0.35
    out[m > 0] = cv2.addWeighted(out[m > 0], 1.0 - alpha, color[m > 0], alpha, 0.0)
    return out


class YNReviewer:
    def __init__(self, rows: list[dict], fields: list[str], output_csv: Path, max_w: int, max_h: int):
        self.rows = rows
        self.fields = fields
        self.output_csv = output_csv
        self.max_w = int(max_w)
        self.max_h = int(max_h)
        self.index = 0
        self.window = "Round1 YN Review"
        self.current = None
        self.scale = 1.0
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

    def _save(self):
        _write_csv(self.output_csv, self.rows, self.fields)
        keep_n, maybe_n, rej_n, pending_n = self._counts()
        print(f"[saved] keep={keep_n} maybe={maybe_n} reject={rej_n} pending={pending_n} -> {self.output_csv}")

    def _counts(self):
        keep_n, maybe_n, rej_n, pending_n = 0, 0, 0, 0
        for r in self.rows:
            d = _normalize_decision(r.get("decision", ""))
            if d == "pass":
                keep_n += 1
            elif d == "maybe":
                maybe_n += 1
            elif d == "reject":
                rej_n += 1
            else:
                pending_n += 1
        return keep_n, maybe_n, rej_n, pending_n

    def _load_display_image(self, row: dict) -> np.ndarray:
        panel_path = _resolve_path(str(row.get("panel_path", "")).strip(), self.output_csv)
        if panel_path and panel_path.exists():
            img = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
            if img is not None:
                return img

        img_path = _resolve_path(str(row.get("image_path", "")).strip(), self.output_csv)
        hard_path = _resolve_path(str(row.get("hard_mask_path", "")).strip(), self.output_csv)
        base = cv2.imread(str(img_path), cv2.IMREAD_COLOR) if img_path and img_path.exists() else None
        if base is None:
            canvas = np.zeros((900, 1400, 3), dtype=np.uint8)
            cv2.putText(canvas, "Image Not Found", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, str(img_path)[:180], (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1, cv2.LINE_AA)
            return canvas

        if hard_path and hard_path.exists():
            m = cv2.imread(str(hard_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                base = _overlay_mask(base, m)
        return base

    def _render(self):
        row = self.rows[self.index]
        image = self._load_display_image(row)
        h, w = image.shape[:2]
        self.scale = min(float(self.max_w) / float(max(1, w)), float(self.max_h) / float(max(1, h)), 1.0)
        self.scale = max(1e-6, self.scale)
        disp = cv2.resize(image, (int(round(w * self.scale)), int(round(h * self.scale))), interpolation=cv2.INTER_LINEAR)

        keep_n, maybe_n, rej_n, pending_n = self._counts()
        sid = str(row.get("id", "")).strip()
        src = str(row.get("source", "")).strip()
        decision = _normalize_decision(row.get("decision", "")) or "-"
        q = str(row.get("quality_post", "") or row.get("quality", ""))
        bq = str(row.get("boundary_quality", ""))
        text1 = f"{self.index + 1}/{len(self.rows)} id:{sid} source:{src}"
        text2 = f"decision:{decision} keep:{keep_n} maybe:{maybe_n} reject:{rej_n} pending:{pending_n}"
        text3 = f"quality:{q} boundary:{bq}"
        text4 = "keys: y=keep  m=maybe-low-weight  n=reject  u=clear  <-/->=prev/next  s=save  q=save&quit"
        cv2.putText(disp, text1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, text2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (180, 255, 180), 2, cv2.LINE_AA)
        cv2.putText(disp, text3, (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (200, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, text4, (12, max(24, disp.shape[0] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.imshow(self.window, disp)

    def run(self):
        if not self.rows:
            print("No rows to review.")
            return
        while True:
            self._render()
            key = cv2.waitKeyEx(20)
            if key < 0:
                continue
            row = self.rows[self.index]
            moved = False
            if key in (ord("y"), ord("Y")):
                row["decision"] = "pass"
                self._save()
                if self.index < len(self.rows) - 1:
                    self.index += 1
                    moved = True
            elif key in (ord("n"), ord("N")):
                row["decision"] = "reject"
                self._save()
                if self.index < len(self.rows) - 1:
                    self.index += 1
                    moved = True
            elif key in (ord("m"), ord("M")):
                row["decision"] = "maybe"
                self._save()
                if self.index < len(self.rows) - 1:
                    self.index += 1
                    moved = True
            elif key in (ord("u"), ord("U"), ord("c"), ord("C")):
                row["decision"] = ""
                self._save()
            elif key in (ord("s"), ord("S")):
                self._save()
            elif key in (ord("q"), ord("Q"), 27):
                self._save()
                break
            elif key in (2555904, ord("d"), ord("D"), 32):  # right / d / space
                if self.index < len(self.rows) - 1:
                    self.index += 1
                    moved = True
            elif key in (2424832, ord("a"), ord("A")):  # left / a
                if self.index > 0:
                    self.index -= 1
                    moved = True

            if moved:
                continue
        cv2.destroyAllWindows()


def cmd_review(args):
    template_csv = Path(args.template_csv)
    output_csv = Path(args.output_csv) if str(args.output_csv).strip() else template_csv.with_name("reviewed_yn.csv")
    if not template_csv.exists():
        raise FileNotFoundError(f"template csv not found: {template_csv}")

    template_rows, template_fields = _read_csv(template_csv)
    if not template_rows:
        raise RuntimeError(f"empty template csv: {template_csv}")
    if "decision" not in template_fields:
        template_fields = ["decision"] + template_fields
        for r in template_rows:
            r["decision"] = ""

    rows = template_rows
    if output_csv.exists():
        out_rows, _ = _read_csv(output_csv)
        out_by_id = _iter_rows_by_id(out_rows)
        merged = []
        for r in template_rows:
            sid = str(r.get("id", "")).strip()
            old = out_by_id.get(sid)
            item = dict(r)
            if old is not None:
                item["decision"] = str(old.get("decision", item.get("decision", ""))).strip()
                if "reason" in item:
                    item["reason"] = str(old.get("reason", item.get("reason", ""))).strip()
            merged.append(item)
        rows = merged

    reviewer = YNReviewer(
        rows=rows,
        fields=template_fields,
        output_csv=output_csv,
        max_w=int(args.max_display_width),
        max_h=int(args.max_display_height),
    )

    if bool(args.start_from_first_pending):
        pending_idx = 0
        for i, r in enumerate(rows):
            if not _normalize_decision(r.get("decision", "")):
                pending_idx = i
                break
        reviewer.index = pending_idx
    else:
        reviewer.index = max(0, min(len(rows) - 1, int(args.start_index)))

    reviewer.run()


def cmd_export_manifest(args):
    reviewed_csv = Path(args.reviewed_csv)
    source_manifest = Path(args.source_manifest)
    output_manifest = Path(args.output_manifest)

    if not reviewed_csv.exists():
        raise FileNotFoundError(f"reviewed csv not found: {reviewed_csv}")
    if not source_manifest.exists():
        raise FileNotFoundError(f"source manifest not found: {source_manifest}")

    reviewed_rows, _ = _read_csv(reviewed_csv)
    source_rows, source_fields = _read_csv(source_manifest)
    decision_by_id = {
        str(r.get("id", "")).strip(): _normalize_decision(r.get("decision", ""))
        for r in reviewed_rows
        if str(r.get("id", "")).strip()
    }
    keep_ids = {sid for sid, d in decision_by_id.items() if d == "pass"}
    maybe_ids = {sid for sid, d in decision_by_id.items() if d == "maybe"}
    export_ids = set(keep_ids)
    if bool(args.include_maybe):
        export_ids.update(maybe_ids)

    kept_rows = []
    for r in source_rows:
        sid = str(r.get("id", "")).strip()
        if sid not in export_ids:
            continue
        row = dict(r)
        decision = decision_by_id.get(sid, "")
        row["review_decision"] = decision
        if decision == "maybe":
            scale = float(args.maybe_weight_scale)
            base_w = _to_float(row.get("pseudo_weight_final", row.get("pseudo_weight", 0.0)), 0.0)
            new_w = max(0.0, min(1.0, base_w * scale))
            row["pseudo_weight"] = new_w
            row["pseudo_weight_final"] = new_w
            row["review_weight_scale"] = scale
        else:
            row["review_weight_scale"] = 1.0
        kept_rows.append(row)

    out_fields = list(source_fields)
    for f in ["review_decision", "review_weight_scale"]:
        if f not in out_fields:
            out_fields.append(f)
    _write_csv(output_manifest, kept_rows, out_fields)

    reject_ids = {
        str(r.get("id", "")).strip()
        for r in reviewed_rows
        if _normalize_decision(r.get("decision", "")) == "reject"
    }
    summary = {
        "reviewed_csv": str(reviewed_csv),
        "source_manifest": str(source_manifest),
        "output_manifest": str(output_manifest),
        "num_review_rows": len(reviewed_rows),
        "num_keep_ids": len(keep_ids),
        "num_maybe_ids": len(maybe_ids),
        "num_reject_ids": len(reject_ids),
        "include_maybe": bool(args.include_maybe),
        "maybe_weight_scale": float(args.maybe_weight_scale),
        "num_output_rows": len(kept_rows),
    }
    print(summary)


def cmd_build_teacher_manifest(args):
    base_manifest = Path(args.base_manifest)
    approved_manifest = Path(args.approved_manifest)
    output_manifest = Path(args.output_manifest)
    keep_subsets = {
        s.strip()
        for s in str(args.keep_subsets).split(",")
        if str(s).strip()
    }
    if not keep_subsets:
        keep_subsets = {"L_small"}

    if not base_manifest.exists():
        raise FileNotFoundError(f"base manifest not found: {base_manifest}")
    if not approved_manifest.exists():
        raise FileNotFoundError(f"approved manifest not found: {approved_manifest}")

    base_rows, base_fields = _read_csv(base_manifest)
    approved_rows, approved_fields = _read_csv(approved_manifest)

    keep_rows = [r for r in base_rows if str(r.get("subset", "")).strip() in keep_subsets]
    merged = keep_rows + approved_rows

    fields = list(base_fields)
    for f in approved_fields:
        if f not in fields:
            fields.append(f)

    _write_csv(output_manifest, merged, fields)
    print(
        {
            "base_manifest": str(base_manifest),
            "approved_manifest": str(approved_manifest),
            "output_manifest": str(output_manifest),
            "keep_subsets": sorted(keep_subsets),
            "num_keep_rows": len(keep_rows),
            "num_approved_rows": len(approved_rows),
            "num_merged_rows": len(merged),
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Round1 manual y/n reviewer and manifest exporter.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-template", help="Build 500-row review template from quality+manifest.")
    p_init.add_argument("--quality-csv", type=str, required=True)
    p_init.add_argument("--candidates-manifest", type=str, required=True)
    p_init.add_argument("--panel-dir", type=str, default="")
    p_init.add_argument("--output-csv", type=str, required=True)
    p_init.add_argument("--sort-by-score", action=argparse.BooleanOptionalAction, default=False)
    p_init.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    p_init.set_defaults(func=cmd_init_template)

    p_review = sub.add_parser("review", help="Interactive y/n/m review with autosave.")
    p_review.add_argument("--template-csv", type=str, required=True)
    p_review.add_argument("--output-csv", type=str, default="")
    p_review.add_argument("--start-index", type=int, default=0)
    p_review.add_argument("--start-from-first-pending", action=argparse.BooleanOptionalAction, default=True)
    p_review.add_argument("--max-display-width", type=int, default=1700)
    p_review.add_argument("--max-display-height", type=int, default=980)
    p_review.set_defaults(func=cmd_review)

    p_export = sub.add_parser("export-manifest", help="Export approved/maybe manifest for downstream training.")
    p_export.add_argument("--reviewed-csv", type=str, required=True)
    p_export.add_argument("--source-manifest", type=str, required=True)
    p_export.add_argument("--output-manifest", type=str, required=True)
    p_export.add_argument("--include-maybe", action=argparse.BooleanOptionalAction, default=True)
    p_export.add_argument("--maybe-weight-scale", type=float, default=0.35)
    p_export.set_defaults(func=cmd_export_manifest)

    p_merge = sub.add_parser("build-teacher-manifest", help="Merge GT subsets + approved pseudo manifest.")
    p_merge.add_argument("--base-manifest", type=str, required=True)
    p_merge.add_argument("--approved-manifest", type=str, required=True)
    p_merge.add_argument("--output-manifest", type=str, required=True)
    p_merge.add_argument("--keep-subsets", type=str, default="L_small")
    p_merge.set_defaults(func=cmd_build_teacher_manifest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
