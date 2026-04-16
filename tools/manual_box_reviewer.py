import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


REQUIRED_FIELDS = ["id", "decision", "x0", "y0", "x1", "y1", "reason"]


def _read_csv(path: Path):
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


def _fmt(v: float) -> str:
    return f"{float(v):.2f}"


def _clamp_box(box: tuple[float, float, float, float], w: int, h: int):
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = max(0.0, min(float(w - 1), x0))
    y0 = max(0.0, min(float(h - 1), y0))
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x1 = max(x1, x0 + 1.0)
    y1 = max(y1, y0 + 1.0)
    return x0, y0, x1, y1


class ManualBoxReviewer:
    def __init__(
        self,
        rows: list[dict],
        fieldnames: list[str],
        output_csv: Path,
        max_w: int,
        max_h: int,
    ):
        self.rows = rows
        self.fieldnames = fieldnames
        self.output_csv = output_csv
        self.max_w = int(max_w)
        self.max_h = int(max_h)
        self.index = 0
        self.window = "Manual Box Review"

        self.current_image = None
        self.current_shape = (1, 1)
        self.scale = 1.0

        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        self.drawn_box = None

        self.auto_box_by_id = {}
        for row in self.rows:
            rid = str(row.get("id", "")).strip()
            self.auto_box_by_id[rid] = (
                _to_float(row.get("x0", 0.0), 0.0),
                _to_float(row.get("y0", 0.0), 0.0),
                _to_float(row.get("x1", 0.0), 0.0),
                _to_float(row.get("y1", 0.0), 0.0),
            )

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if self.current_image is None:
            return
        ox = float(x) / max(1e-6, self.scale)
        oy = float(y) / max(1e-6, self.scale)
        w, h = self.current_shape
        ox = max(0.0, min(float(w - 1), ox))
        oy = max(0.0, min(float(h - 1), oy))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (ox, oy)
            self.drag_end = (ox, oy)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_end = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.drag_end = (ox, oy)
            x0, y0, x1, y1 = _clamp_box(
                (self.drag_start[0], self.drag_start[1], self.drag_end[0], self.drag_end[1]),
                w=w,
                h=h,
            )
            self.drawn_box = (x0, y0, x1, y1)

    def _get_row(self):
        return self.rows[self.index]

    def _get_auto_box(self, row: dict):
        rid = str(row.get("id", "")).strip()
        return self.auto_box_by_id.get(rid, (0.0, 0.0, 1.0, 1.0))

    def _get_current_box(self, row: dict):
        d = str(row.get("decision", "")).strip().lower()
        if d == "override_box":
            return (
                _to_float(row.get("x0", 0.0), 0.0),
                _to_float(row.get("y0", 0.0), 0.0),
                _to_float(row.get("x1", 1.0), 1.0),
                _to_float(row.get("y1", 1.0), 1.0),
            )
        return self._get_auto_box(row)

    def _set_keep_auto(self, row: dict):
        x0, y0, x1, y1 = self._get_auto_box(row)
        row["decision"] = "keep_auto"
        row["x0"], row["y0"], row["x1"], row["y1"] = _fmt(x0), _fmt(y0), _fmt(x1), _fmt(y1)
        if "reason" in row and not str(row.get("reason", "")).strip():
            row["reason"] = ""

    def _set_override(self, row: dict, box: tuple[float, float, float, float]):
        x0, y0, x1, y1 = _clamp_box(box, w=self.current_shape[0], h=self.current_shape[1])
        row["decision"] = "override_box"
        row["x0"], row["y0"], row["x1"], row["y1"] = _fmt(x0), _fmt(y0), _fmt(x1), _fmt(y1)

    def _set_reject(self, row: dict):
        row["decision"] = "reject"
        row["x0"], row["y0"], row["x1"], row["y1"] = "", "", "", ""

    def _save(self):
        _write_csv(self.output_csv, self.rows, self.fieldnames)
        print(f"[saved] {self.output_csv}")

    def _load_image(self, row: dict):
        img_path = str(row.get("image_path", "")).strip()
        image = cv2.imread(img_path, cv2.IMREAD_COLOR) if img_path else None
        if image is None:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(image, "Image Not Found", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(image, img_path[:110], (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
        h, w = image.shape[:2]
        self.current_shape = (w, h)
        self.current_image = image
        self.drawn_box = None
        self.drag_start = None
        self.drag_end = None

    def _draw_box_scaled(self, canvas: np.ndarray, box, color, thickness=2):
        x0, y0, x1, y1 = box
        sx0 = int(round(float(x0) * self.scale))
        sy0 = int(round(float(y0) * self.scale))
        sx1 = int(round(float(x1) * self.scale))
        sy1 = int(round(float(y1) * self.scale))
        cv2.rectangle(canvas, (sx0, sy0), (sx1, sy1), color, thickness)

    def _render(self):
        row = self._get_row()
        base = self.current_image.copy()
        w, h = self.current_shape
        s = min(float(self.max_w) / float(max(1, w)), float(self.max_h) / float(max(1, h)), 1.0)
        self.scale = max(1e-6, s)
        disp = cv2.resize(base, (int(round(w * self.scale)), int(round(h * self.scale))), interpolation=cv2.INTER_LINEAR)

        auto_box = self._get_auto_box(row)
        self._draw_box_scaled(disp, auto_box, (0, 255, 255), 2)

        decision = str(row.get("decision", "")).strip().lower()
        if decision == "override_box":
            ov = self._get_current_box(row)
            self._draw_box_scaled(disp, ov, (0, 255, 0), 2)
        elif decision == "reject":
            cv2.putText(disp, "REJECT", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)

        if self.dragging and self.drag_start is not None and self.drag_end is not None:
            x0, y0, x1, y1 = _clamp_box(
                (self.drag_start[0], self.drag_start[1], self.drag_end[0], self.drag_end[1]),
                w=w,
                h=h,
            )
            self._draw_box_scaled(disp, (x0, y0, x1, y1), (255, 0, 0), 2)
        elif self.drawn_box is not None:
            self._draw_box_scaled(disp, self.drawn_box, (255, 0, 0), 2)

        total = len(self.rows)
        done = sum(1 for r in self.rows if str(r.get("decision", "")).strip())
        rid = str(row.get("id", "")).strip()
        source = str(row.get("source", "")).strip()
        text1 = f"{self.index + 1}/{total} done:{done} id:{rid} source:{source}"
        text2 = f"decision:{str(row.get('decision', '')).strip() or '-'}"
        text3 = "keys: k=keep  o=override(drawn)  r=reject  n=next  p=prev  c=clear_draw  s=save  q=save&quit"
        cv2.putText(disp, text1, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, text2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 255, 180), 2, cv2.LINE_AA)
        cv2.putText(disp, text3, (12, max(24, disp.shape[0] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

        cv2.imshow(self.window, disp)

    def run(self):
        if not self.rows:
            print("No rows to review.")
            return
        self._load_image(self._get_row())
        while True:
            self._render()
            key = cv2.waitKeyEx(20)
            if key < 0:
                continue

            row = self._get_row()
            moved = False

            if key in (ord("k"), ord("K")):
                self._set_keep_auto(row)
                self._save()
                if self.index < len(self.rows) - 1:
                    self.index += 1
                    moved = True
            elif key in (ord("o"), ord("O")):
                if self.drawn_box is not None:
                    self._set_override(row, self.drawn_box)
                    self._save()
                    if self.index < len(self.rows) - 1:
                        self.index += 1
                        moved = True
                else:
                    print("[hint] Draw a box first, then press 'o'.")
            elif key in (ord("r"), ord("R")):
                self._set_reject(row)
                self._save()
                if self.index < len(self.rows) - 1:
                    self.index += 1
                    moved = True
            elif key in (ord("n"), ord("N"), 32, 2555904):  # n / space / right arrow
                if self.index < len(self.rows) - 1:
                    self.index += 1
                    moved = True
            elif key in (ord("p"), ord("P"), 2424832):  # p / left arrow
                if self.index > 0:
                    self.index -= 1
                    moved = True
            elif key in (ord("c"), ord("C")):
                self.drawn_box = None
            elif key in (ord("s"), ord("S")):
                self._save()
            elif key in (ord("q"), ord("Q"), 27):
                self._save()
                break

            if moved:
                self._load_image(self._get_row())

        cv2.destroyAllWindows()


def _prepare_rows(template_csv: Path, output_csv: Path):
    rows, fields = _read_csv(template_csv)
    base_rows = []
    for row in rows:
        item = dict(row)
        for k in REQUIRED_FIELDS:
            item.setdefault(k, "")
        base_rows.append(item)

    fields = REQUIRED_FIELDS + [f for f in fields if f not in REQUIRED_FIELDS]

    if output_csv.exists():
        out_rows, _ = _read_csv(output_csv)
        by_id = {str(r.get("id", "")).strip(): r for r in out_rows}
        for row in base_rows:
            rid = str(row.get("id", "")).strip()
            prev = by_id.get(rid)
            if prev is None:
                continue
            for k in REQUIRED_FIELDS:
                if k in prev:
                    row[k] = prev[k]
    return base_rows, fields


def main():
    parser = argparse.ArgumentParser(description="Interactive manual box reviewer with mouse drawing.")
    parser.add_argument("--template-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-display-width", type=int, default=1600)
    parser.add_argument("--max-display-height", type=int, default=950)
    args = parser.parse_args()

    template_csv = Path(args.template_csv)
    if not template_csv.exists():
        raise FileNotFoundError(f"Template CSV not found: {template_csv}")
    output_csv = Path(args.output_csv) if args.output_csv else template_csv.with_name("box_review.csv")

    rows, fields = _prepare_rows(template_csv, output_csv)
    if not rows:
        raise RuntimeError("No rows loaded from template.")

    reviewer = ManualBoxReviewer(
        rows=rows,
        fieldnames=fields,
        output_csv=output_csv,
        max_w=args.max_display_width,
        max_h=args.max_display_height,
    )
    reviewer.index = max(0, min(len(rows) - 1, int(args.start_index)))
    reviewer.run()


if __name__ == "__main__":
    main()
