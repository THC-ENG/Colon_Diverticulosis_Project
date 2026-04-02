import argparse
import csv
import json
import math
from html import escape
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


def _resolve(path_text: str, base: Path | None = None) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    if base is not None:
        p2 = base / p
        if p2.exists():
            return p2
    return (Path.cwd() / p).resolve()


def _load_manifest_rows(manifest_path: str, subset_filter: set[str]) -> list[dict]:
    out: list[dict] = []
    mpath = Path(manifest_path)
    with open(mpath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subset = str(row.get("subset", "")).strip()
            if subset_filter and subset not in subset_filter:
                continue
            img_text = str(row.get("image_path", "")).strip()
            if not img_text:
                continue
            img = _resolve(img_text, mpath.parent)
            if not img.exists():
                continue
            pid = str(row.get("id", "")).strip() or img.stem
            out.append(
                {
                    "id": pid,
                    "image_path": str(img),
                    "subset": subset,
                    "source": str(row.get("source", "")).strip(),
                    "center": str(row.get("center", "")).strip(),
                }
            )
    return out


def _load_dir_rows(image_dir: str) -> list[dict]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    out = []
    for p in sorted([x for x in Path(image_dir).glob("*") if x.suffix.lower() in exts]):
        out.append({"id": p.stem, "image_path": str(p), "subset": "U_large", "source": "", "center": ""})
    return out


def _load_id_filter(path: str) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x) for x in data}
        if isinstance(data, dict) and "ids" in data and isinstance(data["ids"], list):
            return {str(x) for x in data["ids"]}
        return set()
    ids = set()
    with open(p, "r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if text:
                ids.add(text.split(",")[0])
    return ids


def _load_proposals(path: str) -> dict[str, list[float]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        if isinstance(v, list) and len(v) == 4:
            out[str(k)] = [float(x) for x in v]
    return out


def _edge_from_prob(prob: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = np.clip(edge, 0.0, np.percentile(edge, 99.0) + 1e-6)
    edge = edge / (edge.max() + 1e-6)
    return edge.astype(np.float32)


def _edge_quality(mask: np.ndarray, edge: np.ndarray) -> float:
    fg = (mask > 0).astype(np.uint8)
    if fg.sum() == 0:
        return 0.0

    num_cc, _ = cv2.connectedComponents(fg)
    comp_score = 1.0 / (1.0 + max(0, num_cc - 2))

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = float(sum(cv2.arcLength(c, True) for c in contours)) + 1e-6
    area = float(fg.sum())
    shape_score = np.clip((area / perim) / 6.0, 0.0, 1.0)

    edge_strength = float(np.clip(edge.mean() * 4.0, 0.0, 1.0))
    return float(np.clip(0.5 * edge_strength + 0.3 * comp_score + 0.2 * shape_score, 0.0, 1.0))


def _mask_area_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float((mask > 0).mean())


def _area_prior(area_ratio: float, min_ratio: float, target_ratio: float, max_ratio: float) -> float:
    a = float(area_ratio)
    lo = float(min_ratio)
    mid = float(target_ratio)
    hi = float(max_ratio)
    if hi <= lo:
        return 0.0
    if a < lo or a > hi:
        return 0.0
    if mid <= lo:
        return 1.0 - max(0.0, min(1.0, (a - lo) / (hi - lo + 1e-6)))
    if mid >= hi:
        return max(0.0, min(1.0, (a - lo) / (hi - lo + 1e-6)))
    if a <= mid:
        return max(0.0, min(1.0, (a - lo) / (mid - lo + 1e-6)))
    return max(0.0, min(1.0, (hi - a) / (hi - mid + 1e-6)))


def _center_prior(mask: np.ndarray) -> float:
    fg = np.where(mask > 0)
    if len(fg[0]) == 0:
        return 0.0
    cy = float(np.mean(fg[0]))
    cx = float(np.mean(fg[1]))
    h, w = mask.shape[:2]
    my = 0.5 * (h - 1)
    mx = 0.5 * (w - 1)
    dist = math.sqrt((cy - my) ** 2 + (cx - mx) ** 2)
    max_dist = math.sqrt(my * my + mx * mx) + 1e-6
    return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))


def _parse_scales(text: str) -> list[float]:
    vals = []
    for part in str(text).split(","):
        t = part.strip()
        if not t:
            continue
        try:
            v = float(t)
        except ValueError:
            continue
        if 0.1 <= v <= 1.0:
            vals.append(v)
    if not vals:
        vals = [1.0, 0.85, 0.7, 0.55]
    if 1.0 not in vals:
        vals = [1.0] + vals
    seen = set()
    out = []
    for v in vals:
        key = round(v, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _parse_centers(text: str) -> list[float]:
    vals = []
    for part in str(text).split(","):
        t = part.strip()
        if not t:
            continue
        try:
            v = float(t)
        except ValueError:
            continue
        if 0.05 <= v <= 0.95:
            vals.append(v)
    if not vals:
        vals = [0.3, 0.5, 0.7]
    seen = set()
    out = []
    for v in vals:
        key = round(v, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _box_from_center_scale(w: int, h: int, scale: float, center_x: float, center_y: float) -> list[float]:
    cx = float(center_x) * float(max(1, w - 1))
    cy = float(center_y) * float(max(1, h - 1))
    bw = max(8.0, (w - 1) * float(scale))
    bh = max(8.0, (h - 1) * float(scale))
    x0 = max(0.0, cx - 0.5 * bw)
    y0 = max(0.0, cy - 0.5 * bh)
    x1 = min(float(w - 1), cx + 0.5 * bw)
    y1 = min(float(h - 1), cy + 0.5 * bh)
    return [x0, y0, x1, y1]


def _build_auto_boxes(
    w: int,
    h: int,
    scales: list[float],
    centers: list[float],
    mode: str,
    max_candidates: int,
) -> list[list[float]]:
    mode = str(mode).strip().lower()
    if mode == "single_box":
        return [[0.0, 0.0, float(w - 1), float(h - 1)]]
    boxes = []
    if mode == "multi_box":
        for s in scales:
            boxes.append(_box_from_center_scale(w, h, s, center_x=0.5, center_y=0.5))
    elif mode == "grid_multi_box":
        for s in scales:
            for cy in centers:
                for cx in centers:
                    boxes.append(_box_from_center_scale(w, h, s, center_x=cx, center_y=cy))
    else:
        raise ValueError(f"Unsupported auto proposal mode: {mode}")

    dedup = {}
    for b in boxes:
        key = tuple(round(float(v), 2) for v in b)
        dedup[key] = [float(v) for v in b]
    out = list(dedup.values())
    if max_candidates > 0:
        out = out[: int(max_candidates)]
    if not out:
        out = [[0.0, 0.0, float(w - 1), float(h - 1)]]
    if [0.0, 0.0, float(w - 1), float(h - 1)] not in out:
        out.append([0.0, 0.0, float(w - 1), float(h - 1)])
    if max_candidates > 0:
        out = out[: int(max_candidates)]
    return out


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        dev = base.weight.device
        dt = base.weight.dtype
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features, device=dev, dtype=dt))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = (x @ self.lora_a.t()) @ self.lora_b.t()
        return base_out + self.scale * delta


def inject_lora(module: nn.Module, target_keywords: list[str], rank: int, alpha: int) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(k in name.lower() for k in target_keywords):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            replaced += 1
        else:
            replaced += inject_lora(child, target_keywords, rank, alpha)
    return replaced


def _overlay_mask(image_bgr: np.ndarray, mask_bin: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    image_f = image_bgr.astype(np.float32)
    color = np.zeros_like(image_f)
    color[..., 2] = 255.0
    mask3 = (mask_bin > 0)[..., None]
    blended = image_f * (1.0 - alpha) + color * alpha
    out = np.where(mask3, blended, image_f)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _build_panel(image_bgr: np.ndarray, hard_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask_gray = hard_mask if hard_mask.ndim == 2 else hard_mask.squeeze()
    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    overlay = _overlay_mask(image_bgr, mask_gray, alpha=alpha)
    panel = np.concatenate([image_bgr, mask_bgr, overlay], axis=1)
    h = panel.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "image", (8, min(h - 8, 24)), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        "mask",
        (image_bgr.shape[1] + 8, min(h - 8, 24)),
        font,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "overlay",
        (image_bgr.shape[1] * 2 + 8, min(h - 8, 24)),
        font,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _path_to_uri(path_text: str) -> str:
    if not path_text:
        return ""
    try:
        return Path(path_text).resolve().as_uri()
    except Exception:
        return path_text


def _write_gallery(rows: list[dict], output_html: Path):
    output_html.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda x: float(x.get("quality", 0.0)), reverse=True)
    cards = []
    for row in sorted_rows:
        panel_path = str(row.get("panel_path", "")).strip()
        hard_path = str(row.get("hard_mask_path", "")).strip()
        image_path = str(row.get("image_path", "")).strip()
        preview_uri = _path_to_uri(panel_path if panel_path else hard_path)
        hard_uri = _path_to_uri(hard_path)
        image_uri = _path_to_uri(image_path)
        pid = escape(str(row.get("id", "")))
        conf = float(row.get("conf", 0.0))
        edge_q = float(row.get("edge_quality", 0.0))
        area_ratio = float(row.get("area_ratio", 0.0))
        area_prior = float(row.get("area_prior", 0.0))
        quality = float(row.get("quality", 0.0))
        cards.append(
            (
                "<div class='card'>"
                f"<div class='id'>{pid}</div>"
                f"<img src='{preview_uri}' loading='lazy' alt='{pid}' />"
                f"<div class='meta'>conf={conf:.4f} edge={edge_q:.4f} area={area_ratio:.4f} area_prior={area_prior:.4f} quality={quality:.4f}</div>"
                f"<div class='links'><a href='{image_uri}'>image</a> | <a href='{hard_uri}'>mask</a></div>"
                "</div>"
            )
        )

    html = (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<title>Pseudo Mask Gallery</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:16px;background:#f6f8fb;color:#111;}"
        "h1{margin:0 0 8px 0;font-size:22px;}"
        "p{margin:0 0 16px 0;color:#555;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px;}"
        ".card{background:#fff;border:1px solid #dde3ea;border-radius:8px;padding:10px;box-shadow:0 1px 2px rgba(0,0,0,.05);}"
        ".id{font-weight:700;font-size:13px;margin-bottom:8px;word-break:break-all;}"
        "img{width:100%;height:auto;border-radius:6px;border:1px solid #e5e9ef;background:#000;}"
        ".meta{margin-top:8px;font-size:12px;color:#333;}"
        ".links{margin-top:6px;font-size:12px;}"
        "a{color:#0b5ed7;text-decoration:none;}a:hover{text-decoration:underline;}"
        "</style></head><body>"
        f"<h1>Pseudo Mask Gallery</h1><p>Total samples: {len(sorted_rows)}</p>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>"
    )
    output_html.write_text(html, encoding="utf-8")


def _safe_torch_load(path: str, map_location: str | torch.device | None = None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo labels with MedSAM teacher.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Base MedSAM checkpoint.")
    parser.add_argument("--model-type", type=str, default="vit_b")
    parser.add_argument("--lora-checkpoint", type=str, default="", help="LoRA-adapted teacher checkpoint.")
    parser.add_argument("--data-manifest", type=str, default="")
    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--subset-filter", type=str, default="U_large")
    parser.add_argument("--id-filter", type=str, default="", help="Optional txt/json list of ids to process.")
    parser.add_argument("--proposal-json", type=str, default="", help="Optional id->box mapping.")
    parser.add_argument("--fallback-full-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--auto-proposal-mode",
        type=str,
        default="multi_box",
        choices=["single_box", "multi_box", "grid_multi_box"],
    )
    parser.add_argument("--candidate-box-scales", type=str, default="1.0,0.85,0.7,0.55")
    parser.add_argument("--candidate-box-centers", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--max-candidate-boxes", type=int, default=40)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.002)
    parser.add_argument("--target-mask-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.35)
    parser.add_argument("--round-id", type=int, default=1)
    parser.add_argument("--output-root", type=str, default="runs/flywheel/round1/pseudo")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-exist", action="store_true")
    parser.add_argument("--save-panels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--write-gallery", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_root = Path(args.output_root)
    hard_dir = out_root / "hard_masks"
    soft_dir = out_root / "soft_probs"
    edge_dir = out_root / "edge_probs"
    panel_dir = out_root / "mask_panels"
    out_root.mkdir(parents=True, exist_ok=True)
    hard_dir.mkdir(parents=True, exist_ok=True)
    soft_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)
    if args.save_panels:
        panel_dir.mkdir(parents=True, exist_ok=True)

    subset_filter = {s.strip() for s in args.subset_filter.split(",") if s.strip()}
    if args.data_manifest:
        rows = _load_manifest_rows(args.data_manifest, subset_filter)
    elif args.image_dir:
        rows = _load_dir_rows(args.image_dir)
    else:
        raise ValueError("Provide --data-manifest or --image-dir")

    id_filter = _load_id_filter(args.id_filter)
    if id_filter:
        rows = [r for r in rows if r["id"] in id_filter]
    if not rows:
        raise RuntimeError("No images to process.")

    proposals = _load_proposals(args.proposal_json)
    candidate_scales = _parse_scales(args.candidate_box_scales)
    candidate_centers = _parse_centers(args.candidate_box_centers)

    model = sam_model_registry[args.model_type](checkpoint=None).to(args.device)
    base_state = _safe_torch_load(args.checkpoint, map_location=args.device)
    if isinstance(base_state, dict) and "state_dict" in base_state and isinstance(base_state["state_dict"], dict):
        base_state = base_state["state_dict"]
    missing0, unexpected0 = model.load_state_dict(base_state, strict=False)
    print(f"[base checkpoint] missing={len(missing0)} unexpected={len(unexpected0)} from {args.checkpoint}")
    if args.lora_checkpoint:
        ckpt = _safe_torch_load(args.lora_checkpoint, map_location=args.device)
        state = ckpt.get("sam_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict) and any(".lora_a" in k for k in state.keys()):
            rank = int(ckpt.get("lora_rank", 8)) if isinstance(ckpt, dict) else 8
            alpha = int(ckpt.get("lora_alpha", 16)) if isinstance(ckpt, dict) else 16
            replaced = inject_lora(
                model.image_encoder,
                target_keywords=["qkv", "proj", "q_proj", "k_proj", "v_proj"],
                rank=rank,
                alpha=alpha,
            )
            print(f"[teacher lora] injected layers={replaced}, rank={rank}, alpha={alpha}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[teacher load] missing={len(missing)} unexpected={len(unexpected)} from {args.lora_checkpoint}")
    predictor = SamPredictor(model)

    quality_rows = []
    manifest_rows = []
    for row in tqdm(rows, desc="Pseudo Labeling"):
        pid = row["id"]
        out_hard = hard_dir / f"{pid}.png"
        out_soft = soft_dir / f"{pid}.npy"
        out_edge = edge_dir / f"{pid}.npy"
        out_panel = panel_dir / f"{pid}.jpg"
        core_ready = out_hard.exists() and out_soft.exists() and out_edge.exists()
        panel_ready = (not args.save_panels) or out_panel.exists()
        if args.skip_exist and core_ready and panel_ready:
            continue

        image_bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        predictor.set_image(image_rgb)

        preset_box = proposals.get(pid)
        if preset_box is not None:
            candidate_boxes = [preset_box]
        elif args.fallback_full_image:
            candidate_boxes = _build_auto_boxes(
                w=w,
                h=h,
                scales=candidate_scales,
                centers=candidate_centers,
                mode=args.auto_proposal_mode,
                max_candidates=args.max_candidate_boxes,
            )
        else:
            candidate_boxes = []
        if not candidate_boxes:
            continue

        all_packs = []
        valid_packs = []
        for box in candidate_boxes:
            masks, scores, logits = predictor.predict(
                box=np.array(box, dtype=np.float32)[None, :],
                point_coords=None,
                point_labels=None,
                multimask_output=True,
                return_logits=True,
            )
            for k in range(int(len(scores))):
                hard_k = (masks[k] > 0).astype(np.uint8) * 255
                lowres_logit = logits[k].astype(np.float32)
                lowres_prob = 1.0 / (1.0 + np.exp(-lowres_logit))
                soft_k = cv2.resize(lowres_prob, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                edge_k = _edge_from_prob(soft_k)
                conf_k = float(scores[k])
                edge_q_k = _edge_quality(hard_k, edge_k)
                area_ratio_k = _mask_area_ratio(hard_k)
                area_prior_k = _area_prior(
                    area_ratio_k,
                    min_ratio=args.min_mask_area_ratio,
                    target_ratio=args.target_mask_area_ratio,
                    max_ratio=args.max_mask_area_ratio,
                )
                center_prior_k = _center_prior(hard_k)
                score_k = float(0.45 * conf_k + 0.25 * edge_q_k + 0.20 * area_prior_k + 0.10 * center_prior_k)
                pack = {
                    "hard": hard_k,
                    "soft": soft_k,
                    "edge": edge_k,
                    "conf": conf_k,
                    "edge_quality": edge_q_k,
                    "area_ratio": area_ratio_k,
                    "area_prior": area_prior_k,
                    "center_prior": center_prior_k,
                    "quality": score_k,
                    "prompt_box": [float(x) for x in box],
                }
                all_packs.append(pack)
                if args.min_mask_area_ratio <= area_ratio_k <= args.max_mask_area_ratio:
                    valid_packs.append(pack)
        candidate_pool = valid_packs if valid_packs else all_packs
        if not candidate_pool:
            continue
        best_pack = max(candidate_pool, key=lambda x: float(x["quality"]))

        hard = best_pack["hard"]
        soft = best_pack["soft"]
        edge = best_pack["edge"]
        conf = float(best_pack["conf"])
        edge_q = float(best_pack["edge_quality"])
        area_ratio = float(best_pack["area_ratio"])
        area_prior = float(best_pack["area_prior"])
        center_prior = float(best_pack["center_prior"])
        quality = float(best_pack["quality"])
        prompt_box = best_pack["prompt_box"]

        cv2.imwrite(str(out_hard), hard)
        np.save(out_soft, soft)
        np.save(out_edge, edge)
        panel_path = ""
        if args.save_panels:
            panel = _build_panel(image_bgr, hard, alpha=args.overlay_alpha)
            cv2.imwrite(str(out_panel), panel)
            panel_path = str(out_panel)

        quality_rows.append(
            {
                "id": pid,
                "image_path": row["image_path"],
                "hard_mask_path": str(out_hard),
                "panel_path": panel_path,
                "soft_path": str(out_soft),
                "edge_path": str(out_edge),
                "prompt_box": json.dumps(prompt_box, ensure_ascii=False),
                "subset": row.get("subset", "U_large"),
                "source": row.get("source", ""),
                "center": row.get("center", ""),
                "conf": conf,
                "edge_quality": edge_q,
                "area_ratio": area_ratio,
                "area_prior": area_prior,
                "center_prior": center_prior,
                "quality": quality,
                "round_id": int(args.round_id),
            }
        )
        manifest_rows.append(
            {
                "id": pid,
                "image_path": row["image_path"],
                "mask_path": str(out_hard),
                "subset": f"pseudo_round{int(args.round_id)}",
                "split": "train",
                "source": row.get("source", ""),
                "center": row.get("center", ""),
                "is_labeled": 1,
                "is_pseudo": 1,
                "pseudo_weight": quality,
                "round_id": int(args.round_id),
                "exclude_from_tuning": 0,
                "soft_path": str(out_soft),
                "edge_path": str(out_edge),
            }
        )

    quality_csv = out_root / "pseudo_quality.csv"
    with open(quality_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "image_path",
            "hard_mask_path",
            "panel_path",
            "soft_path",
            "edge_path",
            "prompt_box",
            "subset",
            "source",
            "center",
            "conf",
            "edge_quality",
            "area_ratio",
            "area_prior",
            "center_prior",
            "quality",
            "round_id",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(quality_rows)

    manifest_csv = out_root / "pseudo_candidates_manifest.csv"
    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)

    gallery_html = out_root / "mask_gallery.html"
    if args.write_gallery:
        _write_gallery(quality_rows, gallery_html)

    print(
        json.dumps(
            {
                "num_processed": len(quality_rows),
                "quality_csv": str(quality_csv),
                "pseudo_candidates_manifest": str(manifest_csv),
                "hard_masks_dir": str(hard_dir),
                "mask_panels_dir": str(panel_dir) if args.save_panels else "",
                "mask_gallery_html": str(gallery_html) if args.write_gallery else "",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
