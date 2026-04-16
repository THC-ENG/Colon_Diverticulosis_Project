from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTS = IMAGE_EXTS


@dataclass
class ProtocolSample:
    id: str
    image_path: str
    mask_path: str = ""
    subset: str = ""
    split: str = ""
    source: str = ""
    center: str = ""
    is_labeled: int = 0
    is_pseudo: int = 0
    pseudo_weight: float = 0.0
    round_id: int = 0
    exclude_from_tuning: int = 0
    soft_path: str = ""
    edge_path: str = ""
    tier: str = ""

    def to_row(self) -> dict:
        return asdict(self)


def _to_int(v: str | int | float | None, default: int = 0) -> int:
    if v is None:
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    text = str(v).strip().lower()
    if text == "":
        return default
    if text in {"true", "t", "yes", "y"}:
        return 1
    if text in {"false", "f", "no", "n"}:
        return 0
    return int(float(text))


def _to_float(v: str | int | float | None, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    text = str(v).strip()
    if text == "":
        return default
    return float(text)


def _resolve_path(raw_path: str, manifest_path: Path | None = None) -> str:
    if raw_path is None:
        return ""
    raw = str(raw_path).strip()
    if raw == "":
        return ""

    p = Path(raw)
    if p.is_absolute():
        return str(p)

    candidates = [Path.cwd() / p]
    if manifest_path is not None:
        candidates.append(manifest_path.parent / p)
        candidates.append(manifest_path.parent.parent / p)

    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])


def _normalize_subset(text: str) -> str:
    v = (text or "").strip()
    if v == "":
        return ""
    return v


def _normalize_split(text: str) -> str:
    v = (text or "").strip()
    if v == "":
        return ""
    return v


def _find_mask_for_image(mask_dir: Path, image_stem: str) -> Path | None:
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        p = mask_dir / f"{image_stem}{ext}"
        if p.exists():
            return p
    return None


def load_manifest_samples(manifest_path: str) -> list[ProtocolSample]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: list[ProtocolSample] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "image_path"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Manifest must contain at least columns {sorted(required)}. "
                f"Current columns: {reader.fieldnames}"
            )

        for raw in reader:
            sample = ProtocolSample(
                id=str(raw.get("id", "")).strip(),
                image_path=_resolve_path(raw.get("image_path", ""), path),
                mask_path=_resolve_path(raw.get("mask_path", ""), path),
                subset=_normalize_subset(str(raw.get("subset", ""))),
                split=_normalize_split(str(raw.get("split", ""))),
                source=str(raw.get("source", "")).strip(),
                center=str(raw.get("center", "")).strip(),
                is_labeled=_to_int(raw.get("is_labeled"), default=0),
                is_pseudo=_to_int(raw.get("is_pseudo"), default=0),
                pseudo_weight=_to_float(raw.get("pseudo_weight"), default=0.0),
                round_id=_to_int(raw.get("round_id"), default=0),
                exclude_from_tuning=_to_int(raw.get("exclude_from_tuning"), default=0),
                soft_path=_resolve_path(raw.get("soft_path", ""), path),
                edge_path=_resolve_path(raw.get("edge_path", ""), path),
                tier=str(raw.get("tier", "")).strip(),
            )
            if sample.id == "":
                sample.id = Path(sample.image_path).stem
            rows.append(sample)

    return rows


def load_layout_samples(data_root: str) -> list[ProtocolSample]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    rows: list[ProtocolSample] = []

    l_small_img = root / "L_small" / "images"
    l_small_mask = root / "L_small" / "masks"
    if l_small_img.exists() and l_small_mask.exists():
        for p in sorted([x for x in l_small_img.glob("*") if x.suffix.lower() in IMAGE_EXTS]):
            mask = _find_mask_for_image(l_small_mask, p.stem)
            if mask is None:
                raise ValueError(f"L_small image has no matching mask: {p}")
            rows.append(
                ProtocolSample(
                    id=p.stem,
                    image_path=str(p),
                    mask_path=str(mask),
                    subset="L_small",
                    split="train",
                    source="joint",
                    is_labeled=1,
                    is_pseudo=0,
                    pseudo_weight=1.0,
                    round_id=0,
                    exclude_from_tuning=0,
                )
            )

    u_large_img = root / "U_large" / "images"
    if u_large_img.exists():
        for p in sorted([x for x in u_large_img.glob("*") if x.suffix.lower() in IMAGE_EXTS]):
            rows.append(
                ProtocolSample(
                    id=p.stem,
                    image_path=str(p),
                    mask_path="",
                    subset="U_large",
                    split="unlabeled",
                    source="joint",
                    is_labeled=0,
                    is_pseudo=0,
                    pseudo_weight=0.0,
                    round_id=0,
                    exclude_from_tuning=0,
                )
            )

    external_dir = root / "external"
    if external_dir.exists():
        for center_dir in sorted([x for x in external_dir.iterdir() if x.is_dir()]):
            img_dir = center_dir / "images"
            mask_dir = center_dir / "masks"
            if not img_dir.exists() or not mask_dir.exists():
                continue
            for p in sorted([x for x in img_dir.glob("*") if x.suffix.lower() in IMAGE_EXTS]):
                mask = _find_mask_for_image(mask_dir, p.stem)
                if mask is None:
                    raise ValueError(f"External image has no matching mask: {p}")
                rows.append(
                    ProtocolSample(
                        id=f"{center_dir.name}_{p.stem}",
                        image_path=str(p),
                        mask_path=str(mask),
                        subset="external",
                        split="test",
                        source=center_dir.name,
                        center=center_dir.name,
                        is_labeled=1,
                        is_pseudo=0,
                        pseudo_weight=1.0,
                        round_id=0,
                        exclude_from_tuning=1,
                    )
                )

    if not rows:
        raise RuntimeError(f"No samples found under data root: {root}")
    return rows


def load_protocol_samples(
    data_manifest: str | None = None,
    data_root: str | None = None,
    manifest_mode: str = "prefer",
) -> list[ProtocolSample]:
    mode = (manifest_mode or "prefer").strip().lower()
    if mode not in {"prefer", "only", "off"}:
        raise ValueError(f"Unsupported manifest_mode: {manifest_mode}")

    manifest_path = Path(data_manifest) if data_manifest else None
    use_manifest = (
        mode in {"prefer", "only"}
        and manifest_path is not None
        and manifest_path.exists()
    )

    if use_manifest:
        return load_manifest_samples(str(manifest_path))

    if mode == "only":
        raise FileNotFoundError(f"Manifest mode=only, but manifest not found: {data_manifest}")

    if data_root:
        return load_layout_samples(data_root)

    if manifest_path is not None:
        return load_manifest_samples(str(manifest_path))

    raise ValueError("No valid data source provided. Set --data-manifest or --data-root.")


def validate_protocol_samples(samples: Iterable[ProtocolSample]) -> None:
    rows = list(samples)
    if not rows:
        raise RuntimeError("No protocol samples found.")

    errors: list[str] = []
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for s in rows:
        sid = s.id.strip()
        if sid in seen_ids:
            duplicate_ids.add(sid)
        seen_ids.add(sid)

        img = Path(s.image_path)
        if not img.exists():
            errors.append(f"[missing image] id={sid} path={s.image_path}")

        subset = s.subset.strip().lower()
        has_mask = str(s.mask_path).strip() != ""
        if has_mask and not Path(s.mask_path).exists():
            errors.append(f"[missing mask] id={sid} path={s.mask_path}")

        if subset == "l_small":
            if not has_mask:
                errors.append(f"[L_small requires mask] id={sid}")
            if s.is_labeled != 1:
                errors.append(f"[L_small requires is_labeled=1] id={sid}")

        if subset == "u_large":
            if s.is_labeled != 0 and s.is_pseudo == 0:
                errors.append(f"[U_large unlabeled sample requires is_labeled=0] id={sid}")

        if subset == "external":
            if not has_mask:
                errors.append(f"[external requires mask] id={sid}")
            if s.exclude_from_tuning != 1:
                errors.append(f"[external requires exclude_from_tuning=1] id={sid}")

    if duplicate_ids:
        errors.append(
            f"[duplicate ids] found {len(duplicate_ids)} duplicate ids, "
            "please keep ids unique across the protocol."
        )

    if errors:
        details = "\n".join(errors[:50])
        if len(errors) > 50:
            details += f"\n... and {len(errors) - 50} more"
        raise RuntimeError(f"Protocol validation failed with {len(errors)} issue(s):\n{details}")


def select_protocol_samples(
    samples: Iterable[ProtocolSample],
    include_subsets: set[str] | None = None,
    include_splits: set[str] | None = None,
    exclude_tuning: bool = False,
) -> list[ProtocolSample]:
    out: list[ProtocolSample] = []
    for s in samples:
        if include_subsets is not None and s.subset not in include_subsets:
            continue
        if include_splits is not None and s.split not in include_splits:
            continue
        if exclude_tuning and s.exclude_from_tuning == 1:
            continue
        out.append(s)
    return out


def write_manifest(samples: Iterable[ProtocolSample], output_path: str) -> None:
    rows = list(samples)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "tier",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in rows:
            writer.writerow(s.to_row())


def summarize_samples(samples: Iterable[ProtocolSample]) -> dict:
    rows = list(samples)
    by_subset: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for s in rows:
        by_subset[s.subset] = by_subset.get(s.subset, 0) + 1
        by_split[s.split] = by_split.get(s.split, 0) + 1
    return {
        "num_samples": len(rows),
        "by_subset": by_subset,
        "by_split": by_split,
    }
