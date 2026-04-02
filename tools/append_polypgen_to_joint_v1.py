from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# =========================
# 1) 你先改这里
# =========================

PROJECT_ROOT = Path(r"D:\Embodied AI\Colon_Diverticulosis_Project")
POLYPGEN_ROOT = Path(r"D:\Embodied AI\DATASET\PolypGen")

JOINT_ROOT = PROJECT_ROOT / "data" / "joint_polyp_v1"
MANIFEST_PATH = JOINT_ROOT / "manifest" / "samples_v1.csv"

# 训练中心（进入 U_large）
TRAIN_CENTERS = ["C1", "C2", "C3", "C4"]

# 保留中心（进入 external/PolypGen_holdout）
HOLDOUT_CENTERS = ["C5", "C6"]

# 已有文件是否允许覆盖
OVERWRITE = False

# 如果 manifest 里已经有 PolypGen 记录，是否强行继续
FORCE_APPEND_POLYPGEN = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =========================
# 2) 工具函数
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.stem)]


def normalize_stem(stem: str) -> str:
    """
    兼容类似:
    100H0050
    100H0050_mask
    img_001
    img_001_mask
    """
    s = stem.lower()
    s = s.replace("groundtruth", "")
    s = s.replace("ground_truth", "")
    s = s.replace("ground-truth", "")
    s = s.replace("mask", "")
    s = s.replace("label", "")
    s = s.replace("annotation", "")
    s = s.replace("gt", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def find_center_image_mask_dirs(polypgen_root: Path, center: str) -> Tuple[Path, Path]:
    """
    递归查找:
      data_C1/images_C1
      data_C1/masks_C1
    这类目录
    """
    image_target = f"images_{center}"
    mask_target = f"masks_{center}"

    image_dirs = []
    mask_dirs = []

    for p in polypgen_root.rglob("*"):
        if p.is_dir():
            if p.name.lower() == image_target.lower():
                image_dirs.append(p)
            elif p.name.lower() == mask_target.lower():
                mask_dirs.append(p)

    if not image_dirs:
        raise FileNotFoundError(f"找不到 {center} 的图像目录: {image_target}")
    if not mask_dirs:
        raise FileNotFoundError(f"找不到 {center} 的 mask 目录: {mask_target}")

    # 取层级最浅的一个
    image_dirs = sorted(image_dirs, key=lambda x: (len(x.parts), str(x)))
    mask_dirs = sorted(mask_dirs, key=lambda x: (len(x.parts), str(x)))

    return image_dirs[0], mask_dirs[0]


def list_images(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if is_image_file(p)]
    return sorted(files, key=natural_key)


def pair_images_and_masks(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    images = list_images(image_dir)
    masks = list_images(mask_dir)

    if not images:
        raise RuntimeError(f"图像目录为空: {image_dir}")
    if not masks:
        raise RuntimeError(f"mask 目录为空: {mask_dir}")

    mask_map: Dict[str, List[Path]] = {}
    for m in masks:
        key = normalize_stem(m.stem)
        mask_map.setdefault(key, []).append(m)

    pairs = []
    unmatched = []

    for img in images:
        key = normalize_stem(img.stem)
        cands = mask_map.get(key, [])
        if len(cands) == 1:
            pairs.append((img, cands[0]))
        elif len(cands) > 1:
            pairs.append((img, cands[0]))
        else:
            unmatched.append(img)

    if unmatched:
        print(f"[警告] {image_dir} 中有 {len(unmatched)} 张图没有匹配到 mask，已跳过。示例：")
        for p in unmatched[:10]:
            print("   ", p.name)

    if not pairs:
        raise RuntimeError(f"{image_dir} / {mask_dir} 没有成功配对任何 image-mask")

    return pairs


def safe_copy(src: Path, dst: Path) -> None:
    if dst.exists() and not OVERWRITE:
        raise FileExistsError(f"目标已存在且 OVERWRITE=False: {dst}")
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def rel_to_project(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"manifest 不存在: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required_cols = [
        "id", "image_path", "mask_path", "subset", "split", "source", "center",
        "is_labeled", "is_pseudo", "pseudo_weight", "round_id", "exclude_from_tuning"
    ]
    missing = [c for c in required_cols if c not in reader.fieldnames]
    if missing:
        raise RuntimeError(f"manifest 缺少字段: {missing}")

    return rows


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "id", "image_path", "mask_path", "subset", "split", "source", "center",
        "is_labeled", "is_pseudo", "pseudo_weight", "round_id", "exclude_from_tuning"
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_next_u_index(rows: List[Dict[str, str]]) -> int:
    pat = re.compile(r"^u_(\d+)$", re.IGNORECASE)
    max_idx = 0
    for r in rows:
        m = pat.match(r["id"])
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def infer_next_holdout_index(rows: List[Dict[str, str]], center: str) -> int:
    pat = re.compile(rf"^pgh_{center.lower()}_(\d+)$", re.IGNORECASE)
    max_idx = 0
    for r in rows:
        m = pat.match(r["id"])
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


# =========================
# 3) 主流程
# =========================

def main() -> None:
    # 基础检查
    if not POLYPGEN_ROOT.exists():
        raise FileNotFoundError(f"PolypGen 根目录不存在: {POLYPGEN_ROOT}")
    if not JOINT_ROOT.exists():
        raise FileNotFoundError(f"joint_polyp_v1 目录不存在: {JOINT_ROOT}")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest 不存在: {MANIFEST_PATH}")

    rows = load_manifest(MANIFEST_PATH)

    # 防止重复导入
    existing_polypgen = [r for r in rows if r["source"] == "PolypGen"]
    if existing_polypgen and not FORCE_APPEND_POLYPGEN:
        raise RuntimeError(
            f"manifest 中已存在 {len(existing_polypgen)} 条 PolypGen 记录。\n"
            f"如果你确认要继续追加，请把 FORCE_APPEND_POLYPGEN 改为 True。"
        )

    # 确保目录存在
    ensure_dir(JOINT_ROOT / "U_large" / "images")
    ensure_dir(JOINT_ROOT / "external" / "PolypGen_holdout" / "images")
    ensure_dir(JOINT_ROOT / "external" / "PolypGen_holdout" / "masks")

    next_u_idx = infer_next_u_index(rows)
    holdout_counters = {
        c: infer_next_holdout_index(rows, c) for c in HOLDOUT_CENTERS
    }

    new_rows: List[Dict[str, str]] = []

    # -------------------------
    # A) C1-C4 -> U_large
    # -------------------------
    for center in TRAIN_CENTERS:
        image_dir, mask_dir = find_center_image_mask_dirs(POLYPGEN_ROOT, center)
        pairs = pair_images_and_masks(image_dir, mask_dir)

        print(f"[PolypGen {center}] -> U_large, matched pairs: {len(pairs)}")

        for img_src, _mask_src in pairs:
            new_id = f"u_{next_u_idx:04d}"
            next_u_idx += 1

            img_dst = JOINT_ROOT / "U_large" / "images" / f"{new_id}{img_src.suffix.lower()}"
            safe_copy(img_src, img_dst)

            new_rows.append({
                "id": new_id,
                "image_path": rel_to_project(img_dst),
                "mask_path": "",
                "subset": "U_large",
                "split": "unlabeled",
                "source": "PolypGen",
                "center": center,
                "is_labeled": 0,
                "is_pseudo": 0,
                "pseudo_weight": 0.0,
                "round_id": 0,
                "exclude_from_tuning": 0,
            })

    # -------------------------
    # B) C5-C6 -> external/PolypGen_holdout
    # -------------------------
    for center in HOLDOUT_CENTERS:
        image_dir, mask_dir = find_center_image_mask_dirs(POLYPGEN_ROOT, center)
        pairs = pair_images_and_masks(image_dir, mask_dir)

        print(f"[PolypGen {center}] -> external/PolypGen_holdout, matched pairs: {len(pairs)}")

        for img_src, mask_src in pairs:
            idx = holdout_counters[center]
            holdout_counters[center] += 1

            new_id = f"pgh_{center.lower()}_{idx:04d}"
            img_dst = JOINT_ROOT / "external" / "PolypGen_holdout" / "images" / f"{new_id}{img_src.suffix.lower()}"
            mask_dst = JOINT_ROOT / "external" / "PolypGen_holdout" / "masks" / f"{new_id}{mask_src.suffix.lower()}"

            safe_copy(img_src, img_dst)
            safe_copy(mask_src, mask_dst)

            new_rows.append({
                "id": new_id,
                "image_path": rel_to_project(img_dst),
                "mask_path": rel_to_project(mask_dst),
                "subset": "external",
                "split": "test",
                "source": "PolypGen",
                "center": center,
                "is_labeled": 1,
                "is_pseudo": 0,
                "pseudo_weight": 1.0,
                "round_id": 0,
                "exclude_from_tuning": 1,
            })

    # 备份旧 manifest
    backup_path = MANIFEST_PATH.with_name("samples_v1.before_polypgen_backup.csv")
    shutil.copy2(MANIFEST_PATH, backup_path)

    # 追加并写回
    all_rows = rows + new_rows
    write_manifest(MANIFEST_PATH, all_rows)

    print("\n==============================")
    print("PolypGen 已成功追加到 joint_polyp_v1")
    print(f"新增样本数: {len(new_rows)}")
    print(f"manifest 备份: {backup_path}")
    print(f"manifest 更新: {MANIFEST_PATH}")
    print("==============================")


if __name__ == "__main__":
    main()