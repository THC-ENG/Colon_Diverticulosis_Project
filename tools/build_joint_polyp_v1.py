from __future__ import annotations

import csv
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple



# 项目根目录
PROJECT_ROOT = Path(r"D:\Embodied AI\Colon_Diverticulosis_Project")
# 原始五个数据集所在根目录
RAW_ROOT = Path(r"D:\Embodied AI\DATASET")
# joint_polyp_v1 输出目录
JOINT_ROOT = PROJECT_ROOT / "data" / "joint_polyp_v1"
# L_small 比例（对 Kvasir-SEG 和 CVC-ClinicDB 分别按比例抽样）
L_SMALL_RATIO = 0.10
# 随机种子，保证可复现
RANDOM_SEED = 2026
# 如果目标文件已存在，是否覆盖
OVERWRITE = False
# 图像后缀
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =========================
# 2. 数据集配置
# =========================

DATASETS = {
    "Kvasir-SEG": {
        "root": RAW_ROOT / "Kvasir-SEG",
        "role": "train_pool",
        "prefix": "kvasir",
        "external_name": None,
        "image_dir_candidates": ["images", "image", "Images", "Image"],
        "mask_dir_candidates": ["masks", "mask", "Masks", "Mask", "ground_truth", "ground-truth", "Ground Truth"],
    },
    "CVC-ClinicDB": {
        "root": RAW_ROOT / "CVC-ClinicDB",
        "role": "train_pool",
        "prefix": "clinicdb",
        "external_name": None,
        "image_dir_candidates": ["images", "image", "Original", "original", "Images", "Image"],
        "mask_dir_candidates": ["masks", "mask", "Ground Truth", "ground_truth", "ground-truth", "GT", "gt"],
    },
    "ETIS": {
        "root": RAW_ROOT / "ETIS-LaribPolypDB",
        "role": "external",
        "prefix": "etis",
        "external_name": "ETIS",
        "image_dir_candidates": ["images", "image", "Images", "Image"],
        "mask_dir_candidates": ["masks", "mask", "Masks", "Mask", "ground_truth", "ground-truth", "Ground Truth"],
    },
    "CVC-ColonDB": {
        "root": RAW_ROOT / "CVC-ColonDB",
        "role": "external",
        "prefix": "colondb",
        "external_name": "CVC-ColonDB",
        "image_dir_candidates": ["images", "image", "Images", "Image"],
        "mask_dir_candidates": ["masks", "mask", "Masks", "Mask", "ground_truth", "ground-truth", "Ground Truth"],
    },
    "CVC-300": {
        "root": RAW_ROOT / "CVC-300",
        "role": "external",
        "prefix": "cvc300",
        "external_name": "CVC-300",
        "image_dir_candidates": ["images", "image", "Images", "Image"],
        "mask_dir_candidates": ["masks", "mask", "Masks", "Mask", "ground_truth", "ground-truth", "Ground Truth"],
    },
}

# 工具函数
def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.stem)]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def normalize_stem(stem: str) -> str:
    """
    尽量兼容类似 1 / 1_mask / 1_gt / img1 / img1_mask 这类命名。
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


def find_best_subdir(dataset_root: Path, candidates: List[str]) -> Path:
    """
    递归查找最可能的 image/mask 目录。
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")

    candidates_lower = {c.lower() for c in candidates}
    matches = []

    # 先看 dataset_root 自己是不是候选目录
    if dataset_root.name.lower() in candidates_lower:
        matches.append(dataset_root)

    # 再递归找子目录
    for p in dataset_root.rglob("*"):
        if p.is_dir() and p.name.lower() in candidates_lower:
            matches.append(p)

    if matches:
        # 选层级最浅的
        matches = sorted(matches, key=lambda x: (len(x.relative_to(dataset_root).parts), str(x)))
        return matches[0]

    # 如果没找到候选目录，但 dataset_root 下直接就是图片，也允许直接用
    direct_images = [p for p in dataset_root.iterdir() if is_image_file(p)]
    if direct_images:
        return dataset_root

    raise FileNotFoundError(
        f"在 {dataset_root} 下找不到候选目录: {candidates}。\n"
        f"请手动检查原始数据集目录结构，并修改 DATASETS 配置。"
    )


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
            # 如果有多个候选，优先选 stem 最像的
            exact = [m for m in cands if normalize_stem(m.stem) == key]
            if exact:
                pairs.append((img, exact[0]))
            else:
                pairs.append((img, cands[0]))
        else:
            unmatched.append(img)

    if unmatched:
        print(f"[警告] {image_dir} 中有 {len(unmatched)} 张图没有匹配到 mask。示例:")
        for p in unmatched[:10]:
            print("   ", p.name)

    if not pairs:
        raise RuntimeError(f"没有成功配对任何 image-mask: {image_dir} / {mask_dir}")

    return pairs


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst: Path):
    if dst.exists() and not OVERWRITE:
        raise FileExistsError(f"目标文件已存在，且 OVERWRITE=False: {dst}")
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def rel_to_project(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


# 主流程

def main():
    rng = random.Random(RANDOM_SEED)

    # 目录准备
    ensure_dir(JOINT_ROOT / "L_small" / "images")
    ensure_dir(JOINT_ROOT / "L_small" / "masks")
    ensure_dir(JOINT_ROOT / "U_large" / "images")
    ensure_dir(JOINT_ROOT / "external" / "ETIS" / "images")
    ensure_dir(JOINT_ROOT / "external" / "ETIS" / "masks")
    ensure_dir(JOINT_ROOT / "external" / "CVC-ColonDB" / "images")
    ensure_dir(JOINT_ROOT / "external" / "CVC-ColonDB" / "masks")
    ensure_dir(JOINT_ROOT / "external" / "CVC-300" / "images")
    ensure_dir(JOINT_ROOT / "external" / "CVC-300" / "masks")
    ensure_dir(JOINT_ROOT / "external" / "PolypGen_holdout" / "images")
    ensure_dir(JOINT_ROOT / "external" / "PolypGen_holdout" / "masks")
    ensure_dir(JOINT_ROOT / "manifest")

    manifest_rows = []

    # 计数器
    counters = {
        "kvasir": 1,
        "clinicdb": 1,
        "etis": 1,
        "colondb": 1,
        "cvc300": 1,
        "u": 1,
    }

    # --------
    # A. 训练池：Kvasir-SEG + CVC-ClinicDB
    # --------
    for ds_name in ["Kvasir-SEG", "CVC-ClinicDB"]:
        cfg = DATASETS[ds_name]
        ds_root = cfg["root"]

        image_dir = find_best_subdir(ds_root, cfg["image_dir_candidates"])
        mask_dir = find_best_subdir(ds_root, cfg["mask_dir_candidates"])
        pairs = pair_images_and_masks(image_dir, mask_dir)

        rng.shuffle(pairs)

        n_total = len(pairs)
        n_labeled = max(1, round(n_total * L_SMALL_RATIO))
        labeled_pairs = pairs[:n_labeled]
        unlabeled_pairs = pairs[n_labeled:]

        prefix = cfg["prefix"]

        print(f"[{ds_name}] total={n_total}, L_small={len(labeled_pairs)}, U_large={len(unlabeled_pairs)}")

        # L_small
        for img_src, mask_src in sorted(labeled_pairs, key=lambda x: natural_key(x[0])):
            idx = counters[prefix]
            counters[prefix] += 1

            new_id = f"{prefix}_{idx:04d}"
            img_dst = JOINT_ROOT / "L_small" / "images" / f"{new_id}{img_src.suffix.lower()}"
            mask_dst = JOINT_ROOT / "L_small" / "masks" / f"{new_id}{mask_src.suffix.lower()}"

            safe_copy(img_src, img_dst)
            safe_copy(mask_src, mask_dst)

            manifest_rows.append({
                "id": new_id,
                "image_path": rel_to_project(img_dst),
                "mask_path": rel_to_project(mask_dst),
                "subset": "L_small",
                "split": "train",
                "source": ds_name,
                "center": "",
                "is_labeled": 1,
                "is_pseudo": 0,
                "pseudo_weight": 1.0,
                "round_id": 0,
                "exclude_from_tuning": 0,
            })

        # U_large
        for img_src, mask_src in sorted(unlabeled_pairs, key=lambda x: natural_key(x[0])):
            idx = counters["u"]
            counters["u"] += 1

            new_id = f"u_{idx:04d}"
            img_dst = JOINT_ROOT / "U_large" / "images" / f"{new_id}{img_src.suffix.lower()}"

            safe_copy(img_src, img_dst)

            manifest_rows.append({
                "id": new_id,
                "image_path": rel_to_project(img_dst),
                "mask_path": "",
                "subset": "U_large",
                "split": "unlabeled",
                "source": ds_name,
                "center": "",
                "is_labeled": 0,
                "is_pseudo": 0,
                "pseudo_weight": 0.0,
                "round_id": 0,
                "exclude_from_tuning": 0,
            })

    # --------
    # B. 外部测试集：ETIS / CVC-ColonDB / CVC-300
    # --------
    for ds_name in ["ETIS", "CVC-ColonDB", "CVC-300"]:
        cfg = DATASETS[ds_name]
        ds_root = cfg["root"]

        image_dir = find_best_subdir(ds_root, cfg["image_dir_candidates"])
        mask_dir = find_best_subdir(ds_root, cfg["mask_dir_candidates"])
        pairs = pair_images_and_masks(image_dir, mask_dir)

        prefix = cfg["prefix"]
        external_name = cfg["external_name"]

        print(f"[{ds_name}] external total={len(pairs)}")

        for img_src, mask_src in pairs:
            idx = counters[prefix]
            counters[prefix] += 1

            new_id = f"{prefix}_{idx:04d}"
            img_dst = JOINT_ROOT / "external" / external_name / "images" / f"{new_id}{img_src.suffix.lower()}"
            mask_dst = JOINT_ROOT / "external" / external_name / "masks" / f"{new_id}{mask_src.suffix.lower()}"

            safe_copy(img_src, img_dst)
            safe_copy(mask_src, mask_dst)

            manifest_rows.append({
                "id": new_id,
                "image_path": rel_to_project(img_dst),
                "mask_path": rel_to_project(mask_dst),
                "subset": "external",
                "split": "test",
                "source": ds_name,
                "center": "",
                "is_labeled": 1,
                "is_pseudo": 0,
                "pseudo_weight": 1.0,
                "round_id": 0,
                "exclude_from_tuning": 1,
            })

    # --------
    # C. 写 manifest
    # --------
    manifest_path = JOINT_ROOT / "manifest" / "samples_v1.csv"
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
    ]

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print("\n==============================")
    print("joint_polyp_v1 构建完成")
    print(f"manifest: {manifest_path}")
    print(f"总样本数: {len(manifest_rows)}")
    print("==============================")


if __name__ == "__main__":
    main()