import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data_protocol import ProtocolSample


def generate_boundary_label(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return np.zeros_like(mask_bin, dtype=np.float32)

    radius = max(1, int(radius))
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    eroded = cv2.erode(mask_bin, kernel, iterations=1)
    boundary = (dilated - eroded) > 0
    return boundary.astype(np.float32)


class ColonDataset(Dataset):
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        image_dir,
        mask_dir,
        transform=None,
        use_boundary: bool = False,
        boundary_radius: int = 1,
        include_ids: set[str] | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.use_boundary = use_boundary
        self.boundary_radius = boundary_radius
        self.include_ids = include_ids

        self.exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = [p for p in self.image_dir.glob("*") if p.suffix.lower() in self.exts]
        if self.include_ids is not None:
            files = [p for p in files if p.stem in self.include_ids]
        self.image_files = sorted(files)

    def __len__(self):
        return len(self.image_files)

    def _find_mask_path(self, image_path: Path) -> Path:
        same_name = self.mask_dir / image_path.name
        if same_name.exists():
            return same_name

        for suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            candidate = self.mask_dir / f"{image_path.stem}{suffix}"
            if candidate.exists():
                return candidate

        raise ValueError(f"Cannot find matching mask for image: {image_path}")

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self._find_mask_path(img_path)

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        mask_bin = (mask > 0).astype(np.uint8)

        image_f = image.astype(np.float32) / 255.0
        image_f = (image_f - self.IMAGENET_MEAN) / self.IMAGENET_STD

        image_tensor = torch.from_numpy(image_f).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_bin.astype(np.float32)).unsqueeze(0)

        sample = {
            "id": img_path.stem,
            "image": image_tensor,
            "mask": mask_tensor,
        }

        if self.use_boundary:
            boundary = generate_boundary_label(mask_bin, radius=self.boundary_radius)
            boundary_tensor = torch.from_numpy(boundary).unsqueeze(0)
            sample["boundary"] = boundary_tensor

        return sample


def _load_float_map(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
    else:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Cannot read map file: {p}")
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        arr = raw
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


class ProtocolSegDataset(Dataset):
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        samples: list[ProtocolSample],
        transform=None,
        use_boundary: bool = False,
        boundary_radius: int = 1,
        mask_threshold: int = 127,
    ):
        self.samples = list(samples)
        self.transform = transform
        self.use_boundary = use_boundary
        self.boundary_radius = boundary_radius
        self.mask_threshold = int(mask_threshold)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        image = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {row.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if row.mask_path:
            mask = cv2.imread(str(row.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Cannot read mask: {row.mask_path}")
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        mask_bin = (mask > self.mask_threshold).astype(np.uint8)
        h, w = mask_bin.shape

        distill_soft = np.zeros((h, w), dtype=np.float32)
        distill_edge = np.zeros((h, w), dtype=np.float32)
        has_distill_soft = 0.0
        has_distill_edge = 0.0

        if row.soft_path and Path(row.soft_path).exists():
            soft = _load_float_map(row.soft_path)
            if soft.shape != (h, w):
                soft = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)
            distill_soft = np.clip(soft, 0.0, 1.0).astype(np.float32)
            has_distill_soft = 1.0

        if row.edge_path and Path(row.edge_path).exists():
            edge = _load_float_map(row.edge_path)
            if edge.shape != (h, w):
                edge = cv2.resize(edge, (w, h), interpolation=cv2.INTER_LINEAR)
            distill_edge = np.clip(edge, 0.0, 1.0).astype(np.float32)
            has_distill_edge = 1.0

        image_f = image.astype(np.float32) / 255.0
        image_f = (image_f - self.IMAGENET_MEAN) / self.IMAGENET_STD

        sample = {
            "id": row.id,
            "subset": row.subset,
            "split": row.split,
            "source": row.source,
            "image": torch.from_numpy(image_f).permute(2, 0, 1),
            "mask": torch.from_numpy(mask_bin.astype(np.float32)).unsqueeze(0),
            "is_labeled": torch.tensor(float(row.is_labeled), dtype=torch.float32),
            "is_pseudo": torch.tensor(float(row.is_pseudo), dtype=torch.float32),
            "pseudo_weight": torch.tensor(float(row.pseudo_weight), dtype=torch.float32),
            "round_id": torch.tensor(int(row.round_id), dtype=torch.int64),
            "exclude_from_tuning": torch.tensor(float(row.exclude_from_tuning), dtype=torch.float32),
            "distill_soft": torch.from_numpy(distill_soft).unsqueeze(0),
            "distill_edge": torch.from_numpy(distill_edge).unsqueeze(0),
            "has_distill_soft": torch.tensor(float(has_distill_soft), dtype=torch.float32),
            "has_distill_edge": torch.tensor(float(has_distill_edge), dtype=torch.float32),
        }

        if self.use_boundary:
            boundary = generate_boundary_label(mask_bin, radius=self.boundary_radius)
            sample["boundary"] = torch.from_numpy(boundary).unsqueeze(0)

        return sample


def split_lsmall_train_val(
    samples: list[ProtocolSample],
    val_ratio: float = 0.2,
    seed: int = 42,
    include_pseudo: bool = True,
    pseudo_max_ratio: float = -1.0,
    pseudo_max_count: int = 0,
) -> tuple[list[ProtocolSample], list[ProtocolSample]]:
    lsmall = [
        s
        for s in samples
        if s.subset == "L_small"
        and s.exclude_from_tuning == 0
        and s.is_pseudo == 0
    ]
    pseudo = [
        s
        for s in samples
        if s.is_pseudo == 1
        and s.exclude_from_tuning == 0
        and s.mask_path
    ]

    explicit_val = [s for s in lsmall if s.split == "val"]
    explicit_train = [s for s in lsmall if s.split != "val"]

    if explicit_val:
        train_l = explicit_train
        val_l = explicit_val
    else:
        work = list(lsmall)
        rng = random.Random(seed)
        rng.shuffle(work)
        n_val = max(1, int(round(len(work) * float(val_ratio)))) if len(work) > 1 else 0
        val_l = work[:n_val]
        train_l = work[n_val:]
        for s in train_l:
            s.split = "train"
        for s in val_l:
            s.split = "val"

    train_rows = list(train_l)
    if include_pseudo:
        pseudo_rows = [s for s in pseudo if s.split in {"", "train", "pseudo_train"}]
        rng = random.Random(seed + 17)
        rng.shuffle(pseudo_rows)

        max_count = len(pseudo_rows)
        if pseudo_max_count and pseudo_max_count > 0:
            max_count = min(max_count, int(pseudo_max_count))

        ratio = float(pseudo_max_ratio)
        if ratio >= 0.0:
            if ratio <= 0.0 or len(train_l) <= 0:
                max_count = 0
            elif ratio < 1.0:
                ratio_cap = int((ratio * len(train_l)) / max(1e-6, 1.0 - ratio))
                max_count = min(max_count, max(0, ratio_cap))

        train_rows.extend(pseudo_rows[:max_count])
    return train_rows, list(val_l)
