from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


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
        # Prefer same filename (including suffix), then fallback to common mask suffixes.
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
