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
    def __init__(self, image_dir, mask_dir, transform=None, boundary_radius: int = 1):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.boundary_radius = boundary_radius

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.image_files = sorted(
            [p for p in self.image_dir.glob("*") if p.suffix.lower() in exts]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"

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
        boundary = generate_boundary_label(mask_bin, radius=self.boundary_radius)

        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_bin.astype(np.float32)).unsqueeze(0)
        boundary_tensor = torch.from_numpy(boundary).unsqueeze(0)

        return {
            "id": img_path.stem,
            "image": image_tensor,
            "mask": mask_tensor,
            "boundary": boundary_tensor,
        }
