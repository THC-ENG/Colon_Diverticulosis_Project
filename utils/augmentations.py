import random
from typing import Tuple

import cv2
import numpy as np


def _resize_pair(image: np.ndarray, mask: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = size
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def _is_polypgen_source(source: str) -> bool:
    return "polypgen" in str(source or "").strip().lower()


def _apply_polypgen_color_domain_augment(
    image: np.ndarray,
    h_shift_max: int = 12,
    sat_scale_range: Tuple[float, float] = (0.70, 1.40),
    val_scale_range: Tuple[float, float] = (0.75, 1.30),
    gamma_range: Tuple[float, float] = (0.75, 1.35),
    clahe_prob: float = 0.60,
) -> np.ndarray:
    out = image.copy()

    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_shift = random.randint(-max(0, int(h_shift_max)), max(0, int(h_shift_max)))
    sat_scale = random.uniform(float(sat_scale_range[0]), float(sat_scale_range[1]))
    val_scale = random.uniform(float(val_scale_range[0]), float(val_scale_range[1]))
    hsv[..., 0] = np.mod(hsv[..., 0] + hue_shift, 180.0)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0.0, 255.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0.0, 255.0)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    gamma = random.uniform(float(gamma_range[0]), float(gamma_range[1]))
    gamma = max(1e-6, gamma)
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255.0 for i in range(256)], dtype=np.float32)
    out = cv2.LUT(out, table.astype(np.uint8))

    if random.random() < float(max(0.0, min(1.0, clahe_prob))):
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return out


class TrainAugmentor:
    def __init__(
        self,
        out_size: Tuple[int, int] = (256, 256),
        lesion_crop_prob: float = 0.7,
        elastic_prob: float = 0.15,
        distortion_prob: float = 0.2,
        blur_prob: float = 0.2,
        mask_morph_prob: float = 0.08,
        polypgen_aug_prob: float = 0.7,
        polypgen_h_shift_max: int = 12,
        polypgen_sat_scale_range: Tuple[float, float] = (0.70, 1.40),
        polypgen_val_scale_range: Tuple[float, float] = (0.75, 1.30),
        polypgen_gamma_range: Tuple[float, float] = (0.75, 1.35),
        polypgen_clahe_prob: float = 0.60,
    ):
        self.out_size = out_size
        self.lesion_crop_prob = lesion_crop_prob
        self.elastic_prob = elastic_prob
        self.distortion_prob = distortion_prob
        self.blur_prob = blur_prob
        self.mask_morph_prob = mask_morph_prob
        self.polypgen_aug_prob = polypgen_aug_prob
        self.polypgen_h_shift_max = polypgen_h_shift_max
        self.polypgen_sat_scale_range = polypgen_sat_scale_range
        self.polypgen_val_scale_range = polypgen_val_scale_range
        self.polypgen_gamma_range = polypgen_gamma_range
        self.polypgen_clahe_prob = polypgen_clahe_prob

    def _crop_near_lesion(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape
        if h == 0 or w == 0:
            return image, mask

        scale = random.uniform(0.75, 1.0)
        crop_h = max(int(h * scale), min(h, self.out_size[0]))
        crop_w = max(int(w * scale), min(w, self.out_size[1]))

        ys, xs = np.where(mask > 0)
        use_lesion = ys.size > 0 and random.random() < self.lesion_crop_prob

        if use_lesion:
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            lesion_h = max(1, y_max - y_min + 1)
            lesion_w = max(1, x_max - x_min + 1)
            cy = int((y_min + y_max) / 2 + random.uniform(-0.1, 0.1) * lesion_h)
            cx = int((x_min + x_max) / 2 + random.uniform(-0.1, 0.1) * lesion_w)
            top = np.clip(cy - crop_h // 2, 0, h - crop_h)
            left = np.clip(cx - crop_w // 2, 0, w - crop_w)
        else:
            top = 0 if crop_h == h else random.randint(0, h - crop_h)
            left = 0 if crop_w == w else random.randint(0, w - crop_w)

        bottom = top + crop_h
        right = left + crop_w
        return image[top:bottom, left:right].copy(), mask[top:bottom, left:right].copy()

    def _slight_distortion(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape
        max_shift = max(1, int(0.03 * min(h, w)))

        src = np.float32(
            [
                [0, 0],
                [w - 1, 0],
                [0, h - 1],
                [w - 1, h - 1],
            ]
        )
        jitter = np.random.randint(-max_shift, max_shift + 1, size=(4, 2)).astype(np.float32)
        dst = src + jitter
        dst[:, 0] = np.clip(dst[:, 0], 0, w - 1)
        dst[:, 1] = np.clip(dst[:, 1], 0, h - 1)

        mat = cv2.getPerspectiveTransform(src, dst)
        image = cv2.warpPerspective(
            image,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpPerspective(
            mask,
            mat,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return image, mask

    def _elastic_transform(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape
        alpha = random.uniform(2.0, 6.0)
        sigma = random.uniform(6.0, 10.0)

        dx = (np.random.rand(h, w).astype(np.float32) * 2.0 - 1.0)
        dy = (np.random.rand(h, w).astype(np.float32) * 2.0 - 1.0)
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        map_x = np.clip(grid_x + dx, 0, w - 1)
        map_y = np.clip(grid_y + dy, 0, h - 1)

        image = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.remap(
            mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return image, mask

    def __call__(self, image: np.ndarray, mask: np.ndarray, source: str = "") -> Tuple[np.ndarray, np.ndarray]:
        image, mask = self._crop_near_lesion(image, mask)

        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        if random.random() < 0.25:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        if random.random() < self.distortion_prob:
            image, mask = self._slight_distortion(image, mask)
        if random.random() < self.elastic_prob:
            image, mask = self._elastic_transform(image, mask)

        if random.random() < self.blur_prob:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)

        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-12.0, 12.0)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        if _is_polypgen_source(source) and random.random() < float(max(0.0, min(1.0, self.polypgen_aug_prob))):
            image = _apply_polypgen_color_domain_augment(
                image,
                h_shift_max=self.polypgen_h_shift_max,
                sat_scale_range=self.polypgen_sat_scale_range,
                val_scale_range=self.polypgen_val_scale_range,
                gamma_range=self.polypgen_gamma_range,
                clahe_prob=self.polypgen_clahe_prob,
            )

        if random.random() < self.mask_morph_prob and (mask > 0).any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            if random.random() < 0.5:
                mask = cv2.erode(mask, kernel, iterations=1)
            else:
                mask = cv2.dilate(mask, kernel, iterations=1)

        mask = ((mask > 0).astype(np.uint8) * 255)
        return _resize_pair(image, mask, self.out_size)


class ValAugmentor:
    def __init__(self, out_size: Tuple[int, int] = (256, 256)):
        self.out_size = out_size

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _resize_pair(image, mask, self.out_size)


class DistillTrainAugmentor:
    def __init__(
        self,
        out_size: Tuple[int, int] = (256, 256),
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.2,
        blur_prob: float = 0.15,
        polypgen_aug_prob: float = 0.7,
        polypgen_h_shift_max: int = 12,
        polypgen_sat_scale_range: Tuple[float, float] = (0.70, 1.40),
        polypgen_val_scale_range: Tuple[float, float] = (0.75, 1.30),
        polypgen_gamma_range: Tuple[float, float] = (0.75, 1.35),
        polypgen_clahe_prob: float = 0.60,
    ):
        self.out_size = out_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.blur_prob = blur_prob
        self.polypgen_aug_prob = polypgen_aug_prob
        self.polypgen_h_shift_max = polypgen_h_shift_max
        self.polypgen_sat_scale_range = polypgen_sat_scale_range
        self.polypgen_val_scale_range = polypgen_val_scale_range
        self.polypgen_gamma_range = polypgen_gamma_range
        self.polypgen_clahe_prob = polypgen_clahe_prob

    def __call__(self, image: np.ndarray, mask: np.ndarray, source: str = "") -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.hflip_prob:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < self.vflip_prob:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        alpha = random.uniform(0.92, 1.08)
        beta = random.uniform(-10.0, 10.0)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        if _is_polypgen_source(source) and random.random() < float(max(0.0, min(1.0, self.polypgen_aug_prob))):
            image = _apply_polypgen_color_domain_augment(
                image,
                h_shift_max=self.polypgen_h_shift_max,
                sat_scale_range=self.polypgen_sat_scale_range,
                val_scale_range=self.polypgen_val_scale_range,
                gamma_range=self.polypgen_gamma_range,
                clahe_prob=self.polypgen_clahe_prob,
            )

        if random.random() < self.blur_prob:
            image = cv2.GaussianBlur(image, (3, 3), sigmaX=0)

        mask = ((mask > 0).astype(np.uint8) * 255)
        return _resize_pair(image, mask, self.out_size)
