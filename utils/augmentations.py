import random
from typing import Tuple

import cv2
import numpy as np


def _resize_pair(image: np.ndarray, mask: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = size
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return image, mask


class TrainAugmentor:
    def __init__(
        self,
        out_size: Tuple[int, int] = (256, 256),
        lesion_crop_prob: float = 0.7,
        elastic_prob: float = 0.15,
        distortion_prob: float = 0.2,
        blur_prob: float = 0.2,
        mask_morph_prob: float = 0.08,
    ):
        self.out_size = out_size
        self.lesion_crop_prob = lesion_crop_prob
        self.elastic_prob = elastic_prob
        self.distortion_prob = distortion_prob
        self.blur_prob = blur_prob
        self.mask_morph_prob = mask_morph_prob

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

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
