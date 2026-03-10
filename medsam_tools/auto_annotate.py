import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


def parse_args():
    parser = argparse.ArgumentParser(description="Semi-auto annotation with MedSAM/SAM box prompt.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM/MedSAM checkpoint.")
    parser.add_argument("--model-type", type=str, default="vit_b", help="sam_model_registry key, e.g. vit_b")
    parser.add_argument("--image-dir", type=str, default="data/raw_images")
    parser.add_argument("--output-dir", type=str, default="data/processed_images/masks")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-exist", action="store_true", help="Skip images with existing mask.")
    return parser.parse_args()


def image_files(image_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in image_dir.glob("*") if p.suffix.lower() in exts])


def draw_preview(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    predictor = SamPredictor(model)

    files = image_files(image_dir)
    if not files:
        print(f"[warn] no images found in {image_dir}")
        return

    print("操作说明:")
    print("1) 鼠标框选病灶区域，按 Enter/Space 确认")
    print("2) 在预览窗口中按 Y 保存，按 N 跳过")

    for idx, img_path in enumerate(files, start=1):
        out_path = output_dir / f"{img_path.stem}.png"
        if args.skip_exist and out_path.exists():
            print(f"[{idx}/{len(files)}] skip existing: {img_path.name}")
            continue

        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[{idx}/{len(files)}] failed to read: {img_path.name}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        roi = cv2.selectROI("Select ROI", image_bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI")
        if roi[2] <= 0 or roi[3] <= 0:
            print(f"[{idx}/{len(files)}] no ROI selected: {img_path.name}")
            continue

        x, y, w, h = roi
        box = np.array([x, y, x + w, y + h], dtype=np.float32)
        masks, _, _ = predictor.predict(box=box[None, :], point_coords=None, point_labels=None, multimask_output=False)
        mask_u8 = (masks[0] > 0).astype(np.uint8) * 255

        preview = draw_preview(image_bgr, mask_u8)
        cv2.imshow("Preview (Y=save, N=skip)", preview)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("Preview (Y=save, N=skip)")

        if key in [ord("y"), ord("Y"), 13]:
            cv2.imwrite(str(out_path), mask_u8)
            print(f"[{idx}/{len(files)}] saved: {out_path.name}")
        else:
            print(f"[{idx}/{len(files)}] skipped: {img_path.name}")

    cv2.destroyAllWindows()
    print("Annotation completed.")


if __name__ == "__main__":
    main()
