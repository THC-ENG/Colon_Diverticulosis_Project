# Colon Diverticulosis Segmentation Pipeline

## 1) Data Layout

```text
data/
  raw_images/                     # Unlabeled colonoscopy images
  processed_images/
    images/                       # Final training images
    masks/                        # Final training masks
checkpoints/
results/
```

## 2) MedSAM LoRA Fine-tuning (optional)

```bash
python medsam_tools/finetune_lora.py \
  --checkpoint "D:/Embodied AI/SAM/MedSAM/medsam_vit_b/medsam_vit_b.pth" \
  --image-dir data/processed_images/images \
  --mask-dir data/processed_images/masks \
  --epochs 10 \
  --batch-size 1 \
  --save-path checkpoints/medsam_lora_best.pth
```

## 3) Semi-auto Annotation with Box Prompt

```bash
python medsam_tools/auto_annotate.py \
  --checkpoint "D:/Embodied AI/SAM/MedSAM/medsam_vit_b/medsam_vit_b.pth" \
  --model-type vit_b \
  --image-dir data/raw_images \
  --output-dir data/processed_images/masks \
  --skip-exist
```

## 4) Train Res_Swin_UNet

```bash
python train.py \
  --image-dir data/processed_images/images \
  --mask-dir data/processed_images/masks \
  --epochs 100 \
  --batch-size 4 \
  --img-size 256 \
  --save-path checkpoints/best_model.pth
```

## 5) Inference + Metrics (Dice/IoU)

```bash
python inference_eval.py \
  --image-dir data/processed_images/images \
  --mask-dir data/processed_images/masks \
  --checkpoint checkpoints/best_model.pth \
  --save-dir results/preds \
  --report-path results/metrics_report.json
```

## 6) Network Design Implemented

- Shallow encoder: keep ResNet34 early stages (`conv1`, `layer1`, `layer2`)
- Deep encoder: use Swin-style transformer stages (`SwinStage`) for deep semantic modeling
- Skip connection: attention gating before concatenation (`AttentionGate`)
