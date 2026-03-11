# Colon Diverticulosis Segmentation Project

[![Status](https://img.shields.io/badge/status-active%20research-blue)]()
[![Task](https://img.shields.io/badge/task-medical%20image%20segmentation-green)]()
[![Model](https://img.shields.io/badge/model-CNN--Transformer-orange)]()

A research-oriented project for **colon diverticulosis lesion segmentation** in endoscopic images, with a focus on:

- **MedSAM-assisted mask generation**
- **expert-quality annotation refinement**
- **hybrid CNN-Transformer segmentation**
- **boundary-aware supervision for fine lesion delineation**

This repository is designed as both a **research codebase** and a **project homepage** for ongoing work in medical image segmentation.

---

## Highlights

- Built a complete pipeline for **lesion segmentation** in colonoscopy images
- Deployed and evaluated **SAM / MedSAM** for annotation assistance
- Implemented a **Res/Swin-UNet-style segmentation framework**
- Constructed both **pseudo-labeled** and **expert-labeled** datasets
- Exploring **dual-task supervision** with segmentation and boundary learning
- Continuously improving performance on **small, high-quality medical datasets**

---

## Motivation

Colon diverticulosis lesions are challenging to segment because of:

- subtle visual appearance
- irregular and ambiguous boundaries
- limited availability of fine-grained public annotations
- strong dependence on annotation quality

This project is motivated by the idea that segmentation performance depends not only on the model, but also on the **quality of supervision**. Therefore, the project combines:

1. **foundation-model-assisted annotation** for efficiency
2. **expert-refined labeling** for better supervision
3. **strong segmentation architectures** for accurate lesion delineation

---

## Project Goals

The current goals of this project are:

- build a complete and extensible segmentation workflow
- compare **pseudo-label supervision** with **expert-label supervision**
- reproduce and improve a Transformer-based segmentation baseline
- investigate whether **boundary-aware auxiliary learning** improves lesion edges
- support future paper writing, reports, or academic presentations

---

## Method Overview

The current workflow can be summarized as:

1. **Data organization**
   - standardize image-mask pair structure
   - prepare train/val/test splits

2. **Mask generation with MedSAM**
   - generate pseudo masks from raw images
   - align and clean masks for downstream use

3. **Expert subset construction**
   - create a small but higher-quality annotated dataset
   - use it for controlled experiments on supervision quality

4. **Model training**
   - train a hybrid CNN-Transformer segmentation model
   - optimize with Dice/Focal-style objectives
   - extend to boundary-aware dual-task learning

5. **Inference and evaluation**
   - run prediction on validation/test sets
   - save masks, visualizations, and metrics

---

## Visual Overview

You can place your future figures here.

### Suggested figures
- pipeline figure
- annotation workflow
- qualitative comparison between pseudo masks, expert masks, and model predictions

Example placeholder:

```markdown
![pipeline](assets/pipeline.png)