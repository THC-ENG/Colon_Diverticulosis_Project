# Recommended Public Colonoscopy Datasets (checked on 2026-02-25)

## Priority

1. **Kvasir-SEG** (polyp segmentation, pixel-level masks)
   - https://datasets.simula.no/kvasir-seg/
2. **BKAI-IGH Neopolyp-Small** (polyp segmentation benchmark)
   - https://www.kaggle.com/datasets/debeshjha1/bkai-igh-neopolyp
3. **CVC-ClinicDB**
   - https://polyp.grand-challenge.org/CVCClinicDB/
4. **CVC-ColonDB**
   - https://polyp.grand-challenge.org/CVCColonDB/
5. **ETIS-Larib Polyp DB**
   - https://polyp.grand-challenge.org/EtisLarib/

## Important Note for Diverticulosis

- Public **diverticulosis-specific segmentation** datasets are still scarce.
- Practical strategy:
  1. Use public polyp segmentation sets for pretraining the segmentation pipeline.
  2. Build your own diverticulosis expert-labeled subset using MedSAM-assisted annotation.
  3. Fine-tune on the expert subset for your final task.
