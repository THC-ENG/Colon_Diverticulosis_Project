# Experiment Utilities

## 1)  5-fold 
```bash
python tools/make_kfold_splits.py --image-dir data/processed_images/images/all --num-folds 5 --seed 42 --output data/splits/kfold_5_seed42.json
```

## 2)  3 seeds x 5 folds
```bash
python tools/run_kfold_seeds.py --config configs/train_res_swin_unet.yaml --split-json data/splits/kfold_5_seed42.json --num-folds 5 --seeds 42,43,44 --all-image-dir data/processed_images/images/all --all-mask-dir data/processed_images/masks/all
```

## 3) 两阶段训练脚本
```bash
python tools/run_two_stage.py --config configs/train_res_swin_unet.yaml --pretrain-dataset-root data/public_polyp_dataset --finetune-dataset-root data/expert_dataset --seed 42
```

## 4) 评估脚本
```bash
python inference_eval.py --image-dir data/processed_images/images/test --mask-dir data/processed_images/masks/test --checkpoint checkpoints/best_model.pth --threshold-search --val-image-dir data/processed_images/images/val --val-mask-dir data/processed_images/masks/val --per-sample-report results/per_sample_report.json
```

## 5) 难例筛选
```bash
python tools/select_hard_examples.py --per-sample-report results/per_sample_report.json --top-k 20 --strategy composite --output results/hard_examples.json
```

## 6) 显著性统计聚合
```bash
python tools/aggregate_metrics.py --group baseline=results/ablation/baseline_seed*.json --group swin_fix=results/ablation/swin_fix_seed*.json --metric dice_mean --compare-to baseline --output results/ablation_summary.json
```
