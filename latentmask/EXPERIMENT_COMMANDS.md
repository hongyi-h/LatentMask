# LatentMask Experiment Commands Reference

> Maps every table in the paper to the exact commands needed to reproduce each number.

## Prerequisites

```bash
# 1. Setup
bash setup.sh

# 2. Environment variables (add to ~/.zshrc or ~/.bashrc)
export nnUNet_raw="$HOME/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_data/nnUNet_results"
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"

# 3. Data conversion + nnUNet preprocessing
bash latentmask/scripts/00_convert_datasets.sh
bash latentmask/scripts/01_precompute_vesselness.sh
```

---

## Table 1: Dataset Composition

No experiments needed — descriptive table from dataset stats.

---

## Table 2: Main PE Segmentation Results (5-fold CV on READ-PE)

### Row: nnUNet (pixel-only)
```bash
# Train (5 folds)
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD -tr nnUNetTrainer --npz
done

# Evaluate
python latentmask/scripts/05_compute_metrics.py \
    --results_root $nnUNet_results --dataset_id 100 --trainer nnUNetTrainer --mode cv
```

### Row: nnPU-Seg
```bash
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD -tr nnPUSegTrainer --npz
done

python latentmask/scripts/05_compute_metrics.py \
    --results_root $nnUNet_results --dataset_id 100 --trainer nnPUSegTrainer --mode cv
```

### Row: LatentMask (full)
```bash
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD -tr LatentMaskTrainer --npz
done

python latentmask/scripts/05_compute_metrics.py \
    --results_root $nnUNet_results --dataset_id 100 --trainer LatentMaskTrainer --mode cv
```

### Rows: Mean Teacher, 3D BoxSup, ConfKD-Mixed, WeakMedSAM-3D
> These baselines require separate implementations. Use their official repos:
> - Mean Teacher: https://github.com/CuriousAI/mean-teacher
> - 3D BoxSup: Adapt 2D BoxSup to 3D using nnUNet backbone
> - ConfKD-Mixed: https://github.com/HiLab-git/SSL4MIS (closest codebase)
> - WeakMedSAM-3D: Adapt MedSAM with weak supervision
>
> Train each baseline on Dataset100 + auxiliary data, then evaluate with:
> ```bash
> python latentmask/scripts/05_compute_metrics.py --pred_dir <baseline_preds> \
>     --gt_dir $nnUNet_raw/Dataset100_READPE/labelsTr --output <output.json>
> ```

---

## Table 3: Ablation Study

### Row A0: Naive tri-granularity
```bash
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer_A0_NaiveMixed --npz
done

python latentmask/scripts/05_compute_metrics.py \
    --results_root $nnUNet_results --dataset_id 100 \
    --trainer LatentMaskTrainer_A0_NaiveMixed --mode cv
```

### Row A1: + PU (uniform e=0.5)
```bash
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer_A1_UniformPU --npz
done

python latentmask/scripts/05_compute_metrics.py \
    --results_root $nnUNet_results --dataset_id 100 \
    --trainer LatentMaskTrainer_A1_UniformPU --mode cv
```

### Row A2: + APN (learned e)
```bash
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer_A2_APNOnly --npz
done

python latentmask/scripts/05_compute_metrics.py \
    --results_root $nnUNet_results --dataset_id 100 \
    --trainer LatentMaskTrainer_A2_APNOnly --mode cv
```

### Row A3: + Anatomy reg
```bash
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer_A3_NoRefine --npz
done

python latentmask/scripts/05_compute_metrics.py \
    --results_root $nnUNet_results --dataset_id 100 \
    --trainer LatentMaskTrainer_A3_NoRefine --mode cv
```

### Row A4: Full LatentMask
Same as Table 2 LatentMask row — use `LatentMaskTrainer`.

### Shortcut: Run all ablations at once
```bash
bash latentmask/scripts/03_train_ablations.sh 100 3d_fullres 0
```

---

## Table 4: Annotation Budget Study

Requires training with different subsets of pixel/box/image data. Modify the data splits:

```bash
# "Pixel only" (40 pixel, 0 box, 0 image)
# Unset auxiliary manifests
unset LATENTMASK_RSPECT_MANIFEST
unset LATENTMASK_AUG_RSPECT_MANIFEST
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD -tr LatentMaskTrainer --npz
done

# "Pixel + Box" (20 pixel, 200 box, 0 image)
# Create a sub-split with 20 pixel cases:
python -c "
import json, numpy as np
from pathlib import Path
splits = json.load(open('$nnUNet_preprocessed/Dataset100_READPE/splits_final.json'))
for fold in splits:
    fold['train'] = fold['train'][:int(len(fold['train'])*0.5)]  # Keep 50% pixel
json.dump(splits, open('$nnUNet_preprocessed/Dataset100_READPE/splits_budget_20pix.json','w'))
"
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"
unset LATENTMASK_RSPECT_MANIFEST
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD -tr LatentMaskTrainer --npz
done

# "LatentMask balanced" (12 pixel, 220 box, 4000 image) and "LatentMask full" (20 pixel, 220 box, 4000 image)
# Set both manifests and adjust pixel split accordingly
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"
# Train with appropriate splits
```

Then evaluate each setting with `05_compute_metrics.py --mode cv`.

---

## Table 5: Propensity Calibration Quality

```bash
# After training fold 0 of LatentMask
python latentmask/scripts/06_propensity_analysis.py \
    --checkpoint "$nnUNet_results/Dataset100_READPE/LatentMaskTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth" \
    --dataset_dir "$nnUNet_raw/Dataset100_READPE" \
    --output propensity_calibration.json
```

---

## Table 6: Robustness to Incomplete Weak Labels

```bash
bash latentmask/scripts/07_robustness_test.sh 100 0
```

Or manually for each drop rate:
```bash
for DROP_RATE in 0.0 0.1 0.2 0.3 0.4; do
    export LATENTMASK_WEAK_DROP_RATE=$DROP_RATE
    for TRAINER in LatentMaskTrainer nnPUSegTrainer; do
        for FOLD in 0 1 2 3 4; do
            CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD -tr $TRAINER --npz
        done
        python latentmask/scripts/05_compute_metrics.py \
            --results_root $nnUNet_results --dataset_id 100 --trainer $TRAINER --mode cv
    done
done
```

---

## Table 7: Cross-Center External Generalization

```bash
# Train on Dataset100 (READ-PE) + auxiliary
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD -tr LatentMaskTrainer --npz
done

# Predict on FUMPE (Dataset101)
bash latentmask/scripts/04_eval_external.sh 100 3d_fullres LatentMaskTrainer

# Compute metrics on FUMPE
python latentmask/scripts/05_compute_metrics.py \
    --pred_dir "$nnUNet_results/external_eval/LatentMaskTrainer/dataset101" \
    --gt_dir "$nnUNet_raw/Dataset101_FUMPE/labelsTr" \
    --output external_fumpe.json

# Compute metrics on CAD-PE
python latentmask/scripts/05_compute_metrics.py \
    --pred_dir "$nnUNet_results/external_eval/LatentMaskTrainer/dataset102" \
    --gt_dir "$nnUNet_raw/Dataset102_CADPE/labelsTr" \
    --output external_cadpe.json

# Repeat for each baseline trainer
for TRAINER in nnUNetTrainer nnPUSegTrainer; do
    bash latentmask/scripts/04_eval_external.sh 100 3d_fullres $TRAINER
done
```

---

## Table 8: Lesion-Location Sensitivity Breakdown

Derived from `05_compute_metrics.py` output. Requires post-processing predictions
by anatomical region (proximal / segmental / subsegmental):

```bash
python -c "
from latentmask.utils.metrics import evaluate_case
# Load prediction and GT, then mask by vesselness level before computing metrics
# proximal: vesselness >= 0.6
# segmental: 0.3 <= vesselness < 0.6
# subsegmental: vesselness < 0.3
"
```

---

## Table 9: Lesion-Size Recall Breakdown

Derived from `05_compute_metrics.py` per-case results:
- Large: lesion volume > 1000 voxels
- Medium: 100-1000 voxels
- Small: < 100 voxels

```bash
python -c "
import json
# Load cv_metrics.json, stratify by GT lesion size
# Report recall at each size tier
"
```

---

## Table 10: Training and Inference Efficiency

```bash
# Parameter count
python -c "
from latentmask.trainer.latentmask_trainer import LatentMaskTrainer
# Count parameters from checkpoint
import torch
ckpt = torch.load('checkpoint_best.pth', map_location='cpu', weights_only=False)
n_net = sum(p.numel() for p in ckpt['network_weights'].values())
n_apn = sum(p.numel() for p in ckpt.get('apn_state', {}).values())
print(f'Network: {n_net/1e6:.1f}M, APN: {n_apn/1e6:.1f}M, Total: {(n_net+n_apn)/1e6:.1f}M')
"

# GPU memory: read from training log or nvidia-smi during training
# Training time: parse training log timestamps
# Inference time: time nnUNetv2_predict on test set
time nnUNetv2_predict \
    -i "$nnUNet_raw/Dataset102_CADPE/imagesTr" \
    -o "/tmp/inference_timing" \
    -d 100 -c 3d_fullres -tr LatentMaskTrainer -f 0
```

---

## Table 11: Statistical Significance

```bash
bash latentmask/scripts/08_significance_test.sh 100
```

Or programmatically:
```python
from latentmask.utils.metrics import paired_bootstrap_test

# Load per-case Dice scores for two methods from cv_metrics.json
result = paired_bootstrap_test(latentmask_dices, baseline_dices, n_resamples=10000)
print(f"p = {result['p_value']:.4f}")
```

---

## CAD-PE Secondary Benchmark (91 cases)

Same as above but with Dataset 102:

```bash
# Plan and preprocess CAD-PE
nnUNetv2_plan_and_preprocess -d 102 --verify_dataset_integrity -c 3d_fullres

# Train all methods
for TRAINER in nnUNetTrainer nnPUSegTrainer LatentMaskTrainer; do
    for FOLD in 0 1 2 3 4; do
        CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 102 3d_fullres $FOLD -tr $TRAINER --npz
    done
done

# Evaluate
for TRAINER in nnUNetTrainer nnPUSegTrainer LatentMaskTrainer; do
    python latentmask/scripts/05_compute_metrics.py \
        --results_root $nnUNet_results --dataset_id 102 --trainer $TRAINER --mode cv
done
```

---

## Full Pipeline Summary

```bash
# Step 0: Setup + environment
bash setup.sh
source ~/.zshrc  # pick up env vars

# Step 1: Data conversion
bash latentmask/scripts/00_convert_datasets.sh
bash latentmask/scripts/01_precompute_vesselness.sh

# Step 2: Main method (Table 2, A4 row; Table 3 A4 row)
bash latentmask/scripts/02_train_latentmask.sh 100 3d_fullres LatentMaskTrainer 0

# Step 3: All ablations + baselines (Table 2 other rows; Table 3)
bash latentmask/scripts/03_train_ablations.sh 100 3d_fullres 0

# Step 4: External evaluation (Table 7)
bash latentmask/scripts/04_eval_external.sh 100 3d_fullres LatentMaskTrainer

# Step 5: Compute metrics (Tables 2, 3, 7, 8, 9)
python latentmask/scripts/05_compute_metrics.py --mode cv --dataset_id 100

# Step 6: Propensity analysis (Table 5)
python latentmask/scripts/06_propensity_analysis.py --checkpoint <path> --dataset_dir <path>

# Step 7: Robustness (Table 6)
bash latentmask/scripts/07_robustness_test.sh 100 0

# Step 8: Significance (Table 11)
bash latentmask/scripts/08_significance_test.sh 100

# Step 9: CAD-PE benchmark
bash latentmask/scripts/02_train_latentmask.sh 102 3d_fullres LatentMaskTrainer 0
bash latentmask/scripts/03_train_ablations.sh 102 3d_fullres 0
```

---

## GPU Time Estimate

| Experiment | Folds | Trainers | Est. GPU-hours (A100 40GB) |
|---|---:|---:|---:|
| Table 2 (READ-PE main) | 5 | 7 | ~175 |
| Table 3 (ablations) | 5 | 5 | ~125 |
| Table 4 (budget study) | 5 | 6 settings | ~150 |
| Table 5 (propensity) | 1 | 1 | ~1 |
| Table 6 (robustness) | 5 | 2 × 5 rates | ~100 |
| Table 7 (external) | inference only | 5 | ~5 |
| CAD-PE benchmark | 5 | 7 | ~250 |
| **Total** | | | **~806** |
