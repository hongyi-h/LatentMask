# LatentMask v2 — Detailed Experimental Tables and Launch Commands

> **Note:** All quantitative values are `XX.X` / `PLACEHOLDER` — **no experiments have been run yet**.
> Each table includes launch commands. See `05_real_experiment_replacement_checklist.md` for the full mapping from command → placeholder location.
> **Version**: 2.0 — Redesigned for PropNet (domain-agnostic) + multi-domain validation.

---

## Prerequisites

```bash
# 1. Install environment
bash setup.sh

# 2. Set nnUNet environment variables (~/.zshrc or ~/.bashrc)
export nnUNet_raw="$HOME/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_data/nnUNet_results"
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"

# 3. Data conversion & preprocessing
# --- PE datasets ---
bash latentmask/scripts/00_convert_datasets.sh
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity   # READ-PE
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity   # FUMPE
nnUNetv2_plan_and_preprocess -d 102 --verify_dataset_integrity   # CAD-PE

# --- LiTS dataset ---
bash latentmask/scripts/convert_lits.sh
nnUNetv2_plan_and_preprocess -d 200 --verify_dataset_integrity   # LiTS

# --- PASCAL VOC 2012 ---
bash latentmask/scripts/prepare_voc.sh   # Downloads & builds multi-gran splits

# 4. Build synthetic multi-granularity splits
python latentmask/scripts/build_synthetic_multigran.py \
    --dataset lits --pixel_ratio 0.3 --box_ratio 0.3 --image_ratio 0.4 --seed 42
python latentmask/scripts/build_synthetic_multigran.py \
    --dataset voc --pixel_ratio 0.2 --box_ratio 0.3 --image_ratio 0.5 --seed 42
```

---

## GPU Budget Summary

| Experiment block | Runs | GPU-hours (A100 est.) |
|---|---:|---:|
| PE main comparison (Table 2) | 7 methods × 5 folds | ~700 |
| PE ablation (Table 3) | 5 variants × 5 folds | ~500 |
| LiTS main comparison (Table 4) | 6 methods × 3 folds | ~450 |
| LiTS ablation (Table 5) | 5 variants × 3 folds | ~375 |
| VOC main comparison (Table 6) | 7 methods × 3 runs | ~350 |
| VOC ablation (Table 7) | 5 variants × 3 runs | ~250 |
| Annotation budget study | 5 ratios × 3 benchmarks × 3 runs | ~300 |
| Robustness experiment | 4 missingness levels × 3 benchmarks × 3 runs | ~200 |
| **Total** | | **~3,125** |

---

## Table 1: Dataset Composition (Descriptive)

> **Purpose:** Descriptive overview. No training needed.

| Dataset | Domain | Dim | Supervision | Scale | Usage |
|---|---|---|---|---:|---|
| READ-PE | Medical CT | 3D | Pixel-level masks | 40 exams | Internal CV |
| CAD-PE | Medical CT | 3D | Pixel-level masks | 91 scans | Secondary benchmark |
| FUMPE | Medical CT | 3D | Pixel-level masks | 35 cases | External eval |
| Aug-RSPECT | Medical CT | 3D | Box-level (30K boxes) | 445 studies | Box-level PU |
| RSPECT | Medical CT | 3D | Image-level | 12,195 patients | Image-level PU |
| LiTS | Medical CT | 3D | Pixel → synth 30/30/40 | 131 train, 70 test | Synth multi-gran |
| PASCAL VOC | Natural image | 2D | Pixel → synth 20/30/50 | ~10K train, 1.4K val | Synth multi-gran |

---

## Table 2: PE Segmentation — Main Comparison (Internal 5-fold CV)

> **Purpose:** Core claim — LatentMask outperforms all baselines on real multi-granularity PE data.
> **Protocol:** 5-fold CV on READ-PE (40 exams). Box/image data from Aug-RSPECT and RSPECT during training.

| Method | Pixel | Box | Image | Dice ↑ | HD95 ↓ | Lesion-F1 ↑ | Recall-small ↑ | FP/scan ↓ |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| nnUNet (pixel-only) | ✓ | — | — | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±.XXX | X.XX±.XX |
| Mean Teacher | ✓ | — | ✓ | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±.XXX | X.XX±.XX |
| CPS | ✓ | — | ✓ | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±.XXX | X.XX±.XX |
| nnPU-Seg (uniform e) | ✓ | — | ✓ | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±.XXX | X.XX±.XX |
| 3D BoxSup | — | ✓ | — | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±.XXX | X.XX±.XX |
| Mixed Naive | ✓ | ✓ | ✓ | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±.XXX | X.XX±.XX |
| **LatentMask (ours)** | ✓ | ✓ | ✓ | **XX.X±.XXX** | **XX.X±X.X** | **XX.X±.XXX** | **XX.X±.XXX** | **X.XX±.XX** |

### Launch Commands — Table 2

```bash
# === Row 1: nnUNet (pixel-only) ===
unset LATENTMASK_RSPECT_MANIFEST LATENTMASK_AUG_RSPECT_MANIFEST
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr nnUNetTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer nnUNetTrainer --mode cv

# === Row 2: Mean Teacher ===
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
unset LATENTMASK_AUG_RSPECT_MANIFEST
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr MeanTeacher3DTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer MeanTeacher3DTrainer --mode cv

# === Row 3: CPS ===
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr CPSTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer CPSTrainer --mode cv

# === Row 4: nnPU-Seg (uniform propensity) ===
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
unset LATENTMASK_AUG_RSPECT_MANIFEST
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr nnPUSegTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer nnPUSegTrainer --mode cv

# === Row 5: 3D BoxSup ===
unset LATENTMASK_RSPECT_MANIFEST
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr BoxSup3DTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer BoxSup3DTrainer --mode cv

# === Row 6: Mixed Naive ===
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr MixedNaiveTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer MixedNaiveTrainer --mode cv

# === Row 7: LatentMask (ours) ===
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer LatentMaskTrainer --mode cv
```

---

## Table 3: Ablation Study — PE (5-fold CV on READ-PE)

> **Purpose:** Demonstrate that every component contributes. Key comparison: A1 vs A2 (learned vs uniform propensity).

| Variant | PropNet | PU corr. | Smooth | EMA ref. | Dice ↑ | HD95 ↓ | Recall-small ↑ |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|
| A0: Mixed Naive | ✗ | ✗ | ✗ | ✗ | XX.X±.XXX | XX.X±X.X | XX.X±.XXX |
| A1: + PU (uniform e=0.5) | ✗ | ✓ | ✗ | ✗ | XX.X±.XXX | XX.X±X.X | XX.X±.XXX |
| A2: + PropNet | ✓ | ✓ | ✗ | ✗ | XX.X±.XXX | XX.X±X.X | XX.X±.XXX |
| A3: + Smoothness | ✓ | ✓ | ✓ | ✗ | XX.X±.XXX | XX.X±X.X | XX.X±.XXX |
| A4: Full LatentMask | ✓ | ✓ | ✓ | ✓ | **XX.X±.XXX** | **XX.X±X.X** | **XX.X±.XXX** |

### Launch Commands — Table 3

```bash
# A0: Mixed Naive (same as Table 2 Row 6)
# (see above)

# A1: PU with uniform propensity
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz \
        -p nnUNetPlans \
        --c propnet_mode=uniform,use_smoothness=false,use_ema_refinement=false
done

# A2: PU with PropNet, no smoothness, no EMA
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz \
        --c propnet_mode=learned,use_smoothness=false,use_ema_refinement=false
done

# A3: PU with PropNet + smoothness, no EMA
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz \
        --c propnet_mode=learned,use_smoothness=true,use_ema_refinement=false
done

# A4: Full LatentMask (same as Table 2 Row 7)
# (see above)
```

---

## Table 4: LiTS Liver Tumor — Main Comparison (3-fold CV)

> **Purpose:** Show generality on another 3D medical task with synthetic multi-granularity.
> **Protocol:** 3-fold CV on LiTS training (131 cases). Synthetic degradation: 30% pixel, 30% box, 40% image-level.

| Method | Supervision | Dice ↑ | HD95 ↓ | Lesion-F1 ↑ | Recall-small ↑ |
|---|---|---:|---:|---:|---:|
| nnUNet Oracle | 100% pixel | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±.XXX |
| nnUNet (30% pixel) | 30% pixel only | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±.XXX |
| Mean Teacher | 30% pix + 70% unlab | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±.XXX |
| nnPU-Seg (uniform) | 30% pix + 70% PU | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±.XXX |
| Mixed Naive | 30/30/40 split | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±.XXX |
| **LatentMask (ours)** | 30/30/40 + PropNet | **XX.X±.XXX** | **XX.X±X.X** | **XX.X±.XXX** | **XX.X±.XXX** |

### Launch Commands — Table 4

```bash
# Oracle
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr nnUNetTrainer --npz \
        --c data_split=oracle
done

# 30% pixel only
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr nnUNetTrainer --npz \
        --c data_split=pixel_only
done

# Mean Teacher
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr MeanTeacher3DTrainer --npz \
        --c data_split=semisup
done

# nnPU-Seg
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr nnPUSegTrainer --npz \
        --c data_split=pu
done

# Mixed Naive
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr MixedNaiveTrainer --npz \
        --c data_split=multigran
done

# LatentMask (ours)
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz \
        --c data_split=multigran
done
```

---

## Table 5: LiTS Ablation (3-fold CV)

> **Purpose:** Same ablation structure as PE, on a different domain.

| Variant | Dice ↑ | HD95 ↓ | Recall-small ↑ |
|---|---:|---:|---:|
| A0: Mixed Naive | XX.X±.XXX | XX.X±X.X | XX.X±.XXX |
| A1: + PU (uniform) | XX.X±.XXX | XX.X±X.X | XX.X±.XXX |
| A2: + PropNet | XX.X±.XXX | XX.X±X.X | XX.X±.XXX |
| A3: + Smoothness | XX.X±.XXX | XX.X±X.X | XX.X±.XXX |
| A4: Full LatentMask | **XX.X±.XXX** | **XX.X±X.X** | **XX.X±.XXX** |

Launch commands: same as Table 3 but with `--dataset_id 200` and 3 folds.

---

## Table 6: PASCAL VOC 2012 — Main Comparison (3 runs, mean±std)

> **Purpose:** Prove cross-domain generality on 2D natural images.
> **Protocol:** 3 independent runs with different seeds. Synthetic degradation: 20% pixel, 30% box, 50% image-level.
> **Backbone:** DeepLabV3-ResNet50.

| Method | Supervision | mIoU ↑ | Small-obj IoU ↑ |
|---|---|---:|---:|
| DeepLabV3 Oracle | 100% pixel | XX.X±X.X | XX.X±X.X |
| DeepLabV3 (20% pixel) | 20% pixel only | XX.X±X.X | XX.X±X.X |
| UniMatch (semi-sup) | 20% pix + 80% unlab | XX.X±X.X | XX.X±X.X |
| CAM-baseline | 50% image → CAM | XX.X±X.X | XX.X±X.X |
| BoxSup | 30% box → GrabCut | XX.X±X.X | XX.X±X.X |
| Mixed Naive | 20/30/50 split | XX.X±X.X | XX.X±X.X |
| **LatentMask (ours)** | 20/30/50 + PropNet | **XX.X±X.X** | **XX.X±X.X** |

### Launch Commands — Table 6

```bash
# Oracle
python latentmask/train_voc.py \
    --config configs/voc/oracle.yaml \
    --seed 42 --gpus 1
# Repeat with --seed 43, --seed 44

# 20% pixel only
python latentmask/train_voc.py \
    --config configs/voc/pixel_only_20pct.yaml \
    --seed 42 --gpus 1

# UniMatch
python latentmask/train_voc.py \
    --config configs/voc/unimatch.yaml \
    --seed 42 --gpus 1

# CAM-baseline
python latentmask/train_voc.py \
    --config configs/voc/cam_baseline.yaml \
    --seed 42 --gpus 1

# BoxSup
python latentmask/train_voc.py \
    --config configs/voc/boxsup.yaml \
    --seed 42 --gpus 1

# Mixed Naive
python latentmask/train_voc.py \
    --config configs/voc/mixed_naive.yaml \
    --seed 42 --gpus 1

# LatentMask (ours)
python latentmask/train_voc.py \
    --config configs/voc/latentmask.yaml \
    --seed 42 --gpus 1
```

---

## Table 7: VOC Ablation (3 runs)

| Variant | mIoU ↑ | Small-obj IoU ↑ |
|---|---:|---:|
| A0: Mixed Naive | XX.X±X.X | XX.X±X.X |
| A1: + PU (uniform) | XX.X±X.X | XX.X±X.X |
| A2: + PropNet | XX.X±X.X | XX.X±X.X |
| A3: + Smoothness | XX.X±X.X | XX.X±X.X |
| A4: Full LatentMask | **XX.X±X.X** | **XX.X±X.X** |

---

## Table 8: PE External Evaluation (Cross-Dataset)

> **Purpose:** Show generalization to unseen datasets.
> **Protocol:** Train on READ-PE (pixel) + Aug-RSPECT (box) + RSPECT (image), evaluate on FUMPE and CAD-PE.

| Method | FUMPE Dice ↑ | FUMPE HD95 ↓ | CAD-PE Dice ↑ | CAD-PE HD95 ↓ |
|---|---:|---:|---:|---:|
| nnUNet (pixel-only) | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±X.X |
| Mixed Naive | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±X.X |
| nnPU-Seg (uniform) | XX.X±.XXX | XX.X±X.X | XX.X±.XXX | XX.X±X.X |
| **LatentMask (ours)** | **XX.X±.XXX** | **XX.X±X.X** | **XX.X±.XXX** | **XX.X±X.X** |

### Launch Commands — Table 8

```bash
# Train once on all READ-PE folds combined (no held-out):
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres all \
    -tr LatentMaskTrainer --npz

# Evaluate on FUMPE
python latentmask/scripts/eval_external.py \
    --model_dir $nnUNet_results/Dataset100_READPE/LatentMaskTrainer__nnUNetPlans__3d_fullres/fold_all \
    --test_dir $nnUNet_raw/Dataset101_FUMPE \
    --output fumpe_external_results.json

# Evaluate on CAD-PE
python latentmask/scripts/eval_external.py \
    --model_dir $nnUNet_results/Dataset100_READPE/LatentMaskTrainer__nnUNetPlans__3d_fullres/fold_all \
    --test_dir $nnUNet_raw/Dataset102_CADPE \
    --output cadpe_external_results.json
```

---

## Table 9: Synthetic Missingness Pattern Ablation (PE, 5-fold CV)

> **Purpose:** Which synthetic patterns matter for PropNet training.

| Synthetic patterns | Dice ↑ | Recall-small ↑ |
|---|---:|---:|
| None (uniform propensity) | XX.X±.XXX | XX.X±.XXX |
| Scale-dependent only | XX.X±.XXX | XX.X±.XXX |
| Boundary erosion only | XX.X±.XXX | XX.X±.XXX |
| Component drop only | XX.X±.XXX | XX.X±.XXX |
| Scale + Boundary | XX.X±.XXX | XX.X±.XXX |
| **All three combined** | **XX.X±.XXX** | **XX.X±.XXX** |

### Launch Commands — Table 9

```bash
for PATTERN in uniform scale_only boundary_only component_only scale_boundary all; do
    for FOLD in 0 1 2 3 4; do
        CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
            -tr LatentMaskTrainer --npz \
            --c synthetic_pattern=$PATTERN
    done
done
```

---

## Table 10: Annotation Budget Study — Dice vs Annotation Cost

> **Purpose:** Show that multi-granularity with PU correction is more cost-efficient than pixel-only.
> **Cost model:** 1 pixel annotation = 10 units; 1 box = 2 units; 1 image label = 0.1 units.

| Budget (units) | Pixel-only Dice | Mixed Naive Dice | LatentMask Dice |
|---:|---:|---:|---:|
| 100 | XX.X | XX.X | XX.X |
| 200 | XX.X | XX.X | XX.X |
| 400 | XX.X | XX.X | XX.X |
| 800 | XX.X | XX.X | XX.X |
| 1600 | XX.X | XX.X | XX.X |

### Launch Commands — Table 10

```bash
for BUDGET in 100 200 400 800 1600; do
    python latentmask/scripts/build_budget_split.py \
        --budget $BUDGET --pixel_cost 10 --box_cost 2 --image_cost 0.1 \
        --output_manifest $nnUNet_raw/manifests/budget_${BUDGET}.json

    for FOLD in 0 1 2; do
        # Pixel-only variant
        CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
            -tr nnUNetTrainer --npz \
            --c budget_manifest=budget_${BUDGET}_pixel_only.json

        # Mixed Naive
        CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
            -tr MixedNaiveTrainer --npz \
            --c budget_manifest=budget_${BUDGET}_mixed.json

        # LatentMask
        CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
            -tr LatentMaskTrainer --npz \
            --c budget_manifest=budget_${BUDGET}_mixed.json
    done
done
```

---

## Table 11: Robustness to Missingness Level (LiTS, 3-fold CV)

> **Purpose:** Show LatentMask degrades gracefully as label completeness decreases.

| Pixel/Box/Image ratio | Pixel-only Dice | Mixed Naive Dice | LatentMask Dice | Δ (ours vs naive) |
|---|---:|---:|---:|---:|
| 50/30/20 (mild) | XX.X | XX.X | XX.X | +PLACEHOLDER |
| 30/30/40 (moderate) | XX.X | XX.X | XX.X | +PLACEHOLDER |
| 15/25/60 (severe) | XX.X | XX.X | XX.X | +PLACEHOLDER |
| 5/15/80 (extreme) | XX.X | XX.X | XX.X | +PLACEHOLDER |

### Launch Commands — Table 11

```bash
for SPLIT in "50_30_20" "30_30_40" "15_25_60" "5_15_80"; do
    IFS='_' read -r P B I <<< "$SPLIT"
    python latentmask/scripts/build_synthetic_multigran.py \
        --dataset lits --pixel_ratio 0.${P} --box_ratio 0.${B} --image_ratio 0.${I}

    for FOLD in 0 1 2; do
        CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
            -tr LatentMaskTrainer --npz \
            --c data_split=multigran_${SPLIT}
    done
done
```

---

## Table 12: Optional — PE-specific Enhancement (PropNet + Vesselness Hint)

> **Purpose:** Show domain-specific priors can optionally boost PropNet on PE (but are NOT required).

| Method | PE Dice ↑ | Recall-small ↑ |
|---|---:|---:|
| PropNet (encoder features only) | XX.X±.XXX | XX.X±.XXX |
| PropNet + vesselness hint | XX.X±.XXX | XX.X±.XXX |

```bash
# PropNet + optional vesselness hint
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz \
        --c use_vesselness_hint=true
done
```

---

## Figure → Table Mapping

| Figure | Depends on | Description |
|---|---|---|
| Figure 1 | Architecture design | Method overview diagram (no data needed) |
| Figure 2 | Table 3 A0 vs A4 | Side-by-side PU correction visualization on box data |
| Figure 3 | Tables 2, 4, 6 | PropNet propensity maps across PE, LiTS, VOC (3×3 grid) |
| Figure 4 | Tables 2, 4, 6 | Qualitative segmentation: ours vs baselines (3 domains) |
| Figure 5 | Table 10 | Annotation budget frontier: Dice vs cost (line plot) |
