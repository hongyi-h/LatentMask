# LatentMask v2 — Experiment → Result Replacement Checklist

> **How to use:** Run each launch command block, collect its metrics, then fill in every
> placeholder listed under "Fill locations". Tick the checkbox when done.

---

## Notation

- **02** = `02_experimental_tables.md`
- **03** = `03_paper_en.md`
- **04** = `04_paper_zh.md`
- **T2-R3** = Table 2, Row 3
- **§5.3¶1** = Section 5.3, paragraph 1

---

## 1. PE Main Comparison (Table 2) — 7 methods × 5 folds

Each row produces: Dice, HD95, Lesion-F1, Recall-small, FP/scan (mean±std over 5 folds).

### 1.1 nnUNet (pixel-only)

```bash
unset LATENTMASK_RSPECT_MANIFEST LATENTMASK_AUG_RSPECT_MANIFEST
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr nnUNetTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer nnUNetTrainer --mode cv
```

- [ ] **Fill:** 02 → T2-R1 (5 cells)

### 1.2 Mean Teacher

```bash
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
unset LATENTMASK_AUG_RSPECT_MANIFEST
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr MeanTeacher3DTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer MeanTeacher3DTrainer --mode cv
```

- [ ] **Fill:** 02 → T2-R2 (5 cells)

### 1.3 CPS

```bash
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr CPSTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer CPSTrainer --mode cv
```

- [ ] **Fill:** 02 → T2-R3 (5 cells)

### 1.4 nnPU-Seg (uniform)

```bash
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
unset LATENTMASK_AUG_RSPECT_MANIFEST
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr nnPUSegTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer nnPUSegTrainer --mode cv
```

- [ ] **Fill:** 02 → T2-R4 (5 cells)

### 1.5 3D BoxSup

```bash
unset LATENTMASK_RSPECT_MANIFEST
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr BoxSup3DTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer BoxSup3DTrainer --mode cv
```

- [ ] **Fill:** 02 → T2-R5 (5 cells)

### 1.6 Mixed Naive

```bash
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr MixedNaiveTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer MixedNaiveTrainer --mode cv
```

- [ ] **Fill:** 02 → T2-R6 (5 cells)
- [ ] **Fill:** 02 → T3-R1 (A0: same as T2-R6, Dice/HD95/Recall-small)

### 1.7 LatentMask (ours) ★

```bash
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz
done
python latentmask/scripts/eval_metrics.py \
    --dataset_id 100 --trainer LatentMaskTrainer --mode cv
```

- [ ] **Fill:** 02 → T2-R7 (5 cells)
- [ ] **Fill:** 02 → T3-R5 (A4: same as T2-R7, Dice/HD95/Recall-small)
- [ ] **Fill:** 03 → Abstract ¶1 — "XX.X Dice" (PE headline number)
- [ ] **Fill:** 03 → §1 conclusion ¶ — recompute "+PLACEHOLDER" deltas (ours − Mixed Naive, ours − nnPU-Seg)
- [ ] **Fill:** 03 → §5.3 ¶1 — "XX.X Dice", "+PLACEHOLDER" (all 4 deltas vs baselines)
- [ ] **Fill:** 04 → Abstract — "XX.X Dice"
- [ ] **Fill:** 04 → §1 结论段 — "+PLACEHOLDER分"
- [ ] **Fill:** 04 → §5.3 ¶1 — PE结果段

**Derived deltas (compute after 1.6 + 1.7):**
- ours − Mixed Naive → "+PLACEHOLDER" in 03 Abstract, §1, §5.3; 04 Abstract, §5.3
- ours − nnPU-Seg → "+PLACEHOLDER" in 03 Abstract, §1, §5.3; 04 Abstract, §5.3
- ours − Mixed Naive (Recall-small) → "+PLACEHOLDER" in 03 §5.3; 04 §5.3

---

## 2. PE Ablation (Table 3) — 5 variants × 5 folds

### 2.1 A1: PU with uniform propensity

```bash
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz \
        --c propnet_mode=uniform,use_smoothness=false,use_ema_refinement=false
done
```

- [ ] **Fill:** 02 → T3-R2 (3 cells)

### 2.2 A2: PropNet, no smooth, no EMA

```bash
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz \
        --c propnet_mode=learned,use_smoothness=false,use_ema_refinement=false
done
```

- [ ] **Fill:** 02 → T3-R3 (3 cells)

### 2.3 A3: PropNet + smoothness, no EMA

```bash
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz \
        --c propnet_mode=learned,use_smoothness=true,use_ema_refinement=false
done
```

- [ ] **Fill:** 02 → T3-R4 (3 cells)

**Derived deltas (compute after 2.1–2.3 + 1.6 + 1.7):**
- A0→A1 delta → "+PLACEHOLDER" in 03 §5.4 ¶1; 04 §5.4 (A0→A1)
- A1→A2 delta → "+PLACEHOLDER" in 03 §5.4 ¶1; 04 §5.4 (A1→A2)
- A2→A3 delta → "+PLACEHOLDER" in 03 §5.4 ¶1; 04 §5.4 (A2→A3)
- A3→A4 delta → "+PLACEHOLDER" in 03 §5.4 ¶1; 04 §5.4 (A3→A4)

---

## 3. LiTS Main Comparison (Table 4) — 6 methods × 3 folds

### 3.1 Oracle (100% pixel)

```bash
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr nnUNetTrainer --npz --c data_split=oracle
done
```

- [ ] **Fill:** 02 → T4-R1 (4 cells)

### 3.2 nnUNet (30% pixel)

```bash
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr nnUNetTrainer --npz --c data_split=pixel_only
done
```

- [ ] **Fill:** 02 → T4-R2 (4 cells)

### 3.3 Mean Teacher

```bash
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr MeanTeacher3DTrainer --npz --c data_split=semisup
done
```

- [ ] **Fill:** 02 → T4-R3 (4 cells)

### 3.4 nnPU-Seg

```bash
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr nnPUSegTrainer --npz --c data_split=pu
done
```

- [ ] **Fill:** 02 → T4-R4 (4 cells)

### 3.5 Mixed Naive

```bash
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr MixedNaiveTrainer --npz --c data_split=multigran
done
```

- [ ] **Fill:** 02 → T4-R5 (4 cells)
- [ ] **Fill:** 02 → T5-R1 (A0: same as T4-R5, Dice/HD95/Recall-small)

### 3.6 LatentMask (ours) ★

```bash
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz --c data_split=multigran
done
```

- [ ] **Fill:** 02 → T4-R6 (4 cells)
- [ ] **Fill:** 02 → T5-R5 (A4: same as T4-R6, Dice/HD95/Recall-small)
- [ ] **Fill:** 03 → Abstract — "XX.X%" recovery ratio (ours Dice / oracle Dice × 100)
- [ ] **Fill:** 03 → §5.3 ¶2 — "XX.X Dice", "XX.X" oracle, "XX.X%" recovery, "+PLACEHOLDER" deltas
- [ ] **Fill:** 04 → Abstract — "XX.X%"
- [ ] **Fill:** 04 → §5.3 ¶2 — LiTS结果段 (Dice, Oracle, recovery%, deltas)

---

## 4. LiTS Ablation (Table 5) — 5 variants × 3 folds

Same commands as §2 but `dataset_id=200`, 3 folds, `--c data_split=multigran`.

```bash
# A1
for FOLD in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz \
        --c data_split=multigran,propnet_mode=uniform,use_smoothness=false,use_ema_refinement=false
done
# A2: propnet_mode=learned,use_smoothness=false,use_ema_refinement=false
# A3: propnet_mode=learned,use_smoothness=true,use_ema_refinement=false
```

- [ ] **Fill:** 02 → T5-R2 (A1), T5-R3 (A2), T5-R4 (A3) — 3 cells each

---

## 5. PASCAL VOC Main Comparison (Table 6) — 7 methods × 3 seeds

Each row produces: mIoU, Small-obj IoU (mean±std over 3 seeds).

```bash
# Example for LatentMask:
for SEED in 42 43 44; do
    python latentmask/train_voc.py \
        --config configs/voc/latentmask.yaml --seed $SEED --gpus 1
done
```

See 02 Table 6 Launch Commands for all 7 methods.

- [ ] **Fill:** 02 → T6 — all 7 rows × 2 cells
- [ ] **Fill:** 03 → §5.3 ¶3 — "XX.X mIoU", "+PLACEHOLDER" deltas (4 baselines)
- [ ] **Fill:** 04 → §5.3 ¶3 — VOC结果段

---

## 6. VOC Ablation (Table 7) — 5 variants × 3 seeds

Same ablation flags, using `train_voc.py` configs.

- [ ] **Fill:** 02 → T7 — 5 rows × 2 cells

---

## 7. PE External Evaluation (Table 8) — 4 methods

```bash
# Train on all READ-PE + Aug-RSPECT + RSPECT
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres all \
    -tr LatentMaskTrainer --npz

# Evaluate on FUMPE & CAD-PE
python latentmask/scripts/eval_external.py \
    --model_dir $nnUNet_results/Dataset100_READPE/LatentMaskTrainer__nnUNetPlans__3d_fullres/fold_all \
    --test_dir $nnUNet_raw/Dataset101_FUMPE --output fumpe_results.json

python latentmask/scripts/eval_external.py \
    --model_dir $nnUNet_results/Dataset100_READPE/LatentMaskTrainer__nnUNetPlans__3d_fullres/fold_all \
    --test_dir $nnUNet_raw/Dataset102_CADPE --output cadpe_results.json
```

Repeat for nnUNetTrainer, MixedNaiveTrainer, nnPUSegTrainer.

- [ ] **Fill:** 02 → T8 — 4 rows × 4 cells
- [ ] **Fill:** 03 → §5.5 — "XX.X Dice" (FUMPE, CAD-PE), "+PLACEHOLDER" deltas
- [ ] **Fill:** 04 → §5.5 — 外部评估段 (XX.X Dice + deltas)

---

## 8. Synthetic Missingness Pattern Ablation (Table 9) — 6 patterns × 5 folds

```bash
for PATTERN in uniform scale_only boundary_only component_only scale_boundary all; do
    for FOLD in 0 1 2 3 4; do
        CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
            -tr LatentMaskTrainer --npz \
            --c synthetic_pattern=$PATTERN
    done
done
```

- [ ] **Fill:** 02 → T9 — 6 rows × 2 cells
- [ ] **Fill:** 03 → §5.4 ¶2 — missingness pattern deltas (scale: +PLACEHOLDER, component: +PLACEHOLDER, boundary: +PLACEHOLDER, combined: +PLACEHOLDER)
- [ ] **Fill:** 04 → §5.4 — 合成缺失模式段

---

## 9. Annotation Budget Study (Table 10) — 5 budgets × 3 methods × 3 folds

```bash
for BUDGET in 100 200 400 800 1600; do
    python latentmask/scripts/build_budget_split.py \
        --budget $BUDGET --pixel_cost 10 --box_cost 2 --image_cost 0.1 \
        --output_manifest $nnUNet_raw/manifests/budget_${BUDGET}.json
    for FOLD in 0 1 2; do
        # nnUNetTrainer, MixedNaiveTrainer, LatentMaskTrainer
        CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
            -tr <TRAINER> --npz \
            --c budget_manifest=budget_${BUDGET}_<variant>.json
    done
done
```

- [ ] **Fill:** 02 → T10 — 5 rows × 3 cells
- [ ] **Plot:** Figure 5 (Dice vs budget line chart)

---

## 10. Robustness to Missingness Level (Table 11) — 4 ratios × 3 methods × 3 folds

```bash
for SPLIT in "50_30_20" "30_30_40" "15_25_60" "5_15_80"; do
    IFS='_' read -r P B I <<< "$SPLIT"
    python latentmask/scripts/build_synthetic_multigran.py \
        --dataset lits --pixel_ratio 0.${P} --box_ratio 0.${B} --image_ratio 0.${I}
    for FOLD in 0 1 2; do
        CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres $FOLD \
            -tr LatentMaskTrainer --npz --c data_split=multigran_${SPLIT}
    done
done
```

Repeat for nnUNetTrainer (pixel-only) and MixedNaiveTrainer.

- [ ] **Fill:** 02 → T11 — 4 rows × 4 cells (Δ column = ours − naive)
- [ ] **Fill:** 03 → §5.7 — "+PLACEHOLDER" mild delta, "+PLACEHOLDER" extreme delta
- [ ] **Fill:** 04 → §5.7 — 鲁棒性段

---

## 11. Optional: PropNet + Vesselness Hint (Table 12) — 2 configs × 5 folds

```bash
# Already have PropNet-only from Table 2 Row 7.
# Add vesselness hint:
for FOLD in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres $FOLD \
        -tr LatentMaskTrainer --npz --c use_vesselness_hint=true
done
```

- [ ] **Fill:** 02 → T12 — 2 rows × 2 cells

---

## Quick Reference: Inline Placeholders in Papers

| Placeholder in 03/04 | Source Experiment | Computation |
|---|---|---|
| Abstract "XX.X Dice" (PE) | §1.7 (T2-R7 Dice) | Direct |
| Abstract "+PLACEHOLDER" (vs Naive) | §1.7 − §1.6 | T2-R7 Dice − T2-R6 Dice |
| Abstract "+PLACEHOLDER" (vs nnPU) | §1.7 − §1.4 | T2-R7 Dice − T2-R4 Dice |
| Abstract "XX.X%" (LiTS recovery) | §3.6 / §3.1 | T4-R6 Dice / T4-R1 Dice × 100 |
| Abstract "XX.X mIoU" (VOC) | §5 (T6-R7 mIoU) | Direct |
| Abstract "+PLACEHOLDER" (VOC vs Naive) | §5 | T6-R7 − T6-R6 |
| §1 "+PLACEHOLDER分" (cross-bench) | §2 ablations | min/max of (A4−A1) across PE/LiTS/VOC |
| §5.3 all PE numbers | §1.7, §1.6, §1.4, §1.5 | Direct + pairwise deltas |
| §5.3 all LiTS numbers | §3.6, §3.5, §3.4, §3.1 | Direct + recovery ratio |
| §5.3 all VOC numbers | §5 (T6 rows) | Direct + pairwise deltas |
| §5.4 A0→A1→…→A4 | §2 (T3), §4 (T5), §6 (T7) | Pairwise step deltas |
| §5.4 missingness patterns | §8 (T9) | "all" − "uniform", per-pattern deltas |
| §5.5 external eval | §7 (T8) | Direct + ours − naive deltas |
| §5.7 robustness deltas | §10 (T11) | Δ column: mild vs extreme |

---

## Execution Priority

1. **PE main** (Table 2) + **PE ablation** (Table 3) — fills paper's core claim
2. **LiTS main** (Table 4) + **LiTS ablation** (Table 5) — fills generality claim
3. **VOC main** (Table 6) + **VOC ablation** (Table 7) — fills cross-domain claim
4. **PE external** (Table 8) — fills generalization claim
5. **Missingness patterns** (Table 9) + **Budget** (Table 10) + **Robustness** (Table 11) — fills secondary analyses
6. **Vesselness hint** (Table 12) — optional, PE-specific
