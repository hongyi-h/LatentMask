#!/bin/bash
# LatentMask v5 Experiment Orchestration
# Run milestones sequentially. Check stop/go gates before proceeding.
#
# Usage:
#   bash latentmask/scripts/run_experiments_v5.sh [milestone]
#   e.g. bash latentmask/scripts/run_experiments_v5.sh M0
#        bash latentmask/scripts/run_experiments_v5.sh M1
#        bash latentmask/scripts/run_experiments_v5.sh M2
#        bash latentmask/scripts/run_experiments_v5.sh ALL

set -euo pipefail

DATASET_ID=501
FOLD=0
STEEPNESS=medium
RESULTS_DIR=results/v5

MILESTONE=${1:-M0}

echo "================================================"
echo "LatentMask v5 Experiments — Milestone: $MILESTONE"
echo "================================================"

# ── M0: Sanity Checks ──────────────────────────────────────────────────
run_m0() {
    echo "=== M0: Sanity Checks ==="

    # R001: Verify data conversion (should already be done)
    echo "[R001] Checking nnUNet preprocessed data..."
    if [ ! -d "${nnUNet_preprocessed}/Dataset${DATASET_ID}_LiTS" ]; then
        echo "  Data not found. Running conversion + preprocessing..."
        python -m latentmask.scripts.convert_lits_to_nnunet
        nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity
    else
        echo "  Preprocessed data exists. OK."
    fi

    # R002: Metric sanity check (short run to verify loss computation)
    echo "[R002] Metric sanity — 5-epoch C2 uniform test..."
    python -m latentmask.scripts.launch_training \
        --dataset_id $DATASET_ID --fold $FOLD --neg_mode uniform \
        --steepness $STEEPNESS --num_epochs 5 --seed 42

    # R003: 1-scan overfit test (C4 channel on 1 scan, 50 epochs)
    echo "[R003] Overfit test — 50-epoch C4 on 1 scan..."
    python -m latentmask.scripts.launch_training \
        --dataset_id $DATASET_ID --fold $FOLD --neg_mode channel \
        --steepness $STEEPNESS --num_epochs 50 --pixel_fraction 0.95 \
        --warmup_epochs 5 --ramp_epochs 5 --channel_neg_start 8 --seed 42

    echo "=== M0 Complete. Check outputs before proceeding to M1. ==="
}

# ── M1: Calibration ────────────────────────────────────────────────────
run_m1() {
    echo "=== M1: Calibration (g_theta fitting + 3-fold CV) ==="
    mkdir -p $RESULTS_DIR/m1_calibration

    # R004: Fit g_theta with Hungarian matching
    echo "[R004] Calibration — steepness=$STEEPNESS..."
    python -m latentmask.scripts.run_calibration_cv \
        --dataset_id $DATASET_ID --fold $FOLD --steepness $STEEPNESS \
        --n_folds 3 --output $RESULTS_DIR/m1_calibration/ --seed 42

    # R005: Test with different steepnesses
    for steep in shallow medium steep; do
        echo "[R005] Calibration — steepness=$steep..."
        python -m latentmask.scripts.run_calibration_cv \
            --dataset_id $DATASET_ID --fold $FOLD --steepness $steep \
            --n_folds 3 --output $RESULTS_DIR/m1_calibration/ --seed 42
    done

    echo "=== M1 Complete. Gate G1: ECE < 0.10? ==="
    echo "Check results in $RESULTS_DIR/m1_calibration/"
}

# ── M2: Baselines ──────────────────────────────────────────────────────
run_m2() {
    echo "=== M2: Baselines (C1 pixel-only + C2 uniform) ==="

    # R006-R008: C1 pixel-only × 3 seeds
    for seed in 42 123 456; do
        echo "[R006-R008] C1 none seed=$seed..."
        python -m latentmask.scripts.launch_training \
            --dataset_id $DATASET_ID --fold $FOLD --neg_mode none \
            --steepness $STEEPNESS --seed $seed
    done

    # R009-R011: C2 uniform × 3 seeds
    for seed in 42 123 456; do
        echo "[R009-R011] C2 uniform seed=$seed..."
        python -m latentmask.scripts.launch_training \
            --dataset_id $DATASET_ID --fold $FOLD --neg_mode uniform \
            --steepness $STEEPNESS --seed $seed
    done

    echo "=== M2 Complete. Gate G2: C2 Dice > C1 Dice? ==="
}

# ── M3: Main Method ────────────────────────────────────────────────────
run_m3() {
    echo "=== M3: Main Method (C3 linear + C4 channel) ==="

    # R012-R014: C3 linear × 3 seeds
    for seed in 42 123 456; do
        echo "[R012-R014] C3 linear seed=$seed..."
        python -m latentmask.scripts.launch_training \
            --dataset_id $DATASET_ID --fold $FOLD --neg_mode linear \
            --steepness $STEEPNESS --seed $seed
    done

    # R015-R017: C4 channel × 3 seeds
    for seed in 42 123 456; do
        echo "[R015-R017] C4 channel seed=$seed..."
        python -m latentmask.scripts.launch_training \
            --dataset_id $DATASET_ID --fold $FOLD --neg_mode channel \
            --steepness $STEEPNESS --seed $seed
    done

    echo "=== M3 Complete. Gate G3: C4 Q1-Dice > C2 Q1-Dice? ==="
}

# ── M4: Evaluation ─────────────────────────────────────────────────────
run_m4() {
    echo "=== M4: Full Evaluation (all configs) ==="
    mkdir -p $RESULTS_DIR/m4_eval

    BASE_DIR="${nnUNet_results}/Dataset${DATASET_ID}_LiTS/LatentMaskTrainer__nnUNetPlans__3d_fullres/fold_${FOLD}"
    GT_DIR="${nnUNet_raw}/Dataset${DATASET_ID}_LiTS/labelsTs"

    for mode in none uniform linear channel; do
        for seed in 42 123 456; do
            RUN_DIR="${BASE_DIR}/${mode}_${STEEPNESS}_seed${seed}"
            PRED_DIR="${RUN_DIR}/validation"  # nnUNet validation predictions
            if [ -d "$PRED_DIR" ]; then
                echo "[M4] Evaluating ${mode}_seed${seed}..."
                python -m latentmask.scripts.evaluate \
                    --pred_dir $PRED_DIR --gt_dir $GT_DIR \
                    --output $RESULTS_DIR/m4_eval/${mode}_seed${seed}.json \
                    --fg_label 2
            else
                echo "  Skipping ${mode}_seed${seed}: no predictions found"
            fi
        done
    done

    echo "=== M4 Complete ==="
}

# ── M5: Sensitivity Analysis ───────────────────────────────────────────
run_m5() {
    echo "=== M5: Sensitivity Analysis ==="

    # R020-R022: d_safe sweep {3, 5, 7}
    for d in 3 5 7; do
        echo "[R020-R022] d_safe=$d..."
        python -m latentmask.scripts.launch_training \
            --dataset_id $DATASET_ID --fold $FOLD --neg_mode channel \
            --steepness $STEEPNESS --d_margin $d --seed 42
    done

    # R023-R024: tau_low sweep {0.2, 0.4}
    for tau in 0.2 0.4; do
        echo "[R023-R024] tau_low=$tau..."
        python -m latentmask.scripts.launch_training \
            --dataset_id $DATASET_ID --fold $FOLD --neg_mode channel \
            --steepness $STEEPNESS --tau_low $tau --seed 42
    done

    # R025-R026: alpha_min sweep {0.01, 0.10}
    for amin in 0.01 0.10; do
        echo "[R025-R026] alpha_min=$amin..."
        python -m latentmask.scripts.launch_training \
            --dataset_id $DATASET_ID --fold $FOLD --neg_mode channel \
            --steepness $STEEPNESS --alpha_min $amin --seed 42
    done

    echo "=== M5 Complete ==="
}

# ── Dispatch ────────────────────────────────────────────────────────────
case $MILESTONE in
    M0) run_m0 ;;
    M1) run_m1 ;;
    M2) run_m2 ;;
    M3) run_m3 ;;
    M4) run_m4 ;;
    M5) run_m5 ;;
    ALL)
        run_m0
        run_m1
        run_m2
        run_m3
        run_m4
        run_m5
        ;;
    *)
        echo "Unknown milestone: $MILESTONE"
        echo "Usage: $0 {M0|M1|M2|M3|M4|M5|ALL}"
        exit 1
        ;;
esac
#!/bin/bash
# LatentMask v5 Experiment Commands
# 4-Config Comparison: C1–C4 with CC-based negative supervision
#
# Prerequisites:
#   1. pip install -e nnUNet/
#   2. pip install -e latentmask/
#   3. Set environment variables:
#      export nnUNet_raw=/path/to/nnUNet_raw
#      export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
#      export nnUNet_results=/path/to/nnUNet_results
#   4. Convert and preprocess LiTS data (M0)
#
# Config mapping:
#   C1: ipw_mode=none,    neg_mode=none     (pixel-only lower bound)
#   C2: ipw_mode=channel, neg_mode=uniform  (scaffold + α≡1.0)
#   C3: ipw_mode=channel, neg_mode=linear   (scaffold + α=linear(log m))
#   C4: ipw_mode=channel, neg_mode=channel  (scaffold + α=g_θ(log m))

set -e
SEEDS="42 123 456"
LAUNCH="python -m latentmask.scripts.launch_training"

# ═══════════════════════════════════════════════════════════════════════
# M0: Sanity & Infrastructure
# ═══════════════════════════════════════════════════════════════════════

run_m0_convert() {
    echo "=== M0: Convert LiTS to nnUNet format ==="
    python -m latentmask.scripts.convert_lits_to_nnunet \
        --input_dir data/LiTS_Tiny \
        --output_dir "$nnUNet_raw/Dataset501_LiTS"
}

run_m0_preprocess() {
    echo "=== M0: Plan and preprocess ==="
    nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
}

run_m0_sanity() {
    echo "=== M0-R001: nnUNet overfit check (50 epochs) ==="
    nnUNetv2_train 501 3d_fullres 0 --npz -num_epochs 50

    echo "=== M0-R002: Box-IPW sanity (50 epochs) ==="
    $LAUNCH --dataset_id 501 --fold 0 --ipw_mode channel --neg_mode channel \
        --num_epochs 50 --warmup_epochs 10 --ramp_epochs 10 --seed 42

    echo "=== M0-R003: Metric unit tests ==="
    python -c "
from latentmask.utils.metrics import compute_dice, compute_hd95, compute_size_stratified_metrics
import numpy as np
a = np.zeros((64,64,64), dtype=np.int32); a[10:20,10:20,10:20] = 1
b = np.zeros((64,64,64), dtype=np.int32); b[12:22,12:22,12:22] = 1
print(f'Dice={compute_dice(a,b):.4f}  HD95={compute_hd95(a,b):.2f}')
r = compute_size_stratified_metrics(a,b)
print(f'Overall Dice={r[\"overall_dice\"]:.4f}  Q1 Dice={r[\"per_quintile\"][0][\"dice\"]:.4f}')
print('Metrics OK')
"
}

# ═══════════════════════════════════════════════════════════════════════
# M1: Calibration (CPU-only)
# ═══════════════════════════════════════════════════════════════════════

run_m1_calibration() {
    echo "=== M1: 3-fold cross-validated calibration ==="
    python -m latentmask.scripts.run_calibration_cv \
        --preprocessed_dir "$nnUNet_preprocessed/Dataset501_LiTS/nnUNetPlans_3d_fullres" \
        --output_dir results/m1_calibration_v5
    echo "CHECK: ECE < 0.10 on all folds. g_θ curve is monotone."
}

# ═══════════════════════════════════════════════════════════════════════
# M2: Baselines — C1 (pixel-only) and C2 (scaffold + uniform neg)
# ═══════════════════════════════════════════════════════════════════════

run_m2_c1() {
    echo "=== M2: C1 pixel-only baseline (3 seeds) ==="
    for SEED in $SEEDS; do
        echo "--- C1 seed=$SEED ---"
        $LAUNCH --dataset_id 501 --fold 0 \
            --ipw_mode none --neg_mode none \
            --seed $SEED
    done
}

run_m2_c2() {
    echo "=== M2: C2 scaffold + uniform neg (3 seeds) ==="
    for SEED in $SEEDS; do
        echo "--- C2 seed=$SEED ---"
        $LAUNCH --dataset_id 501 --fold 0 \
            --ipw_mode channel --neg_mode uniform \
            --steepness medium --seed $SEED
    done
}

# ═══════════════════════════════════════════════════════════════════════
# M3: Main Method — C3 (linear neg) and C4 (g_θ channel neg)
# ═══════════════════════════════════════════════════════════════════════

run_m3_c3() {
    echo "=== M3: C3 scaffold + linear neg (3 seeds) ==="
    for SEED in $SEEDS; do
        echo "--- C3 seed=$SEED ---"
        $LAUNCH --dataset_id 501 --fold 0 \
            --ipw_mode channel --neg_mode linear \
            --steepness medium --seed $SEED
    done
}

run_m3_c4() {
    echo "=== M3: C4 scaffold + g_θ channel neg (3 seeds) ==="
    for SEED in $SEEDS; do
        echo "--- C4 seed=$SEED ---"
        $LAUNCH --dataset_id 501 --fold 0 \
            --ipw_mode channel --neg_mode channel \
            --steepness medium --seed $SEED
    done
}

# ═══════════════════════════════════════════════════════════════════════
# M4: Evaluation — Run predictions and evaluate all configs
# ═══════════════════════════════════════════════════════════════════════

run_m4_evaluate() {
    echo "=== M4: Evaluate all trained models ==="
    RESULTS_BASE="$nnUNet_results/Dataset501_LiTS/LatentMaskTrainer__nnUNetPlans__3d_fullres/fold_0"
    GT_DIR="$nnUNet_raw/Dataset501_LiTS/labelsTs"

    for CONFIG in none_none channel_uniform channel_linear channel_channel; do
        IPW=$(echo $CONFIG | cut -d_ -f1)
        NEG=$(echo $CONFIG | cut -d_ -f2)
        for SEED in $SEEDS; do
            SUFFIX="${IPW}_medium_seed${SEED}"
            PRED_DIR="${RESULTS_BASE}/${SUFFIX}/predictions"
            OUT="results/m4_eval/${CONFIG}_seed${SEED}.json"

            if [ -d "$PRED_DIR" ]; then
                echo "--- Evaluating $CONFIG seed=$SEED ---"
                python -m latentmask.scripts.evaluate \
                    --pred_dir "$PRED_DIR" \
                    --gt_dir "$GT_DIR" \
                    --output "$OUT"
            else
                echo "  SKIP: $PRED_DIR not found"
            fi
        done
    done
}

# ═══════════════════════════════════════════════════════════════════════
# M5: Sensitivity Analysis (C4 variants, 1 seed each)
# ═══════════════════════════════════════════════════════════════════════

run_m5_dsafe() {
    echo "=== M5: d_safe sensitivity ==="
    # d_safe=3 (default is 5)
    $LAUNCH --dataset_id 501 --fold 0 \
        --ipw_mode channel --neg_mode channel \
        --d_margin 3 --steepness medium --seed 42

    # Also C2 at d_safe=3 to test AC1
    $LAUNCH --dataset_id 501 --fold 0 \
        --ipw_mode channel --neg_mode uniform \
        --d_margin 3 --steepness medium --seed 42
}

run_m5_tau_low() {
    echo "=== M5: tau_low sensitivity ==="
    for TAU in 0.2 0.4; do
        $LAUNCH --dataset_id 501 --fold 0 \
            --ipw_mode channel --neg_mode channel \
            --tau_low $TAU --steepness medium --seed 42
    done
}

run_m5_alpha_min() {
    echo "=== M5: alpha_min sensitivity ==="
    for AMIN in 0.02 0.10; do
        $LAUNCH --dataset_id 501 --fold 0 \
            --ipw_mode channel --neg_mode channel \
            --alpha_min $AMIN --steepness medium --seed 42
    done
}

# ═══════════════════════════════════════════════════════════════════════
# Combined Execution Targets
# ═══════════════════════════════════════════════════════════════════════

run_all_m2() { run_m2_c1; run_m2_c2; }
run_all_m3() { run_m3_c3; run_m3_c4; }
run_all_m5() { run_m5_dsafe; run_m5_tau_low; run_m5_alpha_min; }

run_full_pipeline() {
    run_m0_sanity
    run_m1_calibration
    run_all_m2
    run_all_m3
    run_m4_evaluate
    run_all_m5
}

# ═══════════════════════════════════════════════════════════════════════
# Usage: source this script, then call functions:
#   source latentmask/scripts/run_experiments_v5.sh
#   run_m0_sanity
#   run_m1_calibration
#   run_m2_c1
#   ...
# Or run the full pipeline:
#   run_full_pipeline
# ═══════════════════════════════════════════════════════════════════════

echo "v5 experiment script loaded. Available functions:"
echo "  run_m0_convert, run_m0_preprocess, run_m0_sanity"
echo "  run_m1_calibration"
echo "  run_m2_c1, run_m2_c2, run_all_m2"
echo "  run_m3_c3, run_m3_c4, run_all_m3"
echo "  run_m4_evaluate"
echo "  run_m5_dsafe, run_m5_tau_low, run_m5_alpha_min, run_all_m5"
echo "  run_full_pipeline"
