#!/bin/bash
# LatentMask Experiment Commands
# Run these on a machine with GPU and nnUNet installed.
#
# Prerequisites:
#   1. pip install -e nnUNet/
#   2. pip install -e latentmask/
#   3. Set environment variables:
#      export nnUNet_raw=/path/to/nnUNet_raw
#      export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
#      export nnUNet_results=/path/to/nnUNet_results
#   4. Convert and preprocess LiTS data (see M0 below)
#
# Result files are saved in each run's output folder under nnUNet_results.
# Additional JSON summaries go to results/ directory.

set -e

# ═══════════════════════════════════════════════════════════════════════
# M0: Sanity & Infrastructure
# ═══════════════════════════════════════════════════════════════════════

# Step 0: Convert LiTS to nnUNet format
# python -m latentmask.scripts.convert_lits_to_nnunet \
#     --input_dir data/LiTS_Tiny \
#     --output_dir $nnUNet_raw/Dataset501_LiTS

# Step 1: Plan and preprocess
# nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity

# R001: nnUNet overfit (standard, no custom code)
# nnUNetv2_train 501 3d_fullres 0 --npz -num_epochs 50

# R002: Single box-IPW sanity run
# python -m latentmask.scripts.launch_training \
#     --dataset_id 501 --fold 0 --ipw_mode channel \
#     --num_epochs 50 --warmup_epochs 10 --ramp_epochs 10 --seed 42

# R003: Metric unit tests (no GPU needed)
# python -m pytest latentmask/tests/test_metrics.py -v   # if tests exist
# or: python -c "from latentmask.utils.metrics import *; print('OK')"

# ═══════════════════════════════════════════════════════════════════════
# M1: Calibration (CPU-only, run in parallel with M0)
# ═══════════════════════════════════════════════════════════════════════

# R004-R024: Cross-validated calibration
# python -m latentmask.scripts.run_calibration_cv \
#     --preprocessed_dir $nnUNet_preprocessed/Dataset501_LiTS/nnUNetPlans_3d_fullres \
#     --output_dir results/m1_calibration
#
# Fallback (no preprocessed data, use raw segmentations):
# python -m latentmask.scripts.run_calibration_cv \
#     --seg_dir "data/LiTS_Tiny/Training Batch" \
#     --output_dir results/m1_calibration

# ═══════════════════════════════════════════════════════════════════════
# M2: Baselines
# ═══════════════════════════════════════════════════════════════════════

# R025-R027: No-box (pixel-only), 3 seeds
run_nobox() {
    for SEED in 42 123 456; do
        python -m latentmask.scripts.launch_training \
            --dataset_id 501 --fold 0 --ipw_mode none \
            --seed $SEED
    done
}

# R028-R030: Uniform PU (w=1), 3 seeds
run_uniform() {
    for SEED in 42 123 456; do
        python -m latentmask.scripts.launch_training \
            --dataset_id 501 --fold 0 --ipw_mode uniform \
            --steepness medium --seed $SEED
    done
}

# R031: Oracle IPW, 1 seed
run_oracle_single() {
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --ipw_mode oracle \
        --steepness medium --seed 42
}

# ═══════════════════════════════════════════════════════════════════════
# M3: Main Method
# ═══════════════════════════════════════════════════════════════════════

# R032-R034: Ours, medium steepness, 3 seeds
run_ours_medium() {
    for SEED in 42 123 456; do
        python -m latentmask.scripts.launch_training \
            --dataset_id 501 --fold 0 --ipw_mode channel \
            --steepness medium --seed $SEED
    done
}

# R035: Ours, shallow, 1 seed
run_ours_shallow() {
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --ipw_mode channel \
        --steepness shallow --seed 42
}

# R036: Ours, steep, 1 seed
run_ours_steep() {
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --ipw_mode channel \
        --steepness steep --seed 42
}

# R037-R040: Uniform + Oracle, shallow + steep
run_baselines_otherstep() {
    for STEEPNESS in shallow steep; do
        python -m latentmask.scripts.launch_training \
            --dataset_id 501 --fold 0 --ipw_mode uniform \
            --steepness $STEEPNESS --seed 42
        python -m latentmask.scripts.launch_training \
            --dataset_id 501 --fold 0 --ipw_mode oracle \
            --steepness $STEEPNESS --seed 42
    done
}

# R041-R042: Oracle, medium, +2 seeds
run_oracle_medium_extra() {
    for SEED in 123 456; do
        python -m latentmask.scripts.launch_training \
            --dataset_id 501 --fold 0 --ipw_mode oracle \
            --steepness medium --seed $SEED
    done
}

# ═══════════════════════════════════════════════════════════════════════
# M4: Ablations
# ═══════════════════════════════════════════════════════════════════════

# R043-R044: π̂ sensitivity
run_pi_sensitivity() {
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --ipw_mode channel \
        --pi_hat_scale 0.8 --seed 42
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --ipw_mode channel \
        --pi_hat_scale 1.2 --seed 42
}

# R045-R047: w_max sweep
run_wmax_sweep() {
    for WMAX in 3 5 20; do
        python -m latentmask.scripts.launch_training \
            --dataset_id 501 --fold 0 --ipw_mode channel \
            --w_max $WMAX --seed 42
    done
}

# R048: No warm-up
run_no_warmup() {
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --ipw_mode channel \
        --warmup_epochs 0 --seed 42
}

# ═══════════════════════════════════════════════════════════════════════
# Usage: source this file then call the function you need
# Example:
#   source latentmask/scripts/run_experiments.sh
#   run_ours_medium
# ═══════════════════════════════════════════════════════════════════════
echo "Experiment functions loaded. Call the function you need."
echo "Available: run_nobox, run_uniform, run_oracle_single, run_ours_medium,"
echo "  run_ours_shallow, run_ours_steep, run_baselines_otherstep,"
echo "  run_oracle_medium_extra, run_pi_sensitivity, run_wmax_sweep, run_no_warmup"
