#!/usr/bin/env python3
"""
05_compute_metrics.py — Compute all LatentMask metrics on predictions.

Usage:
    python 05_compute_metrics.py --pred_dir <pred_dir> --gt_dir <gt_dir> --output <json>
    python 05_compute_metrics.py --results_root <nnUNet_results> --mode cv    # 5-fold CV
    python 05_compute_metrics.py --results_root <nnUNet_results> --mode ext   # External
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from latentmask.utils.metrics import (
    evaluate_case,
    aggregate_results,
    paired_bootstrap_test,
)


def load_nifti_as_binary(path: str | Path) -> np.ndarray:
    """Load NIfTI and binarize."""
    nii = nib.load(str(path))
    arr = nii.get_fdata()
    return (arr > 0).astype(np.uint8)


def get_spacing(path: str | Path) -> tuple:
    """Get voxel spacing from NIfTI header."""
    nii = nib.load(str(path))
    return tuple(nii.header.get_zooms()[:3])


def eval_directory(pred_dir: str, gt_dir: str, output_path: str | None = None) -> dict:
    """Evaluate all predictions in a directory against ground truth."""
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    print(f"Found {len(pred_files)} predictions in {pred_dir}")

    case_results = []
    for pred_path in pred_files:
        case_id = pred_path.name
        gt_path = gt_dir / case_id

        if not gt_path.exists():
            # Try without _0000 suffix
            alt_name = case_id.replace("_0000.nii.gz", ".nii.gz")
            gt_path = gt_dir / alt_name

        if not gt_path.exists():
            print(f"  WARNING: No GT for {case_id}, skipping")
            continue

        pred = load_nifti_as_binary(pred_path)
        gt = load_nifti_as_binary(gt_path)
        spacing = get_spacing(gt_path)

        result = evaluate_case(pred, gt, spacing=spacing)
        result["case_id"] = case_id
        case_results.append(result)
        print(f"  {case_id}: Dice={result['dice']:.3f}, HD95={result['hd95']:.1f}, "
              f"F1={result['lesion_f1']:.3f}")

    agg = aggregate_results(case_results)
    agg["n_cases"] = len(case_results)

    output = {"aggregate": agg, "per_case": case_results}

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    print(f"\n=== Aggregate ({len(case_results)} cases) ===")
    print(f"  Dice: {agg.get('dice_mean', 0):.3f} ± {agg.get('dice_std', 0):.3f}")
    print(f"  HD95: {agg.get('hd95_mean', 0):.1f} ± {agg.get('hd95_std', 0):.1f}")
    print(f"  Lesion-F1: {agg.get('lesion_f1_mean', 0):.3f} ± {agg.get('lesion_f1_std', 0):.3f}")
    print(f"  Recall-small: {agg.get('lesion_recall_small_mean', 0):.3f} ± {agg.get('lesion_recall_small_std', 0):.3f}")
    print(f"  FP/scan: {agg.get('lesion_fp_per_scan_mean', 0):.2f} ± {agg.get('lesion_fp_per_scan_std', 0):.2f}")

    return output


def eval_5fold_cv(results_root: str, dataset_id: int, trainer: str, config: str = "3d_fullres"):
    """Evaluate 5-fold CV results."""
    results_root = Path(results_root)

    # Find the dataset directory
    dataset_dirs = list(results_root.glob(f"Dataset{dataset_id}_*"))
    if not dataset_dirs:
        print(f"No results found for dataset {dataset_id}")
        return

    dataset_dir = dataset_dirs[0]
    trainer_dir = dataset_dir / f"{trainer}__nnUNetPlans__{config}"

    if not trainer_dir.exists():
        print(f"No trainer directory: {trainer_dir}")
        return

    print(f"=== 5-fold CV evaluation for {trainer} on Dataset{dataset_id} ===")

    all_case_results = []
    for fold in range(5):
        fold_dir = trainer_dir / f"fold_{fold}" / "validation"
        gt_dir = Path(os.environ.get("nnUNet_raw", "")) / dataset_dir.name / "labelsTr"

        if not fold_dir.exists():
            print(f"  Fold {fold}: no validation directory, skipping")
            continue

        print(f"\n--- Fold {fold} ---")
        result = eval_directory(str(fold_dir), str(gt_dir))
        for cr in result["per_case"]:
            cr["fold"] = fold
        all_case_results.extend(result["per_case"])

    if all_case_results:
        agg = aggregate_results(all_case_results)
        print(f"\n=== Overall 5-fold CV ({len(all_case_results)} cases) ===")
        print(f"  Dice: {agg.get('dice_mean', 0):.3f} ± {agg.get('dice_std', 0):.3f}")
        print(f"  HD95: {agg.get('hd95_mean', 0):.1f} ± {agg.get('hd95_std', 0):.1f}")
        print(f"  Lesion-F1: {agg.get('lesion_f1_mean', 0):.3f} ± {agg.get('lesion_f1_std', 0):.3f}")
        print(f"  Recall-small: {agg.get('lesion_recall_small_mean', 0):.3f} ± {agg.get('lesion_recall_small_std', 0):.3f}")

        output_path = trainer_dir / "cv_metrics.json"
        with open(output_path, "w") as f:
            json.dump({"aggregate": agg, "per_case": all_case_results}, f, indent=2, default=str)
        print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LatentMask metrics computation")
    parser.add_argument("--pred_dir", type=str, help="Prediction directory")
    parser.add_argument("--gt_dir", type=str, help="Ground truth directory")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--results_root", type=str, help="nnUNet_results root for CV/external mode")
    parser.add_argument("--dataset_id", type=int, default=100)
    parser.add_argument("--trainer", type=str, default="LatentMaskTrainer")
    parser.add_argument("--config", type=str, default="3d_fullres")
    parser.add_argument("--mode", choices=["single", "cv", "ext"], default="single")

    args = parser.parse_args()

    if args.mode == "single":
        eval_directory(args.pred_dir, args.gt_dir, args.output)
    elif args.mode == "cv":
        eval_5fold_cv(args.results_root or os.environ.get("nnUNet_results", ""),
                      args.dataset_id, args.trainer, args.config)
    elif args.mode == "ext":
        ext_pred_dir = Path(args.results_root or os.environ.get("nnUNet_results", ""))
        ext_pred_dir = ext_pred_dir / "external_eval" / args.trainer / f"dataset{args.dataset_id}"
        gt_dir = Path(os.environ.get("nnUNet_raw", ""))
        gt_dirs = list(gt_dir.glob(f"Dataset{args.dataset_id}_*/labelsTr"))
        if gt_dirs:
            eval_directory(str(ext_pred_dir), str(gt_dirs[0]), args.output)


if __name__ == "__main__":
    main()
