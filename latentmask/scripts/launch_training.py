"""Launch LatentMask training experiments.

Usage:
    # R001: Standard nnUNet overfit (no custom code)
    nnUNetv2_train 501 3d_fullres 0 --npz

    # R002: Box-IPW sanity
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --ipw_mode channel \
        --num_epochs 50 --warmup_epochs 10

    # R025-R027: No-box baseline (pixel-only)
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --ipw_mode none --seed 42

    # R028-R030: Uniform PU baseline
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --ipw_mode uniform --seed 42

    # R032-R034: Ours (channel IPW)
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --ipw_mode channel --seed 42
"""
import argparse
import json
import os
import sys
import time

import torch
import numpy as np

from batchgenerators.utilities.file_and_folder_operations import load_json, join

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from latentmask.trainer.latentmask_trainer import LatentMaskTrainer


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Launch LatentMask training')
    parser.add_argument('--dataset_id', type=int, default=501)
    parser.add_argument('--configuration', default='3d_fullres')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    # LatentMask config
    parser.add_argument('--ipw_mode', default='channel',
                        choices=['none', 'uniform', 'channel', 'oracle'])
    parser.add_argument('--steepness', default='medium',
                        choices=['shallow', 'medium', 'steep'])
    parser.add_argument('--pixel_fraction', type=float, default=0.3)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--warmup_epochs', type=int, default=50)
    parser.add_argument('--ramp_epochs', type=int, default=50)
    parser.add_argument('--lambda_box_max', type=float, default=1.0)
    parser.add_argument('--w_max', type=float, default=10.0)
    parser.add_argument('--d_margin', type=int, default=5)
    parser.add_argument('--pi_hat_scale', type=float, default=1.0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Set environment variables for LatentMask config
    os.environ['LM_IPW_MODE'] = args.ipw_mode
    os.environ['LM_STEEPNESS'] = args.steepness
    os.environ['LM_PIXEL_FRACTION'] = str(args.pixel_fraction)
    os.environ['LM_WARMUP_EPOCHS'] = str(args.warmup_epochs)
    os.environ['LM_RAMP_EPOCHS'] = str(args.ramp_epochs)
    os.environ['LM_LAMBDA_BOX_MAX'] = str(args.lambda_box_max)
    os.environ['LM_W_MAX'] = str(args.w_max)
    os.environ['LM_D_MARGIN'] = str(args.d_margin)
    os.environ['LM_PI_HAT_SCALE'] = str(args.pi_hat_scale)

    # Load plans and dataset.json
    dataset_name = f'Dataset{args.dataset_id:03d}_LiTS'
    preprocessed_folder = join(nnUNet_preprocessed, dataset_name)

    if not os.path.isdir(preprocessed_folder):
        print(f"ERROR: Preprocessed data not found at {preprocessed_folder}")
        print(f"Run: nnUNetv2_plan_and_preprocess -d {args.dataset_id} "
              f"--verify_dataset_integrity")
        sys.exit(1)

    plans = load_json(join(preprocessed_folder, 'nnUNetPlans.json'))
    dataset_json = load_json(join(preprocessed_folder, 'dataset.json'))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Log configuration
    run_config = {
        'run_id': f'R_seed{args.seed}_{args.ipw_mode}_{args.steepness}',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
    }
    print(f"=== LatentMask Training ===")
    print(f"  ipw_mode:    {args.ipw_mode}")
    print(f"  steepness:   {args.steepness}")
    print(f"  seed:        {args.seed}")
    print(f"  epochs:      {args.num_epochs}")
    print(f"  warmup:      {args.warmup_epochs}")
    print(f"  device:      {device}")

    # Create trainer
    trainer = LatentMaskTrainer(
        plans=plans,
        configuration=args.configuration,
        fold=args.fold,
        dataset_json=dataset_json,
        device=device,
    )
    trainer.num_epochs = args.num_epochs

    # Save run config
    os.makedirs(trainer.output_folder, exist_ok=True)
    config_path = join(trainer.output_folder, 'run_config.json')
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)

    # Train
    trainer.run_training()

    # Save final results summary
    results_path = join(trainer.output_folder, 'training_complete.json')
    results = {
        'status': 'DONE',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': run_config,
        'output_folder': trainer.output_folder,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete. Results at: {trainer.output_folder}")


if __name__ == '__main__':
    main()
