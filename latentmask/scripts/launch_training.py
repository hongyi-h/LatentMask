"""Launch LatentMask v5 training experiments.

Usage:
    # C1: pixel-only baseline
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --neg_mode none --seed 42

    # C2: scaffold + uniform-neg
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --neg_mode uniform --seed 42

    # C3: scaffold + linear-neg
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --neg_mode linear --seed 42

    # C4: scaffold + g_theta-neg (our method)
    python -m latentmask.scripts.launch_training \
        --dataset_id 501 --fold 0 --neg_mode channel --seed 42
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
    parser = argparse.ArgumentParser(description='Launch LatentMask v5 training')
    parser.add_argument('--dataset_id', type=int, default=501)
    parser.add_argument('--configuration', default='3d_fullres')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    # v5 config
    parser.add_argument('--neg_mode', default='channel',
                        choices=['none', 'uniform', 'linear', 'channel'],
                        help='C1=none, C2=uniform, C3=linear, C4=channel')
    parser.add_argument('--steepness', default='medium',
                        choices=['shallow', 'medium', 'steep'])
    parser.add_argument('--pixel_fraction', type=float, default=0.3)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--warmup_epochs', type=int, default=50)
    parser.add_argument('--ramp_epochs', type=int, default=50)
    parser.add_argument('--channel_neg_start', type=int, default=60)
    parser.add_argument('--lambda_box_max', type=float, default=1.0)
    parser.add_argument('--w_max', type=float, default=10.0)
    parser.add_argument('--d_margin', type=int, default=5)
    parser.add_argument('--device', default='cuda')

    # v5 loss hyperparams
    parser.add_argument('--kappa', type=float, default=1.0)
    parser.add_argument('--beta_fill', type=float, default=1.0)
    parser.add_argument('--gamma_neg', type=float, default=1.0)
    parser.add_argument('--tau_low', type=float, default=0.3)
    parser.add_argument('--tau_high', type=float, default=0.5)
    parser.add_argument('--alpha_min', type=float, default=0.05)

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Set environment variables for LatentMask v5 config
    os.environ['LM_NEG_MODE'] = args.neg_mode
    os.environ['LM_STEEPNESS'] = args.steepness
    os.environ['LM_PIXEL_FRACTION'] = str(args.pixel_fraction)
    os.environ['LM_WARMUP_EPOCHS'] = str(args.warmup_epochs)
    os.environ['LM_RAMP_EPOCHS'] = str(args.ramp_epochs)
    os.environ['LM_CHANNEL_NEG_START'] = str(args.channel_neg_start)
    os.environ['LM_LAMBDA_BOX_MAX'] = str(args.lambda_box_max)
    os.environ['LM_W_MAX'] = str(args.w_max)
    os.environ['LM_D_MARGIN'] = str(args.d_margin)
    os.environ['LM_KAPPA'] = str(args.kappa)
    os.environ['LM_BETA_FILL'] = str(args.beta_fill)
    os.environ['LM_GAMMA_NEG'] = str(args.gamma_neg)
    os.environ['LM_TAU_LOW'] = str(args.tau_low)
    os.environ['LM_TAU_HIGH'] = str(args.tau_high)
    os.environ['LM_ALPHA_MIN'] = str(args.alpha_min)

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

    # Config name for output folder
    config_name = f"{args.neg_mode}_{args.steepness}_seed{args.seed}"

    print(f"=== LatentMask v5 Training ===")
    print(f"  neg_mode:    {args.neg_mode}")
    print(f"  steepness:   {args.steepness}")
    print(f"  seed:        {args.seed}")
    print(f"  epochs:      {args.num_epochs}")
    print(f"  warmup:      {args.warmup_epochs}")
    print(f"  tau_low:     {args.tau_low}")
    print(f"  tau_high:    {args.tau_high}")
    print(f"  alpha_min:   {args.alpha_min}")
    print(f"  gamma_neg:   {args.gamma_neg}")
    print(f"  d_margin:    {args.d_margin}")
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

    # Disambiguate output folder
    trainer.output_folder = join(trainer.output_folder, config_name)
    os.makedirs(trainer.output_folder, exist_ok=True)

    # Re-bind logger
    from datetime import datetime
    from nnunetv2.training.logging.nnunet_logger import MetaLogger
    timestamp = datetime.now()
    trainer.log_file = join(
        trainer.output_folder,
        "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
        (timestamp.year, timestamp.month, timestamp.day,
         timestamp.hour, timestamp.minute, timestamp.second))
    trainer.logger = MetaLogger(trainer.output_folder, False)

    # Save run config
    run_config = {
        'run_id': config_name,
        'version': 'v5',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
    }
    config_path = join(trainer.output_folder, 'run_config.json')
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)

    # Train
    trainer.run_training()

    # Save completion marker
    results_path = join(trainer.output_folder, 'training_complete.json')
    results = {
        'status': 'DONE',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': run_config,
        'output_folder': trainer.output_folder,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete. Output: {trainer.output_folder}")


if __name__ == '__main__':
    main()
