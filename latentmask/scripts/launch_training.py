"""Launch LatentMask v6.1 training.

Calibration artifact (produced by run_calibration_cv.py) must exist at
`{box_annotations_dir}/_calibration_fold{fold}.pkl` before launching
for neg_mode ∈ {linear, channel, inverted}.

Usage (LiTS, C4 / channel mode, fold 0):
    python -m latentmask.scripts.launch_training \
        --dataset_name Dataset501_LiTS --fold 0 \
        --neg_mode channel --fg_label 2 \
        --box_annotations_dir data/box_annotations/P-steep

Usage (BraTS-METS, once the dataset is prepared):
    python -m latentmask.scripts.launch_training \
        --dataset_name Dataset502_BraTSMETS --fold 0 \
        --neg_mode channel --fg_label 1 \
        --box_annotations_dir data/box_annotations_brats/P-steep
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
from latentmask.trainer.latentmask_trainer import (
    LatentMaskTrainer, VALID_NEG_MODES,
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Launch LatentMask v6.1 training')

    # Dataset / fold
    parser.add_argument('--dataset_name', default='Dataset501_LiTS',
                        help='Preprocessed nnUNet dataset folder name')
    parser.add_argument('--configuration', default='3d_fullres')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    # v6.1 core config
    parser.add_argument('--neg_mode', default='channel',
                        choices=sorted(VALID_NEG_MODES),
                        help='C1=none, C2=uniform, C2.5=constant, '
                             'C3=linear, C4=channel, C4-inv=inverted')
    parser.add_argument('--fg_label', type=int, default=2,
                        help='Foreground label (2=LiTS tumor, 1=BraTS-METS mets)')
    parser.add_argument('--box_annotations_dir', default='',
                        help='Directory holding protocol-specific box JSONs + '
                             '_calibration_fold{fold}.pkl + box_segmentations/')
    parser.add_argument('--pixel_fraction', type=float, default=0.3)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--warmup_epochs', type=int, default=50)
    parser.add_argument('--ramp_epochs', type=int, default=50)
    parser.add_argument('--channel_neg_start', type=int, default=60)
    parser.add_argument('--lambda_box_max', type=float, default=1.0)
    parser.add_argument('--w_max', type=float, default=10.0)
    parser.add_argument('--d_margin', type=int, default=5)
    parser.add_argument('--device', default='cuda')

    # Loss hyperparams
    parser.add_argument('--kappa', type=float, default=1.0)
    parser.add_argument('--beta_fill', type=float, default=1.0)
    parser.add_argument('--gamma_neg', type=float, default=1.0)
    parser.add_argument('--tau_low', type=float, default=0.3)
    parser.add_argument('--tau_high', type=float, default=0.5)
    parser.add_argument('--alpha_min', type=float, default=0.05)
    parser.add_argument('--constant_alpha', type=float, default=0.5,
                        help='α for C2.5 constant mode')

    args = parser.parse_args()

    set_seed(args.seed)

    # Export all config via env vars (trainer reads env at __init__)
    os.environ['LM_NEG_MODE'] = args.neg_mode
    os.environ['LM_FG_LABEL'] = str(args.fg_label)
    os.environ['LM_PIXEL_FRACTION'] = str(args.pixel_fraction)
    os.environ['LM_NUM_EPOCHS'] = str(args.num_epochs)
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
    os.environ['LM_CONSTANT_ALPHA'] = str(args.constant_alpha)
    if args.box_annotations_dir:
        os.environ['LM_BOX_ANNOTATIONS_DIR'] = args.box_annotations_dir

    # Fail fast if box-supervised run is missing the calibration artifact
    if args.neg_mode in {'linear', 'channel', 'inverted'}:
        if not args.box_annotations_dir:
            print("ERROR: --box_annotations_dir is required for "
                  f"neg_mode={args.neg_mode}")
            sys.exit(1)
        art = os.path.join(args.box_annotations_dir,
                           f'_calibration_fold{args.fold}.pkl')
        if not os.path.isfile(art):
            print(f"ERROR: calibration artifact missing: {art}")
            print(f"  Run run_calibration_cv.py first "
                  f"(fold={args.fold}, fg_label={args.fg_label}).")
            sys.exit(1)

    preprocessed_folder = join(nnUNet_preprocessed, args.dataset_name)
    if not os.path.isdir(preprocessed_folder):
        print(f"ERROR: Preprocessed data not found: {preprocessed_folder}")
        sys.exit(1)

    plans = load_json(join(preprocessed_folder, 'nnUNetPlans.json'))
    dataset_json = load_json(join(preprocessed_folder, 'dataset.json'))

    if 'continue_training' not in plans:
        plans['continue_training'] = False

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Output folder tag; protocol is inferred from box_annotations_dir name
    proto_tag = (os.path.basename(args.box_annotations_dir.rstrip('/'))
                 if args.box_annotations_dir else 'none')
    config_name = (f"{args.neg_mode}_{proto_tag}_fold{args.fold}"
                   f"_seed{args.seed}")

    print(f"=== LatentMask v6.1 Training ===")
    print(f"  dataset:     {args.dataset_name}")
    print(f"  neg_mode:    {args.neg_mode}")
    print(f"  fg_label:    {args.fg_label}")
    print(f"  protocol:    {proto_tag}")
    print(f"  fold:        {args.fold}")
    print(f"  seed:        {args.seed}")
    print(f"  epochs:      {args.num_epochs}")
    print(f"  device:      {device}")

    trainer = LatentMaskTrainer(
        plans=plans,
        configuration=args.configuration,
        fold=args.fold,
        dataset_json=dataset_json,
        device=device,
    )
    trainer.num_epochs = args.num_epochs

    trainer.output_folder = join(trainer.output_folder, config_name)
    os.makedirs(trainer.output_folder, exist_ok=True)

    from datetime import datetime
    from nnunetv2.training.logging.nnunet_logger import MetaLogger
    timestamp = datetime.now()
    trainer.log_file = join(
        trainer.output_folder,
        "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
        (timestamp.year, timestamp.month, timestamp.day,
         timestamp.hour, timestamp.minute, timestamp.second))
    trainer.logger = MetaLogger(trainer.output_folder, False)

    run_config = {
        'run_id': config_name,
        'version': 'v6.1',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
    }
    with open(join(trainer.output_folder, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)

    trainer.run_training()

    results = {
        'status': 'DONE',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': run_config,
        'output_folder': trainer.output_folder,
    }
    with open(join(trainer.output_folder, 'training_complete.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete. Output: {trainer.output_folder}")


if __name__ == '__main__':
    main()
