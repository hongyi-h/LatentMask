"""LatentMask Trainer: subclass of nnUNetTrainer for box-supervised segmentation.

Overrides:
  - on_train_start: pre-fit isotonic calibrator g_θ, compute π̂
  - run_training: implements [pixel, pixel, box] batch cycle
  - on_train_epoch_start: λ_box ramp schedule
  - on_epoch_end: diagnostic logging (weights, Δ_area)
  - get_dataloaders: creates separate pixel and box dataloaders

Configuration via ipw_mode:
  - 'none': no box loss (pixel-only baseline, R025-R027)
  - 'uniform': w=1 for all boxes (SCAR baseline ≈ 3D-BoxSup, R028-R030)
  - 'channel': isotonic g_θ IPW (our method, R032-R036)
  - 'oracle': true g_true + true sizes (upper bound, R031+)
"""
import json
import os
import sys
from copy import deepcopy
from time import time
from typing import List

import numpy as np
import torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.helpers import empty_cache, dummy_context

from latentmask.losses.bag_pu_loss import compute_batch_box_loss
from latentmask.calibration.isotonic_fit import fit_isotonic, predict_propensity
from latentmask.calibration.channel_simulator import make_channel_func, generate_box_annotations
from latentmask.utils.cc_extraction import extract_connected_components


class LatentMaskTrainer(nnUNetTrainer):

    # ── Configuration defaults ──────────────────────────────────────────
    IPW_MODE = 'channel'       # 'none' | 'uniform' | 'channel' | 'oracle'
    STEEPNESS = 'medium'       # 'shallow' | 'medium' | 'steep'
    PIXEL_FRACTION = 0.3       # fraction of training data with pixel labels
    WARMUP_EPOCHS = 50         # epochs of pixel-only training
    RAMP_EPOCHS = 50           # epochs of λ_box linear ramp
    LAMBDA_BOX_MAX = 1.0       # maximum box loss weight
    W_MAX = 10.0               # IPW weight clipping
    D_MARGIN = 5               # safe zone margin in voxels
    PI_HAT_SCALE = 1.0         # multiplier for π̂ (for sensitivity ablation)
    MIN_CC_SIZE = 10           # minimum CC size to consider
    FG_LABEL = 2               # foreground label for CC extraction (2=tumor for LiTS)
    DIAG_EPOCHS = [50, 100, 150, 200, 250, 300]  # when to log diagnostics

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Override training length
        self.num_epochs = 300

        # Read config from environment (for CLI compatibility)
        self.ipw_mode = os.environ.get('LM_IPW_MODE', self.IPW_MODE)
        self.steepness = os.environ.get('LM_STEEPNESS', self.STEEPNESS)
        self.pixel_fraction = float(os.environ.get('LM_PIXEL_FRACTION',
                                                    self.PIXEL_FRACTION))
        self.warmup_epochs = int(os.environ.get('LM_WARMUP_EPOCHS',
                                                 self.WARMUP_EPOCHS))
        self.ramp_epochs = int(os.environ.get('LM_RAMP_EPOCHS',
                                               self.RAMP_EPOCHS))
        self.lambda_box_max = float(os.environ.get('LM_LAMBDA_BOX_MAX',
                                                     self.LAMBDA_BOX_MAX))
        self.w_max = float(os.environ.get('LM_W_MAX', self.W_MAX))
        self.d_margin = int(os.environ.get('LM_D_MARGIN', self.D_MARGIN))
        self.pi_hat_scale = float(os.environ.get('LM_PI_HAT_SCALE',
                                                   self.PI_HAT_SCALE))

        # Will be set during on_train_start
        self.g_theta = None       # isotonic calibrator
        self.g_theta_s0 = None    # minimum support
        self.pi_hat = None        # class prior
        self.g_true = None        # ground-truth channel (for simulation)
        self.lambda_box = 0.0     # current box loss weight
        self.pixel_keys = []
        self.box_keys = []
        self.box_annotations = {}  # key -> list of box dicts
        self.dataloader_box = None
        self.diag_log = []

        if self.ipw_mode == 'none':
            self.warmup_epochs = self.num_epochs  # never use box loss

    # ── Dataloaders ─────────────────────────────────────────────────────

    def get_dataloaders(self):
        """Create pixel dataloader (standard) + box dataloader (separate)."""
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from batchgenerators.dataloading.single_threaded_augmenter import (
            SingleThreadedAugmenter,
        )
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
            NonDetMultiThreadedAugmenter,
        )

        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(
                self.preprocessed_dataset_folder
            )

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA, do_dummy_2d_data_aug,
            initial_patch_size, mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(self.label_manager.foreground_regions
                     if self.label_manager.has_regions else None),
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(self.label_manager.foreground_regions
                     if self.label_manager.has_regions else None),
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # ── Split training keys ──────────────────────────────────────
        all_keys = sorted(dataset_tr.identifiers)
        rng = np.random.default_rng(seed=12345 + self.fold)
        perm = rng.permutation(len(all_keys)).tolist()
        n_pixel = max(1, int(self.pixel_fraction * len(all_keys)))

        if self.ipw_mode == 'none':
            # Pixel-only baseline: use only the pixel subset
            self.pixel_keys = [all_keys[i] for i in perm[:n_pixel]]
            self.box_keys = []
        else:
            self.pixel_keys = [all_keys[i] for i in perm[:n_pixel]]
            self.box_keys = [all_keys[i] for i in perm[n_pixel:]]
            if len(self.box_keys) == 0:
                self.print_to_log_file(
                    'WARNING: no box keys (dataset too small for split). '
                    'Falling back to pixel-only training.'
                )
                self.ipw_mode = 'none'
                self.warmup_epochs = self.num_epochs

        self.print_to_log_file(
            f"LatentMask split: {len(self.pixel_keys)} pixel, "
            f"{len(self.box_keys)} box, mode={self.ipw_mode}"
        )

        # ── Pixel DataLoader ─────────────────────────────────────────
        # Create a dataset view with only pixel keys
        pixel_dataset = self.dataset_class(
            self.preprocessed_dataset_folder, self.pixel_keys,
            folder_with_segs_from_previous_stage=
            self.folder_with_segs_from_previous_stage,
        )
        dl_pixel = nnUNetDataLoader(
            pixel_dataset, self.batch_size, initial_patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        # ── Box DataLoader ────────────────────────────────────────────
        if len(self.box_keys) > 0:
            box_dataset = self.dataset_class(
                self.preprocessed_dataset_folder, self.box_keys,
                folder_with_segs_from_previous_stage=
                self.folder_with_segs_from_previous_stage,
            )
            dl_box_raw = nnUNetDataLoader(
                box_dataset, self.batch_size, initial_patch_size,
                self.configuration_manager.patch_size, self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None,
                transforms=tr_transforms,
                probabilistic_oversampling=self.probabilistic_oversampling,
            )
            self.dataloader_box = SingleThreadedAugmenter(dl_box_raw, None)
            _ = next(self.dataloader_box)  # prime
        else:
            self.dataloader_box = None

        # ── Validation DataLoader ─────────────────────────────────────
        dl_val = nnUNetDataLoader(
            dataset_val, self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        # Wrap in augmentation threads
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        allowed = get_allowed_n_proc_DA()
        if allowed == 0:
            mt_pixel = SingleThreadedAugmenter(dl_pixel, None)
            mt_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_pixel = NonDetMultiThreadedAugmenter(
                data_loader=dl_pixel, transform=None,
                num_processes=allowed,
                num_cached=max(6, allowed // 2), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002,
            )
            mt_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None,
                num_processes=max(1, allowed // 2),
                num_cached=max(3, allowed // 4), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002,
            )

        _ = next(mt_pixel)
        _ = next(mt_val)
        return mt_pixel, mt_val

    # ── Pre-training calibration ────────────────────────────────────────

    def on_train_start(self):
        super().on_train_start()
        self._prefit_calibration()

    def _prefit_calibration(self):
        """Pre-fit isotonic g_θ and estimate π̂ on pixel subset."""
        self.print_to_log_file("Pre-fitting calibration...")

        if self.ipw_mode == 'none':
            self.print_to_log_file("  ipw_mode=none, skipping calibration.")
            return

        # Collect CCs from pixel-labeled cases
        all_log_sizes = []
        all_fg_counts = []
        all_total_counts = []

        for key in self.pixel_keys:
            data, seg, _, props = self.dataset_class(
                self.preprocessed_dataset_folder, [key],
            ).load_case(key)
            seg_np = seg[0]  # shape (D, H, W)
            ccs = extract_connected_components(seg_np, min_size=self.MIN_CC_SIZE,
                                                fg_label=self.FG_LABEL)
            for cc in ccs:
                all_log_sizes.append(cc['log_size'])
            if self.FG_LABEL is not None:
                all_fg_counts.append(int((seg_np == self.FG_LABEL).sum()))
            else:
                all_fg_counts.append(int((seg_np > 0).sum()))
            all_total_counts.append(int(seg_np.size))

        # Estimate π̂
        total_fg = sum(all_fg_counts)
        total_vox = sum(all_total_counts)
        self.pi_hat = (total_fg / max(total_vox, 1)) * self.pi_hat_scale
        self.print_to_log_file(f"  π̂ = {self.pi_hat:.6f} "
                                f"(scale={self.pi_hat_scale})")

        if len(all_log_sizes) == 0:
            self.print_to_log_file("  WARNING: no CCs found, skipping fit.")
            return

        # Compute μ (median log-CC-size) for channel function
        all_log_sizes = np.array(all_log_sizes)
        mu = float(np.median(all_log_sizes))
        self.print_to_log_file(f"  μ (median log-CC-size) = {mu:.2f}, "
                                f"n_CCs = {len(all_log_sizes)}")

        # Create channel function
        self.g_true = make_channel_func(self.steepness, mu)

        # Simulate channel on pixel CCs → fit isotonic
        rng = np.random.default_rng(seed=42 + self.fold)
        true_probs = self.g_true(all_log_sizes)
        selection = (rng.random(len(all_log_sizes)) < true_probs).astype(float)

        self.g_theta, self.g_theta_s0 = fit_isotonic(all_log_sizes, selection)
        self.print_to_log_file(f"  g_θ fitted: s0={self.g_theta_s0:.2f}")

        # Save calibration results
        calib_path = os.path.join(self.output_folder, 'calibration_prefit.json')
        calib_results = {
            'pi_hat': self.pi_hat,
            'mu': mu,
            's0': self.g_theta_s0,
            'n_ccs': len(all_log_sizes),
            'steepness': self.steepness,
            'ipw_mode': self.ipw_mode,
        }
        with open(calib_path, 'w') as f:
            json.dump(calib_results, f, indent=2)

    # ── λ_box schedule ──────────────────────────────────────────────────

    def _get_lambda_box(self):
        """Compute current λ_box based on epoch."""
        if self.current_epoch < self.warmup_epochs:
            return 0.0
        ramp_end = self.warmup_epochs + self.ramp_epochs
        if self.current_epoch < ramp_end:
            progress = ((self.current_epoch - self.warmup_epochs)
                        / max(self.ramp_epochs, 1))
            return self.lambda_box_max * progress
        return self.lambda_box_max

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.lambda_box = self._get_lambda_box()
        self.print_to_log_file(f"  λ_box = {self.lambda_box:.4f}")

    # ── Training loop ───────────────────────────────────────────────────

    def run_training(self):
        """Main training loop with [pixel, pixel, box] batch cycle."""
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()

            train_outputs = []
            step_counter = 0
            use_box = (self.lambda_box > 0 and self.dataloader_box is not None)

            for batch_id in range(self.num_iterations_per_epoch):
                # Cycle: [pixel, pixel, box]
                is_box_step = (use_box and step_counter % 3 == 2)

                if is_box_step:
                    batch = next(self.dataloader_box)
                    out = self._box_train_step(batch)
                else:
                    batch = next(self.dataloader_train)
                    out = self.train_step(batch)

                train_outputs.append(out)
                step_counter += 1

            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(
                        self.validation_step(next(self.dataloader_val))
                    )
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

    def _box_train_step(self, batch):
        """Training step for box-supervised data."""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            # Deep supervision: use highest resolution only for box loss
            target_full = target[0].to(self.device, non_blocking=True)
        else:
            target_full = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # g_theta function for predict_propensity (CPU-only, define outside autocast)
        if self.g_theta is not None:
            def g_func(log_sizes):
                return predict_propensity(
                    self.g_theta, log_sizes, self.g_theta_s0
                )
        else:
            def g_func(log_sizes):
                return np.ones_like(log_sizes)

        with (autocast(self.device.type, enabled=True)
              if self.device.type == 'cuda' else dummy_context()):
            output = self.network(data)
            # Use highest resolution output for box loss
            out_full = output[0] if isinstance(output, list) else output

            l_box, diag = compute_batch_box_loss(
                out_full, target_full,
                pi_hat=self.pi_hat or 0.01,
                g_theta_func=g_func,
                s0=self.g_theta_s0 or 0.0,
                d_margin=self.d_margin,
                w_max=self.w_max,
                ipw_mode=self.ipw_mode,
                min_cc_size=self.MIN_CC_SIZE,
                fg_label=self.FG_LABEL,
            )

            l = self.lambda_box * l_box

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    # ── Diagnostics ─────────────────────────────────────────────────────

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.current_epoch in self.DIAG_EPOCHS:
            self._log_diagnostics()

    def _log_diagnostics(self):
        """Log weight distribution and Δ_area at checkpoint epochs."""
        diag = {'epoch': self.current_epoch, 'lambda_box': self.lambda_box}

        if self.g_theta is not None and self.ipw_mode != 'none':
            diag['pi_hat'] = self.pi_hat
            diag['ipw_mode'] = self.ipw_mode

        self.diag_log.append(diag)
        diag_path = os.path.join(self.output_folder, 'latentmask_diagnostics.json')
        with open(diag_path, 'w') as f:
            json.dump(self.diag_log, f, indent=2)

        self.print_to_log_file(
            f"  [LatentMask diag] epoch={self.current_epoch}, "
            f"λ_box={self.lambda_box:.4f}"
        )

    def on_train_end(self):
        # Save final diagnostics
        diag_path = os.path.join(self.output_folder,
                                  'latentmask_diagnostics.json')
        with open(diag_path, 'w') as f:
            json.dump(self.diag_log, f, indent=2)

        # Save final config
        config_path = os.path.join(self.output_folder,
                                    'latentmask_config.json')
        config = {
            'ipw_mode': self.ipw_mode,
            'steepness': self.steepness,
            'pixel_fraction': self.pixel_fraction,
            'warmup_epochs': self.warmup_epochs,
            'ramp_epochs': self.ramp_epochs,
            'lambda_box_max': self.lambda_box_max,
            'w_max': self.w_max,
            'd_margin': self.d_margin,
            'pi_hat_scale': self.pi_hat_scale,
            'pi_hat': self.pi_hat,
            'n_pixel_keys': len(self.pixel_keys),
            'n_box_keys': len(self.box_keys),
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        super().on_train_end()
