"""LatentMask Trainer v5: subclass of nnUNetTrainer for box-supervised segmentation.

v5 changes from v3:
  - neg_mode replaces ipw_mode: none/uniform/linear/channel → C1-C4
  - g_theta fitted via Hungarian matching protocol (not simulated channel)
  - CC-level negative supervision (not avg_pool3d)
  - Diagnostics: coverage_ratio, nascent_ratio, mean_alpha
  - Low-coverage fallback: d_safe 5→3 at epoch 100

Configuration via neg_mode:
  - 'none':    C1 pixel-only baseline (no box loss)
  - 'uniform': C2 scaffold + uniform neg (α ≡ 1.0)
  - 'linear':  C3 scaffold + linear neg (α = a + b·log m)
  - 'channel': C4 scaffold + g_θ neg (isotonic, our method)
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

from latentmask.losses.bag_pu_loss import compute_batch_box_loss_v5
from latentmask.calibration.isotonic_fit import (
    fit_g_theta_hungarian, predict_propensity, compute_ece,
)
from latentmask.calibration.channel_simulator import make_channel_func
from latentmask.utils.cc_extraction import (
    extract_connected_components, compute_safe_zone_mask,
)
from scipy import ndimage


class LatentMaskTrainer(nnUNetTrainer):

    # ── Configuration defaults ──────────────────────────────────────────
    NEG_MODE = 'channel'       # 'none' | 'uniform' | 'linear' | 'channel'
    STEEPNESS = 'medium'       # for simulated drop_fn: 'shallow'|'medium'|'steep'
    PIXEL_FRACTION = 0.3       # fraction of training data with pixel labels
    WARMUP_EPOCHS = 50         # epochs of pixel-only training
    RAMP_EPOCHS = 50           # epochs of lambda_box ramp
    CHANNEL_NEG_START = 60     # epoch to enable channel-neg (within ramp)
    LAMBDA_BOX_MAX = 1.0       # maximum box loss weight
    W_MAX = 10.0               # IPW weight clipping
    D_MARGIN = 5               # safe zone margin (d_safe)
    MIN_CC_SIZE = 10           # minimum CC size
    FG_LABEL = 2               # foreground label (2=tumor for LiTS)
    DIAG_EPOCHS = [50, 100, 150, 200, 250, 300]

    # v5 loss hyperparameters
    KAPPA = 1.0                # tightness threshold
    BETA_FILL = 1.0            # L_fill weight
    GAMMA_NEG = 1.0            # L_channel_neg weight (v5: increased from 0.1)
    ALPHA_UPPER = 1.0          # upper fill violation weight
    TAU_LOW = 0.3              # CC extraction low threshold
    TAU_HIGH = 0.5             # CC extraction high threshold
    ALPHA_MIN = 0.05           # minimum alpha for nascent CCs

    # Low-coverage fallback
    COVERAGE_FALLBACK_EPOCH = 100
    COVERAGE_FALLBACK_THRESHOLD = 0.10
    COVERAGE_FALLBACK_D_SAFE = 3

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 300

        # Read config from environment
        self.neg_mode = os.environ.get('LM_NEG_MODE', self.NEG_MODE)
        self.steepness = os.environ.get('LM_STEEPNESS', self.STEEPNESS)
        self.pixel_fraction = float(os.environ.get('LM_PIXEL_FRACTION',
                                                    self.PIXEL_FRACTION))
        self.warmup_epochs = int(os.environ.get('LM_WARMUP_EPOCHS',
                                                 self.WARMUP_EPOCHS))
        self.ramp_epochs = int(os.environ.get('LM_RAMP_EPOCHS',
                                               self.RAMP_EPOCHS))
        self.channel_neg_start = int(os.environ.get('LM_CHANNEL_NEG_START',
                                                     self.CHANNEL_NEG_START))
        self.lambda_box_max = float(os.environ.get('LM_LAMBDA_BOX_MAX',
                                                     self.LAMBDA_BOX_MAX))
        self.w_max = float(os.environ.get('LM_W_MAX', self.W_MAX))
        self.d_margin = int(os.environ.get('LM_D_MARGIN', self.D_MARGIN))
        self.kappa = float(os.environ.get('LM_KAPPA', self.KAPPA))
        self.beta_fill = float(os.environ.get('LM_BETA_FILL', self.BETA_FILL))
        self.gamma_neg = float(os.environ.get('LM_GAMMA_NEG', self.GAMMA_NEG))
        self.alpha_upper = float(os.environ.get('LM_ALPHA_UPPER',
                                                  self.ALPHA_UPPER))
        self.tau_low = float(os.environ.get('LM_TAU_LOW', self.TAU_LOW))
        self.tau_high = float(os.environ.get('LM_TAU_HIGH', self.TAU_HIGH))
        self.alpha_min = float(os.environ.get('LM_ALPHA_MIN', self.ALPHA_MIN))

        # Calibration state (set during on_train_start)
        self.g_theta = None
        self.g_theta_s0 = None
        self.linear_a = 0.0
        self.linear_b = 0.1
        self.pi_hat = None
        self.lambda_box = 0.0
        self.rho_min = 0.15
        self.rho_max = 0.85
        self.pixel_keys = []
        self.box_keys = []
        self.dataloader_box = None
        self.diag_log = []
        self._epoch_box_diags = []
        self._fallback_triggered = False

        if self.neg_mode == 'none':
            self.warmup_epochs = self.num_epochs

    # ── Dataloaders ─────────────────────────────────────────────────────

    def get_dataloaders(self):
        """Create pixel dataloader + box dataloader."""
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

        # Split training keys into pixel / box
        all_keys = sorted(dataset_tr.identifiers)
        rng = np.random.default_rng(seed=12345 + self.fold)
        perm = rng.permutation(len(all_keys)).tolist()
        n_pixel = max(1, int(self.pixel_fraction * len(all_keys)))

        if self.neg_mode == 'none':
            self.pixel_keys = [all_keys[i] for i in perm[:n_pixel]]
            self.box_keys = []
        else:
            self.pixel_keys = [all_keys[i] for i in perm[:n_pixel]]
            self.box_keys = [all_keys[i] for i in perm[n_pixel:]]
            if len(self.box_keys) == 0:
                self.print_to_log_file(
                    'WARNING: no box keys. Falling back to pixel-only.')
                self.neg_mode = 'none'
                self.warmup_epochs = self.num_epochs

        self.print_to_log_file(
            f"LatentMask v5 split: {len(self.pixel_keys)} pixel, "
            f"{len(self.box_keys)} box, neg_mode={self.neg_mode}"
        )

        # Pixel DataLoader
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

        # Box DataLoader
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
            _ = next(self.dataloader_box)
        else:
            self.dataloader_box = None

        # Validation DataLoader
        dl_val = nnUNetDataLoader(
            dataset_val, self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

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

    # ── Pre-training calibration (v5: Hungarian matching) ───────────────

    def on_train_start(self):
        super().on_train_start()
        self._prefit_calibration_v5()

    def _prefit_calibration_v5(self):
        """Pre-fit g_theta via Hungarian matching on pixel-labeled scans."""
        self.print_to_log_file("Pre-fitting calibration (v5 Hungarian)...")

        if self.neg_mode == 'none':
            self.print_to_log_file("  neg_mode=none, skipping calibration.")
            return

        # Load pixel-labeled segmentations
        seg_list = []
        all_fg_counts = []
        all_total_counts = []

        ds = self.dataset_class(self.preprocessed_dataset_folder, self.pixel_keys)
        for key in self.pixel_keys:
            data, seg, _, props = ds.load_case(key)
            seg_np = seg[0]
            seg_list.append(seg_np)
            if self.FG_LABEL is not None:
                all_fg_counts.append(int((seg_np == self.FG_LABEL).sum()))
            else:
                all_fg_counts.append(int((seg_np > 0).sum()))
            all_total_counts.append(int(seg_np.size))

        # Estimate pi_hat
        total_fg = sum(all_fg_counts)
        total_vox = sum(all_total_counts)
        self.pi_hat = total_fg / max(total_vox, 1)
        self.print_to_log_file(f"  pi_hat = {self.pi_hat:.6f}")

        # Create drop function (size-dependent annotation simulation)
        all_ccs_tmp = []
        for seg_np in seg_list:
            ccs = extract_connected_components(
                seg_np, min_size=self.MIN_CC_SIZE, fg_label=self.FG_LABEL)
            all_ccs_tmp.extend(ccs)

        if len(all_ccs_tmp) == 0:
            self.print_to_log_file("  WARNING: no CCs found, skipping fit.")
            return

        all_log_sizes = np.array([cc['log_size'] for cc in all_ccs_tmp])
        mu = float(np.median(all_log_sizes))
        self.print_to_log_file(
            f"  mu={mu:.2f}, n_CCs={len(all_log_sizes)}")

        # Simulated drop function based on steepness
        g_true = make_channel_func(self.steepness, mu)

        def drop_fn(log_sizes):
            return g_true(log_sizes)

        # Fit via Hungarian matching
        rng = np.random.default_rng(seed=42 + self.fold)
        ir, s0, linear_a, linear_b, stats = fit_g_theta_hungarian(
            seg_list, fg_label=self.FG_LABEL,
            min_cc_size=self.MIN_CC_SIZE, iou_threshold=0.3,
            drop_fn=drop_fn, rng=rng)

        self.g_theta = ir
        self.g_theta_s0 = s0
        self.linear_a = linear_a
        self.linear_b = linear_b

        self.print_to_log_file(
            f"  g_theta fitted: s0={s0:.2f}, "
            f"unmatched_rate={stats['unmatched_rate']:.3f}, "
            f"ambiguous_rate={stats['ambiguous_rate']:.3f}")
        self.print_to_log_file(
            f"  linear: a={linear_a:.4f}, b={linear_b:.4f}")

        # Estimate filling rate bounds
        fill_ratios = []
        for seg_np in seg_list:
            ccs = extract_connected_components(
                seg_np, min_size=self.MIN_CC_SIZE, fg_label=self.FG_LABEL)
            for cc in ccs:
                bbox = cc['bbox']
                bbox_vol = 1
                for (s, e) in bbox:
                    bbox_vol *= max(e - s, 1)
                if bbox_vol > 0:
                    fill_ratios.append(cc['size'] / bbox_vol)

        if fill_ratios:
            fill_ratios = np.array(fill_ratios)
            self.rho_min = float(np.percentile(fill_ratios, 10))
            self.rho_max = float(np.percentile(fill_ratios, 95))
            self.print_to_log_file(
                f"  rho bounds: [{self.rho_min:.3f}, {self.rho_max:.3f}]")

        # Save calibration results
        calib_path = os.path.join(self.output_folder, 'calibration_prefit.json')
        calib_results = {
            'version': 'v5',
            'neg_mode': self.neg_mode,
            'pi_hat': self.pi_hat,
            'mu': mu,
            's0': s0,
            'linear_a': linear_a,
            'linear_b': linear_b,
            'rho_min': self.rho_min,
            'rho_max': self.rho_max,
            'steepness': self.steepness,
            **stats,
        }
        with open(calib_path, 'w') as f:
            json.dump(calib_results, f, indent=2)

    # ── Lambda schedule ─────────────────────────────────────────────────

    def _get_lambda_box(self):
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
        self._epoch_box_diags = []
        self.print_to_log_file(f"  lambda_box = {self.lambda_box:.4f}")

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
        """Training step for box-supervised data (v5)."""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target_full = target[0].to(self.device, non_blocking=True)
        else:
            target_full = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # g_theta function for propensity queries
        if self.g_theta is not None:
            def g_func(log_sizes):
                return predict_propensity(
                    self.g_theta, log_sizes, self.g_theta_s0)
        else:
            def g_func(log_sizes):
                return np.ones_like(log_sizes)

        # Channel-neg enabled after channel_neg_start epoch
        enable_neg = (self.current_epoch >= self.channel_neg_start
                      and self.neg_mode != 'none')

        with (autocast(self.device.type, enabled=True)
              if self.device.type == 'cuda' else dummy_context()):
            output = self.network(data)
            out_full = output[0] if isinstance(output, list) else output

            l_box, diag = compute_batch_box_loss_v5(
                out_full, target_full,
                neg_mode=self.neg_mode,
                g_theta_func=g_func,
                s0=self.g_theta_s0 or 0.0,
                linear_a=self.linear_a,
                linear_b=self.linear_b,
                d_margin=self.d_margin,
                w_max=self.w_max,
                ipw_mode=('uniform' if self.neg_mode == 'uniform'
                          else 'channel'),
                min_cc_size=self.MIN_CC_SIZE,
                fg_label=self.FG_LABEL,
                kappa=self.kappa,
                rho_min=self.rho_min,
                rho_max=self.rho_max,
                alpha_upper=self.alpha_upper,
                beta_fill=self.beta_fill,
                gamma_neg=self.gamma_neg,
                tau_low=self.tau_low,
                tau_high=self.tau_high,
                alpha_min=self.alpha_min,
                enable_neg=enable_neg,
            )

            l = self.lambda_box * l_box

        if diag.get('n_samples_with_boxes', 0) > 0:
            self._epoch_box_diags.append(diag)

        if not l.requires_grad:
            return {'loss': 0.0}

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

    # ── Diagnostics & fallback ──────────────────────────────────────────

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.current_epoch in self.DIAG_EPOCHS:
            self._log_diagnostics()

        # Low-coverage fallback at COVERAGE_FALLBACK_EPOCH
        if (self.current_epoch == self.COVERAGE_FALLBACK_EPOCH
                and not self._fallback_triggered):
            self._check_coverage_fallback()

    def _check_coverage_fallback(self):
        """If coverage_ratio < threshold, reduce d_safe."""
        if not self._epoch_box_diags:
            return
        cov_vals = [d.get('coverage_ratio_mean', 0)
                    for d in self._epoch_box_diags
                    if 'coverage_ratio_mean' in d]
        if not cov_vals:
            return
        mean_coverage = np.mean(cov_vals)
        if mean_coverage < self.COVERAGE_FALLBACK_THRESHOLD:
            old_d = self.d_margin
            self.d_margin = self.COVERAGE_FALLBACK_D_SAFE
            self._fallback_triggered = True
            self.print_to_log_file(
                f"  [FALLBACK] coverage_ratio={mean_coverage:.4f} < "
                f"{self.COVERAGE_FALLBACK_THRESHOLD}. "
                f"d_safe: {old_d} -> {self.d_margin}")

    def _log_diagnostics(self):
        """Log v5 diagnostics at checkpoint epochs."""
        diag = {
            'epoch': self.current_epoch,
            'lambda_box': self.lambda_box,
            'neg_mode': self.neg_mode,
            'd_margin': self.d_margin,
            'fallback_triggered': self._fallback_triggered,
        }

        if self._epoch_box_diags:
            for key in ('coverage_ratio_mean', 'nascent_ratio_mean',
                        'mean_alpha_mean', 'tight_loss_mean',
                        'fill_loss_mean', 'neg_loss_mean'):
                vals = [d[key] for d in self._epoch_box_diags if key in d]
                if vals:
                    diag[key] = float(np.mean(vals))
            diag['n_box_steps'] = len(self._epoch_box_diags)
            diag['total_boxes'] = sum(
                d.get('total_boxes', 0) for d in self._epoch_box_diags)
            diag['n_ccs_total'] = sum(
                d.get('n_ccs_total', 0) for d in self._epoch_box_diags)

        self.diag_log.append(diag)
        diag_path = os.path.join(self.output_folder,
                                  'latentmask_diagnostics.json')
        with open(diag_path, 'w') as f:
            json.dump(self.diag_log, f, indent=2)

        self.print_to_log_file(
            f"  [v5 diag] epoch={self.current_epoch}, "
            f"coverage={diag.get('coverage_ratio_mean', 'N/A')}, "
            f"nascent={diag.get('nascent_ratio_mean', 'N/A')}, "
            f"mean_alpha={diag.get('mean_alpha_mean', 'N/A')}")

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
            'version': 'v5',
            'neg_mode': self.neg_mode,
            'steepness': self.steepness,
            'pixel_fraction': self.pixel_fraction,
            'warmup_epochs': self.warmup_epochs,
            'ramp_epochs': self.ramp_epochs,
            'channel_neg_start': self.channel_neg_start,
            'lambda_box_max': self.lambda_box_max,
            'w_max': self.w_max,
            'd_margin': self.d_margin,
            'pi_hat': self.pi_hat,
            'n_pixel_keys': len(self.pixel_keys),
            'n_box_keys': len(self.box_keys),
            'kappa': self.kappa,
            'rho_min': self.rho_min,
            'rho_max': self.rho_max,
            'beta_fill': self.beta_fill,
            'gamma_neg': self.gamma_neg,
            'alpha_upper': self.alpha_upper,
            'tau_low': self.tau_low,
            'tau_high': self.tau_high,
            'alpha_min': self.alpha_min,
            'linear_a': self.linear_a,
            'linear_b': self.linear_b,
            'fallback_triggered': self._fallback_triggered,
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        super().on_train_end()
