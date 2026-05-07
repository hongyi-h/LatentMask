"""LatentMask Trainer v6.1: subclass of nnUNetTrainer for box-supervised segmentation.

v6.1 contract (set by post-Codex review, 2026-04-25):
  - Protocol is UNKNOWN to the trainer. No in-training g_θ re-fit.
  - Calibration (g_θ, linear_a, linear_b, ρ_min, ρ_max, s0, μ) is pre-fitted
    by `run_calibration_cv.py` and pickled to
    `{LM_BOX_ANNOTATIONS_DIR}/_calibration_fold{fold}.pkl`.
  - Trainer loads that artifact at `on_train_start`. If missing, we fail
    loudly — silent fallbacks to simulated channels caused the v5
    leakage-story problem.

Configuration via `neg_mode` (LM_NEG_MODE):
  - 'none':     C1 pixel-only baseline (no box loss)
  - 'uniform':  C2 scaffold + uniform neg (α ≡ 1.0)
  - 'constant': C2.5 scaffold + constant α (LM_CONSTANT_ALPHA, default 0.5)
  - 'linear':   C3 scaffold + linear neg (α = a + b·log m)
  - 'channel':  C4 scaffold + g_θ neg (isotonic, our method)
  - 'inverted': C4-inv directionality control (α = 1 − g_θ + α_min)
"""
import hashlib
import json
import os
import pickle
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

from latentmask.losses.bag_pu_loss import compute_batch_box_loss_v6
from latentmask.calibration.isotonic_fit import predict_propensity
from latentmask.utils.cc_extraction import extract_connected_components
from scipy import ndimage


VALID_NEG_MODES = {'none', 'uniform', 'constant', 'linear',
                   'channel', 'inverted'}


class LatentMaskTrainer(nnUNetTrainer):

    # ── Configuration defaults ──────────────────────────────────────────
    NEG_MODE = 'channel'       # one of VALID_NEG_MODES
    PIXEL_FRACTION = 0.3       # fraction of training data with pixel labels
    WARMUP_EPOCHS = 50         # epochs of pixel-only training
    RAMP_EPOCHS = 50           # epochs of lambda_box ramp
    CHANNEL_NEG_START = 60     # epoch to enable channel-neg (within ramp)
    LAMBDA_BOX_MAX = 1.0       # maximum box loss weight
    W_MAX = 10.0               # IPW weight clipping
    D_MARGIN = 5               # safe zone margin (d_safe)
    MIN_CC_SIZE = 10           # minimum CC size
    FG_LABEL = 2               # foreground label (2=tumor for LiTS; override via LM_FG_LABEL)
    DIAG_EPOCHS = [50, 100, 150, 200, 250, 300]

    # v5/v6 loss hyperparameters
    KAPPA = 1.0                # tightness threshold
    BETA_FILL = 1.0            # L_fill weight
    GAMMA_NEG = 1.0            # L_channel_neg weight
    ALPHA_UPPER = 1.0          # upper fill violation weight
    TAU_LOW = 0.3              # CC extraction low threshold
    TAU_HIGH = 0.5             # CC extraction high threshold
    ALPHA_MIN = 0.05           # minimum alpha for nascent CCs
    CONSTANT_ALPHA = 0.5       # α for C2.5 ('constant')

    # Low-coverage fallback
    COVERAGE_FALLBACK_EPOCH = 100
    COVERAGE_FALLBACK_THRESHOLD = 0.10
    COVERAGE_FALLBACK_D_SAFE = 3

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = int(os.environ.get('LM_NUM_EPOCHS', 300))

        # Read config from environment
        self.neg_mode = os.environ.get('LM_NEG_MODE', self.NEG_MODE)
        if self.neg_mode not in VALID_NEG_MODES:
            raise ValueError(
                f"Invalid LM_NEG_MODE={self.neg_mode!r}. "
                f"Must be one of {sorted(VALID_NEG_MODES)}.")

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
        self.constant_alpha = float(os.environ.get('LM_CONSTANT_ALPHA',
                                                    self.CONSTANT_ALPHA))
        self.fg_label = int(os.environ.get('LM_FG_LABEL', self.FG_LABEL))

        # Calibration state (populated by _load_calibration_artifact)
        self.g_theta = None
        self.s0 = 0.0
        self.linear_a = 0.0
        self.linear_b = 0.1
        self.mu = 0.0
        self.rho_min = 0.15
        self.rho_max = 0.85
        self.pi_hat = None
        self.lambda_box = 0.0
        self.pixel_keys = []
        self.box_keys = []
        self.dataloader_box = None
        self.diag_log = []
        self._epoch_box_diags = []
        self._fallback_triggered = False
        self._calibration_meta = {}

        self.box_annotations_dir = os.environ.get('LM_BOX_ANNOTATIONS_DIR', '')

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
            f"LatentMask v6.1 split: {len(self.pixel_keys)} pixel, "
            f"{len(self.box_keys)} box, neg_mode={self.neg_mode}, "
            f"fg_label={self.fg_label}"
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

        # Box DataLoader (v6: loads box_seg instead of GT seg)
        if len(self.box_keys) > 0:
            box_dataset_raw = self.dataset_class(
                self.preprocessed_dataset_folder, self.box_keys,
                folder_with_segs_from_previous_stage=
                self.folder_with_segs_from_previous_stage,
            )
            if self.box_annotations_dir:
                from latentmask.data.box_seg_dataset import BoxSegDatasetWrapper
                box_seg_dir = os.path.join(self.box_annotations_dir,
                                           'box_segmentations')
                box_dataset = BoxSegDatasetWrapper(box_dataset_raw, box_seg_dir)
                self.print_to_log_file(
                    f"  Box dataloader: using box_seg from {box_seg_dir}")
            else:
                raise RuntimeError(
                    "LM_BOX_ANNOTATIONS_DIR must be set for box-supervised "
                    "training — loading GT seg would leak labels.")

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

    # ── Pre-training calibration (v6.1: LOAD pre-fitted artifact) ──────

    def on_train_start(self):
        super().on_train_start()
        self._load_calibration_artifact()

    def _load_calibration_artifact(self):
        """Load the pickled calibration produced by run_calibration_cv.py.

        Why load instead of re-fit: the v6.1 contract states the method
        treats the protocol as unknown. Re-fitting inside the trainer
        (v5 behavior, using make_channel_func) would (a) require the
        protocol's functional form and (b) produce a different g_θ than
        what gets paper-reported in M1. We load the exact object instead.
        """
        self.print_to_log_file("Loading calibration artifact (v6.1)...")

        # 'uniform' needs no g_θ; 'constant' needs α only; 'none' is pixel-only.
        # The others require a loaded artifact.
        needs_artifact = self.neg_mode in {'linear', 'channel', 'inverted'}
        artifact_path = os.path.join(
            self.box_annotations_dir or '',
            f'_calibration_fold{self.fold}.pkl')

        if not os.path.isfile(artifact_path):
            if needs_artifact:
                raise FileNotFoundError(
                    f"Calibration artifact not found: {artifact_path}. "
                    f"Run `python -m latentmask.scripts.run_calibration_cv "
                    f"--dataset_dir ... --box_annotations_dir "
                    f"{self.box_annotations_dir} --protocol ... "
                    f"--fold {self.fold} --fg_label {self.fg_label}` first.")
            else:
                self.print_to_log_file(
                    f"  neg_mode={self.neg_mode} does not require g_θ; "
                    f"skipping artifact load.")
                return

        with open(artifact_path, 'rb') as f:
            art = pickle.load(f)

        # Identity invariants. Mismatch on any = refuse to train
        # (Codex finding 2026-05-06: fold + fg_label was too thin).
        if art.get('fold') != self.fold:
            raise RuntimeError(
                f"Calibration fold ({art.get('fold')}) != "
                f"training fold ({self.fold})")
        if art.get('fg_label') != self.fg_label:
            raise RuntimeError(
                f"Calibration fg_label ({art.get('fg_label')}) != "
                f"training fg_label ({self.fg_label})")

        # Pin the actual calibration scans. pixel_keys is the trainer-side
        # split (same RNG seed + pixel_fraction as the calibration script);
        # if the two disagree, the g_θ was fit on a different subset than
        # the trainer thinks, and the M1 ECE no longer applies.
        art_hash = art.get('pixel_keys_hash')
        if art_hash is not None:
            train_hash = hashlib.sha256(
                '\n'.join(sorted(self.pixel_keys)).encode()).hexdigest()[:16]
            if art_hash != train_hash:
                raise RuntimeError(
                    f"Calibration pixel_keys_hash ({art_hash}) != "
                    f"training pixel_keys_hash ({train_hash}). "
                    f"Re-run run_calibration_cv.py with matching "
                    f"--pixel_fraction / --fold.")

        self.g_theta = art['g_theta']  # dict with x/y thresholds (version-stable)
        self.s0 = float(art['s0'])
        self.linear_a = float(art['linear_a'])
        self.linear_b = float(art['linear_b'])
        self.mu = float(art['mu'])
        self.rho_min = float(art['rho_min'])
        self.rho_max = float(art['rho_max'])
        self._calibration_meta = {
            'protocol': art.get('protocol'),
            'cv_max_ece': art.get('cv_max_ece'),
            'ece_isotonic_full': art.get('ece_isotonic_full'),
            'gate_pass': art.get('gate_pass'),
            'gate_threshold': art.get('gate_threshold'),
            'n_total_ccs': art.get('n_total_ccs'),
            'artifact_timestamp': art.get('timestamp'),
            'artifact_format_version': art.get('artifact_format_version'),
            'pixel_keys_hash': art.get('pixel_keys_hash'),
            'match_params_hash': art.get('match_params_hash'),
        }

        self.print_to_log_file(
            f"  loaded protocol={self._calibration_meta.get('protocol')}, "
            f"fold={self.fold}, fg_label={self.fg_label}, "
            f"CCs={self._calibration_meta.get('n_total_ccs')}, "
            f"cv_max_ece={self._calibration_meta.get('cv_max_ece')}")
        self.print_to_log_file(
            f"  s0={self.s0:.3f}, mu={self.mu:.3f}, "
            f"linear a={self.linear_a:.4f} b={self.linear_b:.4f}, "
            f"rho=[{self.rho_min:.3f}, {self.rho_max:.3f}]")

        # pi_hat is only used by legacy paths; not required in v6.1.
        # Persist a copy of the loaded artifact in the run folder for audit.
        audit_path = os.path.join(
            self.output_folder, 'calibration_loaded.json')
        audit = {k: v for k, v in self._calibration_meta.items()}
        audit.update({
            'fold': self.fold,
            'fg_label': self.fg_label,
            's0': self.s0, 'mu': self.mu,
            'linear_a': self.linear_a, 'linear_b': self.linear_b,
            'rho_min': self.rho_min, 'rho_max': self.rho_max,
            'artifact_path': artifact_path,
        })
        with open(audit_path, 'w') as f:
            json.dump(audit, f, indent=2)

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
        """Training step for box-supervised data (v6: no label leakage).

        The box dataloader loads box_seg (rectangular box regions) instead of
        GT seg. box_seg goes through the same crop + augment pipeline as data.
        We extract box bounding boxes from the augmented box_seg via find_objects.
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            box_seg_batch = target[0].cpu().numpy()
        else:
            box_seg_batch = target.cpu().numpy()

        # Extract boxes from augmented box_seg for each sample
        B = data.shape[0]
        box_metadata_list = []
        for b in range(B):
            box_seg_np = box_seg_batch[b, 0].astype(np.int32)
            unique_ids = np.unique(box_seg_np)
            unique_ids = unique_ids[unique_ids > 0]

            boxes = []
            if len(unique_ids) > 0:
                from scipy.ndimage import find_objects
                slices = find_objects(box_seg_np)
                for s in slices:
                    if s is not None:
                        bbox = tuple((sl.start, sl.stop) for sl in s)
                        boxes.append({'bbox': bbox})
            box_metadata_list.append(boxes)

        self.optimizer.zero_grad(set_to_none=True)

        # g_theta function for propensity queries (loaded from artifact)
        if self.g_theta is not None:
            def g_func(log_sizes):
                return predict_propensity(self.g_theta, log_sizes, self.s0)
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

            l_box, diag = compute_batch_box_loss_v6(
                out_full, box_metadata_list,
                neg_mode=self.neg_mode,
                g_theta_func=g_func,
                s0=self.s0,
                linear_a=self.linear_a,
                linear_b=self.linear_b,
                constant_alpha=self.constant_alpha,
                d_margin=self.d_margin,
                w_max=self.w_max,
                ipw_mode=('uniform' if self.neg_mode == 'uniform'
                          else 'channel'),
                min_cc_size=self.MIN_CC_SIZE,
                fg_label=self.fg_label,
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
        """Log v6 diagnostics at checkpoint epochs."""
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
            f"  [v6 diag] epoch={self.current_epoch}, "
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
            'version': 'v6.1',
            'neg_mode': self.neg_mode,
            'fg_label': self.fg_label,
            'pixel_fraction': self.pixel_fraction,
            'warmup_epochs': self.warmup_epochs,
            'ramp_epochs': self.ramp_epochs,
            'channel_neg_start': self.channel_neg_start,
            'lambda_box_max': self.lambda_box_max,
            'w_max': self.w_max,
            'd_margin': self.d_margin,
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
            'constant_alpha': self.constant_alpha,
            'linear_a': self.linear_a,
            'linear_b': self.linear_b,
            's0': self.s0,
            'mu': self.mu,
            'fallback_triggered': self._fallback_triggered,
            'calibration_meta': self._calibration_meta,
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        super().on_train_end()
