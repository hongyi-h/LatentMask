"""
LatentMask Trainer — subclasses nnUNetTrainer.

Integrates:
  - Propensity Network (PropNet): domain-agnostic, encoder features only
  - Multi-granularity data loading (pixel / box / image)
  - Propensity-corrected PU losses
  - EMA teacher for Stage 3 refinement
  - Three-stage training schedule

Usage via nnUNet CLI:
    nnUNetv2_train DATASET CONFIG FOLD -tr LatentMaskTrainer
"""

from __future__ import annotations

import os
import itertools
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.helpers import dummy_context

from latentmask.modules.propnet import PropensityNetwork, ConstantPropensity
from latentmask.losses.combined import LatentMaskLoss
from latentmask.trainer.ema import EMATeacher
from latentmask.utils.synthetic_missingness import SyntheticMissingnessGenerator


class LatentMaskTrainer(nnUNetTrainer):
    """
    nnUNetTrainer with LatentMask PU learning extensions.

    Three-stage schedule:
      Stage 1 (epochs 0-49): Warm-up on pixel-only data + PropNet pre-training
      Stage 2 (epochs 50-299): Joint PU training with all granularities
      Stage 3 (epochs 300-399): Propensity-weighted EMA teacher refinement

    Configurable via --c flag:
      propnet_mode=learned|uniform  (default: learned)
      use_smoothness=true|false     (default: true)
      use_ema_refinement=true|false (default: true)
      synthetic_pattern=all|scale_only|boundary_only|component_only|scale_boundary|uniform
      use_vesselness_hint=true|false (default: false, PE-specific optional)
    """

    # ─── Stage boundaries ──────────────────────────────────────────────
    STAGE1_END = 50
    STAGE2_END = 300
    STAGE3_END = 400

    # ─── Loss lambdas ─────────────────────────────────────────────────
    LAMBDA_PIX = 1.0
    LAMBDA_BOX = 1.0
    LAMBDA_IMG = 0.5
    LAMBDA_PROP = 0.5
    LAMBDA_SMOOTH = 0.1
    LAMBDA_REF = 0.4

    # ─── PU hyperparameters ────────────────────────────────────────────
    PI_BASE = 0.15
    EMA_DECAY = 0.999

    # ─── Granularity ratio per iteration cycle ─────────────────────────
    GRAN_SCHEDULE = ["pixel", "pixel", "box", "image"]

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = self.STAGE3_END
        self.save_every = 50

        # Will be set in initialize()
        self.propnet = None
        self.ema_teacher = None
        self.latentmask_loss = None
        self.missingness_gen = None

        # Extra data loaders (box, image) — set in on_train_start
        self.dataloader_box = None
        self.dataloader_image = None
        self.box_iter = None
        self.image_iter = None

        # Paths to auxiliary manifests (set via env vars or config)
        self.rspect_manifest = os.environ.get("LATENTMASK_RSPECT_MANIFEST", "")
        self.aug_rspect_manifest = os.environ.get("LATENTMASK_AUG_RSPECT_MANIFEST", "")

        # Parse --c overrides
        self._propnet_mode = "learned"
        self._use_smoothness = True
        self._use_ema_refinement = True
        self._synthetic_pattern = "all"
        self._use_vesselness_hint = False

    def set_custom_config(self, config_str: str):
        """Parse comma-separated key=value overrides from --c flag."""
        for pair in config_str.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue
            key, val = pair.split("=", 1)
            key, val = key.strip(), val.strip()
            if key == "propnet_mode":
                self._propnet_mode = val
            elif key == "use_smoothness":
                self._use_smoothness = val.lower() == "true"
            elif key == "use_ema_refinement":
                self._use_ema_refinement = val.lower() == "true"
            elif key == "synthetic_pattern":
                self._synthetic_pattern = val
            elif key == "use_vesselness_hint":
                self._use_vesselness_hint = val.lower() == "true"

    @property
    def current_stage(self) -> int:
        if self.current_epoch < self.STAGE1_END:
            return 1
        elif self.current_epoch < self.STAGE2_END:
            return 2
        else:
            return 3 if self._use_ema_refinement else 2

    # ─── Override: initialize ──────────────────────────────────────────

    def initialize(self):
        """Build network + PropNet + loss + optimizer."""
        if not self.was_initialized:
            super().initialize()

            # Determine PropNet input channels from encoder stage
            propnet_in_channels = self._get_encoder_stage3_channels()
            if self._use_vesselness_hint:
                propnet_in_channels += 1  # optional vesselness concatenation

            if self._propnet_mode == "uniform":
                self.propnet = ConstantPropensity(value=0.5).to(self.device)
            else:
                self.propnet = PropensityNetwork(
                    in_channels=propnet_in_channels,
                    hidden_channels=(64, 32),
                    epsilon=0.01,
                    spatial_dims=3,
                ).to(self.device)

            self.latentmask_loss = LatentMaskLoss(
                pi_base=self.PI_BASE,
                lambda_pix=self.LAMBDA_PIX,
                lambda_box=self.LAMBDA_BOX,
                lambda_img=self.LAMBDA_IMG,
                lambda_prop=self.LAMBDA_PROP,
                lambda_smooth=self.LAMBDA_SMOOTH,
                lambda_ref=self.LAMBDA_REF,
            )

            self.missingness_gen = SyntheticMissingnessGenerator(
                pattern=self._synthetic_pattern,
                spatial_dims=3,
            )

            if not self._use_ema_refinement:
                self.num_epochs = self.STAGE2_END

            # Re-create optimizer to include PropNet parameters
            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            self.print_to_log_file(
                f"LatentMask initialized: PropNet mode={self._propnet_mode}, "
                f"params={sum(p.numel() for p in self.propnet.parameters())}, "
                f"pattern={self._synthetic_pattern}, "
                f"smoothness={self._use_smoothness}, "
                f"ema_ref={self._use_ema_refinement}"
            )

    def _get_encoder_stage3_channels(self) -> int:
        """Probe the network to find the number of channels at encoder stage 3."""
        mod = self.network.module if hasattr(self.network, "module") else self.network
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod

        try:
            encoder = mod.encoder
            n_stages = len(encoder.stages)
            target_stage = min(3, n_stages - 1)
            stage = encoder.stages[target_stage]
            if hasattr(stage, "blocks"):
                last_block = stage.blocks[-1]
            else:
                last_block = list(stage.children())[-1]

            for m in reversed(list(last_block.modules())):
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    return m.out_channels

            return 256
        except Exception:
            self.print_to_log_file(
                "WARNING: Could not probe encoder stage 3 channels, defaulting to 256"
            )
            return 256

    # ─── Override: configure_optimizers ─────────────────────────────────

    def configure_optimizers(self):
        """Include PropNet parameters in the optimizer."""
        params = list(self.network.parameters())
        if self.propnet is not None:
            params += list(self.propnet.parameters())

        optimizer = torch.optim.SGD(
            params,
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    # ─── Override: on_train_start ──────────────────────────────────────

    def on_train_start(self):
        super().on_train_start()

        # Initialize EMA teacher
        self.ema_teacher = EMATeacher(self.network, decay=self.EMA_DECAY)
        self.ema_teacher.to(self.device)

        # Set up auxiliary data loaders for box and image data
        self._setup_auxiliary_dataloaders()

        self.print_to_log_file(
            f"LatentMask training: {self.num_epochs} epochs, "
            f"stages at {self.STAGE1_END}/{self.STAGE2_END}/{self.STAGE3_END}"
        )

    def _setup_auxiliary_dataloaders(self):
        """Set up DataLoaders for box-level and image-level data."""
        from latentmask.data.datasets import BoxDataset, ImageDataset
        from torch.utils.data import DataLoader

        patch_size = tuple(self.configuration_manager.patch_size)

        # Box-level loader (Augmented RSPECT)
        if self.aug_rspect_manifest and os.path.isfile(self.aug_rspect_manifest):
            box_ds = BoxDataset(
                manifest_csv=self.aug_rspect_manifest,
                patch_size=patch_size,
            )
            self.dataloader_box = DataLoader(
                box_ds,
                batch_size=1,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True,
            )
            self.print_to_log_file(
                f"Box dataloader: {len(box_ds)} studies from {self.aug_rspect_manifest}"
            )
        else:
            self.dataloader_box = None
            self.print_to_log_file("WARNING: No box-level manifest set. Skipping box data.")

        # Image-level loader (RSPECT)
        if self.rspect_manifest and os.path.isfile(self.rspect_manifest):
            img_ds = ImageDataset(
                manifest_csv=self.rspect_manifest,
                patch_size=patch_size,
            )
            self.dataloader_image = DataLoader(
                img_ds,
                batch_size=2,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True,
            )
            self.print_to_log_file(
                f"Image dataloader: {len(img_ds)} studies from {self.rspect_manifest}"
            )
        else:
            self.dataloader_image = None
            self.print_to_log_file("WARNING: No image-level manifest set. Skipping image data.")

        # Create infinite iterators
        self.box_iter = (
            itertools.cycle(self.dataloader_box) if self.dataloader_box else None
        )
        self.image_iter = (
            itertools.cycle(self.dataloader_image)
            if self.dataloader_image
            else None
        )

    # ─── Override: on_train_epoch_start ────────────────────────────────

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        stage = self.current_stage
        self.print_to_log_file(f"  LatentMask Stage {stage}")

        if self.propnet is not None:
            self.propnet.train()

    # ─── Override: train_step ──────────────────────────────────────────

    def train_step(self, batch: dict) -> dict:
        """
        LatentMask training step.

        In Stage 1: only pixel batches.
        In Stage 2+: cycles through pixel/box/image batches.
        """
        stage = self.current_stage

        if stage == 1:
            return self._train_step_pixel(batch, stage)
        else:
            if not hasattr(self, "_gran_counter"):
                self._gran_counter = 0

            gran = self.GRAN_SCHEDULE[self._gran_counter % len(self.GRAN_SCHEDULE)]
            self._gran_counter += 1

            if gran == "pixel":
                return self._train_step_pixel(batch, stage)
            elif gran == "box" and self.box_iter is not None:
                box_batch = next(self.box_iter)
                return self._train_step_box(box_batch, stage)
            elif gran == "image" and self.image_iter is not None:
                img_batch = next(self.image_iter)
                return self._train_step_image(img_batch, stage)
            else:
                return self._train_step_pixel(batch, stage)

    def _get_propensity(self, encoder_feats: torch.Tensor, data: torch.Tensor | None = None) -> torch.Tensor:
        """Compute propensity from encoder features (+ optional vesselness hint)."""
        if self._use_vesselness_hint and data is not None:
            from latentmask.utils.vesselness import compute_vesselness_tensor
            vesselness = compute_vesselness_tensor(data)
            if vesselness.shape[2:] != encoder_feats.shape[2:]:
                vesselness = F.interpolate(
                    vesselness, size=encoder_feats.shape[2:],
                    mode="trilinear", align_corners=False,
                )
            feats = torch.cat([encoder_feats, vesselness], dim=1)
            return self.propnet(feats)
        return self.propnet(encoder_feats)

    def _train_step_pixel(self, batch: dict, stage: int) -> dict:
        """Train step for pixel-labeled data."""
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)

            if self.enable_deep_supervision and isinstance(output, (list, tuple)):
                pred_hr = output[0]
            else:
                pred_hr = output

            # Standard nnUNet loss for deep supervision
            l_nnunet = self.loss(output, target)

            # Get encoder features for PropNet
            encoder_feats = self._get_encoder_features(data)
            propensity = self._get_propensity(encoder_feats, data)

            pred_prob = torch.sigmoid(pred_hr)

            if isinstance(target, list):
                target_hr = target[0]
            else:
                target_hr = target

            if pred_prob.shape != target_hr.shape:
                target_hr = F.interpolate(
                    target_hr.float(), size=pred_prob.shape[2:], mode="nearest"
                )

            # Resize propensity to match pred spatial dims
            if propensity.shape[2:] != pred_prob.shape[2:]:
                propensity = F.interpolate(
                    propensity, size=pred_prob.shape[2:],
                    mode="trilinear", align_corners=False,
                )

            # Generate synthetic missingness for PropNet training
            synth_propensity = None
            positive_mask = None
            if target_hr.sum() > 0:
                _, synth_propensity = self.missingness_gen.generate_batch(target_hr)
                positive_mask = (target_hr > 0).float()

            # Compute LatentMask losses
            lm_losses = self.latentmask_loss(
                pred=pred_prob,
                propensity=propensity,
                batch_type="pixel",
                stage=stage,
                pixel_target=target_hr,
                synthetic_propensity=synth_propensity,
                positive_mask=positive_mask,
                teacher_pred=self._get_teacher_pred(data) if stage >= 3 else None,
                use_smoothness=self._use_smoothness,
            )

            total_loss = l_nnunet + lm_losses["total"]

        self._backward_and_step(total_loss)

        if stage >= 2:
            self.ema_teacher.update(self.network)

        return {"loss": total_loss.detach().cpu().numpy()}

    def _train_step_box(self, batch: dict, stage: int) -> dict:
        """Train step for box-labeled data."""
        data = batch["data"].to(self.device, non_blocking=True)
        box_mask = batch["box_mask"].to(self.device, non_blocking=True)
        box_target = batch["target"].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)

            if self.enable_deep_supervision and isinstance(output, (list, tuple)):
                pred_hr = output[0]
            else:
                pred_hr = output

            pred_prob = torch.sigmoid(pred_hr)

            encoder_feats = self._get_encoder_features(data)
            propensity = self._get_propensity(encoder_feats, data)

            if propensity.shape[2:] != pred_prob.shape[2:]:
                propensity = F.interpolate(
                    propensity, size=pred_prob.shape[2:],
                    mode="trilinear", align_corners=False,
                )

            lm_losses = self.latentmask_loss(
                pred=pred_prob,
                propensity=propensity,
                batch_type="box",
                stage=stage,
                box_mask=box_mask,
                box_target=box_target,
                teacher_pred=self._get_teacher_pred(data) if stage >= 3 else None,
                use_smoothness=self._use_smoothness,
            )

            total_loss = lm_losses["total"]

        self._backward_and_step(total_loss)
        self.ema_teacher.update(self.network)

        return {"loss": total_loss.detach().cpu().numpy()}

    def _train_step_image(self, batch: dict, stage: int) -> dict:
        """Train step for image-labeled data."""
        data = batch["data"].to(self.device, non_blocking=True)
        image_label = batch["image_label"].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)

            if self.enable_deep_supervision and isinstance(output, (list, tuple)):
                pred_hr = output[0]
            else:
                pred_hr = output

            pred_prob = torch.sigmoid(pred_hr)

            encoder_feats = self._get_encoder_features(data)
            propensity = self._get_propensity(encoder_feats, data)

            if propensity.shape[2:] != pred_prob.shape[2:]:
                propensity = F.interpolate(
                    propensity, size=pred_prob.shape[2:],
                    mode="trilinear", align_corners=False,
                )

            lm_losses = self.latentmask_loss(
                pred=pred_prob,
                propensity=propensity,
                batch_type="image",
                stage=stage,
                image_label=image_label,
                teacher_pred=self._get_teacher_pred(data) if stage >= 3 else None,
                use_smoothness=self._use_smoothness,
            )

            total_loss = lm_losses["total"]

        self._backward_and_step(total_loss)
        self.ema_teacher.update(self.network)

        return {"loss": total_loss.detach().cpu().numpy()}

    # ─── Helpers ───────────────────────────────────────────────────────

    def _backward_and_step(self, loss: torch.Tensor):
        """Backward pass with gradient scaling and clipping."""
        all_params = list(self.network.parameters()) + list(self.propnet.parameters())
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 12)
            self.optimizer.step()

    def _get_encoder_features(self, data: torch.Tensor) -> torch.Tensor:
        """Extract encoder stage 3 features via forward hook."""
        mod = self.network.module if hasattr(self.network, "module") else self.network
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod

        features = {}

        def hook_fn(module, input, output):
            features["stage3"] = output

        try:
            encoder = mod.encoder
            n_stages = len(encoder.stages)
            target_stage = min(3, n_stages - 1)
            handle = encoder.stages[target_stage].register_forward_hook(hook_fn)

            with torch.no_grad():
                _ = encoder(data)

            handle.remove()
            return features["stage3"]
        except Exception:
            B = data.shape[0]
            ch = self._get_encoder_stage3_channels()
            spatial = [s // 8 for s in data.shape[2:]]
            return torch.zeros(B, ch, *spatial, device=data.device)

    @torch.no_grad()
    def _get_teacher_pred(self, data: torch.Tensor) -> torch.Tensor:
        """Get teacher prediction for refinement."""
        teacher_out = self.ema_teacher.forward(data)
        if isinstance(teacher_out, (list, tuple)):
            teacher_out = teacher_out[0]
        return torch.sigmoid(teacher_out).detach()

    # ─── Override: save/load checkpoint ────────────────────────────────

    def save_checkpoint(self, filename: str) -> None:
        """Save checkpoint with PropNet and EMA teacher states."""
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                mod = self.network.module if hasattr(self.network, "module") else self.network
                if hasattr(mod, "_orig_mod"):
                    mod = mod._orig_mod

                checkpoint = {
                    "network_weights": mod.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler else None,
                    "logging": self.logger.get_checkpoint(),
                    "_best_ema": self._best_ema,
                    "current_epoch": self.current_epoch,
                    "init_args": self.my_init_kwargs,
                    "trainer_name": self.__class__.__name__,
                    "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
                    # LatentMask extras
                    "propnet_state": self.propnet.state_dict() if self.propnet else None,
                    "ema_teacher_state": self.ema_teacher.state_dict() if self.ema_teacher else None,
                    "config": {
                        "propnet_mode": self._propnet_mode,
                        "use_smoothness": self._use_smoothness,
                        "use_ema_refinement": self._use_ema_refinement,
                        "synthetic_pattern": self._synthetic_pattern,
                        "use_vesselness_hint": self._use_vesselness_hint,
                    },
                }
                torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str) -> None:
        """Load checkpoint with PropNet and EMA teacher states."""
        if not self.was_initialized:
            self.initialize()

        checkpoint = torch.load(filename, map_location=self.device, weights_only=False)

        mod = self.network.module if hasattr(self.network, "module") else self.network
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod

        mod.load_state_dict(checkpoint["network_weights"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler and checkpoint.get("grad_scaler_state"):
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
        self.current_epoch = checkpoint["current_epoch"]
        self._best_ema = checkpoint.get("_best_ema")
        self.inference_allowed_mirroring_axes = checkpoint.get("inference_allowed_mirroring_axes")
        self.logger.load_checkpoint(checkpoint["logging"])

        if checkpoint.get("propnet_state") and self.propnet is not None:
            self.propnet.load_state_dict(checkpoint["propnet_state"])

        if checkpoint.get("ema_teacher_state") and self.ema_teacher is not None:
            self.ema_teacher.load_state_dict(checkpoint["ema_teacher_state"])

        self.print_to_log_file(
            f"Loaded LatentMask checkpoint from epoch {self.current_epoch}"
        )


# ─── Ablation variants ────────────────────────────────────────────────


class LatentMaskTrainer_A1_UniformPU(LatentMaskTrainer):
    """Ablation A1: PU correction with uniform propensity e=0.5."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._propnet_mode = "uniform"
        self._use_smoothness = False
        self._use_ema_refinement = False


class LatentMaskTrainer_A2_PropNetOnly(LatentMaskTrainer):
    """Ablation A2: PU + learned PropNet, no smoothness, no EMA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_smoothness = False
        self._use_ema_refinement = False


class LatentMaskTrainer_A3_NoRefine(LatentMaskTrainer):
    """Ablation A3: PU + PropNet + smoothness, no EMA teacher refinement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_ema_refinement = False


# ─── Baseline trainers ────────────────────────────────────────────────


class MixedNaiveTrainer(LatentMaskTrainer):
    """
    Baseline: Mixed naive — use all granularities with standard losses.
    No PU correction, no PropNet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._propnet_mode = "uniform"
        self._use_smoothness = False
        self._use_ema_refinement = False
        self.LAMBDA_PROP = 0.0
        self.PI_BASE = 0.0


class nnPUSegTrainer(LatentMaskTrainer):
    """
    Baseline: nnPU-Seg — PU learning with uniform propensity.
    No PropNet, no smoothness, no EMA.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._propnet_mode = "uniform"
        self._use_smoothness = False
        self._use_ema_refinement = False
        self.LAMBDA_PROP = 0.0


class MeanTeacher3DTrainer(nnUNetTrainer):
    """
    Baseline: Mean Teacher semi-supervised.
    Uses pixel data + image-level data (as unlabeled).
    EMA teacher generates pseudo-labels; consistency loss on unlabeled data.
    """

    EMA_DECAY = 0.999

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 400
        self.ema_teacher = None
        self.rspect_manifest = os.environ.get("LATENTMASK_RSPECT_MANIFEST", "")
        self.dataloader_unlabeled = None
        self.unlabeled_iter = None

    def initialize(self):
        if not self.was_initialized:
            super().initialize()

    def on_train_start(self):
        super().on_train_start()
        self.ema_teacher = EMATeacher(self.network, decay=self.EMA_DECAY)
        self.ema_teacher.to(self.device)

        # Set up unlabeled data loader from RSPECT
        if self.rspect_manifest and os.path.isfile(self.rspect_manifest):
            from latentmask.data.datasets import ImageDataset
            from torch.utils.data import DataLoader
            patch_size = tuple(self.configuration_manager.patch_size)
            ds = ImageDataset(manifest_csv=self.rspect_manifest, patch_size=patch_size)
            self.dataloader_unlabeled = DataLoader(
                ds, batch_size=2, shuffle=True, num_workers=2,
                pin_memory=True, drop_last=True,
            )
            self.unlabeled_iter = itertools.cycle(self.dataloader_unlabeled)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            l_sup = self.loss(output, target)

            # Consistency on unlabeled data
            l_consistency = torch.tensor(0.0, device=self.device)
            if self.unlabeled_iter is not None:
                unlabeled_batch = next(self.unlabeled_iter)
                u_data = unlabeled_batch["data"].to(self.device, non_blocking=True)
                u_output = self.network(u_data)
                if isinstance(u_output, (list, tuple)):
                    u_pred = torch.sigmoid(u_output[0])
                else:
                    u_pred = torch.sigmoid(u_output)
                with torch.no_grad():
                    t_out = self.ema_teacher.forward(u_data)
                    if isinstance(t_out, (list, tuple)):
                        t_pred = torch.sigmoid(t_out[0])
                    else:
                        t_pred = torch.sigmoid(t_out)
                l_consistency = F.mse_loss(u_pred, t_pred)

            total = l_sup + 0.5 * l_consistency

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        self.ema_teacher.update(self.network)
        return {"loss": total.detach().cpu().numpy()}


class CPSTrainer(nnUNetTrainer):
    """
    Baseline: Cross Pseudo Supervision.
    Two networks generate pseudo-labels for each other on unlabeled data.
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 400
        self.network2 = None
        self.optimizer2 = None
        self.rspect_manifest = os.environ.get("LATENTMASK_RSPECT_MANIFEST", "")
        self.dataloader_unlabeled = None
        self.unlabeled_iter = None

    def initialize(self):
        if not self.was_initialized:
            super().initialize()
            # Create second network (deep copy)
            self.network2 = deepcopy(self.network).to(self.device)
            self.optimizer2 = torch.optim.SGD(
                self.network2.parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )

    def on_train_start(self):
        super().on_train_start()
        if self.rspect_manifest and os.path.isfile(self.rspect_manifest):
            from latentmask.data.datasets import ImageDataset
            from torch.utils.data import DataLoader
            patch_size = tuple(self.configuration_manager.patch_size)
            ds = ImageDataset(manifest_csv=self.rspect_manifest, patch_size=patch_size)
            self.dataloader_unlabeled = DataLoader(
                ds, batch_size=2, shuffle=True, num_workers=2,
                pin_memory=True, drop_last=True,
            )
            self.unlabeled_iter = itertools.cycle(self.dataloader_unlabeled)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # --- Network 1 supervised ---
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer2.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            out1 = self.network(data)
            out2 = self.network2(data)
            l_sup1 = self.loss(out1, target)
            l_sup2 = self.loss(out2, target)

            l_cps = torch.tensor(0.0, device=self.device)
            if self.unlabeled_iter is not None:
                ub = next(self.unlabeled_iter)
                u_data = ub["data"].to(self.device, non_blocking=True)
                u_out1 = self.network(u_data)
                u_out2 = self.network2(u_data)
                if isinstance(u_out1, (list, tuple)):
                    u_pred1, u_pred2 = torch.sigmoid(u_out1[0]), torch.sigmoid(u_out2[0])
                else:
                    u_pred1, u_pred2 = torch.sigmoid(u_out1), torch.sigmoid(u_out2)

                pseudo1 = (u_pred1.detach() > 0.5).float()
                pseudo2 = (u_pred2.detach() > 0.5).float()
                l_cps = (
                    F.binary_cross_entropy(u_pred1, pseudo2)
                    + F.binary_cross_entropy(u_pred2, pseudo1)
                ) * 0.5

            total = l_sup1 + l_sup2 + l_cps

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self.grad_scaler.unscale_(self.optimizer2)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            torch.nn.utils.clip_grad_norm_(self.network2.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.step(self.optimizer2)
            self.grad_scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            torch.nn.utils.clip_grad_norm_(self.network2.parameters(), 12)
            self.optimizer.step()
            self.optimizer2.step()

        return {"loss": total.detach().cpu().numpy()}


class BoxSup3DTrainer(nnUNetTrainer):
    """
    Baseline: 3D BoxSup — bounding box supervision with GrabCut-style pseudo-masks.
    Uses box annotations from Aug-RSPECT. Inside box = positive, outside = negative.
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 400
        self.aug_rspect_manifest = os.environ.get("LATENTMASK_AUG_RSPECT_MANIFEST", "")
        self.dataloader_box = None
        self.box_iter = None

    def on_train_start(self):
        super().on_train_start()
        if self.aug_rspect_manifest and os.path.isfile(self.aug_rspect_manifest):
            from latentmask.data.datasets import BoxDataset
            from torch.utils.data import DataLoader
            patch_size = tuple(self.configuration_manager.patch_size)
            ds = BoxDataset(manifest_csv=self.aug_rspect_manifest, patch_size=patch_size)
            self.dataloader_box = DataLoader(
                ds, batch_size=1, shuffle=True, num_workers=2,
                pin_memory=True, drop_last=True,
            )
            self.box_iter = itertools.cycle(self.dataloader_box)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            l_sup = self.loss(output, target)

            # Box supervision: use box target as pseudo-mask
            l_box = torch.tensor(0.0, device=self.device)
            if self.box_iter is not None:
                bb = next(self.box_iter)
                b_data = bb["data"].to(self.device, non_blocking=True)
                b_target = bb["target"].to(self.device, non_blocking=True)
                b_output = self.network(b_data)
                if isinstance(b_output, (list, tuple)):
                    b_pred = torch.sigmoid(b_output[0])
                else:
                    b_pred = torch.sigmoid(b_output)
                if b_pred.shape != b_target.shape:
                    b_target = F.interpolate(
                        b_target.float(), size=b_pred.shape[2:], mode="nearest"
                    )
                l_box = F.binary_cross_entropy(b_pred, b_target.float())

            total = l_sup + l_box

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": total.detach().cpu().numpy()}

    def initialize(self):
        super().initialize()
        self.apn = ConstantPropensity(value=0.5).to(self.device)
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
