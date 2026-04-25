# LatentMask v5 review

Date: 2026-04-24

## Bottom line

Current `v5` is not a valid "real box-supervised" experiment.

The core problem is not merely that weak labels are simulated. The core problem is that the
training objective still reads hidden full masks on the box-supervised branch. That breaks the
causal claim that performance gains come from box supervision plus channel modeling.

## Confirmed findings

### F1. `g_theta` is fit to a synthetic channel, not real annotation behavior

Evidence:
- `latentmask/trainer/latentmask_trainer.py` creates `g_true = make_channel_func(...)` and passes
  it as `drop_fn` into `fit_g_theta_hungarian`.
- `latentmask/calibration/isotonic_fit.py` builds `annotation_boxes` by dropping a subset of exact
  GT-derived boxes, then matches them back to the original GT boxes.
- `latentmask/scripts/run_calibration_cv.py` explicitly documents the protocol as
  `GT CCs -> simulated box annotations -> Hungarian matching -> ...`.

Implication:
- This calibration is a self-consistency test under the simulator, not evidence that `g_theta`
  captures real annotator behavior.
- Because `annotation_boxes` are a subset of the same GT boxes, Hungarian matching is almost an
  identity map; it mostly recovers the simulator's keep flags.

### F2. The "box-only" branch still uses full GT masks online during training

Evidence:
- `latentmask/trainer/latentmask_trainer.py` splits `pixel_keys` and `box_keys`, but both loaders
  come from the same nnUNet dataset and load `target`.
- `latentmask/losses/bag_pu_loss.py` reads `target[b, 0]` on every box step.
- The same function extracts GT connected components and boxes directly from that target.

Implication:
- Using full masks to *construct* synthetic weak labels offline is acceptable for a benchmark.
- Using full masks *inside the loss at training time* is label leakage and invalidates the weak
  supervision claim.

### F3. Safe zone uses hidden GT foreground, not observable boxes

Evidence:
- `latentmask/losses/bag_pu_loss.py` calls `compute_safe_zone_mask(seg_np, ...)`.
- `latentmask/utils/cc_extraction.py` defines `compute_safe_zone_mask` from full foreground and also
  provides `compute_safe_zone_from_boxes`, which is not used.

Implication:
- The model gets background information that would not exist in a real box-only setting.
- On LiTS this is stronger than it first appears because the safe zone excludes all foreground,
  including non-target structures already present in the segmentation.

### F4. IPW/scaffold weighting also leaks hidden true lesion size

Evidence:
- `latentmask/losses/bag_pu_loss.py` stores `true_size` in each box record.
- `_compute_sample_loss_v5` prefers `true_size` when computing propensity/IPW weights.

Implication:
- Even if boxes were the intended supervision, the loss still accesses latent GT lesion volume.
- This is a separate leakage path beyond the safe-zone issue.

### F5. Validation/checkpoint selection is misaligned with the main claim

Evidence:
- `nnUNetTrainer.on_validation_epoch_end` logs `mean_fg_dice`.
- `nnUNetTrainer.on_epoch_end` saves `checkpoint_best.pth` using `ema_fg_dice`.
- `latentmask/scripts/evaluate.py` computes `q1_dice` only post hoc.

Implication:
- If the thesis is "C4 helps the smallest lesions", best-checkpoint selection is not optimized for
  the paper's main endpoint.

## Findings that are weaker than they look

### W1. "Box step does not include pixel loss"

This is not automatically a bug.

If the box branch is truly box-only, then pixel Dice/CE is unavailable and should not be used on
those samples. The real question is whether the stochastic objective

`E[L_pixel] + lambda_box * E[L_box]`

is being balanced well. The current `[pixel, pixel, box]` cycle is a design choice. It becomes a
scientific problem only after the leakage above is removed and we can measure whether box steps
help or hurt.

### W2. "LR scheduler is skipped on box steps"

This is not supported by the code.

`nnUNetTrainer.on_train_epoch_start` steps the LR scheduler once per epoch, and
`LatentMaskTrainer.on_train_epoch_start` calls `super().on_train_epoch_start()`. The scheduler is
not supposed to step inside `train_step` or `_box_train_step`.

### W3. `gamma_neg` mismatch

Real issue, but low severity for the main pipeline.

The trainer passes `gamma_neg` explicitly, so the default inconsistency mainly affects ad hoc
direct calls to the loss function.

### W4. CPU connected-components extraction

Likely a real efficiency bottleneck, but not a scientific validity issue.

Do not optimize this before fixing the label-leakage problems.

## What would make the study valid

### Path A: honest synthetic benchmark

Use this only if real box annotations are unavailable soon.

Requirements:
- Generate weak labels offline from full masks.
- Persist only observable weak supervision for box scans: annotation boxes, optional retained-box
  metadata, and nothing else.
- Train the box branch without reading full masks.
- Build safe zones from boxes only.
- Remove `true_size` access on box scans.
- Reframe claims as robustness under controlled missing-box channels, not real annotator behavior.
- Add at least one more dataset if targeting MICCAI.

This path can produce a clean paper, but the claim becomes narrower.

### Path B: real weak-annotation story

Use this if the goal is a stronger MICCAI submission.

Requirements:
- Collect a real box-annotated calibration subset, even if small.
- Fit `g_theta` on actual annotation/miss events, not simulated keep probabilities.
- Keep the training branch strictly box-observable.
- Report calibration quality on held-out real annotation data.

This is the path that actually supports the current narrative around annotation behavior.

## Immediate action list

1. Stop long training runs on the current `v5` branch.
2. Decide whether the paper is synthetic or real-annotation.
3. Refactor the box branch so that it consumes box annotations, not `target` masks.
4. Switch safe-zone construction to `compute_safe_zone_from_boxes`.
5. Remove `true_size` from box-branch loss inputs.
6. Add validation-time Q1 tracking and checkpoint selection aligned with the paper claim.
7. Only after that, tune schedule ratio, `lambda_box`, and efficiency.

## MICCAI risk assessment

As implemented today, the main risk is not weak novelty. The main risk is invalid evidence.

Recent MICCAI weak-supervision papers that were accepted either introduced a real weak annotation
source or clearly defined a concrete incomplete-annotation protocol, and multiple papers validated
on two datasets. Review text from MICCAI 2024 also shows that single-dataset evidence is explicitly
penalized for generalization concerns.
