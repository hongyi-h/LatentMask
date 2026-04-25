# Size-Aware Negative Modulation for Box-Supervised 3D Segmentation under Controlled Incomplete-Box Protocols

**Version**: v6.1 | **Date**: 2026-04-25 | **Supersedes**: v5-final (2026-04-16)
**Target Venue**: MICCAI (primary)

---

## 1. Problem Anchor

Under incomplete-box annotation protocols, training treats missing boxes as negative evidence, suppressing real small objects in the safe zone. The smaller a lesion, the more likely its box is omitted — yet all existing methods apply uniform negative pressure outside annotated regions.

**No existing method modulates safe-zone negative supervision strength based on object size.**

**Non-goals**: NOT a new PU learning framework. NOT a new backbone/architecture. NOT general noisy-label correction. NOT a pseudo-label pipeline. NOT a model of real annotator behavior.

**Constraints**: ~3,400 A100-hours; LiTS (primary); 12 weeks; target MICCAI.

**Success**: (1) Small-lesion detection rate ≥ +5pt over uniform safe zone under size-biased protocols. (2) Gain is directional (C4 > C2.5 constant, C4 > C4-inv inverted). (3) FP/scan does not increase.

## 2. Technical Gap

**No method controls safe-zone negative supervision strength based on object size.** All existing box-supervised methods apply uniform negative pressure outside annotated regions.

| Method | Failure |
|--------|---------|
| 3D-BoxSup (2020) | Uniform safe-zone negatives. No size awareness. |
| BoxInst (CVPR 2021) | Heuristic projection, not size-derived |
| BLPSeg (TIP 2023) | Scribble bias, heuristic, no formal size model |
| MonoBox (AAAI 2025) | Uniform safe zone, no size-aware modulation |
| GeoCoBox (AAAI 2026) | Geometric constraints only, uniform negatives |
| LooBox (ACM MM 2025) | Point-guided, no size-dependent modulation |

**Gap**: No method connects object size to safe-zone negative supervision strength under incomplete-box protocols.

## 3. Framing

### What we claim

Under controlled incomplete-box protocols with size-dependent omissions, uniform outside-box negatives over-suppress small unlabeled objects. Size-aware modulation via a retention prior g_θ(size) — estimated from a small calibration subset — reduces that failure mode.

### What we do NOT claim

- We do NOT model real annotator behavior.
- g_θ is NOT an annotator behavior model. It is a **retention prior estimated from calibration data**.
- We do NOT claim generalization to arbitrary real-world weak annotation scenarios.
- We do NOT assume the omission protocol is known to the method. g_θ is estimated, not derived from the protocol definition.

### Two-term contrast (paper framing)

| Term | Definition |
|------|-----------|
| **Missingness-Agnostic** (baseline) | Observed boxes = complete supervision. Box absence = negative evidence. Uniform α ≡ 1.0. |
| **Missingness-Aware** (ours) | Observed boxes = incomplete observation. Box absence ≠ certain background. α(u) = g_θ(log m(C(u))). |

### Calibration fairness principle

The method assumes a mixed-supervision setting: a small calibration subset has both pixel masks and box annotations; the remaining scans have only boxes. All baselines have equal access to the calibration subset. C3 (linear) fits its slope from the same data. C2 (uniform) chooses not to use it. The question is whether estimating the retention pattern helps, not whether having extra data helps.

## 4. Contribution

**Dominant (single)**: Per-component size-aware negative modulation — one scalar function g_θ (≤10 parameters) estimated from a calibration subset, queried at each safe-zone connected component's predicted mass. Strong α where the protocol would have retained the box, weak α where likely dropped.

**Pre-committed framing**: If g_θ ≈ linear empirically, the contribution still holds (monotone size-aware modulation > uniform). Calibration-specific claims are reduced.

**Primary title**: Size-Aware Negative Modulation for Box-Supervised 3D Segmentation under Controlled Incomplete-Box Protocols

**Fallback title** (if g_θ ≈ linear): Monotone Negative Modulation for Box-Supervised 3D Segmentation

**Explicit non-contributions**: IPW (training stability detail), 3D tightness/fill (scaffold, not novel), isotonic regression (standard tool).

## 5. Method

### 5.1 Complexity Budget

| Component | Status | Params | Role |
|-----------|--------|--------|------|
| nnUNet v2 (3D fullres) | Reused | ~30M | Backbone |
| g_θ: 1D isotonic regression | New | ~10 | Retention prior |
| Box loss terms | New | 0 | Losses only |

Intentionally excluded: pseudo-labels, EMA, contrastive embedding, SAM, spatial monotonicity, RL.

### 5.2 Modeling Proposition: Size-Aware Negative Risk

**Assumptions:**
- **A1 (Size-dependent retention)**: Under the incomplete-box protocol, retention probability depends on component size: g_θ(log s), monotone non-decreasing.
- **A2 (Warmup sufficiency)**: After warmup, predicted components approximate true components sufficiently for g_θ to give meaningful estimates.
- **A3 (Calibration)**: g_θ is well-calibrated (validated by within-protocol ECE < 0.10).

**Proposition.** Under A1–A3, the expected negative loss at u ∈ S decomposes into:
- g_θ(log m(C(u))) · ℓ_neg(u) · P(y*=0 | observed) — **false-alarm term** (penalise)
- (1 - g_θ(log m(C(u)))) · ℓ_neg(u) · P(y*=1 | missed) — **missed-object term** (protect)

Setting α(u) = g_θ(log m(C(u))) retains only the false-alarm suppression term, protecting likely-missed components.

### 5.3 Retention Prior g_θ: Fitting Protocol

**Input**: Calibration subset (31 LiTS scans with both ground-truth masks and simulated box annotations under the target protocol).

**Protocol-unknown assumption**: The method does NOT use the protocol's functional form. It only observes (log_size, was_retained) pairs from the calibration subset and fits isotonic regression.

**Step-by-step:**
1. For each calibration scan, extract all GT connected components. Record voxel count |C_k|.
2. From the offline-generated box annotations (protocol-specific), determine which CCs were retained.
3. Construct training pairs: (x_k, y_k) = (log|C_k|, retention_status).
4. Fit: `IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip').fit(x, y)`
5. Also extract: ρ_min = P10(fill_ratio), ρ_max = P95(fill_ratio) for scaffold bounds.

**Cross-validation**: 3-fold scan-level CV. Report per-fold ECE. Gate: ECE < 0.10.

### 5.4 Channel-Modulated Negative Supervision (L_channel_neg) — Novel Component

**Step 1: Safe zone (from boxes only, no GT masks).**
S = {u : d_chebyshev(u, ∪_j B_j) > d_safe}, d_safe = 5 voxels (default).
Boxes come from pre-generated offline metadata. Safe zone never accesses GT segmentation masks.

**Step 2: Two-threshold CC extraction (every K_cc = 50 steps, detached).**
1. M_low = 𝟙[f_θ > τ_low] ∩ S, where τ_low = 0.3
2. 3D connected components: {C_1, ..., C_n} (scipy.ndimage.label, CPU, <10ms/patch)
3. Per C_k: V_high = {u ∈ C_k : f_θ(u) > τ_high}, where τ_high = 0.5
   - |V_high| > 0 → **mature**: m(C_k) = Σ_{u ∈ C_k} f_θ(u), query g_θ
   - |V_high| = 0 → **nascent** → α_k = α_min

**Step 3: Assignment rules.**
- C_k touches patch face → α_k = 1.0 (conservative: incomplete component)
- Nascent (no high-threshold voxels) → α_k = 0.05 (maximum protection)
- Interior, mature → α_k = clamp(g_θ(log m(C_k)), 0.05, 1.0)
- u ∈ S, no component → α(u) = 1.0 (uniform: no predicted structure)

**Step 4: Loss.**
L_channel_neg = mean_{u ∈ S} [ α(u) · (-log(1 - f_θ(u) + ε)) ]

**IPW weighting for scaffold**: Uses predicted mass only (no GT size). w_j = min(1/g_θ(log m̂_j), w_max), self-normalised.

### 5.5 Box Constraint Scaffold (L_scaffold)

L_scaffold = L_tight + β · L_fill. Reused components, not novel.

**L_tight**: 3D tightness — every axis-aligned slice through the box must have predicted mass ≥ κ.
**L_fill**: Filling rate interval — constrain total predicted mass in each box to [ρ_min·|B_j|, ρ_max·|B_j|].

### 5.6 Full Loss

```
L_box = L_scaffold + γ · L_channel_neg
L_total = λ_pix · (Dice + CE) + λ_box · L_box
```

### 5.7 Training Protocol

| Stage | Epochs | Data | Notes |
|---|---|---|---|
| Pre-fit | 0 | Calibration subset | g_θ (isotonic), ρ_min, ρ_max. Frozen. |
| Warm-up | 1–50 | Pixel only | Dice+CE. |
| Ramp | 51–100 | Pixel + Box | L_scaffold. L_channel_neg from epoch 60. |
| Full | 101–300 | Pixel + Box | All terms. CC every 50 steps. |

Batch cycle: [pixel, pixel, box]. No EMA. No recalibration. No pseudo-labels.

### 5.8 Inference

f_θ only. Zero overhead. g_θ not needed at inference.

### 5.9 Failure Modes

| Failure | Detection | Fallback |
|---------|-----------|----------|
| g_θ miscalibrated | ECE > 0.10 | More calibration data |
| Channel-neg → FP increase | FP/scan > C2 + 1.0 | ↓ γ, ↑ α_min |
| Nascent dominant | nascent_ratio > 0.5 | ↑ τ_low |
| Low coverage | coverage_ratio < 0.10 at epoch 100 | d_safe = 3 |
| g_θ ≈ linear | C4 ≈ C3 | Downgrade to monotone modulation paper |
| C4 ≈ C2.5 | Gain from weaker negatives, not directionality | Fundamental problem — paper dies |

## 6. Validation

### 6.1 Protocols (retention-rate-matched)

All protocols calibrated to the same expected marginal retention rate R = 0.70.

| Protocol | Drop function | Description |
|----------|--------------|-------------|
| P-uniform | drop_prob = 0.30 (constant) | Control: no size dependence |
| P-mild | steepness = 'shallow', scaled to R | Mild size-dependent omission |
| P-steep | steepness = 'steep', scaled to R | Strong size-dependent omission |

### 6.2 Configurations (7 configs)

| Config | Description |
|--------|-------------|
| C0 | Full pixel (upper bound, 104 scans) |
| C1 | Pixel-only (lower bound, 31 scans) |
| C2 | Scaffold + uniform neg (α ≡ 1.0) |
| C2.5 | Scaffold + constant neg (α ≡ 0.5, pre-specified) |
| C3 | Scaffold + linear neg (α = clamp(a + b·log m)) |
| C4 | **Full method** (α = g_θ(log m)) |
| C4-inv | Inverted g_θ (α = clamp(1 − g_θ + α_min)) |

### 6.3 Metrics (LOCKED)

- **V_small**: 500 voxels (fixed absolute threshold)
- **Lesion-level TP**: predicted CC overlaps GT lesion with Dice > 0.10
- **FP burden**: FP lesions per scan

Primary: small-lesion detection rate, FP/scan.
Secondary: Q1 per-lesion Dice, overall Dice.
Reporting: HD95, volume-stratified Dice and detection rate.

### 6.4 Claim Map

| ID | Claim | Evidence |
|----|-------|---------|
| C-P | C4 > C2 on small-lesion detection under P-steep | ≥ +5pt recall, FP/scan ≤ C2 + 0.5 |
| C-S | g_θ-isotonic ≥ linear heuristic | C4 ≥ C3 on Q1 Dice |
| C-D | Gain is directional, not just weaker negatives | C4 > C2.5 and C4 > C4-inv |
| C-R | Robust across retention-matched protocols | C4 > C2 under P-mild and P-steep |
| C-N | No spurious gains when size dependence absent | P-uniform: C4 ≈ C2 |
| AC1 | Not from safe-zone shrinkage | C4 > C2 at d_safe = {3, 5} |
| AC2 | Mechanism is active | coverage_ratio > 0.10 |
| AC3 | Box data adds value via scaffold | C2 > C1 on overall Dice |

### 6.5 Evaluation Protocol

5-fold CV (nnUNet standard) for B1 core configs (C0–C4). Single fold for ablations (C2.5, C4-inv, multi-protocol, sensitivity).

### 6.6 Mechanism Activity (appendix)

- coverage_ratio, nascent_ratio, mean_alpha over epochs
- α histogram at epoch 150 and 300
- g_θ fitting quality: 3-fold CV ECE per protocol

### 6.7 Sensitivity Analysis (appendix)

- d_safe ∈ {3, 5}, τ_low ∈ {0.2, 0.3, 0.4}, α_min ∈ {0.02, 0.05, 0.10}

## 7. Compute & Timeline

| Phase | Duration | GPU-hours |
|---|---|---|
| M0: Fix leakage + offline pipeline | 1 week | 0 |
| M1: Calibration | 0.5 week | 0 |
| M2: Baselines (C0, C1, C2 × 5 folds) | 3 weeks | ~1,125 |
| M3: Main (C3, C4 × 5 folds + C2.5, C4-inv) | 3 weeks | ~900 |
| M3b/c: Multi-protocol + transfer | 2 weeks | ~675 |
| M5: Sensitivity | 1 week | ~450 |
| M6: Polish + writing | 1 week | ~50 |
| **Total** | **~12 weeks** | **~3,400** |

## 8. Red Lines

1. No claim of "real annotator behavior" without real weak annotation evidence.
2. Box branch must NOT access target masks during training. (FIXED in v6 code.)
3. Safe zone computed from boxes only, not GT foreground. (FIXED.)
4. IPW weights from predicted mass only, not true GT size. (FIXED.)
5. g_θ is a "retention prior estimated from calibration data", not an "annotator model."
6. Primary metric is small-lesion detection rate + FP/scan, not overall Dice.
7. Checkpoint selection aligned with primary metric (post-hoc via evaluate.py).

## 9. Claim Conditioning

**If g_θ materially beats linear**: Full title, full claims.

**If g_θ ≈ linear**: "Monotone Negative Modulation for Box-Supervised 3D Segmentation." Contribution is per-component size-aware negative modulation (mechanism is novel regardless). Calibration claims reduced.

**If C4 ≈ C2.5**: Gain is from weaker negatives, not directionality. Paper does not proceed.

**If mechanism inactive (coverage_ratio ≈ 0)**: Fundamental design issue. Report honestly.

---

**Prior art explicitly addressed**: 3D-BoxSup (2020), BLPSeg (TIP 2023), BoxInst (CVPR 2021), MonoBox (AAAI 2025), GeoCoBox (AAAI 2026), LooBox (ACM MM 2025).

**Novelty argument**: One function (g_θ, ≤10 params), one modeling proposition (A1–A3), one per-component α mechanism. No prior work connects object size to safe-zone negative supervision under incomplete-box protocols. Robust: even if g_θ ≈ linear, monotone modulation > uniform.

**Limitations (stated upfront)**:
- Single dataset (LiTS). Multi-dataset validation out of budget.
- Synthetic protocols only. No real weak annotation evidence.
- No external baseline reproduction (GeoCoBox, LooBox — no public code).
