# Novelty Check Report v3 — Channel-Modulated Box Supervision

**Date**: 2025-07-14  
**Proposal**: "Monotone Annotation-Channel Modeling for Size-Aware Box-Supervised Segmentation"  
**Reviewer Model**: GPT-5.4 (xhigh reasoning) via Codex MCP  
**Search Sources**: Google Scholar (2020–2026), arXiv, AAAI/CVPR/ICCV/NeurIPS/MICCAI proceedings  

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Overall Novelty** | **5/10** |
| **Core Novel Claim** | C4: Channel-modulated negative supervision (7/10) |
| **Weakest Claim** | C5: Cross-validated calibration protocol (2/10) |
| **Most Threatening Paper** | Gong et al. 2021 (Instance-Dependent PU with Labeling Bias Estimation) |
| **Venue Fit** | MICCAI: plausible | CVPR/ICCV: borderline |

**Verdict**: Proceed with caution. C4 (channel-modulated negative supervision) is genuinely novel; C1/C2 are domain specializations of known biased-PU machinery; C3 is incremental; C5 is standard evaluation.

---

## 2. Per-Claim Analysis

### C1: Monotone Annotation Channel Model g(|C|) — **5/10**

**Claim**: Model box annotation as size-dependent selection channel with monotone non-decreasing g(s), fitted via isotonic regression.

**Closest Prior Work**:
| Paper | Overlap | Key Difference |
|-------|---------|----------------|
| Bekker et al. (ECML-PKDD 2019) | Biased PU learning beyond SCAR | Generic, not spatial/segmentation-specific |
| Gong et al. (NeurIPS-W 2021) | Instance-dependent labeling bias estimation | Classification, not dense prediction |
| BLPSeg (IEEE TCSVT 2023) | "Annotation probability" removing size interference | Scribble-supervised, learned per-pixel, not monotone isotonic |
| 3D-BoxSup (Front. Neurosci 2020) | PU learning from 3D boxes | Assumes SCAR (uniform), exactly what we argue is wrong |

**Delta**: 1D size-based, monotone, connected-component-level propensity model specialised to box-supervised segmentation. Neither SCAR nor instance-dependent — a middle ground.

**Risk**: Reviewers may say "this is structured SAR/IDPU, not new conceptually."

---

### C2: Calibrated IPW via Isotonic Regression on Soft Positive Mass — **4/10**

**Claim**: Self-normalised Horvitz-Thompson weights w_j = 1/g_θ(log(mass_j)) applied to box-supervision losses, using soft predicted mass as latent size proxy.

**Closest Prior Work**:
| Paper | Overlap | Key Difference |
|-------|---------|----------------|
| Bekker et al. 2019 | IPW for biased PU | Classification, not dense spatial losses |
| Gong et al. 2021 | Explicit labeling-bias estimation + weighting | Classification, not segmentation |
| Lejeune & Sznitman (MedIA 2021) | PU-style debiasing in segmentation | Different setting (partial labels, not box) |

**Delta**: Application of IPW to **box-supervised dense geometric losses** (tightness + fill) with **soft predicted mass as latent proxy**. The soft-mass → true-size convergence argument is original.

**Risk**: "Useful adaptation, but reviewers may call this a structured transfer of SAR/IDPU ideas." — Codex

---

### C3: 3D Tightness + Filling Rate Constraints Replacing MIL — **3/10**

**Claim**: Replace gradient-dead Noisy-OR/LSE MIL with 3D tightness (per-axis-slice mass ≥ κ) + filling rate interval [ρ_min, ρ_max], both IPW-weighted.

**Closest Prior Work**:
| Paper | Overlap | Key Difference |
|-------|---------|----------------|
| BoxInst (CVPR 2021, 439 cit.) | Projection loss (2D tightness) | Our version is 3D + softplus + IPW-weighted |
| BBTP (NeurIPS 2019) | Tightness prior | 2D, no filling rate, no IPW |
| Kervadec et al. (MIDL 2020) | Size/area constraints for segmentation | Fully supervised, not box-supervised |
| MonoBox (AAAI 2025) | Discards MIL in noisy zones, uses monotonicity | Spatial monotonicity, not size-based; 2D polyps |
| GeoCoBox (AAAI 2026) | 3D box-supervised tumor segmentation | Embedding-based, no constraints |
| LooBox (ACM MM 2025) | 3D box-supervised with loose boxes | Self-correcting, no IPW |
| Prompt learning w/ bbox (IEEE Trans 2025) | Tightness + filling rate together | SAM backbone, no IPW |

**Delta**: Primarily engineering — 3D extension of known constraints + IPW weighting. The MIL-gradient-death observation is correct but not independently publishable.

**Risk**: "Box-constraint engineering." Should NOT be sold as main contribution.

---

### C4: Channel-Modulated Negative Supervision — **7/10** ⭐

**Claim**: Safe-zone voxels receive per-voxel negative weight α_i = g_θ(log(local_mass_i)), where local_mass is predicted positive mass in neighbourhood. High g → strong neg (false alarm); low g → weak neg (protect channel miss).

**Closest Prior Work**:
| Paper | Overlap | Key Difference |
|-------|---------|----------------|
| Uniform safe zone (standard) | Negative supervision outside boxes | Uniform, no selection awareness |
| BLPSeg (IEEE TCSVT 2023) | Annotation probability affects learning | Scribble, learned, not monotone-channel-gated |
| Uncertainty-guided box supervision | Attenuates loss in uncertain regions | Uncertainty-based, not annotation-propensity-based |

**Delta**: **No direct prior found** that uses an estimated annotation propensity to attenuate/strengthen outside-box negative supervision based on local predicted mass. This mechanism — high annotation probability → strong negative, low → protect — is the key conceptual contribution.

**Strengths**:
- Logically ties inside-box (IPW) and outside-box (channel-neg) through the same g_θ
- Creates a "unified channel mechanism" narrative
- Addresses a real failure mode (uniform neg suppressing missed small objects)

**Weaknesses**:
- Currently reads as heuristic / engineering rather than principled
- avg_pool3d as local mass estimate needs ablation justification
- Positioned as "selection-aware regulariser" (honest but weakens formality)

---

### C5: Cross-Validated Calibration Fidelity Protocol — **2/10**

**Claim**: 5-fold CV with within-steepness ECE < 0.05, cross-steepness ECE < 0.10, bootstrap 95% CI.

**Assessment**: Calibration evaluation with ECE and bootstrap CI is completely standard (Mehrtash et al. 2020, etc.). The stress-test across steepness regimes is thorough but is evaluation hygiene, not novelty.

---

## 3. Most Threatening Papers (Must Cite & Differentiate)

| Priority | Paper | Year | Threat Level | Threatens |
|----------|-------|------|-------------|-----------|
| 🔴 1 | Gong et al., Instance-Dependent PU w/ Labeling Bias | 2021 | **HIGH** | C1, C2 |
| 🔴 2 | Bekker et al., Biased PU Learning Beyond SCAR | 2019 | **HIGH** | C1, C2 |
| 🟡 3 | MonoBox (AAAI 2025) | 2025 | **MEDIUM** | C1 (name collision: "monotonicity") |
| 🟡 4 | Kervadec et al. (MIDL 2020) | 2020 | **MEDIUM** | C3 |
| 🟡 5 | GeoCoBox (AAAI 2026) | 2026 | **MEDIUM** | Same domain (3D box tumor seg) |
| 🟡 6 | LooBox (ACM MM 2025) | 2025 | **MEDIUM** | Same domain |
| 🟢 7 | Lejeune & Sznitman (MedIA 2021) | 2021 | **LOW** | C2 (PU in segmentation) |
| 🟢 8 | BLPSeg (IEEE TCSVT 2023) | 2023 | **LOW** | C1 (annotation probability) |
| 🟢 9 | ProPU-Net (2025) | 2025 | **LOW** | PU segmentation narrowing conceptual space |

---

## 4. Positioning Recommendations

### DO ✅
1. **Lead with C4** — channel-modulated negative supervision is the paper's real novelty
2. **Frame C1/C2 as "structured specialisation"** of known biased-PU principles, not "new theory"
3. **Present C3 as "3D-compatible optimisation scaffold"** needed so C4 can work, not as standalone contribution
4. **Prove the annotator-bias premise hard** — empirical monotonicity of miss rate vs. lesion size
5. **Show ablations where C4 specifically adds beyond C1+C2+C3** — this is the key experimental proof
6. **Cite Gong et al. 2021 and Bekker et al. 2019 prominently** and differentiate clearly

### DON'T ❌
1. Don't claim "first annotation-channel modeling" — biased PU exists
2. Don't sell isotonic regression as novelty — it's an estimator choice
3. Don't frame C3 as a main contribution — it's incremental
4. Don't use "monotone" prominently in title — creates confusion with MonoBox (AAAI 2025) which uses spatial monotonicity

### Suggested Reframing
**From**: "Monotone Annotation-Channel Modeling for Size-Aware Box-Supervised Segmentation"  
**To**: "Selection-Aware Box Supervision: Channel-Modulated Negative Regularisation for Missing Small Objects"

This reframing:
- Centres C4 (the 7/10 claim)
- Avoids "monotone" confusion with MonoBox
- Signals the real problem (missing small objects)
- Positions channel as the mechanism, not the theory

---

## 5. Gap Analysis for Experiments

### Must-Have Ablations
| Experiment | Purpose |
|------------|---------|
| Full method vs. no C4 (uniform safe zone) | Proves C4 adds value |
| Full method vs. no IPW (uniform weights) | Proves C1/C2 add value |
| Full method vs. MIL (Noisy-OR) baseline | Proves C3 justified |
| Q1-size Dice (channel vs. uniform) | Core small-object claim |
| Safe-zone α distribution on pixel scans | Verifies C4 mechanism |

### Nice-to-Have
| Experiment | Purpose |
|------------|---------|
| Vary avg_pool3d kernel size | Justifies pool_size choice |
| Vary channel_neg_delay | Justifies warmup gating |
| Compare isotonic vs. Platt vs. beta calibration | Justifies isotonic choice |

---

## 6. Overall Verdict

| Aspect | Rating | Comment |
|--------|--------|---------|
| Conceptual novelty | 5/10 | C4 is novel; C1/C2 are adaptations |
| Technical novelty | 6/10 | Unified channel mechanism is clean |
| Domain novelty | 7/10 | No prior work in box-supervised medical segmentation uses annotation-propensity-aware training |
| Venue readiness | MICCAI: ✅ | With strong experiments |
| Venue readiness | CVPR/ICCV: ⚠️ | Only if C4 is central + compelling experiments |

**Recommendation**: Proceed. Reframe to centre C4. Add required ablations. Cite and differentiate from Gong et al. 2021, Bekker et al. 2019, MonoBox 2025.
