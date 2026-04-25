# Novelty Check Report v6 — Selection-Aware Channel-Modulated Negative Regularisation

**Date**: 2026-04-23  
**Proposal**: v5-final — "Selection-Aware Box Supervision via Channel-Modulated Negative Regularisation"  
**Previous Check**: v5 (2026-04-16, score 6/10, PROCEED WITH CAUTION)  
**Phase B Sources**: arXiv (2026-04 listing), manual targeted searches, prior v5 report cross-referenced  
**Phase C Reviewer**: Codex MCP default model (gpt-5.4 at capacity; external sources cross-checked)  

---

## 1. Executive Summary

| Metric | v5 (Apr 16) | v6 (Apr 23) | Change |
|--------|-------------|-------------|--------|
| **Overall Novelty** | 6/10 | **6.5/10** | +0.5 |
| **Core Mechanism (C1)** | 7/10 | **7/10** | — |
| **Fallback (monotone > uniform)** | ~4/10 | **3.5/10** | −0.5 — weaker in isolation |
| **Most Threatening** | ProPU-Nets (MedIA 2025) | ProPU-Nets (MedIA 2025) | Unchanged |
| **New Papers Since v5** | — | None (April 16–23 scan clean) | No new threats |
| **Recommendation** | PROCEED WITH CAUTION | **PROCEED WITH CAUTION** | Unchanged |

**Verdict**: The specific combination remains mechanistically novel — no prior work connects size-dependent annotation propensity to per-component safe-zone negative supervision strength. However, novelty is **compositional** (known ingredients: biased-PU, box-supervised safe zones, CC extraction), requiring strong experimental evidence and tight single-claim framing to publish. The calibration challenge (identifiability of g_θ from incomplete boxes) is the primary new risk surfaced by the fresh cross-model review.

---

## 2. Core Claims Analysis

### C1: Per-Component Channel-Modulated Negative Supervision — **7/10** ⭐

**Claim**: Safe-zone voxels receive per-connected-component weight α(u) = g_θ(log m(C(u))), where m(C) is predicted component mass. Large predicted components → strong negative; small → weak (protect likely missed annotation).

**Cross-model verification (Codex MCP, Apr 23)**: Confirmed — "No listed prior work appears to connect all three pieces: (1) estimated annotation propensity, (2) size-dependent selection bias, (3) component-level modulation of safe-zone negative supervision."

**Narrowing requirement**: Claim must be scoped as "first component-level, size-propensity-modulated safe-zone negative regularizer for box-supervised 3D lesion segmentation" — not "first annotation-propensity segmentation method."

**Closest Prior Work**:
| Paper | Year | Overlap | Key Delta |
|-------|------|---------|-----------|
| 3D-BoxSup | 2020 | PU from 3D boxes, medical | SCAR (uniform). No CC-level modulation. |
| Bekker (ECML 2019) + Gong (NeurIPS-W 2021) | 2019/21 | Biased/instance-dependent PU | Classification, no spatial safe-zone structure |
| ProPU-Nets (MedIA 2025) | 2025 | PU for medical segmentation | Models annotator *certainty*, not size-dependent *propensity*; EM-based, no safe-zone |
| BLPSeg (TCSVT 2023) | 2023 | Per-pixel annotation probability | Scribble-supervised, not CC-level, not box-supervised, not monotone |

**Closest compound attack by reviewer**: "This is 3D-BoxSup + instance-dependent PU + a hand-designed size prior." Defense: CC-level safe-zone placement is a structural choice addressing a specific failure mode, not cosmetic.

---

### C2: Fallback — Monotone Size Modulation > Uniform — **3.5/10** (standalone)

If g_θ ≈ linear, the contribution becomes: "small predicted components in safe zone receive lower negative weight than large ones."

**Cross-model assessment**: This is "weakly novel but easy to dismiss" — reviewers may call it an intuitive heuristic unless framed as data-calibrated propensity regularisation with clear empirical superiority over simple linear/logistic size schedules (C4 > C3 in ablation).

**Implication**: The ablation C4 (g_θ) vs C3 (linear) vs C2 (uniform) is critical. Fallback alone is insufficient for venue acceptance.

---

### C3: Hungarian Matching + Isotonic Fitting Protocol — **2/10** (NOT a contribution)

Standard textbook operations. Report in methods, do not claim.

---

## 3. New Papers Found (Apr 16–23, 2026)

**Scan performed**: arXiv cs.CV April 2026 listing (2,562 entries total as of Apr 23). Targeted searches for: "box supervised segmentation safe zone," "annotation propensity negative supervision," "connected component negative loss," "PU learning segmentation size dependent."

**Result**: **Zero new overlapping papers found** in the April 16–23, 2026 window.

Confirmed non-threats from the scan period:
- BEEP3D (arXiv:2510.12182, Oct 2025 — already in v5): 3D point cloud box-supervised (ScanNetV2/S3DIS), EMA pseudo-masks, not medical, no propensity mechanism. **Safe.**

---

## 4. Most Threatening Papers (Must Cite & Differentiate)

| Priority | Paper | Year | Venue | Threat | Threatens |
|----------|-------|------|-------|--------|-----------|
| 🔴 1 | **ProPU-Nets** (Yi et al.) | 2025 | MedIA | HIGH | Same problem family: PU seg in medical. EM-based uncertainty. Differentiate: they model annotator *certainty in known regions*; you model size-dependent *selection probability* → negative weight |
| 🔴 2 | **Bekker + Gong** | 2019/21 | ECML/NeurIPS-W | HIGH | Non-SCAR PU, instance-dependent labeling. Differentiate: classification setting, no spatial safe-zone structure, no 3D medical |
| 🔴 3 | **3D-BoxSup** | 2020 | MICCAI | HIGH | Direct domain predecessor. Differentiate: SCAR (uniform), no CC-level modulation |
| 🟡 4 | **GeoCoBox** | 2026 | AAAI | MEDIUM | Same domain: 3D box-supervised tumor. Geometric constraints, uniform negatives |
| 🟡 5 | **LooBox** | 2025 | ACM MM | MEDIUM | Same domain: 3D box-supervised tumor. Point-guided correction, no propensity |
| 🟡 6 | **PIASeg** | 2026 | IEEE JBHI | MEDIUM | 3D partial-instance annotation on LiTS. Meta-learning, does not model why instances are missing |
| 🟡 7 | **PADL** | 2024 | MedIA | MEDIUM | Annotator preference modeling. Multi-annotator, fully supervised |
| 🟡 8 | **MonoBox** | 2025 | AAAI | MEDIUM | "Monotonicity" in box supervision — terminology confusion risk. 2D only, no size modulation |
| 🟢 9 | **BLPSeg** | 2023 | TCSVT | LOW | Annotation probability for scribble supervision. Different paradigm |
| 🟢 10 | **BEEP3D** | 2025 | arXiv | LOW | 3D point cloud, EMA pseudo-masks. Different domain and mechanism |

---

## 5. Search Evidence Summary

### Targeted Exact-Match Searches (All Returned Zero Results on arXiv)
- `"annotation propensity" "safe zone" segmentation` → 0 results
- `"box supervised" "connected component" "negative supervision"` → 0 results  
- `"channel modulated" "negative regularization" segmentation` → 0 results
- `"propensity" "safe zone" segmentation "size"` → 0 results
- `"negative supervision weight" "box supervision" "connected component"` → 0 results

### Cross-Model Verification (Codex MCP, April 23, 2026)
- Confirmed: no prior work found connecting annotation propensity (size-dependent) + safe-zone + CC-level negative modulation
- **New risk surfaced**: Identifiability/calibration of g_θ — missed small lesions are unobserved, so box statistics may not reliably identify their annotation probability → must address in paper (pixel subset fitting, 3-fold CV ECE < 0.10)
- **New risk surfaced**: False-positive preservation — weakening all small safe-zone components may preserve spurious predictions alongside true missed lesions → must show lesion-level recall vs. FP tradeoff

---

## 6. Overall Novelty Assessment

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Mechanism Novelty** | 7/10 | No exact prior for CC-level propensity-weighted negative supervision |
| **Conceptual Novelty** | 5/10 | Ingredients all familiar. Composition is original |
| **Domain Novelty** | 6/10 | Box-supervised 3D medical segmentation is active but not saturated |
| **Experimental Novelty** | 4/10 | LiTS-only, standard metrics |
| **Overall** | **6.5/10** | +0.5 vs v5 (v5=6/10). Narrow but real novelty. |

### Recommendation: **PROCEED WITH CAUTION**

### Key Differentiator
"First method to connect size-dependent annotation propensity (estimated from a partial pixel-labeled subset) to per-component safe-zone negative supervision strength in box-supervised 3D lesion segmentation."

### Primary Risks (from cross-model review)
1. **Identifiability**: g_θ is estimated from boxes + pixel labels on a small subset; missed small lesions are unobserved by construction. Reviewers may argue g_θ cannot be calibrated without knowing what was missed.  
   → Mitigation: 3-fold CV ECE < 0.10, unmatched_rate reporting, simulation fallback with explicit limitation statement.

2. **False-positive preservation**: Weakening negatives for small components may preserve both true missed lesions AND spurious predictions.  
   → Mitigation: Report lesion-level recall and per-case FP count alongside Dice; show Q1-Dice gains without FP explosion.

3. **Compositional triviality**: "3D-BoxSup + instance-dependent PU + size prior."  
   → Mitigation: Strict ablation (C4 > C3 > C2 ordering); the CC-level mechanism and the safe-zone linkage are the structural novelties that distinguish from classification-level PU.

4. **Fallback weakness**: If g_θ ≈ linear, the method reduces to a heuristic size-based weighting (3.5/10 novelty alone).  
   → Pre-committed response: "Monotone modulation > uniform is still the first principled use of estimated annotation propensity for safe-zone weighting in 3D box-supervised segmentation."

---

## 7. Positioning Recommendations (Updated)

### DO ✅
1. **Single contribution**: Per-component channel-modulated safe-zone negative supervision (C1 only)
2. **Lead with calibration evidence**: Prove g_θ is fit from independent pixel-annotated scans; report ECE
3. **Address identifiability explicitly in paper**: "g_θ is fit on the 31 pixel-annotated LiTS scans where both GT and box labels are available, enabling principled estimation of the size→propensity curve"
4. **Add lesion-level FP metrics** to Table 1 alongside Dice/HD95
5. **Cite Bekker/Gong prominently** with spatial differentiation: "We instantiate biased-PU reasoning in the spatial safe-zone setting — a combination not addressed in classification-level biased PU"

### DON'T ❌
1. Don't claim the calibration protocol (Hungarian + isotonic) as a contribution
2. Don't oversell theoretical framing (Proposition A1–A3) as a formal theorem
3. Don't use "channel" terminology without explicit definition — confusion with channel attention
4. Don't rely on fallback as primary claim without clear ablation evidence

### Suggested Differentiation Language (Updated)

**vs. ProPU-Nets**: "ProPU-Nets model regions where the annotator *is uncertain* (true-positive neighborhoods); we model which *entire instances* were systematically skipped due to size-dependent selection. The distinction is uncertainty in annotation detail vs. absence of annotation for small lesions."

**vs. 3D-BoxSup (SCAR)**: "3D-BoxSup assumes Selected Completely At Random — uniform propensity regardless of lesion size. Our per-component g_θ replaces this assumption with a data-fitted monotone function, explicitly protecting small safe-zone predictions likely corresponding to annotation gaps."

**vs. Bekker/Gong**: "Biased and instance-dependent PU methods estimate labeling propensity for reweighting classification losses. We exploit the spatial structure of 3D box-supervised segmentation: voxels in the safe zone form connected components whose predicted mass determines annotation propensity, yielding per-voxel negative supervision weights without additional parameters beyond g_θ."

---

## 8. Comparison with Prior Checks

| Aspect | v3 (Jul 2025) | v5 (Apr 2026) | v6 (Apr 2026 +7d) |
|--------|---------------|---------------|--------------------|
| Score | 5/10 | 6/10 | 6.5/10 |
| Core mechanism | avg_pool3d mass | CC-level mass | CC-level mass |
| New threats | — | ProPU-Nets, PIASeg, PADL | None new |
| New risks | — | Calibration | **Identifiability + FP preservation** |
| Recommendation | PROCEED W/ CAUTION | PROCEED W/ CAUTION | PROCEED W/ CAUTION |

**What improved (v5→v6)**: No new competing papers emerged in the 7-day window. Score increased marginally (+0.5) reflecting cleaner single-contribution framing in final proposal vs. v5. Identifiability concern is now explicitly named — allowing the paper to address it proactively.

**What remains unchanged**: Fundamental assessment is stable. Publication depends on experimental evidence quality.

---

## 9. Bottom Line

**The mechanism IS novel** — the specific combination of (size→propensity→CC-level safe-zone α) has no direct predecessor in the box-supervised 3D medical segmentation literature.

**But it is a narrow, compositional novelty** whose key risks are:
- g_θ calibration/identifiability 
- false-positive preservation tradeoff
- reviewer framing as "obvious combination of known tools"

**To publish at MICCAI 2026** (primary target): Strong Q1-Dice gain (+5pt) + ECE < 0.10 + lesion-level FP control + clean 3-way ablation. This is achievable.

**To publish at CVPR/ICCV**: Need second dataset, or deeper theoretical treatment, or striking +10pt gains.

**Honest fallback**: If g_θ ≈ linear AND gains < 5pt, the contribution is insufficient for top venues. Consider MICCAI workshop or domain journal (IEEE JBHI, MedIA).
