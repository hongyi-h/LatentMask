# LatentMask v2 — Research Design (PropNet Redesign)

> **Version**: 2.0 — Complete redesign from APN (anatomy-specific) to PropNet (task-agnostic).
> **Last updated**: 2026-03-28

---

## 1. Project Positioning

**Working title**

`LatentMask: Multi-Granularity Segmentation as Propensity-Corrected Positive-Unlabeled Learning`

**Target venues**

- Primary: CVPR 2027 / ICCV 2027 / ECCV 2026
- Backup: NeurIPS 2027 (if theoretical angle is strengthened)
- NOT MICCAI — method must be fully general

**Core claim**

Mixed-granularity supervised segmentation (pixel + box + image labels) suffers from systematic false-negative propagation when unlabeled regions are naively treated as background. We identify that this is a structured **positive-unlabeled (PU) learning** problem: the label missingness is non-random and depends on object size, boundary clarity, and annotation granularity. We propose a **Propensity Network (PropNet)** that learns to estimate *where labels are likely missing* directly from encoder features—without any domain-specific prior—and derive a **propensity-corrected unbiased risk** for each annotation granularity. Validated on 3D medical (PE, liver tumors) and 2D natural (PASCAL VOC) benchmarks, LatentMask consistently outperforms naive mixed supervision and semi-supervised baselines.

**Why this is CVPR/ICCV-level, not MICCAI:**
1. The PU formulation for multi-granularity segmentation is a **general ML contribution**, not specific to any medical task.
2. PropNet is **domain-agnostic** — it takes only encoder features, no anatomical priors.
3. Validation spans **three diverse domains** (PE in CT, liver tumors in CT, semantic segmentation in natural images).
4. The synthetic missingness framework is a **reusable tool** for any segmentation task with incomplete annotations.

---

## 2. Problem Statement

### 2.1 Motivation (General)

In many segmentation tasks, annotations exist at multiple granularities:
- A small set of images with dense pixel-level masks (expensive, high quality).
- A medium set with bounding boxes around objects of interest (cheaper, incomplete).
- A large set with only image-level labels indicating object presence (cheapest, no localization).

Standard mixed-supervision methods treat weak labels as low-resolution proxies for dense masks. This assumption fails when weak labels are **structurally incomplete**: annotators consistently miss small objects, boundary regions, and low-contrast instances. The missingness is not random—it depends on **object properties** (size, contrast, location) and **annotation granularity**.

### 2.2 Key Observation

This non-random label missingness has a direct analogy: **positive-unlabeled (PU) learning**. In PU learning:
- We observe some labeled positives (P) and unlabeled data (U).
- The labeling process is governed by a **propensity score** $e(x) = P(\text{labeled} \mid \text{positive}, x)$.
- Treating unlabeled data as negative is biased; the solution is to weight the loss by the propensity.

In multi-granularity segmentation:
- Pixel/box/image labels are all **partial observations of the same unknown dense mask**.
- The labeling propensity depends on **object size** (small → more likely missed), **boundary clarity** (fuzzy → more likely wrong), and **annotation granularity** (image-level → no localization at all).
- Multi-granularity segmentation is therefore a **structured PU learning** problem at the pixel/voxel level.

### 2.3 Research Question

**Can we learn an unbiased segmentation model from mixed-granularity annotations by treating them as positive-unlabeled data with learned propensity, and does this approach generalize across diverse segmentation domains?**

---

## 3. Paper Narrative

### 3.1 Two-Sentence Pitch

**Problem:** Multi-granularity segmentation with incomplete weak labels suffers from systematic false-negative propagation because existing methods treat unlabeled regions as background regardless of label missingness patterns.
**Insight:** We show this is equivalent to PU learning with input-dependent propensity, and propose a lightweight Propensity Network that learns where labels are missing from encoder features alone, enabling principled unbiased risk estimation across annotation granularities.

### 3.2 Contribution Framing (4 contributions)

1. **Formulation**: We identify that mixed-granularity segmentation with incomplete labels is structurally a PU learning problem with input-dependent labeling propensity, and derive the corresponding propensity-corrected unbiased risk estimator for pixel, box, and image-level supervision.

2. **Propensity Network (PropNet)**: We propose a lightweight, domain-agnostic module that estimates per-pixel annotation propensity directly from encoder features, trained via synthetic missingness patterns that simulate realistic label incompleteness without domain-specific priors.

3. **LatentMask framework**: We integrate PropNet with multi-granularity PU losses, spatial smoothness regularization, and propensity-weighted teacher-student refinement into a complete training framework applicable to any encoder-decoder segmentation backbone.

4. **Multi-domain validation**: We validate on three diverse benchmarks—3D pulmonary embolism segmentation (real multi-granularity), 3D liver tumor segmentation (synthetic multi-granularity), and 2D PASCAL VOC semantic segmentation (synthetic multi-granularity)—demonstrating consistent improvement across medical and natural image domains.

---

## 4. Method Design

### 4.1 Overview

LatentMask consists of four components:

1. **Shared encoder-decoder** (any backbone: nnUNet for 3D, DeepLabV3 for 2D)
2. **Dense mask head** (segmentation prediction)
3. **Propensity Network (PropNet)** — lightweight, takes ONLY encoder features [architectural novelty]
4. **Propensity-corrected multi-granularity PU risk** [algorithmic novelty]

### Figure 1 (text description)

**Figure 1. LatentMask framework overview.**
Three input streams (pixel-level, box-level, image-level) feed into a shared encoder-decoder that produces dense segmentation logits. A parallel Propensity Network (PropNet) takes intermediate encoder features and outputs a per-pixel propensity map $e(x)$, estimating the probability that each pixel was labeled if positive. The propensity map modulates the PU-corrected loss: pixel-labeled data use standard supervised loss ($e=1$); box/image-labeled data use propensity-corrected PU risk that avoids pushing potentially unlabeled positives toward background. An EMA teacher provides propensity-weighted pseudo-labels for refinement. The framework is domain-agnostic — shown here with both 3D medical and 2D natural image examples.

### 4.2 Problem Formulation

Given an image (2D or 3D volume) $\mathbf{x}$, the unknown dense mask is $\mathbf{z} \in \{0,1\}^{N}$ ($N$ = total pixels/voxels). The model predicts:

$$p = f_\theta(\mathbf{x}), \quad e = g_\phi(\text{enc}(\mathbf{x}))$$

where $p$ is the pixel-wise foreground probability and $e$ is the pixel-wise annotation propensity. Note: $e$ depends **only on encoder features**, not on any domain-specific input.

| Annotation type | Known positives | Unlabeled | Propensity |
|---|---|---|---|
| Pixel-level | All positive pixels | None | $e = 1$ for all pixels |
| Box-level | Pixels inside annotated boxes | Pixels outside boxes | $e$ varies: low for small/boundary regions |
| Image-level positive | None (only image-level) | All pixels | $e$ is low everywhere |
| Image-level negative | N/A | All pixels confirmed negative | N/A |

The propensity $e(v) = P(\text{labeled} \mid z(v)=1, v)$ captures:
- **Size effect**: small objects → lower propensity (more likely missed)
- **Boundary effect**: boundary pixels → lower propensity (less precisely labeled)
- **Contrast effect**: low-contrast regions → lower propensity
- **Granularity effect**: coarser annotation → lower propensity everywhere

### 4.3 Propensity Network (PropNet)

**Architecture:**

PropNet is a lightweight CNN head that takes encoder features and outputs per-pixel propensity:

```
Input: encoder_features (intermediate stage, e.g., stage 3 of encoder)
→ 3×3(×3) Conv, BN, ReLU (64 channels)
→ 3×3(×3) Conv, BN, ReLU (32 channels)
→ 1×1(×1) Conv → Sigmoid → e(x) ∈ (ε, 1-ε)
```

- 3D version for nnUNet: Conv3d with ~0.2M parameters
- 2D version for DeepLabV3: Conv2d with ~0.05M parameters

**Key architectural properties:**
- Takes ONLY encoder features — no domain-specific inputs (no vesselness, no anatomy map)
- Lightweight — negligible overhead compared to backbone (~31M for nnUNet, ~25M for DeepLabV3)
- Backbone-agnostic — same PropNet head attaches to any encoder-decoder architecture
- Produces interpretable output: the propensity map shows WHERE the model expects annotation gaps

**Training signal: Synthetic Missingness**

Since true propensity is unobservable, we train PropNet on pixel-labeled data using synthetic label corruption:

| Pattern | What it simulates | Implementation |
|---|---|---|
| **Scale-dependent drop** | Small objects more likely to be missed | Drop rate $\propto 1/\sqrt{\text{area}(C_i)}$ for each connected component $C_i$ |
| **Boundary erosion** | Edge labels are imprecise | Random morphological erosion with radius $r \sim U(1, 5)$ |
| **Component drop** | Entire instances missed by annotator | Drop each component with probability $p_{\text{drop}}$ weighted by size |

For each pixel-labeled sample:
1. Start with full label $\mathbf{z}$
2. Apply random combination of the three patterns
3. Produce corrupted label $\tilde{\mathbf{z}}$ and ground-truth propensity map $\mathbf{e}^*$
4. Train: $L_{\text{prop}} = \text{BCE}(g_\phi(\text{enc}(\mathbf{x})), \mathbf{e}^*)$

**Why this works:**
- The synthetic missingness patterns correspond to REAL annotation biases (well-documented in medical imaging and COCO-style datasets)
- PropNet learns to associate encoder features (which encode size, shape, contrast) with annotation likelihood
- At inference on box/image-level data, PropNet generalizes because the feature-propensity relationship is consistent

**Optional domain-specific enhancement:** For specific applications, PropNet can accept an additional auxiliary input channel (e.g., vesselness for PE). This is NOT part of the core method but demonstrates extensibility. Ablation shows: PropNet alone < PropNet + domain hint (on PE), but PropNet alone already beats baselines.

### 4.4 Propensity-Corrected Multi-Granularity PU Risk

**Standard PU risk (Kiryo et al., NeurIPS 2017):**

For binary classification with class prior $\pi$ and loss $\ell$:

$$\hat{R}_{PU}(f) = \pi \cdot \hat{R}_P^+(f) + \max\big(0,\; \hat{R}_U^-(f) - \pi \cdot \hat{R}_P^-(f)\big)$$

We adapt this to pixel-level segmentation with learned per-pixel propensity:

#### Pixel-Level Loss ($e = 1$):

Standard supervised segmentation (no PU correction needed):
$$L_{\text{pix}} = \text{Dice}(p, y_{\text{pix}}) + \text{CE}(p, y_{\text{pix}})$$

#### Box-Level PU Loss:

For a box-labeled image with box regions $B$:
$$L_{\text{box}} = L_{\text{box,pos}} + \max\big(0,\; L_{\text{box,unlabeled}} - \pi_{\text{box}} \cdot L_{\text{box,neg\_on\_pos}}\big)$$

- $L_{\text{box,pos}}$: CE loss on positive voxels inside $B$
- $L_{\text{box,neg\_on\_pos}}$: PU correction term using labeled positives
- $L_{\text{box,unlabeled}}$: Propensity-weighted CE on voxels outside $B$:
  $$L_{\text{box,unlabeled}} = \frac{1}{\sum_v e(v)} \sum_{v \notin B} e(v) \cdot \text{CE}(p(v), 0)$$

**Intuition:** In regions where PropNet predicts low propensity (e.g., small object boundary), the negative push is reduced. High-propensity regions outside boxes are confidently treated as negative.

#### Image-Level PU Loss:

For positive images (object present somewhere):
$$L_{\text{img}} = L_{\text{cls}} + \lambda_{\text{vox}} \cdot L_{\text{vox\_pu}}$$

- $L_{\text{cls}} = \text{BCE}(\text{noisy\_or}(p), 1)$: case-level classification
- $L_{\text{vox\_pu}} = \text{mean}_v\big[e(v) \cdot \text{CE}(p(v), 0)\big]$: voxel-level PU suppression only in high-propensity regions

For negative images: $L_{\text{img}} = \text{mean}_v[\text{CE}(p(v), 0)]$ (all pixels confirmed negative).

#### Spatial Smoothness Regularization:

Replace the old domain-specific AnatomyRegularization with a general regularizer:
$$L_{\text{smooth}} = \text{TV}(e) = \sum_v \|\nabla e(v)\|_1$$

This encourages propensity maps to be spatially smooth — nearby pixels should have similar annotation likelihoods.

### 4.5 Total Loss

$$L_{\text{total}} = \lambda_{\text{pix}} L_{\text{pix}} + \lambda_{\text{box}} L_{\text{box}} + \lambda_{\text{img}} L_{\text{img}} + \lambda_{\text{prop}} L_{\text{prop}} + \lambda_{\text{smooth}} L_{\text{smooth}} + \lambda_{\text{ref}} L_{\text{ref}}$$

| Loss | Weight | Active stages | Purpose |
|---|---|---|---|
| $L_{\text{pix}}$ | $\lambda_{\text{pix}} = 1.0$ | All | Supervised pixel loss |
| $L_{\text{box}}$ | $\lambda_{\text{box}} = 1.0$ | Stage 2+ | PU-corrected box loss |
| $L_{\text{img}}$ | $\lambda_{\text{img}} = 0.5$ | Stage 2+ | PU-corrected image loss |
| $L_{\text{prop}}$ | $\lambda_{\text{prop}} = 0.5$ | All | PropNet training (synthetic missingness) |
| $L_{\text{smooth}}$ | $\lambda_{\text{smooth}} = 0.1$ | All | Propensity spatial smoothness |
| $L_{\text{ref}}$ | $\lambda_{\text{ref}} = 0.4$ | Stage 3 | EMA teacher refinement |

### 4.6 Training Schedule

| Stage | Epochs | Data | Active losses | Purpose |
|---|---|---|---|---|
| **Stage 1: Warm-up** | 1–50 | Pixel only | $L_{\text{pix}} + L_{\text{prop}} + L_{\text{smooth}}$ | Backbone + PropNet pre-training |
| **Stage 2: Joint PU** | 51–300 | Pixel + Box + Image | All except $L_{\text{ref}}$ | Multi-granularity PU learning |
| **Stage 3: Refinement** | 301–400 | All | All | Propensity-weighted pseudo-labels |

Multi-granularity batch schedule in Stage 2+: cycle through [pixel, pixel, box, image] per 4 iterations.

---

## 5. Experimental Design

### 5.1 Data Sources

#### Benchmark 1: Pulmonary Embolism (3D CT, Real Multi-Granularity)

| Dataset | Supervision | Scale | Usage |
|---|---|---|---|
| READ-PE / Pixel-PE | Pixel-level masks | ~40 examinations | Internal 5-fold CV |
| CAD-PE | Pixel-level masks | 91 CTPA scans | Secondary main benchmark + external |
| FUMPE | Pixel-level masks | 35 cases | External benchmark |
| Augmented RSPECT | Box-level | 445 studies, 30K boxes | Box-level PU source |
| RSPECT | Image-level | 12,195 patients | Image-level PU source |

**This is the ONLY benchmark with naturally occurring multi-granularity annotations.** The other benchmarks use synthetic degradation.

#### Benchmark 2: Liver Tumor Segmentation (3D CT, Synthetic Multi-Granularity)

| Source | Original | After degradation |
|---|---|---|
| LiTS training (131 cases) | 100% pixel labels | 30% pixel, 30% box (bbox of tumor masks), 40% image-level |
| LiTS test (70 cases) | 100% pixel labels | Held-out evaluation (no degradation) |

**Why LiTS:** CT modality (same as PE → nnUNet directly applicable), high tumor size variability (tiny metastases vs large HCC), well-established benchmark. The scale-dependent missingness assumption is well-motivated: small metastases are harder to annotate.

#### Benchmark 3: PASCAL VOC 2012 (2D, Synthetic Multi-Granularity)

| Source | Original | After degradation |
|---|---|---|
| VOC train+aug (~10K images) | 100% pixel labels | 20% pixel, 30% box (from detection annotations), 50% image-level |
| VOC val (1,449 images) | 100% pixel labels | Held-out evaluation (no degradation) |

**Why VOC:** 2D natural images (proves cross-domain and cross-modality generality), well-established baselines, bounding box annotations already available from detection track.

### 5.2 Synthetic Multi-Granularity Construction Protocol

For LiTS and VOC, degrade full pixel labels to simulate real-world annotation heterogeneity:

1. **Image selection**: Randomly assign each training image to pixel (P), box (B), or image (I) group with prescribed ratios.
2. **Pixel group**: Retain full masks. Additionally, apply scale-dependent thinning within this group (drop 10-30% of small-object pixels) to simulate typical annotation noise.
3. **Box group**: Replace pixel masks with tight bounding boxes of each connected component. This simulates box-level annotation where the annotator drew boxes but did not delineate boundaries.
4. **Image group**: Replace pixel masks with image-level class labels (present/absent per class). This simulates the cheapest annotation.
5. **Within-group missingness**: For box and image groups, additionally drop small components entirely (probability inversely proportional to component area) to simulate real annotator behavior.

### 5.3 Baselines

#### PE Baselines (3D)

| # | Method | Category | Supervision | Implementation |
|---|---|---|---|---|
| 1 | nnUNet (pixel-only) | Full supervision | Pixel | Official nnUNet |
| 2 | Mean Teacher | Semi-supervised | Pixel + unlabeled | [Tarvainen & Valpola 2017], adapt to 3D |
| 3 | Cross Pseudo Supervision (CPS) | Semi-supervised | Pixel + unlabeled | [Chen et al. 2021] |
| 4 | nnPU-Seg | PU learning | Pixel + Image | Kiryo et al. 2017, uniform propensity |
| 5 | 3D BoxSup | Box-supervised | Box | [Dai et al. 2015], adapt to 3D |
| 6 | Mixed Naive | Mixed supervision | Pixel + Box + Image | Treat unlabeled as negative |
| 7 | **LatentMask (ours)** | PU + PropNet | Pixel + Box + Image | This work |

#### LiTS Baselines (3D)

| # | Method | Supervision |
|---|---|---|
| 1 | nnUNet Oracle | 100% pixel (upper bound) |
| 2 | nnUNet (pixel-only) | 30% pixel only |
| 3 | Mean Teacher | 30% pixel + 70% unlabeled |
| 4 | nnPU-Seg | 30% pixel + 70% as PU (uniform) |
| 5 | Mixed Naive | 30% pixel + 30% box + 40% image |
| 6 | **LatentMask (ours)** | 30% pixel + 30% box + 40% image + PropNet |

#### VOC Baselines (2D)

| # | Method | Supervision |
|---|---|---|
| 1 | DeepLabV3 Oracle | 100% pixel (upper bound) |
| 2 | DeepLabV3 (pixel-only) | 20% pixel only |
| 3 | UniMatch | 20% pixel + 80% unlabeled (semi-sup, CVPR 2023) |
| 4 | CAM-baseline | 50% image-level → CAM → pseudo-labels |
| 5 | BoxSup | 30% box → GrabCut → pseudo-labels |
| 6 | Mixed Naive | 20% pixel + 30% box + 50% image |
| 7 | **LatentMask (ours)** | 20% pixel + 30% box + 50% image + PropNet |

### 5.4 Metrics

**3D Medical (PE, LiTS):** Dice, HD95 (mm), Lesion-F1, Small-lesion Recall, FP/scan
**2D Natural (VOC):** mIoU, per-class IoU, small-object IoU (objects < 32²)
**Framework-specific:** Propensity calibration error (ECE of propensity vs empirical missingness on synthetic benchmarks)

### 5.5 Main Claims and Experiments

| Claim | Evidence |
|---|---|
| PU formulation + PropNet outperforms naive mixed supervision | Main comparison (Tables 2, 4, 6) |
| PropNet learns meaningful propensity without domain priors | Propensity visualization (Figure 3) + calibration analysis |
| LatentMask generalizes across 3D medical and 2D natural domains | Three benchmarks (Tables 2, 4, 6) |
| Propensity correction is robust to different missingness levels | Robustness experiment varying degradation ratios |
| Each component contributes | Ablation (Table 3) cross all three benchmarks |
| Multi-granularity data is more efficient than pixel-only | Annotation budget study (Figure 5) |

### 5.6 Ablation Plan

| Variant | PropNet | PU correction | Smoothness reg | EMA refinement |
|---|---|---|---|---|
| A0: Naive mixed supervision | ✗ | ✗ | ✗ | ✗ |
| A1: + PU loss (uniform e=0.5) | ✗ | ✓ | ✗ | ✗ |
| A2: + PropNet (learned propensity) | ✓ | ✓ | ✗ | ✗ |
| A3: + Spatial smoothness | ✓ | ✓ | ✓ | ✗ |
| A4: Full LatentMask | ✓ | ✓ | ✓ | ✓ |

**Key comparison:** A1 vs A2 directly tests whether LEARNED propensity beats uniform propensity.

**Additional ablations:**
- Synthetic missingness patterns: scale-only, boundary-only, component-only, all combined
- PropNet input stage: stage 2 vs stage 3 vs bottleneck features
- Propensity clamping value: ε ∈ {0.01, 0.05, 0.1}
- PE-specific: PropNet vs PropNet + vesselness hint (optional enhancement)

### 5.7 Figure Plan

| Figure | Purpose | Format |
|---|---|---|
| Figure 1 | Method overview: multi-granularity → PU → PropNet → corrected risk | Architecture diagram |
| Figure 2 | PU correction vs naive: side-by-side on box-labeled data | 2-panel comparison |
| Figure 3 | PropNet propensity maps across three domains (PE, LiTS, VOC) | 3×3 grid |
| Figure 4 | Qualitative segmentation comparison (ours vs top baselines) | Multi-panel |
| Figure 5 | Annotation budget frontier (Dice vs annotation cost) | Line plot |

---

## 6. Feasibility Analysis

### 6.1 Engineering Effort

| Component | Effort | Risk |
|---|---|---|
| PropNet (replace APN) | Remove vesselness input from existing APN code | Very low |
| Synthetic missingness v2 | Rewrite scale/boundary/component patterns | Low |
| PE experiments (3D nnUNet) | Already have infrastructure and data | Very low |
| LiTS experiments (3D nnUNet) | Download LiTS + conversion scripts + degradation | Low |
| VOC experiments (2D DeepLabV3) | New 2D training loop + 2D PropNet + 2D PU losses | Medium |
| Baselines | Most have public implementations; adapt to our benchmarks | Medium |

**Total estimated engineering: 4-6 weeks** for one researcher.
**Total estimated GPU time: ~3,000 A100-hours** (PE: ~1,500h, LiTS: ~800h, VOC: ~400h, ablations: ~300h).

### 6.2 Expected Experimental Outcomes

| Benchmark | Expected Dice/mIoU improvement (ours vs mixed naive) | Confidence |
|---|---|---|
| PE (internal) | +3–6 Dice points | 80% |
| PE (external) | +2–4 Dice points | 75% |
| LiTS | +3–5 Dice points | 85% |
| VOC | +2–4 mIoU points | 80% |

### 6.3 Fallback Strategy

If PropNet doesn't provide significant improvement over uniform propensity:
1. The PU framework itself (A1) should still beat naive mixed supervision — **this is the minimum viable paper**.
2. Add curriculum-based propensity (schedule ε from high to low during training) as a simpler alternative.
3. Strengthen the theoretical contribution: prove the unbiased risk property formally, derive variance bounds.

---

## 7. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| PropNet doesn't generalize from synthetic to real missingness | Medium | Synthetic patterns are realistic; multiple pattern combination; validate on PE (real missingness) |
| PU correction is unstable (negative risk values) | Low | Non-negative PU risk (Kiryo et al.) already handles this; propensity clamping |
| VOC 2D experiments showing weaker gains | Medium | VOC is supplementary evidence; PE + LiTS already prove the point in 3D |
| Small pixel-labeled PE set (40 cases) limits PropNet training | Medium | Use data augmentation; PropNet pre-trains on pixel data then generalizes; CAD-PE (91 cases) as additional benchmark |
| Reviewers demand more datasets | Low | Can easily add BraTS or KiTS19 with same pipeline |

---

## 8. Submission Strategy

**Primary**: CVPR 2027 (Nov deadline) or ICCV 2027 (Mar deadline)
- General framing: "multi-granularity segmentation as PU learning"
- Three diverse benchmarks (PE + LiTS + VOC)
- Strong ablation + theoretical justification

**Alternative**: NeurIPS 2027 (May deadline)
- Emphasize theoretical PU contribution
- Add formal proof of unbiased risk property
- Same experiments

---

## 9. What Changed: v1 (APN) → v2 (PropNet)

| Aspect | v1 APN | v2 PropNet |
|---|---|---|
| Core module | Anatomical Propensity Network | Propensity Network |
| Input | encoder_features + **vesselness map** | encoder_features **only** |
| Training signal | Vesselness-based synthetic missingness | Scale/boundary/component synthetic missingness |
| Anatomy regularization | $L_{\text{ana}} = \text{mean}(p \cdot (1-V))$ — vesselness mask | $L_{\text{smooth}} = \text{TV}(e)$ — spatial smoothness |
| APN-specific losses | $L_{\text{apn\_order}}$ (vesselness-propensity ordering) | Removed (no domain-specific ordering) |
| Domain specificity | PE-specific (vesselness, branch order) | **Fully domain-agnostic** |
| Validation | 5 PE datasets only | **PE + LiTS + PASCAL VOC** |
| Target venue | CVPR/ICCV (but MICCAI risk) | **CVPR/ICCV (confident)** |
| Paper framing | "anatomically-structured PU learning" | "multi-granularity segmentation as PU learning" |

**What is preserved from v1:**
- PU learning formulation (core theoretical contribution)
- Multi-granularity batch cycling (pixel/box/image schedule)
- EMA teacher refinement (Stage 3)
- Non-negative PU risk (Kiryo et al.)
- 3-stage training schedule
- nnUNet as 3D backbone

---

## 10. Implementation Checklist

- [ ] Modify `modules/apn.py` → `modules/propnet.py`: remove vesselness input
- [ ] Rewrite `utils/synthetic_missingness.py`: scale/boundary/component patterns (remove vesselness)
- [ ] Update `losses/pu_losses.py`: remove `AnatomyRegularizationLoss`, add `SpatialSmoothnessLoss`
- [ ] Update `losses/combined.py`: remove vesselness from interface
- [ ] Update `trainer/latentmask_trainer.py`: remove vesselness computation, use PropNet
- [ ] Add 2D PropNet variant (`modules/propnet_2d.py`)
- [ ] Add 2D training framework for VOC (`trainer/voc_trainer.py`)
- [ ] Add LiTS data conversion script (`scripts/convert_lits.sh`)
- [ ] Add synthetic degradation script (`scripts/build_synthetic_multigran.py`)
- [ ] Implement nnPU-Seg baseline
- [ ] Implement Mean Teacher 3D baseline
- [ ] Run PE experiments (Tables 2-3)
- [ ] Run LiTS experiments (Tables 4-5)
- [ ] Run VOC experiments (Tables 6-7)
- [ ] Run ablations on all three benchmarks
- [ ] Run annotation budget study
- [ ] Generate figures (propensity maps, qualitative comparison)
