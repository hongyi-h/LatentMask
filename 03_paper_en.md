# LatentMask: Multi-Granularity Segmentation as Propensity-Corrected Positive-Unlabeled Learning

> **Draft status:** All quantitative values are **mock numbers** synchronized with `02_experimental_tables.md`. Replace with real results after experiments.

---

## Abstract

Segmentation with mixed-granularity supervision—pixel masks, bounding boxes, and image-level labels—suffers from systematic false-negative propagation when unlabeled regions are treated as background. We identify that this problem is equivalent to **positive-unlabeled (PU) learning** with input-dependent labeling propensity: annotations are more likely missing for small objects, boundary regions, and coarsely labeled images. We introduce **LatentMask**, a framework that estimates *where labels are likely missing* and corrects the training signal accordingly. Central to LatentMask is the **Propensity Network (PropNet)**, a lightweight, domain-agnostic module that predicts per-pixel annotation propensity from encoder features alone—requiring no task-specific priors. PropNet is trained via synthetic label corruption patterns (scale-dependent drop, boundary erosion, component drop) that simulate realistic annotation biases. We derive a **propensity-corrected unbiased risk** for each annotation granularity and integrate it with a three-stage training schedule including warm-up, joint PU learning, and propensity-weighted teacher refinement. Evaluation on three diverse benchmarks—3D pulmonary embolism segmentation (real multi-granularity), 3D liver tumor segmentation and 2D PASCAL VOC semantic segmentation (both with synthetic multi-granularity)—demonstrates consistent improvement. On the PE benchmark, LatentMask achieves XX.X Dice (+PLACEHOLDER over mixed naive baseline, +PLACEHOLDER over uniform PU). On LiTS, LatentMask recovers XX.X% of oracle performance using only 30% pixel labels. On VOC, LatentMask achieves XX.X mIoU (+PLACEHOLDER over mixed naive) using only 20% pixel labels. Ablation studies confirm that learned propensity is the critical ingredient across all three domains.

---

## 1 Introduction

Accurate segmentation models typically require dense pixel-level annotations, which are expensive to obtain at scale. In practice, annotations exist at multiple granularities: a small set of images with precise pixel masks, a larger set with bounding boxes, and an even larger set with only image-level labels. This creates a realistic training scenario where the goal is to leverage all available annotations despite their heterogeneous granularity.

Existing mixed-supervision methods assume that weaker annotations are noisy but semantically aligned proxies for the true dense mask. For many tasks, however, this assumption is fundamentally flawed. Weak annotations are **structurally incomplete**: annotators consistently miss small objects, boundary details, and low-contrast instances. In medical imaging, radiologists label obvious large lesions but systematically overlook smaller ones. In natural image datasets, small objects are frequently unannotated. This missingness is not random—it depends on object size, contrast, boundary clarity, and the annotation granularity itself.

We observe that this structure has a direct analogy in the machine learning literature: **positive-unlabeled (PU) learning**. In PU learning, we observe labeled positives and unlabeled data, where the labeling process is non-random and governed by a propensity score $e(x) = P(\text{labeled} \mid \text{positive}, x)$. In multi-granularity segmentation, pixel/box/image labels are all partial observations of the same unknown dense mask, and the labeling propensity depends on visual properties of each pixel.

This observation leads to our framework, **LatentMask**, which makes four contributions:

1. **PU formulation for multi-granularity segmentation.** We identify that mixed-granularity supervision with incomplete labels is structurally a PU learning problem with input-dependent labeling propensity, and derive propensity-corrected unbiased risk estimators for pixel, box, and image-level supervision.

2. **Propensity Network (PropNet).** We propose a lightweight module that estimates per-pixel annotation propensity from encoder features alone—no domain-specific priors required. PropNet is trained via synthetic label corruption that simulates realistic annotation biases (scale-dependent drop, boundary erosion, component drop).

3. **Domain-agnostic framework.** LatentMask integrates PropNet with multi-granularity PU losses, spatial smoothness regularization, and propensity-weighted teacher refinement. It attaches to any encoder-decoder backbone (nnUNet for 3D, DeepLabV3 for 2D).

4. **Multi-domain validation.** We validate on three diverse benchmarks spanning 3D medical CT (pulmonary embolism and liver tumors) and 2D natural images (PASCAL VOC), demonstrating consistent improvement across domains.

Our main finding is that learned propensity is the critical ingredient. Standard PU learning with uniform propensity already improves over naive mixed supervision, validating the PU framing. But LatentMask with PropNet improves substantially further (+PLACEHOLDER points across benchmarks), demonstrating that the structure of label missingness can be learned and exploited.

---

## 2 Related Work

### 2.1 Mixed-Supervision Segmentation

Mixed-supervision methods combine strong and weak labels for training [Papandreou et al., 2015; Dai et al., 2015; Liu et al., 2023]. BoxSup [Dai et al., 2015] uses bounding boxes with GrabCut for pseudo-masks. Recent medical imaging work includes confidence-based knowledge distillation for mixed-quality labels [Liu et al., 2023] and multi-source annotation learning. These methods treat weak labels as noisy surrogates for dense masks, assuming alignment between weak and dense annotations. We show this assumption systematically fails when labels are structurally incomplete, and propose PU learning as a principled alternative.

### 2.2 Semi-Supervised Segmentation

Semi-supervised approaches use consistency regularization [Tarvainen and Valpola, 2017], pseudo-labeling [Sohn et al., 2020], and contrastive learning [Chen et al., 2021]. Recent state-of-the-art methods include UniMatch [Yang et al., 2023] and AugSeg [Zhao et al., 2023]. These methods treat unlabeled data as having uniformly unknown labels. LatentMask differs by modeling the **non-uniform structure** of label missingness through propensity estimation.

### 2.3 Positive-Unlabeled Learning

PU learning addresses classification with positive and unlabeled examples [Elkan and Noto, 2008; Bekker and Davis, 2020]. The non-negative PU risk estimator [Kiryo et al., 2017] prevents overfitting. Extensions include variational PU [Chen et al., 2020] and selection-bias-aware methods [Kato et al., 2019]. PU learning has been applied to image classification and anomaly detection, but has not been formulated for multi-granularity pixel-level segmentation with structured propensity. Our work bridges PU learning theory with practical mixed-supervision segmentation.

### 2.4 Propensity Score Methods

Propensity scoring originates in causal inference [Rosenbaum and Rubin, 1983] for correcting selection bias. In machine learning, propensity-weighted learning has been applied to recommendation [Schnabel et al., 2016] and missing data [Little and Rubin, 2019]. We adapt this idea to segmentation annotations, where the "treatment" is "being labeled" and the confounders are visual features—object size, boundary clarity, and annotation granularity.

### 2.5 Gap Summary

No prior work combines (i) multi-granularity supervision formulated as PU learning, (ii) learned per-pixel propensity estimation for label missingness correction, and (iii) domain-agnostic validation across both medical and natural image segmentation. LatentMask addresses all three.

---

## 3 Problem Setting

We consider training data at three granularities:

- **Pixel-level** $\mathcal{D}_\text{pix}$: small set with dense masks. Propensity $e = 1$ (fully labeled).
- **Box-level** $\mathcal{D}_\text{box}$: medium set with bounding boxes around objects. Propensity varies by pixel.
- **Image-level** $\mathcal{D}_\text{img}$: large set with only presence labels. Propensity is low everywhere.

Let $\mathbf{x}$ denote an input image (2D or 3D) and $\mathbf{z} \in \{0,1\}^{N}$ the unknown dense mask ($N$ = total pixels/voxels). The model predicts:

$$p = f_\theta(\mathbf{x}), \quad e = g_\phi(\text{enc}(\mathbf{x}))$$

where $p$ is the pixel-wise foreground probability and $e$ is the pixel-wise annotation propensity. Crucially, $e$ depends **only on encoder features**—no domain-specific inputs.

For a positive pixel $v$, its labeling propensity is:

$$e(v) = P(\text{labeled} \mid z(v) = 1, v)$$

This captures input-dependent label missingness: small objects, boundary pixels, and low-contrast regions have lower $e(v)$.

Standard supervised learning implicitly assumes $e(v) = 1$ everywhere. PU learning with uniform $e$ assumes constant missingness. LatentMask learns per-pixel $e(v)$, enabling fine-grained correction.

---

## 4 Method

### 4.1 Framework Overview

> **[Figure 1 description]** Method overview diagram. Three input streams (pixel-level, box-level, image-level data) feed into a shared encoder-decoder backbone. A parallel Propensity Network (PropNet) — a lightweight CNN head — takes intermediate encoder features and outputs a per-pixel propensity map $e(x) \in (\epsilon, 1-\epsilon)$. The propensity map modulates the PU-corrected loss for each granularity: pixel data uses standard supervised loss ($e=1$); box and image data use propensity-corrected PU risk that avoids suppressing potentially unlabeled positives. An EMA teacher provides propensity-weighted pseudo-labels for refinement in Stage 3. The diagram shows both 3D (nnUNet) and 2D (DeepLabV3) backbone instantiations.

LatentMask consists of four components:

1. **Shared encoder-decoder backbone** (any architecture: nnUNet for 3D, DeepLabV3 for 2D)
2. **Dense mask head** (predicts $p$)
3. **Propensity Network (PropNet)** — lightweight CNN head on encoder features (predicts $e$)
4. **Propensity-corrected multi-granularity PU risk** — per-granularity loss with propensity weighting

### 4.2 Propensity Network (PropNet)

PropNet is a lightweight CNN that takes intermediate encoder features and outputs per-pixel annotation propensity:

$$e = g_\phi(\text{enc}_k(\mathbf{x})) \in (\epsilon, 1-\epsilon)$$

where $\text{enc}_k$ denotes features from encoder stage $k$ (e.g., $k=3$, approximately $8\times$ downsampled). The architecture is:

$$\text{Conv}_{3 \times 3} \text{(64)} \to \text{BN} \to \text{ReLU} \to \text{Conv}_{3 \times 3} \text{(32)} \to \text{BN} \to \text{ReLU} \to \text{Conv}_{1 \times 1} \text{(1)} \to \sigma$$

with $\epsilon$-clamping to prevent degenerate propensity. For 3D inputs, Conv3d is used; for 2D, Conv2d. Parameter count is ~0.2M (3D) or ~0.05M (2D), negligible compared to the backbone.

**Key properties:**
- Domain-agnostic: takes ONLY encoder features—no vesselness, anatomy maps, or task-specific inputs
- Backbone-agnostic: same head architecture attaches to any encoder-decoder
- Interpretable: the propensity map is a human-readable visualization of where the model expects annotation gaps

**Training via Synthetic Missingness.** Since true propensity is unobservable, we train PropNet on pixel-labeled data using synthetic label corruption:

1. **Scale-dependent drop**: Each connected component $C_i$ in the label is dropped with probability $p_{\text{drop}}(C_i) = \alpha / \sqrt{|C_i|}$, where $|C_i|$ is component area and $\alpha$ controls severity. Mimics the universal bias of missing small objects.

2. **Boundary erosion**: Morphological erosion with random radius $r \sim \text{Uniform}(1, R_{\max})$ applied to label boundaries. Mimics imprecise boundary annotation.

3. **Component drop**: Each connected component is dropped entirely with probability $p_{\text{comp}} \propto 1/|C_i|$. Mimics missed instances.

For each pixel-labeled sample $(\mathbf{x}, \mathbf{z})$, we apply a random combination of these patterns to produce corrupted labels $\tilde{\mathbf{z}}$ and ground-truth propensity $\mathbf{e}^* = P(\text{retained} \mid \text{positive})$. PropNet is trained to predict $\mathbf{e}^*$ from encoder features:

$$L_{\text{prop}} = \text{BCE}(g_\phi(\text{enc}_k(\mathbf{x})),\; \mathbf{e}^*)$$

**Why synthetic training works.** The corruption patterns are not arbitrary—they encode well-documented annotation biases across domains. In medical imaging, small lesions have 40–60% miss rates [citation needed]. In COCO, 30% of small objects are unannotated [citation needed]. The scale-dependent pattern captures this universal bias. PropNet learns to associate encoder features (which encode size, shape, contrast) with annotation likelihood, and this association generalizes to real missingness at test time.

### 4.3 Propensity-Corrected Multi-Granularity PU Risk

We adapt the non-negative PU risk [Kiryo et al., 2017] to pixel-level segmentation with per-pixel propensity for each annotation granularity.

**Pixel-Level Loss** ($e = 1$, standard supervised):

$$L_{\text{pix}} = \text{Dice}(p, y_{\text{pix}}) + \text{CE}(p, y_{\text{pix}})$$

**Box-Level PU Loss.** For box-labeled data with box regions $B$:

> **[Figure 2 description]** Two-panel comparison on box-labeled data. Left: Naive approach — a box annotation covers a large tumor but misses a small satellite lesion outside the box. The naive method pushes the satellite region to background (false negative). Right: PU-corrected approach — PropNet assigns low propensity to the small satellite region (shown as blue heatmap), indicating "annotation likely missing here." The PU loss reduces the negative push in low-propensity regions, preserving the satellite lesion in the final prediction.

$$L_{\text{box}} = \pi \cdot L_{\text{pos}} + \max\left(0,\; L_{\text{unlabeled}} - \pi \cdot L_{\text{neg\_on\_pos}}\right)$$

where:
- $L_{\text{pos}} = \text{CE}(p_v, 1)$ for positive voxels inside $B$
- $L_{\text{neg\_on\_pos}} = \text{CE}(p_v, 0)$ for positive voxels (correction term)
- $L_{\text{unlabeled}} = \frac{1}{\sum_v e(v)} \sum_{v \notin B} e(v) \cdot \text{CE}(p_v, 0)$

The propensity weighting ensures that unlabeled regions with low $e(v)$ (likely containing missed positives) receive reduced negative push.

**Image-Level PU Loss.** For positive images ($y_{\text{img}} = 1$):

$$L_{\text{img}} = \text{BCE}(\text{noisy\_or}(p), 1) + \lambda_{\text{vox}} \cdot \overline{e(v) \cdot \text{CE}(p_v, 0)}$$

For negative images ($y_{\text{img}} = 0$): standard voxel-level negative CE (all pixels confirmed negative).

**Spatial Smoothness Regularization:**

$$L_{\text{smooth}} = \text{TV}(e) = \sum_v \|\nabla e(v)\|_1$$

This encourages spatial coherence—nearby pixels should have similar annotation propensity.

### 4.4 Total Loss and Training Schedule

$$L_{\text{total}} = \lambda_{\text{pix}} L_{\text{pix}} + \lambda_{\text{box}} L_{\text{box}} + \lambda_{\text{img}} L_{\text{img}} + \lambda_{\text{prop}} L_{\text{prop}} + \lambda_{\text{smooth}} L_{\text{smooth}} + \lambda_{\text{ref}} L_{\text{ref}}$$

Training proceeds in three stages:

| Stage | Epochs | Data | Key losses |
|---|---|---|---|
| 1. Warm-up | 1–50 | Pixel only | $L_{\text{pix}} + L_{\text{prop}} + L_{\text{smooth}}$ |
| 2. Joint PU | 51–300 | Pixel + Box + Image | All except $L_{\text{ref}}$ |
| 3. Refinement | 301–400 | All | All (with EMA teacher) |

Stage 2 cycles through granularities: [pixel, pixel, box, image] per 4 iterations. Stage 3 uses an EMA teacher to generate pseudo-labels weighted by propensity: high-propensity, high-activation → confident positive; high-propensity, low-activation → confident negative; low-propensity → skip.

---

## 5 Experiments

### 5.1 Benchmarks

We evaluate on three diverse benchmarks to demonstrate domain-agnostic generality:

**Benchmark 1: Pulmonary Embolism (3D CT, Real Multi-Granularity).**
This is the only benchmark with naturally occurring multi-granularity annotations. Pixel-level: READ-PE (40 exams), CAD-PE (91 scans), FUMPE (35 cases). Box-level: Augmented RSPECT (445 studies, 30K boxes). Image-level: RSPECT (12,195 patients). Internal evaluation: 5-fold CV on READ-PE. External: FUMPE and CAD-PE.

**Benchmark 2: LiTS Liver Tumor (3D CT, Synthetic Multi-Granularity).**
LiTS training (131 cases) degraded to 30% pixel / 30% box / 40% image-level. Test on full pixel labels (70 cases). 3-fold CV.

**Benchmark 3: PASCAL VOC 2012 (2D, Synthetic Multi-Granularity).**
VOC train+aug (~10K images) degraded to 20% pixel / 30% box / 50% image-level. Test on val set (1,449 images) with full pixel labels. DeepLabV3-ResNet50 backbone. 3 independent runs.

### 5.2 Baselines

For each benchmark, we compare against: (1) pixel-only baseline (lower bound), (2) oracle with full pixel labels (upper bound, for synthetic benchmarks), (3) semi-supervised methods (Mean Teacher, CPS/UniMatch), (4) weakly-supervised methods (BoxSup, CAM), (5) mixed naive (all granularities, treat unlabeled as negative), (6) nnPU-Seg (PU with uniform propensity). All methods use the same backbone for fair comparison.

### 5.3 Main Results

**Table 2: PE Segmentation (5-fold CV on READ-PE).** LatentMask achieves XX.X Dice, outperforming mixed naive (+PLACEHOLDER), nnPU-Seg with uniform propensity (+PLACEHOLDER), and all semi-supervised baselines. Largest gains appear on small-lesion recall (+PLACEHOLDER vs mixed naive), confirming that propensity-aware PU correction specifically benefits the regions where annotation incompleteness is most severe.

**Table 4: LiTS Liver Tumors (3-fold CV).** LatentMask achieves XX.X Dice using only 30% pixel labels, recovering XX.X% of the oracle (XX.X). The gap to mixed naive is +PLACEHOLDER, and to nnPU-Seg +PLACEHOLDER. Small-lesion recall improves by +PLACEHOLDER over mixed naive, consistent with the PE finding.

**Table 6: PASCAL VOC (3 runs).** LatentMask achieves XX.X mIoU using only 20% pixel labels, outperforming mixed naive (+PLACEHOLDER), UniMatch semi-supervised (+PLACEHOLDER), and BoxSup (+PLACEHOLDER). Small-object IoU improves by +PLACEHOLDER over mixed naive. This confirms that the PU + PropNet framework generalizes beyond medical imaging.

### 5.4 Ablation Studies

> **[Figure 3 description]** A 3×3 grid showing PropNet propensity maps across three domains. Row 1: PE — propensity is high in large proximal emboli regions and low in small subsegmental regions, consistent with clinical annotation patterns. Row 2: LiTS — propensity is high for large tumors and low for small metastases, matching the scale-dependent bias. Row 3: VOC — propensity is high for large objects (e.g., person, car) and low for small objects (e.g., bottle, plant), matching annotation completeness patterns. All three domains show that PropNet learns meaningful, interpretable propensity without domain-specific priors.

**Component ablation (Tables 3, 5, 7).** Consistent across all three benchmarks:
- A0 → A1 (+ PU with uniform $e$): +PLACEHOLDER points. The PU framework itself provides value.
- A1 → A2 (+ PropNet): +PLACEHOLDER points. Learned propensity significantly improves over uniform.
- A2 → A3 (+ smoothness): +PLACEHOLDER points. Spatial regularization helps PropNet generalization.
- A3 → A4 (+ EMA refinement): +PLACEHOLDER points. Propensity-weighted pseudo-labels further refine.

**Synthetic missingness patterns (Table 9).** Scale-dependent drop contributes most (+PLACEHOLDER Dice vs uniform), followed by component drop (+PLACEHOLDER) and boundary erosion (+PLACEHOLDER). All three combined yields the best result (+PLACEHOLDER vs uniform), suggesting the patterns are complementary.

### 5.5 External Generalization

**Table 8: PE cross-dataset evaluation.** LatentMask trained on READ-PE + Aug-RSPECT + RSPECT generalizes well to FUMPE (XX.X Dice) and CAD-PE (XX.X Dice), outperforming mixed naive by +PLACEHOLDER and +PLACEHOLDER respectively. The improvement is larger on external data than internal, suggesting PropNet learns robust propensity patterns rather than overfitting to training distribution.

### 5.6 Annotation Budget Efficiency

> **[Figure 5 description]** Line plot showing Dice score (y-axis) vs annotation budget in cost units (x-axis) for three strategies on PE: pixel-only (red), mixed naive (orange), LatentMask (blue). LatentMask consistently achieves the same Dice at lower cost, e.g., LatentMask at budget 200 ≈ mixed naive at budget 400 ≈ pixel-only at budget 800. The gap widens at lower budgets, demonstrating that PropNet-based PU correction is most valuable in annotation-scarce regimes.

### 5.7 Robustness Analysis

**Table 11: Varying missingness severity (LiTS).** LatentMask's advantage over mixed naive INCREASES as labels become more incomplete: +PLACEHOLDER at mild degradation (50/30/20) vs +PLACEHOLDER at extreme degradation (5/15/80). This confirms that propensity correction is most valuable when annotation incompleteness is severe—exactly the regime where multi-granularity supervision is most needed.

---

## 6 Discussion

**PU learning as a unifying lens for mixed supervision.** Our key conceptual contribution is reframing multi-granularity supervision as positive-unlabeled learning. This provides a principled framework that replaces ad-hoc loss weighting with theoretically grounded risk correction. The PU formulation naturally accommodates any annotation granularity and explains why naive mixed supervision systematically fails.

**What PropNet learns.** Without any domain-specific priors, PropNet consistently learns to assign lower propensity to small objects, boundary regions, and low-contrast areas across all three domains (Figure 3). This aligns with well-documented annotation biases and confirms that encoder features carry sufficient information for propensity estimation.

**Limitations.** (1) PropNet training relies on synthetic missingness patterns that may not perfectly match real-world annotation distributions. Developing adaptive or learned corruption patterns is an important direction. (2) The current framework assumes binary segmentation; extending to multi-class segmentation with class-specific propensity is future work. (3) Domain-specific auxiliary inputs (e.g., vesselness maps for PE) can optionally improve PropNet (Table 12), but we leave systematic exploration of such enhancements to future work.

**Broader impact.** LatentMask reduces the annotation burden for segmentation tasks by effectively leveraging cheap weak labels. This has positive implications for democratizing medical imaging AI, where annotation is a critical bottleneck. No negative societal implications are anticipated.

---

## 7 Conclusion

We introduced LatentMask, a framework that reframes multi-granularity segmentation as propensity-corrected positive-unlabeled learning. The Propensity Network (PropNet) learns where annotations are likely missing from encoder features alone, enabling principled PU risk correction across pixel, box, and image-level supervision. Validated on three diverse benchmarks—3D pulmonary embolism, 3D liver tumors, and 2D PASCAL VOC—LatentMask consistently outperforms naive mixed supervision, semi-supervised methods, and uniform PU baselines, with largest gains on small objects and annotation-scarce regimes. The framework is backbone-agnostic, domain-agnostic, and requires no task-specific priors, making it a general solution for practical mixed-supervision segmentation.

---

## References

> **Note:** All references marked [citation needed] require verification. Citations below are placeholders synchronized with the related work section. Use `04_verified_references.md` for verified BibTeX.

- [Bekker and Davis, 2020] J. Bekker, J. Davis. Learning from positive and unlabeled data: a survey. Machine Learning, 2020.
- [Chen et al., 2020] H. Chen, F. Liu, Y. Wang, L. Zhao, H. Wu. A variational approach for learning from positive and unlabeled data. NeurIPS, 2020.
- [Chen et al., 2021] X. Chen, Y. Yuan, G. Zeng, J. Wang. Semi-supervised semantic segmentation with cross pseudo supervision. CVPR, 2021.
- [Dai et al., 2015] J. Dai, K. He, J. Sun. BoxSup: Exploiting bounding boxes to supervise convolutional networks for semantic segmentation. ICCV, 2015.
- [Elkan and Noto, 2008] C. Elkan, K. Noto. Learning classifiers from only positive and unlabeled data. KDD, 2008.
- [Kato et al., 2019] M. Kato, T. Teshima, J. Honda. Learning from positive and unlabeled data with a selection bias. ICLR, 2019.
- [Kiryo et al., 2017] R. Kiryo, G. Niu, M. du Plessis, M. Sugiyama. Positive-unlabeled learning with non-negative risk estimator. NeurIPS, 2017.
- [Liu et al., 2023] Y. Liu et al. Mixed supervision for medical image segmentation. Medical Image Analysis, 2023.
- [Little and Rubin, 2019] R. Little, D. Rubin. Statistical Analysis with Missing Data. Wiley, 2019.
- [Papandreou et al., 2015] G. Papandreou, L. Chen, K. Murphy, A. Yuille. Weakly- and semi-supervised learning of a deep convolutional network for semantic image segmentation. ICCV, 2015.
- [Rosenbaum and Rubin, 1983] P. Rosenbaum, D. Rubin. The central role of the propensity score in observational studies for causal effects. Biometrika, 1983.
- [Schnabel et al., 2016] T. Schnabel, A. Swaminathan, A. Singh, N. Chandak, T. Joachims. Recommendations as treatments: Debiasing learning and evaluation. ICML, 2016.
- [Sohn et al., 2020] K. Sohn, D. Berthelot, N. Carlini, Z. Zhang, H. Zhang, C. Raffel, E. Cubuk, A. Kurakin, C. Li. FixMatch: Simplifying semi-supervised learning with consistency and confidence. NeurIPS, 2020.
- [Tarvainen and Valpola, 2017] A. Tarvainen, H. Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised learning results. NeurIPS, 2017.
- [Yang et al., 2023] L. Yang, L. Zhuo, L. Qi, Y. Shi, Y. Gao. St++: Make self-training work better for semi-supervised semantic segmentation. CVPR, 2023.
- [Zhao et al., 2023] Z. Zhao et al. AugSeg: Maximizing the utility of full-image augmentations for semi-supervised semantic segmentation. ICCV, 2023.

---

## Appendix (outline)

### A. Proof of Unbiased Risk Property

Sketch: Under the assumption that PropNet correctly estimates the true propensity, the propensity-corrected PU risk is an unbiased estimator of the fully-supervised risk. Formal statement and proof following Kiryo et al. (2017) extended to pixel-level segmentation.

### B. Synthetic Missingness Pattern Details

Full specification of the three corruption patterns, including hyperparameter ranges and the random combination strategy.

### C. Implementation Details

- 3D backbone: nnUNet v2, 3d_fullres configuration, 400 epochs total
- 2D backbone: DeepLabV3-ResNet50, ImageNet pre-trained, 80 epochs
- PropNet: Conv(64)→Conv(32)→Conv(1), attached to encoder stage 3
- Optimizer: SGD (3D) / AdamW (2D), with cosine annealing
- Loss weights: $\lambda_{\text{pix}}=1.0$, $\lambda_{\text{box}}=1.0$, $\lambda_{\text{img}}=0.5$, $\lambda_{\text{prop}}=0.5$, $\lambda_{\text{smooth}}=0.1$, $\lambda_{\text{ref}}=0.4$

### D. Additional Qualitative Results

> **[Figure 4 description]** Multi-panel qualitative comparison across three domains. Each row shows: input image, ground truth, mixed naive prediction, nnPU-Seg prediction, LatentMask prediction, and PropNet propensity map. Row 1 (PE): LatentMask correctly detects a small subsegmental embolus missed by all baselines, with PropNet showing low propensity in that region. Row 2 (LiTS): LatentMask recovers a small liver metastasis that mixed naive misses. Row 3 (VOC): LatentMask correctly segments a small bottle that baselines miss, with corresponding low propensity.

### E. Per-Class VOC Results

Full 21-class IoU breakdown.

### F. Propensity Calibration Analysis

Expected calibration error (ECE) of PropNet propensity on synthetic benchmarks where ground-truth propensity is known.
