# Plan and pre-registered hypotheses

**Written 2026-11-03, before any Part A experiment was run.** The thresholds below are fixed. If a later commit changes
one, the commit message says why, and the report states both the old and the new value. The git history is the record.

Part B's hypotheses (H7–H10) were frozen in a separate commit after Part A and **before** notebook 08 was run.

---

## 0. What we already know (Degrade → Restore → Detect, notebooks 01–06 and 05b)

| Fact | Number | Source |
|---|---|---|
| Clean mAP@[.5:.95] of YOLOv8n on the fixed 300-image COCO val2017 subset | 0.387 | nb 05 |
| DnCNN wins back this share of the mAP lost to Gaussian noise | 70–83% (σ = 15 … 50) | nb 05 |
| BM3D wins back about the same, with 0.6–1.8 dB lower PSNR | 70–76%, statistically tied | nb 05b |
| Low light: CLAHE has the highest PSNR among single methods but the lowest mAP | e.g. ℓ = 0.8: CLAHE 0.095 vs none 0.121 | nb 05 |
| Low light: the best pipeline is brighten, then denoise | gamma → DnCNN: 0.242 at ℓ = 0.8 | nb 05b |
| Spearman(PSNR, mAP) across all runs | 0.82, but reversed inside low light | nb 05 |
| Per-image PSNR *gain* vs per-image AP *gain* | Spearman −0.14 to +0.21 | nb 05b |
| Per-image PSNR vs per-image AP *loss* inside one run (computed for this plan from `prior/05b/per_image_ap.csv`) | mean \|ρ\| ≈ 0.08, max 0.26 | this plan |

So PSNR ranks restorers well for noise and compression, ranks them badly for low light, and says almost nothing about
which images a restorer helps. The question for this project:

> **What does damage do inside the detector, and can a restorer be trained to undo *that* instead of the pixel error?**

---

## 1. Part A: feature-space analysis (notebook 07, no training)

### 1.1 Setup

* **Detector:** YOLOv8n (COCO weights), frozen, `predict()` with the same settings as notebook 05 (conf 0.001, IoU 0.7,
  max_det 300). Forward hooks record the input tensor and the output of every layer except Upsample, Concat and Detect:
  layers 0–9 (backbone + SPPF), 12, 15, 16, 18, 19, 21 (neck). **Neck outputs = layers 15, 18, 21** (the P3/P4/P5
  maps that feed the Detect head).
* **Images:** the same 300 COCO val2017 images and the same damage seeds as notebooks 05/05b. The notebook checks that
  the subset and the per-image AP match 05b exactly.
* **Distances** (per image, per layer, clean vs other; letterbox padding cropped away):
  * `cos` = 1 − mean over positions of the cosine similarity of channel vectors (**primary metric**),
  * `cka` = 1 − linear CKA with positions as samples (Kornblith et al., 2019),
  * `rel_l2` = ‖F − F_clean‖² / ‖F_clean‖²,
  * plus `norm_ratio` = ‖F‖ / ‖F_clean‖ (louder or quieter activations).
* **Neck distance** of an image = mean of its `cos` distance at layers 15, 18, 21. It is fixed in advance so that the
  main tests involve no layer selection.
* **Feature-damage removal** of a restorer at layer l = 1 − mean d_l(restored) / mean d_l(damaged). Pixel removal is
  the same ratio for pixel MSE. mAP recovery = (mAP_restored − mAP_damaged) / (clean mAP − mAP_damaged).

### 1.2 Runs (36)

| Damage | Levels | Restorers |
|---|---|---|
| clean | – | none |
| Gaussian noise | σ = 15, 25, 50 | none, classical (tuned), DnCNN; BM3D at σ = 25, 50 |
| JPEG | q = 30, 10 | none, classical (tuned), DnCNN |
| Low light | ℓ = 0.6, 0.8 | none, classical, DnCNN, CLAHE-L, gamma, gamma → DnCNN, DnCNN → gamma |
| **Dark only** (new: the low-light curve with *no* noise) | ℓ = 0.8 | none, gamma, CLAHE-L, DnCNN |
| Gaussian, second noise seed (new: noise ceiling) | σ = 25 | none |

The second-seed run measures how repeatable per-image AP loss is at all: the Spearman correlation between the AP losses
of two independent noise draws on the same images is a ceiling for any per-image predictor. It is reported, not tested.

### 1.3 Hypotheses

Each has one pass/fail rule. Notebook 07 computes every verdict in code (`rftm/analysis.py::scorecard`).

**H1. Where damage enters the network.**
* **H1a.** For Gaussian noise (σ = 25 and 50, no restoration) the `cos` distance is largest at one of layers 0–2, and
  the neck distance is smaller than the layer-0 distance. *Reasoning: white noise is mostly high frequency, and every
  stride-2 stage averages some of it away.*
* **H1b.** JPEG q = 10 damage survives depth better than Gaussian σ = 50: the ratio neck distance / layer-0 distance is
  higher for JPEG q = 10. *Reasoning: block artifacts are structured and low frequency, so downsampling keeps them.*

**H2. Does DnCNN restore deep features?** (Gaussian, σ = 15, 25, 50)
* **H2a.** DnCNN removes ≥ 60% of the neck damage at every σ.
* **H2b.** It removes a smaller share of the neck damage than of the pixel MSE, at every σ. *(The error it leaves
  behind matters more to YOLO than its pixel size suggests.)*
* **H2c.** Its neck removal is within ±15 percentage points of its mAP recovery at every σ (71–82% in nb 05).

**H3. Why CLAHE fails in low light.**
* **H3a.** At ℓ = 0.6 and 0.8, CLAHE lowers pixel MSE but *raises* the mean `cos` distance at layers 0–4
  (early-layer removal < 0).
* **H3b.** Darkness alone is not the main problem: the dark-only ℓ = 0.8 images lose less than half the mAP that full
  low light ℓ = 0.8 loses (dark-only mAP ≥ 0.387 − ½ × (0.387 − 0.121) ≈ 0.254).
* **H3c.** On dark-only images, both gamma and CLAHE *raise* mAP over doing nothing (brightening helps once there is
  no noise to amplify), the opposite of what they do on real low light.
* **H3d.** Among low-light restorers, gamma → DnCNN has the smallest neck distance at both levels (it had the best mAP).

**H4. Does feature distance predict per-image AP better than PSNR?**
* **H4a.** Inside each damaged run, Spearman(neck distance, per-image AP loss) averages ≥ 0.30 over all runs, and is
  higher than Spearman(−PSNR, AP loss) in ≥ 75% of runs.
* **H4b.** Inside each condition, for each restorer against no restoration: Spearman(drop in neck distance, per-image
  AP gain) averages ≥ 0.30, and beats Spearman(PSNR gain, AP gain) in ≥ 75% of restorer pairs.
* Secondary, not a pass/fail test: the best single layer, chosen on the even-position half of the images and scored
  on the odd half; the same for `cka` and `rel_l2`.

**H5. BM3D vs DnCNN.** At σ = 25 and 50, BM3D's neck distance is within ±15% of DnCNN's, despite the PSNR gap.
*(Feature space should explain the mAP tie that PSNR misses.)*

**H6. Ranking restorers.** In the 7 damaged conditions with more than one restorer (Gaussian 15/25/50, JPEG 30/10,
low light 0.6/0.8), the restorer with the smallest mean neck distance is the one with the highest mAP in ≥ 6 of 7
conditions, and in more conditions than the highest-PSNR restorer manages.

### 1.4 What Part A hands to Part B

Rule fixed now: among {neck mean, layers 0–4 mean, best single layer (chosen on the even half)} × {`cos`, `rel_l2`},
Part B's loss uses the option with the highest held-out (odd-half) Spearman with AP loss **on the Gaussian runs**
(Part B trains on Gaussian noise). Notebook 07 writes it to `results/07/layers_for_partB.json`.

If H4a fails badly (mean ρ < 0.15 for every option), the feature distance is not a good proxy for detection, and Part
B is changed to use the detection loss itself (YOLO's training loss on the clean image's boxes) as the task term, with
the feature loss as a comparison.

---

## 2. Part B: task-driven restoration (notebook 08) · **frozen 2026-11-03, after Part A, before notebook 08 ran**

Changes from the provisional version are marked *(changed)*, with the reason.

### 2.1 Setup

* **Start:** `dncnn_last.pt`. **Detector in the loss:** YOLOv8n, frozen, eval mode (BatchNorm statistics fixed);
  gradients flow through it into the denoiser only.
* **Feature term (from Part A's rule, `results/07/layers_for_partB.json`):** cosine distance at **layer 21** (P5),
  computed on the restored vs the clean crop. Part A found three options within 0.01 of each other, so one extra
  model uses the **neck mean (layers 15, 18, 21)** at the chosen λ\*, as a comparison (exploratory).
* **Loss** *(changed: both terms normalised, so that "λ = 1" really means equal weight at the start)*:
  L = MSE / MSE₀ + λ · D / D₀, where MSE₀ and D₀ are the mean values of the two terms for the *starting* DnCNN on
  20 fixed training batches. The same constants are used for every λ.
* **Data:** COCO val2017 images **not** in the 300-image test subset (4,700). Shuffled with seed 0: 4,200 for training,
  500 for choosing λ. Blind Gaussian noise σ ~ U(0, 55) per crop, as in notebook 04. Random 320 × 320 crops with
  horizontal flips, batch 8.
* **Training** *(changed: a fixed step budget; no step selection, to keep the λ comparison simple)*: Adam, lr 1e-4,
  2,000 steps, lr × 0.1 for the last 500, gradient-norm clip 1.0. Every λ uses the same seed, so all runs see the
  same crops and noise in the same order.
* **Control:** λ = 0 (MSE only, same data, same steps, same seed). Every claim compares against this control.
* **Sweep:** λ ∈ {0, 0.03, 0.1, 0.3, 1, 3}.
* **Choosing λ\*** *(changed: made explicit, and done on the validation split so the test images are not used to
  pick it)*: the λ > 0 with the highest YOLOv8n mAP on the 500 validation images at σ = 50, among those whose
  validation PSNR at σ = 50 is at most 1 dB below the control's.
* **Test:** the 300-image subset (same images and noise seeds as notebooks 05/07) at σ = 0, 15, 25, 50: PSNR and
  mAP for YOLOv8n, **YOLOv8s** (same family, bigger: near transfer) and **Faster R-CNN ResNet-50-FPN v2**
  (torchvision, COCO weights: far transfer, never used in training). Paired bootstrap over images (2,000
  resamples of image-averaged AP, as 05b) for each model against the control.

### 2.2 Hypotheses

* **H7.** At σ = 50, λ\* raises YOLOv8n mAP by ≥ 0.010 over the λ = 0 control, the bootstrap 95% CI of the
  image-averaged AP difference is above 0, and λ\*'s test PSNR is at most 1 dB below the control's.
* **H8.** At σ = 50, λ\*'s Faster R-CNN mAP gain over the control is ≥ 50% of its YOLOv8n gain, and the bootstrap
  95% CI is above 0. If H7 holds and H8 fails, the denoiser is fitting YOLOv8n specifically. YOLOv8s is reported
  alongside, not tested.
* **H9.** The test YOLOv8n mAP at σ = 50, across λ, is highest at some λ below the largest (3): too much feature
  weight lets the output drift from the image.
* **H10.** λ\* does not hurt clean images: YOLOv8n mAP at σ = 0 is at least the control's minus 0.005.

Exploratory, no thresholds: the PSNR–mAP curve over λ for each detector and σ; neck mean vs layer 21 at λ\*;
high-frequency energy of the restoration error (a sign of adversarial-looking patterns).

## 3. Related work to read (and what to take from each)

| Paper | What to look for |
|---|---|
| D. Liu, B. Wen, X. Liu, Z. Wang, T. Huang, *When Image Denoising Meets High-Level Vision Tasks: A Deep Learning Approach*, IJCAI 2018 | Their joint loss (reconstruction + high-level task loss) and the claim that a denoiser trained with one high-level network transfers to others. That claim is what H8 tests for detection. |
| S. Diamond, V. Sitzmann, F. Julca-Aguilar, S. Boyd, G. Wetzstein, F. Heide, *Dirty Pixels: Towards End-to-End Image Processing and Perception*, ACM TOG 2021 (arXiv 2017) | End-to-end training of the image-processing stage with the classifier, especially in low light; how their PSNR and accuracy move in opposite directions. |
| S. Kornblith, M. Norouzi, H. Lee, G. Hinton, *Similarity of Neural Network Representations Revisited*, ICML 2019 | Linear CKA, its invariances, and its bias when samples ≈ features (relevant at P5, where 300 positions meet 256 channels). |
| J. Johnson, A. Alahi, L. Fei-Fei, *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*, ECCV 2016 | Feature-reconstruction loss: the template for Part B's loss. |
| T. Son, J. Kang, N. Kim, S. Cho, S. Kwak, *URIE: Universal Image Enhancement for Visual Recognition in the Wild*, ECCV 2020 | An enhancer trained with a recognition loss and tested on other tasks and networks. |
| C. Michaelis et al., *Benchmarking Robustness in Object Detection*, 2019 | COCO-C: how detectors degrade under corruptions, for context. |

Bibliographic details are from memory and are checked against the PDFs before they go into `paper/refs.bib`.

---

## 4. Risks and how they are handled

* **Per-image AP is coarse** (images with 1–3 objects jump between a few values). The second-seed run gives a noise
  ceiling, so ρ = 0.3 can be read against what is achievable.
* **Multiple comparisons.** Thirteen Part A verdicts; none is promoted to "the finding" after the fact. Exploratory
  results (per-layer curves, CKA, `rel_l2`) are labelled exploratory in the report.
* **Fooling the detector in Part B.** Handled by the λ = 0 control, the held-out second detector (H8) and a check on
  clean images (H10).
* **Compute.** Notebook 07 took ~4 h on a T4 (3.4 h of it BM3D on the CPU). Notebook 08: ~15–20 min of training per λ, plus ~1.5 h of evaluation.
