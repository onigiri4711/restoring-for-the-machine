# Restoring for the Machine

**What does image damage do *inside* an object detector, and can a restorer be trained to undo that instead of the
pixel error?**

A follow-up to [Degrade → Restore → Detect](https://github.com/onigiri4711/degrade-restore-detect), which found that
a blind DnCNN wins back 70–83% of the mAP YOLOv8n loses to Gaussian noise, that BM3D ties it for detection despite
0.6–1.8 dB lower PSNR, that CLAHE has the best PSNR but the worst mAP in low light, and that per-image PSNR gain says
almost nothing about per-image detection gain (Spearman −0.14 to +0.21).

📋 **Plan and pre-registered hypotheses:** [`PLAN.md`](PLAN.md) (written before any experiment; the git history is the record)

## Two parts

| Part | Notebook | What it does | Status |
|---|---|---|---|
| **A · Feature-space analysis** | `07_feature_analysis.ipynb` | Hooks every layer of YOLOv8n and measures how far damaged / restored features are from clean (cosine, linear CKA, relative L2). Where does damage enter? Does DnCNN restore deep features? Why does CLAHE fail? Does feature distance predict per-image AP loss better than PSNR? | done: 8 of 13 hypotheses supported (see the report) |
| **B · Task-driven restoration** | `08_task_driven_dncnn.ipynb` | Fine-tunes DnCNN with pixel MSE + λ × YOLO feature distance (YOLO frozen), sweeps λ, plots PSNR against mAP, and checks on a second detector (Faster R-CNN) that the gain is real and not the denoiser fooling YOLOv8n. | done: 2 of 4 hypotheses supported; +0.01–0.015 mAP at σ = 50 that carries over to Faster R-CNN (see the report) |

## Repository layout

```
PLAN.md                 questions, design, hypotheses H1–H10 with pass/fail rules, related work
rftm/                   shared code
  degradations.py       the seven seeded damage models (identical copy from the first project)
  restorers.py          DnCNN, tuned classical filters, CLAHE-L, auto-gamma, BM3D, chains, dark-only damage
  coco.py               COCO val2017 subset and YOLO -> COCO detections / mAP / per-image AP
  features.py           forward hooks on YOLOv8n, letterbox crop, cosine / CKA / relative-L2 distances
  analysis.py           all Part A analysis on plain tables, including the hypothesis scorecard
  taskloss.py           Part B: differentiable YOLO features, feature loss, COCO crops, the training loop
  detectors.py          Part B test detectors: YOLOv8n / YOLOv8s and Faster R-CNN R50-FPN v2
  analysis8.py          Part B analysis: choosing lambda*, paired bootstrap, scorecard for H7–H10
  plots.py              Part A figures
  kaggle.py             finding inputs on Kaggle
prior/                  results carried over from the first project (inputs and reference numbers)
  01/best_classical.json    tuned classical filter per (damage, level)
  04/dncnn_eval.csv         DnCNN PSNR on CBSD68 / BSD
  05/, 05b/                 mAP sweeps, per-image AP, CIs; 05/subset_ids.json is the frozen 300-image subset
notebooks/              Kaggle notebooks (07 = Part A, 08 = Part B)
results/07/, results/08/ outputs of notebooks 07 and 08 (copied from Kaggle after each run)
figures/                figures for the report
tests/                  test_analysis.py (numpy only), test_features.py (needs torch)
docs/                   notes
```

## Running

Notebooks run on **Kaggle (GPU T4, Internet ON)**. Notebook 07:

1. Add Input → Datasets → `shourjabanerjee/dcnn-last` (the DnCNN checkpoint).
2. Nothing else: the first cell clones this repository. (Or upload the repo as a Kaggle dataset and attach it; it is
   found automatically.)
3. Run once with `QUICK = True` (~10 min, 40 images) to check the pipeline, then with `QUICK = False` as a commit
   (Save Version → Save & Run All, ~1.5–2 h).
4. Download `/kaggle/working/outputs/07/` into `results/07/`.

Notebook 08 works the same way (same dataset input; ~4–5 h; `QUICK = True` first). It deletes COCO from its
working folder at the end, so its output downloads in one go.

Tests: `python tests/test_analysis.py` and `python tests/test_analysis8.py` (no GPU needed), `python tests/test_features.py` (needs torch).

## Reproducibility

Same 300 COCO val2017 images, same damage seeds, same YOLOv8n settings as the first project. Notebook 07 checks the
subset id list, the clean mAP (0.387) and the per-image AP of every shared run against `prior/05b/`.

## References

- D. Liu et al., *When Image Denoising Meets High-Level Vision Tasks: A Deep Learning Approach*, IJCAI 2018.
- S. Diamond et al., *Dirty Pixels: Towards End-to-End Image Processing and Perception*, ACM TOG 2021.
- S. Kornblith et al., *Similarity of Neural Network Representations Revisited*, ICML 2019.
- K. Zhang et al., *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*, IEEE TIP 2017.

Author: Shourja Banerjee
