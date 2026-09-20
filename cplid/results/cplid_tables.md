# CPLID real experimental results (all numbers produced in this run)

Hardware: NVIDIA GeForce RTX 5060 Laptop GPU. Every variant trained from identical COCO-pretrained YOLOv8s weights under one protocol; the only difference is the module inserted at P4 (or in the neck for MAFF).

## Table A. CPLID test split (84 images, 156 instances)

| Variant | P (%) | R (%) | mAP@0.5 (%) | Δ vs base (pp) | mAP@0.5:0.95 (%) | Δ vs base (pp) |
|---|---|---|---|---|---|---|
| YOLOv8s (baseline, unmodified) | 95.69 | 98.11 | 97.80 | — | 81.72 | — |
| + SPPF at P4 | 97.10 | 98.11 | 98.39 | +0.59 | 80.79 | -0.93 |
| + LKA at P4 (VAN-style, no deformable) | 96.97 | 98.11 | 98.71 | +0.91 | 82.67 | +0.96 |
| + Deformable only at P4 (3x3 modulated DCN) | 97.05 | 98.01 | 98.51 | +0.71 | 81.87 | +0.15 |
| + DLKA at P4 (deformable inside large kernel) | 96.59 | 96.97 | 98.57 | +0.78 | 81.32 | -0.40 |
| + MAFF (neck, 3 levels) | 96.67 | 97.35 | 98.15 | +0.36 | 81.81 | +0.09 |
| MADNet (DLKA + MAFF) | 96.79 | 97.07 | 98.30 | +0.50 | 82.10 | +0.38 |

## Table B. Efficiency (uniform protocol: batch 1, 640x640, FP32, no Conv-BN fusion,
60 warm-up + 300 timed forward passes, pre/post-processing excluded)

| Variant | Params (M) | FLOPs (G) | FPS | FPS vs baseline |
|---|---|---|---|---|
| YOLOv8s (baseline, unmodified) | 11.136 | 28.64 | 155.1 | — |
| + SPPF at P4 | 11.301 | 29.17 | 150.7 | 0.97x |
| + LKA at P4 (VAN-style, no deformable) | 11.222 | 28.91 | 143.0 | 0.92x |
| + Deformable only at P4 (3x3 modulated DCN) | 11.789 | 28.85 | 124.3 | 0.80x |
| + DLKA at P4 (deformable inside large kernel) | 11.287 | 29.12 | 44.3 | 0.29x |
| + MAFF (neck, 3 levels) | 11.181 | 28.65 | 111.3 | 0.72x |
| MADNet (DLKA + MAFF) | 11.331 | 29.12 | 39.7 | 0.26x |

## Table C. DLKA against practical drop-in alternatives at the same position (P4)

Directly answers the request to compare against SPPF, VAN-style LKA and deformable-only
rather than against a hypothetical dense 23x23 convolution.

| Block at P4 | mAP@0.5 (%) | Δ vs baseline (pp) | Δ vs DLKA (pp) | Params (M) | FLOPs (G) | FPS |
|---|---|---|---|---|---|---|
| YOLOv8s (baseline, unmodified) | 97.80 | — | -0.78 | 11.136 | 28.64 | 155.1 |
| + SPPF at P4 | 98.39 | +0.59 | -0.18 | 11.301 | 29.17 | 150.7 |
| + LKA at P4 (VAN-style, no deformable) | 98.71 | +0.91 | +0.13 | 11.222 | 28.91 | 143.0 |
| + Deformable only at P4 (3x3 modulated DCN) | 98.51 | +0.71 | -0.06 | 11.789 | 28.85 | 124.3 |
| + DLKA at P4 (deformable inside large kernel) | 98.57 | +0.78 | — | 11.287 | 29.12 | 44.3 |

## Table D. Per-class AP@0.5 on the test split (%)

| Variant | defect | insulator |
|---|---|---|
| YOLOv8s (baseline, unmodified) | 99.50 | 96.10 |
| + SPPF at P4 | 99.50 | 97.28 |
| + LKA at P4 (VAN-style, no deformable) | 99.50 | 97.92 |
| + Deformable only at P4 (3x3 modulated DCN) | 99.50 | 97.52 |
| + DLKA at P4 (deformable inside large kernel) | 99.50 | 97.65 |
| + MAFF (neck, 3 levels) | 99.50 | 96.81 |
| MADNet (DLKA + MAFF) | 99.50 | 97.10 |

## Protocol

- CPLID, 848 images, split 594 / 170 / 84 (train / val / test), 2 classes
- Training-split annotations: insulator 920, defect 174 (instance imbalance 5.29 : 1)
- Identical COCO-pretrained YOLOv8s initialisation for every variant
- 100 epochs, SGD lr0 0.01, momentum 0.937, weight decay 5e-4, cosine schedule, 3 warm-up epochs
- Input 640x640, batch 16, seed 0, early stopping disabled, mosaic closed for the last 10 epochs
- Augmentation: fliplr 0.5, scale 0.5, HSV (0.015 / 0.7 / 0.4), translate 0.1
- Test split held out; never used for model selection
- Single seed per variant; no standard deviations are reported and none are claimed

## Observations that the numbers support

1. A correctly trained, unmodified YOLOv8s already reaches **97.80% mAP@0.5** on this benchmark.
2. Every inserted block moves mAP@0.5 by less than 1 pp in either direction, and the spread across all seven variants is 0.91 pp. With a single seed and an 84-image test split, differences of this size cannot be attributed to the modules.
3. FLOPs do not capture the cost of deformable sampling: DLKA adds only 0.48 G FLOPs over the baseline yet runs at 0.29x its throughput.
4. The *defect* class saturates at 99.50 AP for every variant, so all remaining headroom is in the *insulator* class.