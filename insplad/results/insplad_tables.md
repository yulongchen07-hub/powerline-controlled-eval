# InsPLAD-det real experimental results

Hardware: NVIDIA GeForce RTX 5060 Laptop GPU. All variants start from identical COCO-pretrained YOLOv8s weights and share one protocol; only the inserted module (or the classification loss, for CWSL) differs.

## Table 1. InsPLAD-det held-out test split (2,626 images, 6,324 instances)

| Variant | P (%) | R (%) | mAP@0.5 (%) | Δ (pp) | mAP@0.5:0.95 (%) | Δ (pp) |
|---|---|---|---|---|---|---|
| YOLOv8s (baseline) | 89.66 | 86.21 | 89.29 | — | 73.31 | — |
| + SPPF at P4 | 86.82 | 84.20 | 87.57 | -1.72 | 70.40 | -2.91 |
| + LKA at P4 (VAN-style) | 87.63 | 85.36 | 87.90 | -1.39 | 71.90 | -1.42 |
| + Deformable only at P4 | 86.38 | 86.91 | 89.25 | -0.05 | 72.48 | -0.83 |
| + DLKA at P4 | 88.72 | 85.75 | 89.23 | -0.06 | 72.10 | -1.21 |
| + MAFF (neck) | 89.05 | 84.44 | 88.94 | -0.36 | 72.36 | -0.96 |
| + CWSL (loss only) | 88.44 | 84.67 | 88.15 | -1.15 | 72.61 | -0.70 |
| MADNet (DLKA + MAFF + CWSL) | 89.75 | 83.63 | 89.27 | -0.02 | 72.98 | -0.33 |

## Table 2. Head vs tail performance (the long-tail question)

Head = 7 classes with >= 1000 training instances; tail = 7 classes with < 200. Training-split imbalance is 234 : 1.

| Variant | head mAP@0.5 (%) | tail mAP@0.5 (%) | head-tail gap (pp) | Δ tail vs baseline (pp) |
|---|---|---|---|---|
| YOLOv8s (baseline) | 97.23 | 76.99 | 20.24 | — |
| + SPPF at P4 | 97.06 | 73.02 | 24.03 | -3.97 |
| + LKA at P4 (VAN-style) | 96.82 | 74.04 | 22.77 | -2.95 |
| + Deformable only at P4 | 97.08 | 77.04 | 20.04 | +0.04 |
| + DLKA at P4 | 97.02 | 77.06 | 19.96 | +0.07 |
| + MAFF (neck) | 97.27 | 76.08 | 21.19 | -0.91 |
| + CWSL (loss only) | 97.17 | 74.28 | 22.89 | -2.71 |
| MADNet (DLKA + MAFF + CWSL) | 96.87 | 77.32 | 19.55 | +0.32 |

## Table 3. Efficiency (batch 1, 640x640, FP32, no Conv-BN fusion, 50 warm-up + 200 timed passes)

| Variant | Params (M) | FLOPs (G) | FPS | FPS vs baseline | Train time (min) |
|---|---|---|---|---|---|
| YOLOv8s (baseline) | 11.143 | 28.68 | 133.3 | — | 37.6 |
| + SPPF at P4 | 11.307 | 29.21 | 107.4 | 0.81x | 42.9 |
| + LKA at P4 (VAN-style) | 11.228 | 28.95 | 125.5 | 0.94x | 48.3 |
| + Deformable only at P4 | 11.795 | 28.88 | 135.9 | 1.02x | 52.2 |
| + DLKA at P4 | 11.293 | 29.15 | 39.5 | 0.30x | 54.7 |
| + MAFF (neck) | 11.187 | 28.68 | 110.9 | 0.83x | 45.1 |
| + CWSL (loss only) | 11.143 | 28.68 | 143.3 | 1.08x | 43.9 |
| MADNet (DLKA + MAFF + CWSL) | 11.337 | 29.15 | 36.4 | 0.27x | 65.6 |

## Table 4. Per-class AP@0.5 (%), classes ordered by training frequency

| Class | N train | base | + SPPF at P4 | + LKA at P4 (VAN-style) | + Deformable only at P4 | + DLKA at P4 | + MAFF (neck) | + CWSL (loss only) | MADNet (DLKA + MAFF + CWSL) |
|---|---|---|---|---|---|---|---|---|---|
| stockbridge damper | 4923 | 98.52 | 98.54 | 98.38 | 98.61 | 98.56 | 98.57 | 98.49 | 98.35 |
| yoke suspension | 4484 | 99.14 | 99.17 | 99.13 | 99.12 | 99.14 | 99.24 | 99.11 | 99.16 |
| polymer insulator | 2063 | 99.14 | 99.06 | 99.11 | 98.97 | 99.09 | 99.10 | 99.07 | 98.99 |
| glass insulator | 1674 | 97.95 | 97.83 | 97.40 | 97.26 | 98.04 | 98.00 | 97.78 | 97.65 |
| polymer insulator lower shackle | 1235 | 90.12 | 89.29 | 88.31 | 89.58 | 88.39 | 90.33 | 89.78 | 88.82 |
| yoke | 1124 | 96.65 | 96.60 | 96.24 | 96.82 | 96.78 | 96.53 | 96.89 | 96.39 |
| polymer insulator upper shackle | 1113 | 99.13 | 98.90 | 99.15 | 99.21 | 99.16 | 99.14 | 99.05 | 98.71 |
| vari-grip | 721 | 99.43 | 99.43 | 99.42 | 99.44 | 99.38 | 99.44 | 99.41 | 99.36 |
| spiral damper | 703 | 99.48 | 99.28 | 99.45 | 99.45 | 99.43 | 99.49 | 99.46 | 99.47 |
| lightning rod suspension | 524 | 99.50 | 99.50 | 99.50 | 99.50 | 99.50 | 99.50 | 99.50 | 99.50 |
| tower id plate | 170 | 99.50 | 99.50 | 99.50 | 99.50 | 99.50 | 99.50 | 99.50 | 99.50 |
| lightning rod shackle | 143 | 86.73 | 81.45 | 79.98 | 88.40 | 78.27 | 80.74 | 83.45 | 81.66 |
| glass insulator small shackle | 113 | 54.88 | 52.18 | 59.22 | 49.82 | 61.25 | 59.60 | 50.95 | 54.57 |
| glass insulator big shackle | 99 | 55.63 | 49.53 | 49.11 | 58.79 | 55.49 | 56.97 | 53.78 | 54.43 |
| glass insulator tower shackle | 90 | 69.19 | 53.02 | 58.67 | 64.39 | 66.92 | 61.29 | 64.96 | 68.37 |
| spacer | 54 | 73.52 | 79.60 | 72.31 | 78.86 | 78.51 | 74.95 | 67.81 | 83.18 |
| polymer insulator tower shackle | 41 | 99.50 | 95.86 | 99.50 | 99.50 | 99.50 | 99.50 | 99.50 | 99.50 |

## Protocol

- InsPLAD-det (Vieira-e-Silva et al., 2023), 10,607 UAV images, 18 asset classes
- Split: official train re-partitioned 85:15 into 6,783 train / 1,198 val; the official val split (2,626 images) is held out as test and never used for model selection
- Training-split instances 19,295; imbalance 234 : 1 (stockbridge damper 4,923 vs sphere 21)
- Identical COCO-pretrained YOLOv8s initialisation for every variant
- 30 epochs, SGD lr0 0.01, momentum 0.937, weight decay 5e-4, cosine schedule, 3 warm-up epochs, mosaic closed for the last 10 epochs, early stopping disabled
- Input 640x640, batch 16, seed 0
- Single seed per variant; no standard deviations are reported and none are claimed