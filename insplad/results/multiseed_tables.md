# Multi-seed results (seeds 0, 1, 2 — identical protocol per benchmark)

## InsPLAD-det (test: 2,626 images)

### mAP@0.5 — mean ± std over seeds, and paired Δ vs baseline

| Variant | n seeds | mean ± std (%) | range (%) | paired Δ mean ± std (pp) | Δ per seed (pp) |
|---|---|---|---|---|---|
| YOLOv8s (baseline) | 3 | 89.42 ± 0.37 | [89.13, 89.84] | — | — |
| + SPPF at P4 | 3 | 87.30 ± 0.26 | [87.05, 87.57] | -2.12 ± 0.42 | s0:-1.72 s1:-2.09 s2:-2.56 |
| + LKA at P4 | 3 | 88.19 ± 1.80 | [86.55, 90.11] | -1.24 ± 1.43 | s0:-1.39 s1:-2.58 s2:+0.26 |
| + Deformable only at P4 | 3 | 89.06 ± 0.76 | [88.23, 89.70] | -0.37 ± 0.47 | s0:-0.05 s1:-0.91 s2:-0.14 |
| + DLKA at P4 | 3 | 89.10 ± 0.50 | [88.55, 89.52] | -0.32 ± 0.87 | s0:-0.06 s1:+0.39 s2:-1.29 |
| + MAFF (neck) | 3 | 89.21 ± 0.47 | [88.93, 89.75] | -0.22 ± 0.13 | s0:-0.36 s1:-0.20 s2:-0.09 |
| + CWSL (loss only) | 3 | 88.71 ± 0.78 | [88.15, 89.60] | -0.71 ± 0.45 | s0:-1.15 s1:-0.74 s2:-0.25 |
| MADNet (DLKA + MAFF + CWSL) | 3 | 89.10 ± 0.23 | [88.84, 89.27] | -0.32 ± 0.59 | s0:-0.02 s1:+0.05 s2:-1.00 |

### mAP@0.5:0.95 — mean ± std over seeds, and paired Δ vs baseline

| Variant | n seeds | mean ± std (%) | range (%) | paired Δ mean ± std (pp) | Δ per seed (pp) |
|---|---|---|---|---|---|
| YOLOv8s (baseline) | 3 | 72.73 ± 0.70 | [71.95, 73.31] | — | — |
| + SPPF at P4 | 3 | 70.52 ± 0.12 | [70.40, 70.64] | -2.21 ± 0.74 | s0:-2.91 s1:-1.44 s2:-2.29 |
| + LKA at P4 | 3 | 71.55 ± 1.78 | [69.62, 73.13] | -1.18 ± 1.28 | s0:-1.42 s1:-2.33 s2:+0.20 |
| + Deformable only at P4 | 3 | 72.52 ± 0.86 | [71.68, 73.40] | -0.21 ± 0.65 | s0:-0.83 s1:-0.27 s2:+0.47 |
| + DLKA at P4 | 3 | 72.00 ± 0.11 | [71.88, 72.10] | -0.74 ± 0.69 | s0:-1.21 s1:+0.06 s2:-1.05 |
| + MAFF (neck) | 3 | 72.60 ± 0.39 | [72.36, 73.06] | -0.13 ± 0.74 | s0:-0.96 s1:+0.45 s2:+0.13 |
| + CWSL (loss only) | 3 | 72.33 ± 0.32 | [71.98, 72.61] | -0.40 ± 0.38 | s0:-0.70 s1:+0.03 s2:-0.53 |
| MADNet (DLKA + MAFF + CWSL) | 3 | 72.55 ± 0.49 | [72.02, 72.98] | -0.18 ± 0.82 | s0:-0.33 s1:+0.70 s2:-0.91 |

### Tail mAP@0.5 (7 classes, <200 train instances) — mean ± std over seeds, and paired Δ vs baseline

| Variant | n seeds | mean ± std (%) | range (%) | paired Δ mean ± std (pp) | Δ per seed (pp) |
|---|---|---|---|---|---|
| YOLOv8s (baseline) | 3 | 77.33 ± 0.93 | [76.63, 78.38] | — | — |
| + SPPF at P4 | 3 | 72.28 ± 0.70 | [71.62, 73.02] | -5.06 ± 1.11 | s0:-3.97 s1:-5.01 s2:-6.19 |
| + LKA at P4 | 3 | 74.45 ± 4.17 | [70.50, 78.81] | -2.88 ± 3.27 | s0:-2.95 s1:-6.12 s2:+0.42 |
| + Deformable only at P4 | 3 | 76.58 ± 1.63 | [74.78, 77.94] | -0.75 ± 0.98 | s0:+0.04 s1:-1.85 s2:-0.44 |
| + DLKA at P4 | 3 | 76.74 ± 1.36 | [75.25, 77.90] | -0.60 ± 2.28 | s0:+0.07 s1:+1.28 s2:-3.14 |
| + MAFF (neck) | 3 | 76.78 ± 1.13 | [76.08, 78.09] | -0.55 ± 0.32 | s0:-0.91 s1:-0.44 s2:-0.30 |
| + CWSL (loss only) | 3 | 75.81 ± 2.02 | [74.28, 78.10] | -1.53 ± 1.22 | s0:-2.71 s1:-1.58 s2:-0.28 |
| MADNet (DLKA + MAFF + CWSL) | 3 | 76.90 ± 0.55 | [76.28, 77.32] | -0.43 ± 1.45 | s0:+0.32 s1:+0.48 s2:-2.10 |

### Head mAP@0.5 (7 classes, >=1000 train instances) — mean ± std over seeds, and paired Δ vs baseline

| Variant | n seeds | mean ± std (%) | range (%) | paired Δ mean ± std (pp) | Δ per seed (pp) |
|---|---|---|---|---|---|
| YOLOv8s (baseline) | 3 | 97.21 ± 0.03 | [97.18, 97.23] | — | — |
| + SPPF at P4 | 3 | 97.12 ± 0.06 | [97.06, 97.16] | -0.09 ± 0.08 | s0:-0.18 s1:-0.05 s2:-0.03 |
| + LKA at P4 | 3 | 97.09 ± 0.30 | [96.82, 97.40] | -0.11 ± 0.32 | s0:-0.42 s1:-0.15 s2:+0.23 |
| + Deformable only at P4 | 3 | 97.07 ± 0.21 | [96.86, 97.28] | -0.14 ± 0.22 | s0:-0.15 s1:-0.35 s2:+0.10 |
| + DLKA at P4 | 3 | 97.03 ± 0.15 | [96.89, 97.18] | -0.17 ± 0.17 | s0:-0.21 s1:-0.32 s2:+0.01 |
| + MAFF (neck) | 3 | 97.23 ± 0.05 | [97.17, 97.27] | +0.03 ± 0.06 | s0:+0.04 s1:-0.04 s2:+0.07 |
| + CWSL (loss only) | 3 | 97.01 ± 0.15 | [96.86, 97.17] | -0.20 ± 0.12 | s0:-0.07 s1:-0.21 s2:-0.31 |
| MADNet (DLKA + MAFF + CWSL) | 3 | 96.86 ± 0.01 | [96.85, 96.87] | -0.35 ± 0.03 | s0:-0.37 s1:-0.36 s2:-0.31 |

## CPLID (test: 84 images)

### mAP@0.5 — mean ± std over seeds, and paired Δ vs baseline

| Variant | n seeds | mean ± std (%) | range (%) | paired Δ mean ± std (pp) | Δ per seed (pp) |
|---|---|---|---|---|---|
| YOLOv8s (baseline) | 3 | 98.04 ± 0.57 | [97.63, 98.70] | — | — |
| + SPPF at P4 | 3 | 98.19 ± 0.18 | [98.03, 98.39] | +0.15 ± 0.70 | s0:+0.59 s1:+0.51 s2:-0.66 |
| + LKA at P4 | 3 | 98.64 ± 0.28 | [98.33, 98.88] | +0.60 ± 0.38 | s0:+0.91 s1:+0.70 s2:+0.18 |
| + Deformable only at P4 | 3 | 98.48 ± 0.11 | [98.36, 98.57] | +0.44 ± 0.68 | s0:+0.71 s1:+0.94 s2:-0.34 |
| + DLKA at P4 | 3 | 98.15 ± 0.37 | [97.89, 98.57] | +0.11 ± 0.75 | s0:+0.78 s1:+0.26 s2:-0.70 |
| + MAFF (neck) | 3 | 98.04 ± 0.48 | [97.51, 98.46] | -0.00 ± 1.05 | s0:+0.36 s1:+0.82 s2:-1.19 |
| MADNet (DLKA + MAFF) | 3 | 98.13 ± 0.18 | [97.95, 98.30] | +0.09 ± 0.73 | s0:+0.50 s1:+0.52 s2:-0.75 |

### mAP@0.5:0.95 — mean ± std over seeds, and paired Δ vs baseline

| Variant | n seeds | mean ± std (%) | range (%) | paired Δ mean ± std (pp) | Δ per seed (pp) |
|---|---|---|---|---|---|
| YOLOv8s (baseline) | 3 | 82.28 ± 0.55 | [81.72, 82.83] | — | — |
| + SPPF at P4 | 3 | 81.49 ± 0.67 | [80.79, 82.13] | -0.79 ± 0.12 | s0:-0.93 s1:-0.75 s2:-0.70 |
| + LKA at P4 | 3 | 82.57 ± 0.20 | [82.34, 82.70] | +0.29 ± 0.58 | s0:+0.96 s1:+0.05 s2:-0.13 |
| + Deformable only at P4 | 3 | 81.53 ± 0.38 | [81.12, 81.87] | -0.75 ± 0.93 | s0:+0.15 s1:-0.68 s2:-1.71 |
| + DLKA at P4 | 3 | 80.85 ± 0.51 | [80.31, 81.32] | -1.43 ± 0.90 | s0:-0.40 s1:-1.98 s2:-1.92 |
| + MAFF (neck) | 3 | 81.64 ± 1.36 | [80.20, 82.90] | -0.64 ± 1.74 | s0:+0.09 s1:+0.61 s2:-2.62 |
| MADNet (DLKA + MAFF) | 3 | 81.26 ± 0.74 | [80.69, 82.10] | -1.02 ± 1.21 | s0:+0.38 s1:-1.60 s2:-1.83 |

## Headline checks

### InsPLAD mAP@0.5 vs baseline, per seed
- sppf: Δ per seed = ['-1.72', '-2.09', '-2.56'], seeds above baseline: 0/3
- lka: Δ per seed = ['-1.39', '-2.58', '+0.26'], seeds above baseline: 1/3
- deform: Δ per seed = ['-0.05', '-0.91', '-0.14'], seeds above baseline: 0/3
- dlka: Δ per seed = ['-0.06', '+0.39', '-1.29'], seeds above baseline: 1/3
- maff: Δ per seed = ['-0.36', '-0.20', '-0.09'], seeds above baseline: 0/3
- cwsl: Δ per seed = ['-1.15', '-0.74', '-0.25'], seeds above baseline: 0/3
- madnet: Δ per seed = ['-0.02', '+0.05', '-1.00'], seeds above baseline: 1/3

### InsPLAD tail mAP@0.5 vs baseline, per seed
- sppf: Δ per seed = ['-3.97', '-5.01', '-6.19'], seeds above baseline: 0/3
- lka: Δ per seed = ['-2.95', '-6.12', '+0.42'], seeds above baseline: 1/3
- deform: Δ per seed = ['+0.04', '-1.85', '-0.44'], seeds above baseline: 1/3
- dlka: Δ per seed = ['+0.07', '+1.28', '-3.14'], seeds above baseline: 2/3
- maff: Δ per seed = ['-0.91', '-0.44', '-0.30'], seeds above baseline: 0/3
- cwsl: Δ per seed = ['-2.71', '-1.58', '-0.28'], seeds above baseline: 0/3
- madnet: Δ per seed = ['+0.32', '+0.48', '-2.10'], seeds above baseline: 2/3

### CPLID mAP@0.5 vs baseline, per seed
- sppf: Δ per seed = ['+0.59', '+0.51', '-0.66'], seeds above baseline: 2/3
- lka: Δ per seed = ['+0.91', '+0.70', '+0.18'], seeds above baseline: 3/3
- deform: Δ per seed = ['+0.71', '+0.94', '-0.34'], seeds above baseline: 2/3
- dlka: Δ per seed = ['+0.78', '+0.26', '-0.70'], seeds above baseline: 2/3
- maff: Δ per seed = ['+0.36', '+0.82', '-1.19'], seeds above baseline: 2/3
- madnet: Δ per seed = ['+0.50', '+0.52', '-0.75'], seeds above baseline: 2/3