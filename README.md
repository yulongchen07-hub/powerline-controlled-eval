# Controlled evaluation of attention, deformable and loss-reshaping mechanisms for power line asset detection

Core code and summary results accompanying the paper

> *A Controlled Evaluation of Attention, Deformable and Loss-Reshaping Mechanisms for Power Line Asset Detection: An Unbeaten Baseline, Rare-Class Cost and a FLOPs–Throughput Gap*

Released for academic exchange in connection with that paper.

---

## Scope

This repository holds the core implementation and the summary-level results:

* every script needed to reproduce the study: data preparation, the evaluated blocks, the loss, training, evaluation, throughput measurement and table aggregation;
* the per-run metric table for all 45 runs (`insplad/results/multiseed_raw.csv`, one row per run);
* the summary tables that appear in the paper and the efficiency measurements.

It does not contain the per-epoch training logs, the individual per-run output files, or the trained weights. Those are larger artefacts held in the separate archive named in the paper's Data Availability statement.

---

## What the study does

Eight detector variants are trained under a single controlled protocol. The protocol fixes everything except the mechanism under test:

* the same COCO-pretrained YOLOv8s checkpoint for every variant;
* the same optimiser, learning-rate schedule, augmentation pipeline, input size, batch size and epoch budget;
* the same data splits;
* early stopping disabled, so every variant receives an identical number of updates;
* the inserted block, or for the loss the substituted classification objective, as the only difference.

Each variant is trained three times, with seeds 0, 1 and 2: 45 runs in total.

### Variants

| Variant | What changes |
|---|---|
| `baseline` | unmodified YOLOv8s |
| `sppf` | SPPF block inserted at P4 |
| `lka` | VAN-style large-kernel attention at P4 |
| `deform` | modulated deformable convolution only, at P4 |
| `dlka` | deformable large-kernel attention at P4 |
| `maff` | attention-guided fusion blocks on the three neck outputs |
| `cwsl` | frequency-aware soft-label loss; no architectural change |
| `madnet` | DLKA + MAFF + CWSL combined |

### Benchmarks

* **InsPLAD-det** (primary): 10,607 UAV images, 18 power line asset classes, 349 : 1 training imbalance. The official validation split (2,626 images) is held out as a test set and is never used for model selection.
* **CPLID** (secondary): 848 images, two classes, used only to test whether conclusions transfer.

---

## Layout

```
insplad/
  code/      preparation, blocks, loss, training, evaluation, aggregation
  results/   multiseed_raw.csv        every metric of all 45 runs, one row per run
             multiseed_tables.md, insplad_tables.md   the summary tables
cplid/
  code/      the same scripts for the secondary benchmark
  results/   cplid_tables.md
             efficiency_uniform_protocol.json   parameters, FLOPs and measured throughput
```

| File | Purpose |
|---|---|
| `insplad/code/madnet_modules.py` | DLKA, LKAOnly, DeformOnly, SPPFBlock and MAFF |
| `insplad/code/cwsl_loss.py` | the CWSL classification loss, in the corrected parameterisation of Section 3.4 |
| `insplad/code/prep_insplad.py` | converts InsPLAD-det (COCO) into the YOLO layout and the splits used here |
| `cplid/code/prep_cplid.py` | the deterministic 7:2:1 stratified split of CPLID |
| `insplad/code/train_insplad.py` | the controlled training loop |
| `insplad/code/eval_insplad.py` | evaluation on the held-out test split |
| `cplid/code/fps_uniform.py` | throughput measurement under one uniform protocol |
| `insplad/code/agg_insplad.py`, `cplid/code/aggregate.py` | rebuild the paper's tables from run outputs |
| `insplad/results/multiseed_raw.csv` | the per-run metrics behind every table and figure |

`multiseed_raw.csv` is the file to start from. It carries mAP@0.5, mAP@0.5:0.95, head and tail mAP, precision and recall for each variant and seed on both benchmarks, which is what Tables 1, 2 and 5 of the paper are computed from.

---

## Data

Neither dataset is redistributed here; both are public.

* InsPLAD: https://data.mendeley.com/datasets/5n3fjgvfyz/1
* CPLID: https://github.com/InsulatorData/InsulatorDataSet

`prep_insplad.py` and `prep_cplid.py` reproduce the exact splits used in the paper from the original archives. They are deterministic: the same source archive gives the same splits on any machine.

A small number of images listed in the official InsPLAD annotations are absent from the distributed archive. The preparation script excludes them, so the realised training subset is 6,746 images / 19,167 instances rather than the nominal 6,783 / 19,295. The paper reports the realised figures throughout.

---

## Running it

All scripts read and write under one working directory, taken from the environment variable `WORK_DIR` (default: the current directory). Inside it the scripts expect and create `data/`, `runs/`, `results/` and `tables/`.

```bash
pip install -r requirements.txt
export WORK_DIR=/path/to/work        # optional; defaults to .

# place the original archives under $WORK_DIR/data/ (see the SRC / ROOT constants
# at the top of the preparation scripts), then build the splits
python insplad/code/prep_insplad.py
python cplid/code/prep_cplid.py

# train and evaluate one variant with one seed
python insplad/code/train_insplad.py --variant dlka --seed 0
python insplad/code/eval_insplad.py  --variant dlka --seed 0
```

Training and evaluation need a GPU. The aggregation scripts read the individual per-run output files, which are held in the separate archive rather than here; `multiseed_raw.csv` already carries the aggregated per-run metrics, so the paper's tables can be checked directly from it without rerunning anything.

### Environment used

PyTorch 2.11.0 (CUDA 12.8), Ultralytics 8.4.138, a single NVIDIA GeForce RTX 5060 Laptop GPU (8 GB), Ubuntu 24.04.

### Throughput measurement

Batch size 1, 640×640 input, FP32, Conv-BN fusion disabled, 50 warm-up passes then 200 timed forward passes, pre- and post-processing excluded. Fusion is disabled because several of the inserted blocks contain convolutions without batch normalisation, for which the fusion path does not apply; disabling it for every variant keeps the comparison uniform.

---

## Citing

If you use this material, please cite the paper named at the top of this file.

## Licence

Code and result files here are released under the MIT Licence (see `LICENSE`). The datasets are not redistributed and remain under their own licences.
