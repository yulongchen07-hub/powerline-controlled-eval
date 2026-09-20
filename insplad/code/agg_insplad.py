#!/usr/bin/env python3
"""Aggregate the per-run InsPLAD-det results into the tables used in the paper."""
import json, os, csv
import os
WORK = os.environ.get("WORK_DIR", ".")  # root for data/, runs/, results/ and tables/

RES = os.path.join(WORK, "results_insplad")
OUT = os.path.join(WORK, "tables_insplad")
CNT = os.path.join(WORK, "data/insplad_yolo/class_counts.json")
os.makedirs(OUT, exist_ok=True)

ORDER = ["baseline", "sppf", "lka", "deform", "dlka", "maff", "cwsl", "madnet"]
LABEL = {
    "baseline": "YOLOv8s (baseline)",
    "sppf":     "+ SPPF at P4",
    "lka":      "+ LKA at P4 (VAN-style)",
    "deform":   "+ Deformable only at P4",
    "dlka":     "+ DLKA at P4",
    "maff":     "+ MAFF (neck)",
    "cwsl":     "+ CWSL (loss only)",
    "madnet":   "MADNet (DLKA + MAFF + CWSL)",
}

rows = []
for v in ORDER:
    f = f"{RES}/ins_{v}_s0.json"
    if os.path.exists(f):
        rows.append(json.load(open(f)))
    else:
        print(f"[missing] {v}")
if not rows:
    raise SystemExit("no results")

B = next((r for r in rows if r["variant"] == "baseline"), None)
f2 = lambda x: f"{x*100:.2f}"
def d(r, k):
    if B is None or r["variant"] == "baseline" or r.get(k) is None:
        return "—"
    v = (r[k] - B[k]) * 100
    return f"{v:+.2f}"

cnts = json.load(open(CNT))
names, counts = cnts["names"], cnts["train_counts"]
cls_order = [n for n, _ in sorted(zip(names, counts), key=lambda x: -x[1])]

md = []
md.append("# InsPLAD-det real experimental results\n")
md.append(f"Hardware: {rows[0]['device']}. All variants start from identical COCO-pretrained "
          f"YOLOv8s weights and share one protocol; only the inserted module (or the classification "
          f"loss, for CWSL) differs.\n")

md.append("## Table 1. InsPLAD-det held-out test split (2,626 images, 6,324 instances)\n")
md.append("| Variant | P (%) | R (%) | mAP@0.5 (%) | Δ (pp) | mAP@0.5:0.95 (%) | Δ (pp) |")
md.append("|---|---|---|---|---|---|---|")
for r in rows:
    md.append(f"| {LABEL[r['variant']]} | {f2(r['precision'])} | {f2(r['recall'])} | "
              f"{f2(r['mAP50'])} | {d(r,'mAP50')} | {f2(r['mAP50_95'])} | {d(r,'mAP50_95')} |")

md.append("\n## Table 2. Head vs tail performance (the long-tail question)\n")
md.append("Head = 7 classes with >= 1000 training instances; tail = 7 classes with < 200. "
          "Training-split imbalance is 234 : 1.\n")
md.append("| Variant | head mAP@0.5 (%) | tail mAP@0.5 (%) | head-tail gap (pp) | Δ tail vs baseline (pp) |")
md.append("|---|---|---|---|---|")
for r in rows:
    gap = (r["mAP50_head"] - r["mAP50_tail"]) * 100 if r.get("mAP50_head") else float("nan")
    md.append(f"| {LABEL[r['variant']]} | {f2(r['mAP50_head'])} | {f2(r['mAP50_tail'])} | "
              f"{gap:.2f} | {d(r,'mAP50_tail')} |")

md.append("\n## Table 3. Efficiency (batch 1, 640x640, FP32, no Conv-BN fusion, "
          "50 warm-up + 200 timed passes)\n")
md.append("| Variant | Params (M) | FLOPs (G) | FPS | FPS vs baseline | Train time (min) |")
md.append("|---|---|---|---|---|---|")
for r in rows:
    ratio = "—" if r is B else f"{r['FPS_bs1_fp32_unfused']/B['FPS_bs1_fp32_unfused']:.2f}x"
    md.append(f"| {LABEL[r['variant']]} | {r['params_M']:.3f} | {r['FLOPs_G']:.2f} | "
              f"{r['FPS_bs1_fp32_unfused']:.1f} | {ratio} | {r['train_minutes']} |")

md.append("\n## Table 4. Per-class AP@0.5 (%), classes ordered by training frequency\n")
present = [c for c in cls_order if any(c in r["ap_per_class"] for r in rows)]
md.append("| Class | N train | " + " | ".join(LABEL[r["variant"]].replace("YOLOv8s (baseline)", "base") for r in rows) + " |")
md.append("|---" * (len(rows) + 2) + "|")
cmap = dict(zip(names, counts))
for c in present:
    cells = " | ".join(f"{r['ap_per_class'].get(c, float('nan'))*100:.2f}" for r in rows)
    md.append(f"| {c} | {cmap.get(c,0)} | {cells} |")

md.append("\n## Protocol\n")
p = rows[0]
md.append("- InsPLAD-det (Vieira-e-Silva et al., 2023), 10,607 UAV images, 18 asset classes")
md.append("- Split: official train re-partitioned 85:15 into 6,783 train / 1,198 val; the official "
          "val split (2,626 images) is held out as test and never used for model selection")
md.append(f"- Training-split instances 19,295; imbalance 234 : 1 (stockbridge damper 4,923 vs sphere 21)")
md.append(f"- Identical COCO-pretrained YOLOv8s initialisation for every variant")
md.append(f"- {p['epochs']} epochs, SGD lr0 0.01, momentum 0.937, weight decay 5e-4, cosine schedule, "
          f"3 warm-up epochs, mosaic closed for the last 10 epochs, early stopping disabled")
md.append(f"- Input 640x640, batch 16, seed {p['seed']}")
md.append("- Single seed per variant; no standard deviations are reported and none are claimed")

open(f"{OUT}/insplad_tables.md", "w").write("\n".join(md))

with open(f"{OUT}/insplad_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["variant", "label", "precision_%", "recall_%", "mAP50_%", "mAP50_95_%",
                "head_mAP50_%", "tail_mAP50_%", "params_M", "FLOPs_G", "FPS", "train_min",
                "epochs", "seed"] + [f"AP50_{c}" for c in present])
    for r in rows:
        w.writerow([r["variant"], LABEL[r["variant"]], f2(r["precision"]), f2(r["recall"]),
                    f2(r["mAP50"]), f2(r["mAP50_95"]), f2(r["mAP50_head"]), f2(r["mAP50_tail"]),
                    r["params_M"], r["FLOPs_G"], r["FPS_bs1_fp32_unfused"], r["train_minutes"],
                    r["epochs"], r["seed"]] +
                   [f"{r['ap_per_class'].get(c, float('nan'))*100:.2f}" for c in present])

json.dump(rows, open(f"{OUT}/insplad_results.json", "w"), indent=2)
print("\n".join(md))
