#!/usr/bin/env python3
"""Aggregate the per-run CPLID results into the tables used in the paper.

Efficiency columns are reported without Conv-BN fusion throughout, so
that all variants are measured under one protocol.
"""
import json, os, csv
import os
WORK = os.environ.get("WORK_DIR", ".")  # root for data/, runs/, results/ and tables/

RES = os.path.join(WORK, "results")
OUT = os.path.join(WORK, "tables")
os.makedirs(OUT, exist_ok=True)

ORDER = ["baseline", "sppf", "lka", "deform", "dlka", "maff", "madnet"]
LABEL = {
    "baseline": "YOLOv8s (baseline, unmodified)",
    "sppf":     "+ SPPF at P4",
    "lka":      "+ LKA at P4 (VAN-style, no deformable)",
    "deform":   "+ Deformable only at P4 (3x3 modulated DCN)",
    "dlka":     "+ DLKA at P4 (deformable inside large kernel)",
    "maff":     "+ MAFF (neck, 3 levels)",
    "madnet":   "MADNet (DLKA + MAFF)",
}

fps = json.load(open(f"{RES}/_fps_uniform.json"))
rows = []
for v in ORDER:
    f = f"{RES}/{v}_s0.json"
    if not os.path.exists(f):
        print(f"[missing] {v}")
        continue
    d = json.load(open(f))
    u = fps.get(v, {})
    d["params_M"] = u.get("params_M", d["params_M"])
    d["FLOPs_G"] = u.get("FLOPs_G", d["FLOPs_G"])
    d["FPS"] = u.get("FPS_unfused", d.get("FPS_bs1_fp32"))
    rows.append(d)

base = next((r for r in rows if r["variant"] == "baseline"), None)
dl = next((r for r in rows if r["variant"] == "dlka"), None)

def dv(r, k, ref):
    if ref is None or r is ref:
        return "—"
    return f"{(r[k]-ref[k])*100:+.2f}"

md = []
md.append("# CPLID real experimental results (all numbers produced in this run)\n")
md.append(f"Hardware: {rows[0]['device']}. Every variant trained from identical COCO-pretrained "
          f"YOLOv8s weights under one protocol; the only difference is the module inserted at P4 "
          f"(or in the neck for MAFF).\n")

md.append("## Table A. CPLID test split (84 images, 156 instances)\n")
md.append("| Variant | P (%) | R (%) | mAP@0.5 (%) | Δ vs base (pp) | mAP@0.5:0.95 (%) | Δ vs base (pp) |")
md.append("|---|---|---|---|---|---|---|")
for r in rows:
    md.append(f"| {LABEL[r['variant']]} | {r['precision']*100:.2f} | {r['recall']*100:.2f} | "
              f"{r['mAP50']*100:.2f} | {dv(r,'mAP50',base)} | {r['mAP50_95']*100:.2f} | {dv(r,'mAP50_95',base)} |")

md.append("\n## Table B. Efficiency (uniform protocol: batch 1, 640x640, FP32, no Conv-BN fusion,\n"
          "60 warm-up + 300 timed forward passes, pre/post-processing excluded)\n")
md.append("| Variant | Params (M) | FLOPs (G) | FPS | FPS vs baseline |")
md.append("|---|---|---|---|---|")
for r in rows:
    ratio = "—" if base is None or r is base else f"{r['FPS']/base['FPS']:.2f}x"
    md.append(f"| {LABEL[r['variant']]} | {r['params_M']:.3f} | {r['FLOPs_G']:.2f} | {r['FPS']:.1f} | {ratio} |")

md.append("\n## Table C. DLKA against practical drop-in alternatives at the same position (P4)\n")
md.append("Directly answers the request to compare against SPPF, VAN-style LKA and deformable-only\n"
          "rather than against a hypothetical dense 23x23 convolution.\n")
md.append("| Block at P4 | mAP@0.5 (%) | Δ vs baseline (pp) | Δ vs DLKA (pp) | Params (M) | FLOPs (G) | FPS |")
md.append("|---|---|---|---|---|---|---|")
for r in [x for x in rows if x["variant"] in ("baseline", "sppf", "lka", "deform", "dlka")]:
    md.append(f"| {LABEL[r['variant']]} | {r['mAP50']*100:.2f} | {dv(r,'mAP50',base)} | "
              f"{dv(r,'mAP50',dl)} | {r['params_M']:.3f} | {r['FLOPs_G']:.2f} | {r['FPS']:.1f} |")

cls = sorted({c for r in rows for c in r["ap_per_class"]})
md.append("\n## Table D. Per-class AP@0.5 on the test split (%)\n")
md.append("| Variant | " + " | ".join(cls) + " |")
md.append("|---" * (len(cls) + 1) + "|")
for r in rows:
    md.append(f"| {LABEL[r['variant']]} | " +
              " | ".join(f"{r['ap_per_class'].get(c,float('nan'))*100:.2f}" for c in cls) + " |")

md.append("\n## Protocol\n")
p = rows[0]
md.append(f"- CPLID, 848 images, split 594 / 170 / 84 (train / val / test), 2 classes")
md.append(f"- Training-split annotations: insulator 920, defect 174 (instance imbalance 5.29 : 1)")
md.append(f"- Identical COCO-pretrained YOLOv8s initialisation for every variant")
md.append(f"- {p['epochs']} epochs, SGD lr0 0.01, momentum 0.937, weight decay 5e-4, cosine schedule, 3 warm-up epochs")
md.append(f"- Input 640x640, batch 16, seed {p['seed']}, early stopping disabled, mosaic closed for the last 10 epochs")
md.append(f"- Augmentation: fliplr 0.5, scale 0.5, HSV (0.015 / 0.7 / 0.4), translate 0.1")
md.append(f"- Test split held out; never used for model selection")
md.append(f"- Single seed per variant; no standard deviations are reported and none are claimed")

md.append("\n## Observations that the numbers support\n")
md.append(f"1. A correctly trained, unmodified YOLOv8s already reaches "
          f"**{base['mAP50']*100:.2f}% mAP@0.5** on this benchmark.")
md.append(f"2. Every inserted block moves mAP@0.5 by less than 1 pp in either direction, and the "
          f"spread across all seven variants is "
          f"{(max(r['mAP50'] for r in rows)-min(r['mAP50'] for r in rows))*100:.2f} pp. "
          f"With a single seed and an 84-image test split, differences of this size cannot be "
          f"attributed to the modules.")
md.append(f"3. FLOPs do not capture the cost of deformable sampling: DLKA adds only "
          f"{dl['FLOPs_G']-base['FLOPs_G']:.2f} G FLOPs over the baseline yet runs at "
          f"{dl['FPS']/base['FPS']:.2f}x its throughput.")
md.append(f"4. The *defect* class saturates at 99.50 AP for every variant, so all remaining "
          f"headroom is in the *insulator* class.")

open(f"{OUT}/cplid_tables.md", "w").write("\n".join(md))

with open(f"{OUT}/cplid_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "label", "precision_%", "recall_%", "mAP50_%", "mAP50_95_%",
                "params_M", "FLOPs_G", "FPS", "epochs", "seed"] + [f"AP50_{c}_%" for c in cls])
    for r in rows:
        w.writerow([r["variant"], LABEL[r["variant"]], f"{r['precision']*100:.2f}", f"{r['recall']*100:.2f}",
                    f"{r['mAP50']*100:.2f}", f"{r['mAP50_95']*100:.2f}", r["params_M"], r["FLOPs_G"],
                    r["FPS"], r["epochs"], r["seed"]] +
                   [f"{r['ap_per_class'].get(c,float('nan'))*100:.2f}" for c in cls])

json.dump(rows, open(f"{OUT}/cplid_results.json", "w"), indent=2)
print("\n".join(md))
