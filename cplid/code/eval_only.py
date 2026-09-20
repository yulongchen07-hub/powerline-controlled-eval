#!/usr/bin/env python3
"""Evaluate a trained CPLID checkpoint from its saved best.pt.

Conv-BN fusion is disabled for the same reason as in the InsPLAD-det
evaluation script.
"""
import argparse, json, os, time, sys
import os
WORK = os.environ.get("WORK_DIR", ".")  # root for data/, runs/, results/ and tables/
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import madnet_modules  # noqa: F401  must be importable to deserialise the custom blocks
from ultralytics import YOLO

DATA = os.path.join(WORK, "data/cplid_yolo/cplid.yaml")

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()

name = f"{a.variant}_s{a.seed}"
best = os.path.join(WORK, f"runs/{name}/weights/best.pt")
assert os.path.exists(best), best

ev = YOLO(best)
m = ev.val(data=DATA, split="test", imgsz=a.imgsz, batch=a.batch, plots=False, verbose=False)

from ultralytics.utils.torch_utils import get_flops, get_num_params
net = ev.model
params_M = get_num_params(net) / 1e6
try:
    flops_G = float(get_flops(net, a.imgsz))
except Exception:
    flops_G = float("nan")

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = net.to(dev).eval()
x = torch.zeros(1, 3, a.imgsz, a.imgsz, device=dev)
with torch.no_grad():
    for _ in range(50):
        net(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(200):
        net(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    fps = 200 / (time.time() - t)

tm = None
csvp = os.path.join(WORK, f"runs/{name}/results.csv")
if os.path.exists(csvp):
    try:
        import csv as _csv
        rows = list(_csv.DictReader(open(csvp)))
        k = next((c for c in rows[-1] if "time" in c.lower()), None)
        if k:
            tm = round(float(rows[-1][k]) / 60, 1)
    except Exception:
        pass

res = {
    "variant": a.variant, "seed": a.seed, "epochs": a.epochs,
    "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
    "precision": float(m.box.mp), "recall": float(m.box.mr),
    "ap_per_class": {ev.names[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50)},
    "params_M": round(params_M, 3), "FLOPs_G": round(flops_G, 2),
    "FPS_bs1_fp32": round(fps, 1),
    "train_minutes": tm,
    "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
    "note": "evaluated from saved best.pt (final in-run validation was interrupted)",
}
outp = os.path.join(WORK, f"results/{name}.json")
os.makedirs(os.path.dirname(outp), exist_ok=True)
json.dump(res, open(outp, "w"), indent=2)
print("RESULT " + json.dumps(res))
