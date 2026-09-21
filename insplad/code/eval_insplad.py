#!/usr/bin/env python3
"""Evaluate a trained InsPLAD-det checkpoint on the held-out test split.

Conv-BN fusion is disabled because several of the inserted blocks contain
convolutions without batch normalisation, for which the fusion path does
not apply.
"""
import argparse, json, os, sys, time
import os
WORK = os.environ.get("WORK_DIR", ".")  # root for data/, runs/, results/ and tables/
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import madnet_modules  # noqa: F401
import cwsl_loss       # noqa: F401
from ultralytics import YOLO

DATA = os.path.join(WORK, "data/insplad_yolo/insplad.yaml")
CNTS = os.path.join(WORK, "data/insplad_yolo/class_counts.json")

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=4)
a = ap.parse_args()

name = f"ins_{a.variant}_s{a.seed}"
best = os.path.join(WORK, f"runs/{name}/weights/best.pt")
assert os.path.exists(best), best

ev = YOLO(best)
ev.model.fuse = lambda *x, **k: ev.model          # disable fusion (the custom blocks contain convolutions without BN)
r = ev.val(data=DATA, split="test", imgsz=a.imgsz, batch=a.batch, workers=0,
           plots=False, verbose=False)

from ultralytics.utils.torch_utils import get_flops, get_num_params
net = ev.model
params = get_num_params(net) / 1e6
try:
    fl = float(get_flops(net, a.imgsz))
except Exception:
    fl = float("nan")
net = net.cuda().eval()
x = torch.zeros(1, 3, a.imgsz, a.imgsz).cuda()
with torch.no_grad():
    for _ in range(50):
        net(x)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(200):
        net(x)
    torch.cuda.synchronize(); fps = 200 / (time.time() - t)

meta = json.load(open(CNTS))
names, counts = meta["names"], meta["train_counts"]
ap50 = {ev.names[int(c)]: float(v) for c, v in zip(r.box.ap_class_index, r.box.ap50)}
tail = [ap50[n] for n, k in zip(names, counts) if k < 200 and n in ap50]
head = [ap50[n] for n, k in zip(names, counts) if k >= 1000 and n in ap50]

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

res = {"variant": a.variant, "seed": a.seed, "epochs": a.epochs,
       "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
       "precision": float(r.box.mp), "recall": float(r.box.mr),
       "mAP50_tail": sum(tail)/len(tail) if tail else None,
       "mAP50_head": sum(head)/len(head) if head else None,
       "n_tail_classes": len(tail), "n_head_classes": len(head),
       "ap_per_class": ap50,
       "params_M": round(params, 3), "FLOPs_G": round(fl, 2),
       "FPS_bs1_fp32_unfused": round(fps, 1),
       "train_minutes": tm,
       "device": torch.cuda.get_device_name(0),
       "note": "evaluated from saved best.pt (trainer's own final validation crashed)"}
outp = os.path.join(WORK, f"results_insplad/{name}.json")
os.makedirs(os.path.dirname(outp), exist_ok=True)
json.dump(res, open(outp, "w"), indent=2)
print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "ap_per_class"}))
