#!/usr/bin/env python3
"""Controlled training of the eight InsPLAD-det variants.

All variants start from the same COCO-pretrained YOLOv8s checkpoint and
share identical hyper-parameters, splits, schedule and augmentation;
early stopping is disabled so that every variant receives the same number
of updates. The only difference between variants is the block inserted
after the fourth backbone stage, the MAFF blocks appended to the three
neck outputs, or the substitution of the classification loss for CWSL.
"""
import argparse, json, os, sys, time
import os
WORK = os.environ.get("WORK_DIR", ".")  # root for data/, runs/, results/ and tables/
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from madnet_modules import DLKA, LKAOnly, DeformOnly, SPPFBlock, MAFF
from cwsl_loss import CWSLBCE

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

DATA = os.path.join(WORK, "data/insplad_yolo/insplad.yaml")
CNTS = os.path.join(WORK, "data/insplad_yolo/class_counts.json")
BACKBONE_C4 = 6
NECK_OUTS = [15, 18, 21]

VARIANTS = {
    "baseline": {"c4": None,       "neck": False, "cwsl": False},
    "sppf":     {"c4": SPPFBlock,  "neck": False, "cwsl": False},
    "lka":      {"c4": LKAOnly,    "neck": False, "cwsl": False},
    "deform":   {"c4": DeformOnly, "neck": False, "cwsl": False},
    "dlka":     {"c4": DLKA,       "neck": False, "cwsl": False},
    "maff":     {"c4": None,       "neck": True,  "cwsl": False},
    "cwsl":     {"c4": None,       "neck": False, "cwsl": True},
    "madnet":   {"c4": DLKA,       "neck": True,  "cwsl": True},
}


def probe_channels(model, idxs, imgsz=640):
    seq, got, hooks = model.model, {}, []
    for i in idxs:
        def mk(k):
            def h(_m, _i, o):
                t = o[0] if isinstance(o, (list, tuple)) else o
                got[k] = t.shape[1]
            return h
        hooks.append(seq[i].register_forward_hook(mk(i)))
    tr = model.training
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz, device=dev))
    for h in hooks:
        h.remove()
    model.train(tr)
    return got


def apply_variant(model, variant):
    spec = VARIANTS[variant]
    seq = model.model
    added = []
    targets = ([BACKBONE_C4] if spec["c4"] else []) + (NECK_OUTS if spec["neck"] else [])
    if not targets:
        return added
    ch = probe_channels(model, targets)
    if spec["c4"]:
        lay = seq[BACKBONE_C4]
        blk = spec["c4"](ch[BACKBONE_C4])
        seq[BACKBONE_C4] = nn.Sequential(lay, blk)
        seq[BACKBONE_C4].i, seq[BACKBONE_C4].f = lay.i, lay.f
        seq[BACKBONE_C4].type, seq[BACKBONE_C4].np = f"{spec['c4'].__name__}@{BACKBONE_C4}", 0
        added.append((BACKBONE_C4, spec["c4"].__name__, ch[BACKBONE_C4]))
    if spec["neck"]:
        for idx in NECK_OUTS:
            lay = seq[idx]
            blk = MAFF(ch[idx])
            seq[idx] = nn.Sequential(lay, blk)
            seq[idx].i, seq[idx].f = lay.i, lay.f
            seq[idx].type, seq[idx].np = f"MAFF@{idx}", 0
            added.append((idx, "MAFF", ch[idx]))
    return added


class VariantTrainer(DetectionTrainer):
    variant = "baseline"

    def get_model(self, cfg=None, weights=None, verbose=True):
        m = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        a = apply_variant(m, self.variant)
        if a:
            print(f"[variant={self.variant}] injected: {a}", flush=True)
        return m

    def _setup_train(self, *args, **kwargs):
        super()._setup_train(*args, **kwargs)
        if VARIANTS[self.variant]["cwsl"]:
            counts = json.load(open(CNTS))["train_counts"]
            net = self.model.module if hasattr(self.model, "module") else self.model
            net.criterion = net.init_criterion()      # the criterion is created lazily; initialise it explicitly here
            net.criterion.bce = CWSLBCE(counts).to(self.device)
            print(f"[variant={self.variant}] CWSL classification loss installed "
                  f"({len(counts)} classes, imbalance {max(counts)/max(1,min(counts)):.1f}:1)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    name = f"ins_{a.variant}_s{a.seed}"
    VariantTrainer.variant = a.variant
    t0 = time.time()
    m = YOLO("yolov8s.pt")
    m.train(trainer=VariantTrainer, data=DATA, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
            seed=a.seed, optimizer="SGD", lr0=0.01, momentum=0.937, weight_decay=5e-4,
            warmup_epochs=3.0, cos_lr=True, close_mosaic=10, patience=0, pretrained=True,
            fliplr=0.5, scale=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1,
            val=True, plots=False, workers=a.workers,
            project=os.path.join(WORK, "runs"), name=name, exist_ok=True, verbose=False)
    tmin = (time.time() - t0) / 60

    best = os.path.join(WORK, f"runs/{name}/weights/best.pt")
    ev = YOLO(best)
    ev.model.fuse = lambda *x, **k: ev.model      # MAFF contains convolutions without BN, for which fusion is not applicable
    r = ev.val(data=DATA, split="test", imgsz=a.imgsz, batch=8, workers=0,
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

    names = json.load(open(CNTS))["names"]
    counts = json.load(open(CNTS))["train_counts"]
    ap50 = {ev.names[int(c)]: float(v) for c, v in zip(r.box.ap_class_index, r.box.ap50)}
    # mean AP over the tail classes (fewer than 200 training instances)
    tail = [ap50[n] for n, k in zip(names, counts) if k < 200 and n in ap50]
    head = [ap50[n] for n, k in zip(names, counts) if k >= 1000 and n in ap50]

    res = {"variant": a.variant, "seed": a.seed, "epochs": a.epochs,
           "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
           "precision": float(r.box.mp), "recall": float(r.box.mr),
           "mAP50_tail": sum(tail)/len(tail) if tail else None,
           "mAP50_head": sum(head)/len(head) if head else None,
           "n_tail_classes": len(tail), "n_head_classes": len(head),
           "ap_per_class": ap50,
           "params_M": round(params, 3), "FLOPs_G": round(fl, 2),
           "FPS_bs1_fp32_unfused": round(fps, 1),
           "train_minutes": round(tmin, 1),
           "device": torch.cuda.get_device_name(0)}
    outp = os.path.join(WORK, f"results_insplad/{name}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(res, open(outp, "w"), indent=2)
    print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "ap_per_class"}))


if __name__ == "__main__":
    main()
