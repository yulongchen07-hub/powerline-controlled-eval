#!/usr/bin/env python3
"""Controlled training of the CPLID variants.

Design rules that make the comparison like-for-like:
  - every variant starts from the same COCO-pretrained YOLOv8s weights;
  - optimiser, learning-rate schedule, augmentation, input size, batch
    size and epoch budget are identical across variants;
  - early stopping is disabled so that all variants receive the same
    number of updates;
  - the inserted block (or, for CWSL, the substituted loss) is the only
    difference between runs.
"""
import argparse, json, os, sys, time
import os
WORK = os.environ.get("WORK_DIR", ".")  # root for data/, runs/, results/ and tables/
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from madnet_modules import DLKA, LKAOnly, DeformOnly, SPPFBlock, MAFF

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

DATA = os.path.join(WORK, "data/cplid_yolo/cplid.yaml")
BACKBONE_C4 = 6            # index of the C2f layer that emits P4 in the YOLOv8 backbone
NECK_OUTS = [15, 18, 21]   # the three neck C2f layers feeding the detection head

VARIANTS = {
    "baseline":  {"c4": None,         "neck": False},
    "sppf":      {"c4": SPPFBlock,    "neck": False},
    "lka":       {"c4": LKAOnly,      "neck": False},
    "deform":    {"c4": DeformOnly,   "neck": False},
    "dlka":      {"c4": DLKA,         "neck": False},
    "maff":      {"c4": None,         "neck": True},
    "madnet":    {"c4": DLKA,         "neck": True},
}


def probe_channels(model, indices, imgsz=640):
    """Probe the true output width of a layer with one dry-run forward pass and a hook."""
    seq = model.model
    got = {}
    hooks = []
    for i in indices:
        def mk(idx):
            def hook(_m, _inp, out):
                t = out[0] if isinstance(out, (list, tuple)) else out
                got[idx] = t.shape[1]
            return hook
        hooks.append(seq[i].register_forward_hook(mk(i)))
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz, device=dev))
    for h in hooks:
        h.remove()
    model.train(was_training)
    return got


def apply_variant(model, variant):
    """Wrap the custom block around a layer in place, after the pretrained weights are loaded."""
    spec = VARIANTS[variant]
    seq = model.model            # nn.Sequential of layers
    added = []
    targets = ([BACKBONE_C4] if spec["c4"] is not None else []) + (NECK_OUTS if spec["neck"] else [])
    if not targets:
        return added
    chans = probe_channels(model, targets)
    if spec["c4"] is not None:
        layer = seq[BACKBONE_C4]
        c = chans[BACKBONE_C4]
        blk = spec["c4"](c)
        seq[BACKBONE_C4] = nn.Sequential(layer, blk)
        # keep the attributes Ultralytics relies on
        seq[BACKBONE_C4].i, seq[BACKBONE_C4].f = layer.i, layer.f
        seq[BACKBONE_C4].type, seq[BACKBONE_C4].np = f"{spec['c4'].__name__}@{BACKBONE_C4}", 0
        added.append((BACKBONE_C4, spec["c4"].__name__, c))
    if spec["neck"]:
        for idx in NECK_OUTS:
            layer = seq[idx]
            c = chans[idx]
            blk = MAFF(c)
            seq[idx] = nn.Sequential(layer, blk)
            seq[idx].i, seq[idx].f = layer.i, layer.f
            seq[idx].type, seq[idx].np = f"MAFF@{idx}", 0
            added.append((idx, "MAFF", c))
    return added


class VariantTrainer(DetectionTrainer):
    """Inject the custom block after get_model has loaded the pretrained weights,
    so that every variant starts from identical parameters."""
    variant = "baseline"

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        added = apply_variant(model, self.variant)
        if added:
            print(f"[variant={self.variant}] injected: {added}", flush=True)
        return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or f"{args.variant}_s{args.seed}"
    VariantTrainer.variant = args.variant

    t0 = time.time()
    model = YOLO("yolov8s.pt")          # the single COCO-pretrained starting point shared by all variants
    model.train(
        trainer=VariantTrainer,
        data=DATA,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        optimizer="SGD", lr0=0.01, momentum=0.937, weight_decay=5e-4,
        warmup_epochs=3.0, cos_lr=True,
        close_mosaic=10,
        fliplr=0.5, scale=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1,
        patience=0,                      # early stopping off, so every variant receives the same number of updates
        pretrained=True,
        val=True, plots=False,
        project=os.path.join(WORK, "runs"), name=name, exist_ok=True,
        verbose=False,
    )
    train_min = (time.time() - t0) / 60

    # ---- evaluate on the held-out test split ----
    best = os.path.join(WORK, f"runs/{name}/weights/best.pt")
    ev = YOLO(best)
    m = ev.val(data=DATA, split="test", imgsz=args.imgsz, batch=args.batch,
               plots=False, verbose=False)

    # ---- parameters / FLOPs / throughput ----
    from ultralytics.utils.torch_utils import get_flops, get_num_params
    net = ev.model
    params_M = get_num_params(net) / 1e6
    try:
        flops_G = get_flops(net, args.imgsz)
    except Exception:
        flops_G = float("nan")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev).eval()
    x = torch.zeros(1, 3, args.imgsz, args.imgsz, device=dev)
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

    res = {
        "variant": args.variant, "seed": args.seed, "epochs": args.epochs,
        "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "ap_per_class": {ev.names[int(c)]: float(a)
                         for c, a in zip(m.box.ap_class_index, m.box.ap50)},
        "params_M": round(params_M, 3), "FLOPs_G": round(float(flops_G), 2),
        "FPS_bs1_fp32": round(fps, 1),
        "train_minutes": round(train_min, 1),
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
    }
    outp = os.path.join(WORK, f"results/{name}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(res, f, indent=2)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
