#!/usr/bin/env python3
"""Measure throughput for every variant under one uniform protocol.

Batch size 1, 640x640 input, FP32, no Conv-BN fusion, 50 warm-up passes
followed by 200 timed forward passes, pre- and post-processing excluded.
"""
import sys, os, json, time, torch
import os
WORK = os.environ.get("WORK_DIR", ".")  # root for data/, runs/, results/ and tables/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import madnet_modules  # noqa: F401
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

VS = ["baseline", "sppf", "lka", "deform", "dlka", "maff", "madnet"]
out = {}
for v in VS:
    p = os.path.join(WORK, f"runs/{v}_s0/weights/best.pt")
    if not os.path.exists(p):
        print(f"{v}: missing")
        continue
    m = YOLO(p)
    net = m.model
    params = get_num_params(net) / 1e6
    try:
        fl = float(get_flops(net, 640))
    except Exception:
        fl = float("nan")
    net = net.cuda().eval()          # uniform protocol: fusion is disabled for every variant
    x = torch.zeros(1, 3, 640, 640).cuda()
    with torch.no_grad():
        for _ in range(60):
            net(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(300):
            net(x)
        torch.cuda.synchronize()
        fps = 300 / (time.time() - t)
    out[v] = {"params_M": round(params, 3), "FLOPs_G": round(fl, 2), "FPS_unfused": round(fps, 1)}
    print(f"{v:9s} params={params:7.3f}M FLOPs={fl:6.2f}G FPS={fps:6.1f}", flush=True)
    del net, m
    torch.cuda.empty_cache()

json.dump(out, open(os.path.join(WORK, "results/_fps_uniform.json"), "w"), indent=2)
print("saved _fps_uniform.json")
