#!/usr/bin/env python3
"""Convert InsPLAD-det (COCO format) into the YOLO layout used here.

Split policy: the official validation split is kept whole as a held-out
test set and is never used for model selection; the official training
split is divided 85:15 into training and validation, the latter used only
for monitoring and checkpoint selection during training.
"""
import json, os, shutil, random
import os
WORK = os.environ.get("WORK_DIR", ".")  # root for data/, runs/, results/ and tables/
from collections import Counter, defaultdict

SRC = os.path.join(WORK, "data/insplad_raw/det")
OUT = os.path.join(WORK, "data/insplad_yolo")
SEED = 0

def load(sp):
    d = json.load(open(f"{SRC}/annotations/instances_{sp}.json"))
    cats = sorted(d["categories"], key=lambda c: c["id"])
    imgs = {i["id"]: i for i in d["images"]}
    ann = defaultdict(list)
    for a in d["annotations"]:
        if a.get("iscrowd", 0):
            continue
        ann[a["image_id"]].append(a)
    return d, cats, imgs, ann

dtr, cats, imgs_tr, ann_tr = load("train")
dva, cats_v, imgs_va, ann_va = load("val")

# unify the class index using the categories of the training split
names = [c["name"] for c in cats]
cid2idx = {c["id"]: i for i, c in enumerate(cats)}
print(f"classes ({len(names)}): {names}")

def write_split(split_name, imgs, ann, src_dir, ids):
    os.makedirs(f"{OUT}/images/{split_name}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{split_name}", exist_ok=True)
    cc = Counter()
    n_box = 0
    for iid in ids:
        im = imgs[iid]
        fn = im["file_name"]
        src = os.path.join(src_dir, fn)
        if not os.path.exists(src):
            continue
        stem = os.path.splitext(os.path.basename(fn))[0]
        dst_img = f"{OUT}/images/{split_name}/{stem}.jpg"
        if not os.path.exists(dst_img):
            os.link(src, dst_img) if os.path.samefile(os.path.dirname(src), os.path.dirname(src)) else None
        if not os.path.exists(dst_img):
            shutil.copy(src, dst_img)
        W, H = im["width"], im["height"]
        lines = []
        for a in ann.get(iid, []):
            x, y, w, h = a["bbox"]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(W), x + w), min(float(H), y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            ci = cid2idx[a["category_id"]]
            lines.append(f"{ci} {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            cc[names[ci]] += 1
            n_box += 1
        with open(f"{OUT}/labels/{split_name}/{stem}.txt", "w") as f:
            f.write("\n".join(lines))
    return len(ids), n_box, cc

if os.path.exists(OUT):
    shutil.rmtree(OUT)

# official train -> train/val (85:15), split randomly by image
tr_ids = sorted(imgs_tr.keys())
random.seed(SEED)
random.shuffle(tr_ids)
cut = int(len(tr_ids) * 0.85)
sub_tr, sub_va = tr_ids[:cut], tr_ids[cut:]

n1, b1, c1 = write_split("train", imgs_tr, ann_tr, f"{SRC}/train", sub_tr)
n2, b2, c2 = write_split("val",   imgs_tr, ann_tr, f"{SRC}/train", sub_va)
n3, b3, c3 = write_split("test",  imgs_va, ann_va, f"{SRC}/val",   sorted(imgs_va.keys()))

print(f"\ntrain: {n1} images, {b1} boxes")
print(f"val  : {n2} images, {b2} boxes   (monitoring and checkpoint selection only)")
print(f"test : {n3} images, {b3} boxes   (the official val split, held out throughout)")

print("\n=== per-class training instances (the input to the CWSL frequency weights) ===")
counts = [c1.get(nm, 0) for nm in names]
for nm, k in sorted(zip(names, counts), key=lambda x: -x[1]):
    print(f"  {nm:<38} {k:6d}")
nz = [k for k in counts if k > 0]
print(f"  imbalance max/min = {max(nz)/min(nz):.1f} : 1")

with open(f"{OUT}/insplad.yaml", "w") as f:
    f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n")
    for i, nm in enumerate(names):
        f.write(f"  {i}: {nm}\n")
json.dump({"names": names, "train_counts": counts}, open(f"{OUT}/class_counts.json", "w"), indent=2)
print(f"\nwrote {OUT}/insplad.yaml")
