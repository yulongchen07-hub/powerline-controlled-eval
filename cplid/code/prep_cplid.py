#!/usr/bin/env python3
"""Convert CPLID into the YOLO layout used here.

The 7:2:1 split is stratified over the normal and defective subsets and
is reproduced deterministically from the seed set below.
"""
import os, random, shutil, xml.etree.ElementTree as ET
WORK = os.environ.get("WORK_DIR", ".")  # root for data/, runs/, results/ and tables/
from collections import Counter

ROOT = os.path.join(WORK, "data/InsulatorDataSet")
OUT = os.path.join(WORK, "data/cplid_yolo")
CLASSES = ["insulator", "defect"]
SEED = 0

def parse_xml(p):
    """Return [(cls_name, xmin, ymin, xmax, ymax)] together with (W, H)."""
    if not os.path.exists(p):
        return [], None
    r = ET.parse(p).getroot()
    sz = r.find("size")
    W, H = int(sz.find("width").text), int(sz.find("height").text)
    out = []
    for o in r.findall("object"):
        name = o.find("name").text.strip()
        b = o.find("bndbox")
        out.append((name, float(b.find("xmin").text), float(b.find("ymin").text),
                    float(b.find("xmax").text), float(b.find("ymax").text)))
    return out, (W, H)

def to_yolo(objs, wh):
    W, H = wh
    lines = []
    for name, x1, y1, x2, y2 in objs:
        if name not in CLASSES:
            continue
        # clip to the image bounds so that no box falls outside
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(W), x2), min(float(H), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        lines.append(f"{CLASSES.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines

records = []   # (img_path, [yolo lines], group)

# --- normal insulators: insulator annotations only ---
nd = os.path.join(ROOT, "Normal_Insulators")
for fn in sorted(os.listdir(os.path.join(nd, "images"))):
    stem = os.path.splitext(fn)[0]
    objs, wh = parse_xml(os.path.join(nd, "labels", stem + ".xml"))
    if wh is None:
        continue
    records.append((os.path.join(nd, "images", fn), to_yolo(objs, wh), "normal"))

# --- defective insulators: merge the insulator/ and defect/ annotation sets ---
dd = os.path.join(ROOT, "Defective_Insulators")
for fn in sorted(os.listdir(os.path.join(dd, "images"))):
    stem = os.path.splitext(fn)[0]
    o1, wh1 = parse_xml(os.path.join(dd, "labels", "insulator", stem + ".xml"))
    o2, wh2 = parse_xml(os.path.join(dd, "labels", "defect", stem + ".xml"))
    wh = wh1 or wh2
    if wh is None:
        continue
    records.append((os.path.join(dd, "images", fn), to_yolo(o1 + o2, wh), "defective"))

print(f"total images: {len(records)}")
print("groups:", Counter(r[2] for r in records))

# --- stratified 7:2:1 split, applied within the normal and defective groups ---
random.seed(SEED)
splits = {"train": [], "val": [], "test": []}
for grp in ["normal", "defective"]:
    sub = [r for r in records if r[2] == grp]
    random.shuffle(sub)
    n = len(sub)
    n_tr, n_va = int(round(n * 0.7)), int(round(n * 0.2))
    splits["train"] += sub[:n_tr]
    splits["val"] += sub[n_tr:n_tr + n_va]
    splits["test"] += sub[n_tr + n_va:]

# --- write out ---
if os.path.exists(OUT):
    shutil.rmtree(OUT)
for sp in splits:
    os.makedirs(f"{OUT}/images/{sp}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{sp}", exist_ok=True)

stats = {}
for sp, items in splits.items():
    cc = Counter()
    for i, (img, lines, grp) in enumerate(items):
        base = f"{grp}_{os.path.splitext(os.path.basename(img))[0]}"
        shutil.copy(img, f"{OUT}/images/{sp}/{base}.jpg")
        with open(f"{OUT}/labels/{sp}/{base}.txt", "w") as f:
            f.write("\n".join(lines))
        for l in lines:
            cc[CLASSES[int(l.split()[0])]] += 1
    stats[sp] = (len(items), dict(cc))

print("\n=== split summary ===")
for sp in ["train", "val", "test"]:
    n, cc = stats[sp]
    print(f"{sp:6s} images={n:4d}  annotations={cc}")
tot = sum(stats[s][0] for s in stats)
print(f"{tot} images in total (expected: 594/170/84)")

with open(f"{OUT}/cplid.yaml", "w") as f:
    f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n")
    for i, c in enumerate(CLASSES):
        f.write(f"  {i}: {c}\n")
print(f"\ndataset config: {OUT}/cplid.yaml")
