"""0b: where is the driver, and how big, in renders (instance masks) vs real photos (face detections)?

Usage: python -I tools/crop_geometry.py --data data/main --gate <gate_e dir with gate_e.json> --experts work/experts
All boxes are reported in normalised image coordinates (0..1), so 960x540 renders and 1920x1080 photos compare.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/main")
ap.add_argument("--experts", default="work/experts")
args = ap.parse_args()
data = Path(args.data)
cmap = {c["name"]: tuple(c["color_rgb"]) for c in json.loads((data / "instance_seg" / "instance_seg_class_map.json").read_text())}


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return np.array([xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h])


drv, belt, head = [], [], []
sizes = set()
for p in sorted((data / "instance_seg").glob("sample_*_instance_seg.png")):
    m = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    sizes.add(m.shape[:2])
    d = np.all(m == cmap["driver"], axis=-1)
    b = bbox(d)
    if b is None:
        continue
    drv.append(b)
    bb = bbox(np.all(m == cmap["driver_seatbelt"], axis=-1))
    if bb is not None:
        belt.append(bb)
    ys = np.nonzero(d)[0]
    top = ys.min()
    hm = d.copy()
    hm[top + int(0.22 * (ys.max() - top)) + 1:] = False  # top ~22% of the driver mask ~ head
    hb = bbox(hm)
    if hb is not None:
        head.append(hb)
drv, belt, head = map(np.array, (drv, belt, head))


def summ(name, a):
    if len(a) == 0:
        print(f"{name}: none")
        return
    q = lambda c, f: np.percentile(a[:, c], f)
    print(f"{name} (n={len(a)}): x0 p1={q(0, 1):.3f} x1 p99={q(2, 99):.3f} | y0 p1={q(1, 1):.3f} y1 p99={q(3, 99):.3f} | "
          f"median w={np.median(a[:, 2] - a[:, 0]):.3f} h={np.median(a[:, 3] - a[:, 1]):.3f}")


print("mask sizes:", sizes)
summ("render driver", drv)
summ("render driver head (top 22%)", head)
summ("render driver seatbelt", belt)
head_px = np.median(head[:, 3] - head[:, 1]) * 540
print(f"render head height ~{head_px:.0f} px at native 960x540")

# real photos: face boxes from YuNet (where found)
det_path = Path(args.experts) / "face_detection_yunet_2023mar.onnx"
faces = []
for i in range(30):
    bgr = cv2.imread(str(data / "real" / f"sample_{i:05d}_real.jpg"))
    h, w = bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(str(det_path), "", (w, h), 0.5, 0.3, 50)
    _, f = det.detect(bgr)
    if f is None:
        continue
    f = [x for x in f if x[0] + x[2] / 2 > 0.5 * w]
    if f:
        x = max(f, key=lambda r: r[2] * r[3])
        faces.append([x[0] / w, x[1] / h, (x[0] + x[2]) / w, (x[1] + x[3]) / h])
faces = np.array(faces)
summ("real driver face (YuNet)", faces)
print(f"real face height ~{np.median(faces[:, 3] - faces[:, 1]) * 1080:.0f} px at native 1920x1080; "
      f"~{np.median(faces[:, 3] - faces[:, 1]) * 532:.0f} px in v2's 952x532 input")
