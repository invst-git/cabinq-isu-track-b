"""Gate E: does a real-face expression expert fix v2's driver_emotion errors without adding new ones?

Usage: python -I tools/emotion_expert_gate.py --data data/main --reports work/train_v2/reports.json
           --experts work/experts --out <dir>

Per real photo: YuNet face detection on the full-resolution image (driver = right half), padded crop,
EmotiEffLib enet_b2_8 (AffectNet-8, ONNX) -> z = logit(Happiness) - logit(Neutral). If no face is found
the expert abstains (z = 0). Fusion: s' = s + w*z + b on v2's emotion logit s (P(HAPPY)), w and b fitted by
nested leave-one-out logistic regression (the photo being judged never informs its own fit).
Tested on (i) v2's fold-CV logits and (ii) v2's synthetic-only (zero-shot) stage-1 logits.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

FEATURES_EMOTION = "driver_emotion"
HAPPY, NEUTRAL = 4, 5  # EmotiEffLib 8-class order: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/main")
ap.add_argument("--reports", default="work/train_v2/reports.json")
ap.add_argument("--experts", default="work/experts")
ap.add_argument("--out", required=True)
ap.add_argument("--pad", type=float, default=0.25)
args = ap.parse_args()
data, out = Path(args.data), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

fer = ort.InferenceSession(str(Path(args.experts) / "enet_b2_8.onnx"), providers=["CPUExecutionProvider"])
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def fer_logits(rgb):
    x = cv2.resize(rgb, (260, 260)).astype(np.float64) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    return fer.run(None, {"input": x})[0][0]


def detect_driver_face(bgr):
    h, w = bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(str(Path(args.experts) / "face_detection_yunet_2023mar.onnx"), "", (w, h),
                                    score_threshold=0.5, nms_threshold=0.3, top_k=50)
    _, faces = det.detect(bgr)
    if faces is None:
        return None
    cands = [f for f in faces if f[0] + f[2] / 2 > 0.5 * w]  # the driver sits on the right half of the image
    if not cands:
        return None
    f = max(cands, key=lambda f: f[2] * f[3] * f[14])
    return [float(v) for v in f[:4]] + [float(f[14])]


def crop(rgb, box, pad):
    x, y, bw, bh = box[:4]
    cx, cy, s = x + bw / 2, y + bh / 2, max(bw, bh) * (1 + 2 * pad)
    h, w = rgb.shape[:2]
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    x1, y1 = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
    return rgb[y0:y1, x0:x1]


reports = json.loads(Path(args.reports).read_text())
cv_logit, zs_logit = {}, {}
for tag, res in reports.items():
    if tag.startswith("fold"):
        for key, probs in res["real_test"]["probs"].items():
            cv_logit[key] = probs
zs_tags = [t for t in reports if t.startswith("stage1")]
zs_tag = zs_tags[-1]
for key, probs in reports[zs_tag]["real_all_zero_shot"]["probs"].items():
    zs_logit[key] = probs
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
from isu_core import FEATURES  # noqa: E402
EI = FEATURES.index(FEATURES_EMOTION)


def logit(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


rows = []
montage = []
for i in range(30):
    stem = f"sample_{i:05d}"
    key = f"main/{stem}_real"
    label = json.loads((data / "labels" / f"{stem}.json").read_text())[FEATURES_EMOTION]
    bgr = cv2.imread(str(data / "real" / f"{stem}_real.jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    box = detect_driver_face(bgr)
    if box is None:
        z, ph, face_px = 0.0, None, 0
        tile = np.zeros((160, 160, 3), np.uint8)
    else:
        face = crop(rgb, box, args.pad)
        lg = fer_logits(face)
        z = float(lg[HAPPY] - lg[NEUTRAL])
        e = np.exp(lg - lg.max())
        ph = float(e[HAPPY] / e.sum())
        face_px = int(max(box[2], box[3]))
        tile = cv2.resize(face, (160, 160))
    cv2.putText(tile, f"{i:02d} {label[:1]} z{z:+.1f}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    montage.append(tile)
    rows.append(dict(stem=stem, label=label, y=int(label == "HAPPY"), z=z, p_happy_fer=ph, face_px=face_px,
                     det_score=None if box is None else round(box[4], 3),
                     s_cv=logit(cv_logit[key][EI]), s_zs=logit(zs_logit[key][EI])))

grid = np.vstack([np.hstack(montage[r * 6:(r + 1) * 6]) for r in range(5)])
cv2.imwrite(str(out / "faces_montage.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def fit_wb(s, z, y, l2=0.1, iters=200):
    """Logistic regression on s + w*z + b (s is a fixed offset), small L2 on w and b; Newton steps."""
    w = b = 0.0
    for _ in range(iters):
        t = s + w * z + b
        p = 1 / (1 + np.exp(-t))
        g = np.array([np.sum((p - y) * z) + l2 * w, np.sum(p - y) + l2 * b])
        r = p * (1 - p)
        H = np.array([[np.sum(r * z * z) + l2, np.sum(r * z)], [np.sum(r * z), np.sum(r) + l2]])
        step = np.linalg.solve(H, g)
        w, b = w - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return w, b


def evaluate(col):
    s = np.array([r[col] for r in rows])
    z = np.array([r["z"] for r in rows])
    y = np.array([r["y"] for r in rows])
    base_wrong = [rows[i]["stem"] for i in range(30) if (s[i] > 0) != y[i]]
    fused, params = np.zeros(30), []
    for i in range(30):
        m = np.arange(30) != i
        w, b = fit_wb(s[m], z[m], y[m])
        params.append((w, b))
        fused[i] = s[i] + w * z[i] + b
    fused_wrong = [rows[i]["stem"] for i in range(30) if (fused[i] > 0) != y[i]]
    w_all, b_all = fit_wb(s, z, y)
    return dict(base_wrong=base_wrong, fused_wrong=fused_wrong,
                fixed=sorted(set(base_wrong) - set(fused_wrong)), new=sorted(set(fused_wrong) - set(base_wrong)),
                w_range=[round(min(p[0] for p in params), 3), round(max(p[0] for p in params), 3)],
                b_range=[round(min(p[1] for p in params), 3), round(max(p[1] for p in params), 3)],
                w_all=round(w_all, 3), b_all=round(b_all, 3),
                fused=dict((rows[i]["stem"], round(float(fused[i]), 2)) for i in range(30)))


res = {"cv": evaluate("s_cv"), "zero_shot": evaluate("s_zs"), "zero_shot_tag": zs_tag, "rows": rows}
(out / "gate_e.json").write_text(json.dumps(res, indent=2))

print(f"faces detected: {sum(r['face_px'] > 0 for r in rows)}/30 (driver face size px: "
      f"{sorted(r['face_px'] for r in rows if r['face_px'])})")
print("expert alone (z > 0 -> HAPPY): wrong on",
      [r["stem"][-2:] + r["label"][:1] for r in rows if r["face_px"] and (r["z"] > 0) != r["y"]])
print(" stem  label    z     pFER   s_cv    s_zs")
for r in rows:
    pf = "  -  " if r["p_happy_fer"] is None else f"{r['p_happy_fer']:.2f}"
    print(f"  {r['stem'][-2:]}  {r['label']:7s} {r['z']:+6.2f}  {pf:>5s}  {r['s_cv']:+6.2f}  {r['s_zs']:+6.2f}")
for k in ("cv", "zero_shot"):
    e = res[k]
    print(f"[{k}] base wrong {e['base_wrong']} -> fused wrong {e['fused_wrong']} | fixed {e['fixed']} new {e['new']} | "
          f"LOO w {e['w_range']} b {e['b_range']} | all-30 fit w={e['w_all']} b={e['b_all']}")
