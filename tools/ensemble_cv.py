"""Judge a logit-average ensemble on the real-photo fold CV, from the runs' saved reports (no GPU).

Usage: python -I tools/ensemble_cv.py work/train_v2/reports.json work/train_v4/reports.json [--splits a b]
Every run must use the same splits (checked when --splits is given): each real photo is then scored by fold
models that never trained on it, in every member, and the per-photo member logits can be averaged.
Prints per-member and ensemble CV errors, log-loss, and the smallest margins (|logit|) per feature.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
from isu_core import FEATURES, VALUES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("reports", nargs="+")
ap.add_argument("--splits", nargs="*")
ap.add_argument("--labels", default="data/main/labels")
args = ap.parse_args()

if args.splits:
    s = [json.loads(Path(p).read_text()) for p in args.splits]
    assert all(x["folds"] == s[0]["folds"] for x in s), "members were trained on different CV folds"


def cv_logits(path):
    rep = json.loads(Path(path).read_text())
    out = {}
    for tag, res in rep.items():
        if tag.startswith("fold"):
            for key, probs in res["real_test"]["probs"].items():
                p = np.clip(np.array(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
                out[key] = np.log(p / (1 - p))
    return out


members = [cv_logits(p) for p in args.reports]
keys = sorted(members[0])
assert all(sorted(m) == keys for m in members), "members cover different photos"
Y = np.zeros((len(keys), len(FEATURES)))
M = np.zeros_like(Y)
for n, key in enumerate(keys):
    stem = key.split("/")[1].rsplit("_", 1)[0]
    lab = json.loads((Path(args.labels) / f"{stem}.json").read_text())
    for i, f in enumerate(FEATURES):
        v = lab.get(f)
        if v is not None:
            Y[n, i] = VALUES[f].index(str(v).strip().upper())
            M[n, i] = 1


def judge(name, L):
    wrong = ((L > 0) != (Y > 0.5)) & (M > 0)
    p = 1 / (1 + np.exp(-L))
    ll = -(Y * np.log(np.clip(p, 1e-9, 1)) + (1 - Y) * np.log(np.clip(1 - p, 1e-9, 1)))
    ll = float((ll * M).sum() / M.sum())
    signed = np.where(Y > 0.5, L, -L)  # >0 means correct; the size is the margin
    worst = sorted(((float(signed[r, c]), keys[r].split("/")[1][7:12], FEATURES[c])
                    for r, c in zip(*np.nonzero(M))), key=lambda t: t[0])[:6]
    errs = [(keys[r].split("/")[1][7:12], FEATURES[c], round(float(L[r, c]), 2)) for r, c in zip(*np.nonzero(wrong))]
    print(f"{name:12s} errors {int(wrong.sum())}/{int(M.sum())}  logloss {ll:.4f}  errors={errs}")
    print(f"{'':12s} 6 smallest signed margins: {[(round(a, 2), b, c) for a, b, c in worst]}")


for path, m in zip(args.reports, members):
    judge(Path(path).parent.name, np.stack([m[k] for k in keys]))
ens = np.mean([np.stack([m[k] for k in keys]) for m in members], axis=0)
judge("ENSEMBLE", ens)
