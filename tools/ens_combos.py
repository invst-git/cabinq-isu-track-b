"""Score every member combination from the Kaggle ensemble check's per-member logits (ens_logits.json).

Usage: python -I tools/ens_combos.py work/enscheck/ens_logits.json [--data data]
For each combination (mean of member logits): synthetic hold-out errors, values within 1 logit of the
threshold, the smallest synthetic margin, and errors on the 30 seen real photos (seen in training: a
sanity check only, not a generalisation measure).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
from isu_core import FEATURES, VALUES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("logits")
ap.add_argument("--data", default="data")
args = ap.parse_args()

L = json.loads(Path(args.logits).read_text())
members = [m for m in next(iter(L.values())) if m != "ENSEMBLE"]
labels = {}
for key in L:
    branch, rest = key.split("/")
    labels[key] = json.loads((Path(args.data) / branch / "labels" / f"{rest.rsplit('_', 1)[0]}.json").read_text())

for r in range(1, len(members) + 1):
    for combo in itertools.combinations(members, r):
        sim_wrong, real_wrong, near, worst = [], [], 0, (1e9, "")
        for key, per in L.items():
            lg = np.mean([per[m] for m in combo], axis=0)
            for i, f in enumerate(FEATURES):
                v = labels[key].get(f)
                if v is None:
                    continue
                y = VALUES[f].index(str(v).strip().upper())
                signed = lg[i] if y else -lg[i]
                if key.endswith("_sim"):
                    near += abs(lg[i]) < 1.0
                    worst = min(worst, (round(float(signed), 2), f"{key.split('/')[1]}:{f}"))
                    if signed <= 0:
                        sim_wrong.append(f"{key.split('/')[1]}:{f}")
                elif signed <= 0:
                    real_wrong.append(f"{key.split('/')[1]}:{f}")
        print(f"{'+'.join(combo):14s} synthetic hold-out errors {len(sim_wrong):2d}  near(<1) {near:2d}  "
              f"min margin {worst[0]:6.2f} ({worst[1]})  seen-real errors {len(real_wrong)}"
              + (f"  sim wrong: {sim_wrong[:6]}" if sim_wrong else ""))
