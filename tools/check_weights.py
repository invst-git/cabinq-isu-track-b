"""Parity check: run trained weights locally through submission/isu_core.Engine and compare with Kaggle.

Usage (project root):
    python -I tools/check_weights.py --weights work/train_v1/soup --reports work/train_v1/reports.json \
        --splits work/train_v1/splits.json --data data/main

Local errors on the 30 real photos must equal the Kaggle soup report's real_all_seen errors, and local
renders that were in the Kaggle hold-out must be as accurate as Kaggle reported. Any difference means
preprocessing or library versions drift between training (Kaggle) and inference (evaluation image).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))

from PIL import Image  # noqa: E402

from isu_core import Engine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--reports")
ap.add_argument("--splits")
ap.add_argument("--data", default=str(ROOT / "data" / "main"))
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

data = Path(args.data)
t0 = time.perf_counter()
eng = Engine(Path(args.weights), args.device)
eng.warmup(1)
print(f"loaded {args.weights} on {args.device} in {time.perf_counter() - t0:.1f}s; meta stage={eng.meta.get('stage')} "
      f"torch(train)={eng.meta.get('torch')} transformers(train)={eng.meta.get('transformers')}")

items = [(f"main/{p.stem.replace('_real', '')}_real", p) for p in sorted((data / "real").glob("*_real.jpg"))]
items += [(f"main/{p.stem}", p) for p in sorted((data / "images").glob("*_sim.png"))]
holdout = set(json.loads(Path(args.splits).read_text())["holdout"]) if args.splits else set()

errors, n_feat, times = {}, {"real": 0, "sim_holdout": 0, "sim_other": 0}, []
errs_by_group = {k: 0 for k in n_feat}
for key, path in items:
    stem = path.stem.replace("_real", "").replace("_sim", "")
    label = json.loads((data / "labels" / f"{stem}.json").read_text())
    with Image.open(path) as im:
        img = im.convert("RGB")
    t = time.perf_counter()
    pred = eng.predict(img)
    times.append(time.perf_counter() - t)
    group = "real" if key.endswith("_real") else ("sim_holdout" if key in holdout else "sim_other")
    for f, v in label.items():
        if f in pred and v is not None:
            n_feat[group] += 1
            if pred[f] != str(v).strip().upper():
                errs_by_group[group] += 1
                errors.setdefault(key, []).append(f)

for g in n_feat:
    if n_feat[g]:
        print(f"{g:12s}: acc {1 - errs_by_group[g] / n_feat[g]:.4f} ({errs_by_group[g]}/{n_feat[g]} wrong)")
print(f"CPU latency: mean {sum(times) / len(times):.3f}s, max {max(times):.3f}s over {len(times)} images")

if args.reports:
    rep = json.loads(Path(args.reports).read_text())
    stage = eng.meta.get("stage", "soup")
    kaggle = rep.get(stage, {}).get("real_all_seen") or rep.get(stage, {}).get("real_test")
    if kaggle is not None:
        k_err = {(e["key"], e["feature"]) for e in kaggle["errors"]}
        l_err = {(k, f) for k, fs in errors.items() if k.endswith("_real") for f in fs}
        same = k_err == l_err
        print(f"PARITY real photos vs Kaggle '{stage}': {'MATCH' if same else 'MISMATCH'} "
              f"(kaggle {len(k_err)} errors, local {len(l_err)})")
        for e in sorted(k_err ^ l_err):
            print(f"  differs: {e} ({'kaggle only' if e in k_err else 'local only'})")
for k, fs in sorted(errors.items()):
    print(f"  error {k}: {fs}")
