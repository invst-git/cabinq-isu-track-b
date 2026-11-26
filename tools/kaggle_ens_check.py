"""Ensemble gate on Kaggle (T4): every attached soup alone and their mean-logit ensemble, on the 500-render
synthetic hold-out and the 30 (seen) real photos. Errors, near-threshold counts, smallest margins, T4 timing.
Writes /kaggle/working/ens_check.json and ens_logits.json. Bundled with isu_core by kaggle/build_check_kernel.py.
"""
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from isu_core import FEATURES, VALUES, EnsembleEngine, Engine, prep  # BUNDLE-STRIP

OUT = Path("/kaggle/working")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==__TRANSFORMERS_PIN__", "timm==1.0.31"],
               check=False)
root = Path("/tmp/isu")
for t in sorted(Path("/kaggle/input").rglob("isu_*.tar")):
    branch = t.stem.replace("isu_", "")
    if not (root / branch / "labels").is_dir():
        with tarfile.open(t) as tf:
            tf.extractall(root, filter="data")
soups = sorted(p.parent for p in Path("/kaggle/input").rglob("soup/model.safetensors"))
names = [json.loads((s / "meta.json").read_text()).get("variant", s.parent.name) for s in soups]
splits = [json.loads((s.parent / "splits.json").read_text()) for s in soups]
assert all(sp["holdout"] == splits[0]["holdout"] for sp in splits), "members used different hold-outs"
ens_dir = Path("/tmp/ens")
ens_dir.mkdir(exist_ok=True)
for n, s in zip(names, soups):
    (ens_dir / n).symlink_to(s, target_is_directory=True)
(ens_dir / "ensemble.json").write_text(json.dumps({"members": names, "combine": "mean_logit"}))
print("members:", list(zip(names, map(str, soups))), flush=True)

eng = EnsembleEngine(ens_dir, "cuda")
eng.warmup()
keys = splits[0]["holdout"] + [f"main/sample_{i:05d}_real" for i in range(30)]


def paths(key):
    branch, rest = key.split("/")
    stem, kind = rest.rsplit("_", 1)
    img = root / branch / ("images" if kind == "sim" else "real") / f"{stem}_{kind}.{'png' if kind == 'sim' else 'jpg'}"
    return img, root / branch / "labels" / f"{stem}.json"


res = {n: {"wrong": [], "near1": 0, "min": [1e9, None]} for n in names + ["ENSEMBLE"]}
logits_out, times = {}, []
for key in keys:
    ip, lp = paths(key)
    with Image.open(ip) as im:
        image = im.convert("RGB")
    label = json.loads(lp.read_text())
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.inference_mode():
        x = torch.from_numpy(prep(image)).unsqueeze(0).to(eng.device)
        per = [m.logits_from_u8(x)[0].cpu().numpy() for m in eng.members]
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t)
    allm = dict(zip(names, per))
    allm["ENSEMBLE"] = np.mean(per, axis=0)
    logits_out[key] = {k: [round(float(v), 4) for v in lg] for k, lg in allm.items()}
    for n, lg in allm.items():
        for i, f in enumerate(FEATURES):
            v = label.get(f)
            if v is None:
                continue
            y = VALUES[f].index(str(v).strip().upper())
            if (lg[i] > 0) != bool(y):
                res[n]["wrong"].append([key, f, round(float(lg[i]), 3)])
            if key.endswith("_sim"):
                m = abs(float(lg[i]))
                res[n]["near1"] += m < 1.0
                if m < res[n]["min"][0]:
                    res[n]["min"] = [round(m, 3), f"{key}:{f}"]
for n in res:
    res[n]["n_wrong_holdout"] = sum(1 for w in res[n]["wrong"] if w[0].endswith("_sim"))
    res[n]["n_wrong_real_seen"] = sum(1 for w in res[n]["wrong"] if w[0].endswith("_real"))
res["t4_ms_mean_all_members"] = round(1000 * float(np.mean(times[5:])), 1)
(OUT / "ens_check.json").write_text(json.dumps(res, indent=2))
(OUT / "ens_logits.json").write_text(json.dumps(logits_out))
print(json.dumps({n: {k: v for k, v in r.items() if k != "wrong"} if isinstance(r, dict) else r
                  for n, r in res.items()}, indent=1), flush=True)
