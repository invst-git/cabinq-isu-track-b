"""Collect every number used in the 27 Nov report from saved run artefacts (no hand-typed results).

Run from the repository root with requirements.txt installed (needs torch for the teaser prediction):
    python -I reports/isu_report_27nov/compute_metrics.py
Writes reports/isu_report_27nov/data/metrics.json (read by make_figures.py and make_tables.py).
"""
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submission"))
from isu_core import FEATURES, VALUES  # noqa: E402

OUT = ROOT / "reports" / "isu_report_27nov" / "data"
RUNS = ["v1", "v2", "v2s1", "v4"]
LABELS = ROOT / "data"


def label(key):
    branch, rest = key.split("/")
    return json.loads((LABELS / branch / "labels" / f"{rest.rsplit('_', 1)[0]}.json").read_text())


def yvec(lab):
    y, m = np.zeros(len(FEATURES)), np.zeros(len(FEATURES))
    for i, f in enumerate(FEATURES):
        v = lab.get(f)
        if v is not None:
            y[i], m[i] = VALUES[f].index(str(v).strip().upper()), 1
    return y, m


def n_params(st_path):
    with open(st_path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    total = back = 0
    for k, v in header.items():
        if k == "__metadata__":
            continue
        c = int(np.prod(v["shape"]))
        total += c
        back += c if k.startswith("backbone.") else 0
    return total, back


def parse_log(path):
    steps, times = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        t = re.match(r"\[(\d\d):(\d\d):(\d\d)\]", line)
        if t:
            times.append(int(t[1]) * 3600 + int(t[2]) * 60 + int(t[3]))
        m = re.search(r"\] (stage1|fold\d) step (\d+)/(\d+) loss ([\d.]+) \| ([\d.]+)s/step", line)
        if m:
            steps.append({"phase": m[1], "step": int(m[2]), "total": int(m[3]), "loss": float(m[4]), "s_per_step": float(m[5])})
    wall = 0
    for a, b in zip(times, times[1:]):
        wall += (b - a) % 86400
    return steps, wall


res = {"features": FEATURES, "runs": {}}

# ---- per-run development results (Kaggle reports.json) ----
for v in RUNS:
    rep = json.loads((ROOT / f"work/train_{v}/reports.json").read_text())
    meta = json.loads((ROOT / f"work/train_{v}/soup/meta.json").read_text()) if (ROOT / f"work/train_{v}/soup/meta.json").exists() else {}
    r = {"stage1": [], "folds": [], "soup": {}, "cfg": meta.get("cfg", {}), "input_size": meta.get("input_size")}
    for tag, x in rep.items():
        if tag.startswith("stage1"):
            r["stage1"].append({"step": int(tag.split()[-1]), "holdout_wrong": x["holdout_sim"]["n_wrong"],
                                "holdout_n": x["holdout_sim"]["n_features"],
                                "zeroshot_wrong": x["real_all_zero_shot"]["n_wrong"],
                                "zeroshot_n": x["real_all_zero_shot"]["n_features"],
                                "zeroshot_per_feature": x["real_all_zero_shot"]["per_feature_wrong"]})
        elif tag.startswith("fold"):
            r["folds"].append({"fold": int(tag[4:]), "holdout_wrong": x["holdout_sim"]["n_wrong"],
                               "real_wrong": x["real_test"]["n_wrong"], "real_n": x["real_test"]["n_features"],
                               "real_per_feature": x["real_test"]["per_feature_wrong"]})
        elif tag == "soup":
            r["soup"] = {"holdout_wrong": x["holdout_sim"]["n_wrong"], "real_seen_wrong": x["real_all_seen"]["n_wrong"]}
    r["cv_wrong"] = sum(f["real_wrong"] for f in r["folds"])
    r["cv_n"] = sum(f["real_n"] for f in r["folds"])
    steps, wall = parse_log(ROOT / f"work/train_{v}/train_log.txt")
    r["train_steps"] = steps
    r["train_wall_h"] = round(wall / 3600, 2)
    st = ROOT / f"work/train_{v}/soup/model.safetensors"
    if st.exists():
        r["params_total"], r["params_backbone"] = n_params(st)
    res["runs"][v] = r

# ---- real-photo CV logits (fold models scored on photos they never trained on) ----
def cv_logits(v):
    rep = json.loads((ROOT / f"work/train_{v}/reports.json").read_text())
    out = {}
    for tag, x in rep.items():
        if tag.startswith("fold") and "probs" in x["real_test"]:
            for key, p in x["real_test"]["probs"].items():
                p = np.clip(np.array(p, dtype=np.float64), 1e-6, 1 - 1e-6)
                out[key] = np.log(p / (1 - p))
    return out


cv = {v: cv_logits(v) for v in ["v2", "v2s1", "v4"]}
keys = sorted(cv["v2"])
Y = np.zeros((len(keys), len(FEATURES)))
M = np.zeros_like(Y)
for n, k in enumerate(keys):
    Y[n], M[n] = yvec(label(k))


def judge(L):
    signed = np.where(Y > 0.5, L, -L)
    wrong = (signed <= 0) & (M > 0)
    p = 1 / (1 + np.exp(-L))
    ll = -(Y * np.log(np.clip(p, 1e-9, 1)) + (1 - Y) * np.log(np.clip(1 - p, 1e-9, 1)))
    errs = [{"photo": keys[r].split("/")[1][7:12], "feature": FEATURES[c], "logit": round(float(L[r, c]), 2)}
            for r, c in zip(*np.nonzero(wrong))]
    return {"wrong": int(wrong.sum()), "n": int(M.sum()), "logloss": round(float((ll * M).sum() / M.sum()), 4),
            "errors": errs, "signed_margins": [round(float(s), 3) for s in signed[M > 0]]}


combos = {"v2": ["v2"], "v2s1": ["v2s1"], "v4": ["v4"], "v2+v2s1": ["v2", "v2s1"], "v2+v4": ["v2", "v4"],
          "v2+v2s1+v4": ["v2", "v2s1", "v4"]}
res["cv"] = {name: judge(np.mean([np.stack([cv[m][k] for k in keys]) for m in ms], axis=0)) for name, ms in combos.items()}

# ---- synthetic hold-out margins from the Kaggle ensemble check (per-member logits) ----
EL = json.loads((ROOT / "work/enscheck/ens_logits.json").read_text())
sim_keys = [k for k in EL if k.endswith("_sim")]
res["holdout"] = {}
for name, ms in combos.items():
    signed = []
    for k in sim_keys:
        lg = np.mean([EL[k][m] for m in ms], axis=0)
        y, m = yvec(label(k))
        signed += [float(lg[i] if y[i] else -lg[i]) for i in range(len(FEATURES)) if m[i]]
    s = np.array(signed)
    res["holdout"][name] = {"n": len(s), "wrong": int((s <= 0).sum()), "min": round(float(s.min()), 2),
                            "near1": int((np.abs(s) < 1).sum()), "q01": round(float(np.quantile(s, 0.01)), 2),
                            "median": round(float(np.median(s)), 2), "signed_margins": [round(x, 3) for x in s]}
res["holdout_scenes"] = len(sim_keys)
res["ens_t4_ms_three_members"] = json.loads((ROOT / "work/enscheck/ens_check.json").read_text())["t4_ms_mean_all_members"]

# ---- official evaluations ----
res["official"] = json.loads((OUT / "official_results.json").read_text())

# ---- other measured facts ----
chk = json.loads((ROOT / "work/checks_0d/checks.json").read_text())
res["cuda_graph"] = {"eager_ms": chk["B_cuda_graph"]["eager_ms_mean_p99"], "graph_ms": chk["B_cuda_graph"]["graph_ms_mean_p99"],
                     "bit_equal": chk["B_cuda_graph"]["probs_bit_equal"]}
res["dinov3_fp16_cos_min"] = chk["C_dinov3_b_fp16"]["1088x608"]["fp16_vs_fp32_cos_min"]

# ---- teaser: v2 soup prediction on an unseen synthetic hold-out render with many occupants/objects ----
split = json.loads((ROOT / "work/train_v2/splits.json").read_text())
cands = [k for k in split["holdout"] if k.startswith("main/")
         and (LABELS / "main" / "images" / f"{k.split('/')[1]}.png").exists()]
presence = ["passenger_codriver", "passenger_rear_seat_left", "passenger_rear_seat_right", "driver_phone",
            "driver_safety_belt", "phone_codriver_seat", "colabottle_codriver_seat", "colacan_codriver_seat", "suitcase",
            "baby_seat"]
score = lambda k: sum(str(label(k).get(f, "")).upper() == "YES" for f in presence)
best = max(cands, key=lambda k: (score(k), k))
res["teaser"] = {"key": best, "label": label(best)}
try:
    import torch
    from PIL import Image

    from isu_core import load_engine
    torch.set_grad_enabled(False)
    eng = load_engine(ROOT / "work/submission_v2/weights", "cpu")
    img_path = LABELS / "main" / "images" / f"{best.split('/')[1]}.png"
    with Image.open(img_path) as im:
        image = im.convert("RGB")
    pred = eng.predict(image)
    res["teaser"].update({"image": img_path.relative_to(ROOT).as_posix(), "pred": pred,
                          "n_correct": sum(str(pred[f]) == str(label(best).get(f)).upper() for f in FEATURES if label(best).get(f) is not None),
                          "n_labelled": sum(label(best).get(f) is not None for f in FEATURES)})
except Exception as exc:  # the figure falls back to labels only
    res["teaser"]["error"] = repr(exc)

(OUT / "metrics.json").write_text(json.dumps(res, indent=1))
print("wrote", OUT / "metrics.json")
for v in RUNS:
    r = res["runs"][v]
    print(v, "cv", r["cv_wrong"], "/", r["cv_n"], "| folds", [f["real_wrong"] for f in r["folds"]],
          "| stage1", [(s["holdout_wrong"], s["zeroshot_wrong"]) for s in r["stage1"]], "| soup", r["soup"],
          "| wall h", r["train_wall_h"], "| params", r.get("params_total"), r.get("params_backbone"), "| in", r["input_size"])
for n, x in res["cv"].items():
    print("CV", n, x["wrong"], x["n"], x["logloss"], [(e["photo"], e["feature"]) for e in x["errors"]])
for n, x in res["holdout"].items():
    print("HO", n, {k: v for k, v in x.items() if k != "signed_margins"})
print("teaser", res["teaser"].get("key"), res["teaser"].get("n_correct"), res["teaser"].get("n_labelled"), res["teaser"].get("error"))
