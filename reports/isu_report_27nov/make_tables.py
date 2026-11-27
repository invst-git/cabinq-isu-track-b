"""LaTeX tables and number macros for the 27 Nov report, generated from data/metrics.json.

    python reports/isu_report_27nov/make_tables.py
Writes tab_official.tex, tab_dev.tex and numbers.tex next to this script.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = json.loads((HERE / "data" / "metrics.json").read_text())
OFF = M["official"]
N_REAL = 479  # real-photo features in the scored pool; checked against every official real accuracy below


def off(name):
    e = OFF[name]["evaluations"][-1]["metrics"]
    n_all, correct = int(e["evaluated_features"]), int(e["correct_features"])
    syn_wrong = 0 if e["accuracy_synthetic"] == 1 else None
    assert syn_wrong == 0, f"{name}: synthetic errors present, derive counts differently"
    real_wrong = n_all - correct
    assert abs((1 - real_wrong / N_REAL) - e["accuracy_real"]) < 5e-5, name
    return {"pr": OFF[name]["pr"], "correct": correct, "n_all": n_all, "real_ok": N_REAL - real_wrong,
            "syn_ok": n_all - N_REAL, "n_syn": n_all - N_REAL,
            "real_exact": round(e["scene_exact_match_real"] * e["scenes_real"]), "n_real_scenes": int(e["scenes_real"]),
            "lat": e["latency_seconds_mean"], "commit": OFF[name]["evaluations"][-1]["commit"][:7]}


fmt = lambda n: f"{n:,}".replace(",", "{,}")
bold = lambda s, on: f"\\textbf{{{s}}}" if on else s
names1 = [("v1", "v1"), ("v2", "v2 (final)"), ("v2+v2s1", "v2 + v2s1"), ("v2+v4", "v2 + v4")]
O = {n: off(n) for n, _ in names1}
best = {k: max(o[k] for o in O.values()) for k in ("correct", "real_ok", "syn_ok", "real_exact")}
best_lat = min(o["lat"] for o in O.values())
rows = []
for name, label in names1:
    o = O[name]
    rows.append(f"{label} & \\#{o['pr']} & {bold(fmt(o['correct']), o['correct'] == best['correct'])} & "
                f"{bold(o['real_ok'], o['real_ok'] == best['real_ok'])} & {bold(fmt(o['syn_ok']), o['syn_ok'] == best['syn_ok'])} & "
                f"{bold(str(o['real_exact']) + '/' + str(o['n_real_scenes']), o['real_exact'] == best['real_exact'])} & "
                f"{bold(f'{o[chr(108) + chr(97) + chr(116)]:.3f}', o['lat'] == best_lat)} \\\\")
o2 = off("v2")
tab1 = r"""\begin{table}[t]
\caption{Official evaluation results of the submitted variants on an NVIDIA L4: correct answers out of %s (%d on real photos, %s on renders), real scenes answered without error, and mean latency per image. Bold indicates the best values.}
\label{tab:official}
\small
\begin{tabular}{@{}lcrrrcr@{}}
\toprule
System & PR & All & Real & Synth. & Real exact & s/img \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (fmt(o2["n_all"]), N_REAL, fmt(o2["n_syn"]), "\n".join(rows))
(HERE / "tab_official.tex").write_text(tab1)

# development table
R, CV, HO = M["runs"], M["cv"], M["holdout"]
bb = {"v1": "DINOv2-B", "v2": "DINOv2-B", "v2s1": "DINOv2-B", "v4": "DINOv3-B"}
singles, ens = ["v1", "v2", "v2s1", "v4"], ["v2+v2s1", "v2+v4", "v2+v2s1+v4"]
ho_err = {**{v: R[v]["soup"]["holdout_wrong"] for v in singles}, **{n: HO[n]["wrong"] for n in ens}}
cv_err = {**{v: R[v]["cv_wrong"] for v in singles}, **{n: CV[n]["wrong"] for n in ens}}
b_ho, b_cv = min(ho_err.values()), min(cv_err.values())
b_mn = max(HO[n]["min"] for n in singles[1:] + ens)
b_ll = min(CV[n]["logloss"] for n in singles[1:] + ens)
dev = []
for v in singles:
    w, h = R[v]["input_size"]
    mn = bold(f"${HO[v]['min']:.1f}$", HO[v]["min"] == b_mn) if v in HO else "--"
    ll = bold(f"{CV[v]['logloss']:.3f}", CV[v]["logloss"] == b_ll) if v in CV else "--"
    dev.append(f"{v} & {w}$\\times${h} & {bold(ho_err[v], ho_err[v] == b_ho)} & {mn} & "
               f"{bold(cv_err[v], cv_err[v] == b_cv)} & {ll} & {R[v]['train_wall_h']:.1f} \\\\")
dev.append(r"\midrule")
for n, lab in [("v2+v2s1", "v2 + v2s1"), ("v2+v4", "v2 + v4"), ("v2+v2s1+v4", "v2 + v2s1 + v4")]:
    dev.append(f"{lab} & -- & {bold(ho_err[n], ho_err[n] == b_ho)} & "
               f"{bold(f'{HO[n][chr(109) + chr(105) + chr(110)]:.1f}', HO[n]['min'] == b_mn)} & {bold(cv_err[n], cv_err[n] == b_cv)} & "
               f"{bold(f'{CV[n][chr(108) + chr(111) + chr(103) + chr(108) + chr(111) + chr(115) + chr(115)]:.3f}', CV[n]['logloss'] == b_ll)} & -- \\\\")
tab2 = r"""\begin{table}[t]
\caption{Development results. Synthetic hold-out: errors and smallest signed margin (logits) over %s answers on %d unseen renders. Real-photo CV: errors and log-loss over %d answers. Training time in hours on 2$\times$T4; margin and log-loss are unavailable for v1. Bold indicates the best values.}
\label{tab:dev}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcrrrrr@{}}
\toprule
& & \multicolumn{2}{c}{Synth.\ hold-out} & \multicolumn{2}{c}{Real-photo CV} & Train \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Variant & Input & errors & margin & errors & log-loss & h \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (fmt(HO["v2"]["n"]), M["holdout_scenes"] - 30 if M["holdout_scenes"] > 500 else M["holdout_scenes"], CV["v2"]["n"], "\n".join(dev))
(HERE / "tab_dev.tex").write_text(tab2)

# number macros used in the prose
v2 = R["v2"]
c = v2["cfg"]
mac = {
    "nAll": fmt(o2["n_all"]), "nReal": str(N_REAL), "nSyn": fmt(o2["n_syn"]),
    "vTwoCorrect": fmt(o2["correct"]), "vOneCorrect": fmt(off("v1")["correct"]),
    "vTwoLat": f"{o2['lat'] * 1000:.0f}", "vOneLat": f"{off('v1')['lat'] * 1000:.0f}",
    "ensLat": f"{off('v2+v4')['lat'] * 1000:.0f}",
    "nParams": f"{v2['params_total'] / 1e6:.1f}", "nParamsBB": f"{v2['params_backbone'] / 1e6:.1f}",
    "nParamsHead": f"{(v2['params_total'] - v2['params_backbone']) / 1e6:.2f}",
    "sOneEpochs": str(c["s1_epochs"]), "sTwoSteps": str(c["s2_steps"]), "batch": str(c["batch"]),
    "sOneBbLr": f"{c['s1_bb_lr']:.0e}".replace("e-0", "\\times10^{-") + "}",
    "layerDecay": str(c["layer_decay"]), "weightDecay": str(c["wd"]), "emaOne": str(c["s1_ema"]), "emaTwo": str(c["s2_ema"]),
    "nHoldout": str(c["holdout"]), "nHoldoutAns": fmt(HO["v2"]["n"]), "nCvAns": str(CV["v2"]["n"]),
    "vTwoTrainH": f"{v2['train_wall_h']:.1f}", "vTwoMinMargin": f"{HO['v2']['min']:.1f}",
    "graphEager": f"{M['cuda_graph']['eager_ms'][0]:.1f}", "graphGraph": f"{M['cuda_graph']['graph_ms'][0]:.1f}",
}
(HERE / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
print(tab1)
print(tab2)
print(open(HERE / "numbers.tex").read())
