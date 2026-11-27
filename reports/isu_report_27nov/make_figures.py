"""Figures for the 27 Nov report, drawn only from data/metrics.json (written by compute_metrics.py).

Run from the repository root with a Python that has matplotlib (requirements-report.txt):
    python reports/isu_report_27nov/make_figures.py
Writes figures/*.pdf (fonts embedded as TrueType for the ACM checker) and *.png previews.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figures"
M = json.loads((HERE / "data" / "metrics.json").read_text())

COL_W = 3.33  # ACM sigconf column width (in)
FULL_W = 7.0
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8986", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # reference palette slots 1-4, fixed order
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1: teaser
def teaser():
    t = M["teaser"]
    pred = t["pred"]
    img = Image.open(ROOT / t["image"]).convert("RGB")
    fig = plt.figure(figsize=(COL_W, 2.36))
    ax = fig.add_axes([0, 0.31, 1, 0.69])
    ax.imshow(img)
    ax.set_axis_off()
    yn = lambda f: "belt" if pred[f] == "YES" else "no belt"
    rows = [
        ("Driver", f"{pred['driver_gender']}, {pred['driver_tshirt_color']} shirt, {pred['driver_emotion']}, "
                   f"phone {pred['driver_phone']}, {yn('driver_safety_belt')}"),
        ("Co-driver", f"{pred['passenger_codriver_gender']}, {pred['passenger_codriver_tshirt_color']} shirt, "
                      f"{pred['passenger_codriver_emotion']}, {yn('codriver_safety_belt')}"),
        ("Rear left", f"{pred['passenger_rear_left_gender']}, {pred['passenger_rear_left_tshirt_color']} shirt, "
                      f"{pred['passenger_rear_left_emotion']}, {yn('passenger_rear_left_safety_belt')}"),
        ("Rear right", f"{pred['passenger_rear_right_gender']}, {pred['passenger_rear_right_tshirt_color']} shirt, "
                       f"{pred['passenger_rear_right_emotion']}, {yn('passenger_rear_right_safety_belt')}"),
        ("Objects", f"suitcase {pred['suitcase']} ({pred['suitcase_color']}, {pred['suitcase_location']}, "
                    f"{pred['suitcase_pose']}); baby seat {pred['baby_seat']}; co-driver seat: phone "
                    f"{pred['phone_codriver_seat']}, bottle {pred['colabottle_codriver_seat']}, can {pred['colacan_codriver_seat']}"),
    ]
    tx = fig.add_axes([0, 0, 1, 0.295])
    tx.set_axis_off()
    for i, (who, txt) in enumerate(rows):
        y = 0.92 - i * 0.205
        tx.text(0.0, y, who, fontsize=6.2, fontweight="bold", color=INK, va="top")
        tx.text(0.17, y, txt.lower().replace("_", " "), fontsize=6.2, color=INK2, va="top", wrap=True)
    save(fig, "fig_teaser")


# ---------------------------------------------------------------- Fig. 2: method schematic
def method():
    v2 = M["runs"]["v2"]
    w, h = v2["input_size"]
    pb = v2["params_backbone"] / 1e6
    n_sim = 32 - v2["cfg"]["s2_real_per_batch"] - v2["cfg"]["s2_twin_per_batch"]
    fig, ax = plt.subplots(figsize=(COL_W, 1.62))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.set_axis_off()

    def box(x0, x1, y0, y1, text, fill="#f3f2ef", edge=MUTED, bold=False):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=fill, ec=edge, lw=0.6))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=5.1, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, lw=0.6, color=INK2))

    ax.text(0, 48.5, "Inference", fontsize=6, color=INK2, fontweight="bold", va="top")
    ax.text(0, 22.5, "Training", fontsize=6, color=INK2, fontweight="bold", va="top")
    top = (29, 43)
    xs = [(0, 14), (17.5, 37.5), (41, 58), (61.5, 81), (84.5, 100)]
    labels = [f"image\n{w}$\\times${h}", f"DINOv2-B/14\n{pb:.1f} M\nparams",
              f"{w // 14}$\\times${h // 14}\ntokens\n+ CLS", f"{len(M['features'])} queries\ncross-\nattention",
              f"{len(M['features'])} logits\n$\\rightarrow$\nanswers"]
    for (a, b), lab in zip(xs, labels):
        box(a, b, *top, lab, fill="#e8f0fb" if lab.startswith("DINO") else "#f3f2ef")
    for (a, b), (c, d) in zip(xs, xs[1:]):
        arrow(b, sum(top) / 2, c, sum(top) / 2)
    bot = (3, 18)
    tx = [(0, 24), (27.5, 72.5), (76, 100)]
    s1 = v2["cfg"]["s1_epochs"]
    tl = [f"Stage 1\nrenders only\n{s1} epochs",
          f"Stage 2: 5 folds $\\times$ {v2['cfg']['s2_steps']} steps\nbatch: {n_sim} renders +\n{v2['cfg']['s2_real_per_batch']} photos + {v2['cfg']['s2_twin_per_batch']} twins",
          "average the\n5 fold\nmodels"]
    for (a, b), lab in zip(tx, tl):
        box(a, b, *bot, lab)
    for (a, b), (c, d) in zip(tx, tx[1:]):
        arrow(b, sum(bot) / 2, c, sum(bot) / 2)
    ax.add_patch(FancyArrowPatch((86.5, bot[1]), (28.5, top[0]), connectionstyle="arc3,rad=0.18", arrowstyle="-|>",
                                 mutation_scale=5, lw=0.6, color=INK2, linestyle=(0, (2, 1.5))))
    ax.text(50, 25.3, "shipped weights", fontsize=5.4, color=INK2, ha="center")
    save(fig, "fig_method")


# ---------------------------------------------------------------- Fig. 3: results
def results():
    groups = ["driver_emotion", "driver_phone", "driver_safety_belt"]
    names = ["v1", "v2", "v2s1", "v4", "v2+v2s1", "v2+v4", "v2+v2s1+v4"]
    counts = {}
    for n in names:
        if n == "v1":
            per = {}
            for f in M["runs"]["v1"]["folds"]:
                for k, c in f["real_per_feature"].items():
                    per[k] = per.get(k, 0) + c
        else:
            per = {}
            for e in M["cv"][n]["errors"]:
                per[e["feature"]] = per.get(e["feature"], 0) + 1
        counts[n] = [per.get(g, 0) for g in groups] + [sum(c for k, c in per.items() if k not in groups)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 1.78), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.32})
    # (a) real-photo CV errors by feature
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    seg_names = ["driver emotion", "driver phone", "driver belt", "other"]
    for j, (lab, col) in enumerate(zip(seg_names, SERIES)):
        vals = np.array([counts[n][j] for n in names], float)
        a1.barh(y, vals, left=left, height=0.62, color=col, edgecolor="white", linewidth=0.8, label=lab)
        left += vals
    for yi, tot in zip(y, left):
        a1.text(tot + 0.12, yi, f"{int(tot)}", va="center", fontsize=6.5, color=INK)
    a1.set_yticks(y)
    a1.set_yticklabels(names)
    a1.set_xlim(0, 7.2)
    a1.set_xlabel(f"wrong answers in 5-fold real-photo CV (of {M['cv']['v2']['n']})")
    a1.xaxis.grid(True, color=GRID, lw=0.5)
    a1.set_axisbelow(True)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a1.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, handlelength=1.0,
              columnspacing=0.9, handletextpad=0.4)
    fig.text(0.005, 0.97, "(a)", fontweight="bold", fontsize=7)

    # (b) signed margins on the 500-render synthetic hold-out
    hn = ["v2", "v2s1", "v4", "v2+v2s1", "v2+v4", "v2+v2s1+v4"]
    data = [np.array(M["holdout"][n]["signed_margins"]) for n in hn]
    pos = np.arange(len(hn))
    bp = a2.boxplot(data, positions=pos, widths=0.5, whis=(1, 99), showfliers=False, patch_artist=True,
                    medianprops={"color": INK, "lw": 0.8}, whiskerprops={"color": INK2, "lw": 0.6},
                    capprops={"color": INK2, "lw": 0.6}, boxprops={"lw": 0.6, "edgecolor": INK2})
    for b in bp["boxes"]:
        b.set_facecolor("#e8f0fb")
    for p, d, n in zip(pos, data, hn):
        lo = d[d < np.quantile(d, 0.01)]
        a2.scatter(np.full(len(lo), p) + np.random.RandomState(0).uniform(-0.12, 0.12, len(lo)), lo, s=2.5,
                   color=MUTED, lw=0, zorder=3)
        neg = d[d <= 0]
        if len(neg):
            a2.scatter(np.full(len(neg), p), neg, marker="x", s=12, color=INK, lw=0.8, zorder=4)
            a2.annotate(f"{len(neg)} wrong", (p, neg.min()), xytext=(p + 0.35, neg.min() - 1.0), fontsize=6,
                        color=INK, arrowprops={"arrowstyle": "-", "lw": 0.5, "color": INK2})
    a2.axhline(0, color=INK2, lw=0.6, ls=(0, (3, 2)))
    a2.set_xticks(pos)
    a2.set_xticklabels([n.replace("+", "+\n", 1) for n in hn])
    a2.set_ylabel("signed margin (logit)")
    a2.yaxis.grid(True, color=GRID, lw=0.5)
    a2.set_axisbelow(True)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    a2.set_ylim(-4.5, 25)
    fig.text(0.515, 0.97, "(b)", fontweight="bold", fontsize=7)
    save(fig, "fig_results")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    if (ROOT / M["teaser"]["image"]).exists():
        teaser()
    else:  # the render comes from the dataset (tools/download_data.py); keep the committed figure
        print("dataset render not found; keeping figures/fig_teaser.pdf")
    method()
    results()
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))
