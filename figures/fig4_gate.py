"""
fig4_gate.py
============
Fig. 4: LS-ADLC gate behaviour on the real TUEV training pool, one panel per
anatomical group, at final print size (7.48 in wide).

For every training window the gate reads the out-of-fold, distance-weighted
k-NN vote (k=15) stored in data/oof_pG_<group>.npz (column of `knn_w` whose
`knn_grid` entry is 15), exactly as src/train.py does in `--lsadlc adlc`
mode. A window is flagged when vote < tau (0.80) and its label is not BCKG
(configs/config.yaml, lsadlc_train.adlc). Bars split each class into
vote = 0, 0 < vote < tau (both flagged) and vote >= tau (not flagged); the
number printed at the right is the flagged count, which matches the
"ADLC gate" line of the pLS training logs.

Writes figures/output/fig_gate.{png,pdf} and prints the per-class counts used in the text.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, ".."))                 # repository root
OUTS = [os.path.join(HERE, "output")]
plt.rcParams["font.family"] = "DejaVu Sans"
GROUPS = ["temporal", "central", "parasagittal"]
NAMES = ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"]
K, TAU, EXCL = 15, 0.80, 5
C_ZERO, C_LOW, C_KEEP = "#a31515", "#e8a0a0", "#d9d9d9"


def gate_table(group):
    d = np.load(os.path.join(R, "data", f"oof_pG_{group}.npz"))
    v = d["knn_w"][:, list(d["knn_grid"]).index(K)]
    y = d["y"].astype(int)
    rows = []
    for c in range(len(NAMES)):
        vc = v[y == c]
        rows.append(dict(n=len(vc), zero=int((vc == 0).sum()),
                         low=int(((vc > 0) & (vc < TAU)).sum()), keep=int((vc >= TAU).sum()),
                         flagged=0 if c == EXCL else int((vc < TAU).sum())))
    return rows


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.25), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.17, wspace=0.12)
    ypos = np.arange(len(NAMES))[::-1]
    for ax, g in zip(axes, GROUPS):
        T = gate_table(g)
        print(g, {NAMES[c]: (t["flagged"], round(100 * t["flagged"] / t["n"], 1),
                             round(100 * t["zero"] / t["n"], 1)) for c, t in enumerate(T)})
        for c, (t, yy) in enumerate(zip(T, ypos)):
            z, lo, ke = (100 * t[k] / t["n"] for k in ("zero", "low", "keep"))
            hatch = "/////" if c == EXCL else None
            ax.barh(yy, z, color=C_ZERO if c != EXCL else "#bdbdbd", height=0.68,
                    hatch=hatch, edgecolor="white", lw=0)
            ax.barh(yy, lo, left=z, color=C_LOW if c != EXCL else "#e6e6e6", height=0.68,
                    hatch=hatch, edgecolor="white", lw=0)
            ax.barh(yy, ke, left=z + lo, color=C_KEEP, height=0.68, lw=0)
            lab = "excluded" if c == EXCL else f"{t['flagged']:,} ({100 * t['flagged'] / t['n']:.0f}%)"
            ax.text(101.5, yy, lab, fontsize=5.8, va="center", ha="left",
                    color="#555555" if c == EXCL else "#222222", clip_on=False)
        ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=6.2, length=2, pad=1.5)
        ax.set_title(f"{g.capitalize()} group", fontsize=6.8, fontweight="bold", pad=3)
        ax.set_xlabel("Training windows of the class (%)", fontsize=6.2, labelpad=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_linewidth(0.6); ax.spines["bottom"].set_linewidth(0.6)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(NAMES, fontsize=6.4)
    # right margin room for the flagged-count labels
    for ax in axes:
        pos = ax.get_position(); ax.set_position([pos.x0, pos.y0, pos.width * 0.80, pos.height])
    fig.legend(handles=[Patch(color=C_ZERO, label="vote = 0 (flagged)"),
                        Patch(color=C_LOW, label=f"0 < vote < {TAU:.2f} (flagged)"),
                        Patch(color=C_KEEP, label=f"vote ≥ {TAU:.2f} (not flagged)"),
                        Patch(facecolor="#bdbdbd", hatch="/////", edgecolor="white",
                              label="BCKG, excluded by rule")],
               loc="upper center", ncol=4, fontsize=6.0, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.4)
    for d in OUTS:
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(d, f"fig_gate.{ext}"), dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("-> fig_gate.png/.pdf (7.48 x 2.25 in, print size)")


if __name__ == "__main__":
    main()
