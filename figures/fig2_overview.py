"""
fig2_overview.py
================
Fig. 2 (overview), drawn at final print size (7.48 in, double-column
width; data coordinates are inches, so every font size below
is its printed size; smallest text 6 pt).

(a) Inference pipeline shared by both arms: 22-channel bipolar TCP window ->
preprocessing -> anatomical partition (temporal, central, parasagittal) -> one backbone
per group -> product-rule fusion -> six-class prediction.
(b) The pLS arm's training-time LS-ADLC as its two real stages: Stage A once
before training (out-of-fold pG embeddings -> k-NN vote -> gate); Stage B
every epoch at the latent z of each branch.

Values from the code/config behind the reported runs: notch 60/120 Hz,
band-pass 0.5-80 Hz, 250 Hz, 5-s windows, DWT db4 level 6; per-group
parameter counts from saved checkpoints; OOF 5-fold GroupKFold with a
15-epoch encoder per fold; k = 15, w = 1/(d + eps); threshold 0.80; BCKG
excluded; alpha = 0.5; 4-5 copies per epoch; synthetic rows weighted 1.0 in
the cross-entropy loss. EEG traces and the 2-D scatter are illustrative.

Writes figures/output/fig1_overview.{png,pdf}.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = [os.path.join(HERE, "output")]
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"

FW, FH = 7.48, 4.55
INK, GREY = "#222222", "#555555"
RED, RED_FILL, RED_TXT = "#a31515", "#FCE9E9", "#7a0000"
GRP = [("Temporal", "#D3E3F5", "#2a5a8c", "8 ch"),
       ("Central", "#FBE3CB", "#b05a00", "6 ch"),
       ("Parasagittal", "#D9EDCB", "#2e7d32", "8 ch")]
EDGE = [g[2] for g in GRP]


def box(ax, x, y, w, h, fc, ec, lw=0.8, r=0.04, ls="solid", z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arr(ax, a, b, c=INK, lw=0.8, ls="solid", ms=6, cs="arc3,rad=0", z=4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=c,
                                 linestyle=ls, connectionstyle=cs, shrinkA=0, shrinkB=0, zorder=z))


def t(ax, x, y, s, size=6.2, w="normal", c=INK, ha="center", va="center", st="normal", z=6):
    ax.text(x, y, s, fontsize=size, fontweight=w, color=c, ha=ha, va=va, style=st, zorder=z,
            linespacing=1.15)


POS = {"FP1": (-0.27, 0.93), "FP2": (0.27, 0.93), "F7": (-0.78, 0.58), "F3": (-0.40, 0.58),
       "F4": (0.40, 0.58), "F8": (0.78, 0.58), "A1": (-1.02, 0.0), "T3": (-0.80, 0.0),
       "C3": (-0.40, 0.0), "CZ": (0.0, 0.0), "C4": (0.40, 0.0), "T4": (0.80, 0.0),
       "A2": (1.02, 0.0), "T5": (-0.78, -0.58), "P3": (-0.40, -0.58), "P4": (0.40, -0.58),
       "T6": (0.78, -0.58), "O1": (-0.27, -0.93), "O2": (0.27, -0.93)}
MONTAGE = [("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"), ("FP2", "F8"), ("F8", "T4"),
           ("T4", "T6"), ("T6", "O2"), ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"),
           ("C4", "T4"), ("T4", "A2"), ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
           ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2")]


def inset(fig, x, y, w, h):
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])


def head(fig, x, y, w, h):
    ax = inset(fig, x, y, w, h)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.15, 1.22); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.05, fill=False, lw=0.8, ec=INK))
    ax.add_patch(Polygon([(-0.09, 1.04), (0, 1.17), (0.09, 1.04)], fc="white", ec=INK, lw=0.7, zorder=5))
    for s in (-1, 1):
        ax.add_patch(Wedge((s * 1.05, 0), 0.10, 250 if s < 0 else 70, 290 if s < 0 else 110,
                           fc="white", ec=INK, lw=0.6, zorder=5))
    for i, (a, b) in enumerate(MONTAGE):
        g = 0 if i <= 7 else (1 if i <= 13 else 2)
        (xa, ya), (xb, yb) = POS[a], POS[b]
        ax.plot([xa, xb], [ya, yb], color=EDGE[g], lw=1.9, solid_capstyle="round", zorder=3)
    for px, py in POS.values():
        ax.scatter([px], [py], s=2.5, c=INK, zorder=6)


def eeg(fig, x, y, w, h):
    ax = inset(fig, x, y, w, h)
    rs = np.random.RandomState(4)
    tt = np.linspace(0, 5, 400)
    for i in range(22):
        g = 0 if i <= 7 else (1 if i <= 13 else 2)
        sig = 0.30 * np.sin(2 * np.pi * (1.2 + 0.3 * rs.rand()) * tt + rs.rand() * 6)
        sig += 0.12 * np.sin(2 * np.pi * 9 * tt + rs.rand() * 6) + 0.08 * rs.randn(400).cumsum() / 10
        if i == 3:
            sig += 1.0 * np.exp(-((tt - 2.6) / 0.06) ** 2)
        ax.plot(tt, sig - i, color="#3a3a3a", lw=0.35)
        ax.add_patch(plt.Rectangle((-0.55, -i - 0.42), 0.32, 0.84, color=EDGE[g], lw=0))
    ax.set_xlim(-0.6, 5.05); ax.set_ylim(-22, 1.1); ax.axis("off")


def main():
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")

    # ===================== (a) inference pipeline =====================
    A0, A1 = 2.34, 4.5
    box(ax, 0.04, A0, FW - 0.08, A1 - A0, "#FAFAF8", "#c8c8c8", lw=0.6, r=0.06, z=0)
    t(ax, 0.12, A1 - 0.13, "(a)", 8.5, "bold", ha="left")
    t(ax, 0.42, A1 - 0.13, "Inference pipeline (identical in the pG and pLS arms)", 7.2, "bold", ha="left")

    yc = 3.33                                       # vertical centre of the lanes
    eeg(fig, 0.12, 2.67, 0.86, 1.4)
    t(ax, 0.55, 2.52, "22-ch TCP window\n5 s · 250 Hz", 6.0)
    arr(ax, (1.02, yc), (1.14, yc))

    box(ax, 1.16, 2.78, 0.86, 1.1, "#F1F1F1", "#8a8a8a", lw=0.7)
    t(ax, 1.59, 3.72, "Preprocessing", 6.6, "bold")
    t(ax, 1.59, 3.2, "Notch 60/120 Hz\nBand-pass\n0.5–80 Hz\nDWT db4, level 6", 6.0)
    arr(ax, (2.04, yc), (2.16, yc))

    head(fig, 2.14, 2.7, 1.05, 1.15)
    t(ax, 2.665, 3.97, "Anatomical partition", 6.6, "bold")

    lanes = [3.9, 3.33, 2.76]
    bx, bw, bh = 3.55, 1.22, 0.44
    cx, cw = 4.95, 0.5
    fx, fw, fy, fh = 5.62, 0.86, 3.03, 0.6
    for (name, fc, ec, nch), y in zip(GRP, lanes):
        arr(ax, (3.2, yc + (y - yc) * 0.45), (bx, y), c=ec, lw=1.0)
        box(ax, bx, y - bh / 2, bw, bh, fc, ec, lw=0.9)
        t(ax, bx + bw / 2, y + 0.08, f"{name} backbone", 6.4, "bold")
        t(ax, bx + bw / 2, y - 0.11, f"{nch} · 1.82 M params", 6.0)
        ax.add_patch(Circle((bx + bw - 0.07, y - bh / 2), 0.035, fc=RED, ec="white", lw=0.4, zorder=7))
        arr(ax, (bx + bw, y), (cx, y))
        box(ax, cx, y - 0.13, cw, 0.26, "white", ec, lw=0.8, r=0.03)
        t(ax, cx + cw / 2, y, rf"$\mathbf{{p}}_{{{name[0]}}}$", 7.0)
        arr(ax, (cx + cw, y), (fx, fy + fh / 2 + (y - yc) * 0.35))
    box(ax, fx, fy, fw, fh, "#E8DFF3", "#5b3a86", lw=0.9)
    t(ax, fx + fw / 2, fy + 0.41, "Product rule", 6.6, "bold", c="#3a2460")
    t(ax, fx + fw / 2, fy + 0.18, r"$\frac{1}{3}\sum_g \log\mathbf{p}_g$", 7.0, c="#3a2460")
    arr(ax, (fx + fw, yc), (6.62, yc), c="#3a2460")
    box(ax, 6.64, 2.78, 0.74, 1.1, "#EDEDED", "#555555", lw=0.8)
    t(ax, 7.01, 3.68, "Prediction", 6.6, "bold")
    t(ax, 7.01, 3.2, "SPSW · GPED\nPLED · EYEM\nARTF · BCKG", 6.0)
    t(ax, 1.2, 2.45, "Both arms: 5,462,482 parameters", 6.0, st="italic", c=GREY, ha="left")
    ax.add_patch(Circle((5.62, 2.45), 0.035, fc=RED, ec="white", lw=0.4, zorder=7))
    t(ax, 5.70, 2.45, "latent z (Stage B acts here)", 6.0, st="italic", c=RED_TXT, ha="left")

    # ===================== (b) training-time LS-ADLC =====================
    B0, B1 = 0.04, 2.22
    box(ax, 0.04, B0, FW - 0.08, B1 - B0, "white", RED, lw=0.9, r=0.06, z=0)
    t(ax, 0.12, B1 - 0.13, "(b)", 8.5, "bold", ha="left")
    t(ax, 0.42, B1 - 0.13, "LS-ADLC: pLS arm only, run in each branch, training time only",
      7.2, "bold", c=RED_TXT, ha="left")

    def stage(x0, w, title):
        box(ax, x0, 0.14, w, 1.78, RED_FILL, RED, lw=0.7, r=0.05, ls=(0, (3, 1.5)))
        t(ax, x0 + 0.07, 1.78, title, 6.6, "bold", c=RED_TXT, ha="left")

    def step(x, w, head_txt, body):
        box(ax, x, 0.24, w, 1.38, "white", "#c98b8b", lw=0.6, r=0.04)
        t(ax, x + w / 2, 1.44, head_txt, 6.5, "bold")
        if body:
            t(ax, x + w / 2, 0.86, body, 6.0)

    SAx, SAw = 0.12, 3.5
    stage(SAx, SAw, "Stage A · once, before training")
    sw, sg = 1.04, 0.1
    xs = [SAx + 0.08 + i * (sw + sg) for i in range(3)]
    step(xs[0], sw, "Out-of-fold z", "5-fold GroupKFold\nover subjects;\n15-epoch pG\nencoder per fold")
    step(xs[1], sw, "k-NN vote", "k = 15\n" r"$w=1/(d+\varepsilon)$" "\nshare of same-\nlabel neighbours")
    step(xs[2], sw, "Gate", "flag if\nvote < 0.80\nand label ≠ BCKG")
    for i in range(2):
        arr(ax, (xs[i] + sw + 0.005, 0.93), (xs[i + 1] - 0.005, 0.93), c=RED, lw=0.9)

    SBx, SBw = 3.86, 3.5
    arr(ax, (SAx + SAw + 0.01, 1.03), (SBx - 0.01, 1.03), c=RED, lw=1.2, ms=8)
    stage(SBx, SBw, "Stage B · every epoch, at z")
    xb = [SBx + 0.08 + i * (sw + sg) for i in range(3)]
    step(xb[0], sw, "Flagged z", "")
    ins = inset(fig, xb[0] + 0.06, 0.3, sw - 0.12, 0.98)
    z0 = np.array([0.0, 0.0])
    for p in np.array([[-0.72, 0.45], [0.68, 0.55], [-0.5, -0.62], [0.78, -0.42], [0.06, 0.86]]):
        ins.add_patch(FancyArrowPatch(tuple(z0), tuple(p), arrowstyle="-|>", mutation_scale=4,
                                      lw=0.6, color="#c98a00", linestyle=(0, (2, 1)), shrinkA=4, shrinkB=2.5))
        ins.scatter([p[0]], [p[1]], s=11, c="#f1c232", edgecolor="#a67c00", lw=0.4, zorder=3)
    ins.scatter([0], [0], marker="*", s=42, c=RED, edgecolor="black", lw=0.4, zorder=4)
    ins.text(0, -1.05, "z → copies z′", fontsize=6.0, ha="center")
    ins.set_xlim(-1.05, 1.05); ins.set_ylim(-1.25, 1.05); ins.axis("off")
    step(xb[1], sw, "Latent noise", r"$z'=z+\alpha\sigma_c\varepsilon$" "\n" r"$\alpha=0.5$" "\n4–5 copies\nper epoch")
    step(xb[2], sw, "Same head", "real + synthetic\nrows, one\ncross-entropy\nloss (weight 1.0)")
    for i in range(2):
        arr(ax, (xb[i] + sw + 0.005, 0.93), (xb[i + 1] - 0.005, 0.93), c=RED, lw=0.9)

    # Stage B feeds the latent taps of panel (a)
    arr(ax, (5.25, 1.92), (4.74, 2.53), c=RED, lw=0.9, ls=(0, (3, 1.5)), ms=6, cs="arc3,rad=0.25")

    for d in OUTS:
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(d, f"fig1_overview.{ext}"), dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("-> fig1_overview.png/.pdf (7.48 x 4.55 in, print size)")


if __name__ == "__main__":
    main()
