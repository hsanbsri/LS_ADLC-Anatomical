"""
fig3_backbone.py
================
Fig. 3: the shared CNN-ResBiLSTM-Attention backbone, drawn as a horizontal
flow at final print size (7.48 in wide; font sizes are printed sizes).

Every block is read from src/model.py and configs/config.yaml:
stem Conv1d(k=7, stride 2) + BN + GELU (64 ch); four residual blocks
(64/128/128/256 ch, stride 2, k=7; Conv-BN-GELU, dropout, Conv-BN, 1x1
Conv-BN shortcut when the shape changes, GELU after the sum); BiLSTM
(128 units per direction); attention pooling concatenated with global
max-pooling over time; an MLP over the DWT sub-band features
(128 -> 64, GELU); concatenation -> latent projection z (256-d, GELU) ->
head (dropout 0.3, linear) -> six-class softmax.

Writes figures/output/fig1_backbone.{png,pdf}.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = [os.path.join(HERE, "output")]
plt.rcParams["font.family"] = "DejaVu Sans"
FW, FH = 7.48, 2.62
INK = "#222222"
C_IN, C_CNN, C_SEQ, C_POOL, C_DWT, C_LAT, C_OUT = ("#E8EEF7", "#EEF4FB", "#EEF4FB", "#FFF4DD",
                                                    "#E8EEF7", "#F3E9F7", "#EDEDED")
RED = "#a31515"


def box(ax, x, y, w, h, title, sub="", fc="#ffffff", ec="#555555", lw=0.8, tsize=6.6, ssize=6.0,
        tcol=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, fontsize=tsize, fontweight="bold", ha="center",
                va="center", color=tcol, zorder=5)
        ax.text(x + w / 2, y + h * 0.3, sub, fontsize=ssize, ha="center", va="center", color=INK,
                zorder=5, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, fontsize=tsize, fontweight="bold", ha="center",
                va="center", color=tcol, zorder=5, linespacing=1.15)
    return (x, y, w, h)


def arr(ax, a, b, c=INK, lw=0.8, cs="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, lw=lw, color=c,
                                 connectionstyle=cs, shrinkA=0, shrinkB=0, zorder=4))


def right(b, dy=0.0):
    return (b[0] + b[2], b[1] + b[3] / 2 + dy)


def left(b, dy=0.0):
    return (b[0], b[1] + b[3] / 2 + dy)


def main():
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")

    yT, h = 1.62, 0.62                      # main (signal) row
    inp = box(ax, 0.04, yT, 0.74, h, "EEG window", "C ch × 1,250\nsamples", fc=C_IN)
    stem = box(ax, 0.9, yT, 0.86, h, "Conv stem", "Conv1D k=7, s=2\nBN, GELU · 64 ch", fc=C_CNN)
    res = box(ax, 1.88, yT, 0.98, h, "Residual block ×4", "64/128/128/256 ch\nstride 2 each", fc=C_CNN,
              ec="#2a5a8c", lw=1.0)
    lstm = box(ax, 2.98, yT, 0.76, h, "BiLSTM", "128 units\nper direction", fc=C_SEQ)
    att = box(ax, 3.88, yT + 0.36, 0.98, 0.34, "Attention pooling", fc=C_POOL, tsize=6.0)
    mp = box(ax, 3.88, yT - 0.08, 0.98, 0.34, "Global max-pool", fc=C_POOL, tsize=6.0)
    cat = box(ax, 4.98, yT - 0.08, 0.46, 0.78, "Concat", fc="#F6F6F6", tsize=6.2)
    lat = box(ax, 5.56, yT, 0.86, h, "Latent z", "Linear, GELU\n256-d", fc=C_LAT, ec=RED, lw=1.0, tcol=RED)
    out = box(ax, 6.54, yT, 0.9, h, "Head", "Dropout 0.3, linear,\nsoftmax (6 classes)", fc=C_OUT, ssize=5.8)

    arr(ax, right(inp), left(stem)); arr(ax, right(stem), left(res)); arr(ax, right(res), left(lstm))
    arr(ax, right(lstm, 0.12), left(att), cs="arc3,rad=-0.15")
    arr(ax, right(lstm, -0.12), left(mp), cs="arc3,rad=0.15")
    arr(ax, right(att), (cat[0], att[1] + att[3] / 2))
    arr(ax, right(mp), (cat[0], mp[1] + mp[3] / 2))
    arr(ax, (cat[0] + cat[2], yT + h / 2), left(lat)); arr(ax, right(lat), left(out))

    # DWT branch
    yB = 0.62
    dwt = box(ax, 2.98, yB, 1.18, 0.5, "DWT sub-band features", "db4, level 6", fc=C_DWT, tsize=6.0)
    mlp = box(ax, 4.26, yB, 0.72, 0.5, "MLP", "128 → 64, GELU", fc=C_DWT, ssize=5.8)
    arr(ax, right(dwt), left(mlp))
    arr(ax, right(mlp), (cat[0] + cat[2] / 2, cat[1]), cs="arc3,rad=0.3")

    # residual-block inset: main path with the shortcut drawn as an arc above it
    ix, iy, iw, ih = 0.04, 0.12, 2.82, 1.2
    ax.add_patch(FancyBboxPatch((ix, iy), iw, ih, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="white", ec="#2a5a8c", lw=0.7, ls=(0, (3, 1.5)), zorder=2))
    ax.text(ix + 0.08, iy + ih - 0.13, "Residual block", fontsize=6.4, fontweight="bold",
            color="#2a5a8c", ha="left", va="center", zorder=5)
    by, bh = iy + 0.2, 0.36
    b1 = box(ax, ix + 0.2, by, 0.9, bh, "Conv1D, BN, GELU", tsize=5.9)
    b2 = box(ax, ix + 1.2, by, 0.5, bh, "Dropout", tsize=5.9)
    b3 = box(ax, ix + 1.8, by, 0.62, bh, "Conv1D, BN", tsize=5.9)
    sm = box(ax, ix + 2.5, by, 0.26, bh, "+", tsize=8.0)
    arr(ax, (ix + 0.06, by + bh / 2), left(b1))
    arr(ax, right(b1), left(b2)); arr(ax, right(b2), left(b3)); arr(ax, right(b3), left(sm))
    top = by + bh + 0.26
    ax.plot([ix + 0.12, ix + 0.12, sm[0] + sm[2] / 2], [by + bh / 2, top, top], color="#2a5a8c",
            lw=0.7, ls=(0, (2, 1)), zorder=4)
    arr(ax, (sm[0] + sm[2] / 2, top), (sm[0] + sm[2] / 2, by + bh), c="#2a5a8c", lw=0.7)
    ax.text(ix + 1.3, top + 0.08, "shortcut (1×1 Conv1D + BN when the shape changes)", fontsize=5.8,
            ha="center", va="bottom", color="#2a5a8c", zorder=5)
    ax.text(sm[0] + sm[2], by - 0.04, "then GELU", fontsize=5.8, ha="right", va="top", zorder=5)
    ax.plot([res[0] + 0.15, ix + iw - 0.4], [res[1], iy + ih], color="#2a5a8c", lw=0.5,
            ls=(0, (2, 1.5)), zorder=1)

    ax.text(5.0, 0.3, "C: 22 (full montage) or 6–8 (per anatomical group).",
            fontsize=6.0, style="italic", color="#555555", ha="left", va="center")

    for d in OUTS:
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(d, f"fig1_backbone.{ext}"), dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("-> fig1_backbone.png/.pdf (7.48 x 2.62 in, print size)")


if __name__ == "__main__":
    main()
