"""
fig6_per_class.py
=================
Fig. 6: bar-chart visualization of Table 7 (per-class F1, batch 2,
seeds 100-124). Two panels:
  A. Fused-ensemble F1 per class, pG vs. pLS (all 6 classes).
  B. SPSW-F1 per anatomical group (temporal/central/parasagittal), pG vs.
     pLS -- visualizes the asymmetric per-branch effect discussed in
     Section 4 (gain concentrated in the parasagittal branch).

All values are copied directly from Table 7 (analysis/per_class_f1.py
is the source of record); this script does not recompute them. Drawn at final
print size (7.48 in wide, fonts >= 6 pt) with no in-figure footnote.
Writes figures/output/fig6_per_class_bar.{png,pdf}.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"]
PG_FUSED = [18.84, 88.64, 58.82, 50.71, 50.17, 92.89]
PLS_FUSED = [28.25, 88.90, 60.18, 53.55, 47.41, 92.62]

GROUPS = ["Temporal", "Central", "Parasagittal"]
PG_SPSW_GROUP = [16.43, 15.30, 24.89]
PLS_SPSW_GROUP = [13.54, 15.40, 39.45]

PG_COLOR = "#5b7fa6"
PLS_COLOR = "#d98a2b"


def panel_a(ax):
    x = np.arange(len(CLASSES))
    w = 0.36
    b1 = ax.bar(x - w / 2, PG_FUSED, width=w, color=PG_COLOR, label="pG")
    b2 = ax.bar(x + w / 2, PLS_FUSED, width=w, color=PLS_COLOR, label="pLS")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 1.0, f"{h:.2f}",
                    ha="center", va="bottom", rotation=90, fontsize=5.6)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=7)
    ax.set_ylabel("Fused-ensemble F1 (%)", fontsize=7)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("A. Per-class F1, fused ensemble", fontsize=7.5, fontweight="bold")
    # Opaque white backing is required here (unlike panel B): this panel's
    # bars are tall enough (GPED/BCKG ~88-93) that almost any legend
    # position sits in front of a bar, and a transparent (frameon=False)
    # legend over a colored bar becomes unreadable -- that was the bug.
    ax.legend(fontsize=6.5, frameon=True, facecolor="white", framealpha=0.95,
              edgecolor="#cccccc", loc="upper center", ncol=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Highlight the SPSW bars (headline result).
    ax.annotate("+9.42", xy=(0, PLS_FUSED[0] + 13), ha="center",
                fontsize=7, fontweight="bold", color="#8a4b00")


def panel_b(ax):
    x = np.arange(len(GROUPS))
    w = 0.36
    b1 = ax.bar(x - w / 2, PG_SPSW_GROUP, width=w, color=PG_COLOR, label="pG")
    b2 = ax.bar(x + w / 2, PLS_SPSW_GROUP, width=w, color=PLS_COLOR, label="pLS")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.8, f"{h:.2f}",
                    ha="center", fontsize=5.8)

    # Each delta label floats just above its own bar pair (not a fixed
    # height), so none of the three ever lands inside the upper-left
    # legend box regardless of that group's bar heights.
    deltas = [p - g for p, g in zip(PLS_SPSW_GROUP, PG_SPSW_GROUP)]
    for xi, g, p, d in zip(x, PG_SPSW_GROUP, PLS_SPSW_GROUP, deltas):
        sign = "+" if d >= 0 else ""
        label_y = max(g, p) + 3.0
        ax.text(xi, label_y, f"{sign}{d:.2f}", ha="center", fontsize=6.6,
                fontweight="bold", color="#8a4b00" if d > 1 else "#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, fontsize=7)
    ax.set_ylabel("Group-level SPSW-F1 (%)", fontsize=7)
    ax.set_ylim(0, 50)
    ax.set_title("B. SPSW-F1 per anatomical group model", fontsize=7.5, fontweight="bold")
    ax.legend(fontsize=6.5, frameon=True, facecolor="white", framealpha=0.95,
              edgecolor="#cccccc", loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 2.9))
    panel_a(axes[0])
    panel_b(axes[1])

    for ax in axes:
        ax.tick_params(labelsize=6.5, length=2)
    plt.tight_layout(w_pad=2.0)
    png_path = os.path.join(OUT_DIR, "fig6_per_class_bar.png")
    pdf_path = os.path.join(OUT_DIR, "fig6_per_class_bar.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path}")
    print(f"  wrote {pdf_path}")


if __name__ == "__main__":
    main()
