"""
fig5_replication.py
===================
Fig. 5: forest-style plot of the pLS-vs-pG effect (balanced accuracy,
SPSW-F1) across the two independent 25-seed batches. Numbers are taken
directly from Tables 7 and 8 (analysis/paired_ttest_table.py is the
source of record; the literal values are reproduced here so
this script has no runtime dependency on that file's exact schema).
Writes figures/output/fig5_replication.{png,pdf}.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# metric -> [(batch_label, delta, ci_lo, ci_hi, q), ...]
DATA = {
    "Balanced accuracy (pLS - pG, pp)": [
        ("Batch 1 (seeds 42-66)", 2.57, 1.08, 4.06, 0.0065),
        ("Batch 2 (seeds 100-124)", 2.53, 0.68, 4.37, 0.0185),
    ],
    "SPSW-F1 (pLS - pG, pp)": [
        ("Batch 1 (seeds 42-66)", 9.58, 2.57, 16.58, 0.0188),
        ("Batch 2 (seeds 100-124)", 9.42, 2.65, 16.19, 0.0185),
    ],
}


def main():
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex=False)

    for ax, (metric, rows) in zip(axes, DATA.items()):
        ys = list(range(len(rows)))[::-1]
        for y, (label, delta, lo, hi, q) in zip(ys, rows):
            err_lo = delta - lo
            err_hi = hi - delta
            ax.errorbar(delta, y, xerr=[[err_lo], [err_hi]], fmt="o",
                         color="#2a5a8c", ecolor="#2a5a8c", capsize=4,
                         markersize=6, linewidth=1.6)
            star = "***" if q <= 0.001 else "**" if q <= 0.01 else "*" if q <= 0.05 else "ns"   # notation of Table 4
            ax.text(hi + 0.6, y, f"q={q:.4f}{star}", va="center", fontsize=8.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.set_xlabel(metric, fontsize=9.5)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        xmax = max(r[3] for r in rows) + 3.5
        xmin = min(0, min(r[2] for r in rows) - 1.0)
        ax.set_xlim(xmin, xmax)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)

    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, "fig5_replication.png")
    pdf_path = os.path.join(OUT_DIR, "fig5_replication.pdf")
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  wrote {png_path}")
    print(f"  wrote {pdf_path}")


if __name__ == "__main__":
    main()
