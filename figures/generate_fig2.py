import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
OUT = os.path.dirname(os.path.abspath(__file__))  # this figures/ directory

# Verified numbers (Tables 3-4 of the manuscript; independently re-derived, matched to reported precision)
rows = [
    ("Balanced accuracy (Δ pLS − pG, pp)", [
        ("Batch 1 (seeds 42–66, exploratory)", 2.57, 1.08, 4.06),
        ("Batch 2 (seeds 100–124, confirmatory)", 2.53, 0.68, 4.37),
    ]),
    ("SPSW-F1 (Δ pLS − pG, pp)", [
        ("Batch 1 (seeds 42–66, exploratory)", 9.58, 2.57, 16.58),
        ("Batch 2 (seeds 100–124, confirmatory)", 9.42, 2.65, 16.19),
    ]),
]

fig, axes = plt.subplots(2, 1, figsize=(6.5, 3.6), sharex=False)
plt.rcParams.update({"font.size": 9, "font.family": "serif"})

for ax, (title, pts) in zip(axes, rows):
    ypos = list(range(len(pts)))[::-1]
    for y, (label, d, lo, hi) in zip(ypos, pts):
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color="black",
                    ecolor="black", elinewidth=1.2, capsize=4, markersize=5)
        ax.text(hi + 0.6, y, f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=8)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels([p[0] for p in pts], fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlim(-2, 22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.6, len(pts) - 0.4)

axes[-1].set_xlabel("Point difference, pLS − pG (percentage points), 95% CI", fontsize=9)
fig.suptitle("Figure 2. Replication consistency across two independent 25-seed batches", fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_replication.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig2_replication.png", dpi=300, bbox_inches="tight")
print("saved fig2")
