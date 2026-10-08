import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

import os
OUT = os.path.dirname(os.path.abspath(__file__))  # this figures/ directory
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(7.2, 7.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12.6)
ax.axis("off")

def box(x, y, w, h, text, fc="white", ec="black", lw=1.2, ls="solid", fs=8.5, bold=False, linespacing=1.6):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
             fontweight="bold" if bold else "normal", linespacing=linespacing, zorder=3)

def arrow(x1, y1, x2, y2, ls="solid", color="black", rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                         linewidth=1.1, color=color, linestyle=ls, zorder=1,
                         connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)

# Input
box(2.8, 11.2, 4.4, 1.0, "22-channel bipolar TCP (ACNS) montage\n(single backbone input, no annotation info)", bold=True)

group_x = [0.6, 3.6, 6.6]
group_w = 2.3
labels = ["Temporal\n(ch 0–7)", "Central\n(ch 8–13)", "Parasagittal\n(ch 14–21)"]
for gx, lab in zip(group_x, labels):
    arrow(5.0, 11.2, gx + group_w / 2, 9.9)
    box(gx, 9.4, group_w, 0.9, lab, fc="#eef2f7")

backbone_text = ("CNN-Res-BiLSTM-\nAttention backbone\n(independently trained)\n\n"
                 "4 ResBlocks → BiLSTM →\nAttn+max-pool\n⊕ group's DWT subbands")
for gx in group_x:
    arrow(gx + group_w / 2, 9.4, gx + group_w / 2, 8.75)
    box(gx, 6.5, group_w, 2.25, backbone_text, fc="#f7f7f7", fs=6.9, linespacing=1.7)

# LS-ADLC dashed branch (training-time only, pLS arm)
ladlc_x, ladlc_y, ladlc_w, ladlc_h = 3.25, 4.5, 3.5, 1.5
box(ladlc_x, ladlc_y, ladlc_w, ladlc_h,
    "LS-ADLC (training-time only, pLS arm)\nOOF k-NN confidence gate (k=15)\n→ 10 latent synthetic copies,\nα=0.5 (excl. BCKG)",
    fc="white", ec="#555555", ls=(0, (5, 3)), fs=7.3, linespacing=1.6)

arrow(group_x[0] + group_w / 2, 6.5, ladlc_x, ladlc_y + ladlc_h * 0.65, ls=(0, (5, 3)), color="#777777", rad=-0.15)
arrow(group_x[1] + group_w / 2, 6.5, group_x[1] + group_w / 2, ladlc_y + ladlc_h, ls=(0, (5, 3)), color="#777777")
arrow(group_x[2] + group_w / 2, 6.5, ladlc_x + ladlc_w, ladlc_y + ladlc_h * 0.65, ls=(0, (5, 3)), color="#777777", rad=0.15)

ax.text(5.0, 4.05, "present only in the pLS arm; absent in pG (control) —\n"
                    "the only configuration difference between the two arms",
        ha="center", va="top", fontsize=7, style="italic", color="#444444")

for gx in group_x:
    arrow(gx + group_w / 2, 6.5, gx + group_w / 2, 3.35)
    box(gx + 0.45, 2.85, group_w - 0.9, 0.5, "6-class probs", fc="#eef2f7", fs=7.3)

for gx in group_x:
    arrow(gx + group_w / 2, 2.85, 5.0, 2.15)
box(2.6, 1.35, 4.8, 0.8, "Evaluation-time fusion (both arms, identical):\nmean log-probability (product rule, Kittler et al. 1998)", fs=7.4)

arrow(5.0, 1.35, 5.0, 0.8)
box(3.1, 0.05, 3.8, 0.7, "6-class prediction\n(SPSW / GPED / PLED / EYEM / ARTF / BCKG)", bold=True, fs=7.6)

ax.plot([7.4, 7.9], [12.0, 12.0], color="black", lw=1.1)
ax.text(8.0, 12.0, "present in both arms (pG, pLS)", va="center", fontsize=6.8)
ax.plot([7.4, 7.9], [11.65, 11.65], color="#555555", lw=1.0, linestyle=(0, (5, 3)))
ax.text(8.0, 11.65, "present only in the pLS arm", va="center", fontsize=6.8)

ax.text(5.0, 12.35, "Figure 1. Three-branch anatomically-partitioned ensemble with per-group LS-ADLC",
        ha="center", fontsize=10.5, fontweight="bold")

fig.tight_layout()
fig.savefig(f"{OUT}/fig1_architecture.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig1_architecture.png", dpi=300, bbox_inches="tight")
print("saved fig1")
