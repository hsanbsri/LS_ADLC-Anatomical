"""precision_recall.py: per-class precision and recall, pG vs. pLS, TUEV batch 2 (Section 3.4).

Fused ensemble (Eq. 1) per class, and SPSW precision/recall of each anatomical
group model; means over seeds 100-124 (zero_division = 0).

Input:  results/{pG,pLS}_<group>_armA_seed<s>_evalprob.npy, data/win_eval.h5
Usage:  python analysis/precision_recall.py
"""
import os
import numpy as np
import h5py
from sklearn.metrics import precision_recall_fscore_support as prf

R = os.environ.get("LSADLC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
G = ["temporal", "central", "parasagittal"]
N = ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"]
SEEDS = range(100, 125)
y = h5py.File(f"{R}/data/win_eval.h5", "r")["y"][:].astype(int)


def fused(arm, s):
    return np.mean([np.log(np.clip(np.load(f"{R}/results/{arm}_{g}_armA_seed{s}_evalprob.npy"), 1e-12, 1))
                    for g in G], 0).argmax(1)


def single(arm, g, s):
    return np.load(f"{R}/results/{arm}_{g}_armA_seed{s}_evalprob.npy").argmax(1)


def avg(fn):
    P, Rc = [], []
    for s in SEEDS:
        p, r, _, _ = prf(y, fn(s), labels=range(6), zero_division=0); P.append(p); Rc.append(r)
    return 100 * np.mean(P, 0), 100 * np.mean(Rc, 0)


print(f"BCKG share of evaluation windows: {100 * np.mean(y == 5):.1f}%")
a, b = avg(lambda s: fused("pG", s)), avg(lambda s: fused("pLS", s))
for i, n in enumerate(N):
    print(f"fused {n}: precision {a[0][i]:.1f} -> {b[0][i]:.1f} | recall {a[1][i]:.1f} -> {b[1][i]:.1f}")
for g in G:
    a, b = avg(lambda s: single("pG", g, s)), avg(lambda s: single("pLS", g, s))
    print(f"{g} SPSW: precision {a[0][0]:.1f} -> {b[0][0]:.1f} | recall {a[1][0]:.1f} -> {b[1][0]:.1f}")
