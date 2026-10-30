"""capacity_matched_single.py: three-group ensemble vs. single full-montage backbones (Table 4).

TUEV, seeds 42-66. Contrasts, each with a paired t-test over seeds,
Benjamini-Hochberg correction (m = 4 metrics) and the 95% CI of the mean
paired difference:
  pG - pE1wide   ensemble vs. the capacity-matched single backbone
                 (pE1wide: 5,473,414 parameters vs. 5,462,482 for pG)
  pE1wide - pE1  effect of capacity alone
  pG - pE1       ensemble vs. the default single backbone (1,866,822 parameters)
pG is fused by Eq. 1.

Input:  results/pG_*_evalprob.npy, results/pE1_strict_*_evalprob.npy,
        results/pE1wide_*_evalprob.npy, data/win_eval.h5
Output: analysis/output/capacity_matched_single_results.json
Usage:  python analysis/capacity_matched_single.py
"""
import numpy as np
import h5py
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
import json, os

R = os.environ.get("LSADLC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
G = ["temporal", "central", "parasagittal"]; SEEDS = range(42, 67); KEYS = ["balAcc", "kappa", "wF1", "SPSW"]
y = h5py.File(f"{R}/data/win_eval.h5", "r")["y"][:].astype(int)
met = lambda pr: [100 * balanced_accuracy_score(y, pr), 100 * cohen_kappa_score(y, pr),
                  100 * f1_score(y, pr, average="weighted"), 100 * f1_score(y, pr, labels=[0], average=None)[0]]
def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty_like(p); m = len(p); prev = 1.0
    for r, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * m / (m - r)); q[i] = prev
    return q
pG = np.array([met(np.mean([np.log(np.clip(np.load(f"{R}/results/pG_{g}_armA_seed{s}_evalprob.npy"), 1e-12, 1))
                             for g in G], 0).argmax(1)) for s in SEEDS])
pE1 = np.array([met(np.load(f"{R}/results/pE1_strict_armA_seed{s}_evalprob.npy").argmax(1)) for s in SEEDS])
pW = np.array([met(np.load(f"{R}/results/pE1wide_armA_seed{s}_evalprob.npy").argmax(1)) for s in SEEDS])
out = {}
for name, A, B in (("pG - pE1wide", pG, pW), ("pE1wide - pE1", pW, pE1), ("pG - pE1", pG, pE1)):
    D = A - B; p = [stats.ttest_rel(A[:, i], B[:, i]).pvalue for i in range(4)]; q = bh(p); res = {}
    print(f"\n{name}")
    for i, k in enumerate(KEYS):
        h = stats.t.ppf(0.975, 24) * D[:, i].std(ddof=1) / 5
        print(f"  {k:6s} {A[:, i].mean():6.2f} vs {B[:, i].mean():6.2f}  delta {D[:, i].mean():+6.2f} "
              f"[{D[:, i].mean() - h:+.2f}, {D[:, i].mean() + h:+.2f}]  p={p[i]:.3g} q={q[i]:.3g} fav {(D[:, i] > 0).sum()}/25")
        res[k] = dict(a=float(A[:, i].mean()), b=float(B[:, i].mean()), delta=float(D[:, i].mean()),
                      ci=[float(D[:, i].mean() - h), float(D[:, i].mean() + h)], p=float(p[i]), q=float(q[i]), fav=int((D[:, i] > 0).sum()))
    out[name] = res
json.dump(out, open(os.path.join(OUT, "capacity_matched_single_results.json"), "w"), indent=1)
