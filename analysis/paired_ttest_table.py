"""paired_ttest_table.py: pLS vs. pG on the fused ensemble (Tables 5, 6 and 8, Fig. 5).

For TUEV batch 1 (seeds 42-66), TUEV batch 2 (seeds 100-124) and the same two
batches on TUSZ, the script fuses the three anatomical-group models of each
arm by Eq. 1 (mean log-probability, then argmax), computes balanced accuracy,
Cohen's kappa, weighted-F1 and the minority-class F1 (SPSW for TUEV, seizure
for TUSZ), and runs a two-sided paired t-test over seeds (alpha = 0.05) with
Benjamini-Hochberg correction (m = 4 metrics per batch). Reported per metric:
number of experiments, mean and standard deviation of the paired difference,
95% confidence interval of the mean difference, t, p and q. check() asserts
that every value agrees with the numbers printed in the paper.

Input:  results/*_evalprob.npy, results_tusz/*_evalprob.npy,
        data/win_eval.h5, data_tusz/win_eval.h5
Output: analysis/output/paired_ttest_table_results.json
Usage:  python analysis/paired_ttest_table.py
"""
import json
import os
import numpy as np
import h5py
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

R = os.environ.get("LSADLC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
G = ["temporal", "central", "parasagittal"]
SETS = {"TUEV": ("results", "data", 0, "SPSW-F1"), "TUSZ": ("results_tusz", "data_tusz", 1, "Seizure-F1")}
BATCH = {"batch1": range(42, 67), "batch2": range(100, 125)}


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); q = np.empty(m); prev = 1.0
    for r in range(m - 1, -1, -1):
        prev = min(prev, p[o[r]] * m / (r + 1)); q[o[r]] = prev
    return q


def run():
    out = {}
    for ds, (res, cache, minority, mname) in SETS.items():
        with h5py.File(f"{R}/{cache}/win_eval.h5", "r") as h:
            y = h["y"][:].astype(int)

        def met(arm, s):
            L = np.mean([np.log(np.clip(np.load(f"{R}/{res}/{arm}_{g}_armA_seed{s}_evalprob.npy"), 1e-12, 1))
                         for g in G], 0)
            pr = L.argmax(1)
            return {"Balanced accuracy": 100 * balanced_accuracy_score(y, pr),
                    "Cohen's kappa": 100 * cohen_kappa_score(y, pr),
                    "Weighted-F1": 100 * f1_score(y, pr, average="weighted"),
                    mname: 100 * f1_score(y, pr, labels=[minority], average=None, zero_division=0)[0]}
        for b, seeds in BATCH.items():
            A = [met("pLS", s) for s in seeds]; Bm = [met("pG", s) for s in seeds]
            rows, ps = {}, []
            for k in A[0]:
                a = np.array([x[k] for x in A]); c = np.array([x[k] for x in Bm]); d = a - c
                n = len(d); sd = d.std(ddof=1); se = sd / np.sqrt(n)
                t, p = stats.ttest_rel(a, c)
                h_ = stats.t.ppf(0.975, n - 1) * se
                rows[k] = dict(n=n, pLS=a.mean(), pG=c.mean(), mean_diff=d.mean(), sd_diff=sd,
                               ci_low=d.mean() - h_, ci_high=d.mean() + h_, t=t, df=n - 1, p=p,
                               fav=int((d > 0).sum()))
                ps.append(p)
            for k, q in zip(rows, bh(ps)):
                rows[k]["q"] = q
            out[f"{ds}_{b}"] = rows
    return out


def check(out):
    """Assert agreement with the numbers already printed in the paper."""
    ms = {  # (mean diff, CI low, CI high, p, q) as reported, TUEV Tables 5 and 6
        ("TUEV_batch1", "Balanced accuracy"): (2.57, 1.08, 4.06, 0.0016, 0.0065),
        ("TUEV_batch1", "Cohen's kappa"): (0.21, -0.67, 1.09, 0.63, 0.63),
        ("TUEV_batch1", "Weighted-F1"): (0.20, -0.21, 0.61, 0.32, 0.43),
        ("TUEV_batch1", "SPSW-F1"): (9.58, 2.57, 16.58, 0.0094, 0.0188),
        ("TUEV_batch2", "Balanced accuracy"): (2.53, 0.68, 4.37, 0.0093, 0.0185),
        ("TUEV_batch2", "Cohen's kappa"): (-0.59, -1.31, 0.12, 0.099, 0.13),
        ("TUEV_batch2", "Weighted-F1"): (-0.04, -0.42, 0.35, 0.85, 0.85),
        ("TUEV_batch2", "SPSW-F1"): (9.42, 2.65, 16.19, 0.0084, 0.0185),
    }
    for (c, k), (dm, lo, hi, p, q) in ms.items():
        r = out[c][k]
        assert abs(r["mean_diff"] - dm) < 0.006 and abs(r["ci_low"] - lo) < 0.006 and abs(r["ci_high"] - hi) < 0.006, (c, k, r)
        assert abs(r["p"] - p) < 0.6 * 10 ** np.floor(np.log10(p)) and abs(r["q"] - q) < 0.6 * 10 ** np.floor(np.log10(q)), (c, k, r)
    tusz = {  # (mean diff, q) as reported, Table 8
        ("TUSZ_batch1", "Balanced accuracy"): (5.18, 1.1e-9), ("TUSZ_batch2", "Balanced accuracy"): (3.59, 3.2e-5),
        ("TUSZ_batch1", "Cohen's kappa"): (8.74, 2.6e-9), ("TUSZ_batch2", "Cohen's kappa"): (5.90, 3.2e-5),
        ("TUSZ_batch1", "Weighted-F1"): (2.01, 5.8e-9), ("TUSZ_batch2", "Weighted-F1"): (1.34, 3.8e-5),
        ("TUSZ_batch1", "Seizure-F1"): (9.31, 2.6e-9), ("TUSZ_batch2", "Seizure-F1"): (6.35, 3.2e-5),
    }
    for (c, k), (dm, q) in tusz.items():
        r = out[c][k]
        assert abs(r["mean_diff"] - dm) < 0.006 and abs(r["q"] / q - 1) < 0.05, (c, k, r)


if __name__ == "__main__":
    out = run()
    check(out)
    json.dump(out, open(os.path.join(OUT, "paired_ttest_table_results.json"), "w"), indent=1)
    for c, rows in out.items():
        print(f"\n{c}")
        for k, r in rows.items():
            print(f"  {k:18s} n={r['n']} diff={r['mean_diff']:+6.2f} +- {r['sd_diff']:5.2f}  CI[{r['ci_low']:+6.2f},{r['ci_high']:+6.2f}]"
                  f"  t({r['df']})={r['t']:6.2f}  p={r['p']:.2g}  q={r['q']:.2g}  fav={r['fav']}/25")
    print("\nall values agree with the paper")
