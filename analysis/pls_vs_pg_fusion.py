"""pls_vs_pg_fusion.py — verification script for this paper's confirmatory contrast.

Adapts the verified reference `scripts/analisis_group.py` (which compares the
pG fused ensemble against pE1_strict) to instead compare the pLS fused
ensemble against the pG fused ensemble, at identical per-group architecture
and parameter count (the only config difference is lsadlc_train.enabled).

Fusion rule (unchanged from analisis_group.py): per-row mean of
log-clipped class probabilities across the three anatomical-group models
(temporal, central, parasagittal) — product rule / geometric-mean
combination (Kittler, Hatef, Duin & Matas, 1998) — argmax for the hard
label. Metrics computed on the held-out eval set: balanced accuracy,
Cohen's kappa, weighted-F1 (all x100), and SPSW-only F1 (class index 0) x100.

Statistics: paired t-test per seed (pairing = same seed, pLS vs. pG arm),
run separately for batch 1 (seeds 42-66) and batch 2 (seeds 100-124),
then Benjamini-Hochberg correction with m=4 within each batch (the four
metrics: balAcc, kappa, wF1, SPSW), matching the pre-specified analysis
plan recorded in README.md / 01_outline.md.

Read-only: loads only existing `results/*_evalprob.npy` and
`data/win_eval.h5`; writes nothing outside this project's data/analysis/.
"""
import json
import os

import h5py
import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

R = os.environ.get("LSADLC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUPS = ["temporal", "central", "parasagittal"]
KEYS = ["balAcc", "kappa", "wF1", "SPSW"]

BATCH1 = list(range(42, 67))   # exploratory batch (informed the LS-ADLC gate choice upstream)
BATCH2 = list(range(100, 125))  # confirmatory replication batch, criterion locked before inspection


def bh_correct(pvals):
    """Benjamini-Hochberg FDR correction. Returns q-values in the original order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_sorted = ranked * m / (np.arange(1, m + 1))
    # enforce monotonicity from the largest p-value down
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(m, dtype=float)
    q[order] = np.clip(q_sorted, 0, 1)
    return q


def met(p, y):
    pr = p.argmax(1)
    return dict(
        balAcc=100 * balanced_accuracy_score(y, pr),
        kappa=100 * cohen_kappa_score(y, pr),
        wF1=100 * f1_score(y, pr, average="weighted"),
        SPSW=100 * f1_score(y, pr, labels=[0], average=None)[0],
    )


def fuse(arm, seed):
    P = [np.load(f"{R}/results/{arm}_{g}_armA_seed{seed}_evalprob.npy") for g in GROUPS]
    return np.mean([np.log(np.clip(p, 1e-12, 1)) for p in P], axis=0)


def run_batch(seeds, y, label):
    rows_ls, rows_g = [], []
    for s in seeds:
        rows_ls.append(met(fuse("pLS", s), y))
        rows_g.append(met(fuse("pG", s), y))
    out = {"batch": label, "seeds": [seeds[0], seeds[-1]], "n": len(seeds), "per_metric": {}}
    pvals = []
    deltas_mean = []
    for k in KEYS:
        a = np.array([r[k] for r in rows_ls])  # pLS
        b = np.array([r[k] for r in rows_g])   # pG
        d = a - b
        tstat, p = stats.ttest_rel(a, b)
        pvals.append(p)
        deltas_mean.append(d.mean())
        out["per_metric"][k] = {
            "pLS_mean": float(a.mean()), "pG_mean": float(b.mean()),
            "delta_mean": float(d.mean()), "p_raw": float(p),
            "n_seeds_positive": int((d > 0).sum()),
        }
    q = bh_correct(pvals)
    for k, qv in zip(KEYS, q):
        out["per_metric"][k]["q_bh_m4"] = float(qv)
    return out


def main():
    y = h5py.File(f"{R}/data/win_eval.h5", "r")["y"][:].astype(int)
    results = {
        "batch1_seeds42_66": run_batch(BATCH1, y, "batch1_exploratory_seeds42-66"),
        "batch2_seeds100_124": run_batch(BATCH2, y, "batch2_confirmatory_seeds100-124"),
    }
    print(json.dumps(results, indent=2))
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "pls_vs_pg_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
