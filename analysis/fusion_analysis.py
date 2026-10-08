"""Gabung tiga model per kelompok (pG_*) dengan rata-rata log-probabilitas, 6 kelas murni.
Log-prob rata-rata setara rata-rata logit untuk argmax (konstanta per baris tak mengubah argmax).
Pembanding: pE1_strict pada seed yang sama (berpasangan)."""
import os, sys, json, numpy as np, h5py
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
R = os.environ.get("LSADLC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUPS = ["pG_temporal", "pG_central", "pG_parasagittal"]
seeds = [int(s) for s in sys.argv[1:]] or list(range(42, 67))
y = h5py.File(f"{R}/data/win_eval.h5", "r")["y"][:].astype(int)
def met(p):
    pr = p.argmax(1)
    return dict(balAcc=100*balanced_accuracy_score(y, pr), kappa=100*cohen_kappa_score(y, pr),
                wF1=100*f1_score(y, pr, average="weighted"), SPSW=100*f1_score(y, pr, labels=[0], average=None)[0])
rows = {"ens": [], "pE1": []}; per = {g: [] for g in GROUPS}
for s in seeds:
    P = [np.load(f"{R}/results/{g}_armA_seed{s}_evalprob.npy") for g in GROUPS]
    ens = np.mean([np.log(np.clip(p, 1e-12, 1)) for p in P], axis=0)
    rows["ens"].append(met(ens)); rows["pE1"].append(met(np.load(f"{R}/results/pE1_strict_armA_seed{s}_evalprob.npy")))
    for g, p in zip(GROUPS, P): per[g].append(met(p))
keys = ["balAcc", "kappa", "wF1", "SPSW"]
for g in GROUPS:
    print(f"{g:16s} n={len(seeds)} " + " ".join(f"{k} {np.mean([r[k] for r in per[g]]):.2f}" for k in keys))
for k in keys:
    a = np.array([r[k] for r in rows["ens"]]); b = np.array([r[k] for r in rows["pE1"]])
    d = a - b; p = stats.ttest_rel(a, b).pvalue if len(d) > 1 else float("nan")
    print(f"{k:6s} ens {a.mean():6.2f} | pE1 {b.mean():6.2f} | Δ {d.mean():+.2f} p={p:.4g} seed+ {(d>0).sum()}/{len(d)}")
