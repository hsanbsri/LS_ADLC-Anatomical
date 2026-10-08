"""table5_per_class.py — per-class F1 for the three group models and the
fused ensemble, pG vs. pLS, batch 2 (seeds 100-124, the confirmatory batch).

Read-only. Group-level per-class F1 comes straight from each run's logged
`eval.per_class_f1` (JSON). Fused-ensemble per-class F1 is computed from the
same log-probability fusion used throughout this project
(`scripts/analisis_group.py` fusion rule), applied to `*_evalprob.npy`.
"""
import json
import os

import h5py
import numpy as np
from sklearn.metrics import f1_score

R = os.environ.get("LSADLC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUPS = ["temporal", "central", "parasagittal"]
CLASSES = ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"]
BATCH2 = list(range(100, 125))


def fuse(arm, seed):
    P = [np.load(f"{R}/results/{arm}_{g}_armA_seed{seed}_evalprob.npy") for g in GROUPS]
    return np.mean([np.log(np.clip(p, 1e-12, 1)) for p in P], axis=0)


def main():
    y = h5py.File(f"{R}/data/win_eval.h5", "r")["y"][:].astype(int)

    out = {"batch": "batch2_seeds100-124", "n_seeds": len(BATCH2), "groups": {}, "fused": {}}

    # per-group, per-class F1 straight from logged eval metrics
    for arm in ["pG", "pLS"]:
        for g in GROUPS:
            vals = {c: [] for c in CLASSES}
            for s in BATCH2:
                d = json.load(open(f"{R}/results/{arm}_{g}_armA_seed{s}.json"))
                pcf = d["eval"]["per_class_f1"]
                for c in CLASSES:
                    vals[c].append(pcf[c] * 100)
            out["groups"][f"{arm}_{g}"] = {c: float(np.mean(v)) for c, v in vals.items()}

    # fused ensemble, per-class F1, computed from evalprob fusion
    for arm in ["pG", "pLS"]:
        per_class_runs = {c: [] for c in CLASSES}
        for s in BATCH2:
            ens = fuse(arm, s)
            pr = ens.argmax(1)
            f1s = f1_score(y, pr, average=None, labels=list(range(6)))
            for ci, c in enumerate(CLASSES):
                per_class_runs[c].append(f1s[ci] * 100)
        out["fused"][arm] = {c: float(np.mean(v)) for c, v in per_class_runs.items()}

    print(json.dumps(out, indent=2))
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "table5_per_class_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
