# LS-ADLC-Anatomical

Code and analysis for:

> **Confidence-Driven Latent-Space Augmentation in an Anatomically
> Partitioned Multichannel-EEG Ensemble for Epileptiform Event
> Classification: A Replicated, Capacity-Matched Evaluation**
> Muhammad Hasan Basri. Manuscript in preparation / under review.

This repository trains and evaluates a three-branch, anatomically
partitioned ensemble (temporal / central / parasagittal channel groups) on
6-class EEG event classification (TUEV), with and without a per-group
latent-space confidence-gated augmentation mechanism (**LS-ADLC**, an
extension of ADLC [Sunaryono et al., 2025] into a latent-space,
out-of-fold, per-anatomical-group setting). The confirmatory result
(LS-ADLC vs. no-LS-ADLC, at identical ensemble capacity) was replicated on
two independent 25-seed batches with a pass/fail criterion fixed before
the second batch was inspected.

## What this repository contains

- **`src/`** — model and training code: backbone architecture
  (`model.py`), channel-partition wrapper, data loading/config (`common.py`,
  `features.py`), the training loop (`train.py`), and the out-of-fold k-NN
  confidence-gate generator for LS-ADLC (`oof_encoder.py`).
- **`data_prep/`** — turns raw TUEV `.edf`/`.rec` files into the cached
  HDF5/`.npy` files `src/train.py` and `src/oof_encoder.py` read
  (`build_windows.py`, `precompute_feats.py`).
- **`scripts/`** — shell wrappers that reproduce the paper's two arms
  across a seed range (see **Reproducing the paper's results** below).
- **`analysis/`** — standalone scripts that compute every statistic
  reported in the paper directly from the JSON/`.npy` outputs of
  `scripts/`, independent of the training code.
- **`figures/`** — scripts that regenerate Figures 1-2 from the paper.
- **`configs/config.yaml`** — single source of truth for all
  hyperparameters; every run overrides specific keys with `--set
  a.b.c=value` rather than editing this file (see **Configuration**).

## What this repository does *not* contain

- The TUEV dataset itself (see **Dataset access** below — it is not
  redistributable).
- Trained model checkpoints or cached features (large binary files;
  `.gitignore`d by design — regenerate them with the steps below).
- Any code related to other, unrelated experiments from the broader
  research project this work was developed alongside. This repository is
  scoped to exactly the one architecture and the one confirmatory claim
  described in the paper above.

## Installation

```bash
git clone https://github.com/hsanbsri/LS_ADLC-Anatomical-.git
cd LS_ADLC-Anatomical-
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12, CUDA-capable GPU (16 GB+ VRAM recommended;
`gpu_cache` in `configs/config.yaml` can be set to `false` to trade speed
for lower memory use). `src/train.py` and `src/oof_encoder.py` run on CPU
but are substantially slower.

## Dataset access

This work uses the **TUH EEG Events Corpus (TUEV)**, version 2.0.0, a
subset of the TUH EEG Corpus maintained by the Neural Engineering Data
Consortium (NEDC) at Temple University. It is **not included** in this
repository and is not freely redistributable; it is available at no cost
to registered users after agreeing to the corpus's terms of use. Start at
the NEDC/ISIP EEG resources page (`isip.piconepress.com`) or contact
`help@nedcdata.org` for current registration instructions.

**If you use TUEV, please cite it as its own documentation requests:**

```bibtex
@inproceedings{harati2015tuev,
  author    = {Harati, A. and Golmohammadi, M. and Lopez, S. and Obeid, I. and Picone, J.},
  title     = {Improved {EEG} Event Classification Using Differential Energy},
  booktitle = {Proc. IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
  year      = {2015},
  pages     = {1--4},
  address   = {Philadelphia, PA, USA}
}

@article{obeid2016tuh,
  author  = {Obeid, I. and Picone, J.},
  title   = {The {Temple University Hospital EEG} Data Corpus},
  journal = {Frontiers in Neuroscience},
  volume  = {10},
  pages   = {196},
  year    = {2016},
  doi     = {10.3389/fnins.2016.00196}
}
```

Once you have access, place the corpus so that `configs/config.yaml`'s
`paths.dataset_root` (default: `../dataset`, relative to this repo —
override with `--set paths.dataset_root=/your/path`) points at a directory
containing an `edf/` subfolder with TUEV's `train/` and `eval/` splits, as
distributed by NEDC.

## Reproducing the paper's results

**Step 1 — build the window cache** from raw EDF/`.rec` files (run once;
~hours depending on disk I/O, not GPU-bound):

```bash
python3 data_prep/build_windows.py --split train   # writes data/win_train.h5, X_train.npy
python3 data_prep/build_windows.py --split eval    # writes data/win_eval.h5, X_eval.npy
python3 data_prep/precompute_feats.py --split train --arm A   # writes data/feat_train_A.npy (DWT subband features)
python3 data_prep/precompute_feats.py --split eval  --arm A   # writes data/feat_eval_A.npy
```

**Step 2 — run the three arms** (each script skips a seed/tag combination
if its output JSON already exists, so interrupted runs resume safely):

```bash
# optional, descriptive-only comparator (Table 2 in the paper; the
# confirmatory claim does not depend on this)
bash scripts/run_pE1_baseline.sh 42 66

# required for LS-ADLC: per-group out-of-fold k-NN confidence gates
bash scripts/run_oof_knn.sh

# control arm: three independent per-group backbones, no LS-ADLC
bash scripts/run_pG.sh 42 66

# treatment arm: same, + per-group LS-ADLC
bash scripts/run_pLS.sh 42 66
```

or, as one call: `bash scripts/run_all.sh 42 66`.

The paper used **seeds 42-66** (batch 1, exploratory) and an independently
run **seeds 100-124** (batch 2, confirmatory — its pass/fail criterion was
fixed before these seeds were inspected). To reproduce the replication
batch: `bash scripts/run_all.sh 100 124` (after `run_oof_knn.sh` has
already been run once — the OOF gate does not depend on the evaluation
seed).

**Step 3 — compute the paper's statistics:**

```bash
python3 analysis/pls_vs_pg_fusion.py      # Tables 3-4: pLS vs. pG, both batches, paired tests + BH correction
python3 analysis/fusion_analysis.py       # Table 2: pG vs. pE1 (descriptive, capacity-confounded)
python3 analysis/table5_per_class.py      # Table 5: per-class F1, batch 2
python3 figures/generate_fig1.py          # Figure 1: architecture schematic
python3 figures/generate_fig2.py          # Figure 2: replication forest plot
```

All analysis and figure scripts resolve paths relative to the repository
root automatically; set `LSADLC_ROOT=/path/to/repo` if you run them from
elsewhere.

## Configuration

`configs/config.yaml` documents every switch with inline comments. The two
arms in this paper differ by exactly one flag block, visible directly in
`scripts/run_pG.sh` vs. `scripts/run_pLS.sh`: `--lsadlc adlc` plus
`--set lsadlc_train.adlc.knn_file=...`. Every other training flag
(optimizer, schedule, augmentation, split, determinism) is identical
between the two arms and is passed explicitly on the command line — never
assume a default from the file reflects what a given run used; read the
`--set` flags in the shell script that produced it, or the run's own
output JSON (which logs its full effective configuration).

## Anatomical channel partition

Fixed, 22-channel bipolar TCP (ACNS) montage, split by index (original to
this paper, not adapted from a prior published grouping — see the
manuscript's Methods):

| Group | Channel indices | Count |
|---|---|---|
| Temporal | 0-7 | 8 |
| Central | 8-13 | 6 |
| Parasagittal | 14-21 | 8 |

## Citation

If you use this code, please cite the manuscript (BibTeX to be added once
a DOI/preprint is assigned) and the TUEV/TUH EEG Corpus citations above,
and, if you build on the LS-ADLC mechanism specifically, the original ADLC
method:

```bibtex
@article{sunaryono2025adlc,
  author  = {Sunaryono, D. and Sarno, R. and Sabilla, S. I. and Putri, R. A. and Amri, T. C. and Mirda, I. and Siswantoro, J.},
  title   = {Augmented Data Low Confidence ({ADLC}): A Confidence-Driven Data Augmentation Framework With Ensemble Optimization for Enhanced Machine Learning Performance},
  journal = {IEEE Access},
  volume  = {13},
  pages   = {201439--201459},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3636729}
}
```

## License

MIT — see `LICENSE`.

## Contact

Muhammad Hasan Basri — issues and questions via this repository's GitHub
Issues page.
