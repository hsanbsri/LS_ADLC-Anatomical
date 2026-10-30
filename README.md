# LS-ADLC in an anatomically partitioned EEG ensemble

Code for the paper

> **Anatomically Partitioned Latent-Space Augmentation of Low-Confidence Samples for Rare Epileptiform Event Classification in Multichannel EEG**
> Muhammad Hasan Basri, Dwi Sunaryono, Yudhi Purwananto, Riyanarto Sarno.
> Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya, Indonesia.
> (under review)

The 22-channel bipolar TCP montage is divided into three anatomical groups
(temporal, central, parasagittal). Each group is classified by its own
CNN-ResBiLSTM-Attention backbone, and the three groups are fused by the
product rule. **LS-ADLC** is a latent-space extension of Augmented Data Low
Confidence (ADLC, Sunaryono et al., 2025). Before training, an out-of-fold
latent k-NN vote flags low-confidence minority-class windows. During
training, Gaussian-perturbed latent copies of the flagged windows are added
in each branch. LS-ADLC adds no parameters, so the two arms compared in the
paper have identical capacity:

| Arm | Description |
|---|---|
| **pG** | three-group ensemble, no LS-ADLC (control) |
| **pLS** | the same ensemble trained with LS-ADLC (proposed) |

## Main results

Means of 25 seeds per batch, fused ensemble, evaluation pool of each corpus.
Values are in %. Δ is pLS − pG, with Benjamini–Hochberg q (paired t-test,
m = 4 metrics per batch).

| Corpus, batch | Metric | pG | pLS | Δ | q |
|---|---|---:|---:|---:|---:|
| TUEV, batch 2 (seeds 100–124, confirmatory) | Balanced accuracy | 57.26 | 59.79 | +2.53 | 0.019 |
| | SPSW-F1 | 18.84 | 28.25 | +9.42 | 0.019 |
| | Cohen's kappa | 71.22 | 70.63 | −0.59 | 0.13 |
| | Weighted-F1 | 84.80 | 84.77 | −0.04 | 0.85 |
| TUSZ, batch 2 (seeds 100–124) | Balanced accuracy | 67.28 | 70.87 | +3.59 | 3.2e-5 |
| | Seizure-F1 | 50.33 | 56.68 | +6.35 | 3.2e-5 |
| | Cohen's kappa | 44.51 | 50.40 | +5.90 | 3.2e-5 |
| | Weighted-F1 | 85.11 | 86.45 | +1.34 | 3.8e-5 |

Batch 1 (seeds 42–66) and the full statistics (mean ± SD of the paired
differences, 95% CI, t, p) are in Tables 5, 6 and 8 of the paper. All of
them are reproduced by `analysis/paired_ttest_table.py`.

## Repository layout

```
configs/      config.yaml (TUEV), config_tusz.yaml (TUSZ), all hyperparameters
data_prep/    raw EDF -> cached windows (HDF5) and DWT features (.npy)
src/          model, training loop, LS-ADLC gate
scripts/      run scripts of every training run reported in the paper
analysis/     statistics of Tables 4-8 and Section 3.4
figures/      Figs. 1-6 of the paper
```

| File | Role | Paper |
|---|---|---|
| `src/model.py` | CNN-ResBiLSTM-Attention backbone, channel-subset wrapper | Sect. 2.3, Fig. 3 |
| `src/oof_encoder.py` | out-of-fold latent k-NN confidence gate (Stage A) | Algorithm 1, Eqs. 2–3 |
| `src/train.py` | training of one group backbone, with or without LS-ADLC (Stage B), checkpoint selection, evaluation | Algorithm 2, Eqs. 4–5 |
| `analysis/*.py` | product-rule fusion of the three groups, metrics and statistics | Algorithm S1 (Supplementary Material), Eq. 1, Sect. 2.8 |
| `src/common.py`, `src/features.py` | configuration, montage, filtering, DWT features | Sect. 2.2 |
| `src/latent_ops.py`, `src/lsadlc_v2.py` | alternative gates and latent operators imported by `train.py`, not used by the paper's configuration | |

## Installation

```bash
git clone <repository-url> ls-adlc && cd ls-adlc
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.11 (CUDA 12.8) and one NVIDIA RTX 4080
SUPER (16 GB). Training needs a CUDA GPU. All runs use deterministic mode
(`runtime.deterministic=true`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Results
are therefore bit-reproducible on the same hardware and software stack.
Other GPUs or library versions may change the last digits.

## Data

Neither corpus can be redistributed. Both are available free of charge to
registered users of the Neural Engineering Data Consortium (NEDC), Temple
University (https://isip.piconepress.com):

- **TUEV**, TUH EEG Events Corpus, v2.0.0: 6-class event classification
  (SPSW, GPED, PLED, EYEM, ARTF, BCKG).
- **TUSZ**, TUH EEG Seizure Corpus, v2.0.5: binary seizure detection.

Both corpora use their official, disjoint train/eval pools. For
checkpoint selection, the last 20% of the training subjects in alphabetical
order are held out (`split.mode=alphabetical_tail`). The TUSZ dev pool is
not used.

## Reproduction

Every command is run from the repository root. Each run script skips a run
whose output file already exists, so an interrupted queue can be restarted.

### 1. Build the caches (CPU, once)

TUEV: `paths.dataset_root` in `configs/config.yaml` (default `../dataset`)
must contain `edf/train/` and `edf/eval/` as distributed by NEDC. Otherwise,
add `--set paths.dataset_root=/path/to/tuev` to the two `build_windows.py`
calls.

```bash
python data_prep/build_windows.py    --split train          # data/win_train.h5
python data_prep/build_windows.py    --split eval           # data/win_eval.h5
python data_prep/precompute_feats.py --split train --arm A  # data/feat_train_A.npy
python data_prep/precompute_feats.py --split eval  --arm A  # data/feat_eval_A.npy
python data_prep/export_npy.py       --split train          # optional: data/X_train.npy (memory-mapped)
```

TUSZ (`--tusz-root` is the TUSZ `edf/` directory with `train/`, `dev/`, `eval/`):

```bash
C=configs/config_tusz.yaml
for s in train eval; do
  python data_prep/build_tusz.py           --config $C --tusz-root /path/to/tusz/v2.0.5/edf --split $s
  python data_prep/finalize_tusz_cache.py  --config $C --split $s
  python data_prep/precompute_feats_tusz.py --config $C --split $s
  python data_prep/export_npy.py           --config $C --split $s   # optional
done
```

### 2. Train (GPU)

```bash
bash scripts/run_all.sh tuev 42 66      # TUEV batch 1 (+ single-backbone controls pE1, pE1wide)
bash scripts/run_all.sh tuev 100 124    # TUEV batch 2
bash scripts/run_all.sh tusz 42 66      # TUSZ batch 1
bash scripts/run_all.sh tusz 100 124    # TUSZ batch 2
```

`run_all.sh` calls the individual scripts, which can also be run alone:

| Script | Runs |
|---|---|
| `run_oof_knn.sh <corpus>` | LS-ADLC gate for the three groups, `data*/oof_pG_<group>.npz` (once per corpus, independent of the seed) |
| `run_pG.sh <corpus> <s1> <s2>` | control arm, 3 groups × seeds |
| `run_pLS.sh <corpus> <s1> <s2>` | LS-ADLC arm, 3 groups × seeds |
| `run_pE1_baseline.sh <s1> <s2>` | single full-montage backbone pE1, 1,866,822 parameters (TUEV, Table 4) |
| `run_pE1wide.sh <s1> <s2>` | capacity-matched single backbone pE1wide, 5,473,414 parameters (TUEV, Table 4) |

All settings are passed explicitly on the command line from
`scripts/flags.sh`. pG and pLS differ only in `--lsadlc adlc` and the gate
file. Each run writes `results*/<tag>_armA_seed<s>.json`, which holds the
full effective configuration and metrics, together with the evaluation
probabilities `*_evalprob.npy`, and saves the selected checkpoint in
`checkpoints*/`. One run of the paper uses the following settings
(Table 2):

| Setting | Value |
|---|---|
| Input | 5-s windows at 250 Hz, notch 60/120 Hz, band-pass 0.5–80 Hz, DWT (db4, level 6) features |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, 2 warm-up epochs then cosine decay, batch 256, ≤ 40 epochs |
| Checkpoint selection | best internal-validation Cohen's kappa, patience 10 |
| Regularization | dropout 0.1 / 0.3 (head), channel dropout p = 0.3 over 12 channels, gradient clipping 1.0 |
| LS-ADLC gate | out-of-fold k-NN vote, k = 15, distance-weighted, 5 subject-grouped folds, 15 epochs per fold, τ = 0.80, majority class excluded |
| LS-ADLC synthesis | z + α σ_c ε with α = 0.5, copy ratio 0.45, 0.50 or 0.55 of 10 copies, i.e. 4 or 5 copies per flagged window (Eq. 5) |

### 3. Statistics and figures

```bash
python analysis/capacity_matched_single.py   # Table 4: pG vs. pE1 and pE1wide
python analysis/paired_ttest_table.py        # Tables 5, 6, 8: pLS vs. pG, TUEV and TUSZ, both batches
python analysis/per_class_f1.py              # Table 7: per-class F1, TUEV batch 2
python analysis/precision_recall.py          # Section 3.4: precision and recall
python figures/make_figures.py               # Figs. 1-6 -> figures/output/
```

The analysis scripts read only the `*_evalprob.npy`/JSON outputs and the
evaluation labels. Their results are written to `analysis/output/`.
`paired_ttest_table.py` asserts that every value agrees with the paper. The
scripts locate the repository root automatically. Set `LSADLC_ROOT` to point
them at a different results tree. `fig1_examples.py` also reads the raw TUEV
EDF files.

## Anatomical partition

Channel indices refer to the 22-channel TCP montage in `configs/config.yaml`
(`channels.montage`).

| Group | Channels | Indices |
|---|---|---|
| Temporal | FP1-F7, F7-T3, T3-T5, T5-O1, FP2-F8, F8-T4, T4-T6, T6-O2 | 0–7 |
| Central | A1-T3, T3-C3, C3-CZ, CZ-C4, C4-T4, T4-A2 | 8–13 |
| Parasagittal | FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2 | 14–21 |

## Citation

The citation of the paper will be added upon publication. Please also cite
the corpora and the original ADLC method:

```bibtex
@article{obeid2016,
  author  = {Obeid, Iyad and Picone, Joseph},
  title   = {The {Temple University Hospital} {EEG} Data Corpus},
  journal = {Frontiers in Neuroscience}, year = {2016}, volume = {10}, pages = {196},
  doi     = {10.3389/fnins.2016.00196}
}
@inproceedings{harati2015,
  author    = {Harati, A. and Golmohammadi, M. and Lopez, S. and Obeid, I. and Picone, J.},
  title     = {Improved {EEG} event classification using differential energy},
  booktitle = {2015 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
  year      = {2015}, pages = {1--4}, doi = {10.1109/spmb.2015.7405421}
}
@article{shah2018,
  author  = {Shah, Vinit and von Weltin, Eva and Lopez, Silvia and McHugh, James Riley and Veloso, Lillian
             and Golmohammadi, Meysam and Obeid, Iyad and Picone, Joseph},
  title   = {The {Temple University Hospital} Seizure Detection Corpus},
  journal = {Frontiers in Neuroinformatics}, year = {2018}, volume = {12}, pages = {83},
  doi     = {10.3389/fninf.2018.00083}
}
@article{sunaryono2025,
  author  = {Sunaryono, Dwi and Sarno, Riyanarto and Sabilla, Shoffi Izza and Putri, Rizqy Ahsana
             and Amri, Taufiq Choirul and Mirda, Irfan and Siswantoro, Joko},
  title   = {Augmented Data Low Confidence ({ADLC}): A Confidence-Driven Data Augmentation Framework
             With Ensemble Optimization for Enhanced Machine Learning Performance},
  journal = {IEEE Access}, year = {2025}, volume = {13}, pages = {201439--201459},
  doi     = {10.1109/access.2025.3636729}
}
```

## License

MIT, see `LICENSE`.
