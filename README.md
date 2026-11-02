# LS-ADLC

**Anatomically Partitioned Latent-Space Augmentation of Low-Confidence Samples for Rare Epileptiform Event Classification in Multichannel EEG**

A training method for imbalanced EEG classification. The 22-channel bipolar montage is split into three anatomical groups (temporal, central, parasagittal), each group is modeled by its own CNN-ResBiLSTM-Attention backbone, and the three streams are fused by the product rule. LS-ADLC identifies low-confidence minority samples in the latent space of each group and adds Gaussian-perturbed latent copies of them during training. It adds no parameters. The method is evaluated on TUEV and TUSZ.

## Paper

Authors: Muhammad Hasan Basri, Dwi Sunaryono\*, Yudhi Purwananto, Riyanarto Sarno (\*corresponding author)

"Anatomically Partitioned Latent-Space Augmentation of Low-Confidence Samples for Rare Epileptiform Event Classification in Multichannel EEG"

Submit journal: Intelligent Medicine, under review.

## Abstract

Spike-and-sharp waves and other rare epileptiform events are among the most clinically relevant targets of automated electroencephalogram (EEG) interpretation, yet classifiers trained on clinical recordings are biased toward the dominant background class. Augmented Data Low Confidence (ADLC) synthesizes samples for low-confidence cases, but it operates in the input space of a single model and has not been evaluated with model capacity held fixed. Latent-Space ADLC (LS-ADLC) moves the confidence gate and the synthesis into the latent space of a neural network and applies them independently to each scalp region. The 22-channel bipolar montage is partitioned into temporal, central, and parasagittal groups, each processed by a dedicated convolutional-recurrent backbone. Within every group, low-confidence minority samples are identified by an out-of-fold k-nearest-neighbor vote with subject-disjoint folds, and Gaussian-perturbed latent copies of them are added during training. With identical parameter counts in both ensembles, the comparison isolates the augmentation from model size. On the Temple University Hospital EEG Events Corpus, LS-ADLC raised balanced accuracy by 2.5 to 2.6 points and spike-and-sharp-wave F1 by 9.4 to 9.6 points in two separate 25-seed batches (paired t-test, p<0.01), meeting a replication criterion fixed in advance, and reached balanced accuracy of 59.8%, Cohen's kappa of 70.6%, and weighted F1-score of 84.8%. On the Temple University Hospital EEG Seizure Corpus, all four metrics improved in both batches. Region-wise confidence-gated latent augmentation improves rare epileptiform event recognition without enlarging the model.

## Repository Structure

```
LS_ADLC-Anatomical/
├── README.md                          # this file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT license
├── configs/
│   ├── config.yaml                    # TUEV hyperparameters
│   └── config_tusz.yaml               # TUSZ hyperparameters
├── data/                              # caches go here (see data/README.md)
│   └── README.md                      # how to obtain TUEV / TUSZ
├── results/                           # experiment outputs (created at runtime, git-ignored)
├── scripts/                           # run scripts of every training run
│   ├── flags.sh                       # settings shared by all runs
│   ├── run_all.sh                     # full queue for one corpus and one seed batch
│   ├── run_oof_knn.sh                 # LS-ADLC gate of the three groups
│   ├── run_pG.sh                      # control ensemble
│   ├── run_pLS.sh                     # LS-ADLC ensemble
│   ├── run_pE1_baseline.sh            # single full-montage backbone
│   └── run_pE1wide.sh                 # capacity-matched single backbone
└── src/                               # all source code
    ├── common.py                      # configuration, montage, filtering
    ├── features.py                    # DWT features
    ├── model.py                       # CNN-ResBiLSTM-Attention, channel-subset wrapper
    ├── latent_ops.py                  # latent operators imported by train.py
    ├── lsadlc_v2.py                   # alternative gates imported by train.py
    │
    │  ── Stage A · data preparation ──
    ├── build_windows.py               # TUEV: raw EDF to cached windows
    ├── precompute_feats.py            # TUEV: DWT features
    ├── build_tusz.py                  # TUSZ: raw EDF to cached windows
    ├── finalize_tusz_cache.py         # TUSZ: finalize the window cache
    ├── precompute_feats_tusz.py       # TUSZ: DWT features
    ├── export_npy.py                  # optional memory-mapped export
    │
    │  ── Stage B · training ──
    ├── oof_encoder.py                 # out-of-fold latent k-NN confidence gate
    ├── train.py                       # one group backbone, with or without LS-ADLC
    │
    │  ── Stage C · evaluation ──
    ├── ensemble_eval.py               # product-rule fusion, paired tests (Tables 5, 6, 8)
    ├── ensemble_per_class.py          # per-class F1 (Table 7)
    ├── ensemble_precision_recall.py   # precision and recall (Section 3.4)
    └── baseline_single_backbone.py    # ensemble vs. single backbones (Table 4)
```

Run every command from the repository root. Results are written to `results/` (git-ignored) as JSON and `.npy` files, and checkpoints to `checkpoints/`.

## Datasets

Two public scalp-EEG corpora are used. Neither can be redistributed.

| Dataset | Task | Source |
|---|---|---|
| TUEV (TUH EEG Events Corpus, v2.0.0) | 6-class event classification (SPSW, GPED, PLED, EYEM, ARTF, BCKG) | [NEDC](https://isip.piconepress.com) |
| TUSZ (TUH EEG Seizure Corpus, v2.0.5) | Binary seizure detection | [NEDC](https://isip.piconepress.com) |

See `data/README.md` for access and the expected folder layout.

## Installation

```bash
git clone https://github.com/hsanbsri/LS_ADLC-Anatomical.git
cd LS_ADLC-Anatomical
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.11 (CUDA 12.8) and one NVIDIA RTX 4080 SUPER (16 GB). A CUDA GPU is required for training. All runs use deterministic mode (`runtime.deterministic=true`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`), so results are bit-reproducible on the same hardware and software stack.

## Usage

### Stage A: data preparation

TUEV (`paths.dataset_root` in `configs/config.yaml` must contain `edf/train/` and `edf/eval/`):

```bash
python src/build_windows.py    --split train           # data/win_train.h5
python src/build_windows.py    --split eval            # data/win_eval.h5
python src/precompute_feats.py --split train --arm A   # data/feat_train_A.npy
python src/precompute_feats.py --split eval  --arm A   # data/feat_eval_A.npy
python src/export_npy.py       --split train           # optional: data/X_train.npy
```

TUSZ (`--tusz-root` is the TUSZ `edf/` directory):

```bash
C=configs/config_tusz.yaml
for s in train eval
do
  python src/build_tusz.py            --config $C --tusz-root /path/to/tusz/v2.0.5/edf --split $s
  python src/finalize_tusz_cache.py   --config $C --split $s
  python src/precompute_feats_tusz.py --config $C --split $s
  python src/export_npy.py            --config $C --split $s
done
```

### Stage B: training

```bash
bash scripts/run_all.sh tuev 42 66      # TUEV batch 1, with single-backbone controls
bash scripts/run_all.sh tuev 100 124    # TUEV batch 2
bash scripts/run_all.sh tusz 42 66      # TUSZ batch 1
bash scripts/run_all.sh tusz 100 124    # TUSZ batch 2
```

`run_all.sh` calls the individual scripts in `scripts/`, which can also be run alone. A run whose output file already exists is skipped, so an interrupted queue can be restarted. All settings are passed on the command line from `scripts/flags.sh`. pG and pLS differ only in `--lsadlc adlc` and the gate file.

### Stage C: evaluation

```bash
python src/baseline_single_backbone.py    # Table 4
python src/ensemble_eval.py               # Tables 5, 6, 8
python src/ensemble_per_class.py          # Table 7
python src/ensemble_precision_recall.py   # Section 3.4
```

These scripts read only the evaluation probabilities, the JSON outputs and the evaluation labels. Set `LSADLC_ROOT` to read a different results tree.

## Output

| File | Contents |
|---|---|
| `results*/<tag>_armA_seed<s>.json` | Effective configuration and metrics of one run |
| `results*/<tag>_armA_seed<s>_evalprob.npy` | Evaluation probabilities of one run |
| `data*/oof_pG_<group>.npz` | LS-ADLC gate of one anatomical group |
| `checkpoints*/` | Selected checkpoint of each run |
| `results/analysis/ensemble_eval_results.json` | Paired tests of both corpora and both batches |
| `results/analysis/ensemble_per_class_results.json` | Per-class F1 |
| `results/analysis/baseline_single_backbone_results.json` | Ensemble vs. single backbones |

## Architecture

```
Input EEG (22 bipolar channels, 5 s @ 250 Hz = 1250 samples)
        │
        ▼
Channel grouping (3 anatomical groups)
 ├── Temporal      (8 channels)
 ├── Central       (6 channels)
 └── Parasagittal  (8 channels)
        │  (per group)
        ▼
CNN → ResBiLSTM → Attention + DWT features
        │  (training only, per group)
        ▼
LS-ADLC: out-of-fold k-NN gate → Gaussian-perturbed latent copies
        │
        ▼
Product-rule fusion (mean log-probability, then argmax)
        │
        ▼
Event class
```

| Group | Channels | Indices |
|---|---|---|
| Temporal | FP1-F7, F7-T3, T3-T5, T5-O1, FP2-F8, F8-T4, T4-T6, T6-O2 | 0–7 |
| Central | A1-T3, T3-C3, C3-CZ, CZ-C4, C4-T4, T4-A2 | 8–13 |
| Parasagittal | FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2 | 14–21 |

## Methods

| Component | Description |
|---|---|
| pLS (proposed) | Three-group ensemble trained with LS-ADLC |
| pG | The same ensemble without LS-ADLC (control) |
| pE1 | Single full-montage backbone, 1,866,822 parameters |
| pE1wide | Capacity-matched single backbone, 5,473,414 parameters |

| Setting | Value |
|---|---|
| Input | 5-s windows at 250 Hz, notch 60/120 Hz, band-pass 0.5–80 Hz, DWT (db4, level 6) features |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, 2 warm-up epochs then cosine decay, batch 256, ≤ 40 epochs |
| Checkpoint selection | Best internal-validation Cohen's kappa, patience 10 |
| Regularization | Dropout 0.1 / 0.3 (head), channel dropout p = 0.3 over 12 channels, gradient clipping 1.0 |
| LS-ADLC gate | Out-of-fold k-NN vote, k = 15, distance-weighted, 5 subject-grouped folds, 15 epochs per fold, τ = 0.80, majority class excluded |
| LS-ADLC synthesis | z + α σ_c ε with α = 0.5, copy ratio 0.45, 0.50 or 0.55 of 10 copies, i.e. 4 or 5 copies per flagged window (Eq. 5) |

Both corpora use their official, disjoint train/eval pools. For checkpoint selection, the last 20% of the training subjects in alphabetical order are held out. The TUSZ dev pool is not used.

## Results

Means of 25 seeds per batch, fused ensemble, evaluation pool of each corpus. Values are in %. Δ is pLS − pG, with Benjamini–Hochberg q (paired t-test, m = 4 metrics per batch).

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

Batch 1 (seeds 42–66) and the full statistics are in Tables 5, 6 and 8 of the paper.

## Citation

If you use this code, please cite:

```bibtex
@article{basri2026lsadlc,
  title   = {Anatomically Partitioned Latent-Space Augmentation of Low-Confidence Samples
             for Rare Epileptiform Event Classification in Multichannel {EEG}},
  author  = {Basri, Muhammad Hasan and Sunaryono, Dwi and
             Purwananto, Yudhi and Sarno, Riyanarto},
  journal = {Intelligent Medicine},
  year    = {2026},
  note    = {Under review}
}
```

Please also cite the corpora and the original ADLC method:

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

Released under the MIT License.

## Contact

Dwi Sunaryono (corresponding author), dwi@its.ac.id
Department of Informatics, Institut Teknologi Sepuluh Nopember (ITS), Surabaya, Indonesia
