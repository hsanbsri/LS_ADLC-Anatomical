# Data

Neither corpus can be redistributed. Both are available free of charge to registered users of the Neural Engineering Data Consortium (NEDC), Temple University: https://isip.piconepress.com

| Corpus | Version | Expected input |
|---|---|---|
| TUEV | v2.0.0 | `paths.dataset_root` in `configs/config.yaml` (default `../dataset`) contains `edf/train/` and `edf/eval/`. Otherwise add `--set paths.dataset_root=/path/to/tuev` to the `build_windows.py` calls |
| TUSZ | v2.0.5 | `--tusz-root` points to the TUSZ `edf/` directory with `train/`, `dev/`, `eval/` |

The Stage A scripts in `src/` write the caches below.

| File | Contents |
|---|---|
| `data/win_train.h5`, `data/win_eval.h5` | TUEV windows |
| `data/feat_train_A.npy`, `data/feat_eval_A.npy` | TUEV DWT features |
| `data/X_train.npy` | Optional memory-mapped export |
| `data_tusz/` | The same files for TUSZ |
| `data*/oof_pG_<group>.npz` | LS-ADLC gate, written by `scripts/run_oof_knn.sh` |
