#!/bin/bash
# Shared settings of every run reported in the paper. Sourced by the run_*.sh scripts.
# All flags are stated explicitly; never rely on a default in configs/*.yaml.
#
# Usage inside a script:  source "$(dirname "${BASH_SOURCE[0]}")/flags.sh" <tuev|tusz>

CORPUS=${1:-tuev}
case "$CORPUS" in
  tuev) CONFIG="configs/config.yaml";      RES="results";      CKPT="checkpoints";      DATA="data";      GPU_CACHE=true ;;
  tusz) CONFIG="configs/config_tusz.yaml"; RES="results_tusz"; CKPT="checkpoints_tusz"; DATA="data_tusz"; GPU_CACHE=false ;;
  *) echo "unknown corpus '$CORPUS' (use tuev or tusz)"; exit 1 ;;
esac
PY=${PYTHON:-python3}
export CUBLAS_WORKSPACE_CONFIG=:4096:8          # deterministic cuBLAS

# Anatomical groups of the 22-channel TCP montage (Section 2.4 of the paper)
GROUP_NAMES=(temporal central parasagittal)
GROUP_IDS=("[0,1,2,3,4,5,6,7]" "[8,9,10,11,12,13]" "[14,15,16,17,18,19,20,21]")

# Split, determinism, checkpoint selection on validation Cohen's kappa
TRAIN_BASE="--config $CONFIG --arm A --monitor cohen_kappa --set runtime.deterministic=true \
--set periodicity.enabled=false --set laterality.enabled=false --set runtime.gpu_cache=$GPU_CACHE \
--set split.mode=alphabetical_tail --set split.val_fraction=0.2 \
--set paths.results_dir=$RES --set paths.ckpt_dir=$CKPT"

# Backbone recipe shared by all arms: channel dropout (p=0.3, 12 of 22 channels) and global max pooling
BACKBONE="--set training.augment.channel_dropout.enabled=true --set training.augment.channel_dropout.p=0.3 \
--set training.augment.channel_dropout.n_channels=12 --set training.augment.channel_dropout.protect_query=false \
--set model.max_pool.enabled=true --set model.ann_channel_branch.enabled=false"

# Per-group models (pG / pLS)
GROUP_FLAGS="$TRAIN_BASE $BACKBONE --set model.channel_query.enabled=false --set model.chain_expert.enabled=false"

# Out-of-fold k-NN gate (Algorithm 1): 5 subject-disjoint folds, 15 epochs per fold (defaults of oof_encoder.py)
OOF_FLAGS="--config $CONFIG --keep-cd --knn --set split.mode=alphabetical_tail --set split.val_fraction=0.2 \
--set periodicity.enabled=false --set laterality.enabled=false $BACKBONE"

# Capacity-matched single full-montage backbone (pE1wide, Table 4): 5,473,414 parameters
WIDE="--set model.temporal.stem_channels=113 --set model.temporal.res_blocks=[113,227,227,453] \
--set model.lstm.hidden=227 --set model.attention.dim=227"

mkdir -p logs "$RES" "$CKPT" "$DATA"
