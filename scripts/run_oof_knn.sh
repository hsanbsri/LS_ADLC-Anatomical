#!/bin/bash
# Step 1 of 3: per-group out-of-fold (OOF) k-NN confidence estimates.
# Required before run_pLS.sh, which reads these as the LS-ADLC gate.
# Usage: bash scripts/run_oof_knn.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs data
PY=${PYTHON:-python3}

COMMON="--keep-cd --knn --set split.mode=alphabetical_tail --set split.val_fraction=0.2 \
--set periodicity.enabled=false --set laterality.enabled=false --set model.max_pool.enabled=true \
--set model.ann_channel_branch.enabled=false --set training.augment.channel_dropout.enabled=true \
--set training.augment.channel_dropout.p=0.3 --set training.augment.channel_dropout.n_channels=12 \
--set training.augment.channel_dropout.protect_query=false"

run() {
  name=$1; ids=$2
  [ -f "data/oof_${name}.npz" ] && { echo "skip $name (already exists)"; return; }
  $PY src/oof_encoder.py $COMMON --set model.channel_subset="$ids" --out "oof_${name}.npz" \
    > "logs/oof_${name}.log" 2>&1 || { echo "FAILED: $name"; exit 1; }
  echo "$(date +%H:%M) done: $name"
}

run pG_temporal "[0,1,2,3,4,5,6,7]"
run pG_central "[8,9,10,11,12,13]"
run pG_parasagittal "[14,15,16,17,18,19,20,21]"
echo OOF_DONE
