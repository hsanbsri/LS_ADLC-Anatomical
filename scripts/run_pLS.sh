#!/bin/bash
# Step 2b of 3: treatment arm (pLS) — identical to run_pG.sh, plus the
# per-group LS-ADLC confidence-gated latent-space augmentation. Requires
# run_oof_knn.sh to have been run first (reads data/oof_pG_<group>.npz).
# Usage: bash scripts/run_pLS.sh <seed_start> <seed_end>
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results checkpoints
PY=${PYTHON:-python3}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

S1=${1:-42}; S2=${2:-42}
P="--arm A --monitor cohen_kappa --set runtime.deterministic=true --set periodicity.enabled=false \
--set laterality.enabled=false --set runtime.gpu_cache=true --set split.mode=alphabetical_tail \
--set split.val_fraction=0.2"
E="$P --set training.augment.channel_dropout.enabled=true --set training.augment.channel_dropout.p=0.3 \
--set training.augment.channel_dropout.n_channels=12 --set training.augment.channel_dropout.protect_query=false \
--set model.max_pool.enabled=true --set model.ann_channel_branch.enabled=false \
--set model.channel_query.enabled=false"

run() {
  name=$1; ids=$2; s=$3
  [ -f "results/pLS_${name}_armA_seed${s}.json" ] && { echo "skip $name seed $s"; return; }
  $PY src/train.py $E --variant baseline --seed "$s" --tag "pLS_${name}" --lsadlc adlc \
    --set model.channel_subset="$ids" --set lsadlc_train.adlc.knn_file="oof_pG_${name}.npz" \
    > "logs/pLS_${name}_seed${s}.log" 2>&1 || { echo "FAILED: $name seed $s"; exit 1; }
  echo "$(date +%H:%M) done: pLS_${name} seed $s"
}

for s in $(seq "$S1" "$S2"); do
  run temporal "[0,1,2,3,4,5,6,7]" "$s"
  run central "[8,9,10,11,12,13]" "$s"
  run parasagittal "[14,15,16,17,18,19,20,21]" "$s"
done
echo PLS_DONE
