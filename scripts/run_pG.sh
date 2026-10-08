#!/bin/bash
# Step 2a of 3: control arm (pG) — three independently trained per-group
# backbones, NO LS-ADLC. Same training recipe as the single-backbone
# baseline except the input is restricted to one anatomical group.
# Usage: bash scripts/run_pG.sh <seed_start> <seed_end>
#   e.g. bash scripts/run_pG.sh 42 66
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
  tag=$1; ids=$2; s=$3
  [ -f "results/${tag}_armA_seed${s}.json" ] && return
  $PY src/train.py $E --variant baseline --seed "$s" --tag "$tag" --set model.channel_subset="$ids" \
    > "logs/${tag}_seed${s}.log" 2>&1 || { echo "FAILED: $tag seed $s"; exit 1; }
  echo "$(date +%H:%M) done: $tag seed $s"
}

for s in $(seq "$S1" "$S2"); do
  run pG_temporal "[0,1,2,3,4,5,6,7]" "$s"
  run pG_central "[8,9,10,11,12,13]" "$s"
  run pG_parasagittal "[14,15,16,17,18,19,20,21]" "$s"
done
echo PG_DONE
