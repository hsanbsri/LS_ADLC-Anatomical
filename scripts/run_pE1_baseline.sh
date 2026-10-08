#!/bin/bash
# Step 0 of 3 (optional): single full-montage backbone baseline (pE1).
# Same recipe as pG/pLS but with NO channel partition (all 22 channels,
# one model). Only needed to reproduce the descriptive, capacity-confounded
# pG-vs-pE1 comparison (Table 2 in the paper); the paper's one confirmatory
# claim (pLS vs. pG) does not depend on this baseline.
# Usage: bash scripts/run_pE1_baseline.sh <seed_start> <seed_end>
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

for s in $(seq "$S1" "$S2"); do
  [ -f "results/pE1_strict_armA_seed${s}.json" ] && continue
  $PY src/train.py $E --variant baseline --seed "$s" --tag pE1_strict \
    > "logs/pE1_strict_seed${s}.log" 2>&1 || { echo "FAILED: pE1 seed $s"; exit 1; }
  echo "$(date +%H:%M) done: pE1_strict seed $s"
done
echo PE1_DONE
