#!/bin/bash
# Algorithm 1: per-group out-of-fold k-NN confidence gate of LS-ADLC.
# Run once per corpus, before run_pLS.sh; the gate does not depend on the training seed.
# Usage: bash scripts/run_oof_knn.sh <tuev|tusz>
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/flags.sh "${1:-tuev}"
for i in 0 1 2; do
  name=${GROUP_NAMES[$i]}; ids=${GROUP_IDS[$i]}
  [ -f "$DATA/oof_pG_${name}.npz" ] && { echo "skip $name (exists)"; continue; }
  $PY src/oof_encoder.py $OOF_FLAGS --set model.channel_subset="$ids" --out "oof_pG_${name}.npz" \
    > "logs/${CORPUS}_oof_pG_${name}.log" 2>&1 || { echo "FAILED: OOF $name"; exit 1; }
  echo "$(date +%H:%M) done: OOF gate $name ($CORPUS)"
done
