#!/bin/bash
# Capacity-matched single full-montage backbone pE1wide (5,473,414 parameters, Table 4). TUEV only.
# Usage: bash scripts/run_pE1wide.sh <first_seed> <last_seed>            (paper: 42 66)
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/flags.sh tuev
S1=${1:-42}; S2=${2:-42}
for s in $(seq "$S1" "$S2"); do
  [ -f "$RES/pE1wide_armA_seed${s}.json" ] && continue
  $PY src/train.py $TRAIN_BASE $BACKBONE $WIDE --seed "$s" --tag pE1wide \
    > "logs/tuev_pE1wide_seed${s}.log" 2>&1 || { echo "FAILED: pE1wide seed $s"; exit 1; }
  echo "$(date +%H:%M) done: pE1wide seed $s"
done
