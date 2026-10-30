#!/bin/bash
# Single full-montage backbone pE1 (1,866,822 parameters; exploratory comparison, Table 4). TUEV only.
# Usage: bash scripts/run_pE1_baseline.sh <first_seed> <last_seed>       (paper: 42 66)
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/flags.sh tuev
S1=${1:-42}; S2=${2:-42}
for s in $(seq "$S1" "$S2"); do
  [ -f "$RES/pE1_strict_armA_seed${s}.json" ] && continue
  $PY src/train.py $TRAIN_BASE $BACKBONE --set model.channel_query.enabled=false --variant baseline \
    --seed "$s" --tag pE1_strict > "logs/tuev_pE1_strict_seed${s}.log" 2>&1 || { echo "FAILED: pE1 seed $s"; exit 1; }
  echo "$(date +%H:%M) done: pE1 seed $s"
done
