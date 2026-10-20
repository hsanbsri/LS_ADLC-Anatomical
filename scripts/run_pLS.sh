#!/bin/bash
# Treatment arm pLS: the same three per-group backbones trained with LS-ADLC (Algorithm 2).
# Requires the gate files of run_oof_knn.sh.
# Usage: bash scripts/run_pLS.sh <tuev|tusz> <first_seed> <last_seed>    (paper: 42 66 and 100 124)
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/flags.sh "${1:-tuev}"
S1=${2:-42}; S2=${3:-42}
for s in $(seq "$S1" "$S2"); do
  for i in 0 1 2; do
    name=${GROUP_NAMES[$i]}; ids=${GROUP_IDS[$i]}; tag=pLS_${name}
    [ -f "$DATA/oof_pG_${name}.npz" ] || { echo "missing $DATA/oof_pG_${name}.npz: run scripts/run_oof_knn.sh $CORPUS first"; exit 1; }
    [ -f "$RES/${tag}_armA_seed${s}.json" ] && continue
    $PY src/train.py $GROUP_FLAGS --variant baseline --seed "$s" --tag "$tag" --lsadlc adlc \
      --set model.channel_subset="$ids" --set lsadlc_train.adlc.knn_file="oof_pG_${name}.npz" \
      > "logs/${CORPUS}_${tag}_seed${s}.log" 2>&1 || { echo "FAILED: $tag seed $s"; exit 1; }
    echo "$(date +%H:%M) done: $tag seed $s ($CORPUS)"
  done
done
