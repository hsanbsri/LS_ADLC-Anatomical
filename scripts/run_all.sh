#!/bin/bash
# Every training run of the paper for one corpus and one seed batch.
# Usage: bash scripts/run_all.sh <tuev|tusz> <first_seed> <last_seed>
#   paper: tuev 42 66, tuev 100 124, tusz 42 66, tusz 100 124
# (the single-backbone controls pE1/pE1wide were run for TUEV seeds 42-66 only)
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C=${1:-tuev}; S1=${2:-42}; S2=${3:-66}
bash scripts/run_oof_knn.sh "$C"            || exit 1
bash scripts/run_pG.sh  "$C" "$S1" "$S2"    || exit 1
bash scripts/run_pLS.sh "$C" "$S1" "$S2"    || exit 1
if [ "$C" = "tuev" ] && [ "$S1" = "42" ]; then
  bash scripts/run_pE1_baseline.sh "$S1" "$S2" || exit 1
  bash scripts/run_pE1wide.sh "$S1" "$S2"      || exit 1
fi
echo "all runs done: $C seeds $S1-$S2"
