#!/bin/bash
# Convenience wrapper — baseline, OOF gates, control arm, treatment arm,
# for one seed range. See README.md for what each step produces.
# Usage: bash scripts/run_all.sh <seed_start> <seed_end>
#   e.g. bash scripts/run_all.sh 42 66
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
S1=${1:-42}; S2=${2:-66}

echo "$(date +%H:%M) [1/4] pE1 baseline (optional, descriptive comparator), seeds $S1-$S2"
bash scripts/run_pE1_baseline.sh "$S1" "$S2" || { echo "pE1 step failed"; exit 1; }

echo "$(date +%H:%M) [2/4] OOF k-NN gates"
bash scripts/run_oof_knn.sh || { echo "OOF step failed"; exit 1; }

echo "$(date +%H:%M) [3/4] pG (control), seeds $S1-$S2"
bash scripts/run_pG.sh "$S1" "$S2" || { echo "pG step failed"; exit 1; }

echo "$(date +%H:%M) [4/4] pLS (LS-ADLC), seeds $S1-$S2"
bash scripts/run_pLS.sh "$S1" "$S2" || { echo "pLS step failed"; exit 1; }

echo ALL_DONE
