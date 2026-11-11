#!/bin/bash
# Train the paper's method (Reinforce-Ada-Seq) on one experiment and seed.
#
# Usage:  bash scripts/method.sh [seq_balance|seq_pos]
# Env:    EXPERIMENT (seqverify_main|seqverify_hard), SEED, STEPS, RESULTS_DIR
set -e
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
variant="${1:-seq_balance}"
python method/reinforce_ada.py --variant "$variant" \
  --experiment "${EXPERIMENT:-seqverify_main}" --seed "${SEED:-1}" \
  ${STEPS:+--steps "$STEPS"} --output-dir "${RESULTS_DIR:-checkpoints/all_results}"
