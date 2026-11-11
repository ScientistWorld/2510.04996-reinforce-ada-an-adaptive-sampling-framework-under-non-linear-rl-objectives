#!/bin/bash
# Train a fixed-group GRPO baseline on one experiment and seed.
#
# Usage:  bash scripts/baseline.sh grpo_n4|grpo_n8|grpo_n16
# Env:    EXPERIMENT (seqverify_main|seqverify_hard), SEED, STEPS, RESULTS_DIR
set -e
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
name="${1:-grpo_n4}"
case "$name" in
  grpo_n4|grpo_n8|grpo_n16) group_size="${name#grpo_n}" ;;
  *) echo "Unknown baseline: $name" >&2; exit 2 ;;
esac
python baseline/grpo.py --group-size "$group_size" \
  --experiment "${EXPERIMENT:-seqverify_main}" --seed "${SEED:-1}" \
  ${STEPS:+--steps "$STEPS"} --output-dir "${RESULTS_DIR:-checkpoints/all_results}"
