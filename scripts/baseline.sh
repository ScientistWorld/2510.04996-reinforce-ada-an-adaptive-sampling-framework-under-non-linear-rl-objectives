#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

name="${1:-grpo_n4}"
steps="${STEPS:-220}"
seed="${SEED:-17}"
out_dir="${RESULTS_DIR:-checkpoints/all_results}"
dataset_name="${DATASET_NAME:-mixed_difficulty}"
experiment="${EXPERIMENT:-synthetic_signal_recovery}"

case "$name" in
  grpo_n4)
    python baseline/grpo.py --group-size 4 --steps "$steps" --seed "$seed" --dataset-name "$dataset_name" --experiment "$experiment" --output-dir "$out_dir"
    ;;
  grpo_n16)
    python baseline/grpo.py --group-size 16 --steps "$steps" --seed "$seed" --dataset-name "$dataset_name" --experiment "$experiment" --output-dir "$out_dir"
    ;;
  *)
    echo "Unknown baseline: $name" >&2
    exit 2
    ;;
esac
