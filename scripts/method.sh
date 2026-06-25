#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

variant="${1:-seq_balance}"
steps="${STEPS:-220}"
seed="${SEED:-17}"
out_dir="${RESULTS_DIR:-checkpoints/all_results}"
dataset_name="${DATASET_NAME:-mixed_difficulty}"
experiment="${EXPERIMENT:-synthetic_signal_recovery}"

python method/reinforce_ada.py --variant "$variant" --steps "$steps" --seed "$seed" --dataset-name "$dataset_name" --experiment "$experiment" --output-dir "$out_dir"
bash scripts/evaluate.sh "$out_dir"
