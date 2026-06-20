#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

variant="${1:-seq_balance}"
steps="${STEPS:-220}"
seed="${SEED:-17}"
out_dir="${RESULTS_DIR:-checkpoints/all_results}"

python method/reinforce_ada.py --variant "$variant" --steps "$steps" --seed "$seed" --output-dir "$out_dir"
bash scripts/evaluate.sh "$out_dir"
