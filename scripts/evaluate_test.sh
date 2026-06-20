#!/bin/bash
# Held-out test-slice evaluator. Mirrors scripts/evaluate.sh's artifact contract:
# pass the same directory of method result JSON files and write a test score
# file. When train scores exist, the evaluator scores exactly that method set.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

results_dir="${1:-checkpoints/all_results}"
python eval/test/evaluate.py --results-dir "$results_dir" --scores scoring/scores_test.json --train-scores scoring/scores_train.json
