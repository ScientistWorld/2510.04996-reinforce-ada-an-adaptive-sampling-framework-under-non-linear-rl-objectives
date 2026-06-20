#!/bin/bash
# Visible train-slice evaluator. Mirrors scripts/evaluate.sh's artifact contract:
# pass a directory of method result JSON files, enumerate every method artifact,
# and write a score file with experiment and metric keys from reference.json.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

results_dir="${1:-checkpoints/all_results}"
python eval/train/evaluate.py --results-dir "$results_dir" --scores scoring/scores_train.json
