#!/bin/bash
# Visible-split scorer: every method artifact in checkpoints/all_results is
# scored on the visible dev prompts (Ave@32 against verifier hashes) and the
# results are written to scoring/scores_train.json.
#
# Usage: bash scripts/evaluate_train.sh [results_dir]
set -e
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
mkdir -p scoring
python eval/train/evaluate.py --results-dir "${1:-checkpoints/all_results}" --scores scoring/scores_train.json
