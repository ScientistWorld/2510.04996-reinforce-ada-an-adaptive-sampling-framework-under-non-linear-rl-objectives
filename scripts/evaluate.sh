#!/bin/bash
# Score run artifacts on the visible dev prompts and write scoring/scores.json.
#
# Artifact contract (any method): one JSON file per training run in the
# results directory with fields
#   "method": name, "experiment": "seqverify_main" | "seqverify_hard",
#   "final_weights": (L, d, V) nested list = (8, 16, 16).
# Optional "history" rows feed unscored, self-reported cost diagnostics.
# This script only uses code in eval/; it never imports method/.
#
# Usage: bash scripts/evaluate.sh [results_dir]   (default checkpoints/all_results)
set -e
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
mkdir -p scoring
python eval/evaluate_results.py --results-dir "${1:-checkpoints/all_results}" --scores scoring/scores.json
