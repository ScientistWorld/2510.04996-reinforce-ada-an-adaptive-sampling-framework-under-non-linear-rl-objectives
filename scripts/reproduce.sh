#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

rm -rf checkpoints/all_results
mkdir -p checkpoints/all_results

export RESULTS_DIR=checkpoints/all_results
export STEPS="${STEPS:-220}"
export SEED="${SEED:-17}"

bash scripts/baseline.sh grpo_n4
bash scripts/baseline.sh grpo_n16
python method/reinforce_ada.py --variant seq_pos --steps "$STEPS" --seed "$SEED" --output-dir "$RESULTS_DIR"
python method/reinforce_ada.py --variant seq_balance --steps "$STEPS" --seed "$SEED" --output-dir "$RESULTS_DIR"
bash scripts/evaluate.sh "$RESULTS_DIR"
