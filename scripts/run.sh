#!/bin/bash
# Reproduce the reference end to end: train GRPO-n4, GRPO-n16, Reinforce-Ada-
# Seq-Pos and Reinforce-Ada-Seq-Balance (5 seeds x 2 experiments), then score
# the visible split (scores.json, scores_train.json) and the hidden test
# (scores_test.json), and validate the workspace.
#
# The hardening audit (variants, brute force, shortcuts through the hidden
# test) is reproducible with:
#   rm -rf checkpoints/reproduction checkpoints/hardening checkpoints/hardening_staged hardening/audit
#   cp -r checkpoints/all_results checkpoints/reproduction
#   python hardening/audit.py && python hardening/analyze.py
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PARALLEL="${PARALLEL:-16}" bash scripts/reproduce.sh > /dev/null
bash scripts/evaluate_train.sh > /dev/null
bash scripts/evaluate_test.sh
python validate.py
