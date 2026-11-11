#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
bash redteam/shortcut_failed_policy_negative_trace/run.sh
bash redteam/shortcut_rebuild_eval_answers/run.sh
python redteam/analyze.py
