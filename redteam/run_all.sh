#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
for name in reference seed_17 seed_23 seed_31 quality_less_training quality_more_training quality_lower_lr baseline_grpo4 baseline_grpo16 baseline_seqpos control_initial_policy control_uniform shortcut_rebuild_answers shortcut_bruteforce_train shortcut_forged_trace shortcut_negative_trace shortcut_choose_test_size shortcut_replace_data shortcut_uniform_free shortcut_summary_fields shortcut_nan_weights shortcut_omit_method shortcut_unscored_experiment shortcut_empty_artifact shortcut_stale_scores shortcut_failed_policy_negative_trace shortcut_rebuild_eval_answers; do
    echo "Audit attempt: $name"
    bash "redteam/$name/run.sh"
done
python redteam/analyze.py
