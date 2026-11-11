#!/bin/bash
# Job 2 of the hardening iteration: calibration v2 (batch / lr / steps with
# greedy-accuracy metrics) and a quick evaluator smoke test.
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import numpy, os; print('numpy', numpy.__version__, 'cpu_count', os.cpu_count())"

echo "=== calibration v2"
time python hardening/calibrate.py 2>&1 | tee hardening/calibration/log_v2.txt

echo "=== evaluator smoke (10 steps)"
STEPS=10 bash scripts/reproduce.sh > /dev/null
bash scripts/evaluate_train.sh > /dev/null
bash scripts/evaluate_test.sh > /dev/null
python - << 'PY'
import json
for f in ["scores", "scores_train", "scores_test"]:
    d = json.load(open(f"scoring/{f}.json"))["experiments"]
    for e, v in d.items():
        for m, r in v["results"].items():
            print(f, e, m, {k: round(x, 4) for k, x in r.items()})
PY
rm -rf checkpoints/reproduction && cp -r checkpoints/all_results checkpoints/reproduction
AUDIT_STEPS=10 AUDIT_ONLY="reference shortcut_sharpen_reference_x2 shortcut_forged_history shortcut_nan_weights shortcut_stale_scores shortcut_unscored_experiment" python hardening/audit.py
