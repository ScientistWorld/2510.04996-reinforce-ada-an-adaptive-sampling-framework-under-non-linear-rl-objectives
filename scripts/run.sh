#!/bin/bash
# Job 1 of the hardening iteration: build the problem, calibrate the shared
# learning rate / step count, and smoke-test the whole pipeline at low steps.
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -V
python -c "import numpy; print('numpy', numpy.__version__)"
echo "cpus: $(nproc)"

echo "=== build problem"
time python eval/test/build_problem.py

echo "=== trainer smoke"
for m in seq_balance seq_pos; do EXPERIMENT=seqverify_hard SEED=9 STEPS=3 RESULTS_DIR=checkpoints/smoke bash scripts/method.sh $m; done
for b in grpo_n4 grpo_n16; do EXPERIMENT=seqverify_main SEED=9 STEPS=3 RESULTS_DIR=checkpoints/smoke bash scripts/baseline.sh $b; done
rm -rf checkpoints/smoke

echo "=== calibration"
time python hardening/calibrate.py 2>&1 | tee hardening/calibration/log.txt

echo "=== pipeline smoke (20 steps)"
STEPS=20 bash scripts/reproduce.sh > /dev/null
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
AUDIT_STEPS=20 AUDIT_SKIP="bruteforce_search_sft_n64 bruteforce_search_sft_n2048" python hardening/audit.py
python hardening/analyze.py || echo "analyze failed"
python validate.py || echo "validate failed (expected before final scores)"
