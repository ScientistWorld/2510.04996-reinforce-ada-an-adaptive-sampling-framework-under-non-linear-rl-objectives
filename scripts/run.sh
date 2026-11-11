#!/bin/bash
# Final hardening job: reproduce the reference and baselines from the final
# code, score train/test, then run the hardening audit (variants, brute force,
# shortcuts) through the unchanged hidden test and analyse ranks and noise.
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import numpy, os; print('numpy', numpy.__version__, 'cpu_count', os.cpu_count())"

echo "=== reproduce: 4 methods x 2 experiments x 5 seeds"
time PARALLEL=16 bash scripts/reproduce.sh > /dev/null
bash scripts/evaluate_train.sh > /dev/null
bash scripts/evaluate_test.sh
cat scoring/scores.json

echo "=== hardening audit"
rm -rf checkpoints/reproduction checkpoints/hardening checkpoints/hardening_staged hardening/audit
cp -r checkpoints/all_results checkpoints/reproduction
time python hardening/audit.py
python hardening/analyze.py | tee hardening/analysis.md
python validate.py || echo "validate: reference.json results are filled in after this job"
