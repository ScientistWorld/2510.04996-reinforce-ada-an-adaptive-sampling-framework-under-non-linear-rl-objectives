#!/bin/bash

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

python -V
python -c "import numpy; print('numpy', numpy.__version__)"
bash scripts/evaluate_train.sh checkpoints/all_results
bash scripts/evaluate_test.sh checkpoints/all_results
