#!/bin/bash
# Reproduce the reference results end to end: train GRPO-n4, GRPO-n16,
# Reinforce-Ada-Seq-Pos and Reinforce-Ada-Seq-Balance on both experiments with
# five seeds each (the hidden test scores five runs per method and experiment),
# then score the visible split into scoring/scores.json.
#
# Env: SEEDS (default "1 2 3 4 5"), EXPERIMENTS, METHODS, STEPS, PARALLEL
set -e
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export RESULTS_DIR="${RESULTS_DIR:-checkpoints/all_results}"
SEEDS="${SEEDS:-1 2 3 4 5}"
EXPERIMENTS="${EXPERIMENTS:-seqverify_main seqverify_hard}"
METHODS="${METHODS:-grpo_n4 grpo_n16 seq_pos seq_balance}"
PARALLEL="${PARALLEL:-$(nproc)}"

rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
for experiment in $EXPERIMENTS; do
  for seed in $SEEDS; do
    for m in $METHODS; do
      echo "$m $experiment $seed"
    done
  done
done | xargs -P "$PARALLEL" -L 1 bash -c '
  m=$0; export EXPERIMENT=$1 SEED=$2
  case "$m" in
    grpo_*) bash scripts/baseline.sh "$m" ;;
    *) bash scripts/method.sh "$m" ;;
  esac'
bash scripts/evaluate.sh "$RESULTS_DIR"
