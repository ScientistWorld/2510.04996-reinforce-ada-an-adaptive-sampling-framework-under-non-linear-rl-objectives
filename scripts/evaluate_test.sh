#!/bin/bash
# Hidden-test scorer. Runs with no arguments: scores every method artifact in
# checkpoints/all_results against the hidden prompts in eval/test/ and writes
# scoring/scores_test.json. Never trains; reads only method, experiment and
# final_weights from each artifact.
set -e
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
mkdir -p scoring
python eval/test/evaluate.py --results-dir "${1:-checkpoints/all_results}" --scores scoring/scores_test.json
