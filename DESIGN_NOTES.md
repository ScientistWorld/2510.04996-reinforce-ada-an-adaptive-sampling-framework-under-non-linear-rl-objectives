# Design Notes

## Checked

- Compared `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`; train calls `eval/train/evaluate.py` and writes `scoring/scores_train.json`, test calls `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Checked that both train and test evaluators read method result JSON artifacts from `checkpoints/all_results` instead of importing or rerunning `method/`.
- Checked for train/test cross-imports; the evaluators duplicate their slice logic and only import neutral data utilities.
- Checked that no `scoring/scores_train.json` or `scoring/scores_test.json` stubs exist.
- Ran `bash -n` on both train/test scripts.
- Validated `scoring/reference.json` parses and that every active metric has `higher_is_better`, `coefficient`, and `description`.

## Changed

- Added `hard_eval_pass1` to both slices, computed on the initially hardest prompts within each slice. This directly probes the paper's signal-recovery claim and makes visible-slice tuning less likely to transfer by accident.
- Added `easy_eval_pass1` to both slices, computed on the initially easiest prompts within each slice. This catches methods that improve hard prompts by sacrificing broad prompt coverage.
- Added `train_eval_gap` to both slices, penalizing methods whose train-prompt pass rate exceeds held-out pass rate. This gives the test slice a way to expose prompt memorization or train-distribution overfit.
- Reweighted `scoring/reference.json` so the train/test harness gives all active weight to `synthetic_signal_recovery`, the only experiment the train/test scripts actually compute.
- Set archival paper-table experiment weights and metric coefficients to zero while leaving their reported results untouched.
- Added missing metric descriptions throughout `scoring/reference.json`.

## Runtime Concerns

- The new metrics are artifact-only and should be lightweight, but the next runner should confirm the method artifacts always include `final_weights`, `history`, `seed`, `num_train`, and `num_eval`.
- The hard/easy subsets are selected separately inside train and test slices using initial-policy correct-action probability. With the default 160 eval prompts this gives moderate subset sizes; if future runs shrink `num_eval`, these metrics may become noisy.
- `scripts/evaluate.sh` and `eval/evaluate_results.py` were out of scope for this iteration and still emit only the legacy synthetic metrics. The train/test split is updated, but a later cleanup should decide whether the legacy visible `scores.json` path should mirror the expanded metric set.
