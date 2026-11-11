# Evaluation

## What Is Scored

Each method is scored on two experiments that share one evaluation prompt set and differ in their training prompts:

- `seqverify_main` (weight 0.6): training prompts of moderate difficulty for the base policy.
- `seqverify_hard` (weight 0.4): training prompts that are hard for the base policy.

Scored metrics (all higher is better) are computed from the trained policy weights alone:

| Metric | Coefficient | Meaning |
|---|---:|---|
| `greedy_acc` | 1.0 | Fraction of evaluation prompts whose most likely response is correct |
| `hard_greedy_acc` | 0.5 | The same on the 40% of evaluation prompts with the lowest base-policy pass rate |
| `easy_greedy_acc` | 0.25 | The same on the 25% of evaluation prompts with the highest base-policy pass rate (constraint: do not sacrifice easy prompts) |

Greedy accuracy is Pass@1 in the limit of zero sampling temperature, so rescaling the weights cannot change it. `pass1` (temperature-1 Pass@1, the estimator the paper reports) is written as an unscored diagnostic. `scripts/evaluate.sh` also summarises three self-reported training-cost diagnostics from each run's history: verifier calls per prompt, update responses per prompt and zero-variance update groups. They are unscored, because the evaluator cannot check them.

## Splits

- `scripts/evaluate_train.sh` scores 1000 visible evaluation prompts (answers stored only as verifier hashes; greedy responses are verified exactly; `pass1` is an Ave@32 estimate) and writes `scoring/scores_train.json`.
- `scripts/evaluate_test.sh` (no arguments) scores a hidden set of 4000 prompts from the same distribution with exact metrics, and writes `scoring/scores_test.json`. Only this file is ranked.

## Artifact Contract

Write one JSON file per training run into `checkpoints/all_results/`:

```json
{"method": "<name>", "experiment": "seqverify_main" | "seqverify_hard", "seed": 1, "final_weights": [[[...]]]}
```

`final_weights` must have shape (8, 16, 16). Any other fields (`history`, `config`, ...) are ignored by the scorers, except for the unscored diagnostics in `scoring/scores.json`. Every method found in the directory is scored on both experiments.

- Each method must supply **5 distinct runs per experiment**.
- A run whose weights are missing, of the wrong shape, non-finite, or produce non-finite logits counts as failed, and contributes 0 to every metric.
- Missing runs, and exact duplicates of another run, are filled with 0.
- Artifacts naming an unknown experiment are ignored, so skipping an experiment scores 0 there.
- Method names must not start with `_`.

## Reference Numbers (hidden test, mean of 5 runs)

| Method | main greedy | main hard | main easy | hard-set greedy | hard-set hard | hard-set easy |
|---|---:|---:|---:|---:|---:|---:|
| base policy (no training) | 0.148 | 0.000 | 0.559 | 0.148 | 0.000 | 0.559 |
| GRPO, 4 responses per prompt | 0.416 | 0.208 | 0.703 | 0.430 | 0.252 | 0.677 |
| GRPO, 16 responses per prompt | 0.451 | 0.232 | 0.740 | 0.478 | 0.289 | 0.738 |
| reference (the paper's method) | 0.486 | 0.249 | 0.792 | 0.507 | 0.289 | 0.811 |

Seed-to-seed spread of the reference: greedy accuracy range 0.011 (main) and 0.005 (hard) across 5 seeds. Runs that share a seed see the same prompt batches, so comparisons between methods are paired.

## Running

`bash scripts/reproduce.sh` trains the reference and the baselines (5 seeds × 2 experiments) into `checkpoints/all_results/`. Then run `bash scripts/evaluate_train.sh` and `bash scripts/evaluate_test.sh`. The evaluators never import `method/`.
