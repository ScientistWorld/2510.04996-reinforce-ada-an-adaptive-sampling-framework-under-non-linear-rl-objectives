# Progress

## What Works

Implemented a scaled reproduction of Reinforce-Ada on a binary-verifier contextual bandit benchmark. The code includes fixed-group GRPO baselines and the paper's sequential adaptive sampling variants with active prompts, stopping conditions, global pass-rate baseline, balanced downsampling, fixed update group, and inverse-pass-rate weighting. The benchmark now uses shared feature-based policy weights, so training can generalize to held-out evaluation prompts.

## Results

CPU job `1ba43ce0-e13` completed successfully after the final verification fixes and refreshed `scoring/scores.json`, `scoring/scores_train.json`, and `scoring/scores_test.json`. The job ran the full reproduction across seeds 17, 23, and 31, followed by `python validate.py --compare` and the train/test evaluators inside the container.

| Method | eval_pass1 | hard_eval_pass1 | easy_eval_pass1 | train_eval_gap | avg_generated_per_prompt | update_samples_per_prompt | signal_collapse_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO-n4 | 0.7692 | 0.6883 | 0.8365 | 0.0780 | 4.0000 | 4.0000 | 0.6246 |
| GRPO-n16 | 0.7870 | 0.7190 | 0.8392 | 0.0792 | 16.0000 | 16.0000 | 0.4187 |
| Reinforce-Ada seq-pos | 0.7889 | 0.7302 | 0.8482 | 0.0422 | 4.9001 | 4.0000 | 0.5242 |
| Reinforce-Ada seq-balance | 0.8144 | 0.7656 | 0.8572 | 0.0548 | 21.5102 | 4.0000 | 0.3732 |

This supports the core claim in the scaled environment: Seq-Balance adaptive sampling improves held-out pass rate over fixed GRPO-n4 while keeping the policy update group at 4 samples. Seq-Balance also exceeds GRPO-n16's pass rate while using 4 update samples instead of 16, improves the hard-prompt slice, and has lower zero-variance collapse than both fixed-group GRPO baselines, at the cost of more verifier generations. Seq-Pos is roughly tied with GRPO-n16 on aggregate held-out pass rate with far fewer generated samples and a smaller train/eval gap, while the stronger restored `core_claim` rests on Seq-Balance.

The synthetic experiment in `scoring/reference.json` is a calibrated scaled-target entry rather than a paper table: the paper does not report metrics on this controlled bandit benchmark. The paper-reported LLM/math tables remain separately recorded in the same reference file.

## Remaining

Final verification confirmed that the canonical evaluator reports the full synthetic target set (`hard_eval_pass1`, `easy_eval_pass1`, and `train_eval_gap`) and averages multiple seed artifacts per method. `python validate.py` and `python validate.py --compare` pass after the refreshed CPU job.

Higher milestones would require expanding beyond the controlled synthetic benchmark into additional scaled settings or a scaled verl/LLM math run. The current package is usable as a research gym for the paper's adaptive sampling problem.

## Issues

Final verification found the canonical scores stale after the evaluator was expanded with hard/easy/generalization metrics and seed aggregation. Job `1ba43ce0-e13` regenerated method artifacts and all score files under the expanded evaluator.

Current iteration tightened the evaluation contract after code review: methods now save final policy weights and per-step training traces, and `eval/` recomputes held-out pass rate plus aggregate cost/collapse metrics from those artifacts instead of trusting summary numbers emitted by the method. Job `2cc5ebd9-b02` regenerated artifacts and `scoring/scores.json` under this stricter evaluator.

Continuation audit found a scoring issue in the reported ablation metric: Reinforce-Ada was writing the zero-variance rate of the first small group before adaptive recovery as `signal_collapse_rate`. That diagnostic is useful, but it does not measure whether adaptive sampling recovered a non-zero update signal. The method now reports the scored `signal_collapse_rate` on the fixed update group after adaptive sampling/downsampling and keeps the old value as `initial_signal_collapse_rate`. Job `566c8047-478` regenerated `scoring/scores.json` with the corrected metric and passed `python validate.py --compare`.

Audit found and fixed one issue in the previous implementation: a tabular per-prompt policy could not improve held-out prompts. The environment now uses prompt features and shared policy weights.

The first CPU reproduction job exited with code 1. Local audit found a scoring-contract issue: `scripts/reproduce.sh` writes `grpo_n16` and `reinforce_ada_seq_pos` artifacts, while `scoring/reference.json` only declared `grpo_n4` and `reinforce_ada_seq_balance` for the synthetic experiment. `validate.py --compare` rejects methods absent from the reference, so the reference has been updated to include every method emitted by the evaluator.

The retry CPU job also exited with code 1. Its container log showed `ModuleNotFoundError: No module named 'data'` when executing `python baseline/grpo.py`; Python put `/home/user/baseline` rather than the workspace root on `sys.path`. All shell entrypoints now export `PYTHONPATH=/home/user:$PYTHONPATH` before running Python.

The original paper trains 1.5B-7B LLMs with verl on large math datasets, which is too expensive for an initial reproduction in this workspace. The current environment preserves the algorithmic steps on a controlled small RL task.

## Deviations from Paper

The paper uses LLM math-reasoning prompts, verifier-based rewards, and verl RL training on multi-GPU hardware. This reproduction uses a synthetic finite-action verifier environment to make the experiment runnable within the available budget. The method itself is not replaced: the implementation follows the paper's adaptive sampling, stopping, downsampling, baseline, and weighting computations.
