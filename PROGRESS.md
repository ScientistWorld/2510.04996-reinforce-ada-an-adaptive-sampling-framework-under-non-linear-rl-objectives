# Progress

## What Works

Implemented a scaled reproduction of Reinforce-Ada on a binary-verifier contextual bandit benchmark. The code includes fixed-group GRPO baselines and the paper's sequential adaptive sampling variants with active prompts, stopping conditions, global pass-rate baseline, balanced downsampling, fixed update group, and inverse-pass-rate weighting. The benchmark now uses shared feature-based policy weights, so training can generalize to held-out evaluation prompts.

## Results

CPU job `0bb02000-a55` completed successfully and wrote `scoring/scores.json`.

| Method | eval_pass1 | avg_generated_per_prompt | update_samples_per_prompt | signal_collapse_rate |
| --- | ---: | ---: | ---: | ---: |
| GRPO-n4 | 0.8048 | 4.0000 | 4.0000 | 0.6298 |
| GRPO-n16 | 0.8146 | 16.0000 | 16.0000 | 0.4151 |
| Reinforce-Ada seq-pos | 0.8187 | 4.9511 | 4.0000 | 0.5917 |
| Reinforce-Ada seq-balance | 0.8269 | 21.3750 | 4.0000 | 0.6624 |

This supports the core claim in the scaled environment: adaptive sampling improves held-out pass rate over fixed GRPO-n4 while keeping the policy update group at 4 samples. Seq-balance also exceeds GRPO-n16's pass rate while using 4 update samples instead of 16, at the cost of more verifier generations.

The synthetic experiment in `scoring/reference.json` is a calibrated scaled-target entry rather than a paper table: the paper does not report metrics on this controlled bandit benchmark. The paper-reported LLM/math tables remain separately recorded in the same reference file.

## Remaining

Higher milestones would require expanding beyond the controlled single benchmark: additional difficulty settings, more seeds, or a scaled verl/LLM math run. The current package is usable as a research gym for the paper's adaptive sampling problem.

## Issues

Audit found and fixed one issue in the previous implementation: a tabular per-prompt policy could not improve held-out prompts. The environment now uses prompt features and shared policy weights.

The first CPU reproduction job exited with code 1. Local audit found a scoring-contract issue: `scripts/reproduce.sh` writes `grpo_n16` and `reinforce_ada_seq_pos` artifacts, while `scoring/reference.json` only declared `grpo_n4` and `reinforce_ada_seq_balance` for the synthetic experiment. `validate.py --compare` rejects methods absent from the reference, so the reference has been updated to include every method emitted by the evaluator.

The retry CPU job also exited with code 1. Its container log showed `ModuleNotFoundError: No module named 'data'` when executing `python baseline/grpo.py`; Python put `/home/user/baseline` rather than the workspace root on `sys.path`. All shell entrypoints now export `PYTHONPATH=/home/user:$PYTHONPATH` before running Python.

The original paper trains 1.5B-7B LLMs with verl on large math datasets, which is too expensive for an initial reproduction in this workspace. The current environment preserves the algorithmic steps on a controlled small RL task.

## Deviations from Paper

The paper uses LLM math-reasoning prompts, verifier-based rewards, and verl RL training on multi-GPU hardware. This reproduction uses a synthetic finite-action verifier environment to make the experiment runnable within the available budget. The method itself is not replaced: the implementation follows the paper's adaptive sampling, stopping, downsampling, baseline, and weighting computations.
