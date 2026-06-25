# Verification Report

## Initial Assessment

The workspace entered final verification at milestone `core_claim`. The implementation was a scaled reproduction of the paper's adaptive sampling method on a synthetic binary-verifier RL benchmark, with reusable baseline, method, data, and evaluation scripts. The claimed milestone was plausible, but the workspace needed a final consistency check because recent code changes had altered the method update order and an attempted hard-setting expansion had not produced checked-in scores.

## Issues Found

- Critical, fixed: `scores.json` did not include the attempted `hard_signal_recovery` expansion, while `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, `scoring/reference.json`, and `scripts/reproduce.sh` still advertised or generated that extra experiment.
- Critical, fixed: the checked-in scores needed to be refreshed after the method and baseline were changed to sample a whole batch from a frozen policy snapshot and apply one accumulated batch gradient.
- Minor, fixed: the previous verification report and progress notes referenced an older final-verification job and older scores.
- Residual limitation: the gym validates the core claim on a controlled synthetic benchmark, not the paper's full LLM math-reasoning setup.

## Fixes Applied

- Narrowed the public gym target back to the verified `synthetic_signal_recovery` experiment by aligning `scripts/reproduce.sh`, `scoring/reference.json`, `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, and `briefing/evaluation.md`.
- Submitted CPU verification job `f7ca695b-4f2`, which regenerated `scoring/scores.json`, `scoring/scores_train.json`, and `scoring/scores_test.json` from the current implementation.
- Removed generated job logs from the repository after reviewing them; `jobs/` remains ignored and is not part of the gym deliverable.
- Updated `PROGRESS.md` and `MILESTONES.md` with the final refreshed results.

## Final Judgment

The workspace is ready to serve as a research gym in both improve mode and from-scratch mode. The method-agnostic files describe the task and metrics without exposing the paper's algorithm, `eval/` reads method artifacts without importing `method/`, train/test evaluators are independent and method-enumerating, and `python validate.py --compare` passes.

Final reproduced core result: Reinforce-Ada Seq-Balance reaches `eval_pass1=0.8081` versus GRPO-n4 `0.7633` and GRPO-n16 `0.7869`, while keeping `update_samples_per_prompt=4`.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: synthetic benchmark rather than the original full-scale LLM math-reasoning experiments
