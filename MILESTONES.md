# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

### [2026-06-25 04:27] - core_claim
- Final verification job `1ba43ce0-e13` completed successfully and refreshed canonical, train, and test scores across seeds 17, 23, and 31.
- `python validate.py` and `python validate.py --compare` pass after the refreshed run.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8144` versus GRPO-n4 `0.7692` and GRPO-n16 `0.7870` while keeping the policy-update group size fixed at 4 samples.
- The expanded evaluator now includes hard/easy held-out slices and train/eval gap metrics, and train/test score files use matching method keys.

## Stop Justification
- Completed at milestone `core_claim`.
