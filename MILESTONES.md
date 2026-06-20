# Reproduction Milestones

**Current: method_runs**

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

## Stop Justification
- Not stopping at `method_runs`; a fresh job is being submitted to restore `core_claim` after audit fixes.
