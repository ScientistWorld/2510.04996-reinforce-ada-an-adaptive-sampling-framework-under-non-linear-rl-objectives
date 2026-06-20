# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-20 10:50] - core_claim
- Implemented and validated a scaled binary-verifier reproduction of Reinforce-Ada sequential adaptive sampling.
- Reproduced the core claim on the controlled benchmark: adaptive sampling improves held-out pass rate over GRPO-n4 while preserving the fixed 4-sample policy update group, and seq-balance exceeds GRPO-n16 pass rate with lower update cost.

### [2026-06-20 11:15] - method_runs
- Continuation audit reset the milestone to `method_runs`.
- Found and fixed a scoring issue: the Reinforce-Ada artifact previously reported first-group zero-variance frequency as `signal_collapse_rate`; the scored metric now measures zero-variance in the actual update group after adaptive recovery/downsampling, with the old diagnostic kept as `initial_signal_collapse_rate`.

## Stop Justification
- Fresh corrected results are pending from the next reproduction job.
