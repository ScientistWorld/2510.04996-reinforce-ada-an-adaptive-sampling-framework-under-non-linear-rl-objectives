# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-20 10:50] - core_claim
- Implemented and validated a scaled binary-verifier reproduction of Reinforce-Ada sequential adaptive sampling.
- Reproduced the core claim on the controlled benchmark: adaptive sampling improves held-out pass rate over GRPO-n4 while preserving the fixed 4-sample policy update group, and seq-balance exceeds GRPO-n16 pass rate with lower update cost.

### [2026-06-20 11:15] - method_runs
- Continuation audit reset the milestone to `method_runs`.
- Found and fixed a scoring issue: the Reinforce-Ada artifact previously reported first-group zero-variance frequency as `signal_collapse_rate`; the scored metric now measures zero-variance in the actual update group after adaptive recovery/downsampling, with the old diagnostic kept as `initial_signal_collapse_rate`.

### [2026-06-20 11:25] - core_claim
- CPU job `566c8047-478` completed successfully in the container and ran `scripts/reproduce.sh` followed by `python validate.py --compare`.
- Corrected results support the scaled core claim: Reinforce-Ada seq-balance improves held-out pass rate over GRPO-n4 and GRPO-n16 while keeping a 4-sample update group and reducing post-recovery update-group signal collapse.

## Stop Justification
- Reached `core_claim` with a packaged, validated scaled reproduction. Higher milestones would require additional seeds/settings or a much larger LLM/verl reproduction.
