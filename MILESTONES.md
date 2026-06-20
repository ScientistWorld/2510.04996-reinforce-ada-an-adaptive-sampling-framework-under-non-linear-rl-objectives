# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-20 10:50] - core_claim
- Implemented and validated a scaled binary-verifier reproduction of Reinforce-Ada sequential adaptive sampling.
- Reproduced the core claim on the controlled benchmark: adaptive sampling improves held-out pass rate over GRPO-n4 while preserving the fixed 4-sample policy update group, and seq-balance exceeds GRPO-n16 pass rate with lower update cost.

## Stop Justification
- Minimum target `core_claim` is reached in the scaled environment. Higher milestones would require additional seeds/settings or a much larger LLM/verl reproduction.
