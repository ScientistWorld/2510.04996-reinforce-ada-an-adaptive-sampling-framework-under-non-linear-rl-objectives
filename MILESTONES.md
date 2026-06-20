# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

## Stop Justification
- Completed at milestone `core_claim`. Higher milestones would require broader LLM/verl math runs or additional scaled settings; the current package is a usable research gym for the paper's central adaptive-sampling claim.
