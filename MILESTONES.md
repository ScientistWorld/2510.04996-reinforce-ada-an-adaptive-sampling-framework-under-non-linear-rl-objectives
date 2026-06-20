# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-20 10:45] - none
- Read the paper and identified Reinforce-Ada's adaptive sampling algorithms, baselines, metrics, and reported target tables.
- Implemented the scaled benchmark and scripts, but no compute job has completed yet.

### [2026-06-20 11:10] - none
- Audited the previous implementation and found that tabular per-prompt parameters could not improve held-out evaluation prompts.
- Fixed the benchmark to use prompt features and shared categorical policy weights while keeping the Reinforce-Ada sampling and weighting steps unchanged.

### [2026-06-20 11:35] - none
- Reviewed the failed CPU job report. No artifacts were written.
- Fixed the scoring reference to declare every method emitted by the reproduction script, preventing `validate.py --compare` from rejecting valid generated scores.

### [2026-06-20 11:50] - none
- Read the retry job logs from `jobs/5fd284a6-05e`; execution failed before training with `ModuleNotFoundError: No module named 'data'`.
- Fixed script entrypoints to export the workspace root on `PYTHONPATH` before invoking baseline, method, or eval Python modules.

### [2026-06-20 12:00] - core_claim
- Ran the full scaled reproduction successfully in CPU job `0bb02000-a55`.
- Reinforce-Ada seq-balance reached `eval_pass1=0.8269`, above GRPO-n4 at `0.8048` with the same `update_samples_per_prompt=4`.
- The evaluator wrote `scoring/scores.json`, and `python validate.py --compare` passed inside the container.

## Stop Justification
- Stopping at `core_claim`: the actual Reinforce-Ada adaptive sampling algorithm runs end to end on a non-trivial controlled verifier task and demonstrates the core accuracy-vs-update-cost claim. Larger LLM math runs remain out of scope for the current budget.
