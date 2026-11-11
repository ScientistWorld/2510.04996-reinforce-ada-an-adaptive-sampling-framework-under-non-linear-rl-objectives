# Evaluation Targets

All targets are computed by `scripts/evaluate_test.sh` (hidden) and `scripts/evaluate_train.sh` (visible) from the trained policy weights in `checkpoints/all_results/` (5 runs per method and experiment).

## Primary: seqverify_main
Held-out accuracy after training on curated moderate-difficulty prompts.
- **Metric**: `greedy_acc` (higher is better, coefficient 1.0)
- **Run**: `bash scripts/reproduce.sh && bash scripts/evaluate_test.sh`

## Primary: seqverify_hard
Held-out accuracy after training on prompts that are hard for the starting policy (signal is scarce).
- **Metric**: `greedy_acc` (higher is better, coefficient 1.0)
- **Run**: `bash scripts/reproduce.sh && bash scripts/evaluate_test.sh`

## Robustness: seqverify_main, seqverify_hard
Recovery of prompts that start out hardest (lowest base-policy pass rate; 40% of the evaluation prompts).
- **Metric**: `hard_greedy_acc` (higher is better, coefficient 0.5)
- **Run**: `bash scripts/evaluate_test.sh`

## Constraint: seqverify_main, seqverify_hard
Gains on hard prompts must not come from giving up prompts that start out easiest (top 25% by base-policy pass rate).
- **Metric**: `easy_greedy_acc` (higher is better, coefficient 0.25)
- **Run**: `bash scripts/evaluate_test.sh`

## Constraint (protocol, not measurable from outputs)
The training budget is fixed by `scoring/CONSTRAINTS.md`: 600 updates of 256 prompts, at most 32 sampled responses per prompt per update, and 4 responses per prompt in each update. The hidden test sees only final weights, so these limits are checked by review, not by a score. The self-reported diagnostics `avg_generated_per_prompt`, `update_samples_per_prompt` and `signal_collapse_rate` in `scoring/scores.json` document them (coefficient 0).
