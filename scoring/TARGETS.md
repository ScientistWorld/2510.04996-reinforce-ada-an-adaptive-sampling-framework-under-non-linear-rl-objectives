# Evaluation Targets

## Primary: synthetic_signal_recovery
Tests whether a verifier-trained policy improves held-out pass rate on a non-trivial binary-reward prompt distribution.
- **Metric**: `eval_pass1` (higher is better)
- **Run**: `bash scripts/reproduce.sh && bash scripts/evaluate.sh checkpoints/all_results`

## Constraint: synthetic_signal_recovery
Tests whether gains preserve the paper's intended training-cost constraint instead of using a larger policy-update group.
- **Metric**: `update_samples_per_prompt` (lower is better)
- **Run**: `bash scripts/evaluate.sh checkpoints/all_results`

## Constraint: synthetic_signal_recovery
Tests whether sample efficiency remains visible in verifier-call cost.
- **Metric**: `avg_generated_per_prompt` (lower is better when accuracy is comparable)
- **Run**: `bash scripts/evaluate.sh checkpoints/all_results`

## Ablation: synthetic_signal_recovery
Tests whether the method reduces the zero-variance reward-group failure mode that motivates the paper.
- **Metric**: `signal_collapse_rate` (lower is better)
- **Run**: `bash scripts/evaluate.sh checkpoints/all_results`
