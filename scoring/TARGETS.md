# Evaluation Targets

## Primary: synthetic_signal_recovery
Tests whether a verifier-trained policy improves held-out pass rate on a non-trivial binary-reward prompt distribution.
- **Metric**: `eval_pass1` (higher is better)
- **Run**: `bash scripts/reproduce.sh && bash scripts/evaluate.sh checkpoints/all_results`

## Robustness: synthetic_signal_recovery
Tests whether gains apply to initially hard held-out prompts, where small uniform groups are most likely to lose signal.
- **Metric**: `hard_eval_pass1` (higher is better)
- **Run**: `bash scripts/evaluate.sh checkpoints/all_results`

## Constraint: synthetic_signal_recovery
Tests whether hard-prompt gains do not come from sacrificing initially easy held-out prompts.
- **Metric**: `easy_eval_pass1` (higher is better)
- **Run**: `bash scripts/evaluate.sh checkpoints/all_results`

## Constraint: synthetic_signal_recovery
Tests whether improvements generalize beyond memorizing the sampled training prompts.
- **Metric**: `train_eval_gap` (lower is better)
- **Run**: `bash scripts/evaluate.sh checkpoints/all_results`

## Constraint: synthetic_signal_recovery
Tests whether gains preserve the paper's intended training-cost constraint instead of using a larger policy-update group.
- **Metric**: `update_samples_per_prompt` (lower is better)
- **Run**: `bash scripts/evaluate.sh checkpoints/all_results`

## Constraint: synthetic_signal_recovery
Tests whether sample efficiency remains visible in verifier-call cost.
- **Metric**: `avg_generated_per_prompt` (lower is better when accuracy is comparable)
- **Run**: `bash scripts/evaluate.sh checkpoints/all_results`

## Ablation: synthetic_signal_recovery
Tests whether the method reduces the zero-variance reward-group failure mode in the actual fixed-size policy-update group.
- **Metric**: `signal_collapse_rate` (lower is better)
- **Run**: `bash scripts/evaluate.sh checkpoints/all_results`
