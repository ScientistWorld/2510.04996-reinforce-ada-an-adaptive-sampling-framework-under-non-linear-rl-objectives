# Scientist Constraints

## Environment

Use the provided synthetic binary-verifier prompt distribution from `data/contextual_bandit.py` with the same default train/eval sizes, action count, and seed unless explicitly reporting a separate non-comparable experiment. The benchmark is about sampling and credit assignment, not changing the task distribution.

## Policy Class

Use the same linear categorical policy over the provided prompt features and fixed action set. Replacing it with a larger neural model, an oracle value function, or direct access to held-out correct actions would measure extra modeling capacity rather than the sampling method.

## Training Cost

For comparable scores, keep `update_samples_per_prompt` at 4 for the proposed method family. Reporting higher values is allowed only if the score includes the constraint metric and is not claimed as a strict improvement over the fixed-update-cost setting.

## Evaluation

Evaluation must read artifacts written by a method run and must not regenerate training, inspect method internals, or import from `method/`. The eval split must not be used for training updates.

## Compliance Checklist

- [ ] Uses the default `data/contextual_bandit.py` benchmark configuration for comparable scores
- [ ] Uses the shared linear categorical policy over the provided prompt features
- [ ] Does not train on the eval split
- [ ] Reports `update_samples_per_prompt` and `avg_generated_per_prompt`
- [ ] Evaluates through `scripts/evaluate.sh` without importing `method/`
