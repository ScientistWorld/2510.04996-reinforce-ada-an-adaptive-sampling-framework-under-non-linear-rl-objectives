# Paper's Method

## Key Contribution

The paper introduces Reinforce-Ada, an adaptive sampling framework for RL with binary verifier rewards under non-linear objectives. It addresses signal collapse in small fixed response groups by spending generation budget where prompt-level signal is scarce.

## Approach

The theoretical framework optimizes a transformed pass-rate objective, such as `log p`, whose gradient weights difficult prompts by the derivative of the transform. The paper realizes this weighting through adaptive response allocation.

Reinforce-Ada-Seq samples prompts in sequential rounds. A prompt remains active until an exit condition is met, such as collecting enough positive responses or enough positive and negative responses. The variable response pool is then downsampled to a fixed update group, while the full pool provides a prompt-level baseline. The final update uses the fixed group with an explicit inverse-pass-rate weight so the backward-pass batch size stays comparable to small-group GRPO.

Reinforce-Ada-Est instead maintains online pass-rate estimates and allocates per-prompt budgets proportional to inverse square-root difficulty, with a residual explicit weight in the advantage.

## Main Claims

Primary claim: adaptive sampling recovers learning signal lost by fixed small-group GRPO and improves math-reasoning RL performance under similar or lower training cost than uniformly increasing group size.

Secondary claims: the balanced sequential variant is more robust than positive-only sampling late in training; gains are larger on hard prompt subsets; adaptive sampling improves the reward/entropy trade-off and small-budget pass@k behavior.
