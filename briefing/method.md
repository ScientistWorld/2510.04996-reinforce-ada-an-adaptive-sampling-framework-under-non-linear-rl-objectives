# Paper's Method

## Key Contribution

The paper introduces Reinforce-Ada, an adaptive sampling framework for RL with binary verifier rewards under non-linear objectives. It addresses signal collapse in small fixed response groups by spending generation budget where prompt-level signal is scarce.

## Approach

The theoretical framework optimizes a transformed pass-rate objective, such as `log p`, whose gradient weights difficult prompts by the derivative of the transform. The paper realizes this weighting through adaptive response allocation.

Reinforce-Ada-Seq (Algorithm 2) samples prompts in sequential rounds. A prompt remains active until an exit condition is met:
- Seq-Pos exits once it has at least `K_pos` correct responses.
- Seq-Balance exits once it has at least `K_pos` correct and `K_neg` incorrect responses.
- Every prompt stops at `N_max` responses.

The variable response pool is then downsampled to a fixed update group of `n` responses, aiming for n/2 correct and n/2 incorrect. The full pool's mean `p_hat` is the prompt's baseline. Each kept response gets advantage `r - p_hat`, multiplied by the explicit log-objective weight `1 / p_hat`. The backward-pass batch therefore stays the size of small-group GRPO.

Reinforce-Ada-Est instead maintains online pass-rate estimates and allocates per-prompt budgets proportional to inverse square-root difficulty, with a residual explicit weight in the advantage.

## Implementation in This Gym (`method/reinforce_ada.py`)

- Paper settings: `N_max = 32` in 4 rounds of `M = 8`, update group `n = 4`, and `K_pos = K_neg = n/2 = 2`.
- Each batch is sampled from a frozen policy snapshot, followed by one on-policy Adam step. PPO ratio clipping is therefore inactive, as in a single update per batch.
- Shared with the GRPO baselines (`baseline/grpo.py`): learning rate 0.1, 10-step warm-up, entropy bonus 1e-4, 256 prompts per step and 600 steps. Batch-order streams are identical for every method.
- The reference is `reinforce_ada_seq_balance`. `reinforce_ada_seq_pos` is the positive-focused variant.
- Ada-Est is not implemented.

## Main Claims

Primary claim: adaptive sampling recovers learning signal lost by fixed small-group GRPO and improves math-reasoning RL performance under similar or lower training cost than uniformly increasing group size.

Secondary claims:
- the balanced sequential variant is more robust than positive-only sampling late in training;
- gains are larger on hard prompt subsets;
- adaptive sampling improves the reward/entropy trade-off and small-budget pass@k behavior.
