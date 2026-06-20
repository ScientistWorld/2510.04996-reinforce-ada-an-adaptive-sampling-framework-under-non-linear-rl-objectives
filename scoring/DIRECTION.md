# Research Direction

## Research Problem

Improve reinforcement learning from binary verifier feedback when small response groups frequently produce no reward variance. The scientific question is how to allocate sampling and construct policy-gradient updates so sparse prompt-level signal is recovered efficiently.

## Core Contribution

The relevant contribution area is adaptive response sampling and prompt-dependent weighting under non-linear pass-rate objectives. Improvements should address how samples are allocated across prompts, how prompt-level baselines are estimated, or how fixed-size update groups preserve recovered signal.

## Approach Scope

Work within policy-gradient methods for binary verifier rewards. Changes may alter stopping criteria, budget allocation rules, pass-rate estimators, downsampling rules, advantage weights, or combinations of adaptive sampling with group-normalized updates.

## Out of Bounds

- Switching to supervised learning on correct actions instead of learning from sampled verifier rewards.
- Using an oracle that exposes the correct action except through sampled rewards.
- Improving scores primarily by increasing policy class capacity, training on the evaluation split, or expanding the update group size without treating it as a cost.
- Reframing the benchmark as a planning/search problem where the policy-gradient signal recovery issue no longer exists.
