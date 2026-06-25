# Research Direction

## Research Problem

Improve reinforcement learning from binary verifier feedback when small response groups frequently produce no reward variance. The scientific question is how to allocate sampling and construct policy-gradient updates so sparse prompt-level signal is recovered efficiently.

## Core Contribution

The relevant contribution area is sample-efficient policy-gradient learning under sparse binary rewards. Improvements should address how limited verifier calls are spent, how reward statistics are estimated, or how fixed-size update groups preserve useful learning signal.

## Approach Scope

Work within policy-gradient methods for binary verifier rewards. Changes may alter how candidate actions are sampled, how reward statistics are used, or how update groups are formed, as long as they preserve the benchmark task and cost accounting.

## Out of Bounds

- Switching to supervised learning on correct actions instead of learning from sampled verifier rewards.
- Using an oracle that exposes the correct action except through sampled rewards.
- Improving scores primarily by increasing policy class capacity, training on the evaluation split, or expanding the update group size without treating it as a cost.
- Reframing the benchmark as a planning/search problem where the policy-gradient signal recovery issue no longer exists.
