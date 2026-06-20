# Problem Definition

## Research Question

How can reinforcement learning for verifier-scored reasoning tasks obtain useful policy-gradient signal when each prompt is expensive to sample and binary rewards are often all correct or all incorrect within a small response group?

## Why It Matters

Small response groups are attractive because generation and training are costly. However, when a group has no reward variation, common group-normalized policy-gradient updates vanish. A successful solution should improve learning on difficult and saturated prompts without simply multiplying the training batch size.

## Success Criteria

A successful system should increase held-out verifier pass rate compared with a fixed small-group baseline, recover useful signal from prompts that would otherwise produce zero-variance groups, and keep the number of samples used for the policy update fixed so gains do not come from a larger backward pass.
