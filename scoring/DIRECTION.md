# Research Direction

## Research Problem

Improve reinforcement learning from binary verifier feedback when small response groups frequently produce no reward variance. The scientific question is how to allocate a fixed sampling budget and construct fixed-size policy-gradient updates so that sparse prompt-level signal is recovered efficiently.

## Core Contribution

The relevant contribution area is sample-efficient policy-gradient learning under sparse binary rewards. Improvements should address:
- how a limited number of verifier calls is spent across prompts and rounds;
- how reward statistics and pass rates are estimated and used to weight updates;
- how fixed-size update groups preserve useful learning signal.

## Approach Scope

Work within online policy-gradient methods for binary verifier rewards, under the fixed training budget in `scoring/CONSTRAINTS.md`. Changes may alter how candidate responses are sampled, how reward statistics are used, how advantages are formed and how update groups are selected.

## Out of Bounds

- Spending more budget than the protocol allows: more updates, larger batches or update groups, more than 32 sampled responses per prompt per update, or up-front answer search followed by supervised fitting.
- Inverting the answer hashes, enumerating the response space, or using evaluation prompts or answers for training.
- Changing the policy class, the base-policy starting point or the problem files.
- Selecting the best runs out of many seeds instead of reporting 5 independent runs.
