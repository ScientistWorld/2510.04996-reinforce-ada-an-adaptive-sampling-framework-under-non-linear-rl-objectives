# Problem Definition

## Research Question

How should a reinforcement-learning loop spend a limited sampling budget so that a policy trained only from binary verifier feedback learns as much as possible, when small groups of sampled responses are often all correct or all incorrect and therefore carry no learning signal?

## The Task in This Gym

The gym is a scaled stand-in for RL on verifier-scored reasoning prompts:

- A **prompt** is a feature vector `x` (16 dimensions).
- A **response** is a sequence of 8 tokens, each one of 16 symbols (16^8, about 4.3e9 possible responses per prompt). For every prompt, exactly one response is correct.
- The **policy class** is fixed: a factorized linear-softmax policy `pi_W(r | x) = prod_l softmax(x @ W[l])[r_l]`, with `W` of shape (8, 16, 16). A fixed set of base-policy weights (the "base model") is the starting point of every run.
- The **verifier** returns 1 only for the correct response. For training prompts, the answers exist only as salted, iterated hashes, so a learner finds out what is correct only by sampling responses and checking them.
- **Training prompts** were curated from the base policy with a difficulty filter: prompts that the base policy solves in 1–6 of 16 samples (main set), or in 1–2 of 16 (hard set). Evaluation prompts are unfiltered draws from the same prompt distribution, so they mix easy, hard and practically unsolved prompts.

## Why It Matters

Generating and verifying responses is the dominant cost of RL for reasoning models, and the backward pass grows with the number of responses used per update. With few responses per prompt, many groups have zero reward variance and contribute no gradient. With many, the signal is clear but the cost scales linearly. The open question is how to use a fixed budget well.

## Success Criteria

At a fixed training budget, a successful method raises the fraction of held-out prompts the trained policy answers correctly. It should help prompts that start out hard, without giving up prompts that start out easy. The budget is fixed: the number of updates, the prompts per update, the responses sampled per prompt per update and the responses used per prompt in each update (see `scoring/CONSTRAINTS.md`).
