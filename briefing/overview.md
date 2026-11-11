# Overview

- **Paper ID:** 2510.04996
- **Title:** Reinforce-Ada: An Adaptive Sampling Framework under Non-linear RL Objectives
- **Domain:** Reinforcement learning for language-model reasoning
- **TL;DR:** The paper reallocates verifier-sampling budget across prompts to recover zero-variance RL signal that fixed small response groups miss.

## Short Summary

Small-group RL methods such as GRPO often see all-correct or all-incorrect verifier rewards for a prompt, which makes group-normalized advantages vanish. Reinforce-Ada derives adaptive sampling from non-linear pass-rate objectives and implements it with estimation-based and sequential sampling variants. The reported experiments show better math-reasoning accuracy than GRPO at comparable update cost, especially on hard prompts.

## Key Results

- On Qwen2.5-Math-1.5B, the paper reports weighted average accuracy of 47.6 for Reinforce-Ada-Seq-Balance versus 45.3 for GRPO-n4 and 47.3 for GRPO-n16.
- On the hard prompt subset, Seq-Balance improves the reported weighted average from 41.9 to 45.5.
- This gym reproduces the method on a sequence-response binary-verifier task (8-token responses over 16 symbols, hashed training answers, hidden evaluation prompts).
  - Seq-Balance reaches held-out greedy accuracy of 0.486 (main) and 0.507 (hard training set).
  - GRPO-n4 reaches 0.416 and 0.430; GRPO-n16 reaches 0.451 and 0.478.
  - Seq-Balance uses 4 update responses per prompt and about 12–13 generated responses per prompt per step; GRPO-n16 uses 16 of each.
