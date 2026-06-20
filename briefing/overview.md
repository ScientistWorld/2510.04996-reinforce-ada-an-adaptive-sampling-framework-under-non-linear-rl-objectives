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
- The scaled reproduction implements the same sequential sampling, downsampling, global-baseline, and inverse-pass-rate weighting steps on a controlled verifier RL benchmark.
