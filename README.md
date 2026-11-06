# Reinforce-Ada: An Adaptive Sampling Framework under Non-linear RL Objectives

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2510.04996"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2510.04996-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2510.04996](https://arxiv.org/abs/2510.04996) |
| **Domain** | Reinforcement learning for language-model reasoning |
| **TL;DR** | The paper reallocates verifier-sampling budget across prompts to recover zero-variance RL signal that fixed small response groups miss. |

## Summary

Small-group RL methods such as GRPO often see all-correct or all-incorrect verifier rewards for a prompt, which makes group-normalized advantages vanish. Reinforce-Ada derives adaptive sampling from non-linear pass-rate objectives and implements it with estimation-based and sequential sampling variants. The reported experiments show better math-reasoning accuracy than GRPO at comparable update cost, especially on hard prompts.

## Problem

How can reinforcement learning for verifier-scored reasoning tasks obtain useful policy-gradient signal when each prompt is expensive to sample and binary rewards are often all correct or all incorrect within a small response group?

## The paper's method

The paper introduces Reinforce-Ada, an adaptive sampling framework for RL with binary verifier rewards under non-linear objectives. It addresses signal collapse in small fixed response groups by spending generation budget where prompt-level signal is scarce.

## Key results of the paper

- On Qwen2.5-Math-1.5B, the paper reports weighted average accuracy of 47.6 for Reinforce-Ada-Seq-Balance versus 45.3 for GRPO-n4 and 47.3 for GRPO-n16.
- On the hard prompt subset, Seq-Balance improves the reported weighted average from 41.9 to 45.5.
- The scaled reproduction implements the same sequential sampling, downsampling, global-baseline, and inverse-pass-rate weighting steps on a controlled verifier RL benchmark.

## Run it

```bash
pip install sciworld
sw research 2510.04996 --mode innovate --agent claude-code
sw evaluate 2510.04996
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

