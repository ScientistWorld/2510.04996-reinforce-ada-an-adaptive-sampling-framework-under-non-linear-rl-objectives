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

---

# Problem Definition

## Research Question

How can reinforcement learning for verifier-scored reasoning tasks obtain useful policy-gradient signal when each prompt is expensive to sample and binary rewards are often all correct or all incorrect within a small response group?

## Why It Matters

Small response groups are attractive because generation and training are costly. However, when a group has no reward variation, common group-normalized policy-gradient updates vanish. A successful solution should improve learning on difficult and saturated prompts without simply multiplying the training batch size.

## Success Criteria

A successful system should increase held-out verifier pass rate compared with a fixed small-group baseline, recover useful signal from prompts that would otherwise produce zero-variance groups, and keep the number of samples used for the policy update fixed so gains do not come from a larger backward pass.

---

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

---

# Evaluation

## Metrics

- `eval_pass1`: mean probability that the learned policy samples the verifier-correct action on held-out prompts. Higher is better.
- `signal_collapse_rate`: fraction of prompts whose first fixed-size reward group has zero variance. Lower is better.
- `avg_generated_per_prompt`: average number of verifier calls used per prompt. Lower is better when accuracy is comparable.
- `update_samples_per_prompt`: number of samples included in the policy update per prompt. This is a constraint metric; lower values mean lower training cost.

## Baselines and Targets

The paper reports math-reasoning accuracy targets for fixed-group GRPO and adaptive methods on Qwen2.5-Math, Llama, and Qwen3 backbones. The scaled reproduction uses a synthetic binary-verifier RL benchmark and compares fixed-group GRPO at `n=4` and `n=16` against methods that must keep `update_samples_per_prompt=4`.

## Evaluation Protocol

Run `bash scripts/reproduce.sh` to train all included methods on the same generated train/eval splits with the same random seed. Then run `bash scripts/evaluate.sh checkpoints/all_results` to score the JSON artifacts. Evaluation code must not import from `method/`; it only reads method outputs.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 0m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 1m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 0m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 3 total (1 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:45] - none
- Read the paper and identified Reinforce-Ada's adaptive sampling algorithms, baselines, metrics, and reported target tables.
- Implemented the scaled benchmark and scripts, but no compute job has completed yet.

### [2026-06-20 11:10] - none
- Audited the previous implementation and found that tabular per-prompt parameters could not improve held-out evaluation prompts.
- Fixed the benchmark to use prompt features and shared categorical policy weights while keeping the Reinforce-Ada sampling and weighting steps unchanged.

### [2026-06-20 11:35] - none
- Reviewed the failed CPU job report. No artifacts were written.
- Fixed the scoring reference to declare every method emitted by the reproduction script, preventing `validate.py --compare` from rejecting valid generated scores.

### [2026-06-20 11:50] - none
- Read the retry job logs from `jobs/5fd284a6-05e`; execution failed before training with `ModuleNotFoundError: No module named 'data'`.
- Fixed script entrypoints to export the workspace root on `PYTHONPATH` before invoking baseline, method, or eval Python modules.

### [2026-06-20 12:00] - core_claim
- Ran the full scaled reproduction successfully in CPU job `0bb02000-a55`.
- Reinforce-Ada seq-balance reached `eval_pass1=0.8269`, above GRPO-n4 at `0.8048` with the same `update_samples_per_prompt=4`.
- The evaluator wrote `scoring/scores.json`, and `python validate.py --compare` passed inside the container.

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
