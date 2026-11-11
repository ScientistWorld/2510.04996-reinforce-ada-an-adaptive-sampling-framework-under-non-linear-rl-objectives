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

---

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

---

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

---

# Evaluation

## What Is Scored

Each method is scored on two experiments that share one evaluation prompt set and differ in their training prompts:

- `seqverify_main` (weight 0.6): training prompts of moderate difficulty for the base policy.
- `seqverify_hard` (weight 0.4): training prompts that are hard for the base policy.

Scored metrics (all higher is better) are computed from the trained policy weights alone:

| Metric | Coefficient | Meaning |
|---|---:|---|
| `greedy_acc` | 1.0 | Fraction of evaluation prompts whose most likely response is correct |
| `hard_greedy_acc` | 0.5 | The same on the 40% of evaluation prompts with the lowest base-policy pass rate |
| `easy_greedy_acc` | 0.25 | The same on the 25% of evaluation prompts with the highest base-policy pass rate (constraint: do not sacrifice easy prompts) |

Greedy accuracy is Pass@1 in the limit of zero sampling temperature, so rescaling the weights cannot change it. `pass1` (temperature-1 Pass@1, the estimator the paper reports) is written as an unscored diagnostic. `scripts/evaluate.sh` also summarises three self-reported training-cost diagnostics from each run's history: verifier calls per prompt, update responses per prompt and zero-variance update groups. They are unscored, because the evaluator cannot check them.

## Splits

- `scripts/evaluate_train.sh` scores 1000 visible evaluation prompts (answers stored only as verifier hashes; greedy responses are verified exactly; `pass1` is an Ave@32 estimate) and writes `scoring/scores_train.json`.
- `scripts/evaluate_test.sh` (no arguments) scores a hidden set of 4000 prompts from the same distribution with exact metrics, and writes `scoring/scores_test.json`. Only this file is ranked.

## Artifact Contract

Write one JSON file per training run into `checkpoints/all_results/`:

```json
{"method": "<name>", "experiment": "seqverify_main" | "seqverify_hard", "seed": 1, "final_weights": [[[...]]]}
```

`final_weights` must have shape (8, 16, 16). Any other fields (`history`, `config`, ...) are ignored by the scorers, except for the unscored diagnostics in `scoring/scores.json`. Every method found in the directory is scored on both experiments.

- Each method must supply **5 distinct runs per experiment**.
- A run whose weights are missing, of the wrong shape, non-finite, or produce non-finite logits counts as failed, and contributes 0 to every metric.
- Missing runs, and exact duplicates of another run, are filled with 0.
- Artifacts naming an unknown experiment are ignored, so skipping an experiment scores 0 there.
- Method names must not start with `_`.

## Reference Numbers (hidden test, mean of 5 runs)

| Method | main greedy | main hard | main easy | hard-set greedy | hard-set hard | hard-set easy |
|---|---:|---:|---:|---:|---:|---:|
| base policy (no training) | 0.148 | 0.000 | 0.559 | 0.148 | 0.000 | 0.559 |
| GRPO, 4 responses per prompt | 0.416 | 0.208 | 0.703 | 0.430 | 0.252 | 0.677 |
| GRPO, 16 responses per prompt | 0.451 | 0.232 | 0.740 | 0.478 | 0.289 | 0.738 |
| reference (the paper's method) | 0.486 | 0.249 | 0.792 | 0.507 | 0.289 | 0.811 |

Seed-to-seed spread of the reference: greedy accuracy range 0.011 (main) and 0.005 (hard) across 5 seeds. Runs that share a seed see the same prompt batches, so comparisons between methods are paired.

## Running

`bash scripts/reproduce.sh` trains the reference and the baselines (5 seeds × 2 experiments) into `checkpoints/all_results/`. Then run `bash scripts/evaluate_train.sh` and `bash scripts/evaluate_test.sh`. The evaluators never import `method/`.

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

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:50] - core_claim
- Implemented and validated a scaled binary-verifier reproduction of Reinforce-Ada sequential adaptive sampling.
- Reproduced the core claim on the controlled benchmark: adaptive sampling improves held-out pass rate over GRPO-n4 while preserving the fixed 4-sample policy update group, and seq-balance exceeds GRPO-n16 pass rate with lower update cost.

### [2026-06-20 11:15] - method_runs
- Continuation audit reset the milestone to `method_runs`.
- Found and fixed a scoring issue: the Reinforce-Ada artifact previously reported first-group zero-variance frequency as `signal_collapse_rate`; the scored metric now measures zero-variance in the actual update group after adaptive recovery/downsampling, with the old diagnostic kept as `initial_signal_collapse_rate`.

### [2026-06-20 11:25] - core_claim
- CPU job `566c8047-478` completed successfully in the container and ran `scripts/reproduce.sh` followed by `python validate.py --compare`.
- Corrected results support the scaled core claim: Reinforce-Ada seq-balance improves held-out pass rate over GRPO-n4 and GRPO-n16 while keeping a 4-sample update group and reducing post-recovery update-group signal collapse.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 5 total (3 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 5 total (3 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 5 total (3 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: error
- **Working time**: 16m | **GPU**: 0.0h
- **Jobs**: 6 total (4 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 6 total (4 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 6 total (4 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 6 total (4 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 7 total (5 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

### [2026-06-25 04:27] - core_claim
- Final verification job `1ba43ce0-e13` completed successfully and refreshed canonical, train, and test scores across seeds 17, 23, and 31.
- `python validate.py` and `python validate.py --compare` pass after the refreshed run.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8144` versus GRPO-n4 `0.7692` and GRPO-n16 `0.7870` while keeping the policy-update group size fixed at 4 samples.
- The expanded evaluator now includes hard/easy held-out slices and train/eval gap metrics, and train/test score files use matching method keys.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 13m | **GPU**: 0.0h
- **Jobs**: 7 total (5 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

### [2026-06-25 04:27] - core_claim
- Final verification job `1ba43ce0-e13` completed successfully and refreshed canonical, train, and test scores across seeds 17, 23, and 31.
- `python validate.py` and `python validate.py --compare` pass after the refreshed run.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8144` versus GRPO-n4 `0.7692` and GRPO-n16 `0.7870` while keeping the policy-update group size fixed at 4 samples.
- The expanded evaluator now includes hard/easy held-out slices and train/eval gap metrics, and train/test score files use matching method keys.

### [2026-06-25 04:34] - core_claim
- Prepared a new expansion job for a hard initial-policy benchmark modeled after the paper's hard-prompt ablation.
- Tightened algorithm fidelity: GRPO and Reinforce-Ada now sample each batch under a frozen policy snapshot, construct all update groups, and apply one accumulated batch gradient.
- Extended canonical, train, and test evaluators to group artifacts by experiment and use each artifact's dataset name.
- Current milestone remains `core_claim` until the submitted job refreshes scores and confirms the hard setting.

### [2026-06-25 04:41] - core_claim
- Final verification found that the hard initial-policy expansion was not present in `scores.json`, so the public gym target was narrowed back to the verified `synthetic_signal_recovery` experiment.
- `scripts/reproduce.sh`, `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, and `scoring/reference.json` now match the checked-in score structure.
- A CPU verification job is being submitted to refresh canonical, train, and test scores from the current batch-snapshot implementation.

### [2026-06-25 04:44] - core_claim
- CPU verification job `f7ca695b-4f2` completed successfully.
- The job refreshed canonical, train, and test scores from the current batch-snapshot implementation and `python validate.py --compare` passed.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8081` versus GRPO-n4 `0.7633` and GRPO-n16 `0.7869` while keeping `update_samples_per_prompt=4`.
- Final verification judged the workspace ready as a synthetic `core_claim` research gym.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 9 total (7 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

### [2026-06-25 04:27] - core_claim
- Final verification job `1ba43ce0-e13` completed successfully and refreshed canonical, train, and test scores across seeds 17, 23, and 31.
- `python validate.py` and `python validate.py --compare` pass after the refreshed run.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8144` versus GRPO-n4 `0.7692` and GRPO-n16 `0.7870` while keeping the policy-update group size fixed at 4 samples.
- The expanded evaluator now includes hard/easy held-out slices and train/eval gap metrics, and train/test score files use matching method keys.

### [2026-06-25 04:34] - core_claim
- Prepared a new expansion job for a hard initial-policy benchmark modeled after the paper's hard-prompt ablation.
- Tightened algorithm fidelity: GRPO and Reinforce-Ada now sample each batch under a frozen policy snapshot, construct all update groups, and apply one accumulated batch gradient.
- Extended canonical, train, and test evaluators to group artifacts by experiment and use each artifact's dataset name.
- Current milestone remains `core_claim` until the submitted job refreshes scores and confirms the hard setting.

### [2026-06-25 04:41] - core_claim
- Final verification found that the hard initial-policy expansion was not present in `scores.json`, so the public gym target was narrowed back to the verified `synthetic_signal_recovery` experiment.
- `scripts/reproduce.sh`, `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, and `scoring/reference.json` now match the checked-in score structure.
- A CPU verification job is being submitted to refresh canonical, train, and test scores from the current batch-snapshot implementation.

### [2026-06-25 04:44] - core_claim
- CPU verification job `f7ca695b-4f2` completed successfully.
- The job refreshed canonical, train, and test scores from the current batch-snapshot implementation and `python validate.py --compare` passed.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8081` versus GRPO-n4 `0.7633` and GRPO-n16 `0.7869` while keeping `update_samples_per_prompt=4`.
- Final verification judged the workspace ready as a synthetic `core_claim` research gym.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim` | **Status**: error
- **Working time**: 0m | **GPU**: 0.0h
- **Jobs**: 9 total (7 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

### [2026-06-25 04:27] - core_claim
- Final verification job `1ba43ce0-e13` completed successfully and refreshed canonical, train, and test scores across seeds 17, 23, and 31.
- `python validate.py` and `python validate.py --compare` pass after the refreshed run.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8144` versus GRPO-n4 `0.7692` and GRPO-n16 `0.7870` while keeping the policy-update group size fixed at 4 samples.
- The expanded evaluator now includes hard/easy held-out slices and train/eval gap metrics, and train/test score files use matching method keys.

### [2026-06-25 04:34] - core_claim
- Prepared a new expansion job for a hard initial-policy benchmark modeled after the paper's hard-prompt ablation.
- Tightened algorithm fidelity: GRPO and Reinforce-Ada now sample each batch under a frozen policy snapshot, construct all update groups, and apply one accumulated batch gradient.
- Extended canonical, train, and test evaluators to group artifacts by experiment and use each artifact's dataset name.
- Current milestone remains `core_claim` until the submitted job refreshes scores and confirms the hard setting.

### [2026-06-25 04:41] - core_claim
- Final verification found that the hard initial-policy expansion was not present in `scores.json`, so the public gym target was narrowed back to the verified `synthetic_signal_recovery` experiment.
- `scripts/reproduce.sh`, `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, and `scoring/reference.json` now match the checked-in score structure.
- A CPU verification job is being submitted to refresh canonical, train, and test scores from the current batch-snapshot implementation.

### [2026-06-25 04:44] - core_claim
- CPU verification job `f7ca695b-4f2` completed successfully.
- The job refreshed canonical, train, and test scores from the current batch-snapshot implementation and `python validate.py --compare` passed.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8081` versus GRPO-n4 `0.7633` and GRPO-n16 `0.7869` while keeping `update_samples_per_prompt=4`.
- Final verification judged the workspace ready as a synthetic `core_claim` research gym.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 37m | **GPU**: 0.0h
- **Jobs**: 12 total (10 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 10:54] - core_claim
- Implemented Reinforce-Ada sequential adaptive sampling on a scaled binary-verifier RL benchmark.
- Ran fixed-group GRPO-n4 and GRPO-n16 baselines plus Reinforce-Ada Seq-Pos and Seq-Balance through `scripts/reproduce.sh`.
- Job `566c8047-478` completed successfully and `python validate.py --compare` passed.
- Core claim is supported in the scaled environment: adaptive sampling improves held-out pass rate over GRPO-n4 while keeping the policy-update group size fixed at 4, and Seq-Balance exceeds GRPO-n16's pass rate with a smaller update group.

### [2026-06-20 11:04] - method_runs
- Continuation audit reset the active milestone while tightening the evaluator.
- Fixed evaluation to recompute held-out pass rate from saved policy weights and aggregate training-cost metrics from per-step traces.
- Fixed Reinforce-Ada downsampling to fill from remaining collected samples before replacement.
- A fresh CPU job is required to restore `core_claim` under the stricter evaluator.

### [2026-06-20 11:13] - core_claim
- Job `2cc5ebd9-b02` completed successfully with the stricter evaluator and `python validate.py --compare` passed inside the container.
- The evaluator recomputed held-out pass rate from saved policy weights and cost/collapse metrics from per-step traces, without importing from `method/`.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8269` versus GRPO-n4 `0.8048` with the same policy-update group size of 4 samples.
- Seq-Balance also exceeded GRPO-n16's `eval_pass1=0.8146` while using 4 update samples instead of 16, and reduced signal-collapse rate to `0.3778` versus GRPO-n4 `0.6116` and GRPO-n16 `0.4082`.

### [2026-06-25 04:27] - core_claim
- Final verification job `1ba43ce0-e13` completed successfully and refreshed canonical, train, and test scores across seeds 17, 23, and 31.
- `python validate.py` and `python validate.py --compare` pass after the refreshed run.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8144` versus GRPO-n4 `0.7692` and GRPO-n16 `0.7870` while keeping the policy-update group size fixed at 4 samples.
- The expanded evaluator now includes hard/easy held-out slices and train/eval gap metrics, and train/test score files use matching method keys.

### [2026-06-25 04:34] - core_claim
- Prepared a new expansion job for a hard initial-policy benchmark modeled after the paper's hard-prompt ablation.
- Tightened algorithm fidelity: GRPO and Reinforce-Ada now sample each batch under a frozen policy snapshot, construct all update groups, and apply one accumulated batch gradient.
- Extended canonical, train, and test evaluators to group artifacts by experiment and use each artifact's dataset name.
- Current milestone remains `core_claim` until the submitted job refreshes scores and confirms the hard setting.

### [2026-06-25 04:41] - core_claim
- Final verification found that the hard initial-policy expansion was not present in `scores.json`, so the public gym target was narrowed back to the verified `synthetic_signal_recovery` experiment.
- `scripts/reproduce.sh`, `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, and `scoring/reference.json` now match the checked-in score structure.
- A CPU verification job is being submitted to refresh canonical, train, and test scores from the current batch-snapshot implementation.

### [2026-06-25 04:44] - core_claim
- CPU verification job `f7ca695b-4f2` completed successfully.
- The job refreshed canonical, train, and test scores from the current batch-snapshot implementation and `python validate.py --compare` passed.
- Reinforce-Ada Seq-Balance achieved `eval_pass1=0.8081` versus GRPO-n4 `0.7633` and GRPO-n16 `0.7869` while keeping `update_samples_per_prompt=4`.
- Final verification judged the workspace ready as a synthetic `core_claim` research gym.


### [2026-11-11] - core_claim (hardening, final iteration)
- Replaced the flat 8-action bandit with a sequence-response binary-verifier task (8 tokens × 16 symbols).
  - Training answers sit behind a hashed verifier; the teacher and the hidden test answers live only in `eval/test/`.
  - Training prompts use the paper's base-model difficulty filter; a hard training set reproduces the Table 4 ablation.
- Restored paper settings: Adam with warm-up and entropy bonus, 4 rounds of 8 samples (N_max 32), 600 steps; batch 256.
- Hidden test reads only final weights.
  - Scores greedy accuracy (overall, hard slice, easy slice) and requires 5 runs per method.
  - Failed, missing or duplicate runs score worst values.
- Job `5a68e270-72c` results:
  - Seq-Balance 0.486 / 0.507 against GRPO-n4 0.416 / 0.430 and GRPO-n16 0.451 / 0.478.
  - Seq-Balance uses 12.4 generated and 4 update responses per prompt, against 16 and 16 for GRPO-n16.
- Full audit in `HARDENING.md`.

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
