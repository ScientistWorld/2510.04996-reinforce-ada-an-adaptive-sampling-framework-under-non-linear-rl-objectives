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
- `hard_eval_pass1`: mean correct-action probability on the held-out prompts that were hardest under the initial policy. Higher is better.
- `easy_eval_pass1`: mean correct-action probability on the held-out prompts that were easiest under the initial policy. Higher is better.
- `train_eval_gap`: positive gap between train-prompt and held-out pass rates. Lower is better.
- `signal_collapse_rate`: fraction of policy-update groups whose binary rewards have zero variance. Lower is better.
- `avg_generated_per_prompt`: average number of verifier calls used per prompt. Lower is better when accuracy is comparable.
- `update_samples_per_prompt`: number of samples included in the policy update per prompt. This is a constraint metric; lower values mean lower training cost.

## Baselines and Targets

The paper reports math-reasoning accuracy targets for several reinforcement-learning methods on multiple language-model backbones. The scored scaled reproduction uses a synthetic binary-verifier RL benchmark and compares fixed small-group and larger-group policy-gradient baselines against constrained methods that keep `update_samples_per_prompt=4`.

## Evaluation Protocol

Run `bash scripts/reproduce.sh` to train all included methods on the same generated train/eval splits with the configured seeds. Then run `bash scripts/evaluate.sh checkpoints/all_results` to score the JSON artifacts. Evaluation code must not import from `method/`; it only reads method outputs.

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


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
