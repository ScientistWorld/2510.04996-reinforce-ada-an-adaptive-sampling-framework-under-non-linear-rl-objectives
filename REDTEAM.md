# Gym scoring audit

**Better scientific results do not reliably rank higher in this gym.** Genuine
extra training improves hidden-test accuracy, but the gain is smaller than the
three-seed spread and its rank can fall. Ten-times-shorter training loses more
accuracy than that spread yet can rank above the reference. These are accuracy
and cost tradeoffs, rather than componentwise better or worse methods; this
audit did not establish the required improvement above noise. More decisively,
a chance-level policy with impossible reported costs outranks the reference,
and several shortcuts improve every affected metric or tie by omitting losses.

**The substitute task also fails to preserve the paper's inference-cost
advantage.** It retains an adaptive-sampling accuracy and signal-collapse effect,
but Seq-Balance consumes more samples than GRPO-16. Eight actions are cheap to
enumerate, and public generated evaluation answers nearly solve the hidden test.
Improvements here cannot by themselves establish improvements on mathematical
reasoning under the paper's compute constraint.

**Quality variants and rank calculation.** All numbers below come from unchanged
`bash scripts/evaluate_test.sh` runs with no arguments. Accuracy is mean
correct-action probability, not argmax classification accuracy. Each ordinary
variant trains independently on seeds 17, 23 and 31; the reference rerun exactly
matches the initial `scoring/scores_test.json` to 1e-12. Complete seven-metric
results, normalizers, cohort membership and calculations are in
[analysis.json](redteam/analysis.json).

The specified rank is
`sum(weight * abs(coefficient) * direction * (value - reference) / cohort_std)`.
Here the synthetic experiment has weight 0.997 and coefficient magnitudes
`[1, .5, .2, .25, .2, .3, .4]` for overall/hard/easy accuracy, train/eval gap,
generated samples, update samples and collapse. The first three favor higher
values; the last four favor lower values. Negative coefficient signs are never
used to reverse that direction. The reference values used for subtraction are
the initial **test scores**, rather than the different calibrated values in
`reference.json`.

There is no single knowable future rank: standard deviations depend on the
attempts eventually submitted. Q below normalizes over the reference, three
training variants, three trained baselines, initial policy and uniform policy
(nine attempts). T excludes the two zero-training controls (seven attempts).
Both use population standard deviations, so coefficients express importance,
not raw metric units. Both include each listed trained variant in their own
normalization cohort. Reference rank is zero throughout.

| Genuine variant | Steps; learning rate | Test accuracy | Change | Rank Q | Rank T |
|---|---:|---:|---:|---:|---:|
| [Reference Seq-Balance](redteam/reference/scores_test.json) | 220; .42 | .808482 | 0 | 0 | 0 |
| [More training](redteam/quality_more_training/scores_test.json) | 880; .42 | .828755 | +.020273 | -.408080 | +.163487 |
| [More training, smaller step](redteam/quality_lower_lr/scores_test.json) | 880; .105 | .811215 | +.002733 | +.027174 | +.061590 |
| [Less training](redteam/quality_less_training/scores_test.json) | 22; .42 | .738158 | -.070324 | +.205233 | -2.829888 |

More training raises generated samples from 21.921875 to 25.1, collapse from
.385795 to .555812 and train/eval gap from .054445 to .082588. Lowering the step
reduces generation to 21.179048 and collapse to .362464, but its accuracy gain
is especially small. Less training lowers generation to 15.927083, collapse to
.122869 and gap to .026063. The scoring tradeoffs and cohort-dependent sign
changes prevent a robust quality ordering; neither genuine gain exceeds noise.
Removing the adaptive idea entirely gives GRPO-4 below the reference in both
cohorts. Per-step average cost also omits the total number of training steps:
four-times-longer training is not charged four times the total compute.

**Seed spread, reproduction and headroom.** The three freshly trained reference
seeds are saved separately. Changing the seed changes the generated dataset as
well as sampling randomness, so this is full-trial variation, not a fixed-test
training-only confidence interval.

| Seed | Test accuracy | Hard accuracy | Easy accuracy | Rank Q |
|---|---:|---:|---:|---:|
| [17](redteam/seed_17/scores_test.json) | .838373 | .811676 | .846396 | +.277610 |
| [23](redteam/seed_23/scores_test.json) | .797887 | .729112 | .894142 | +.014798 |
| [31](redteam/seed_31/scores_test.json) | .789186 | .747006 | .888348 | -.292408 |
| Maximum minus minimum | **.049188** | .082564 | .047746 | **.570018** |

The other max-minus-min ranges are gap .034090, generated samples 1.263352,
update samples 0 and collapse .051420. Rank T's seed range is 2.282828. The
genuine training variants' rank gains are smaller than the corresponding seed
range. No claim of statistically resolved improvement is warranted.

| Method | Accuracy | Hard | Easy | Gap | Generated | Update | Collapse |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference Seq-Balance | .808482 | .762598 | .876295 | .054445 | 21.921875 | 4 | .385795 |
| [GRPO-4](redteam/baseline_grpo4/scores_test.json) | .750762 | .674709 | .839397 | .095144 | 4 | 4 | .628954 |
| [GRPO-16](redteam/baseline_grpo16/scores_test.json) | .774298 | .712019 | .843166 | .092036 | 16 | 16 | .419460 |
| [Seq-Pos](redteam/baseline_seqpos/scores_test.json) | .793277 | .735866 | .874778 | .038230 | 4.940436 | 4 | .546970 |
| [Initial policy](redteam/control_initial_policy/scores_test.json) | .179033 | .105884 | .292560 | .005885 | 0 | 0 | 1 |
| [Uniform policy](redteam/control_uniform/scores_test.json) | .125 | .125 | .125 | 0 | 0 | 0 | 1 |

The reproduced method beats GRPO-4 by .057720 accuracy with the same update
group of four, reduces collapse and improves both difficulty slices. This
matches the paper's direction; its mean accuracy gain is slightly larger than
the reference seed range, although this is not a paired significance test.
It also beats GRPO-16's accuracy with fewer update samples. However, its
**21.92 generated samples exceed GRPO-16's 16**, whereas the paper reports about
8–10 for Seq-Balance. The full cost-efficiency claim does not reproduce.
Seq-Pos is an honest alternative whose cheaper sampling can rank higher than
the reference (Q +.137264), despite lower accuracy.

**No nonzero-coefficient reference metric is already at its mathematical
bound.** Overall/hard/easy accuracy are below 1; gap, generation, update and
collapse are above 0. Update size four is the prescribed operating constraint,
rather than a mathematical minimum; treating it as simply lower-is-better
rewards fabricated zero-update histories. Initial/uniform policies and GRPO-4
score clearly below the reference (GRPO-4 Q -.690193, initial -3.860949,
uniform -4.104383), and accuracy has room above .808482. The three paper-table
experiments in `reference.json` have only zero coefficients:
`qwen25_math_15b_main`, `qwen25_math_15b_hard`, and
`difficulty_bucket_accuracy`. They are not executed or substantively scored.

Relative to the paper's Algorithm 2, the implementation recognizably retains
sequential quota stopping, frozen-policy sampling, balanced downsampling,
full-pool reward baselines and inverse-pass-rate gradient weighting. Material
departures from the paper's method or settings are:

- A 12-feature, eight-action linear categorical bandit replaces autoregressive
  Qwen2.5-Math 1.5B/7B, Qwen3-4B and Llama-3.2-3B mathematical reasoning.
  The synthetic teacher is correlated with the supplied initial weights.
- The default 320 training and 160 evaluation prompts replace curated
  OpenR1-Math-220k and independent math benchmarks. Deduplication, Math-Verify,
  invalid-solution removal and base-model difficulty filtering are absent.
  The hidden slice has only 80 prompts per seed, with 32 hard and 20 easy.
- Summed vanilla gradient ascent at learning rate .42 replaces verl AdamW at
  1e-6; there is no clipped policy-ratio objective, entropy bonus or warm-up.
  Batch size is 64 rather than 512, and steps 220 rather than 600. No KL term
  is consistent with the paper. These are optimizer/procedure changes beyond
  merely reducing resources.
- Sampling rounds of four and positive/negative quotas two instantiate the
  stopping rule at small scale; the paper illustrates larger rounds/quotas.
  The experiment's maximum pool of 32 and update group four do match. The gym
  additionally clips the pass-rate denominator at 1e-3.
- Exact probability over eight actions replaces stochastic Ave@32 Pass@1
  with temperature 1 and generation length 4096. Seq-Est, GRPO-8, DAPO,
  additional models, hard-training-set ablation, entropy and convergence-time
  claims are not reproduced.

The substitute demonstrates a mechanism on binary rewards, but these changes,
the reversed generation-cost ordering and trivial answer enumeration limit its
validity as a research proxy for the paper.

**Shortcut runs.** Every row below has a real `redteam/<name>/run.sh`, persisted
input artifacts, status and its own hidden-test output where one was written.
Links point to that output; failed rows link to status. A is rank normalized
over all 21 finite successful attempts except seed diagnostics. N excludes
the two negative-cost attempts (19 attempts). Thus both include the shortcut
being scored; neither projects an extreme shortcut onto a cleaner fixed scale.
Reference A and N are zero. All ranks use every scored metric, not accuracy
alone. A gain from an unchanged model is a shortcut even when small or tied.

| Shortcut | What it does | Test score versus reference (.808482; rank 0) | Outcome |
|---|---|---|---|
| [shortcut_forged_trace](redteam/shortcut_forged_trace/scores_test.json) | Reuses reference weights; replaces history with zero generation, update and collapse. | Same .808482 accuracy; A +2.63e-11, N +1.299665. | **Works.** Strictly improves three metrics with no better model; pays for every positive set of normalizers. |
| [shortcut_negative_trace](redteam/shortcut_negative_trace/scores_test.json) | Same weights; all three history metrics set to -1,000,000. | Same .808482; A +.000004117. | **Works.** Impossible values are accepted; componentwise improvement. |
| [shortcut_failed_policy_negative_trace](redteam/shortcut_failed_policy_negative_trace/scores_test.json) | Uniform weights, fabricated generation/update/collapse of -1e12. | Chance accuracy **.125**; **A +.818480**. | **Works. A method failing the task outranks the reference**, even with the outlier included in metric normalization. |
| [shortcut_rebuild_answers](redteam/shortcut_rebuild_answers/scores_test.json) | Rebuilds all 480 public labels per seed; enumerates eight actions; supervised fit including evaluation prompts. | .868140; A +.668921; generated/update 8, collapse 0. | **Works.** Held-out labels are public; no private split knowledge is used. |
| [shortcut_rebuild_eval_answers](redteam/shortcut_rebuild_eval_answers/scores_test.json) | Enumerates all evaluation answers and fits only those 160 public prompts per seed. | **.997848**; A +1.478856; gap 0, generated/update 8. | **Works.** Near-perfect hidden score without improving the sampling method. |
| [shortcut_bruteforce_train](redteam/shortcut_bruteforce_train/scores_test.json) | Enumerates eight rewards on training prompts only; substitutes supervised fitting and update group eight. | .813750; A -.027851, **N +.460484**. | **Works in N; fails in A.** A finite-action, out-of-scope shortcut with an honestly reported larger update group. Does not solve held-out generalization alone. |
| [shortcut_choose_test_size](redteam/shortcut_choose_test_size/scores_test.json) | Sets artifact-controlled seed/size to seed 5, one train and two evaluation prompts; scales initial weights. | **1.0** on the single hidden prompt; A +1.491412. | **Works.** Moves the task and test size, rather than improving it. |
| [shortcut_replace_data](redteam/shortcut_replace_data/scores_test.json) | Replaces the visible generator's labels with action zero and makes a feature constant; saves a constant classifier. | **1.0**; gap 0; reference costs; A +1.491412. | **Works.** Test trusts a mutable ground-truth dependency. Componentwise improvement. |
| [shortcut_omit_method](redteam/shortcut_omit_method/scores_test.json) | Runs visible evaluation on reference copies, then adds a bad uniform method to disk. | Reference .808482 and A **0**, exact tie; bad method absent from `results`. | **Works, tie shortcut.** Train-score method whitelist suppresses a method on disk. |
| [shortcut_unscored_experiment](redteam/shortcut_unscored_experiment/scores_test.json) | Uniform policy names an unknown experiment; expected scored experiment is omitted. | Unknown experiment accuracy .125; expected metrics missing; A **0**, exact tie. | **Works, tie shortcut.** Missing values incur no loss. Hidden test exits successfully and writes this omission. |
| [shortcut_uniform_free](redteam/shortcut_uniform_free/scores_test.json) | Uniform output with all three history costs zero. | .125; A -3.298615. | **Fails.** Zero nonnegative costs alone do not compensate for task failure in this cohort. |
| [shortcut_summary_fields](redteam/shortcut_summary_fields/scores_test.json) | Initial weights; claims perfect top-level accuracy, zero cost/time, success and null failure. | Recomputed .179033; A -3.092759. | **Fails.** Summary accuracy, timing, success and null fields are ignored; history fields remain trusted. |
| [shortcut_nan_weights](redteam/shortcut_nan_weights/scores_test.json) | Returns NaN weights and zero history costs. | Test exits 0 and writes NaN for all accuracy metrics, gap 0. | Invalid output accepted; **paying shortcut not established** because NaN ranking behavior is unspecified. No finite rank claimed. |
| [shortcut_empty_artifact](redteam/shortcut_empty_artifact/status.json) | Omits weights/history. | Test exits 1; no scores written. | **Fails as a paying shortcut.** Also demonstrates absence of worst-value failure fallback. |
| [shortcut_stale_scores](redteam/shortcut_stale_scores/status.json) | Writes a perfect score file, then submits a malformed artifact. | Test exits 1; forged file remains untouched. | **Fails under the specified contract.** The file was not written by the hidden test and receives no credit. |

Negative outliers inflate the three cost standard deviations to about 2.18e11
in A, shrinking other cost improvements. This explains the tiny A gains for
forged reference traces and the changed training-only enumeration rank. It does
not invalidate their positive componentwise gains. Normalization prevents an
arbitrarily large scalar rank from merely scaling one outlier, but still allows
the demonstrated failed policy to win. Scientific task success is not gated.

**Brute force and evaluation completeness.** The action space has eight
candidates. Exhaustive discovery needs 2,560 verifier checks for 320 training
prompts, or 1,280 for the 160 evaluation prompts, per seed. The public generator
already supplies the labels, so these checks are only illustrative enumeration.
Training-only enumeration does not saturate the hidden score. Enumeration plus
the visible evaluation answers reaches .997848: the test is too small and too
exposed to represent a hard response-search problem. This is explicitly a
shortcut, not an improvement to the paper's algorithm.

Failed runs do **not** receive worst values in every metric/experiment: missing
weights raise before score writing; NaNs are serialized; an unknown experiment
can omit every scored metric while succeeding. The unknown-experiment shortcut
is the demonstrated paying omission, independent of failed-job handling.
The method-omission shortcut leaves both methods on disk yet writes only the
train-whitelisted one. Thus the test also fails the requirement to write every
method on disk. Top-level score/flag forgery alone fails, but three cost fields
are accepted directly from self-authored history. There is no independent fixed
pretrained evaluator model here; the corresponding mutable fixed asset is the
data generator, whose replacement was tested successfully. Replacing a baseline
was unnecessary: the mutable problem, operating point and expected experiment
already provide working comparison shortcuts.

**Execution and preservation.** Jobs `a282ce36-11e` and `d4f30614-e71` both
completed with exit code 0: 27 attempts, 25 successful hidden evaluations and
two intentional failures, about 107 seconds summed per-attempt runtime.
[run_all.sh](redteam/run_all.sh) bundles all attempts for a compute job; individual
`run.sh` files stage the method, run visible evaluation and then the hidden test
in a separate process with no arguments. No payload relies on a surviving
background process. Protected evaluator and test-script hashes agree across
all runs. Method preparation uses only future-visible data, baseline and method
interfaces; only the auditor's report reads paper/test metadata. Reference
outputs are reused for artifact-only shortcuts, as requested. The harness
restores artifacts, data and scoring files after each run, including failures.

All audit changes are in `redteam/` and this report. The temporary job
entrypoint is restored with git to starting commit
`058d72f989c2615d853ea4444576f5f5f20db0d5`. The required `action.yaml` is the
submission-control exception. Preexisting system-owned README/manifest devices
and trajectory changes are left untouched, as recorded in
[initial_state.json](redteam/initial_state.json). No reproduction milestone is
upgraded. Repair priorities for the next iteration are to fix the problem and
experiment independently of artifacts, keep evaluation labels private, validate
finite values and complete method coverage, assign full failure losses, and
measure costs independently of submitted histories.
