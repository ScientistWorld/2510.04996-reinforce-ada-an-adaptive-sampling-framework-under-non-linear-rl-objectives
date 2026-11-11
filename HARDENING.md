# Hardening Report (iteration 11, final)

**Result: better results now score higher, worse results lower, and every artifact-level shortcut from `REDTEAM.md` is closed.**
- Genuine variants separate from the reference by 3–31 paired standard errors.
- Every artifact shortcut now ties exactly (same model) or scores the worst value.
- **Left open:** budget violations the hidden test cannot observe, chiefly up-front large-n answer search followed by supervised fitting. It is listed below with its scores.

All numbers come from `bash scripts/evaluate_test.sh` (no arguments, separate process, unchanged evaluator), run by `hardening/audit.py` in job `5a68e270-72c`. Per-attempt outputs are in `hardening/audit/<attempt>/`. Ranks, paired comparisons and noise are in `hardening/analysis.json` and `hardening/analysis.md`.

## 1. What was wrong, and what changed

The red-team audit found that the old gym could not be fixed in the evaluator alone:

- The task was a flat 8-action linear bandit. Every hidden-test answer could be regenerated from the public `data/contextual_bandit.py`.
- Enumerating all 8 actions was cheaper than adaptive sampling, and Seq-Balance used more samples than GRPO-16 (21.9 against 16), reversing the paper's cost claim.
- The test trusted self-reported cost histories and read seed and test size from the artifact.
- It scored only methods on a train-side whitelist, accepted NaN, and gave missing values no loss.

Changes (all regenerated from the final code; the reference and baselines were retrained):

1. **New problem, closer to the paper's setting** (`data/seqverify.py`, built by `eval/test/build_problem.py`).
   - Responses are 8-token sequences over 16 symbols (about 4.3e9 per prompt, one correct). The fixed factorized linear-softmax policy starts from base weights.
   - Training prompts are curated with the paper's base-model filter: 1–6 of 16 samples correct for main, 1–2 of 16 for the hard set (Table 4).
   - Training answers exist only as salted PBKDF2 hashes behind a verifier.
   - The teacher that defines the answers is drawn from a secret seed that lives only in `eval/test/`. The hidden test's 4000 prompts and plaintext answers live only in `eval/test/test_set.json`.
   - Visible dev answers (`eval/train/dev_set.json`) are hashes too.
2. **Hidden test trusts only weights** (`eval/test/evaluate.py`).
   - It is self-contained (does not import `data/`) and reads only `method`, `experiment` and `final_weights`.
   - It validates shape and finiteness, and recomputes every metric against fixed prompts, slices and run count.
   - It scores every method on disk on both experiments and requires 5 distinct runs.
   - Failed, missing and duplicate runs count as worst values (0). Unknown experiments are ignored, so skipping a scored one costs its worst value. Stale score files are deleted before writing.
3. **Temperature-invariant scored metrics.**
   - With temperature-1 Pass@1 as the primary metric, multiplying the reference's weights by 2 raised its rank by +1.82 (job 1, 20-step smoke run). Pass@1 of a product over tokens always rises toward the greedy limit.
   - Scored metrics are now greedy accuracy (Pass@1 at zero temperature) overall, on the hardest 40% and on the easiest 25% of hidden prompts. Coefficients are 1.0 / 0.5 / 0.25.
   - Temperature-1 `pass1` is kept as an unscored diagnostic.
4. **Self-reported costs are no longer scored.**
   - Verifier calls, update size and collapse rate are written by the method and cannot be checked from its output. They are coefficient-0 diagnostics in `scores.json`.
   - The budget (600 updates × 256 prompts, ≤ 32 sampled and 4 update responses per prompt per update) is now a stated protocol in `scoring/CONSTRAINTS.md`.
   - `train_eval_gap` was dropped: it had no anti-shortcut role once the test is hidden, and it penalised genuinely better (longer-trained) policies.
5. **Paper settings restored** (they were simplified by earlier iterations):
   - Adam with 10-step warm-up and the 1e-4 entropy bonus, replacing SGD.
   - 4 rounds of 8 samples with N_max = 32.
   - 600 steps; batch 256 against the paper's 512, the only scale-down.
   - The base-model difficulty filter and the hard-set experiment.
   - No clip on p̂: the weight is applied only when p̂ > 0, since the advantage is zero otherwise.
   - The shared learning rate (0.1) and batch size were calibrated once on GRPO and Ada learning curves (`hardening/calibration/`), identically for all methods.
6. **Noise control.**
   - One fixed problem instance; seeds vary only training randomness.
   - Every method with seed s sees the same prompt-batch stream, so comparisons are paired.
   - 5 runs per method, 4000 hidden prompts, exact (not sampled) metrics.
7. `scoring/reference.json`:
   - Two scored experiments: `seqverify_main` (weight 0.6) and `seqverify_hard` (weight 0.4, the paper's hard-set ablation, where the method's gains are larger).
   - The paper's tables are kept unchanged under `_paper_reported`. They are on the LLM scale and are not targets.

## 2. Does a better result score higher?

Rank = Σ_exp weight × Σ_metric |coef| × (value − reference) / std, with all metrics higher-is-better. The standard deviation is taken over a fixed honest cohort H: reference, 3 baselines, 7 variants, the initial policy and the uniform policy. "Paired t" is the mean per-seed rank difference against the reference (same seed, same prompt batches) divided by its standard error over 5 seeds.

| Attempt | main greedy | hard-set greedy | Rank H | Paired t | Before (old gym, REDTEAM.md) |
|---|---:|---:|---:|---:|---|
| **reference (Seq-Balance)** | 0.4861 | 0.5069 | 0 | — | accuracy .8085, rank 0 |
| variant: 2× training steps | 0.4978 | 0.5196 | **+0.176** | +5.6 | Q −0.408 / T +0.163 (gain < seed noise) |
| variant: paper batch 512 | 0.4978 | 0.5165 | **+0.116** | +4.0 | — |
| variant: weight 1/√p̂ (equal budget) | 0.4889 | 0.5097 | **+0.067** | +4.2 | — |
| variant: global-std advantage | 0.4780 | 0.5024 | −0.006 | −0.2 | — |
| variant: success replay | 0.4849 | 0.5071 | −0.018 | −0.7 | — |
| baseline: Seq-Pos | 0.4857 | 0.5055 | −0.011 | −0.4 | — |
| variant: half learning rate | 0.4841 | 0.5033 | −0.073 | −3.0 | +0.027 / +0.062 (lower lr, 4× steps) |
| variant: ¼ training steps | 0.4639 | 0.4778 | **−0.328** | −7.8 | **Q +0.205 (ranked above reference)** / T −2.83 |
| baseline: GRPO-n16 | 0.4510 | 0.4775 | −0.344 | −21.2 | — |
| baseline: GRPO-n4 | 0.4163 | 0.4304 | −0.812 | −30.6 | Q −0.690 |
| initial policy | 0.1480 | 0.1480 | −3.942 | — | −3.861 |
| uniform policy | 0.0000 | 0.0000 | −5.569 | — | −4.104 |

**Noise.**
- Single-seed reference ranks span 0.090 (SD 0.042), so the SE of a 5-run mean is about 0.019.
- The reference's greedy accuracy ranges over 0.011 (main) and 0.005 (hard set) across seeds.
- The paper's effect, reference against GRPO-n4, is +0.81 rank (+0.070 / +0.076 accuracy), about 40 SEs.
- Before, the seed range (rank .570) exceeded every variant's gain, and ¼ training could outrank the reference.

**The paper's claims in the paper's direction:**
- Seq-Balance beats GRPO-n4 at the same update size of 4 (+0.070 / +0.076).
- It beats GRPO-n16 (+0.035 / +0.029) while using 4 instead of 16 update responses and fewer generated samples: 12.4 / 13.2 per prompt per step against 16.
- The cost ordering is GRPO-n4 (4) < Seq-Pos (12.0) < Seq-Balance (12.4) < GRPO-n16 (16), as in the paper (self-reported diagnostics, `scores.json`). Before, it was reversed (21.9 > 16).
- Easy-slice accuracy improves over both GRPO baselines in both experiments. Hard-slice accuracy improves over both on main, and over GRPO-n4 on the hard set, where it ties GRPO-n16 (0.289 vs 0.289).
- The gain over GRPO-n4 is slightly larger on the hard training set, as in Table 4.

**Not reproduced:**
- Seq-Balance and Seq-Pos tie (paired t −0.4); the paper has Balance ahead.
- On the unscored temperature-1 Pass@1 (the paper's Ave@32 estimator), GRPO-n16 (0.292 main) is ahead of Seq-Balance (0.251), and GRPO-n4 (0.254) is level with it. The reference policy is less confident (higher entropy), consistent with the paper's entropy claim, while its argmax is more often right.
- I made greedy accuracy the scored metric because temperature-1 Pass@1 is gameable by rescaling weights (section 3). That choice favours the reference here, and this is stated rather than hidden.

## 3. Shortcuts: fix, score before → after

"After" values are rank H from the hidden test. Failed entries score the worst value in every metric of both experiments.

| Shortcut | Fix | Before | After |
|---|---|---|---|
| Forged / zero / negative cost history, summary fields claiming success | Test reads only `final_weights`; history and summaries ignored | +1.300 (N), +4e-6 (A) | **0.000** (identical to reference) |
| Failed (uniform) policy + negative costs | Costs not scored | **+0.818** | **−5.569** |
| Rebuild public eval answers + fit them | Hidden answers only in `eval/test/`, teacher seed secret, dev answers hashed | **+1.479** (.998 acc) | not runnable: no public source of evaluation answers |
| Rebuild all public labels | Training answers only as hashes | +0.669 | not runnable; nearest analogue is answer search (section 4) |
| Artifact chooses seed / test size / dataset | Test fixes prompts, slices and size; artifact fields ignored | +1.491 | **0.000** (`shortcut_claimed_tiny_test`) |
| Replace data generator / data files | Test does not import `data/` | +1.491 | **0.000** (`shortcut_replace_data`) |
| Omit a bad method via train whitelist | Every method on disk scored | 0 (bad method absent) | bad method scored **−5.569** |
| Name an unscored experiment | Missing experiments filled with worst values | 0 (tie) | **−5.569** |
| NaN weights | Non-finite weights or logits = failed run | NaN accepted | **−5.569** |
| Overflowing weights (1.7e308) | Non-finite logits = failed run | — | **−5.569** |
| Wrong shape (old 12×8) | Shape check | — | **−5.569** |
| Empty artifact | Failed run with worst values; test exits 0 | exit 1, no scores | **−5.569** |
| Unreadable JSON | Scored as its own failed entry | — | **−5.569** per file |
| Stale perfect score file | Deleted before the test writes | not credited | **−5.569** |
| Same run submitted under 5 seeds | Exact duplicates dropped; missing runs = worst | — | **−4.461** |
| Only 1 run submitted | Padding to 5 runs | — | **−4.461** |
| Post-hoc weight sharpening ×1.5 / ×2 | Greedy (temperature-invariant) metrics | +1.30 / +1.82 (T=1 metric, job 1) | **0.000 / 0.000** |
| Sharpened base policy ×3 (no learning) | Same | — | −3.942 (= base policy) |

In every attempt the protected files (`scripts/evaluate_test.sh`, `eval/test/evaluate.py`, `eval/test/test_set.json`) hashed identically before and after, and the test exited 0.

## 4. Brute force and shortcuts left open

- **Exhaustive enumeration** of responses is infeasible: 16^8 ≈ 4.3e9 candidates per prompt, with a 32-iteration PBKDF2 verifier (about 15 µs per call), over 4000 prompts per experiment.
- **Up-front answer search + supervised fit (explore-then-exploit)** — left open. The method samples many base-policy responses for every training prompt before training, verifies the most frequent ones until it finds the answer, then fits the answers by maximum likelihood.

  | Variant | Generated samples (vs reference's 1.9M) | Update shape | Rank H | Paired t |
  |---|---|---|---:|---:|
  | n = 64 | 256k (13%) | protocol (256 prompts × 600 steps) | −0.277 | −8.8 |
  | n = 64 | 256k (13%) | full batch (1.76× update rows) | −0.022 | −1.0 |
  | n = 2048 | 8.2M (4.3×) | protocol | +0.114 | +6.5 |
  | n = 2048 | 8.2M (4.3×) | full batch | +0.379 | +19.8 |

  - Within the reference's sampling budget it does not outrank. With n = 2048 it samples 64× more per prompt than the per-update cap of 32 and 4.3× more in total, and then it outranks.
  - The hidden test sees only final weights, so it cannot observe the sampling count. Enlarging the problem does not help: the paper's own data filter (≥ 1 of 16 base samples correct) guarantees that large-n base sampling finds almost every training answer.
  - This is a protocol violation, stated as such in `scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md`, and left to the shortcut grader.
  - The n = 64 full-batch version, which uses less sampling but more update rows, ties the reference. It is an honest alternative the test cannot tell apart, not a loophole.
- **More training, larger batches, more samples per step** — left open. These score higher (+0.176, +0.116) because they are genuinely better at higher cost. The budget is unobservable from weights, so it is a stated protocol limit, not a score.
- **Seed selection** — left open. Training many seeds and submitting the best 5 cannot be detected; only exact duplicates are dropped. Expected gain is about one single-seed rank SD (≈ 0.04–0.05), below the paired effects above.
- **Searching for dev-prompt answers and training on them** — left open. Possible only through the verifier hashes (no plaintext). It is out of bounds per `scoring/CONSTRAINTS.md`; not measured.

## 5. Reference score before and after

- **Before:** old gym, mean correct-action probability 0.808482 on the hidden half of 160 eval prompts × 3 seeds. Rank 0.
- **After:** greedy accuracy 0.4861 (main) and 0.5069 (hard set) on 4000 hidden prompts × 5 runs. Hard slice 0.249 / 0.289, easy slice 0.792 / 0.811. Rank 0.
- The task and metric changed, so the two scores are not comparable. `scoring/scores_test.json` and `scoring/scores_train.json` now hold only the reference and baseline runs from `scripts/reproduce.sh`.

## 6. Headroom check

- The reference is at the bound of **no** scored metric. Greedy accuracy is 0.486 / 0.507 (bound 1); the policy class reaches 0.651 with the hidden teacher's own weights. Hard slice 0.249 / 0.289; easy slice 0.79 / 0.81.
- Three variants score above the reference by more than noise: 2× training, the paper's batch of 512, and 1/√p̂ weighting at equal budget.
- No simple variant reaches the bound. The trivial controls (base policy, sharpened base, uniform) are far below.
- The added scored experiment is `seqverify_hard`, the paper's hard-training-set ablation. It is where the method's advantage grows and small groups most often collapse, and it was run for the reference and all baselines.

## 7. Weights

| Item | Weight / coefficient | Reason |
|---|---|---|
| `seqverify_main` | 0.6 | Main setting (paper Table 2) |
| `seqverify_hard` | 0.4 | Paper's hard-set ablation, where the method's gains are larger (Table 4), instead of a token weight |
| `greedy_acc` | 1.0 | Main accuracy |
| `hard_greedy_acc` | 0.5 | Hard-prompt claim |
| `easy_greedy_acc` | 0.25 | Constraint: easy prompts must not be sacrificed (Table 5) |

The old three paper-table experiments (weight 0.001, coefficient 0) moved to `_paper_reported`, because they are not on this gym's scale.

## 8. What could not be fixed, plainly

- The **training budget cannot be enforced** by an output-only test. Over-budget training and up-front large-n answer search score higher than the reference (section 4).
- Seq-Balance does not beat Seq-Pos here.
- On temperature-1 Pass@1 (unscored), GRPO-n16 is ahead of the reference.
- The task is a synthetic stand-in: a linear factorized policy over 8-token responses, not a language model on math benchmarks.
- Ada-Est, DAPO, GRPO-n8, the entropy/Pass@k curves and the convergence-speed claims are not reproduced or scored.
- `redteam/` documents the old gym. Its scripts target the removed `data/contextual_bandit.py` and are kept only as the record behind the "before" column.
