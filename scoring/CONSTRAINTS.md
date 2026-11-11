# Scientist Constraints

## Problem and Data

Use the committed problem instance: `data/seqverify/` (base-policy weights, training prompts with answer hashes) through `data/seqverify.py`. Do not regenerate, relabel or replace it. Training prompts for `seqverify_main` come from `train_main.json`, and for `seqverify_hard` from `train_hard.json`. The visible evaluation prompts in `eval/train/dev_set.json` are for measurement only: do not train on them, and do not search for their answers.

## Policy Class

The submitted policy is the fixed factorized linear-softmax policy: weights of shape (8, 16, 16), applied as `softmax(x @ W[l])` per position. Every run starts from the provided base-policy weights. A different model class, extra features or hand-written weights measure something other than learning from verifier feedback.

## Learning Signal

Learn only from responses sampled from the current policy and scored by the provided verifier (`data.seqverify.Verifier`). Do not attempt to invert the answer hashes. Do not enumerate responses outside what the sampling budget allows. Do not use any information about evaluation answers.

## Training Budget (fixed; the hidden test cannot observe it)

| Quantity | Limit |
|---|---|
| policy updates per run | 600 |
| prompts per update | 256, drawn by `data.seqverify.batch_schedule(seed, ...)` |
| responses sampled and verified per prompt per update | at most 32, sampled online from the current policy |
| responses per prompt used in the update (backward pass) | 4 |
| optimizer | Adam, lr 0.1, 10-step linear warm-up, entropy bonus 1e-4 (as in `data.seqverify.PROTOCOL`) |
| runs per method and experiment | 5 independent seeds; report all of them, not a selection |

A method may change how responses are allocated across prompts within these limits, how advantages and weights are computed, and how update groups are formed. Training longer, using larger batches or larger update groups, sampling more than 32 responses per prompt per update, or searching for answers ahead of training are budget changes, not method improvements. Such runs are not comparable, even though the hidden test would score them higher.

## Evaluation

Write artifacts in the contract described in `briefing/evaluation.md`. The evaluators read only `method`, `experiment` and `final_weights`. They recompute every metric and never import `method/`.

## Compliance Checklist

- [ ] Uses the committed problem files and the fixed policy class, starting from the base-policy weights
- [ ] Learns only from verifier rewards of responses sampled within the per-update budget
- [ ] 600 updates × 256 prompts, ≤ 32 sampled and 4 update responses per prompt per update
- [ ] 5 runs per experiment, all reported
- [ ] Does not train on, or search over, evaluation prompts
