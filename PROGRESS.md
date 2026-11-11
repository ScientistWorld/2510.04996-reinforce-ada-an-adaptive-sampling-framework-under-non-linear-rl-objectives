# Progress

## What Works

Reinforce-Ada-Seq (Seq-Balance and Seq-Pos, Algorithm 2) and fixed-group GRPO baselines run on a sequence-response binary-verifier RL task:
- responses are 8 tokens over 16 symbols, with one correct response per prompt;
- training answers are behind a hashed verifier, and the fixed factorized linear-softmax policy starts from base weights;
- training prompts are curated with the paper's base-model difficulty filter.

Paper settings:
- N_max = 32 in 4 rounds of 8, update group 4, K_pos = K_neg = 2, explicit 1/p̂ weight;
- Adam with warm-up and entropy bonus, 600 steps.

Batch 256 against the paper's 512 is the one scale-down. The hidden test (`eval/test/`) scores greedy accuracy from final weights only, on 4000 hidden prompts.

## Results (hidden test, 5 runs; job `5a68e270-72c`)

| Method | main greedy | hard-set greedy | generated / prompt / step | update responses / prompt |
|---|---:|---:|---:|---:|
| GRPO-n4 | 0.416 | 0.430 | 4 | 4 |
| GRPO-n16 | 0.451 | 0.478 | 16 | 16 |
| Seq-Pos | 0.486 | 0.506 | 12.0 / 13.0 | 4 |
| Seq-Balance (reference) | 0.486 | 0.507 | 12.4 / 13.2 | 4 |

The core claim is reproduced in the scaled setting. Adaptive sampling beats GRPO-n4 at the same update size by about 40 paired standard errors. It also beats GRPO-n16 with fewer generated samples and 4× fewer update responses. The cost ordering matches the paper.

## Remaining

- Seq-Balance does not beat Seq-Pos.
- On temperature-1 Pass@1 (unscored diagnostic), GRPO-n16 is ahead.
- Ada-Est, DAPO, GRPO-n8, and the entropy and Pass@k analyses are not implemented.

See `HARDENING.md` for the audit and the shortcuts left open.

## Deviations from Paper

- Synthetic substitute: a linear factorized policy over 8-token responses instead of LLMs on math benchmarks.
- Batch 256 instead of 512, and a learning rate fitted to the linear model.
- Exact greedy accuracy on hidden prompts is scored, instead of Ave@32 at temperature 1. The reason is in `HARDENING.md` section 1.3.
