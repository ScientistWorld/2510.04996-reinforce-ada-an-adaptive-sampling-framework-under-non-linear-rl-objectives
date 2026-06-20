# Progress

## What Works

Implemented a scaled reproduction of Reinforce-Ada on a binary-verifier contextual bandit benchmark. The code includes fixed-group GRPO baselines and the paper's sequential adaptive sampling variants with active prompts, stopping conditions, global pass-rate baseline, balanced downsampling, fixed update group, and inverse-pass-rate weighting.

## Results

Pending compute job. The submitted run will write `scoring/scores.json` from actual artifacts and compare with `scoring/reference.json`.

## Remaining

Run the job, inspect whether Reinforce-Ada improves `eval_pass1` over GRPO-n4 while keeping `update_samples_per_prompt=4`, and iterate if the synthetic setting is too easy or too hard.

## Issues

The original paper trains 1.5B-7B LLMs with verl on large math datasets, which is too expensive for an initial reproduction in this workspace. The current environment preserves the algorithmic steps on a controlled small RL task.

## Deviations from Paper

The paper uses LLM math-reasoning prompts, verifier-based rewards, and verl RL training on multi-GPU hardware. This reproduction uses a synthetic finite-action verifier environment to make the experiment runnable within the available budget. The method itself is not replaced: the implementation follows the paper's adaptive sampling, stopping, downsampling, baseline, and weighting computations.
