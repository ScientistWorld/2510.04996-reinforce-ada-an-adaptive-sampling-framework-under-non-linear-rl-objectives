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
