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

The paper reports math-reasoning accuracy targets for several reinforcement-learning methods on multiple language-model backbones. The scaled reproduction uses a synthetic binary-verifier RL benchmark and compares fixed small-group and larger-group policy-gradient baselines against constrained methods that keep `update_samples_per_prompt=4`.

## Evaluation Protocol

Run `bash scripts/reproduce.sh` to train all included methods on the same generated train/eval splits with the configured seeds. Then run `bash scripts/evaluate.sh checkpoints/all_results` to score the JSON artifacts. Evaluation code must not import from `method/`; it only reads method outputs.
