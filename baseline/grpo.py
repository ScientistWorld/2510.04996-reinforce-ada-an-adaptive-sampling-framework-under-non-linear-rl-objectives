"""Fixed-group GRPO baseline for the synthetic verifier environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from data.contextual_bandit import make_dataset, softmax, subset_indices


def policy_probs(features: np.ndarray, weights: np.ndarray, prompt: int) -> np.ndarray:
    return softmax(features[prompt : prompt + 1] @ weights)[0]


def reinforce_gradient(
    weights: np.ndarray,
    features: np.ndarray,
    prompt: int,
    actions: Sequence[int],
    advantages: Sequence[float],
) -> np.ndarray:
    probs = policy_probs(features, weights, prompt)
    grad = np.zeros_like(probs)
    for action, adv in zip(actions, advantages):
        one_hot = np.zeros_like(probs)
        one_hot[action] = 1.0
        grad += adv * (one_hot - probs)
    return np.outer(features[prompt], grad / max(1, len(actions)))


def train_grpo(
    group_size: int,
    steps: int,
    seed: int,
    output_dir: Path,
    dataset_name: str = "mixed_difficulty",
    experiment: str = "synthetic_signal_recovery",
    num_train: int = 320,
    num_eval: int = 160,
    batch_size: int = 64,
    lr: float = 0.42,
) -> dict:
    rng = np.random.default_rng(seed)
    dataset = make_dataset(name=dataset_name, num_train=num_train, num_eval=num_eval, seed=seed)
    weights = dataset.initial_weights.copy()
    train_ids = subset_indices(dataset, "train")
    eval_ids = subset_indices(dataset, "eval")
    history = []

    for step in range(steps):
        batch = rng.choice(train_ids, size=batch_size, replace=False)
        policy_snapshot = weights.copy()
        batch_gradient = np.zeros_like(weights)
        zero_signal = 0
        rewards_seen = []
        for prompt in batch:
            probs = policy_probs(dataset.features, policy_snapshot, int(prompt))
            actions = rng.choice(dataset.num_actions, size=group_size, p=probs)
            rewards = (actions == dataset.correct_actions[prompt]).astype(float)
            rewards_seen.extend(rewards.tolist())
            mean = float(rewards.mean())
            std = float(rewards.std())
            if std <= 1e-12:
                zero_signal += 1
                advantages = np.zeros_like(rewards)
            else:
                advantages = (rewards - mean) / (std + 1e-6)
            batch_gradient += reinforce_gradient(policy_snapshot, dataset.features, int(prompt), actions, advantages)

        weights += lr * batch_gradient

        eval_pass1 = None
        if step % 10 == 0 or step == steps - 1:
            eval_probs = softmax(dataset.features[eval_ids] @ weights)
            correct_probs = eval_probs[np.arange(len(eval_ids)), dataset.correct_actions[eval_ids]]
            eval_pass1 = float(correct_probs.mean())
        history.append(
            {
                "step": step,
                "eval_pass1": eval_pass1,
                "generated_total": int(group_size * batch_size),
                "batch_size": int(batch_size),
                "update_group_size": int(group_size),
                "zero_signal_update_groups": int(zero_signal),
                "batch_reward": float(np.mean(rewards_seen)),
            }
        )

    eval_probs = softmax(dataset.features[eval_ids] @ weights)
    correct_probs = eval_probs[np.arange(len(eval_ids)), dataset.correct_actions[eval_ids]]
    train_probs = softmax(dataset.features[train_ids] @ weights)
    train_correct_probs = train_probs[np.arange(len(train_ids)), dataset.correct_actions[train_ids]]
    result = {
        "method": f"grpo_n{group_size}",
        "experiment": experiment,
        "dataset_name": dataset_name,
        "seed": seed,
        "steps": steps,
        "num_train": num_train,
        "num_eval": num_eval,
        "eval_pass1": float(correct_probs.mean()),
        "train_pass1": float(train_correct_probs.mean()),
        "avg_generated_per_prompt": float(group_size),
        "update_samples_per_prompt": float(group_size),
        "signal_collapse_rate": float(np.mean([h["zero_signal_update_groups"] / h["batch_size"] for h in history])),
        "final_weights": weights.tolist(),
        "history": history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{experiment}_{result['method']}_seed{seed}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dataset-name", default="mixed_difficulty")
    parser.add_argument("--experiment", default="synthetic_signal_recovery")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/baselines"))
    args = parser.parse_args()
    result = train_grpo(
        args.group_size,
        args.steps,
        args.seed,
        args.output_dir,
        dataset_name=args.dataset_name,
        experiment=args.experiment,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
