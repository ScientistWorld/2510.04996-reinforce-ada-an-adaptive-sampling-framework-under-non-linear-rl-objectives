"""Reinforce-Ada implementation for the synthetic verifier environment.

This is a scaled implementation of Algorithm 2 from the paper: prompts are
sampled in sequential rounds, active prompts keep receiving responses until a
positive or balanced exit condition is met, the variable pool is downsampled to
a fixed update group, and gradients use a global pass-rate baseline with the
log-objective inverse-pass-rate weight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from data.contextual_bandit import make_dataset, softmax, subset_indices


def sample_actions(rng: np.random.Generator, probs: np.ndarray, n: int) -> np.ndarray:
    return rng.choice(probs.shape[0], size=n, p=probs)


def reinforce_update(logits: np.ndarray, prompt: int, actions: Sequence[int], advantages: Sequence[float], lr: float) -> None:
    probs = softmax(logits[prompt : prompt + 1])[0]
    grad = np.zeros_like(probs)
    for action, adv in zip(actions, advantages):
        one_hot = np.zeros_like(probs)
        one_hot[action] = 1.0
        grad += adv * (one_hot - probs)
    logits[prompt] += lr * grad / max(1, len(actions))


def balanced_downsample(
    rng: np.random.Generator,
    pool: List[Tuple[int, float]],
    update_group_size: int,
) -> List[Tuple[int, float]]:
    positives = [item for item in pool if item[1] > 0.5]
    negatives = [item for item in pool if item[1] < 0.5]
    half = update_group_size // 2
    selected: List[Tuple[int, float]] = []

    def take(items: List[Tuple[int, float]], count: int) -> List[Tuple[int, float]]:
        if not items or count <= 0:
            return []
        idx = rng.choice(len(items), size=min(count, len(items)), replace=False)
        return [items[int(i)] for i in np.atleast_1d(idx)]

    selected.extend(take(positives, half))
    selected.extend(take(negatives, update_group_size - len(selected)))
    if len(selected) < update_group_size:
        remainder = positives if len(positives) > len(selected) else negatives
        selected.extend(take(remainder, update_group_size - len(selected)))
    if len(selected) < update_group_size:
        idx = rng.choice(len(pool), size=update_group_size - len(selected), replace=True)
        selected.extend(pool[int(i)] for i in np.atleast_1d(idx))
    rng.shuffle(selected)
    return selected[:update_group_size]


def train_one(
    variant: str,
    steps: int,
    seed: int,
    output_dir: Path,
    num_train: int = 320,
    num_eval: int = 160,
    batch_size: int = 64,
    update_group_size: int = 4,
    samples_per_round: int = 4,
    max_samples: int = 32,
    k_pos: int = 2,
    k_neg: int = 2,
    lr: float = 0.42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    dataset = make_dataset(num_train=num_train, num_eval=num_eval, seed=seed)
    logits = dataset.initial_logits.copy()
    train_ids = subset_indices(dataset, "train")
    eval_ids = subset_indices(dataset, "eval")

    history = []
    for step in range(steps):
        batch = rng.choice(train_ids, size=batch_size, replace=False)
        generated = 0
        zero_signal_before_recovery = 0
        prompt_pass_rates = []

        for prompt in batch:
            pool: List[Tuple[int, float]] = []
            active = True
            while active and len(pool) < max_samples:
                probs = softmax(logits[prompt : prompt + 1])[0]
                actions = sample_actions(rng, probs, samples_per_round)
                rewards = (actions == dataset.correct_actions[prompt]).astype(float)
                pool.extend((int(a), float(r)) for a, r in zip(actions, rewards))
                generated += len(actions)

                pos = sum(r for _, r in pool)
                neg = len(pool) - pos
                if variant == "seq_pos":
                    active = pos < k_pos
                elif variant == "seq_balance":
                    active = not (pos >= k_pos and neg >= k_neg)
                else:
                    raise ValueError(f"unknown variant: {variant}")

            first_group_rewards = [r for _, r in pool[:update_group_size]]
            if len(set(first_group_rewards)) == 1:
                zero_signal_before_recovery += 1

            pass_rate = float(np.mean([r for _, r in pool]))
            prompt_pass_rates.append(pass_rate)
            update_group = balanced_downsample(rng, pool, update_group_size)
            baseline = np.clip(pass_rate, 1e-3, 1.0)
            weight = 1.0 / baseline
            advantages = [(r - pass_rate) * weight for _, r in update_group]
            actions = [a for a, _ in update_group]
            reinforce_update(logits, int(prompt), actions, advantages, lr)

        if step % 10 == 0 or step == steps - 1:
            eval_probs = softmax(logits[eval_ids])
            correct_probs = eval_probs[np.arange(len(eval_ids)), dataset.correct_actions[eval_ids]]
            history.append(
                {
                    "step": step,
                    "eval_pass1": float(correct_probs.mean()),
                    "avg_generated_per_prompt": generated / batch_size,
                    "signal_collapse_rate_initial_group": zero_signal_before_recovery / batch_size,
                    "prompt_level_train_reward": float(np.mean(prompt_pass_rates)),
                }
            )

    eval_probs = softmax(logits[eval_ids])
    correct_probs = eval_probs[np.arange(len(eval_ids)), dataset.correct_actions[eval_ids]]
    train_probs = softmax(logits[train_ids])
    train_correct_probs = train_probs[np.arange(len(train_ids)), dataset.correct_actions[train_ids]]

    result = {
        "method": f"reinforce_ada_{variant}",
        "experiment": "synthetic_signal_recovery",
        "seed": seed,
        "steps": steps,
        "eval_pass1": float(correct_probs.mean()),
        "train_pass1": float(train_correct_probs.mean()),
        "avg_generated_per_prompt": float(np.mean([h["avg_generated_per_prompt"] for h in history])),
        "update_samples_per_prompt": float(update_group_size),
        "signal_collapse_rate_initial_group": float(np.mean([h["signal_collapse_rate_initial_group"] for h in history])),
        "history": history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{result['method']}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["seq_pos", "seq_balance"], default="seq_balance")
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/reinforce_ada"))
    args = parser.parse_args()
    result = train_one(args.variant, args.steps, args.seed, args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
