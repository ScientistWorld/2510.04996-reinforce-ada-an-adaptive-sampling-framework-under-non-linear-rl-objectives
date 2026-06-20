"""Synthetic verifier environment for adaptive RL sampling experiments.

The paper trains LLM policies on prompts with binary verifier rewards.  This
module provides the same problem interface at small scale: each prompt has one
correct action, a policy samples candidate actions, and the verifier returns
1 only for the correct action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BanditDataset:
    name: str
    correct_actions: np.ndarray
    initial_logits: np.ndarray
    split: np.ndarray

    @property
    def num_prompts(self) -> int:
        return int(self.correct_actions.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.initial_logits.shape[1])


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def make_dataset(
    name: str = "mixed_difficulty",
    num_train: int = 320,
    num_eval: int = 160,
    num_actions: int = 8,
    seed: int = 17,
) -> BanditDataset:
    """Create prompts with non-trivial low, medium, and high pass rates.

    The initial policy assigns the correct action probabilities from a beta
    mixture.  Many prompts start at p < 0.15, so fixed n=4 GRPO frequently sees
    all-zero groups, while the problem remains learnable because the correct
    action is in the finite action set and can be discovered by sampling.
    """

    rng = np.random.default_rng(seed)
    total = num_train + num_eval
    correct = rng.integers(0, num_actions, size=total)

    # Mixture chosen so a trivial baseline is far from solved, but there are
    # enough positives for adaptive samplers to recover signal.
    components = rng.choice(3, size=total, p=[0.55, 0.35, 0.10])
    target_p = np.empty(total)
    target_p[components == 0] = rng.beta(1.2, 12.0, size=(components == 0).sum())
    target_p[components == 1] = rng.beta(2.5, 5.5, size=(components == 1).sum())
    target_p[components == 2] = rng.beta(5.0, 3.0, size=(components == 2).sum())
    target_p = np.clip(target_p, 0.025, 0.82)

    logits = rng.normal(0.0, 0.04, size=(total, num_actions))
    for i, p in enumerate(target_p):
        wrong_mass = (1.0 - p) / (num_actions - 1)
        probs = np.full(num_actions, wrong_mass)
        probs[correct[i]] = p
        logits[i] += np.log(probs)

    split = np.array(["train"] * num_train + ["eval"] * num_eval)
    return BanditDataset(name, correct, logits, split)


def subset_indices(dataset: BanditDataset, split: str) -> np.ndarray:
    return np.flatnonzero(dataset.split == split)


def reward_for_actions(correct_actions: np.ndarray, prompt_ids: Iterable[int], actions: np.ndarray) -> np.ndarray:
    prompt_ids = np.asarray(list(prompt_ids), dtype=np.int64)
    return (actions == correct_actions[prompt_ids]).astype(np.float64)
