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
    features: np.ndarray
    correct_actions: np.ndarray
    initial_weights: np.ndarray
    split: np.ndarray

    @property
    def num_prompts(self) -> int:
        return int(self.correct_actions.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.initial_weights.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def make_dataset(
    name: str = "mixed_difficulty",
    num_train: int = 320,
    num_eval: int = 160,
    num_actions: int = 8,
    feature_dim: int = 12,
    seed: int = 17,
) -> BanditDataset:
    """Create prompts with learnable structure and varied initial pass rates."""

    rng = np.random.default_rng(seed)
    total = num_train + num_eval
    features = rng.normal(0.0, 1.0, size=(total, feature_dim))
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12

    teacher = rng.normal(0.0, 1.0, size=(feature_dim, num_actions))
    teacher_logits = features @ teacher
    margins = rng.normal(0.0, 0.15, size=teacher_logits.shape)
    correct = np.argmax(teacher_logits + margins, axis=1)

    # The initial policy is related to the teacher but underfit and noisy. This
    # creates a non-trivial baseline with many low-probability correct actions,
    # while still allowing policy-gradient updates to transfer to eval prompts.
    initial_weights = 0.35 * teacher + rng.normal(0.0, 0.55, size=teacher.shape)

    split = np.array(["train"] * num_train + ["eval"] * num_eval)
    return BanditDataset(name, features, correct, initial_weights, split)


def subset_indices(dataset: BanditDataset, split: str) -> np.ndarray:
    return np.flatnonzero(dataset.split == split)


def reward_for_actions(correct_actions: np.ndarray, prompt_ids: Iterable[int], actions: np.ndarray) -> np.ndarray:
    prompt_ids = np.asarray(list(prompt_ids), dtype=np.int64)
    return (actions == correct_actions[prompt_ids]).astype(np.float64)
