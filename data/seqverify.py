"""Public interface of the sequence-response binary-verifier RL problem.

The paper trains language-model policies on prompts whose sampled responses
are checked by a binary verifier. This module provides the same interface at
small scale:

* a prompt is a feature vector ``x`` in R^d;
* a response is a sequence of ``L`` tokens, each one of ``V`` symbols, so the
  response space has V**L (16**8, about 4.3e9) members and cannot be enumerated;
* the policy class is fixed: a factorized linear-softmax policy
  ``pi_W(r | x) = prod_l softmax(x @ W[l])[r_l]`` with ``W`` of shape (L, d, V);
  ``initial_weights`` plays the role of the base model;
* a response earns reward 1 iff the verifier accepts it. Training-prompt
  answers are never stored in the clear: the verifier compares a salted,
  iterated hash of the sampled response against the stored hash, so a
  learner only finds out what is correct by sampling and verifying.

Nothing in this module (or in ``data/seqverify/``) is read by the hidden test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "seqverify"

EXPERIMENTS = ("seqverify_main", "seqverify_hard")
TRAIN_FILES = {
    "seqverify_main": "train_main.json",
    "seqverify_hard": "train_hard.json",
}

# The fixed training protocol every comparable run follows (paper Sec. 4.1,
# scaled). The hidden test cannot observe training, so these limits are part
# of the task statement (see scoring/CONSTRAINTS.md), not something it checks.
PROTOCOL = {
    "batch_size": 64,            # prompts per policy update (paper: 512)
    "steps": 300,                # policy updates per run (paper: 600)
    "update_group_size": 4,      # responses per prompt in the backward pass (paper: n = 4)
    "max_samples_per_prompt": 32,  # verifier calls per prompt per step, upper limit (paper: N_max = 32)
    "lr": 0.03,                  # Adam learning rate, identical for every method (paper: AdamW 1e-6, untuned)
    "warmup_steps": 10,          # linear learning-rate warm-up (paper: 10 steps)
    "entropy_coef": 1e-4,        # entropy bonus (paper: 1e-4)
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "runs_per_experiment": 5,    # independent training runs (seeds) scored per method
}


@dataclass(frozen=True)
class Problem:
    d: int
    L: int
    V: int
    salt: bytes
    hash_iterations: int
    initial_weights: np.ndarray  # (L, d, V)


@dataclass(frozen=True)
class TrainSet:
    experiment: str
    uids: tuple
    features: np.ndarray  # (N, d)
    answer_hashes: tuple

    @property
    def num_prompts(self) -> int:
        return int(self.features.shape[0])


def load_problem(data_dir: Path = DATA_DIR) -> Problem:
    meta = json.loads((data_dir / "problem.json").read_text())
    weights = np.asarray(meta["initial_weights"], dtype=np.float64)
    return Problem(
        d=int(meta["d"]),
        L=int(meta["L"]),
        V=int(meta["V"]),
        salt=meta["salt"].encode(),
        hash_iterations=int(meta["hash"]["iterations"]),
        initial_weights=weights,
    )


def load_train_set(experiment: str, data_dir: Path = DATA_DIR) -> TrainSet:
    if experiment not in TRAIN_FILES:
        raise ValueError(f"unknown experiment {experiment!r}; expected one of {EXPERIMENTS}")
    raw = json.loads((data_dir / TRAIN_FILES[experiment]).read_text())
    return TrainSet(
        experiment=experiment,
        uids=tuple(raw["uids"]),
        features=np.asarray(raw["features"], dtype=np.float64),
        answer_hashes=tuple(raw["answer_hashes"]),
    )


def response_hash(salt: bytes, iterations: int, uid: str, response: Sequence[int]) -> str:
    token_bytes = bytes(int(t) for t in response)
    return hashlib.pbkdf2_hmac("sha256", token_bytes, salt + uid.encode(), iterations).hex()


class Verifier:
    """Binary verifier for the training prompts of one experiment."""

    def __init__(self, problem: Problem, train_set: TrainSet):
        self._salt = problem.salt
        self._iterations = problem.hash_iterations
        self._uids = train_set.uids
        self._hashes = train_set.answer_hashes
        self.calls = 0

    def __call__(self, prompt_index: int, responses: np.ndarray) -> np.ndarray:
        uid = self._uids[int(prompt_index)]
        target = self._hashes[int(prompt_index)]
        responses = np.atleast_2d(np.asarray(responses, dtype=np.int64))
        self.calls += int(responses.shape[0])
        return np.array(
            [1.0 if response_hash(self._salt, self._iterations, uid, r) == target else 0.0 for r in responses],
            dtype=np.float64,
        )


# ── Policy class ───────────────────────────────────────────────────────────────


def policy_probs(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Per-position token distributions, shape (B, L, V)."""
    logits = np.einsum("bd,ldv->blv", np.atleast_2d(features), weights)
    logits -= logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs


def sample_responses(rng: np.random.Generator, probs: np.ndarray, n: int) -> np.ndarray:
    """Draw n responses per prompt. probs (B, L, V) -> tokens (B, n, L)."""
    cdf = np.cumsum(probs, axis=-1)
    cdf[..., -1] = 1.0
    u = rng.random((probs.shape[0], n, probs.shape[1]))
    tokens = (u[..., None] > cdf[:, None, :, :]).sum(axis=-1)
    return np.minimum(tokens, probs.shape[-1] - 1)


def logprob_gradient(features: np.ndarray, responses: np.ndarray, probs: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Token-mean of coef_i * grad log pi(response_i | x_i) for R rows.

    features (R, d), responses (R, L), probs (R, L, V), coef (R,) -> (L, d, V).
    """
    R, L = responses.shape
    V = probs.shape[-1]
    onehot = np.zeros_like(probs)
    np.put_along_axis(onehot, responses[..., None].astype(np.int64), 1.0, axis=-1)
    delta = coef[:, None, None] * (onehot - probs)
    return np.einsum("rd,rlv->ldv", features, delta) / float(R * L)


def entropy_gradient(features: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Gradient of the token-mean policy entropy over R rows -> (L, d, V)."""
    logp = np.log(np.clip(probs, 1e-300, None))
    ent = -(probs * logp).sum(axis=-1, keepdims=True)
    dz = -probs * (logp + ent)
    R, L = probs.shape[:2]
    return np.einsum("rd,rlv->ldv", features, dz) / float(R * L)


class Adam:
    """Adam ascent step shared by every method (paper: AdamW, no tuning)."""

    def __init__(self, shape, betas=PROTOCOL["adam_betas"], eps=PROTOCOL["adam_eps"]):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0

    def ascend(self, params: np.ndarray, grad: np.ndarray, lr: float) -> None:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        params += lr * m_hat / (np.sqrt(v_hat) + self.eps)


def learning_rate(step: int, base_lr: float = PROTOCOL["lr"], warmup: int = PROTOCOL["warmup_steps"]) -> float:
    return base_lr * min(1.0, (step + 1) / float(max(1, warmup)))


def batch_schedule(seed: int, num_prompts: int, batch_size: int):
    """Prompt-batch stream; identical for every method given the same seed (paired runs)."""
    rng = np.random.default_rng([int(seed), 0])
    while True:
        yield rng.choice(num_prompts, size=batch_size, replace=False)


def sampling_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), 1])


def write_artifact(output_dir: Path, artifact: dict) -> Path:
    """Write a run artifact in the format the evaluators read."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{artifact['experiment']}_{artifact['method']}_seed{artifact['seed']}.json"
    path.write_text(json.dumps(artifact))
    return path
