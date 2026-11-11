"""Build the seqverify problem instance (one-off tool; outputs are committed).

This file lives in eval/test/ because it can regenerate the hidden test
answers: the teacher that defines every correct response is drawn from
SECRET_SEED and is never written anywhere public.

Outputs
  data/seqverify/problem.json     public: sizes, salt, hash scheme, base-policy weights
  data/seqverify/train_main.json  public: training prompts (features + answer hashes),
                                  paper's difficulty filter (1-6 of 16 base samples correct)
  data/seqverify/train_hard.json  public: hard training prompts (1-2 of 16 correct), paper Table 4
  eval/train/dev_set.json         visible evaluation prompts (features + answer hashes)
  eval/test/test_set.json         hidden evaluation prompts (features + plaintext answers)
  eval/test/build_summary.json    hidden: summary statistics of the generated sets

Usage: python eval/test/build_problem.py   (deterministic; overwrites the files)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SECRET_SEED = 482_913_557_061

D, L, V = 16, 8, 16
ALPHA = 6.0          # weight of the teacher in the base policy
BETA = 1.5           # weight of the base policy's own (wrong) structure
LABEL_NOISE = 0.05   # small per-token label noise -> a pass-rate ceiling below 1
N_TEST, N_DEV, N_TRAIN = 4000, 1000, 4000
FILTER_SAMPLES = 16
FILTER_RANGES = {"seqverify_main": (1, 6), "seqverify_hard": (1, 2)}
TRAIN_FILES = {"seqverify_main": "train_main.json", "seqverify_hard": "train_hard.json"}
UID_PREFIX = {"seqverify_main": "main", "seqverify_hard": "hard"}
SALT = "reinforce-ada/seqverify/v1"
HASH_ITERATIONS = 32


def response_hash(uid: str, tokens) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", bytes(int(t) for t in tokens), SALT.encode() + uid.encode(), HASH_ITERATIONS
    ).hex()


def log_probs(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    logits = np.einsum("nd,ldv->nlv", x, weights)
    logits -= logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


def answer_pass_rate(weights: np.ndarray, x: np.ndarray, answers: np.ndarray) -> np.ndarray:
    lp = log_probs(weights, x)
    picked = np.take_along_axis(lp, answers[..., None], axis=-1)[..., 0]
    return np.exp(picked.sum(axis=-1))


def main() -> None:
    rng = np.random.default_rng(SECRET_SEED)
    teacher = rng.normal(size=(L, D, V))
    base_noise = rng.normal(size=(L, D, V))
    w0 = np.round(ALPHA * teacher + BETA * base_noise, 6)

    def draw(n: int):
        x = rng.normal(size=(n, D))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        x = np.round(x, 6)
        clean = np.einsum("nd,ldv->nlv", x, teacher)
        y = np.argmax(clean + LABEL_NOISE * rng.normal(size=clean.shape), axis=-1)
        return x, y.astype(np.int64)

    def base_correct_counts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        probs = np.exp(log_probs(w0, x))
        cdf = np.cumsum(probs, axis=-1)
        cdf[..., -1] = 1.0
        u = rng.random((x.shape[0], FILTER_SAMPLES, L))
        tokens = np.minimum((u[..., None] > cdf[:, None]).sum(-1), V - 1)
        return (tokens == y[:, None, :]).all(axis=-1).sum(axis=-1)

    summary = {"d": D, "L": L, "V": V, "alpha": ALPHA, "beta": BETA, "label_noise": LABEL_NOISE}

    test_x, test_y = draw(N_TEST)
    test_p0 = answer_pass_rate(w0, test_x, test_y)
    dev_x, dev_y = draw(N_DEV)
    dev_p0 = answer_pass_rate(w0, dev_x, dev_y)

    train_sets = {}
    for experiment, (lo, hi) in FILTER_RANGES.items():
        kept_x, kept_y, drawn = [], [], 0
        while sum(len(k) for k in kept_x) < N_TRAIN:
            x, y = draw(4000)
            drawn += len(x)
            counts = base_correct_counts(x, y)
            keep = (counts >= lo) & (counts <= hi)
            kept_x.append(x[keep])
            kept_y.append(y[keep])
        x = np.concatenate(kept_x)[:N_TRAIN]
        y = np.concatenate(kept_y)[:N_TRAIN]
        train_sets[experiment] = (x, y)
        p0 = answer_pass_rate(w0, x, y)
        summary[experiment] = {
            "candidates_drawn": drawn,
            "kept_fraction": float(sum(len(k) for k in kept_x) / drawn),
            "base_pass_rate_mean": float(p0.mean()),
            "base_pass_rate_quartiles": [float(q) for q in np.quantile(p0, [0.25, 0.5, 0.75])],
        }

    teacher_policy = 64.0 * teacher
    summary["test"] = {
        "size": N_TEST,
        "base_pass_rate_mean": float(test_p0.mean()),
        "base_pass_rate_quartiles": [float(q) for q in np.quantile(test_p0, [0.25, 0.5, 0.75])],
        "sharpened_base_x3_pass_rate": float(answer_pass_rate(3.0 * w0, test_x, test_y).mean()),
        "teacher_policy_pass_rate": float(answer_pass_rate(teacher_policy, test_x, test_y).mean()),
    }
    summary["dev"] = {"size": N_DEV, "base_pass_rate_mean": float(dev_p0.mean())}

    data_dir = ROOT / "data" / "seqverify"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "problem.json").write_text(json.dumps({
        "name": "seqverify",
        "description": "Sequence-response binary-verifier RL problem; see data/seqverify.py.",
        "d": D, "L": L, "V": V,
        "salt": SALT,
        "hash": {"scheme": "pbkdf2_hmac_sha256(password=bytes(tokens), salt=salt+uid)", "iterations": HASH_ITERATIONS},
        "initial_weights": w0.tolist(),
    }))
    for experiment, (x, y) in train_sets.items():
        lo, hi = FILTER_RANGES[experiment]
        uids = [f"{UID_PREFIX[experiment]}-{i:05d}" for i in range(len(x))]
        (data_dir / TRAIN_FILES[experiment]).write_text(json.dumps({
            "experiment": experiment,
            "filter": f"kept prompts with {lo}-{hi} of {FILTER_SAMPLES} base-policy samples correct",
            "uids": uids,
            "features": x.tolist(),
            "answer_hashes": [response_hash(u, t) for u, t in zip(uids, y)],
        }))

    dev_uids = [f"dev-{i:05d}" for i in range(N_DEV)]
    (ROOT / "eval" / "train" / "dev_set.json").write_text(json.dumps({
        "description": "Visible evaluation prompts: answers only as verifier hashes.",
        "d": D, "L": L, "V": V, "salt": SALT, "hash_iterations": HASH_ITERATIONS,
        "uids": dev_uids,
        "features": dev_x.tolist(),
        "answer_hashes": [response_hash(u, t) for u, t in zip(dev_uids, dev_y)],
        "base_pass_rate": dev_p0.tolist(),
    }))
    (ROOT / "eval" / "test" / "test_set.json").write_text(json.dumps({
        "description": "Hidden evaluation prompts with plaintext answers.",
        "d": D, "L": L, "V": V,
        "features": test_x.tolist(),
        "answers": test_y.tolist(),
        "base_pass_rate": test_p0.tolist(),
    }))
    (ROOT / "eval" / "test" / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
