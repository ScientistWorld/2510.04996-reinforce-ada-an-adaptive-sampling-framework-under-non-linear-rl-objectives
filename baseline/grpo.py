"""Fixed-group GRPO baseline on the seqverify problem.

Each prompt of the batch receives ``group_size`` responses from the frozen
policy snapshot; advantages are group-normalised, ``(r - mean) / (std + 1e-6)``
with the unbiased group standard deviation (verl's GRPO), and are zero when
every response in the group has the same reward. Same optimizer, learning rate,
warm-up, entropy bonus and prompt-batch stream as Reinforce-Ada.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.seqverify import (
    PROTOCOL,
    Adam,
    Verifier,
    batch_schedule,
    entropy_gradient,
    learning_rate,
    load_problem,
    load_train_set,
    logprob_gradient,
    policy_probs,
    sample_responses,
    sampling_rng,
    write_artifact,
)


def train_grpo(
    group_size: int = 4,
    experiment: str = "seqverify_main",
    seed: int = 1,
    output_dir: Path = Path("checkpoints/all_results"),
    steps: int = PROTOCOL["steps"],
    lr: float = PROTOCOL["lr"],
    batch_size: int = PROTOCOL["batch_size"],
    entropy_coef: float = PROTOCOL["entropy_coef"],
    method_name: str | None = None,
    snapshot_steps=(),
) -> dict:
    problem = load_problem()
    train_set = load_train_set(experiment)
    verifier = Verifier(problem, train_set)
    features = train_set.features
    weights = problem.initial_weights.copy()
    optimizer = Adam(weights.shape)
    batches = batch_schedule(seed, train_set.num_prompts, batch_size)
    rng = sampling_rng(seed)
    L = problem.L
    snapshot_steps = set(int(s) for s in snapshot_steps)
    snapshots, history = {}, []

    for step in range(steps):
        batch = next(batches)
        probs = policy_probs(weights, features[batch])
        drawn = sample_responses(rng, probs, group_size)  # (B, n, L)
        rewards = np.stack([verifier(batch[i], drawn[i]) for i in range(len(batch))])
        mean = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, ddof=1, keepdims=True) if group_size > 1 else np.zeros_like(mean)
        adv = np.where(std > 0, (rewards - mean) / (std + 1e-6), 0.0)
        rows_x = np.repeat(features[batch], group_size, axis=0)
        rows_p = np.repeat(probs, group_size, axis=0)
        grad = logprob_gradient(rows_x, drawn.reshape(-1, L), rows_p, adv.reshape(-1))
        grad += entropy_coef * entropy_gradient(rows_x, rows_p)
        optimizer.ascend(weights, grad, learning_rate(step, lr))
        history.append({
            "step": step,
            "batch_size": int(batch_size),
            "generated": int(rewards.size),
            "update_rows": int(rewards.size),
            "zero_variance_groups": int((std[:, 0] == 0).sum()),
            "mean_pool_reward": float(rewards.mean()),
            "rounds": 1,
        })
        if step + 1 in snapshot_steps:
            snapshots[str(step + 1)] = weights.tolist()

    artifact = {
        "method": method_name or f"grpo_n{group_size}",
        "experiment": experiment,
        "seed": int(seed),
        "final_weights": weights.tolist(),
        "config": {"group_size": group_size, "steps": steps, "lr": lr, "batch_size": batch_size,
                   "entropy_coef": entropy_coef},
        "verifier_calls": int(verifier.calls),
        "history": history,
    }
    if snapshots:
        artifact["snapshots"] = snapshots
    write_artifact(Path(output_dir), artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--experiment", default="seqverify_main")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=PROTOCOL["steps"])
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/all_results"))
    args = parser.parse_args()
    artifact = train_grpo(args.group_size, args.experiment, args.seed, args.output_dir, steps=args.steps)
    print(json.dumps({
        "method": artifact["method"], "experiment": artifact["experiment"], "seed": artifact["seed"],
        "verifier_calls": artifact["verifier_calls"],
        "final_mean_pool_reward": artifact["history"][-1]["mean_pool_reward"],
    }))


if __name__ == "__main__":
    main()
