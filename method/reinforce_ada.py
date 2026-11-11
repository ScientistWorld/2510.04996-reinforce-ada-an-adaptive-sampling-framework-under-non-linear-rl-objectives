"""Reinforce-Ada-Seq (Algorithm 2 of the paper) on the seqverify problem.

One training iteration (``train_step``):

Phase 1, adaptive sampling. Every prompt of the batch starts active. In each
round, every active prompt receives ``round_size`` (M) fresh responses from the
frozen policy snapshot, each checked by the verifier. A prompt leaves the
active set once its exit condition holds -- Seq-Pos: at least ``k_pos`` correct
responses; Seq-Balance: at least ``k_pos`` correct and ``k_neg`` incorrect -- or
once ``max_samples`` (N_max) responses have been drawn for it.

Phase 2, static batch. The pool mean ``p_hat = N_pos / N_x`` is the prompt's
global baseline. The pool is down-sampled to a fixed update group of ``n``
responses aiming for n/2 correct and n/2 incorrect (filling from the other side
when one is short). Each kept response gets advantage ``r - p_hat``, multiplied
by the explicit log-objective weight ``1 / p_hat`` that re-introduces the
weight removed by down-sampling (Sec. 3.2). The batch policy gradient is applied
with one Adam step (single on-policy update, so PPO ratio clipping is inactive).

Paper settings kept: N_max = 32 in 4 rounds of M = 8, update group n = 4,
K_pos = K_neg = n/2, same optimizer and learning rate as the GRPO baselines.
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

EXIT_RULES = {
    "seq_pos": lambda pos, neg, k_pos, k_neg: pos >= k_pos,
    "seq_balance": lambda pos, neg, k_pos, k_neg: pos >= k_pos and neg >= k_neg,
}


def adaptive_pools(rng, verifier, batch_ids, probs, variant, round_size, max_samples, k_pos, k_neg):
    """Phase 1: sequential sampling with elimination. Returns per-prompt (responses, rewards)."""
    exit_rule = EXIT_RULES[variant]
    B = len(batch_ids)
    responses = [[] for _ in range(B)]
    rewards = [[] for _ in range(B)]
    positives = np.zeros(B)
    drawn_count = np.zeros(B, dtype=np.int64)
    active = np.ones(B, dtype=bool)
    rounds_used = 0
    for _ in range(max(1, max_samples // round_size)):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        rounds_used += 1
        drawn = sample_responses(rng, probs[idx], round_size)
        for j, i in enumerate(idx):
            r = verifier(batch_ids[i], drawn[j])
            responses[i].append(drawn[j])
            rewards[i].append(r)
            positives[i] += r.sum()
            drawn_count[i] += len(r)
            if exit_rule(positives[i], drawn_count[i] - positives[i], k_pos, k_neg):
                active[i] = False
    pools = [(np.concatenate(responses[i]), np.concatenate(rewards[i])) for i in range(B)]
    return pools, rounds_used


def balanced_downsample(rng, rewards: np.ndarray, n: int) -> np.ndarray:
    """Indices of an n-response update group: n/2 correct and n/2 incorrect when available."""
    pos = rng.permutation(np.flatnonzero(rewards > 0.5))
    neg = rng.permutation(np.flatnonzero(rewards <= 0.5))
    take_pos = pos[: n // 2]
    take_neg = neg[: n - len(take_pos)]
    chosen = np.concatenate([take_pos, take_neg]).astype(np.int64)
    if len(chosen) < n:
        rest = np.concatenate([pos[len(take_pos):], neg[len(take_neg):]])
        chosen = np.concatenate([chosen, rest[: n - len(chosen)]]).astype(np.int64)
    if len(chosen) < n:  # pool smaller than n (cannot happen when round_size >= n)
        chosen = np.concatenate([chosen, rng.choice(len(rewards), size=n - len(chosen))]).astype(np.int64)
    return chosen


def explicit_weight(p_hat: float, weighting: str) -> float:
    if p_hat <= 0.0:
        return 0.0  # pool without a correct response: r - p_hat is zero for every sample anyway
    if weighting == "inverse":
        return 1.0 / p_hat
    if weighting == "sqrt":
        return 1.0 / np.sqrt(p_hat)
    if weighting == "none":
        return 1.0
    if weighting == "std":  # global-statistics normalisation, (r - p) / sqrt(p (1 - p)) (paper App. B, Remark 1)
        return 1.0 / np.sqrt(p_hat * (1.0 - p_hat)) if p_hat < 1.0 else 0.0
    raise ValueError(f"unknown weighting {weighting!r}")


def train(
    variant: str = "seq_balance",
    experiment: str = "seqverify_main",
    seed: int = 1,
    output_dir: Path = Path("checkpoints/all_results"),
    steps: int = PROTOCOL["steps"],
    lr: float = PROTOCOL["lr"],
    batch_size: int = PROTOCOL["batch_size"],
    update_group_size: int = PROTOCOL["update_group_size"],
    round_size: int = 8,
    max_samples: int = PROTOCOL["max_samples_per_prompt"],
    k_pos: int = 2,
    k_neg: int = 2,
    weighting: str = "inverse",
    entropy_coef: float = PROTOCOL["entropy_coef"],
    method_name: str | None = None,
    snapshot_steps=(),
    pool_hook=None,
) -> dict:
    """Train one run and write its artifact. ``pool_hook(batch_ids, pools)``, if
    given, may edit the Phase-1 pools before Phase 2 (used only by audit variants)."""
    problem = load_problem()
    train_set = load_train_set(experiment)
    verifier = Verifier(problem, train_set)
    features = train_set.features
    weights = problem.initial_weights.copy()
    optimizer = Adam(weights.shape)
    batches = batch_schedule(seed, train_set.num_prompts, batch_size)
    rng = sampling_rng(seed)
    n = update_group_size
    snapshot_steps = set(int(s) for s in snapshot_steps)
    snapshots, history = {}, []

    for step in range(steps):
        batch = next(batches)
        probs = policy_probs(weights, features[batch])  # frozen snapshot for the whole batch
        pools, rounds_used = adaptive_pools(
            rng, verifier, batch, probs, variant, round_size, max_samples, k_pos, k_neg
        )
        generated = sum(len(rewards) for _, rewards in pools)
        if pool_hook is not None:
            pools = pool_hook(batch, pools)
        rows_x, rows_resp, rows_p, rows_adv = [], [], [], []
        zero_variance, pool_means = 0, []
        for i, (responses, rewards) in enumerate(pools):
            p_hat = float(rewards.mean())
            pool_means.append(p_hat)
            chosen = balanced_downsample(rng, rewards, n)
            group_rewards = rewards[chosen]
            if group_rewards.min() == group_rewards.max():
                zero_variance += 1
            adv = (group_rewards - p_hat) * explicit_weight(p_hat, weighting)
            rows_x.append(np.repeat(features[batch[i]][None], n, axis=0))
            rows_resp.append(responses[chosen])
            rows_p.append(np.repeat(probs[i][None], n, axis=0))
            rows_adv.append(adv)
        rows_x = np.concatenate(rows_x)
        rows_p = np.concatenate(rows_p)
        grad = logprob_gradient(rows_x, np.concatenate(rows_resp), rows_p, np.concatenate(rows_adv))
        grad += entropy_coef * entropy_gradient(rows_x, rows_p)
        optimizer.ascend(weights, grad, learning_rate(step, lr))
        history.append({
            "step": step,
            "batch_size": int(batch_size),
            "generated": int(generated),
            "update_rows": int(len(rows_x)),
            "zero_variance_groups": int(zero_variance),
            "mean_pool_reward": float(np.mean(pool_means)),
            "rounds": int(rounds_used),
        })
        if step + 1 in snapshot_steps:
            snapshots[str(step + 1)] = weights.tolist()

    artifact = {
        "method": method_name or f"reinforce_ada_{variant}",
        "experiment": experiment,
        "seed": int(seed),
        "final_weights": weights.tolist(),
        "config": {
            "variant": variant, "steps": steps, "lr": lr, "batch_size": batch_size,
            "update_group_size": n, "round_size": round_size, "max_samples": max_samples,
            "k_pos": k_pos, "k_neg": k_neg, "weighting": weighting, "entropy_coef": entropy_coef,
        },
        "verifier_calls": int(verifier.calls),
        "history": history,
    }
    if snapshots:
        artifact["snapshots"] = snapshots
    write_artifact(Path(output_dir), artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(EXIT_RULES), default="seq_balance")
    parser.add_argument("--experiment", default="seqverify_main")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=PROTOCOL["steps"])
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/all_results"))
    args = parser.parse_args()
    artifact = train(args.variant, args.experiment, args.seed, args.output_dir, steps=args.steps)
    h = artifact["history"]
    print(json.dumps({
        "method": artifact["method"], "experiment": artifact["experiment"], "seed": artifact["seed"],
        "verifier_calls": artifact["verifier_calls"],
        "avg_generated_per_prompt": float(np.mean([r["generated"] / r["batch_size"] for r in h])),
        "final_mean_pool_reward": h[-1]["mean_pool_reward"],
    }))


if __name__ == "__main__":
    main()
