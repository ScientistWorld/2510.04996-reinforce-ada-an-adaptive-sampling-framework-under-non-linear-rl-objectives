"""Held-out test-slice evaluator for synthetic signal recovery artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from data.contextual_bandit import make_dataset, softmax, subset_indices


EXPERIMENT = "synthetic_signal_recovery"
SPLIT_SEED = 9071


def heldout_eval_ids(num_train: int, num_eval: int, seed: int) -> np.ndarray:
    dataset = make_dataset(num_train=num_train, num_eval=num_eval, seed=seed)
    eval_ids = subset_indices(dataset, "eval")
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(eval_ids)
    return np.sort(shuffled[len(shuffled) // 2 :])


def history_metrics(history: Iterable[dict]) -> dict:
    rows = list(history)
    if not rows:
        raise ValueError("artifact missing history")
    return {
        "avg_generated_per_prompt": float(np.mean([h["generated_total"] / h["batch_size"] for h in rows])),
        "update_samples_per_prompt": float(np.mean([h["update_group_size"] for h in rows])),
        "signal_collapse_rate": float(np.mean([h["zero_signal_update_groups"] / h["batch_size"] for h in rows])),
    }


def recompute_metrics(data: dict) -> dict:
    if "final_weights" not in data:
        raise ValueError(f"{data.get('method', '<unknown>')} artifact missing final_weights")
    seed = int(data.get("seed", 17))
    num_train = int(data.get("num_train", 320))
    num_eval = int(data.get("num_eval", 160))
    dataset = make_dataset(num_train=num_train, num_eval=num_eval, seed=seed)
    prompt_ids = heldout_eval_ids(num_train, num_eval, seed)
    weights = np.asarray(data["final_weights"], dtype=np.float64)
    probs = softmax(dataset.features[prompt_ids] @ weights)
    correct_probs = probs[np.arange(len(prompt_ids)), dataset.correct_actions[prompt_ids]]

    metrics = {
        "eval_pass1": float(correct_probs.mean()),
    }
    metrics.update(history_metrics(data.get("history", [])))
    return metrics


def train_score_methods(train_scores: Path) -> list[str]:
    if not train_scores.is_file():
        return []
    data = json.loads(train_scores.read_text())
    results = data["experiments"][EXPERIMENT]["results"]
    return sorted(results)


def load_artifacts(results_dir: Path) -> Dict[str, dict]:
    artifacts: Dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        artifacts[data["method"]] = data
    if not artifacts:
        raise FileNotFoundError(f"no result JSON files found in {results_dir}")
    return artifacts


def load_results(results_dir: Path, train_scores: Path) -> Dict[str, dict]:
    artifacts = load_artifacts(results_dir)
    methods = train_score_methods(train_scores) or sorted(artifacts)
    missing = [method for method in methods if method not in artifacts]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"missing artifacts for train-scored methods: {missing_list}")
    return {method: recompute_metrics(artifacts[method]) for method in methods}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("checkpoints/all_results"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_test.json"))
    parser.add_argument("--train-scores", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    scores = {
        "_metadata": {
            "slice": "test",
            "split_seed": SPLIT_SEED,
            "split": "second half of a deterministic shuffle of held-out eval prompts",
        },
        "experiments": {
            EXPERIMENT: {
                "results": load_results(args.results_dir, args.train_scores),
            }
        },
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
