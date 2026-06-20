"""Evaluate method artifacts without importing implementation code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from data.contextual_bandit import make_dataset, softmax, subset_indices


def recompute_metrics(data: dict) -> dict:
    if "final_weights" not in data:
        raise ValueError(f"{data.get('method', '<unknown>')} artifact missing final_weights")
    seed = int(data.get("seed", 17))
    num_train = int(data.get("num_train", 320))
    num_eval = int(data.get("num_eval", 160))
    dataset = make_dataset(num_train=num_train, num_eval=num_eval, seed=seed)
    eval_ids = subset_indices(dataset, "eval")
    weights = np.asarray(data["final_weights"], dtype=np.float64)
    eval_probs = softmax(dataset.features[eval_ids] @ weights)
    correct_probs = eval_probs[np.arange(len(eval_ids)), dataset.correct_actions[eval_ids]]

    history = data.get("history", [])
    if not history:
        raise ValueError(f"{data.get('method', '<unknown>')} artifact missing history")
    avg_generated = np.mean([h["generated_total"] / h["batch_size"] for h in history])
    update_samples = np.mean([h["update_group_size"] for h in history])
    collapse = np.mean([h["zero_signal_update_groups"] / h["batch_size"] for h in history])

    return {
        "eval_pass1": float(correct_probs.mean()),
        "avg_generated_per_prompt": float(avg_generated),
        "update_samples_per_prompt": float(update_samples),
        "signal_collapse_rate": float(collapse),
    }


def load_results(results_dir: Path) -> Dict[str, dict]:
    loaded: Dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        method = data["method"]
        loaded[method] = recompute_metrics(data)
    if not loaded:
        raise FileNotFoundError(f"no result JSON files found in {results_dir}")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("checkpoints/all_results"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()
    scores = {
        "experiments": {
            "synthetic_signal_recovery": {
                "results": load_results(args.results_dir),
            }
        }
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
