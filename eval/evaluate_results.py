"""Evaluate method artifacts without importing implementation code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def load_results(results_dir: Path) -> Dict[str, dict]:
    loaded: Dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        method = data["method"]
        loaded[method] = {
            "eval_pass1": data["eval_pass1"],
            "avg_generated_per_prompt": data["avg_generated_per_prompt"],
            "update_samples_per_prompt": data["update_samples_per_prompt"],
            "signal_collapse_rate": data.get(
                "signal_collapse_rate",
                data.get("signal_collapse_rate_initial_group"),
            ),
        }
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
