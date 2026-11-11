"""Canonical visible scoring (scoring/scores.json) for seqverify artifacts.

Scored metrics come from the visible evaluator (eval/train/evaluate.py).
In addition, three training-cost diagnostics are summarised from each run's
self-reported ``history`` (verifier calls, update-group size, zero-variance
update groups). They document the paper's cost claims for the reproduction,
but they are written by the method and cannot be checked from its outputs, so
they carry a zero coefficient and are never part of the hidden test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.train.evaluate import EXPERIMENTS, DevSet, score

DIAGNOSTICS = ("avg_generated_per_prompt", "update_samples_per_prompt", "signal_collapse_rate")


def history_diagnostics(history) -> dict | None:
    try:
        rows = list(history)
        if not rows:
            return None
        return {
            "avg_generated_per_prompt": float(np.mean([h["generated"] / h["batch_size"] for h in rows])),
            "update_samples_per_prompt": float(np.mean([h["update_rows"] / h["batch_size"] for h in rows])),
            "signal_collapse_rate": float(np.mean([h["zero_variance_groups"] / h["batch_size"] for h in rows])),
        }
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def self_reported(results_dir: Path) -> dict:
    out: dict = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict) or data.get("experiment") not in EXPERIMENTS:
            continue
        diag = history_diagnostics(data.get("history", []))
        if diag is not None and isinstance(data.get("method"), str):
            out.setdefault(data["experiment"], {}).setdefault(data["method"], []).append(diag)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("checkpoints/all_results"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()
    experiments, _ = score(args.results_dir, DevSet())
    reported = self_reported(args.results_dir)
    for experiment, methods in reported.items():
        for method, rows in methods.items():
            if method in experiments[experiment]["results"]:
                for key in DIAGNOSTICS:
                    experiments[experiment]["results"][method][key] = float(np.mean([r[key] for r in rows]))
    scores = {
        "_metadata": {
            "slice": "visible dev prompts (same estimator as scores_train.json)",
            "diagnostics": "avg_generated_per_prompt, update_samples_per_prompt and signal_collapse_rate are "
                           "self-reported by each run and unscored (coefficient 0)",
        },
        "experiments": experiments,
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
