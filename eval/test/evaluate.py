"""Hidden-test scorer for seqverify run artifacts.

Reads every ``*.json`` file in the results directory (default
``checkpoints/all_results``), scores every method found there and writes
``scoring/scores_test.json``.

Scored metrics are greedy (argmax-decoding) accuracies: the fraction of
prompts whose most likely response is verified correct, overall and on fixed
hard / easy slices (by base-policy pass rate). Greedy accuracy is the limit of
Pass@1 as the sampling temperature goes to zero, so rescaling a policy's
weights (changing its temperature) cannot move it. ``pass1`` -- the exact
temperature-1 Pass@1 the paper reports -- is written as an unscored diagnostic.

Only three fields of an artifact are read: ``method`` (name), ``experiment``
and ``final_weights`` (the trained policy, shape (L, d, V)). Every scored
number is recomputed here from those weights against the hidden prompts and
plaintext answers in ``eval/test/test_set.json``; nothing else the method wrote
is trusted. Everything on the other side of the comparison is fixed here: the
test prompts and answers, the difficulty slices, the number of runs per method
and the metric definitions.

Failure handling: an artifact whose weights are missing, non-numeric, of the
wrong shape, non-finite, or produce impossible pass rates counts as a failed
run and contributes the worst value (0) of every metric. Each method must
supply ``REQUIRED_RUNS`` distinct runs per experiment; missing runs (including
an experiment the method skipped entirely, and exact duplicates of another run)
are filled with worst values. Artifacts naming an unknown experiment are
ignored, so they never stand in for a scored one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST_SET = HERE / "test_set.json"

EXPERIMENTS = ("seqverify_main", "seqverify_hard")
REQUIRED_RUNS = 5
HARD_FRACTION = 0.40
EASY_FRACTION = 0.25
METRICS = ("greedy_acc", "hard_greedy_acc", "easy_greedy_acc", "pass1")
WORST = {metric: 0.0 for metric in METRICS}


class HiddenTest:
    def __init__(self, path: Path = TEST_SET):
        raw = json.loads(path.read_text())
        self.features = np.asarray(raw["features"], dtype=np.float64)
        self.answers = np.asarray(raw["answers"], dtype=np.int64)
        base = np.asarray(raw["base_pass_rate"], dtype=np.float64)
        self.shape = (int(raw["L"]), int(raw["d"]), int(raw["V"]))
        n = len(base)
        order = np.argsort(base, kind="stable")
        self.hard = np.sort(order[: int(round(n * HARD_FRACTION))])
        self.easy = np.sort(order[n - int(round(n * EASY_FRACTION)):])

    def metrics(self, weights: np.ndarray) -> dict:
        with np.errstate(all="ignore"):
            logits = np.einsum("nd,ldv->nlv", self.features, weights)
            if not np.all(np.isfinite(logits)):
                raise ValueError("weights produce non-finite logits")
            greedy = logits.argmax(axis=-1)
            logits = logits - logits.max(axis=-1, keepdims=True)
            log_norm = np.log(np.exp(logits).sum(axis=-1, keepdims=True))
            picked = np.take_along_axis(logits - log_norm, self.answers[..., None], axis=-1)[..., 0]
            p = np.exp(picked.sum(axis=-1))
        if not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0 + 1e-9:
            raise ValueError("weights produce impossible pass rates")
        correct = (greedy == self.answers).all(axis=-1).astype(np.float64)
        return {
            "greedy_acc": float(correct.mean()),
            "hard_greedy_acc": float(correct[self.hard].mean()),
            "easy_greedy_acc": float(correct[self.easy].mean()),
            "pass1": float(np.clip(p, 0.0, 1.0).mean()),
        }


def parse_weights(raw, shape):
    if raw is None:
        return None, "missing final_weights"
    try:
        weights = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None, "final_weights is not a numeric array"
    if weights.shape != shape:
        return None, f"final_weights shape {weights.shape} != required {shape}"
    if not np.all(np.isfinite(weights)):
        return None, "final_weights contains non-finite values"
    return weights, None


def fingerprint(weights: np.ndarray) -> str:
    return hashlib.sha256((np.round(weights, 6) + 0.0).tobytes()).hexdigest()


def new_entry() -> dict:
    return {
        "runs": {e: [] for e in EXPERIMENTS},
        "failed": {e: [] for e in EXPERIMENTS},
        "fingerprints": {e: set() for e in EXPERIMENTS},
        "duplicates": {e: 0 for e in EXPERIMENTS},
        "ignored": [],
    }


def collect(results_dir: Path, test: HiddenTest) -> dict:
    entries: dict = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # unreadable artifact: failed run of its own entry
            data = None
            reason = f"{path.name}: unreadable JSON ({type(exc).__name__})"
        if not isinstance(data, dict):
            entry = entries.setdefault(f"unreadable:{path.stem}", new_entry())
            for experiment in EXPERIMENTS:
                entry["failed"][experiment].append(reason if data is None else f"{path.name}: not a JSON object")
            continue
        method = data.get("method")
        if not isinstance(method, str) or not method.strip() or method.startswith("_"):
            entry = entries.setdefault(f"unnamed:{path.stem}", new_entry())
            for experiment in EXPERIMENTS:
                entry["failed"][experiment].append(f"{path.name}: invalid method name")
            continue
        entry = entries.setdefault(method, new_entry())
        experiment = data.get("experiment")
        if experiment not in EXPERIMENTS:
            entry["ignored"].append(f"{path.name}: experiment {experiment!r} is not scored")
            continue
        weights, error = parse_weights(data.get("final_weights"), test.shape)
        if error is None:
            try:
                metrics = test.metrics(weights)
            except ValueError as exc:
                error = str(exc)
        if error is not None:
            entry["failed"][experiment].append(f"{path.name}: {error}")
            continue
        key = fingerprint(weights)
        if key in entry["fingerprints"][experiment]:
            entry["duplicates"][experiment] += 1
            continue
        entry["fingerprints"][experiment].add(key)
        entry["runs"][experiment].append(metrics)
    return entries


def aggregate(entries: dict):
    experiments = {e: {"results": {}} for e in EXPERIMENTS}
    diagnostics = {}
    for method in sorted(entries):
        entry = entries[method]
        diagnostics[method] = {"ignored_artifacts": entry["ignored"]}
        for experiment in EXPERIMENTS:
            slots = list(entry["runs"][experiment]) + [WORST] * len(entry["failed"][experiment])
            padded = max(0, REQUIRED_RUNS - len(slots))
            slots += [WORST] * padded
            experiments[experiment]["results"][method] = {
                metric: float(np.mean([slot[metric] for slot in slots])) for metric in METRICS
            }
            diagnostics[method][experiment] = {
                "valid_runs": len(entry["runs"][experiment]),
                "failed_runs": len(entry["failed"][experiment]),
                "duplicate_runs_dropped": entry["duplicates"][experiment],
                "missing_runs_filled_with_worst": padded,
                "failure_reasons": entry["failed"][experiment][:20],
            }
    return experiments, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("checkpoints/all_results"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_test.json"))
    args = parser.parse_args()

    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.unlink(missing_ok=True)  # never leave a stale score file behind

    test = HiddenTest()
    entries = collect(args.results_dir, test) if args.results_dir.is_dir() else {}
    experiments, diagnostics = aggregate(entries)
    scores = {
        "_metadata": {
            "slice": "test",
            "test_prompts": int(len(test.answers)),
            "hard_prompts": int(len(test.hard)),
            "easy_prompts": int(len(test.easy)),
            "required_runs_per_experiment": REQUIRED_RUNS,
            "worst_values": WORST,
            "greedy_acc": "fraction of prompts whose argmax response is correct (scored)",
            "pass1": "exact temperature-1 Pass@1 (diagnostic, unscored)",
        },
        "_diagnostics": diagnostics,
        "experiments": experiments,
    }
    if not entries:
        scores["_metadata"]["error"] = f"no artifacts found in {args.results_dir}"
    args.scores.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps({e: v["results"] for e, v in experiments.items()}, indent=2))
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
