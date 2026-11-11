"""Visible (development) scorer for seqverify run artifacts.

Same artifact contract as the hidden test: reads every ``*.json`` in the
results directory, scores every method found, and writes
``scoring/scores_train.json``. Only ``method``, ``experiment`` and
``final_weights`` are read from an artifact.

The visible prompts in ``eval/train/dev_set.json`` come from the same
distribution as the hidden test prompts, but their answers are stored only as
verifier hashes. Pass rates are therefore estimated the way the paper reports
them: sample ``MC_SAMPLES`` responses per prompt (Ave@32, temperature 1) with a
fixed random stream and verify each one. ``pass8`` uses the unbiased pass@k
estimator over those samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEV_SET = HERE / "dev_set.json"

EXPERIMENTS = ("seqverify_main", "seqverify_hard")
REQUIRED_RUNS = 5
HARD_FRACTION = 0.40
EASY_FRACTION = 0.25
PASS_K = 8
MC_SAMPLES = 32
MC_SEED = 20251006
METRICS = ("pass1", "hard_pass1", "easy_pass1", "pass8")
WORST = {metric: 0.0 for metric in METRICS}


class DevSet:
    def __init__(self, path: Path = DEV_SET):
        raw = json.loads(path.read_text())
        self.features = np.asarray(raw["features"], dtype=np.float64)
        self.uids = list(raw["uids"])
        self.hashes = list(raw["answer_hashes"])
        self.salt = raw["salt"].encode()
        self.iterations = int(raw["hash_iterations"])
        self.shape = (int(raw["L"]), int(raw["d"]), int(raw["V"]))
        base = np.asarray(raw["base_pass_rate"], dtype=np.float64)
        n = len(base)
        order = np.argsort(base, kind="stable")
        self.hard = np.sort(order[: int(round(n * HARD_FRACTION))])
        self.easy = np.sort(order[n - int(round(n * EASY_FRACTION)):])
        self._pass_k_table = np.array(
            [1.0 - comb(MC_SAMPLES - c, PASS_K) / comb(MC_SAMPLES, PASS_K) for c in range(MC_SAMPLES + 1)]
        )

    def correct_counts(self, weights: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            logits = np.einsum("nd,ldv->nlv", self.features, weights)
            logits = logits - logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / probs.sum(axis=-1, keepdims=True)
        if not np.all(np.isfinite(probs)):
            raise ValueError("weights produce non-finite token probabilities")
        cdf = np.cumsum(probs, axis=-1)
        cdf[..., -1] = 1.0
        rng = np.random.default_rng(MC_SEED)
        u = rng.random((probs.shape[0], MC_SAMPLES, probs.shape[1]))
        tokens = np.minimum((u[..., None] > cdf[:, None]).sum(-1), probs.shape[-1] - 1)
        counts = np.zeros(len(self.uids), dtype=np.int64)
        for i, (uid, target) in enumerate(zip(self.uids, self.hashes)):
            seen = {}
            salt = self.salt + uid.encode()
            for row in tokens[i]:
                key = row.tobytes()
                if key not in seen:
                    digest = hashlib.pbkdf2_hmac("sha256", bytes(int(t) for t in row), salt, self.iterations).hex()
                    seen[key] = digest == target
                counts[i] += int(seen[key])
        return counts

    def metrics(self, weights: np.ndarray) -> dict:
        counts = self.correct_counts(weights)
        p = counts / float(MC_SAMPLES)
        pk = self._pass_k_table[counts]
        return {
            "pass1": float(p.mean()),
            "hard_pass1": float(p[self.hard].mean()),
            "easy_pass1": float(p[self.easy].mean()),
            "pass8": float(pk.mean()),
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


def weights_key(weights: np.ndarray) -> str:
    return hashlib.sha256((np.round(weights, 6) + 0.0).tobytes()).hexdigest()


def read_artifacts(results_dir: Path) -> list:
    """Return (method, experiment, weights-or-error) records for every artifact."""
    records = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = None
        if not isinstance(data, dict):
            records.append((f"unreadable:{path.stem}", None, f"{path.name}: unreadable artifact"))
            continue
        method = data.get("method")
        if not isinstance(method, str) or not method.strip() or method.startswith("_"):
            records.append((f"unnamed:{path.stem}", None, f"{path.name}: invalid method name"))
            continue
        records.append((method, data.get("experiment"), data.get("final_weights"), path.name))
    return records


def score(results_dir: Path, dev: DevSet) -> tuple:
    per_method: dict = {}
    for record in read_artifacts(results_dir) if results_dir.is_dir() else []:
        if len(record) == 3:  # unreadable / unnamed: failed run in every experiment
            method, _, reason = record
            state = per_method.setdefault(method, {"runs": {}, "failed": {}, "seen": {}, "ignored": []})
            for experiment in EXPERIMENTS:
                state["failed"].setdefault(experiment, []).append(reason)
            continue
        method, experiment, raw_weights, name = record
        state = per_method.setdefault(method, {"runs": {}, "failed": {}, "seen": {}, "ignored": []})
        if experiment not in EXPERIMENTS:
            state["ignored"].append(f"{name}: experiment {experiment!r} is not scored")
            continue
        weights, error = parse_weights(raw_weights, dev.shape)
        if error is None:
            key = weights_key(weights)
            if key in state["seen"].setdefault(experiment, set()):
                continue
            try:
                metrics = dev.metrics(weights)
            except ValueError as exc:
                error = str(exc)
            else:
                state["seen"][experiment].add(key)
                state["runs"].setdefault(experiment, []).append(metrics)
                continue
        state["failed"].setdefault(experiment, []).append(f"{name}: {error}")

    experiments = {e: {"results": {}} for e in EXPERIMENTS}
    diagnostics = {}
    for method in sorted(per_method):
        state = per_method[method]
        diagnostics[method] = {"ignored_artifacts": state["ignored"]}
        for experiment in EXPERIMENTS:
            runs = state["runs"].get(experiment, [])
            failed = state["failed"].get(experiment, [])
            slots = runs + [WORST] * len(failed)
            padded = max(0, REQUIRED_RUNS - len(slots))
            slots += [WORST] * padded
            experiments[experiment]["results"][method] = {
                metric: float(np.mean([s[metric] for s in slots])) for metric in METRICS
            }
            diagnostics[method][experiment] = {
                "valid_runs": len(runs),
                "failed_runs": len(failed),
                "missing_runs_filled_with_worst": padded,
                "failure_reasons": failed[:20],
            }
    return experiments, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("checkpoints/all_results"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.unlink(missing_ok=True)
    dev = DevSet()
    experiments, diagnostics = score(args.results_dir, dev)
    scores = {
        "_metadata": {
            "slice": "train",
            "dev_prompts": len(dev.uids),
            "samples_per_prompt": MC_SAMPLES,
            "required_runs_per_experiment": REQUIRED_RUNS,
            "worst_values": WORST,
            "pass_rate": "Ave@32 estimate of the probability that a sampled response is correct",
        },
        "_diagnostics": diagnostics,
        "experiments": experiments,
    }
    args.scores.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps({e: v["results"] for e, v in experiments.items()}, indent=2))
    return 0 if diagnostics else 1


if __name__ == "__main__":
    raise SystemExit(main())
