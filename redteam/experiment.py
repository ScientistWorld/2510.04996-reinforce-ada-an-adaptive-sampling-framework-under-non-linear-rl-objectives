"""Real audit attempts: stage outputs, then run the unchanged hidden test.

Payload preparation uses visible data/baseline/method interfaces only. Test
evaluation always runs as a separate process with no arguments. No process or
Python-environment modification is used. All temporary gym edits are restored.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
RESULTS = ROOT / 'checkpoints/all_results'
EXP = 'synthetic_signal_recovery'
SEEDS = [17, 23, 31]
REFERENCE = 'reinforce_ada_seq_balance'


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True) + '\n')


def source_hashes():
    paths = ['scripts/evaluate_test.sh', 'eval/test/evaluate.py',
             'eval/train/evaluate.py', 'eval/evaluate_results.py']
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


@contextlib.contextmanager
def isolated_attempt(name):
    work = AUDIT / '.runtime' / name
    work.mkdir(parents=True, exist_ok=True)
    paths = ['scoring/scores_train.json', 'scoring/scores_test.json',
             'data/contextual_bandit.py']
    originals = {p: (ROOT / p).read_bytes() if (ROOT / p).exists() else None for p in paths}
    before_hashes = source_hashes()
    backup = work / 'original_results'
    if backup.exists():
        raise RuntimeError(f'Unrestored previous run: {backup}')
    had_results = RESULTS.exists()
    if had_results:
        RESULTS.rename(backup)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for p in ['scoring/scores_train.json', 'scoring/scores_test.json']:
        (ROOT / p).unlink(missing_ok=True)
    try:
        yield before_hashes
    finally:
        shutil.rmtree(RESULTS, ignore_errors=True)
        if had_results:
            backup.rename(RESULTS)
        for p, content in originals.items():
            if content is None:
                (ROOT / p).unlink(missing_ok=True)
            else:
                (ROOT / p).write_bytes(content)
        if before_hashes != source_hashes():
            raise RuntimeError('Audit changed a protected evaluator file')


def train_case(name, kind, steps, lr=0.42):
    from method.reinforce_ada import train_one
    from baseline.grpo import train_grpo
    cache = AUDIT / name / 'artifacts'
    if len(list(cache.glob('*.json'))) != 3:
        cache.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            if kind == 'ada':
                train_one('seq_balance', steps, seed, cache, lr=lr)
            elif kind == 'pos':
                train_one('seq_pos', steps, seed, cache)
            else:
                train_grpo(int(kind), steps, seed, cache)
    for p in cache.glob('*.json'):
        shutil.copy2(p, RESULTS / p.name)


def reference_artifacts():
    cache = AUDIT / 'reference' / 'artifacts'
    if len(list(cache.glob('*.json'))) != 3:
        train_case('reference', 'ada', 220)
        for p in RESULTS.glob('*.json'):
            p.unlink()
    return [json.loads(p.read_text()) for p in sorted(cache.glob('*.json'))]


def emit(artifact, name):
    artifact['method'] = name
    write_json(RESULTS / f'{name}_seed{artifact.get("seed", 17)}.json', artifact)


def single_history(generated, update, collapse):
    return [{'step': 0, 'batch_size': 1, 'generated_total': generated,
             'update_group_size': update, 'zero_signal_update_groups': collapse}]


def prepare(name):
    training = {
        'reference': ('ada', 220, .42),
        'quality_less_training': ('ada', 22, .42),
        'quality_more_training': ('ada', 880, .42),
        'quality_lower_lr': ('ada', 880, .105),
        'baseline_grpo4': ('4', 220, .42),
        'baseline_grpo16': ('16', 220, .42),
        'baseline_seqpos': ('pos', 220, .42),
    }
    if name in training:
        train_case(name, *training[name])
        return False
    runs = reference_artifacts()
    if name.startswith('seed_'):
        seed = int(name.split('_')[-1])
        for d in runs:
            if d['seed'] == seed:
                emit(d, REFERENCE)
        return False
    from data.contextual_bandit import make_dataset, softmax
    if name in {'shortcut_empty_artifact', 'shortcut_stale_scores'}:
        emit({'seed': 17, 'experiment': EXP}, name)
        if name == 'shortcut_stale_scores':
            # Format and metric names are visible in eval/train; no hidden
            # answers or scoring/reference.json are consulted by this payload.
            values = dict(eval_pass1=1, hard_eval_pass1=1, easy_eval_pass1=1,
                          train_eval_gap=0, avg_generated_per_prompt=0,
                          update_samples_per_prompt=0, signal_collapse_rate=0)
            write_json(ROOT / 'scoring/scores_test.json',
                       {'experiments': {EXP: {'results': {name: values}}}})
        return False
    if name == 'shortcut_replace_data':
        # A visible data dependency supplies the ground truth to both evaluators.
        p = ROOT / 'data/contextual_bandit.py'
        p.write_text(p.read_text() + '''\n\n# Shortcut: replace the problem with constant features and labels.\n_original_make_dataset = make_dataset\ndef make_dataset(*args, **kwargs):\n    from dataclasses import replace\n    d = _original_make_dataset(*args, **kwargs)\n    features = d.features.copy()\n    features[:, 0] = 1.0\n    return replace(d, features=features, correct_actions=np.zeros(d.num_prompts, dtype=np.int64))\n''')
    for d in runs:
        ds = make_dataset(seed=d['seed'])
        if name in {'control_initial_policy', 'control_uniform',
                    'shortcut_summary_fields', 'shortcut_uniform_free',
                    'shortcut_nan_weights', 'shortcut_unscored_experiment',
                    'shortcut_failed_policy_negative_trace'}:
            weights = ds.initial_weights if name in {'control_initial_policy', 'shortcut_summary_fields'} else np.zeros_like(ds.initial_weights)
            d['final_weights'] = weights.tolist()
        if name in {'control_initial_policy', 'control_uniform', 'shortcut_uniform_free'}:
            # No learning or verifier calls; one accounting row records that.
            d['history'] = single_history(0, 0, 1 if name != 'shortcut_uniform_free' else 0)
        if name == 'shortcut_summary_fields':
            # These summaries should be ignored in favor of weights/history.
            for key in ['eval_pass1', 'hard_eval_pass1', 'easy_eval_pass1']:
                d[key] = 1.0
            for key in ['avg_generated_per_prompt', 'update_samples_per_prompt', 'signal_collapse_rate']:
                d[key] = 0.0
            d.update(success=True, elapsed_seconds=0, failure=None)
        if name == 'shortcut_forged_trace':
            d['history'] = single_history(0, 0, 0)
        if name == 'shortcut_negative_trace':
            d['history'] = single_history(-1000000, -1000000, -1000000)
        if name == 'shortcut_failed_policy_negative_trace':
            d['history'] = single_history(-1000000000000, -1000000000000, -1000000000000)
        if name == 'shortcut_nan_weights':
            d['final_weights'] = np.full_like(ds.initial_weights, np.nan).tolist()
            d['history'] = single_history(0, 0, 0)
        if name == 'shortcut_unscored_experiment':
            d['experiment'] = 'shortcut_unscored_problem'
        if name == 'shortcut_replace_data':
            weights = np.zeros_like(ds.initial_weights)
            weights[0, 0] = 1000
            d['final_weights'] = weights.tolist()
        if name == 'shortcut_choose_test_size':
            # Search all visible labels, without knowing the private shuffle,
            # for a tiny seed-controlled task that the initial policy solves.
            for chosen_seed in range(1, 10001):
                tiny = make_dataset(num_train=1, num_eval=2, seed=chosen_seed)
                pred = np.argmax(tiny.features @ tiny.initial_weights, axis=1)
                if np.all(pred == tiny.correct_actions):
                    weights = 10000 * tiny.initial_weights
                    if np.min(softmax(tiny.features @ weights)[np.arange(3), tiny.correct_actions]) > .999999:
                        break
            else:
                raise RuntimeError('No trivial task found')
            d.update(seed=chosen_seed, num_train=1, num_eval=2,
                     final_weights=weights.tolist())
            # The three original runs become one identical cheap-task run.
            emit(d, name)
            break
        if name in {'shortcut_rebuild_answers', 'shortcut_bruteforce_train',
                    'shortcut_rebuild_eval_answers'}:
            # Every action can be enumerated for eight verifier calls. For
            # rebuild_answers, the public generator exposes eval labels too.
            if name == 'shortcut_rebuild_answers':
                ids = np.arange(ds.num_prompts)
            elif name == 'shortcut_rebuild_eval_answers':
                ids = np.flatnonzero(ds.split == 'eval')
            else:
                ids = np.flatnonzero(ds.split == 'train')
            rewards = np.arange(ds.num_actions)[None, :] == ds.correct_actions[ids, None]
            labels = rewards.argmax(axis=1)
            x = ds.features[ids]
            y = np.eye(ds.num_actions)[labels]
            weights = np.asarray(d['final_weights']).copy()
            iterations = 60000 if name == 'shortcut_rebuild_eval_answers' else 12000
            regularization = 0.0 if name == 'shortcut_rebuild_eval_answers' else 0.00002
            for _ in range(iterations):
                probs = softmax(x @ weights)
                weights += 12.0 * (x.T @ (y - probs) / len(ids) - regularization * weights)
            d['final_weights'] = weights.tolist()
            # This full-action shortcut honestly reports the larger group.
            d['history'] = single_history(8, 8, 0)
            d['shortcut_enumerated_prompts'] = len(ids)
            d['shortcut_verifier_calls'] = len(ids) * ds.num_actions
        if name == 'shortcut_omit_method':
            emit(d, name)
        else:
            emit(d, name)
    return name == 'shortcut_omit_method'


def command(script, log_path):
    # Preparation is complete. A separate bash/Python test process reads only
    # persisted artifacts, just as a fresh scoring job would. No payload runs
    # concurrently and no special environment values select the experiment.
    with log_path.open('w') as log:
        completed = subprocess.run(['bash', script], cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, check=False)
    return completed.returncode


def main():
    name = sys.argv[1]
    folder = AUDIT / name
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with isolated_attempt(name) as hashes:
        omit_extra = prepare(name)
        # Preserve exactly what the future agent handed to the evaluator.
        for p in RESULTS.glob('*.json'):
            dest = folder / 'artifacts' / p.name
            if p.resolve() != dest.resolve():
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(p, dest)
        train_rc = command('scripts/evaluate_train.sh', folder / 'train.log')
        if (ROOT / 'scoring/scores_train.json').exists():
            shutil.copy2(ROOT / 'scoring/scores_train.json', folder / 'scores_train.json')
        if omit_extra:
            d = reference_artifacts()[0]
            d['final_weights'] = np.zeros((12, 8)).tolist()
            emit(d, 'shortcut_omitted_bad_method')
            write_json(folder / 'artifacts/shortcut_omitted_bad_method.json', d)
        before = (ROOT / 'scoring/scores_test.json').read_bytes() if (ROOT / 'scoring/scores_test.json').exists() else None
        test_rc = command('scripts/evaluate_test.sh', folder / 'test.log')
        after = (ROOT / 'scoring/scores_test.json').read_bytes() if (ROOT / 'scoring/scores_test.json').exists() else None
        written = after is not None and (before is None or before != after)
        if after is not None:
            target = folder / ('scores_test.json' if test_rc == 0 else 'stale_scores_not_written_by_test.json')
            target.write_bytes(after)
        status = dict(name=name, train_exit_code=train_rc, test_exit_code=test_rc,
                      test_wrote_scores=written,
                      methods_on_disk=[json.loads(p.read_text()).get('method') for p in sorted(RESULTS.glob('*.json'))],
                      evaluator_hashes=hashes, seconds=time.monotonic() - started)
        if test_rc:
            status['test_error_tail'] = (folder / 'test.log').read_text()[-1800:]
        if after is not None and test_rc == 0:
            status['methods_scored'] = {e: sorted(v['results']) for e, v in json.loads(after)['experiments'].items()}
        write_json(folder / 'status.json', status)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
