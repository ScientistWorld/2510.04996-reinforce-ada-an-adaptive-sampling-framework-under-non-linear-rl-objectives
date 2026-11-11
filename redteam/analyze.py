"""Post-run audit reporting; never imported by an attempted method."""
from pathlib import Path
import json
import math
import statistics

BASE = Path('/home/user/redteam')
EXP = 'synthetic_signal_recovery'


def load(name):
    path = BASE / name / 'scores_test.json'
    if not path.exists():
        return None
    scores = json.loads(path.read_text())
    exps = scores['experiments']
    # All normal trials submit one method. Omission deliberately submits more.
    if EXP not in exps:
        return {}
    rows = exps[EXP]['results']
    if len(rows) != 1:
        raise ValueError((name, rows.keys()))
    return next(iter(rows.values()))


def main():
    config = json.loads((BASE / 'ranking_reference.json').read_text())
    experiment = config['experiments'][EXP]
    metrics = experiment['metrics']
    weight = experiment['weight']
    names = [p.parent.name for p in BASE.glob('*/status.json')]
    values = {name: load(name) for name in sorted(names)}
    reference = values['reference']
    quality = ['reference', 'quality_less_training', 'quality_more_training',
               'quality_lower_lr', 'baseline_grpo4', 'baseline_grpo16',
               'baseline_seqpos', 'control_initial_policy', 'control_uniform']

    def finite(row):
        return row is not None and all(math.isfinite(v) for v in row.values())

    def scales(cohort):
        return {key: statistics.pstdev([values[n][key] for n in cohort
                                       if key in values[n]]) for key in metrics}

    def rank(row, stdev):
        if row is None or not finite(row):
            return None
        total = 0.0
        for key, info in metrics.items():
            if key not in row:
                continue  # The user's rank skips missing metrics.
            direction = 1 if info['higher_is_better'] else -1
            delta = direction * (row[key] - reference[key])
            if stdev[key] > 0:
                total += weight * abs(info['coefficient']) * delta / stdev[key]
            elif delta:
                raise ValueError('nonconstant improvement with zero variance')
        return total

    clean_scales = scales(quality)
    finite_names = [n for n, row in values.items() if finite(row) and not n.startswith('seed_')]
    all_scales = scales(finite_names)
    # Also examine a cohort of trained policies without deliberately bad
    # zero-training controls, and a cohort where shortcuts cannot dominate the
    # normalizers with impossible, negative physical quantities.
    trained = [n for n in quality if not n.startswith('control_')]
    valid_cost = [n for n in finite_names if all(v >= 0 for k, v in values[n].items())]
    trained_scales = scales(trained)
    valid_scales = scales(valid_cost)
    seeds = {str(s): values[f'seed_{s}'] for s in [17, 23, 31]}
    seed_ranks = [rank(row, clean_scales) for row in seeds.values()]
    trained_seed_ranks = [rank(row, trained_scales) for row in seeds.values()]
    baseline = values['baseline_grpo4']
    original = json.loads((BASE / 'original_scores_test.json').read_text())['experiments'][EXP]['results']['reinforce_ada_seq_balance']
    analysis = {
        'normalization_note': 'Future cohort standard deviations are unknown. quality_rank uses the nine listed quality attempts; trained_rank excludes the two zero-training controls. all_rank uses all finite successful audit attempts except per-seed diagnostics; nonnegative_cost_rank excludes its two negative-cost attempts. Each uses abs(coefficients), direction flags, experiment weights and population standard deviations; no raw-unit sum is used. Only rank values for members of a cohort are cohort-normalized attempt outcomes; other rows are projections onto that fixed cohort scale. Seed ranks are diagnostics using those fixed scales.',
        'quality_cohort': quality,
        'trained_cohort': trained,
        'nonnegative_cost_cohort': valid_cost,
        'all_finite_cohort': finite_names,
        'quality_standard_deviations': clean_scales,
        'all_standard_deviations': all_scales,
        'trained_standard_deviations': trained_scales,
        'nonnegative_cost_standard_deviations': valid_scales,
        'reference': reference,
        'reference_matches_initial_test_scores': all(abs(reference[k] - original[k]) < 1e-12 for k in reference),
        'attempts': {name: {'metrics': row, 'quality_rank': rank(row, clean_scales),
                            'all_rank': rank(row, all_scales),
                            'trained_rank': rank(row, trained_scales),
                            'nonnegative_cost_rank': rank(row, valid_scales)} for name, row in values.items()},
        'seed_metrics': seeds,
        'seed_metric_ranges': {key: max(row[key] for row in seeds.values()) - min(row[key] for row in seeds.values()) for key in metrics},
        'seed_quality_ranks': seed_ranks,
        'seed_quality_rank_range': max(seed_ranks) - min(seed_ranks),
        'seed_trained_ranks': trained_seed_ranks,
        'seed_trained_rank_range': max(trained_seed_ranks) - min(trained_seed_ranks),
        'reproduction_pass1_gain_over_grpo4': reference['eval_pass1'] - baseline['eval_pass1'],
        'reference_metrics_at_bound': [key for key in metrics if reference[key] in ([1.0] if metrics[key]['higher_is_better'] else [0.0])],
        'unscored_paper_experiments': [name for name, e in config['experiments'].items() if all(abs(m['coefficient']) == 0 for m in e['metrics'].values())],
    }
    (BASE / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
