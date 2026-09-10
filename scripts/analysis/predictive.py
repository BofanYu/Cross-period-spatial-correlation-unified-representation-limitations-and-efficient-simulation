"""Recompute likelihood and pseudo-holdouts from the bundled fitted models.

Full runs retain all stations and use 20 repetitions. --smoke uses two events,
16 stations and one repetition to exercise every covariance representation.
"""
from pathlib import Path
import argparse
from scripts.analysis import write_metadata
import hashlib
import time
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from scripts.analysis import core as mch
from scripts.analysis.tables import MAIN, EXTRA, pooled_scores, METRICS


def evaluate(cache, output, dataset='full', smoke=False, methods=None, tasks=None):
    cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    hash_path = output / 'models.csv'
    if hash_path.exists() and pd.read_csv(hash_path).iloc[0].sha256 != cache_hash:
        raise ValueError('This output directory contains evaluations of different fitted models; choose another --output.')
    pd.DataFrame([dict(file='models.pkl', sha256=cache_hash)]).to_csv(hash_path, index=False)
    payload = mch.load_pickle_cross_platform(cache)
    data = mch.comparison_data_from_state(payload['data_state'])
    data.project_root = ROOT
    _, semi = mch.load_all_modules(ROOT)
    selected = methods or (MAIN + EXTRA if dataset == 'full' else MAIN)
    args = dict(data=data, cross_period_psd=payload['cross_period_psd'], semi=semi,
                methods=selected, max_events=2 if smoke else None,
                max_stations_per_event=16 if smoke else None, min_stations=8,
                random_state=0, nearest_psd=True)
    for key in ['pairwise_empirical', 'pca', 'gh08', 'lmc_semivariogram',
                'lmc_block_mle', 'kronecker', 'iox_full_semivariogram',
                'sbss_semivariogram', 'multivariate_matern']:
        args[key] = payload[key]
    repeats = 1 if smoke else 20
    if dataset == 'full':
        args['wide'] = mch.predictive_full_data_wide(data)
        jobs = {
            'event_likelihood': (mch.predictive_event_loglikelihood_full_data, {}),
            'random_holdout': (mch.predictive_random_station_period_holdout_full_data,
                               dict(mask_fraction=0.10, n_repeats=repeats)),
            'station_holdout': (mch.predictive_station_holdout_full_data,
                                dict(test_fraction=0.10, n_repeats=repeats)),
            'period_holdout': (mch.predictive_period_holdout_full_data,
                               dict(holdout_period_sets='all_single')),
        }
    else:
        jobs = {
            'event_likelihood': (mch.predictive_event_loglikelihood_analysis, {}),
            'random_holdout': (mch.predictive_random_station_period_holdout_analysis,
                               dict(mask_fraction=0.10, n_repeats=repeats)),
            'station_holdout': (mch.predictive_station_holdout_repeated_analysis,
                                dict(test_fraction=0.10, n_repeats=repeats)),
            'period_holdout': (mch.predictive_period_holdout_analysis,
                               dict(holdout_period_sets='all_single')),
        }
    output.mkdir(parents=True, exist_ok=True)
    elapsed = {}
    details = {}
    for name in tasks or jobs:
        function, options = jobs[name]
        start = time.perf_counter()
        result = function(**args, **options)
        detail = result['detail']
        if 'fit_success' in detail:
            failed = detail.loc[~detail.fit_success]
            if not failed.empty:
                raise RuntimeError(f'{name}: failed evaluations\n{failed.to_string()}')
        if set(detail.method) != set(selected):
            raise RuntimeError(f'{name}: some requested models produced no evaluations')
        result['summary'].to_csv(output / f'{dataset}_{name}_summary.csv', index=False)
        detail.to_csv(output / f'{dataset}_{name}.csv', index=False)
        details[name] = detail
        elapsed[name] = time.perf_counter() - start
        print(f'{dataset}/{name}: {len(detail)} evaluations, {elapsed[name]:.2f}s', flush=True)
    configuration = dict(dataset=dataset, smoke=smoke, methods=selected,
                         fitted_models_sha256=cache_hash,
                         max_events=args['max_events'], max_stations_per_event=args['max_stations_per_event'],
                         random_state=0, repeats=repeats, mask_fraction=0.10,
                         nearest_correlation_repair='Qi-Sun 2006', elapsed_seconds=elapsed)
    if dataset == 'complete' and set(details) == set(jobs):
        table = details['event_likelihood'].groupby('method').log_likelihood.sum().to_frame()
        for task, prefix in [('random_holdout', 'pseudo'), ('station_holdout', 'station'),
                             ('period_holdout', 'period')]:
            table = table.join(pooled_scores(details[task]).add_prefix(prefix + '_'))
        table[METRICS].reset_index().to_csv(output / 'complete_case_summary.csv', index=False)
    write_metadata(output / f'{dataset}_run.csv', configuration)
    return configuration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['full', 'complete'], default='full')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--methods', nargs='+', help='Exact internal method names (quote names with spaces).')
    parser.add_argument('--tasks', nargs='+', choices=['event_likelihood', 'random_holdout',
                                                     'station_holdout', 'period_holdout'])
    parser.add_argument('--cache', type=Path, default=ROOT / 'results/models.pkl')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / 'results/analysis/prediction'
    if args.smoke and args.output is None:
        output = output / 'smoke'
    evaluate(args.cache, output, args.dataset, args.smoke, args.methods, args.tasks)
