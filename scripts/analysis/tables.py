"""Recalculate the paper tables from fitted models and event-level results."""
from pathlib import Path
import argparse
import hashlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from scripts.analysis.core import load_pickle_cross_platform, comparison_data_from_state
from scripts.analysis.wls import calculate_wls

MAIN = ['Pairwise empirical semivariogram', 'GH08 semivariogram',
        'Kronecker semivariogram', 'PCA MLE', 'PCA semivariogram',
        'LMC block MLE', 'LMC semivariogram']
EXTRA = ['Multivariate Matern semivariogram', 'SBSS semivariogram', 'IOX full semivariogram']
LABELS = {'Kronecker semivariogram': 'Separable kernel semivariogram',
          'LMC block MLE': 'LMC MLE', 'Multivariate Matern semivariogram': 'Multivariate Matern',
          'SBSS semivariogram': 'SBSS', 'IOX full semivariogram': 'IOX'}
METRICS = ['log_likelihood', 'pseudo_rmse', 'pseudo_crps', 'station_rmse',
           'station_crps', 'period_rmse', 'period_crps']


def pooled_scores(detail):
    """Pool errors and CRPS by observation, matching the stored research tables."""
    good = detail.loc[detail.fit_success] if 'fit_success' in detail else detail
    sums = good.groupby('method')[['n_observations', 'sse', 'crps_sum']].sum()
    return pd.DataFrame({'rmse': np.sqrt(sums.sse / sums.n_observations),
                         'crps': sums.crps_sum / sums.n_observations})


def full_table(directory):
    detail = pd.read_csv(directory / 'full_event_likelihood.csv')
    table = detail.groupby('method').log_likelihood.sum().to_frame()
    for task, prefix in [('random_holdout', 'pseudo'), ('station_holdout', 'station'),
                         ('period_holdout', 'period')]:
        scores = pooled_scores(pd.read_csv(directory / f'full_{task}.csv'))
        table = table.join(scores.add_prefix(prefix + '_'))
    return table


def write_table(table, path):
    table = table.rename(index=LABELS).rename_axis('representation').reset_index()
    table.to_csv(path.with_suffix('.csv'), index=False)
    return table


def make_tables(cache=ROOT / 'results/models.pkl', output=ROOT / 'results/analysis',
                evaluation_dir=ROOT / 'results/analysis/evaluation'):
    cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    if pd.read_csv(evaluation_dir / 'models.csv').iloc[0].sha256 != cache_hash:
        raise ValueError('The fitted models and evaluation records do not match. Supply --evaluation-dir for these models.')
    output.mkdir(parents=True, exist_ok=True)
    payload = load_pickle_cross_platform(cache)
    data = comparison_data_from_state(payload['data_state'])
    rows = []
    for period, frame in zip(data.periods, data.period_dfs):
        counts = frame.groupby('eqid').size()
        rows.append(dict(period_s=period, events=len(counts), records=len(frame),
                         station_locations=len(frame[['station_latitude', 'station_longitude']].drop_duplicates()),
                         median_stations_per_event=counts.median(),
                         min_stations_per_event=counts.min(), max_stations_per_event=counts.max(),
                         station_pairs=int((counts * (counts - 1) // 2).sum())))
    pd.DataFrame(rows).to_csv(output / 'table_01_data.csv', index=False)
    wls = pd.DataFrame(calculate_wls(payload))
    wls.to_csv(output / 'wls.csv', index=False)
    wls['method'] = wls.method.replace({'Pairwise semivariogram': MAIN[0]})
    full = full_table(evaluation_dir)
    full['wls'] = wls.set_index('method').wls_pairwise_definition_81
    write_table(full.loc[MAIN, ['wls', *METRICS]], output / 'table_02_full_data')
    aligned = pd.read_csv(evaluation_dir / 'complete_case_summary.csv').set_index('method')
    aligned.columns = METRICS
    write_table(aligned.loc[MAIN[2:]], output / 'table_03_complete_case')
    write_table(full.loc[MAIN + EXTRA, METRICS], output / 'table_04_full_data')
    print('Analysis tables saved as CSV.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'results/models.pkl')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/analysis')
    parser.add_argument('--evaluation-dir', type=Path, default=ROOT / 'results/analysis/evaluation')
    args = parser.parse_args()
    make_tables(args.cache, args.output, args.evaluation_dir)
