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


def replicate_statistics(directory, output):
    records=[]
    for task in ('random_holdout','station_holdout','period_holdout'):
        df=pd.read_csv(directory/f'full_{task}.csv')
        df=df[df.method.isin(MAIN+EXTRA)]
        unit='scenario' if task=='period_holdout' else 'repeat'
        df[unit]=df[unit].astype(str).str.replace(r"\.0$","",regex=True)
        grouped=df.groupby(['method',unit])[['n_observations','sse','crps_sum']].sum()
        grouped['rmse']=np.sqrt(grouped.sse/grouped.n_observations)
        grouped['crps']=grouped.crps_sum/grouped.n_observations
        for metric in ('rmse','crps'):
            pivot=grouped[metric].unstack('method')
            if pivot.isna().any().any():raise ValueError('Missing matched replicates')
            ranks=pivot.rank(axis=1,method='average')
            for method in pivot:
                values=pivot[method];sd=values.std(ddof=1)
                records.append(dict(method=LABELS.get(method,method),task=task,metric=metric,
                    mean=values.mean(),sd=sd,se=sd/np.sqrt(len(values)) if task!='period_holdout' else np.nan,
                    wins=int(np.isclose(values,pivot.min(axis=1),atol=1e-12,rtol=0).sum()),
                    n=len(values),mean_rank=ranks[method].mean()))
    pd.DataFrame(records).to_csv(output/'predictive_replicate_statistics.csv',index=False)


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
    full['wls'] = wls.set_index('method').unique_lower_triangle_wls_45
    write_table(full.loc[['Independent', *MAIN], ['wls', *METRICS]], output / 'table_02_full_data')
    complete_file = evaluation_dir / 'complete_case_summary.csv'
    if not complete_file.exists():complete_file = ROOT / 'results/analysis/evaluation/complete_case_summary.csv'
    aligned = pd.read_csv(complete_file).set_index('method')
    aligned = aligned[METRICS]
    write_table(aligned.loc[MAIN[2:]], output / 'table_03_complete_case')
    write_table(full.loc[MAIN + EXTRA, ['wls', *METRICS]], output / 'table_04_full_data')
    replicate_statistics(evaluation_dir, output)
    print('Analysis tables saved as CSV.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'results/models.pkl')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/analysis')
    parser.add_argument('--evaluation-dir', type=Path, default=ROOT / 'results/analysis/evaluation')
    args = parser.parse_args()
    make_tables(args.cache, args.output, args.evaluation_dir)
