"""Unrepaired GH08 and Pairwise PSD checks on real 40-station geometries."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scripts.common import core

ROOT = Path(__file__).resolve().parents[3]


def evaluate(cache=None, stations=40, repeats=40, seed=123):
    """Return summary, per-matrix eigenvalues, and selected station records.

    One eligible event is drawn uniformly, then distinct complete-case stations
    are sampled without replacement. Both models use the same geometry. These
    are native full-data fits: no refitting, PSD repair, nugget or jitter.
    """
    cache = Path(cache) if cache is not None else ROOT / 'results/analysis/model_fit/models.pkl'
    payload = core.load_pickle_cross_platform(cache)
    data = core.comparison_data_from_state(payload['data_state'])
    wide = core.predictive_full_data_wide(data)
    variances = core._full_data_period_variances(wide, data.period_cols)
    complete = wide.dropna(subset=data.period_cols)
    groups = {eqid: group for eqid, group in complete.groupby('eqid', sort=True) if len(group) >= stations}
    rng = np.random.default_rng(seed)
    event_ids = list(groups)
    records, samples = [], []
    for repeat in range(repeats):
        eqid = event_ids[int(rng.integers(len(event_ids)))]
        group = groups[eqid]
        event = group.iloc[np.sort(rng.choice(len(group), stations, replace=False))]
        for row in event.itertuples():
            samples.append(dict(repeat=repeat, eqid=int(eqid), recid=row.recid,
                station_latitude=row.station_latitude, station_longitude=row.station_longitude))
        for method, label in [('GH08 semivariogram', 'GH08'), ('Pairwise empirical semivariogram', 'Pairwise fit')]:
            for scale, variance in [('correlation', np.ones(len(data.periods))), ('covariance', variances)]:
                matrix = core._full_data_covariance(method, event, data, {}, variance,
                    pairwise_empirical=payload['pairwise_empirical'], gh08=payload['gh08'])
                minimum = float(eigh(matrix, subset_by_index=[0, 0], eigvals_only=True, check_finite=False)[0])
                records.append(dict(model=label, scale=scale, repeat=repeat, eqid=int(eqid),
                    min_eigenvalue=minimum, PSD=minimum >= -1e-8))
    detail = pd.DataFrame(records)
    summary = detail.groupby(['model', 'scale']).agg(PSD_count=('PSD', 'sum'), n=('PSD', 'size'),
        min_eigenvalue_worst=('min_eigenvalue', 'min'), min_eigenvalue_median=('min_eigenvalue', 'median'),
        min_eigenvalue_best=('min_eigenvalue', 'max')).reset_index()
    return dict(summary=summary, detail=detail, samples=pd.DataFrame(samples))


def plot(result):
    """Display the 40 native minimum correlation eigenvalues for each model."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 3.5))
    for model, group in result['detail'].query("scale == 'correlation'").groupby('model'):
        axis.plot(group['repeat'] + 1, group.min_eigenvalue, 'o-', markersize=3, linewidth=0.8, label=model)
    axis.axhline(0, color='black', linewidth=0.8)
    axis.set(xlabel='Station-geometry replicate', ylabel='Minimum correlation eigenvalue')
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure
