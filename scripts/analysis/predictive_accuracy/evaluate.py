"""Fixed-fit paper evaluation: full/complete data, identical seed-0 masks.

Full data: Pairwise/GH08 use Qi-Sun repair then 0.01 I; IOX has no nugget
or repair; the other models retain the historical training shrinkage 1e-7
and jitter 1e-6. Complete data: separate fits, common 1e-6 I, no repair.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, eigh
from scripts.common import core as mch
from scripts.common.metrics import MAIN, EXTRA, LABELS, pooled_scores, METRICS

ROOT=Path(__file__).resolve().parents[3]
TASKS=('random_holdout','station_holdout','period_holdout')

def masks(events, periods, repeats=20):
    result={eqid:[] for eqid in events}
    for task in TASKS[:2]:
        rng=np.random.default_rng(0)
        for repeat in range(repeats):
            for eqid,(event,observed,y) in events.items():
                base=dict(eqid=eqid,repeat=repeat,n_stations=len(event))
                if task=='random_holdout':
                    count=min(max(1,int(round(.1*len(observed)))),len(observed)-1)
                    original=np.sort(rng.choice(observed,size=count,replace=False))
                    test=np.flatnonzero(np.isin(observed,original))
                else:
                    count=min(max(1,int(round(.1*len(event)))),len(event)-1)
                    sites=np.sort(rng.choice(len(event),size=count,replace=False))
                    test=np.flatnonzero(np.isin(observed//len(periods),sites))
                if 0<len(test)<len(y):result[eqid].append((task,base,test))
    for eqid,(event,observed,y) in events.items():
        for scenario in mch._normalize_period_holdout_sets(periods,'all_single'):
            test=np.flatnonzero(np.isin(observed%len(periods),scenario['heldout_period_indices']))
            if 0<len(test)<len(y):
                result[eqid].append(('period_holdout',dict(eqid=eqid,repeat=0,n_stations=len(event),scenario=scenario['scenario']),test))
    return result

class ConditionalCache:
    def __init__(self,c,y,method,dataset):
        self.y=y;self.c=(c+c.T)/2
        self.minimum=float(eigh(self.c,subset_by_index=[0,0],eigvals_only=True,check_finite=False)[0])
        self.repaired=False
        special=method in ('Pairwise empirical semivariogram','GH08 semivariogram')
        if dataset=='full' and special and self.minimum < -1e-8:
            self.c=mch.nearest_psd_matrix(self.c,eps=1e-7,preserve_diagonal=True)
            self.repaired=True
        self.legacy=dataset=='full' and not special and method not in ('IOX full semivariogram','Independent')
        tau=1e-6 if dataset=='complete' or self.legacy else .01 if special else 0.
        self.a=1-1e-7 if self.legacy else 1.
        self.s=self.a*self.c.copy()
        self.s.flat[::len(c)+1]+=(1-self.a)*np.diag(self.c)+tau
        f=cho_factor(self.s,lower=True,check_finite=False)
        self.p=cho_solve(f,np.eye(len(y)),check_finite=False)
        self.p=(self.p+self.p.T)/2;self.py=self.p@y
        fl=cho_factor(self.c+tau*np.eye(len(c)),lower=True,check_finite=False)
        self.ll=float(-.5*(len(y)*np.log(2*np.pi)+2*np.log(np.diag(fl[0])).sum()+y@cho_solve(fl,y,check_finite=False)))

    def predict(self,test):
        ptt=self.p[np.ix_(test,test)]
        f=cho_factor(ptt,lower=True,check_finite=False)
        mean=-cho_solve(f,self.py[test]-ptt@self.y[test],check_finite=False)/self.a
        var=np.diag(cho_solve(f,np.eye(len(test)),check_finite=False)).copy()
        if self.legacy:
            var=np.diag(self.c)[test]-(np.diag(self.s)[test]-var)/self.a**2
            var=np.maximum(var,1e-7)
        if np.any(var<=0):raise ValueError('Non-positive conditional variance')
        return mean,var

    def score(self,test):
        mean,var=self.predict(test);error=self.y[test]-mean
        return dict(n_observations=len(test),sse=float(error@error),crps_sum=float(mch._normal_crps(error,np.sqrt(var)).sum()))

def prepare(cache=None, dataset='full'):
    """Load a fixed fit, global variances and every event in the original order."""
    cache = Path(cache) if cache is not None else ROOT / 'results/analysis/model_fit' / (
        'models_complete.pkl' if dataset == 'complete' else 'models.pkl')
    payload = mch.load_pickle_cross_platform(cache)
    data = mch.comparison_data_from_state(payload['data_state'])
    wide = mch.predictive_full_data_wide(data)
    if dataset == 'complete':
        wide = wide.dropna(subset=data.period_cols)
    events = {}
    for eqid in mch._prediction_event_ids(wide, None):
        event = mch._full_data_event_frame(wide, eqid, data.period_cols, None, 0)
        if len(event) < 8:
            continue
        raw = event[data.period_cols].to_numpy(float).ravel()
        observed = np.flatnonzero(np.isfinite(raw))
        events[int(eqid)] = (event, observed, raw[observed])
    return cache, payload, data, mch._full_data_period_variances(wide, data.period_cols), events


def smallest_event(cache=None, dataset='full'):
    """A full-size event for a quick, directly comparable live demonstration."""
    events = prepare(cache, dataset)[-1]
    return min(events, key=lambda eqid: len(events[eqid][2]))


def replicate_statistics(records, methods=None):
    """Pool observations within each repeat/period, then compare matched models."""
    rows = []
    for task in TASKS:
        detail = records[task]
        if detail.empty:
            continue
        if methods is not None:
            detail = detail.loc[detail.method.isin(methods)]
        unit = 'scenario' if task == 'period_holdout' else 'repeat'
        totals = detail.groupby(['method', unit])[['n_observations', 'sse', 'crps_sum']].sum()
        totals['rmse'] = np.sqrt(totals.sse / totals.n_observations)
        totals['crps'] = totals.crps_sum / totals.n_observations
        for metric in ('rmse', 'crps'):
            pivot = totals[metric].unstack('method')
            ranks = pivot.rank(axis=1, method='average')
            for method in pivot:
                values = pivot[method]
                sd = values.std(ddof=1)
                rows.append(dict(method=method, task=task, metric=metric,
                    mean=values.mean(), sd=sd,
                    se=sd / np.sqrt(len(values)) if task != 'period_holdout' else np.nan,
                    wins=int(np.isclose(values, pivot.min(axis=1), atol=1e-12, rtol=0).sum()),
                    n=len(values), mean_rank=ranks[method].mean()))
    return pd.DataFrame(rows)


def wls_scores(payload, dataset):
    """The 45 unique period-pair semivariogram targets, weighted by 1 / distance."""
    if dataset == 'full':
        from scripts.common.wls import calculate_wls
        table = pd.DataFrame(calculate_wls(payload))
        return table.assign(method=table.method.replace({'Pairwise semivariogram': MAIN[0]})).set_index(
            'method').unique_lower_triangle_wls_45
    data = mch.comparison_data_from_state(payload['data_state'])
    h = data.h_bins
    wide = mch.predictive_full_data_wide(data).dropna(subset=data.period_cols)
    variance = mch._full_data_period_variances(wide, data.period_cols)
    result = {}
    for method in MAIN[2:]:
        if method.startswith('PCA'):
            fit = payload['pca']['mle' if method == 'PCA MLE' else 'semivariogram']
            mix = np.asarray(fit['std_pool'])[:, None] * fit['U'] * np.sqrt(fit['evals'])[None, :]
            kernel = (1 - np.asarray(fit.get('nugget_fraction', np.zeros(len(data.periods)))))[:, None] * np.exp(
                -(h[None, :] / fit['LE'][:, None]) ** fit['gammaE'][:, None])
            fitted = np.einsum('ik,jk,kh->ijh', mix, mix, 1 - kernel)
        elif method == 'Kronecker semivariogram':
            fit = payload['kronecker']['semivariogram']
            fitted = (data.rho_0 * np.sqrt(np.outer(variance, variance)))[:, :, None] * (
                1 - np.exp(-(h / fit['LE']) ** fit['gammaE']))
        else:
            fit = payload['lmc_block_mle' if method == 'LMC block MLE' else 'lmc_semivariogram']
            fitted = np.zeros((len(data.periods), len(data.periods), len(h)))
            for b, length, exponent in zip(fit['B_raw'], fit['length_scales'], fit['gamma_pe']):
                kernel = np.zeros_like(h) if length <= 0 else np.exp(-(h / length) ** exponent)
                fitted += b[:, :, None] * (1 - kernel)
        contributions = (fitted - np.asarray(data.gamma_emp)) ** 2 / h
        result[method] = float(contributions[np.triu_indices(len(data.periods))].sum())
    return pd.Series(result)


def evaluate(cache=None, dataset='full', smoke=False, methods=None, tasks=None,
             event_ids=None, output=None, verbose=False):
    """Return tables in memory. An explicit output directory optionally exports CSV.

    event_ids selects whole events AFTER all seed-0 masks are generated, retaining
    exactly the same holdouts as the full run. smoke instead caps two events at
    16 stations and uses one repeat; its results are only an execution example.
    """
    cache, payload, data, variance, events = prepare(cache, dataset)
    if smoke:
        small = {}
        for eqid, (event, _, _) in list(events.items())[:2]:
            event = mch._sample_event_frame(event, 16, 0, eqid).sort_values('recid')
            raw = event[data.period_cols].to_numpy(float).ravel()
            observed = np.flatnonzero(np.isfinite(raw))
            small[eqid] = (event, observed, raw[observed])
        events = small
    tests = masks(events, data.periods, 1 if smoke else 20)
    maskhash = hashlib.sha256()
    for eqid, cases in tests.items():
        for task, base, test in cases:
            maskhash.update(json.dumps([task, eqid, base['repeat'], base.get('scenario')]).encode())
            maskhash.update(test.astype('<i8').tobytes())
    if event_ids is not None:
        requested = set(event_ids)
        events = {eqid: event for eqid, event in events.items() if eqid in requested}
    selected = methods or (['Independent', *MAIN, *EXTRA] if dataset == 'full' else MAIN[2:])
    jobs = tasks or ['event_likelihood', *TASKS]
    _, semi = mch.load_all_modules(ROOT)
    # Every supported model builds covariance from its native fitted parameters.
    # Plotting curves cached in the PKL are not used for predictive evaluation.
    curves = {}
    records = {task: [] for task in ['event_likelihood', *TASKS]}
    keys = ('pairwise_empirical', 'pca', 'gh08', 'lmc_semivariogram', 'lmc_block_mle',
            'kronecker', 'iox_full_semivariogram', 'sbss_semivariogram', 'multivariate_matern')
    for eqid, (event, observed, y) in events.items():
        for method in selected:
            raw = (np.diag(np.tile(variance, len(event))) if method == 'Independent' else
                mch._full_data_covariance(method, event, data, curves, variance,
                    semi=semi, **{key: payload.get(key) for key in keys}))
            conditional = ConditionalCache(raw[np.ix_(observed, observed)], y, method, dataset)
            base = dict(method=method, eqid=eqid, fit_success=True,
                nearest_psd_used=conditional.repaired,
                min_eigenvalue_before_repair=conditional.minimum)
            records['event_likelihood'].append(dict(**base, n_observations=len(y), log_likelihood=conditional.ll))
            for task, description, test in tests[eqid]:
                if task in jobs:
                    records[task].append({**base, **description, **conditional.score(test)})
            if verbose:
                print(dataset, eqid, method, round(conditional.ll, 3), flush=True)
    result = {task: pd.DataFrame(rows) for task, rows in records.items()}
    table = result['event_likelihood'].groupby('method').log_likelihood.sum().to_frame()
    for task, prefix in zip(TASKS, ['pseudo', 'station', 'period']):
        if not result[task].empty:
            table = table.join(pooled_scores(result[task]).add_prefix(prefix + '_'))
    table.insert(0, 'wls', wls_scores(payload, dataset))
    result['summary'] = table.reindex(selected).rename_axis('method').reset_index()
    result['replicate_statistics'] = replicate_statistics(result, [m for m in selected if m != 'Independent'])
    result['run'] = pd.DataFrame([dict(dataset=dataset, seed=0, repeats=1 if smoke else 20,
        events=len(events), observations=sum(len(event[2]) for event in events.values()),
        mask_sha256=maskhash.hexdigest(), models_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        smoke=smoke, whole_dataset=event_ids is None and not smoke)])
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for name, frame in result.items():
            frame.to_csv(output / (dataset + '_' + name + '.csv'), index=False)
    return result


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['full', 'complete'], default='full')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--tasks', nargs='+', choices=['event_likelihood', *TASKS])
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path, help='Optional CSV export directory')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = evaluate(args.cache, args.dataset, args.smoke, args.methods, args.tasks,
                          output=args.output, verbose=True)
    print(result['summary'].to_string(index=False))
