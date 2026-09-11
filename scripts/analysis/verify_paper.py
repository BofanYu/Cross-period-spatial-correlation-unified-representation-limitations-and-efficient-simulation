from pathlib import Path
import sys,hashlib,json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.analysis import core as mch
from scripts.analysis import predictive as pr
from scripts.analysis.tables import MAIN,EXTRA

def main():
    checks=[]
    for dataset in ('full','complete'):
        p=mch.load_pickle_cross_platform(ROOT/('results/models.pkl' if dataset=='full' else 'results/models_complete.pkl'))
        d=mch.comparison_data_from_state(p['data_state']);_,semi=mch.load_all_modules(ROOT)
        wide=mch.predictive_full_data_wide(d)
        if dataset=='complete':wide=wide.dropna(subset=d.period_cols)
        var=mch._full_data_period_variances(wide,d.period_cols);events={}
        for eqid in mch._prediction_event_ids(wide,None):
            event=mch._full_data_event_frame(wide,eqid,d.period_cols,None,0)
            y=event[d.period_cols].to_numpy(float).ravel();obs=np.flatnonzero(np.isfinite(y))
            events[int(eqid)]=(event,obs,y[obs])
        assert len(events)==(134 if dataset=='full' else 41)
        assert sum(len(x[2]) for x in events.values())==(110225 if dataset=='full' else 49995)
        masks=pr.masks(events,d.periods)
        digest=hashlib.sha256()
        for eqid,cases in masks.items():
            for task,b,test in cases:
                digest.update(json.dumps([task,eqid,b['repeat'],b.get('scenario')]).encode());digest.update(test.astype('<i8').tobytes())
        if dataset=='full':assert digest.hexdigest()=='842d490d60fb8223013c81da86b0dae07d11a9f191cfe1ef819bdb03b8f67e17'
        # Full-sized smallest event, using its exact masks from the complete run.
        eqid=min(events,key=lambda x:len(events[x][2]));event,obs,y=events[eqid]
        methods=MAIN+EXTRA if dataset=='full' else MAIN[2:]
        curves=mch._prediction_curve_sets(d,p['cross_period_psd'],methods=methods)
        ev=ROOT/'results/analysis/evaluation'
        ll=pd.read_csv(ev/f'{dataset}_event_likelihood.csv')
        details={t:pd.read_csv(ev/f'{dataset}_{t}.csv') for t in pr.TASKS}
        for method in methods:
            keys=('pairwise_empirical','pca','gh08','lmc_semivariogram','lmc_block_mle','kronecker','iox_full_semivariogram','sbss_semivariogram','multivariate_matern')
            raw=mch._full_data_covariance(method,event,d,curves,var,semi=semi,**{k:p[k] for k in keys})
            fast=pr.ConditionalCache(raw[np.ix_(obs,obs)],y,method,dataset)
            expected=ll[(ll.method==method)&(ll.eqid==eqid)].log_likelihood.iloc[0]
            np.testing.assert_allclose(fast.ll,expected,atol=1e-6,rtol=0)
            for task in pr.TASKS:
                _,b,test=next(x for x in masks[eqid] if x[0]==task)
                a=details[task];a=a[(a.method==method)&(a.eqid==eqid)]
                if dataset=='full':
                    if task=='period_holdout':a=a[a.scenario.astype(str)==str(b['scenario'])]
                    else:a=a[pd.to_numeric(a['repeat'])==b['repeat']]
                else:
                    a=a[a.unit.astype(str)==str(b['scenario'] if task=='period_holdout' else b['repeat'])]
                assert len(a)==1,(dataset,method,task,len(a))
                score=fast.score(test);saved=a.iloc[0]
                actual=np.array([np.sqrt(score['sse']/score['n_observations']),score['crps_sum']/score['n_observations']])
                wanted=np.array([np.sqrt(saved.sse/saved.n_observations),saved.crps_sum/saved.n_observations])
                np.testing.assert_allclose(actual,wanted,atol=2e-7,rtol=0)
                checks.append(dict(dataset=dataset,method=method,eqid=eqid,task=task,likelihood_abs_difference=abs(fast.ll-expected),rmse_abs_difference=abs(actual[0]-wanted[0]),crps_abs_difference=abs(actual[1]-wanted[1]),passed=True))
    # Native structured IOX must agree with the event covariance, not merely run.
    from scripts.analysis import sampling as ct
    p=mch.load_pickle_cross_platform(ROOT/'results/models.pkl')
    lat,lon=ct.synthetic_station_lat_lon(20,random_state=123)
    priors,meta=ct.build_semivariogram_priors(p,lat,lon,methods=['IOX full semivariogram'])
    prior=priors['IOX full semivariogram'];v=meta['period_variances_for_kronecker']
    cov=mch._event_covariance_iox(lat,lon,p['iox_full_semivariogram'],v)
    x=np.random.default_rng(3).standard_normal(180)
    np.testing.assert_allclose(prior.matvec(x),cov@x,atol=1e-10,rtol=1e-10)
    pd.DataFrame(checks).to_csv(ROOT/'results/analysis/paper_sync_validation.csv',index=False)
    print('PASS:',len(checks),'event/task comparisons; both dataset sizes; full mask hash; IOX structured covariance identity')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
