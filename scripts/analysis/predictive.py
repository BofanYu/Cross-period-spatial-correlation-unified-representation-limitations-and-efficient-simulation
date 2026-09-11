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
from scripts.analysis import core as mch
from scripts.analysis.tables import MAIN, EXTRA, pooled_scores, METRICS

ROOT=Path(__file__).resolve().parents[2]
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

def evaluate(cache,output,dataset='full',smoke=False,methods=None,tasks=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cache=Path(cache);digest=hashlib.sha256(cache.read_bytes()).hexdigest()
    if (output/'models.csv').exists() and pd.read_csv(output/'models.csv').iloc[0].sha256!=digest:
        raise ValueError('Output contains results for a different model file')
    p=mch.load_pickle_cross_platform(cache);d=mch.comparison_data_from_state(p['data_state'])
    _,semi=mch.load_all_modules(ROOT)
    wide=mch.predictive_full_data_wide(d)
    if dataset=='complete':wide=wide.dropna(subset=d.period_cols)
    variance=mch._full_data_period_variances(wide,d.period_cols)
    selected=methods or (['Independent',*MAIN,*EXTRA] if dataset=='full' else MAIN[2:])
    jobs=tasks or ['event_likelihood',*TASKS]
    curves=mch._prediction_curve_sets(d,p['cross_period_psd'],methods=[m for m in selected if m!='Independent'])
    events={}
    for eqid in mch._prediction_event_ids(wide,2 if smoke else None):
        event=mch._full_data_event_frame(wide,eqid,d.period_cols,16 if smoke else None,0)
        if len(event)<8:continue
        raw=event[d.period_cols].to_numpy(float).ravel();obs=np.flatnonzero(np.isfinite(raw))
        events[int(eqid)]=(event,obs,raw[obs])
    tests=masks(events,d.periods,1 if smoke else 20)
    maskhash=hashlib.sha256()
    for eqid,cases in tests.items():
        for task,b,test in cases:
            maskhash.update(json.dumps([task,eqid,b['repeat'],b.get('scenario')]).encode())
            maskhash.update(test.astype('<i8').tobytes())
    records={task:[] for task in ['event_likelihood',*TASKS]}
    keys=('pairwise_empirical','pca','gh08','lmc_semivariogram','lmc_block_mle','kronecker','iox_full_semivariogram','sbss_semivariogram','multivariate_matern')
    for eqid,(event,obs,y) in events.items():
        for method in selected:
            raw=(np.diag(np.tile(variance,len(event))) if method=='Independent' else
                 mch._full_data_covariance(method,event,d,curves,variance,semi=semi,**{k:p.get(k) for k in keys}))
            fast=ConditionalCache(raw[np.ix_(obs,obs)],y,method,dataset)
            base=dict(method=method,eqid=eqid,fit_success=True,nearest_psd_used=fast.repaired,min_eigenvalue_before_repair=fast.minimum)
            records['event_likelihood'].append(dict(**base,n_observations=len(y),log_likelihood=fast.ll))
            for task,b,test in tests[eqid]:
                if task in jobs:records[task].append({**base,**b,**fast.score(test)})
            print(dataset,eqid,method,round(fast.ll,3),flush=True)
    for task in jobs:pd.DataFrame(records[task]).to_csv(output/f'{dataset}_{task}.csv',index=False)
    pd.DataFrame([dict(file=cache.name,sha256=digest)]).to_csv(output/'models.csv',index=False)
    if set(jobs)=={'event_likelihood',*TASKS}:
        table=pd.DataFrame(records['event_likelihood']).groupby('method').log_likelihood.sum().to_frame()
        for task,prefix in zip(TASKS,['pseudo','station','period']):
            table=table.join(pooled_scores(pd.DataFrame(records[task])).add_prefix(prefix+'_'))
        table.to_csv(output/('complete_case_summary.csv' if dataset=='complete' else 'full_summary.csv'))
    pd.DataFrame([dict(dataset=dataset,seed=0,repeats=1 if smoke else 20,events=len(events),observations=sum(len(x[2]) for x in events.values()),mask_sha256=maskhash.hexdigest(),models_sha256=digest,smoke=smoke)]).to_csv(output/'run.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['full','complete'],default='full')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--tasks',nargs='+',choices=['event_likelihood',*TASKS])
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    cache=args.cache or ROOT/('results/models_complete.pkl' if args.dataset=='complete' else 'results/models.pkl')
    output=args.output or ROOT/'results/analysis/prediction'/args.dataset/('smoke' if args.smoke else 'full_run')
    evaluate(cache,output,args.dataset,args.smoke,args.methods,args.tasks)
