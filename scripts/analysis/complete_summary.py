"""Merge a complete-case LMC evaluation and regenerate five-model statistics."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scripts.analysis import core as c
from scripts.analysis.tables import MAIN,LABELS,pooled_scores

REPO=Path(__file__).resolve().parents[2]

def main():
    ev=REPO/'results/analysis/evaluation';out=REPO/'results/analysis/complete_case';fresh=REPO/'results/analysis/prediction/complete/lmc_wls45'
    method='LMC semivariogram'
    all_details=[]
    for task in ['event_likelihood','random_holdout','station_holdout','period_holdout']:
        path=ev/f'complete_{task}.csv';a=pd.read_csv(path);b=pd.read_csv(fresh/path.name)
        if task!='event_likelihood':
            b['task']=task;b['unit']=b['scenario'] if task=='period_holdout' else b['repeat'].astype(str)
        joined=pd.concat([a[a.method!=method],b],ignore_index=True)
        joined.to_csv(path,index=False)
        if task!='event_likelihood':all_details.append(joined)
        else:joined.to_csv(out/'event_log_likelihood.csv',index=False)
    details=pd.concat(all_details,ignore_index=True)
    random_rows=details.task!='period_holdout'
    details.loc[random_rows,'unit']=pd.to_numeric(details.loc[random_rows,'unit']).astype(int).astype(str)
    details.to_csv(out/'holdout_event_details.csv',index=False)
    table=pd.read_csv(ev/'complete_event_likelihood.csv').groupby('method').log_likelihood.sum().to_frame()
    units=[]
    for task,prefix in zip(['random_holdout','station_holdout','period_holdout'],['pseudo','station','period']):
        a=details[details.task==task];table=table.join(pooled_scores(a).add_prefix(prefix+'_'))
        agg=a.groupby(['method','unit'])[['sse','crps_sum','n_observations']].sum().reset_index()
        agg['rmse']=np.sqrt(agg.sse/agg.n_observations);agg['crps']=agg.crps_sum/agg.n_observations;agg['task']=task
        units.append(agg[['method','task','unit','rmse','crps']])
    table=table.loc[MAIN[2:]];table.to_csv(ev/'complete_case_summary.csv')
    units=pd.concat(units,ignore_index=True);units.to_csv(out/'replicate_and_period_scores.csv',index=False)
    stats=[]
    for task in units.task.unique():
        for metric in ['rmse','crps']:
            pivot=units[units.task==task].pivot(index='unit',columns='method',values=metric);ranks=pivot.rank(axis=1,method='average')
            for m in pivot:
                x=pivot[m];stats.append(dict(model=LABELS.get(m,m),task=task,metric=metric,mean=x.mean(),sd=x.std(ddof=1),se=x.std(ddof=1)/np.sqrt(len(x)) if task!='period_holdout' else np.nan,wins=int(np.isclose(x,pivot.min(axis=1),atol=1e-12,rtol=0).sum()),mean_rank=ranks[m].mean(),n=len(x)))
    pd.DataFrame(stats).to_csv(out/'replicate_statistics.csv',index=False)
    p=c.load_pickle_cross_platform(REPO/'results/models_complete.pkl');d=c.comparison_data_from_state(p['data_state']);h=d.h_bins
    wide=c.predictive_full_data_wide(d).dropna(subset=d.period_cols);var=c._full_data_period_variances(wide,d.period_cols)
    wls={}
    for m in MAIN[2:]:
        if m.startswith('PCA'):
            f=p['pca']['mle' if m=='PCA MLE' else 'semivariogram'];mix=np.asarray(f['std_pool'])[:,None]*f['U']*np.sqrt(f['evals'])[None,:]
            kernels=(1-np.asarray(f.get('nugget_fraction',np.zeros(9))))[:,None]*np.exp(-(h[None,:]/f['LE'][:,None])**f['gammaE'][:,None])
            g=np.einsum('ik,jk,kh->ijh',mix,mix,1-kernels)
        elif m=='Kronecker semivariogram':
            f=p['kronecker']['semivariogram'];g=(d.rho_0*np.sqrt(np.outer(var,var)))[:,:,None]*(1-np.exp(-(h/f['LE'])**f['gammaE']))
        else:
            f=p['lmc_block_mle' if m=='LMC block MLE' else 'lmc_semivariogram'];g=np.zeros((9,9,len(h)))
            for b,l,e in zip(f['B_raw'],f['length_scales'],f['gamma_pe']):
                kernel=np.zeros_like(h) if l<=0 else np.exp(-(h/l)**e);g+=b[:,:,None]*(1-kernel)
        delta=g-np.asarray(d.gamma_emp);wls[m]=float((delta**2/h)[np.triu_indices(9)].sum())
    table.insert(0,'WLS',pd.Series(wls));table.rename(index=LABELS).rename_axis('model').to_csv(out/'summary.csv')
    pd.DataFrame([dict(file='models_complete.pkl',sha256=hashlib.sha256((REPO/'results/models_complete.pkl').read_bytes()).hexdigest())]).to_csv(out/'models.csv',index=False)
    pd.DataFrame([dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),source='Event-level evaluation; see PAPER_SYNC.md') for f in sorted(ev.glob('*.csv')) if f.name!='sources.csv']).to_csv(ev/'sources.csv',index=False)
    print(table.to_string())

if __name__ == "__main__":
    main()
