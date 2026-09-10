"""Check sampler covariance, draw ordering and dense/structured LMC equivalence."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
from scripts.analysis import sampling as ct
from scripts.analysis.benchmark_sampling import build_dense_lmc


def main():
    payload=ct.load_cache(ROOT/"results/models.pkl")
    lat,lon=ct.synthetic_station_lat_lon(8,random_state=456)
    idx=(0,3,6)
    rows=[]
    with threadpool_limits(limits=1):
        priors,_=ct.build_semivariogram_priors(payload,lat,lon,period_indices=idx)
        dense,assembled=build_dense_lmc(ct,payload,lat,lon,idx,retain_assembled=True)
        lmc_cov=priors["LMC"].dense_covariance_for_validation()
        np.testing.assert_allclose(assembled,lmc_cov,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(dense.lower@dense.lower.T,lmc_cov,atol=1e-12,rtol=1e-12)
        for name,prior in priors.items():
            expected=prior.dense_covariance_for_validation()
            draws=prior.sample(np.random.default_rng(123),n_samples=40000)
            assert draws.shape==(8,3,40000)
            empirical=np.cov(draws.reshape(24,40000))
            relative=np.linalg.norm(empirical-expected)/np.linalg.norm(expected)
            assert relative < .045, (name,relative)
            rows.append(dict(method=name,relative_covariance_error=float(relative),draws=40000))
    target=ROOT/"results/analysis/sampling_validation.csv"
    target.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).assign(lmc_dense_structured_max_error=float(np.max(abs(assembled-lmc_cov)))).to_csv(target, index=False)
    print(f"Sampling checks passed: {target}")



if __name__=="__main__":
    main()
