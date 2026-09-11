"""Rerun unconditional sampling benchmarks, keeping paper reference measurements unchanged."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import platform
from scripts.analysis import write_metadata
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from threadpoolctl import threadpool_info, threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
from scripts.analysis import sampling as ct

class DenseLMCConventionalPrior:
    def __init__(self, lower, n_stations, n_periods):
        self.lower = lower
        self.n_stations = n_stations
        self.n_periods = n_periods

    def sample(self, rng, n_samples=1):
        z = rng.standard_normal((self.n_stations * self.n_periods, int(n_samples)))
        return (self.lower @ z).reshape(
            self.n_stations, self.n_periods, int(n_samples), order="C"
        )

def build_dense_lmc(ct, payload, lat, lon, period_indices, max_bytes=None, *, retain_assembled=False):
    """Assemble every covariance entry, then factor the entire (n*q)^2 matrix.

    Station-major ordering: Sigma[i*q+r,j*q+s] = sum_k K_k[i,j] B_k[r,s].
    Only small B matrices receive the same PSD cleanup as the existing LMC.
    No spatial factorization, Kronecker factorization, or latent-field sampling
    is used. The generic dense Cholesky routine is the existing benchmark's.
    """
    idx = ct.normalize_period_indices(len(payload["data_state"]["periods"]), period_indices)
    n, q = len(lat), len(idx)
    covariance_bytes = (n * q) ** 2 * np.dtype(float).itemsize
    if max_bytes is not None and covariance_bytes > max_bytes:
        raise MemoryError(f"Full covariance needs {covariance_bytes} bytes; limit={max_bytes}")
    distance = ct.haversine_distance_matrix(lat, lon)
    fit = payload["lmc_semivariogram"]
    scales = np.asarray(fit["length_scales"], dtype=float)
    exponents = np.atleast_1d(np.asarray(fit["gamma_pe"], dtype=float))
    if len(exponents) == 1:
        exponents = np.repeat(exponents, len(scales))
    assert len(fit["B_raw"]) == len(scales) == len(exponents)
    # Fortran order lets LAPACK overwrite this same allocation during Cholesky,
    # avoiding another full-matrix allocation while remaining a generic dense
    # factorization. Station/period entries still use station-major indexing.
    covariance = np.zeros((n * q, n * q), dtype=float, order="F")
    for b_raw, ell, gamma in zip(fit["B_raw"], scales, exponents):
        b = np.asarray(b_raw, dtype=float)[np.ix_(idx, idx)]
        a, _ = ct.psd_rectangular_factor(b)
        b = a @ a.T  # Exactly the effective B used by LMCPrior.
        kernel = ct.powered_exponential_kernel(distance, ell, gamma)
        for r in range(q):
            for s in range(q):
                covariance[r::q, s::q] += b[r, s] * kernel
    assembled = covariance.copy() if retain_assembled else None
    lower = cholesky(
        covariance, lower=True, overwrite_a=True, check_finite=False
    )
    if not np.shares_memory(lower, covariance):
        raise MemoryError("Dense Cholesky unexpectedly allocated a second full matrix")
    return DenseLMCConventionalPrior(lower, n, q), assembled

METHODS = ("Kronecker", "PCA", "LMC", "LMC conventional method", "IOX full semivariogram", "SBSS semivariogram", "Multivariate Matern semivariogram")


def measure(payload, n, q, draws, methods, repeat, seed, max_bytes, network_size):
    lat_all,lon_all=ct.synthetic_station_lat_lon(network_size,random_state=seed)
    lat,lon=lat_all[:n],lon_all[:n]
    rows=[]
    for method in methods:
        started=time.perf_counter()
        if method=="LMC conventional method":
            dense_lat,dense_lon=ct.synthetic_station_lat_lon(n,random_state=seed)
            prior,_=build_dense_lmc(ct,payload,dense_lat,dense_lon,tuple(range(q)),max_bytes)
        else:
            priors,_=ct.build_semivariogram_priors(payload,lat,lon,
                period_indices=tuple(range(q)),methods=[method],max_dense_covariance_bytes=max_bytes)
            prior=priors[method]
            del priors
        setup=time.perf_counter()-started
        started=time.perf_counter()
        samples=prior.sample(np.random.default_rng(seed+repeat),n_samples=draws)
        elapsed=time.perf_counter()-started
        assert samples.shape==(n,q,draws) and np.isfinite(samples).all()
        rows.append(dict(method=method,n_stations=n,n_periods=q,n_realizations=draws,repeat=repeat,
            setup_seconds=setup,sampling_seconds_total=elapsed,
            sampling_seconds_per_realization=elapsed/draws,total_seconds=setup+elapsed,
            first_realization_seconds=setup+elapsed/draws))
        print(f"{method}: n={n}, m={q}, draws={draws}, total={setup+elapsed:.4g} s",flush=True)
        del prior,samples
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=ROOT/"results/models.pkl")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results/analysis/sampling")
    parser.add_argument("--stations",nargs="+",type=int,default=[40])
    parser.add_argument("--periods",nargs="+",type=int,default=[9])
    parser.add_argument("--realizations",nargs="+",type=int,default=[100])
    parser.add_argument("--methods",nargs="+",default=list(METHODS))
    parser.add_argument("--repeats",type=int,default=1)
    parser.add_argument("--seed",type=int,default=123)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--max-dense-gb",type=float,default=1.)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    payload=ct.load_cache(args.cache)
    rows=[]
    with threadpool_limits(limits=args.threads):
        for n in args.stations:
            for q in args.periods:
                for draws in args.realizations:
                    for repeat in range(args.repeats):
                        rows.extend(measure(payload,n,q,draws,args.methods,repeat,args.seed,int(args.max_dense_gb*1e9),max(args.stations)))
    detail=pd.DataFrame(rows)
    detail.to_csv(args.output_dir/"sampling_detail.csv",index=False)
    metrics=[c for c in detail if c.endswith("seconds") or c.startswith("sampling_seconds")]
    summary=detail.groupby(["method","n_stations","n_periods","n_realizations"],sort=False)[metrics].median().reset_index()
    summary.to_csv(args.output_dir/"sampling_summary.csv",index=False)
    metadata={"python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,
        "threads":args.threads,"seed":args.seed,"repeats":args.repeats,"blas":threadpool_info(),
        "timing_definition":"First realization = setup + batch average sampling per realization; fit/data loading and station generation excluded.",
        "iox_definition":"Native structured IOX: period mixing followed by marginal PE spatial factors, zero nugget. Archived figure timings are historical and not overwritten."}
    for library in metadata["blas"]: library.pop("filepath",None)
    write_metadata(args.output_dir / "run_config.csv", metadata)


if __name__=="__main__":
    main()
