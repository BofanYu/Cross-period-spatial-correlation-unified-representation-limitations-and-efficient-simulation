"""Rerun unconditional sampling benchmarks, keeping paper reference measurements unchanged."""
from __future__ import annotations
from pathlib import Path
import time
import platform
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from threadpoolctl import threadpool_info, threadpool_limits
ROOT=Path(__file__).resolve().parents[3]
from scripts.analysis.computation import samplers as ct

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


def measure(payload, n, q, draws, methods, repeat, seed, max_bytes, network_size, iox_dense=True):
    lat_all,lon_all=ct.synthetic_station_lat_lon(network_size,random_state=seed)
    lat,lon=lat_all[:n],lon_all[:n]
    rows=[]
    for method in methods:
        started=time.perf_counter()
        if method=="LMC conventional method":
            dense_lat,dense_lon=ct.synthetic_station_lat_lon(n,random_state=seed)
            prior,_=build_dense_lmc(ct,payload,dense_lat,dense_lon,tuple(range(q)),max_bytes)
        elif method == "IOX full semivariogram" and iox_dense:
            idx = tuple(range(q))
            columns = payload["data_state"]["period_cols"][:q]
            variance = np.var(payload["data_state"]["df_pca"][columns].to_numpy(float), axis=0)
            prior = ct._dense_curve_prior(payload, method, ct.haversine_distance_matrix(lat, lon),
                                          idx, variance, factor_jitter=1e-11,
                                          max_covariance_bytes=max_bytes)
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


def run_sampling(payload, stations=(40,), periods=(9,), realizations=(100,),
                 methods=("Kronecker", "PCA", "LMC"), repeats=1, seed=123,
                 max_dense_gb=1., iox_dense=True):
    """Return measured timings; dense IOX matches the supplementary comparator."""
    rows = []
    with threadpool_limits(limits=1):
        for n in stations:
            for q in periods:
                for draws in realizations:
                    for repeat in range(repeats):
                        for method in methods:
                            try:
                                rows.extend(measure(payload, n, q, draws, (method,), repeat, seed,
                                                    int(max_dense_gb * 1e9), max(stations), iox_dense))
                            except MemoryError as exc:
                                rows.append(dict(method=method, n_stations=n, n_periods=q,
                                    n_realizations=draws, repeat=repeat, status="not measured",
                                    reason=str(exc), setup_seconds=np.nan, sampling_seconds_total=np.nan,
                                    sampling_seconds_per_realization=np.nan, total_seconds=np.nan,
                                    first_realization_seconds=np.nan))
    return pd.DataFrame(rows)


def plot_summary(detail, sweep):
    """Convert new timing rows to the same columns used by the article plots."""
    groups = ["method", "n_stations", "n_periods", "n_realizations"]
    values = ["setup_seconds", "sampling_seconds_total", "sampling_seconds_per_realization",
              "total_seconds", "first_realization_seconds"]
    summary = detail.groupby(groups, as_index=False)[values].median()
    summary = summary.rename(columns={key: "median_" + key for key in values})
    summary["method"] = summary.method.replace({"Kronecker": "Kronecker semivariogram",
        "PCA": "PCA semivariogram", "LMC": "LMC semivariogram"})
    summary["station_count"] = summary.n_stations
    summary["period_count"] = summary.n_periods
    summary["n_realizations"] = summary.n_realizations
    return summary
