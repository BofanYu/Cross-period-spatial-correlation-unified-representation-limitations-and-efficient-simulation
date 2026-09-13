"""Unconditional samplers for the paper's multi-period spatial models.

Outputs are ordered (station, period, realization). LMC, PCA, Separable and
SBSS use spatial/period factors; the archived IOX timing comparator used dense cached
curve covariance, as do Matérn, Pairwise and GH08. Pairwise and GH08 require
the same finite-network nearest-correlation repair used in the paper.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path, PosixPath, PureWindowsPath
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

Array = np.ndarray


# -----------------------------------------------------------------------------
# Cache loading and geometry
# -----------------------------------------------------------------------------


class _CrossPlatformPathUnpickler(pickle.Unpickler):
    """Load pickles containing pathlib paths created on another OS."""

    def find_class(self, module: str, name: str):
        if module == "pathlib" and name in {"WindowsPath", "PureWindowsPath"}:
            return PureWindowsPath
        if module == "pathlib" and name in {"PosixPath", "PurePosixPath"}:
            return PosixPath
        return super().find_class(module, name)


def load_cache(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        payload = _CrossPlatformPathUnpickler(handle).load()
    required = {"data_state", "pca", "kronecker", "lmc_semivariogram"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"Cache is missing required keys: {sorted(missing)}")
    return payload


def synthetic_station_lat_lon(
    n_stations: int,
    *,
    center_lat: float = 34.05,
    center_lon: float = -118.25,
    region_km: float = 120.0,
    random_state: int = 123,
) -> tuple[Array, Array]:
    """Generate reproducible stations in a local square region."""
    if n_stations <= 0:
        raise ValueError("n_stations must be positive")
    rng = np.random.default_rng(int(random_state))
    east_km = rng.uniform(-0.5 * region_km, 0.5 * region_km, n_stations)
    north_km = rng.uniform(-0.5 * region_km, 0.5 * region_km, n_stations)
    lat = center_lat + north_km / 111.0
    lon = center_lon + east_km / (111.0 * np.cos(np.radians(center_lat)))
    return lat.astype(float), lon.astype(float)


def haversine_distance_matrix(lat: Array, lon: Array) -> Array:
    """All-pairs great-circle distances in kilometres."""
    lat_r = np.radians(np.asarray(lat, dtype=float))
    lon_r = np.radians(np.asarray(lon, dtype=float))
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    aa = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_r)[:, None]
        * np.cos(lat_r)[None, :]
        * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * 6371.0 * np.arcsin(np.minimum(1.0, np.sqrt(aa)))


def powered_exponential_kernel(distance_km: Array, length_scale: float, exponent: float) -> Array:
    """Powered-exponential correlation, with ell<=0 interpreted as a nugget."""
    distance_km = np.asarray(distance_km, dtype=float)
    if float(length_scale) <= 0.0:
        out = np.zeros_like(distance_km)
        out[np.isclose(distance_km, 0.0)] = 1.0
        return out
    safe = np.maximum(distance_km, 0.0)
    out = np.exp(-((safe / float(length_scale)) ** float(exponent)))
    out[np.isclose(distance_km, 0.0)] = 1.0
    return 0.5 * (out + out.T)


# -----------------------------------------------------------------------------
# Numerically stable factors
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorDiagnostics:
    jitter: float = 0.0
    repaired: bool = False
    rank: int | None = None
    identity: bool = False


def stable_cholesky(
    matrix: Array,
    *,
    base_jitter: float = 1e-11,
    max_tries: int = 8,
    identity_tolerance: float = 1e-13,
) -> tuple[Array | None, FactorDiagnostics]:
    """Fast Cholesky, with an identity shortcut and eigenvalue-repair fallback.

    ``None`` is returned as the factor for a numerical identity matrix; callers
    should then apply the identity directly rather than multiplying by it.
    """
    a = np.asarray(matrix, dtype=float)
    a = 0.5 * (a + a.T)
    n = a.shape[0]
    if a.ndim != 2 or a.shape[1] != n:
        raise ValueError("matrix must be square")

    diag = np.diag(a)
    if np.allclose(diag, 1.0, rtol=0.0, atol=identity_tolerance):
        off = a - np.eye(n)
        if np.max(np.abs(off)) <= identity_tolerance:
            return None, FactorDiagnostics(identity=True, rank=n)

    scale = max(float(np.mean(np.abs(diag))), 1.0)
    last_error: Exception | None = None
    for attempt in range(max_tries):
        jitter = 0.0 if attempt == 0 else base_jitter * scale * (10.0 ** (attempt - 1))
        try:
            lower = np.linalg.cholesky(a + jitter * np.eye(n))
            return lower, FactorDiagnostics(jitter=float(jitter), rank=n)
        except np.linalg.LinAlgError as exc:
            last_error = exc

    # Rare fallback: clip only numerically negative eigenvalues, then factor.
    evals, evecs = np.linalg.eigh(a)
    floor = base_jitter * scale
    evals_clipped = np.maximum(evals, floor)
    repaired = (evals.min() < -100.0 * floor) or not np.allclose(evals, evals_clipped)
    repaired_a = (evecs * evals_clipped) @ evecs.T
    repaired_a = 0.5 * (repaired_a + repaired_a.T)
    try:
        lower = np.linalg.cholesky(repaired_a)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise np.linalg.LinAlgError(f"Cholesky failed after repair: {last_error}; {exc}") from exc
    return lower, FactorDiagnostics(jitter=float(floor), repaired=bool(repaired), rank=n)


def psd_rectangular_factor(
    matrix: Array,
    *,
    relative_tolerance: float = 1e-11,
) -> tuple[Array, FactorDiagnostics]:
    """Return G with G @ G.T equal to a PSD matrix, retaining its numerical rank."""
    a = np.asarray(matrix, dtype=float)
    a = 0.5 * (a + a.T)
    evals, evecs = np.linalg.eigh(a)
    scale = max(float(np.max(np.abs(evals))), 1.0)
    tol = relative_tolerance * scale
    if evals.min() < -100.0 * tol:
        # The cached object should be PSD.  Clip a materially negative part but
        # report it so the smoke diagnostics expose the repair.
        repaired = True
    else:
        repaired = bool(evals.min() < 0.0)
    positive = evals > tol
    if not np.any(positive):
        raise np.linalg.LinAlgError("PSD factor has zero numerical rank")
    factor = evecs[:, positive] * np.sqrt(np.maximum(evals[positive], 0.0))[None, :]
    return factor, FactorDiagnostics(repaired=repaired, rank=int(np.sum(positive)))


def apply_factor(lower: Array | None, z: Array) -> Array:
    return z if lower is None else lower @ z




# -----------------------------------------------------------------------------
# Structured Gaussian prior classes
# -----------------------------------------------------------------------------


class StructuredPrior:
    """Interface shared by the three fitted Gaussian residual models."""

    name: str
    n_stations: int
    n_periods: int

    def sample(self, rng: np.random.Generator, n_samples: int = 1) -> Array:
        """Return an array with shape (station, period, sample)."""
        raise NotImplementedError

    def matvec(self, flat_vector: Array) -> Array:
        """Apply the full station-period covariance in C-order, matrix-free."""
        raise NotImplementedError

    def diagonal_flat(self) -> Array:
        raise NotImplementedError

    def zero_lag_period_covariance(self) -> Array:
        """Within-station cross-period covariance used by block preconditioning."""
        raise NotImplementedError

    def dense_covariance_for_validation(self) -> Array:
        """Small-n validation helper; never used by the scalable samplers."""
        eye = np.eye(self.n_stations * self.n_periods)
        return np.column_stack([self.matvec(eye[:, j]) for j in range(eye.shape[1])])


@dataclass
class KroneckerPrior(StructuredPrior):
    spatial_cov: Array
    period_cov: Array
    spatial_lower: Array | None
    period_lower: Array | None
    spatial_factor_info: FactorDiagnostics
    period_factor_info: FactorDiagnostics
    name: str = "Kronecker"

    @property
    def n_stations(self) -> int:
        return int(self.spatial_cov.shape[0])

    @property
    def n_periods(self) -> int:
        return int(self.period_cov.shape[0])

    def sample(self, rng: np.random.Generator, n_samples: int = 1) -> Array:
        z = rng.standard_normal((self.n_stations, self.n_periods, int(n_samples)))
        if self.spatial_lower is None:
            left = z
        else:
            # Flatten period/sample into one BLAS-friendly right-hand-side
            # dimension.  This is algebraically identical to the former 3-D
            # einsum, but avoids its slow generic contraction path.
            left = (
                self.spatial_lower
                @ z.reshape(self.n_stations, self.n_periods * int(n_samples))
            ).reshape(self.n_stations, self.n_periods, int(n_samples))
        if self.period_lower is None:
            return left
        return np.einsum(
            "iqs,pq->ips", left, self.period_lower, optimize=True
        )

    def matvec(self, flat_vector: Array) -> Array:
        v = np.asarray(flat_vector, dtype=float).reshape(self.n_stations, self.n_periods, order="C")
        out = self.spatial_cov @ v @ self.period_cov
        return out.reshape(-1, order="C")

    def diagonal_flat(self) -> Array:
        diag = np.outer(np.diag(self.spatial_cov), np.diag(self.period_cov))
        return diag.reshape(-1, order="C")

    def zero_lag_period_covariance(self) -> Array:
        return float(self.spatial_cov[0, 0]) * self.period_cov

    def dense_covariance_for_validation(self) -> Array:
        return np.kron(self.spatial_cov, self.period_cov)




@dataclass
class PCAPrior(StructuredPrior):
    spatial_covariances: list[Array]
    spatial_lowers: list[Array | None]
    spatial_factor_info: list[FactorDiagnostics]
    loadings: Array  # q x k
    eigenvalues: Array  # k
    period_std: Array  # q
    name: str = "PCA"

    @property
    def n_stations(self) -> int:
        return int(self.spatial_covariances[0].shape[0])

    @property
    def n_periods(self) -> int:
        return int(self.loadings.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.loadings.shape[1])

    @property
    def period_factors(self) -> Array:
        # Column k is g_k such that B_k = g_k g_k^T.
        return self.period_std[:, None] * self.loadings * np.sqrt(self.eigenvalues)[None, :]


    def sample_latent_scores(self, rng: np.random.Generator, n_samples: int = 1) -> Array:
        scores = np.empty((self.n_stations, self.n_components, int(n_samples)), dtype=float)
        for k, lower in enumerate(self.spatial_lowers):
            z = rng.standard_normal((self.n_stations, int(n_samples)))
            scores[:, k, :] = np.sqrt(self.eigenvalues[k]) * apply_factor(lower, z)
        return scores

    def scores_to_periods(self, scores: Array) -> Array:
        return np.einsum("nks,qk,q->nqs", scores, self.loadings, self.period_std)

    def sample(self, rng: np.random.Generator, n_samples: int = 1) -> Array:
        return self.scores_to_periods(self.sample_latent_scores(rng, n_samples))

    def matvec(self, flat_vector: Array) -> Array:
        v = np.asarray(flat_vector, dtype=float).reshape(self.n_stations, self.n_periods, order="C")
        out = np.zeros_like(v)
        for spatial, g in zip(self.spatial_covariances, self.period_factors.T):
            # K V (g g^T) = [K (V g)] g^T, exploiting rank one.
            out += (spatial @ (v @ g))[:, None] * g[None, :]
        return out.reshape(-1, order="C")

    def diagonal_flat(self) -> Array:
        diag = np.zeros((self.n_stations, self.n_periods), dtype=float)
        for spatial, g in zip(self.spatial_covariances, self.period_factors.T):
            diag += np.diag(spatial)[:, None] * (g * g)[None, :]
        return diag.reshape(-1, order="C")

    def zero_lag_period_covariance(self) -> Array:
        out = np.zeros((self.n_periods, self.n_periods), dtype=float)
        for spatial, g in zip(self.spatial_covariances, self.period_factors.T):
            out += float(spatial[0, 0]) * np.outer(g, g)
        return 0.5 * (out + out.T)

    def dense_covariance_for_validation(self) -> Array:
        out = np.zeros(
            (self.n_stations * self.n_periods, self.n_stations * self.n_periods),
            dtype=float,
        )
        for spatial, g in zip(self.spatial_covariances, self.period_factors.T):
            out += np.kron(spatial, np.outer(g, g))
        return 0.5 * (out + out.T)



@dataclass
class SBSSPrior(PCAPrior):
    """Independent SBSS latent fields with one batched period-mixing step."""

    name: str = "SBSS semivariogram"


@dataclass
class LMCComponent:
    spatial_cov: Array
    spatial_lower: Array | None
    spatial_factor_info: FactorDiagnostics
    period_factor: Array  # q x rank; B = A A.T
    period_factor_info: FactorDiagnostics


@dataclass
class LMCPrior(StructuredPrior):
    components: list[LMCComponent]
    name: str = "LMC"

    @property
    def n_stations(self) -> int:
        return int(self.components[0].spatial_cov.shape[0])

    @property
    def n_periods(self) -> int:
        return int(self.components[0].period_factor.shape[0])

    def sample(self, rng: np.random.Generator, n_samples: int = 1) -> Array:
        out = np.zeros((self.n_stations, self.n_periods, int(n_samples)), dtype=float)
        for component in self.components:
            rank = component.period_factor.shape[1]
            z = rng.standard_normal((self.n_stations, rank, int(n_samples)))
            if component.spatial_lower is None:
                spatial_draw = z
            else:
                # Use one 2-D GEMM for every component instead of a generic
                # 3-D einsum over the station dimension.
                spatial_draw = (
                    component.spatial_lower
                    @ z.reshape(self.n_stations, rank * int(n_samples))
                ).reshape(self.n_stations, rank, int(n_samples))
            out += np.einsum(
                "nrs,qr->nqs",
                spatial_draw,
                component.period_factor,
                optimize=True,
            )
        return out

    def matvec(self, flat_vector: Array) -> Array:
        v = np.asarray(flat_vector, dtype=float).reshape(self.n_stations, self.n_periods, order="C")
        out = np.zeros_like(v)
        for component in self.components:
            a = component.period_factor
            # K V B, with B=A A.T, evaluated without forming B.
            out += component.spatial_cov @ (v @ a) @ a.T
        return out.reshape(-1, order="C")

    def diagonal_flat(self) -> Array:
        diag = np.zeros((self.n_stations, self.n_periods), dtype=float)
        for component in self.components:
            b_diag = np.sum(component.period_factor**2, axis=1)
            diag += np.diag(component.spatial_cov)[:, None] * b_diag[None, :]
        return diag.reshape(-1, order="C")

    def zero_lag_period_covariance(self) -> Array:
        out = np.zeros((self.n_periods, self.n_periods), dtype=float)
        for component in self.components:
            out += float(component.spatial_cov[0, 0]) * (
                component.period_factor @ component.period_factor.T
            )
        return 0.5 * (out + out.T)

    def dense_covariance_for_validation(self) -> Array:
        out = np.zeros(
            (self.n_stations * self.n_periods, self.n_stations * self.n_periods),
            dtype=float,
        )
        for component in self.components:
            b = component.period_factor @ component.period_factor.T
            out += np.kron(component.spatial_cov, b)
        return 0.5 * (out + out.T)


@dataclass
class DenseCurvePrior(StructuredPrior):
    """Exact finite-network prior built from a matrix-valued correlation curve."""

    covariance: Array
    lower: Array | None
    factor_info: FactorDiagnostics
    station_count: int
    period_count: int
    name: str

    @property
    def n_stations(self) -> int:
        return int(self.station_count)

    @property
    def n_periods(self) -> int:
        return int(self.period_count)

    def sample(self, rng: np.random.Generator, n_samples: int = 1) -> Array:
        z = rng.standard_normal(
            (self.n_stations * self.n_periods, int(n_samples))
        )
        flat = z if self.lower is None else self.lower @ z
        return flat.reshape(
            self.n_stations, self.n_periods, int(n_samples), order="C"
        )

    def matvec(self, flat_vector: Array) -> Array:
        return self.covariance @ np.asarray(flat_vector, dtype=float)

    def diagonal_flat(self) -> Array:
        return np.diag(self.covariance).copy()

    def zero_lag_period_covariance(self) -> Array:
        return self.covariance[: self.n_periods, : self.n_periods].copy()

    def dense_covariance_for_validation(self) -> Array:
        return self.covariance.copy()


# -----------------------------------------------------------------------------
# Model extraction from the cached semivariogram fits
# -----------------------------------------------------------------------------


def normalize_period_indices(n_periods_total: int, period_indices: Sequence[int] | None) -> Array:
    if period_indices is None:
        return np.arange(n_periods_total, dtype=int)
    idx = np.asarray(period_indices, dtype=int)
    if idx.ndim != 1 or len(idx) == 0:
        raise ValueError("period_indices must be a non-empty one-dimensional sequence")
    if np.any(idx < 0) or np.any(idx >= n_periods_total):
        raise IndexError("period index out of range")
    return np.unique(idx)


def _dense_curve_prior(
    payload: Mapping,
    method: str,
    distance: Array,
    period_idx: Array,
    period_variances: Array,
    *,
    factor_jitter: float,
    max_covariance_bytes: int,
) -> DenseCurvePrior:
    """Build the exact finite-network covariance represented by cached curves."""
    curve_names = {
        "Pairwise empirical": "Pairwise empirical semivariogram",
        "GH08": "GH08 semivariogram",
        "IOX full semivariogram": "IOX full semivariogram",
        "Multivariate Matern semivariogram": "Multivariate Matern semivariogram",
    }
    cache_name = curve_names[method]
    h_grid, curves_all = payload["cross_period_psd"]["curves"][cache_name]
    h_grid = np.asarray(h_grid, dtype=float)
    curves = np.asarray(curves_all, dtype=float)[np.ix_(period_idx, period_idx, np.arange(len(h_grid)))]
    n = int(distance.shape[0])
    q = int(len(period_idx))
    dimension = n * q
    covariance_bytes = dimension * dimension * np.dtype(float).itemsize
    if covariance_bytes > int(max_covariance_bytes):
        raise MemoryError(
            f"{method} exact dense covariance needs {covariance_bytes / 1e6:.1f} MB "
            f"for {n} stations x {q} periods; configured ceiling is "
            f"{int(max_covariance_bytes) / 1e6:.1f} MB"
        )

    # The cached comparison grids stop at 100 or 120 km.  Beyond the fitted
    # range, retain the final fitted correlation rather than extrapolating a
    # polynomial through the tail.
    flat_distance = np.asarray(distance, dtype=float).ravel()
    scale = np.sqrt(np.outer(period_variances, period_variances))
    covariance_4d = np.empty((n, q, n, q), dtype=float)
    for i in range(q):
        for j in range(q):
            rho = np.interp(
                flat_distance,
                h_grid,
                curves[i, j],
                left=float(curves[i, j, 0]),
                right=float(curves[i, j, -1]),
            ).reshape(n, n)
            covariance_4d[:, i, :, j] = scale[i, j] * rho
    covariance = covariance_4d.reshape(dimension, dimension)
    covariance = 0.5 * (covariance + covariance.T)
    if method in {"Pairwise empirical", "GH08"}:
        # These pairwise/screening constructions are not guaranteed to define
        # a PSD covariance on a finite station-period grid.  Apply the same
        # fixed-diagonal Qi-Sun (2006) nearest-correlation repair used by the
        # predictive/simulation diagnostics before attempting factorization.
        from scripts.common.core import nearest_psd_matrix

        covariance = nearest_psd_matrix(
            covariance,
            eps=max(float(factor_jitter), 1.0e-8),
            preserve_diagonal=True,
        )
        lower, initial_info = stable_cholesky(
            covariance, base_jitter=factor_jitter
        )
        factor_info = FactorDiagnostics(
            jitter=initial_info.jitter,
            repaired=True,
            rank=initial_info.rank,
            identity=initial_info.identity,
        )
    else:
        lower, factor_info = stable_cholesky(
            covariance, base_jitter=factor_jitter
        )
    return DenseCurvePrior(
        covariance=covariance,
        lower=lower,
        factor_info=factor_info,
        station_count=n,
        period_count=q,
        name=method,
    )


def _sbss_prior(
    payload: Mapping,
    distance: Array,
    period_idx: Array,
    period_variances: Array,
    *,
    factor_jitter: float,
) -> SBSSPrior:
    """Build SBSS as independent latent fields with batched period mixing."""
    fit = payload["sbss_semivariogram"]
    mixing = np.asarray(fit["mixing_raw"], dtype=float)[period_idx, :]
    zero_covariance = mixing @ mixing.T
    model_sd = np.sqrt(np.clip(np.diag(zero_covariance), 1e-15, None))
    target_sd = np.sqrt(np.clip(period_variances, 1e-15, None))
    # Rescale the native mixing so its lag-zero marginal variances match the
    # same empirical residual variances used for all other benchmark methods.
    mixing = mixing * (target_sd / model_sd)[:, None]
    latent = fit["latent_parameters"]
    if isinstance(latent, pd.DataFrame):
        length_scales = latent["length_scale_km"].to_numpy(dtype=float)
        exponents = latent["powered_exponent"].to_numpy(dtype=float)
    else:
        length_scales = np.asarray(latent["length_scale_km"], dtype=float)
        exponents = np.asarray(latent["powered_exponent"], dtype=float)

    spatial_covariances: list[Array] = []
    spatial_lowers: list[Array | None] = []
    spatial_infos: list[FactorDiagnostics] = []
    for length_scale, exponent in zip(length_scales, exponents):
        spatial = powered_exponential_kernel(distance, length_scale, exponent)
        lower, spatial_info = stable_cholesky(
            spatial, base_jitter=factor_jitter
        )
        spatial_covariances.append(spatial)
        spatial_lowers.append(lower)
        spatial_infos.append(spatial_info)
    return SBSSPrior(
        spatial_covariances=spatial_covariances,
        spatial_lowers=spatial_lowers,
        spatial_factor_info=spatial_infos,
        loadings=mixing,
        eigenvalues=np.ones(mixing.shape[1], dtype=float),
        period_std=np.ones(mixing.shape[0], dtype=float),
    )


def build_semivariogram_priors(
    payload: Mapping,
    lat: Array,
    lon: Array,
    *,
    period_indices: Sequence[int] | None = None,
    factor_jitter: float = 1e-11,
    methods: Sequence[str] | None = None,
    max_dense_covariance_bytes: int = 350_000_000,
) -> tuple[dict[str, StructuredPrior], dict]:
    """Build selected native priors for one station network.

    Parameters
    ----------
    methods
        Any subset of the implemented cached models. Restricting this
        argument is important for method-specific setup timing because unused
        covariance matrices and Cholesky factors are not constructed.
    """
    allowed = {
        "Pairwise empirical",
        "GH08",
        "Kronecker",
        "PCA",
        "LMC",
        "IOX full semivariogram",
        "SBSS semivariogram",
        "Multivariate Matern semivariogram",
    }
    wanted = allowed if methods is None else set(methods)
    unknown = wanted.difference(allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if not wanted:
        raise ValueError("methods must contain at least one model")

    data = payload["data_state"]
    periods_all = np.asarray(data["periods"], dtype=float)
    idx = normalize_period_indices(len(periods_all), period_indices)
    periods = periods_all[idx]
    period_cols_all = list(data["period_cols"])
    period_cols = [period_cols_all[i] for i in idx]
    distance = haversine_distance_matrix(lat, lon)

    y = data["df_pca"][period_cols].to_numpy(dtype=float)
    period_variances = np.var(y, axis=0)
    rho0 = np.asarray(data["rho_0"], dtype=float)[np.ix_(idx, idx)]
    priors: dict[str, StructuredPrior] = {}

    if "Kronecker" in wanted:
        kron_fit = payload["kronecker"]["semivariogram"]
        k_space = powered_exponential_kernel(distance, kron_fit["LE"], kron_fit["gammaE"])
        c_period = rho0 * np.sqrt(np.outer(period_variances, period_variances))
        l_space, info_space = stable_cholesky(k_space, base_jitter=factor_jitter)
        l_period, info_period = stable_cholesky(c_period, base_jitter=factor_jitter)
        priors["Kronecker"] = KroneckerPrior(
            spatial_cov=k_space,
            period_cov=c_period,
            spatial_lower=l_space,
            period_lower=l_period,
            spatial_factor_info=info_space,
            period_factor_info=info_period,
        )

    if "PCA" in wanted:
        # A restriction to selected output periods retains every latent PC from
        # the fitted full model; only the loading rows and period SDs are subset.
        pca_fit = payload["pca"]["semivariogram"]
        loadings = np.asarray(pca_fit["U"], dtype=float)[idx, :]
        eigenvalues = np.asarray(pca_fit["evals"], dtype=float)
        period_std = np.asarray(pca_fit["std_pool"], dtype=float)[idx]
        pca_spatial: list[Array] = []
        pca_lowers: list[Array | None] = []
        pca_infos: list[FactorDiagnostics] = []
        for ell, gamma in zip(np.asarray(pca_fit["LE"]), np.asarray(pca_fit["gammaE"])):
            k_pc = powered_exponential_kernel(distance, float(ell), float(gamma))
            lower, info = stable_cholesky(k_pc, base_jitter=factor_jitter)
            pca_spatial.append(k_pc)
            pca_lowers.append(lower)
            pca_infos.append(info)
        priors["PCA"] = PCAPrior(
            spatial_covariances=pca_spatial,
            spatial_lowers=pca_lowers,
            spatial_factor_info=pca_infos,
            loadings=loadings,
            eigenvalues=eigenvalues,
            period_std=period_std,
        )

    if "LMC" in wanted:
        # Retain the numerical rank of each B matrix instead of forcing a
        # full-rank Cholesky by adding jitter.
        lmc_fit = payload["lmc_semivariogram"]
        b_matrices = [
            np.asarray(b, dtype=float)[np.ix_(idx, idx)]
            for b in lmc_fit["B_raw"]
        ]
        length_scales = np.asarray(lmc_fit["length_scales"], dtype=float)
        exponents = np.asarray(lmc_fit["gamma_pe"], dtype=float)
        if len(exponents) == 1:
            exponents = np.repeat(exponents, len(length_scales))
        components: list[LMCComponent] = []
        for b, ell, gamma in zip(b_matrices, length_scales, exponents):
            k_lmc = powered_exponential_kernel(distance, float(ell), float(gamma))
            lower, spatial_info = stable_cholesky(k_lmc, base_jitter=factor_jitter)
            a, period_info = psd_rectangular_factor(b)
            components.append(
                LMCComponent(
                    spatial_cov=k_lmc,
                    spatial_lower=lower,
                    spatial_factor_info=spatial_info,
                    period_factor=a,
                    period_factor_info=period_info,
                )
            )
        priors["LMC"] = LMCPrior(components=components)

    if "SBSS semivariogram" in wanted:
        priors["SBSS semivariogram"] = _sbss_prior(
            payload,
            distance,
            idx,
            period_variances,
            factor_jitter=factor_jitter,
        )

    if "IOX full semivariogram" in wanted:
        from scripts.analysis.computation.iox_sampling import build_iox_prior
        priors["IOX full semivariogram"] = build_iox_prior(payload, lat, lon, idx, period_variances)

    for dense_method in (
        "Pairwise empirical",
        "GH08",
        "Multivariate Matern semivariogram",
    ):
        if dense_method in wanted:
            priors[dense_method] = _dense_curve_prior(
                payload,
                dense_method,
                distance,
                idx,
                period_variances,
                factor_jitter=factor_jitter,
                max_covariance_bytes=int(max_dense_covariance_bytes),
            )

    metadata = {
        "period_indices": idx,
        "periods": periods,
        "period_cols": period_cols,
        "period_variances_for_kronecker": period_variances,
        "rho0_min_eigenvalue": float(
            np.linalg.eigvalsh(0.5 * (rho0 + rho0.T)).min()
        ),
    }
    return priors, metadata


# -----------------------------------------------------------------------------

def format_log_time_axis(ax, values=None, base: float = 10.0):
    """Use math-style log ticks and expand limits to full powers of ten."""
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    ax.set_yscale("log", base=base)
    if values is not None:
        positive = np.asarray(values, dtype=float)
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
        if positive.size:
            lo_exp = np.floor(np.log10(float(np.nanmin(positive))))
            hi_exp = np.ceil(np.log10(float(np.nanmax(positive))))
            y_min = base ** lo_exp
            y_max = base ** hi_exp
            if y_max <= y_min:
                y_max = y_min * base
            ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(LogLocator(base=base))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=base))
    ax.yaxis.set_minor_locator(LogLocator(base=base, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    return ax
