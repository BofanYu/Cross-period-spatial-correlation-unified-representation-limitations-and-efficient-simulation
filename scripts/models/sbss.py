"""Spatial blind source separation for the nine-period common-record data.

The data preparation, empirical variograms, powered-exponential latent fits,
and PCA-style reconstruction follow methods_comparison.ipynb.  PCA's single
correlation-matrix eigendecomposition is replaced by SBSS whitening and joint
diagonalization of nine event-block ring local-covariance matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize


PERIODS = np.asarray((0.01, 0.03, 0.06, 0.10, 0.30, 0.60, 1.00, 3.00, 6.00))
RINGS_KM = tuple((float(inner), float(inner + 10)) for inner in range(0, 90, 10))


@dataclass
class CommonData:
    frame: pd.DataFrame
    period_columns: list[str]
    periods: np.ndarray
    values: np.ndarray
    standardized_values: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    period_rms: np.ndarray


@dataclass
class EmpiricalVariograms:
    h_bins: np.ndarray
    gamma: np.ndarray
    counts: np.ndarray


@dataclass
class LatentFit:
    scores: np.ndarray
    empirical: EmpiricalVariograms
    sills: np.ndarray
    length_scales: np.ndarray
    exponents: np.ndarray
    wss: np.ndarray
    rho_fine: np.ndarray
    optimizer_rows: list[dict]


@dataclass
class SBSSFit:
    periods: np.ndarray
    rings_km: tuple[tuple[float, float], ...]
    scatter_zero: np.ndarray
    ring_scatters: np.ndarray
    ring_ordered_pair_counts: np.ndarray
    whitening: np.ndarray
    whitened_ring_scatters: np.ndarray
    joint_rotation: np.ndarray
    unmixing_standardized: np.ndarray
    mixing_standardized: np.ndarray
    mixing_raw: np.ndarray
    diagonalized_ring_scatters: np.ndarray
    spatial_profiles: np.ndarray
    spatial_scores: np.ndarray
    latent: LatentFit
    h_fine: np.ndarray
    covariance_fine: np.ndarray
    correlation_fine: np.ndarray
    semivariogram_fine: np.ndarray
    empirical_observed: EmpiricalVariograms
    empirical_auto_sills: np.ndarray
    cross_sills: np.ndarray
    wss_45: float
    joint_diagnostics: dict


def _column_to_period(column: str) -> float:
    return float(column.split("WT")[-1]) / 100.0


def select_period_columns(frame: pd.DataFrame, periods=PERIODS) -> list[str]:
    available = [name for name in frame.columns if name.startswith("scaled_deltaW") and "WT" in name]
    pairs = [(_column_to_period(name), name) for name in available]
    selected: list[str] = []
    for period in periods:
        matches = [name for value, name in pairs if np.isclose(value, period, rtol=0.0, atol=1.0e-10)]
        if not matches:
            raise KeyError(f"No scaled_deltaW column found for T={period:g}s")
        selected.append(matches[0])
    return selected


def load_pca_common_data(path: str | Path) -> CommonData:
    """Apply the exact common-record and no-recentering PCA notebook preparation."""
    source = pd.read_csv(path)
    required_metadata = ["eqid", "station_latitude", "station_longitude"]
    missing = [name for name in required_metadata if name not in source.columns]
    if missing:
        raise KeyError(f"Input is missing columns: {missing}")
    period_columns = select_period_columns(source, PERIODS)
    frame = source[[*required_metadata, *period_columns]].dropna().copy()
    frame["eqid"] = frame["eqid"].astype(str)
    # Stable sorting makes every event a contiguous block. It changes no
    # estimate, but lets the event-wise matrix operations share one index map.
    frame = frame.sort_values("eqid", kind="stable").reset_index(drop=True)
    values = frame[period_columns].to_numpy(dtype=float)

    # This deliberately matches corr_matrix() in the notebook: residuals are
    # not re-centered and the divisor is N, not N-1.
    covariance = (values.T @ values) / len(values)
    period_rms = np.sqrt(np.clip(np.diag(covariance), 1.0e-15, None))
    correlation = covariance / np.outer(period_rms, period_rms)
    standardized = values / period_rms[None, :]
    return CommonData(
        frame=frame,
        period_columns=period_columns,
        periods=PERIODS.copy(),
        values=values,
        standardized_values=standardized,
        covariance=covariance,
        correlation=correlation,
        period_rms=period_rms,
    )


def haversine_km(lat1, lon1, lat2, lon2):
    """The same spherical distance implementation used by the PCA notebook."""
    earth_radius_km = 6371.0
    dlat = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(np.asarray(lat1)))
        * np.cos(np.radians(np.asarray(lat2)))
        * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _event_distance_matrix(frame: pd.DataFrame) -> np.ndarray:
    lat = frame["station_latitude"].to_numpy(dtype=float)
    lon = frame["station_longitude"].to_numpy(dtype=float)
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def compute_ring_scatters(
    data: CommonData,
    rings_km=RINGS_KM,
    min_stations_per_event: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute event-block ring scatters without a dense all-event kernel."""
    q = len(data.periods)
    sums = np.zeros((len(rings_km), q, q), dtype=float)
    counts = np.zeros(len(rings_km), dtype=np.int64)
    cursor = 0
    for _, event in data.frame.groupby("eqid", sort=False):
        n = len(event)
        event_standardized = data.standardized_values[cursor : cursor + n]
        cursor += n
        if n < min_stations_per_event:
            continue
        distance = _event_distance_matrix(event)
        for index, (inner, outer) in enumerate(rings_km):
            # SpatialBSS ring convention: inner < d <= outer.  Thus R(0,10)
            # excludes self-pairs and all rings contain distinct locations.
            kernel = ((distance > inner) & (distance <= outer)).astype(float)
            counts[index] += int(kernel.sum())
            sums[index] += event_standardized.T @ kernel @ event_standardized
    if cursor != len(data.frame):
        raise RuntimeError("Event groups did not cover the common-record data")
    return sums / len(data.frame), counts


def whiten_scatter_matrices(scatter_zero: np.ndarray, ring_scatters: np.ndarray):
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (scatter_zero + scatter_zero.T))
    if eigenvalues.min() <= 0:
        raise ValueError("The zero-lag scatter is not positive definite")
    whitening = (eigenvectors / np.sqrt(eigenvalues)[None, :]).T
    whitened = np.asarray(
        [whitening @ matrix @ whitening.T for matrix in ring_scatters], dtype=float
    )
    whitened = 0.5 * (whitened + np.swapaxes(whitened, 1, 2))
    return whitening, whitened, eigenvalues


def _skew_matrix(parameters: np.ndarray, q: int) -> np.ndarray:
    matrix = np.zeros((q, q), dtype=float)
    indices = np.tril_indices(q, k=-1)
    matrix[indices] = parameters
    matrix[(indices[1], indices[0])] = -parameters
    return matrix


def _offdiagonal_objective(rotation: np.ndarray, matrices: np.ndarray) -> float:
    total = 0.0
    for matrix in matrices:
        transformed = rotation @ matrix @ rotation.T
        off = transformed - np.diag(np.diag(transformed))
        total += float(np.sum(off * off))
    return total


def joint_diagonalize_symmetric(matrices: np.ndarray) -> tuple[np.ndarray, dict]:
    """Approximate joint diagonalization over O(q) using Givens-equivalent rotations.

    A real skew-symmetric matrix has q(q-1)/2 free plane-rotation parameters;
    its exponential is orthogonal.  Optimizing these parameters is equivalent
    to optimizing a product of Givens rotations but avoids order dependence.
    """
    matrices = np.asarray(matrices, dtype=float)
    q = matrices.shape[1]
    free = q * (q - 1) // 2
    identity = np.eye(q)
    bases: list[tuple[str, np.ndarray]] = [("identity", identity)]
    for index, matrix in enumerate(matrices):
        _, eigenvectors = np.linalg.eigh(matrix)
        bases.append((f"eigen_ring_{index + 1}", eigenvectors.T))
    weighted = sum((index + 1.0) * matrix for index, matrix in enumerate(matrices))
    _, weighted_vectors = np.linalg.eigh(weighted)
    bases.append(("eigen_weighted_sum", weighted_vectors.T))

    for index, (name, base) in enumerate(bases):
        if np.linalg.det(base) < 0:
            base = base.copy()
            base[0] *= -1.0
            bases[index] = (name, base)

    initial = [(name, base, _offdiagonal_objective(base, matrices)) for name, base in bases]
    selected = sorted(initial, key=lambda item: item[2])[:4]
    rows: list[dict] = []
    results: list[tuple[object, np.ndarray, str]] = []

    for name, base, initial_objective in selected:
        def objective(parameters):
            rotation = expm(_skew_matrix(parameters, q)) @ base
            return _offdiagonal_objective(rotation, matrices)

        result = minimize(
            objective,
            np.zeros(free),
            method="L-BFGS-B",
            bounds=[(-np.pi, np.pi)] * free,
            options={"maxiter": 1500, "maxls": 50, "ftol": 1.0e-14, "gtol": 1.0e-9},
        )
        rotation = expm(_skew_matrix(result.x, q)) @ base
        results.append((result, rotation, name))
        rows.append(
            {
                "start": name,
                "initial_objective": initial_objective,
                "final_objective": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
            }
        )

    best_result, best_rotation, best_name = min(results, key=lambda item: float(item[0].fun))
    denominator = float(sum(np.sum(np.diag(matrix) ** 2) for matrix in matrices))
    diagnostics = {
        "selected_start": best_name,
        "success": bool(best_result.success),
        "message": str(best_result.message),
        "free_rotation_parameters": free,
        "initial_candidates": [
            {"name": name, "objective": float(value)} for name, _, value in initial
        ],
        "optimized_starts": rows,
        "final_offdiagonal_objective_before_ordering": float(best_result.fun),
        "offdiagonal_to_initial_diagonal_energy_ratio": float(best_result.fun / max(denominator, 1.0e-30)),
        "orthogonality_error": float(np.linalg.norm(best_rotation @ best_rotation.T - np.eye(q))),
    }
    return best_rotation, diagnostics


def empirical_event_variograms(
    frame: pd.DataFrame,
    values: np.ndarray,
    bin_size_km: float = 2.0,
    max_distance_km: float = 100.0,
    min_stations_per_event: int = 5,
) -> EmpiricalVariograms:
    """Match PCA: ordered within-event pairs, nearest 2-km bins, pair-count pooling."""
    q = values.shape[1]
    n_bins = int(round(max_distance_km / bin_size_km))
    sums = np.zeros((q, q, n_bins), dtype=float)
    counts = np.zeros(n_bins, dtype=np.int64)
    cursor = 0
    for _, event in frame.groupby("eqid", sort=False):
        n = len(event)
        event_values = values[cursor : cursor + n]
        cursor += n
        if n < min_stations_per_event:
            continue
        distance = _event_distance_matrix(event)
        distance_bin = np.round(distance / bin_size_km).astype(np.int64)
        for bin_number in range(1, n_bins + 1):
            a, b = np.where(distance_bin == bin_number)
            pair_count = len(a)
            if pair_count == 0:
                continue
            delta = event_values[a] - event_values[b]
            sums[:, :, bin_number - 1] += 0.5 * delta.T @ delta
            counts[bin_number - 1] += pair_count
    if cursor != len(frame):
        raise RuntimeError("Event groups did not cover values")
    gamma = np.zeros_like(sums)
    np.divide(sums, counts[None, None, :], out=gamma, where=counts[None, None, :] > 0)
    h = np.arange(1, n_bins + 1, dtype=float) * bin_size_km
    return EmpiricalVariograms(h, gamma, counts)


def _fit_powered_exponential(gamma, h, sill, initial=(20.0, 0.4)):
    def objective(parameters):
        length_scale, exponent = parameters
        prediction = sill * (1.0 - np.exp(-((h / length_scale) ** exponent)))
        return float(np.sum((gamma - prediction) ** 2 / h))

    starts = (initial, (8.0, 0.2), (40.0, 0.7), (100.0, 1.2))
    results = [
        minimize(
            objective,
            start,
            bounds=[(1.0, 200.0), (0.05, 1.5)],
            method="L-BFGS-B",
            options={"maxiter": 1500, "ftol": 1.0e-14, "gtol": 1.0e-9},
        )
        for start in starts
    ]
    return min(results, key=lambda result: float(result.fun))


def fit_latent_variograms(
    data: CommonData,
    unmixing_standardized: np.ndarray,
    h_fine: np.ndarray,
    tail_bins: int = 10,
) -> LatentFit:
    scores = data.standardized_values @ unmixing_standardized.T
    empirical = empirical_event_variograms(data.frame, scores)
    q = scores.shape[1]
    sills = np.asarray([empirical.gamma[k, k, -tail_bins:].mean() for k in range(q)])
    length_scales = np.empty(q)
    exponents = np.empty(q)
    wss = np.empty(q)
    rows: list[dict] = []

    for k in range(q):
        if sills[k] <= 0.01:
            length_scales[k], exponents[k], wss[k] = 1.0, 1.0, np.nan
            rows.append(
                {
                    "latent_component": k + 1,
                    "tail10_sill": sills[k],
                    "length_scale_km": 1.0,
                    "powered_exponent": 1.0,
                    "wss": np.nan,
                    "success": False,
                    "message": "Tail sill <= 0.01; notebook fallback used",
                    "iterations": 0,
                    "function_evaluations": 0,
                }
            )
            continue
        result = _fit_powered_exponential(
            empirical.gamma[k, k], empirical.h_bins, sills[k]
        )
        length_scales[k], exponents[k] = result.x
        wss[k] = float(result.fun)
        rows.append(
            {
                "latent_component": k + 1,
                "tail10_sill": sills[k],
                "length_scale_km": length_scales[k],
                "powered_exponent": exponents[k],
                "wss": wss[k],
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
            }
        )

    rho_fine = np.exp(
        -((np.maximum(h_fine[:, None], 1.0e-12) / length_scales[None, :]) ** exponents[None, :])
    )
    rho_fine[0] = 1.0
    return LatentFit(
        scores=scores,
        empirical=empirical,
        sills=sills,
        length_scales=length_scales,
        exponents=exponents,
        wss=wss,
        rho_fine=rho_fine,
        optimizer_rows=rows,
    )


def _reconstruct_covariance(mixing_raw: np.ndarray, rho_latent: np.ndarray) -> np.ndarray:
    curves = np.empty((len(rho_latent), len(mixing_raw), len(mixing_raw)), dtype=float)
    for index, rho in enumerate(rho_latent):
        curves[index] = (mixing_raw * rho[None, :]) @ mixing_raw.T
    return np.moveaxis(curves, 0, 2)


def fit_sbss(
    data: CommonData,
    rings_km=RINGS_KM,
    h_fine: np.ndarray | None = None,
    tail_bins: int = 10,
) -> SBSSFit:
    if h_fine is None:
        h_fine = np.linspace(0.0, 120.0, 400)

    ring_scatters, ring_pair_counts = compute_ring_scatters(data, rings_km)
    whitening, white_ring_scatters, eigenvalues = whiten_scatter_matrices(
        data.correlation, ring_scatters
    )
    rotation, joint_diagnostics = joint_diagonalize_symmetric(white_ring_scatters)

    # Order components exactly like SpatialBSS/PCA reporting: decreasing sum
    # of squared pseudo-eigenvalues across all local scatters.
    preliminary = np.asarray(
        [rotation @ matrix @ rotation.T for matrix in white_ring_scatters]
    )
    profiles = np.diagonal(preliminary, axis1=1, axis2=2)
    scores = np.sum(profiles * profiles, axis=0)
    order = np.argsort(scores)[::-1]
    rotation = rotation[order]

    unmixing = rotation @ whitening
    mixing_standardized = np.linalg.inv(unmixing)
    for k in range(len(data.periods)):
        anchor = int(np.argmax(np.abs(mixing_standardized[:, k])))
        if mixing_standardized[anchor, k] < 0:
            unmixing[k] *= -1.0
            rotation[k] *= -1.0
            mixing_standardized[:, k] *= -1.0

    diagonalized = np.asarray(
        [rotation @ matrix @ rotation.T for matrix in white_ring_scatters]
    )
    profiles = np.diagonal(diagonalized, axis1=1, axis2=2)
    spatial_scores = np.sum(profiles * profiles, axis=0)
    mixing_raw = data.period_rms[:, None] * mixing_standardized

    latent = fit_latent_variograms(data, unmixing, np.asarray(h_fine), tail_bins=tail_bins)
    covariance_fine = _reconstruct_covariance(mixing_raw, latent.rho_fine)
    c0 = covariance_fine[:, :, 0]
    sd0 = np.sqrt(np.clip(np.diag(c0), 1.0e-15, None))
    correlation_fine = covariance_fine / np.outer(sd0, sd0)[:, :, None]
    correlation_fine[:, :, 0] = data.correlation

    empirical_observed = empirical_event_variograms(data.frame, data.values)
    auto_sills = np.asarray(
        [empirical_observed.gamma[i, i, -tail_bins:].mean() for i in range(len(data.periods))]
    )
    cross_sills = data.correlation * np.sqrt(np.outer(auto_sills, auto_sills))

    # Match the notebook's PCA cross-variogram conversion rather than using
    # raw sample covariance amplitudes.  Cross decay is made relative to its
    # fitted lag-zero correlation and amplitude uses rho0*sqrt(auto sills).
    semivariogram = np.empty_like(correlation_fine)
    for i in range(len(data.periods)):
        semivariogram[i, i] = auto_sills[i] * (1.0 - correlation_fine[i, i])
        for j in range(i):
            rho0 = data.correlation[i, j]
            if abs(rho0) < 1.0e-12:
                relative = np.zeros_like(h_fine)
                relative[0] = 1.0
            else:
                relative = correlation_fine[i, j] / rho0
            curve = cross_sills[i, j] * (1.0 - relative)
            semivariogram[i, j] = semivariogram[j, i] = curve
    semivariogram[:, :, 0] = 0.0

    fitted_at_bins = np.empty_like(empirical_observed.gamma)
    for i in range(len(data.periods)):
        for j in range(len(data.periods)):
            fitted_at_bins[i, j] = np.interp(
                empirical_observed.h_bins, h_fine, semivariogram[i, j]
            )
    residual = empirical_observed.gamma - fitted_at_bins
    wss_45 = float(np.sum((residual * residual) / empirical_observed.h_bins[None, None, :]) / 2.0)
    # Diagonal was halved by the symmetric /2 above; add half of each diagonal
    # again so the result is exactly the lower-triangle 45-curve sum.
    wss_45 += 0.5 * float(
        sum(np.sum(residual[i, i] ** 2 / empirical_observed.h_bins) for i in range(len(data.periods)))
    )

    offdiag_final = float(
        sum(
            np.sum((matrix - np.diag(np.diag(matrix))) ** 2)
            for matrix in diagonalized
        )
    )
    profile_separation = min(
        max(abs(profiles[:, i] - profiles[:, j]))
        for i in range(len(data.periods))
        for j in range(i)
    )
    joint_diagnostics.update(
        {
            "scatter_zero_eigenvalues": eigenvalues.tolist(),
            "final_offdiagonal_objective": offdiag_final,
            "whitening_error": float(
                np.linalg.norm(unmixing @ data.correlation @ unmixing.T - np.eye(len(data.periods)))
            ),
            "reconstruction_error_correlation": float(
                np.linalg.norm(mixing_standardized @ mixing_standardized.T - data.correlation)
            ),
            "minimum_profile_separation": float(profile_separation),
        }
    )

    return SBSSFit(
        periods=data.periods.copy(),
        rings_km=tuple(rings_km),
        scatter_zero=data.correlation.copy(),
        ring_scatters=ring_scatters,
        ring_ordered_pair_counts=ring_pair_counts,
        whitening=whitening,
        whitened_ring_scatters=white_ring_scatters,
        joint_rotation=rotation,
        unmixing_standardized=unmixing,
        mixing_standardized=mixing_standardized,
        mixing_raw=mixing_raw,
        diagonalized_ring_scatters=diagonalized,
        spatial_profiles=profiles,
        spatial_scores=spatial_scores,
        latent=latent,
        h_fine=np.asarray(h_fine),
        covariance_fine=covariance_fine,
        correlation_fine=correlation_fine,
        semivariogram_fine=semivariogram,
        empirical_observed=empirical_observed,
        empirical_auto_sills=auto_sills,
        cross_sills=cross_sills,
        wss_45=wss_45,
        joint_diagnostics=joint_diagnostics,
    )


def semivariogram_long_frame(fit: SBSSFit) -> pd.DataFrame:
    rows: list[dict] = []
    empirical = fit.empirical_observed
    for i in range(len(fit.periods)):
        for j in range(i + 1):
            fitted = np.interp(empirical.h_bins, fit.h_fine, fit.semivariogram_fine[i, j])
            for b, h in enumerate(empirical.h_bins):
                residual = empirical.gamma[i, j, b] - fitted[b]
                rows.append(
                    {
                        "period_i_s": fit.periods[i],
                        "period_j_s": fit.periods[j],
                        "curve_type": "auto" if i == j else "cross",
                        "distance_km": h,
                        "empirical_semivariogram": empirical.gamma[i, j, b],
                        "fitted_semivariogram": fitted[b],
                        "residual": residual,
                        "weight_1_over_h": 1.0 / h,
                        "weighted_squared_residual": residual * residual / h,
                        "ordered_station_pair_count": int(empirical.counts[b]),
                    }
                )
    return pd.DataFrame(rows)


def marginal_correlation_frame(fit: SBSSFit) -> pd.DataFrame:
    table = pd.DataFrame({"distance_km": fit.h_fine})
    for i, period in enumerate(fit.periods):
        table[f"rho_T{period:g}s"] = fit.correlation_fine[i, i]
    return table


def latent_correlation_frame(fit: SBSSFit) -> pd.DataFrame:
    table = pd.DataFrame({"distance_km": fit.h_fine})
    for k in range(len(fit.periods)):
        table[f"rho_latent_{k + 1}"] = fit.latent.rho_fine[:, k]
    return table
