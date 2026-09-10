"""Two-stage isotropic flexible multivariate Matern WLS fitting.

Stage 1 fits marginal alpha_ii and nu_ii from the nine auto-semivariograms
and fixes sigma_ii to the mean of the last ten supported bins.  Stage 2
keeps all marginal quantities fixed and jointly fits Delta_A, Delta_B and
the full R_A, R_B and R_V correlation matrices to the 36 cross curves.

The parameterization follows Theorem 1 of Apanasovich, Genton & Sun
(2012, JASA).  R_A and R_B have nonnegative entries and all three R
matrices remain positive definite by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln, kv


PERIOD_SPECS = (
    ("T001", 0.01),
    ("T003", 0.03),
    ("T006", 0.06),
    ("T010", 0.10),
    ("T030", 0.30),
    ("T060", 0.60),
    ("T100", 1.00),
    ("T300", 3.00),
    ("T600", 6.00),
)

REQUIRED_COLUMNS = (
    "recid",
    "eqid",
    "station_latitude",
    "station_longitude",
    "scaled_deltaW",
)


@dataclass
class EmpiricalVariograms:
    periods: np.ndarray
    h_bins: np.ndarray
    gamma: np.ndarray
    counts: np.ndarray
    record_counts: np.ndarray
    event_counts: np.ndarray
    rho_empirical: np.ndarray
    input_summary: pd.DataFrame


@dataclass
class MarginalFit:
    sigma_diag: np.ndarray
    alpha_diag: np.ndarray
    nu_diag: np.ndarray
    wss: float
    optimizer_rows: list[dict]


@dataclass
class FlexibleMaternFit:
    periods: np.ndarray
    h_bins: np.ndarray
    sigma: np.ndarray
    rho: np.ndarray
    rho_empirical: np.ndarray
    alpha: np.ndarray
    nu: np.ndarray
    delta_a: float
    delta_b: float
    r_a: np.ndarray
    r_b: np.ndarray
    r_v: np.ndarray
    v_matrix: np.ndarray
    marginal: MarginalFit
    joint_optimizer: dict
    wss_marginal: float
    wss_cross: float
    wss_total: float


def load_period_datasets(input_dir: str | Path) -> tuple[np.ndarray, list[pd.DataFrame], pd.DataFrame]:
    """Load all nine single-period files without a complete-case filter."""
    input_dir = Path(input_dir)
    periods: list[float] = []
    frames: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for code, period in PERIOD_SPECS:
        path = input_dir / f"data_ngawest2_{code}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        source = pd.read_csv(path)
        missing = [column for column in REQUIRED_COLUMNS if column not in source.columns]
        if missing:
            raise KeyError(f"{path.name} is missing required columns: {missing}")

        frame = source.loc[:, REQUIRED_COLUMNS].dropna().copy()
        duplicates = int(frame["recid"].duplicated().sum())
        if duplicates:
            raise ValueError(f"{path.name} contains {duplicates} duplicate recid values")
        frame["eqid"] = frame["eqid"].astype(str)
        event_sizes = frame.groupby("eqid", sort=False).size()

        periods.append(period)
        frames.append(frame)
        summaries.append(
            {
                "period_s": period,
                "file": path.name,
                "source_rows": int(len(source)),
                "usable_rows": int(len(frame)),
                "unique_recids": int(frame["recid"].nunique()),
                "events": int(frame["eqid"].nunique()),
                "events_with_at_least_5_records": int((event_sizes >= 5).sum()),
                "min_records_per_event": int(event_sizes.min()),
                "median_records_per_event": float(event_sizes.median()),
                "max_records_per_event": int(event_sizes.max()),
            }
        )

    return np.asarray(periods, dtype=float), frames, pd.DataFrame(summaries)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres; no anisotropy is used."""
    earth_radius_km = 6371.0
    dlat = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    lat1_rad = np.radians(np.asarray(lat1))
    lat2_rad = np.radians(np.asarray(lat2))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def pairwise_common_frame(frame_i: pd.DataFrame, frame_j: pd.DataFrame, same_period: bool) -> pd.DataFrame:
    """Use every record common to the requested period pair."""
    left = frame_i.rename(columns={"scaled_deltaW": "eps_i"})
    if same_period:
        out = left.copy()
        out["eps_j"] = out["eps_i"]
        return out

    right = frame_j.loc[:, ["recid", "eqid", "scaled_deltaW"]].rename(
        columns={"eqid": "eqid_j", "scaled_deltaW": "eps_j"}
    )
    out = left.merge(right, on="recid", how="inner", validate="one_to_one")
    if bool((out["eqid"] != out["eqid_j"]).any()):
        raise ValueError("A recid was assigned to different events in two period files")
    return out.drop(columns="eqid_j")


def _event_binned_cross_semivariogram(
    frame: pd.DataFrame,
    bin_size_km: float,
    max_distance_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one event's unordered station-pair cross-semivariogram."""
    n_bins = int(round(max_distance_km / bin_size_km))
    n = len(frame)
    if n < 2:
        return np.zeros(n_bins), np.zeros(n_bins, dtype=np.int64)

    index_a, index_b = np.triu_indices(n, k=1)
    lat = frame["station_latitude"].to_numpy(dtype=float)
    lon = frame["station_longitude"].to_numpy(dtype=float)
    distance = haversine_km(lat[index_a], lon[index_a], lat[index_b], lon[index_b])
    bin_index = np.rint(distance / bin_size_km).astype(int)
    keep = (bin_index >= 1) & (bin_index <= n_bins)
    if not bool(keep.any()):
        return np.zeros(n_bins), np.zeros(n_bins, dtype=np.int64)

    eps_i = frame["eps_i"].to_numpy(dtype=float)
    eps_j = frame["eps_j"].to_numpy(dtype=float)
    product = 0.5 * (eps_i[index_a] - eps_i[index_b]) * (eps_j[index_a] - eps_j[index_b])
    bins0 = bin_index[keep] - 1
    sums = np.bincount(bins0, weights=product[keep], minlength=n_bins).astype(float)
    counts = np.bincount(bins0, minlength=n_bins).astype(np.int64)
    return sums, counts


def pool_pairwise_event_semivariograms(
    periods: np.ndarray,
    frames: list[pd.DataFrame],
    input_summary: pd.DataFrame,
    bin_size_km: float = 2.0,
    max_distance_km: float = 100.0,
    min_stations_per_event: int = 5,
) -> EmpiricalVariograms:
    """Compute within event first, then pool station-pair sums and counts."""
    p = len(periods)
    n_bins = int(round(max_distance_km / bin_size_km))
    h_bins = np.arange(1, n_bins + 1, dtype=float) * bin_size_km
    gamma = np.full((p, p, n_bins), np.nan, dtype=float)
    counts = np.zeros((p, p, n_bins), dtype=np.int64)
    record_counts = np.zeros((p, p), dtype=np.int64)
    event_counts = np.zeros((p, p), dtype=np.int64)
    rho_empirical = np.eye(p, dtype=float)

    for i in range(p):
        for j in range(i + 1):
            pair = pairwise_common_frame(frames[i], frames[j], same_period=(i == j))
            record_counts[i, j] = record_counts[j, i] = len(pair)
            if i != j:
                rho_ij = float(np.corrcoef(pair["eps_i"], pair["eps_j"])[0, 1])
                rho_empirical[i, j] = rho_empirical[j, i] = rho_ij

            sum_all = np.zeros(n_bins, dtype=float)
            count_all = np.zeros(n_bins, dtype=np.int64)
            eligible_events = 0
            for _, event in pair.groupby("eqid", sort=False):
                if len(event) < min_stations_per_event:
                    continue
                event_sum, event_count = _event_binned_cross_semivariogram(
                    event, bin_size_km=bin_size_km, max_distance_km=max_distance_km
                )
                sum_all += event_sum
                count_all += event_count
                eligible_events += 1

            curve = np.full(n_bins, np.nan, dtype=float)
            np.divide(sum_all, count_all, out=curve, where=count_all > 0)
            gamma[i, j] = gamma[j, i] = curve
            counts[i, j] = counts[j, i] = count_all
            event_counts[i, j] = event_counts[j, i] = eligible_events

    return EmpiricalVariograms(
        periods=np.asarray(periods, dtype=float),
        h_bins=h_bins,
        gamma=gamma,
        counts=counts,
        record_counts=record_counts,
        event_counts=event_counts,
        rho_empirical=rho_empirical,
        input_summary=input_summary.copy(),
    )


def tail_auto_sills(empirical: EmpiricalVariograms, tail_bins: int = 10) -> np.ndarray:
    """Fix sigma_ii to the mean of the last ten supported auto bins."""
    sigma_diag = np.zeros(len(empirical.periods), dtype=float)
    for i, period in enumerate(empirical.periods):
        valid = (empirical.counts[i, i] > 0) & np.isfinite(empirical.gamma[i, i])
        values = empirical.gamma[i, i, valid]
        if len(values) < tail_bins:
            raise ValueError(f"Period {period:g}s has fewer than {tail_bins} supported bins")
        sigma_diag[i] = float(np.mean(values[-tail_bins:]))
        if sigma_diag[i] <= 0:
            raise ValueError(f"Nonpositive tail sill for period {period:g}s")
    return sigma_diag


def matern_correlation(h, nu: float, alpha: float) -> np.ndarray:
    """AGS rate-parameterized Matern M(h | nu, alpha)."""
    h = np.asarray(h, dtype=float)
    x = float(alpha) * np.abs(h)
    out = np.ones_like(x, dtype=float)
    positive = x > 0
    xp = x[positive]
    if xp.size:
        bessel = kv(float(nu), xp)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            log_value = (1.0 - float(nu)) * np.log(2.0) - gammaln(float(nu))
            log_value += float(nu) * np.log(xp) + np.log(bessel)
            values = np.exp(log_value)
        out[positive] = np.clip(np.where(np.isfinite(values), values, 0.0), 0.0, 1.0)
    return out


def theoretical_semivariogram(h, sigma: float, nu: float, alpha: float) -> np.ndarray:
    return float(sigma) * (1.0 - matern_correlation(h, nu=float(nu), alpha=float(alpha)))


def _curve_wss(gamma, counts, h, sigma, nu, alpha) -> float:
    valid = (np.asarray(counts) > 0) & np.isfinite(gamma)
    if not bool(valid.any()):
        return 0.0
    prediction = theoretical_semivariogram(h[valid], sigma, nu, alpha)
    return float(np.sum((1.0 / h[valid]) * (gamma[valid] - prediction) ** 2))


def _fit_marginal_curve(gamma, counts, h, sigma) -> tuple[float, float, float, bool, str, int, int]:
    """Fit one marginal alpha_ii and nu_ii with sigma_ii fixed."""
    bounds = [(np.log(1.0e-4), np.log(2.0)), (np.log(0.05), np.log(5.0))]
    starts = ((0.01, 0.2), (0.03, 0.5), (0.08, 1.0), (0.20, 2.0))

    def objective(log_parameters):
        alpha, nu = np.exp(log_parameters)
        return _curve_wss(gamma, counts, h, sigma, nu, alpha)

    results = [
        minimize(objective, np.log(start), method="L-BFGS-B", bounds=bounds)
        for start in starts
    ]
    best = min(results, key=lambda result: float(result.fun))
    alpha, nu = np.exp(best.x)
    return (
        float(alpha),
        float(nu),
        float(best.fun),
        bool(best.success),
        str(best.message),
        int(best.nit),
        int(best.nfev),
    )


def fit_marginal_models(empirical: EmpiricalVariograms, tail_bins: int = 10) -> MarginalFit:
    """Stage 1: fit only the nine marginal models."""
    sigma_diag = tail_auto_sills(empirical, tail_bins=tail_bins)
    alpha_diag = np.zeros(len(empirical.periods), dtype=float)
    nu_diag = np.zeros(len(empirical.periods), dtype=float)
    rows: list[dict] = []

    for i, period in enumerate(empirical.periods):
        alpha, nu, wss, success, message, iterations, evaluations = _fit_marginal_curve(
            empirical.gamma[i, i], empirical.counts[i, i], empirical.h_bins, sigma_diag[i]
        )
        alpha_diag[i] = alpha
        nu_diag[i] = nu
        rows.append(
            {
                "i": i,
                "period_s": period,
                "sigma_ii_tail_mean": sigma_diag[i],
                "alpha_ii_per_km": alpha,
                "nu_ii": nu,
                "marginal_wss": wss,
                "success": success,
                "message": message,
                "iterations": iterations,
                "function_evaluations": evaluations,
            }
        )
    return MarginalFit(
        sigma_diag=sigma_diag,
        alpha_diag=alpha_diag,
        nu_diag=nu_diag,
        wss=float(sum(row["marginal_wss"] for row in rows)),
        optimizer_rows=rows,
    )


def _nearest_correlation(matrix: np.ndarray, iterations: int = 200) -> np.ndarray:
    """Return a numerically positive-definite correlation matrix."""
    y = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    np.fill_diagonal(y, 1.0)
    correction = np.zeros_like(y)
    for _ in range(iterations):
        residual = y - correction
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (residual + residual.T))
        psd = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        correction = psd - residual
        diagonal = np.sqrt(np.clip(np.diag(psd), 1.0e-15, None))
        updated = psd / np.outer(diagonal, diagonal)
        np.fill_diagonal(updated, 1.0)
        if np.max(np.abs(updated - y)) < 1.0e-12:
            y = updated
            break
        y = updated
    jitter = 1.0e-6
    y = (1.0 - jitter) * y + jitter * np.eye(len(y))
    return 0.5 * (y + y.T)


def _softplus(values):
    values = np.asarray(values, dtype=float)
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def _inverse_softplus(values):
    values = np.clip(np.asarray(values, dtype=float), 1.0e-12, None)
    return values + np.log(-np.expm1(-values))


def _factor_state(parameters: np.ndarray, p: int, positive_factor: bool) -> dict:
    """Correlation and derivative state for a normalized lower factor."""
    parameters = np.asarray(parameters, dtype=float)
    expected = p * (p - 1) // 2
    if parameters.size != expected:
        raise ValueError(f"Expected {expected} factor parameters, got {parameters.size}")
    lower = np.eye(p, dtype=float)
    values = _softplus(parameters) if positive_factor else parameters
    cursor = 0
    for i in range(1, p):
        lower[i, :i] = values[cursor : cursor + i]
        cursor += i
    gram = lower @ lower.T
    row_norm = np.sqrt(np.diag(gram))
    unit_rows = lower / row_norm[:, None]
    correlation = unit_rows @ unit_rows.T
    np.fill_diagonal(correlation, 1.0)
    return {
        "correlation": 0.5 * (correlation + correlation.T),
        "lower": lower,
        "row_norm": row_norm,
        "unit_rows": unit_rows,
        "parameter_slope": expit(parameters) if positive_factor else np.ones_like(parameters),
    }


def _factor_correlation(parameters: np.ndarray, p: int, positive_factor: bool) -> np.ndarray:
    """Correlation from a lower factor with unit diagonal and free subdiagonal."""
    return _factor_state(parameters, p, positive_factor)["correlation"]


def _factor_parameter_gradient(state: dict, correlation_gradient: np.ndarray) -> np.ndarray:
    """Back-propagate off-diagonal dL/dR through R=normalize(L)normalize(L)'."""
    unit_rows = state["unit_rows"]
    gradient_unit = correlation_gradient @ unit_rows
    radial = np.sum(unit_rows * gradient_unit, axis=1)
    gradient_lower = (
        gradient_unit - unit_rows * radial[:, None]
    ) / state["row_norm"][:, None]
    values = []
    for i in range(1, len(unit_rows)):
        values.extend(gradient_lower[i, :i].tolist())
    return np.asarray(values, dtype=float) * state["parameter_slope"]


def _correlation_to_factor_parameters(correlation: np.ndarray, positive_factor: bool) -> np.ndarray:
    """Inverse initialization map for the factor correlation parameterization."""
    correlation = _nearest_correlation(correlation)
    lower = np.linalg.cholesky(correlation)
    values = []
    for i in range(1, len(correlation)):
        scaled_row = lower[i, :i] / lower[i, i]
        if positive_factor:
            scaled_row = np.clip(scaled_row, 1.0e-10, None)
            scaled_row = _inverse_softplus(scaled_row)
        values.extend(scaled_row.tolist())
    return np.asarray(values, dtype=float)


def _equicorrelation(p: int, correlation: float) -> np.ndarray:
    matrix = np.full((p, p), float(correlation), dtype=float)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _ags_v_multiplier(nu, alpha, delta_a: float, spatial_dimension: int) -> np.ndarray:
    diag_nu = np.diag(nu)
    average_nu = 0.5 * (diag_nu[:, None] + diag_nu[None, :])
    log_multiplier = (
        (2.0 * delta_a + diag_nu[:, None] + diag_nu[None, :]) * np.log(alpha)
        + np.vectorize(gammaln)(nu + spatial_dimension / 2.0)
        - np.vectorize(gammaln)(average_nu + spatial_dimension / 2.0)
        - np.vectorize(gammaln)(nu)
    )
    return np.exp(np.clip(log_multiplier, -700.0, 700.0))


def _unpack_joint_parameters(
    parameters: np.ndarray,
    marginal: MarginalFit,
    spatial_dimension: int,
) -> dict:
    p = len(marginal.sigma_diag)
    n_correlation = p * (p - 1) // 2
    parameters = np.asarray(parameters, dtype=float)
    delta_a = float(np.exp(parameters[0]))
    delta_b = float(np.exp(parameters[1]))
    cursor = 2
    r_a = _factor_correlation(parameters[cursor : cursor + n_correlation], p, positive_factor=True)
    cursor += n_correlation
    r_b = _factor_correlation(parameters[cursor : cursor + n_correlation], p, positive_factor=True)
    cursor += n_correlation
    r_v = _factor_correlation(parameters[cursor : cursor + n_correlation], p, positive_factor=False)

    average_nu = 0.5 * (marginal.nu_diag[:, None] + marginal.nu_diag[None, :])
    average_alpha_sq = 0.5 * (
        marginal.alpha_diag[:, None] ** 2 + marginal.alpha_diag[None, :] ** 2
    )
    nu = average_nu + delta_a * (1.0 - r_a)
    alpha = np.sqrt(np.clip(average_alpha_sq + delta_b * (1.0 - r_b), 1.0e-15, None))
    np.fill_diagonal(nu, marginal.nu_diag)
    np.fill_diagonal(alpha, marginal.alpha_diag)

    multiplier = _ags_v_multiplier(nu, alpha, delta_a, spatial_dimension)
    v_diag = marginal.sigma_diag * np.diag(multiplier)
    v_matrix = np.sqrt(np.outer(v_diag, v_diag)) * r_v
    sigma = np.divide(v_matrix, multiplier, out=np.zeros_like(v_matrix), where=multiplier > 0)
    sigma = 0.5 * (sigma + sigma.T)
    np.fill_diagonal(sigma, marginal.sigma_diag)
    rho = sigma / np.sqrt(np.outer(marginal.sigma_diag, marginal.sigma_diag))
    rho = 0.5 * (rho + rho.T)
    np.fill_diagonal(rho, 1.0)

    return {
        "delta_a": delta_a,
        "delta_b": delta_b,
        "r_a": r_a,
        "r_b": r_b,
        "r_v": r_v,
        "nu": nu,
        "alpha": alpha,
        "v_matrix": v_matrix,
        "sigma": sigma,
        "rho": rho,
    }


def _joint_objective_and_gradient(
    parameters: np.ndarray,
    empirical: EmpiricalVariograms,
    marginal: MarginalFit,
    spatial_dimension: int,
) -> tuple[float, np.ndarray]:
    """Cross WLS and analytic chain gradient for all 110 joint parameters."""
    p = len(marginal.sigma_diag)
    n_correlation = p * (p - 1) // 2
    parameters = np.asarray(parameters, dtype=float)
    delta_a = float(np.exp(parameters[0]))
    delta_b = float(np.exp(parameters[1]))
    cursor = 2
    parameters_a = parameters[cursor : cursor + n_correlation]
    cursor += n_correlation
    parameters_b = parameters[cursor : cursor + n_correlation]
    cursor += n_correlation
    parameters_v = parameters[cursor : cursor + n_correlation]
    state_a = _factor_state(parameters_a, p, positive_factor=True)
    state_b = _factor_state(parameters_b, p, positive_factor=True)
    state_v = _factor_state(parameters_v, p, positive_factor=False)
    r_a = state_a["correlation"]
    r_b = state_b["correlation"]
    r_v = state_v["correlation"]

    gradient_r_a = np.zeros((p, p), dtype=float)
    gradient_r_b = np.zeros((p, p), dtype=float)
    gradient_r_v = np.zeros((p, p), dtype=float)
    gradient_delta_a = 0.0
    gradient_delta_b = 0.0
    objective = 0.0

    for i in range(p):
        for j in range(i):
            nu_average = 0.5 * (marginal.nu_diag[i] + marginal.nu_diag[j])
            alpha_sq_average = 0.5 * (
                marginal.alpha_diag[i] ** 2 + marginal.alpha_diag[j] ** 2
            )
            nu = nu_average + delta_a * (1.0 - r_a[i, j])
            alpha = np.sqrt(
                max(alpha_sq_average + delta_b * (1.0 - r_b[i, j]), 1.0e-15)
            )

            log_multiplier = (
                (2.0 * delta_a + marginal.nu_diag[i] + marginal.nu_diag[j]) * np.log(alpha)
                + gammaln(nu + spatial_dimension / 2.0)
                - gammaln(nu_average + spatial_dimension / 2.0)
                - gammaln(nu)
            )
            log_v_scale = 0.5 * (
                np.log(marginal.sigma_diag[i])
                + (2.0 * delta_a + 2.0 * marginal.nu_diag[i]) * np.log(marginal.alpha_diag[i])
                - gammaln(marginal.nu_diag[i])
                + np.log(marginal.sigma_diag[j])
                + (2.0 * delta_a + 2.0 * marginal.nu_diag[j]) * np.log(marginal.alpha_diag[j])
                - gammaln(marginal.nu_diag[j])
            )
            sigma_per_r_v = float(np.exp(np.clip(log_v_scale - log_multiplier, -700.0, 700.0)))
            sigma = sigma_per_r_v * r_v[i, j]

            valid = (empirical.counts[i, j] > 0) & np.isfinite(empirical.gamma[i, j])
            h = empirical.h_bins[valid]
            observed = empirical.gamma[i, j, valid]
            weights = 1.0 / h
            matern = matern_correlation(h, nu, alpha)
            basis = 1.0 - matern
            prediction = sigma * basis
            residual = observed - prediction
            objective += float(np.sum(weights * residual**2))
            gradient_prediction = -2.0 * weights * residual
            gradient_sigma = float(np.sum(gradient_prediction * basis))

            # Local numerical derivatives of Matern M with respect to alpha
            # and nu avoid a global 110-dimensional finite-difference pass.
            step_alpha = max(1.0e-6, 1.0e-4 * alpha)
            alpha_minus = max(alpha - step_alpha, 1.0e-8)
            derivative_m_alpha = (
                matern_correlation(h, nu, alpha + step_alpha)
                - matern_correlation(h, nu, alpha_minus)
            ) / (alpha + step_alpha - alpha_minus)
            step_nu = max(1.0e-6, 1.0e-4 * nu)
            nu_minus = max(nu - step_nu, 1.0e-8)
            derivative_m_nu = (
                matern_correlation(h, nu + step_nu, alpha)
                - matern_correlation(h, nu_minus, alpha)
            ) / (nu + step_nu - nu_minus)
            gradient_alpha_direct = float(
                np.sum(gradient_prediction * (-sigma * derivative_m_alpha))
            )
            gradient_nu_direct = float(
                np.sum(gradient_prediction * (-sigma * derivative_m_nu))
            )

            derivative_log_multiplier_nu = float(
                digamma(nu + spatial_dimension / 2.0) - digamma(nu)
            )
            derivative_log_multiplier_alpha = (
                2.0 * delta_a + marginal.nu_diag[i] + marginal.nu_diag[j]
            ) / alpha
            gradient_nu_total = (
                gradient_nu_direct - gradient_sigma * sigma * derivative_log_multiplier_nu
            )
            gradient_alpha_total = (
                gradient_alpha_direct - gradient_sigma * sigma * derivative_log_multiplier_alpha
            )

            gradient_delta_a += (
                gradient_sigma
                * sigma
                * (
                    np.log(marginal.alpha_diag[i])
                    + np.log(marginal.alpha_diag[j])
                    - 2.0 * np.log(alpha)
                )
                + gradient_nu_total * (1.0 - r_a[i, j])
            )
            gradient_delta_b += gradient_alpha_total * (1.0 - r_b[i, j]) / (2.0 * alpha)
            gradient_a_ij = -gradient_nu_total * delta_a
            gradient_b_ij = -gradient_alpha_total * delta_b / (2.0 * alpha)
            gradient_v_ij = gradient_sigma * sigma_per_r_v
            gradient_r_a[i, j] = gradient_r_a[j, i] = gradient_a_ij
            gradient_r_b[i, j] = gradient_r_b[j, i] = gradient_b_ij
            gradient_r_v[i, j] = gradient_r_v[j, i] = gradient_v_ij

    gradient = np.concatenate(
        [
            [gradient_delta_a * delta_a, gradient_delta_b * delta_b],
            _factor_parameter_gradient(state_a, gradient_r_a),
            _factor_parameter_gradient(state_b, gradient_r_b),
            _factor_parameter_gradient(state_v, gradient_r_v),
        ]
    )
    return float(objective), gradient


def _cross_wss(empirical: EmpiricalVariograms, model: dict) -> float:
    total = 0.0
    p = len(empirical.periods)
    for i in range(p):
        for j in range(i):
            total += _curve_wss(
                empirical.gamma[i, j],
                empirical.counts[i, j],
                empirical.h_bins,
                model["sigma"][i, j],
                model["nu"][i, j],
                model["alpha"][i, j],
            )
    return float(total)


def _joint_start(
    empirical: EmpiricalVariograms,
    marginal: MarginalFit,
    r_ab_correlation: float,
    delta_a: float,
    delta_b: float,
    spatial_dimension: int,
) -> np.ndarray:
    """Construct a valid start using empirical colocated rho only for R_V initialization."""
    p = len(empirical.periods)
    r_a = _equicorrelation(p, r_ab_correlation)
    r_b = _equicorrelation(p, r_ab_correlation)
    parameters_a = _correlation_to_factor_parameters(r_a, positive_factor=True)
    parameters_b = _correlation_to_factor_parameters(r_b, positive_factor=True)

    temporary = np.concatenate(
        [[np.log(delta_a), np.log(delta_b)], parameters_a, parameters_b, np.zeros(p * (p - 1) // 2)]
    )
    shape_model = _unpack_joint_parameters(temporary, marginal, spatial_dimension)
    target_sigma = empirical.rho_empirical * np.sqrt(
        np.outer(marginal.sigma_diag, marginal.sigma_diag)
    )
    multiplier = _ags_v_multiplier(
        shape_model["nu"], shape_model["alpha"], delta_a, spatial_dimension
    )
    target_v = target_sigma * multiplier
    v_diag = marginal.sigma_diag * np.diag(multiplier)
    target_r_v = np.divide(
        target_v,
        np.sqrt(np.outer(v_diag, v_diag)),
        out=np.eye(p),
        where=np.outer(v_diag, v_diag) > 0,
    )
    target_r_v = np.clip(0.5 * (target_r_v + target_r_v.T), -1.0, 1.0)
    np.fill_diagonal(target_r_v, 1.0)
    r_v = _nearest_correlation(target_r_v)
    parameters_v = _correlation_to_factor_parameters(r_v, positive_factor=False)
    return np.concatenate(
        [[np.log(delta_a), np.log(delta_b)], parameters_a, parameters_b, parameters_v]
    )


def fit_joint_cross_model(
    empirical: EmpiricalVariograms,
    marginal: MarginalFit,
    spatial_dimension: int = 2,
) -> tuple[dict, dict]:
    """Stage 2: jointly fit Delta_A, Delta_B, R_A, R_B and R_V."""
    p = len(empirical.periods)
    n_correlation = p * (p - 1) // 2
    starts = {
        "moderate_RAB": _joint_start(
            empirical, marginal, r_ab_correlation=0.50, delta_a=0.25, delta_b=0.005,
            spatial_dimension=spatial_dimension,
        ),
        "high_RAB": _joint_start(
            empirical, marginal, r_ab_correlation=0.80, delta_a=0.50, delta_b=0.010,
            spatial_dimension=spatial_dimension,
        ),
    }

    def objective_and_gradient(parameters):
        return _joint_objective_and_gradient(
            parameters, empirical, marginal, spatial_dimension
        )

    bounds = (
        [(np.log(1.0e-6), np.log(5.0)), (np.log(1.0e-10), np.log(1.0))]
        + [(-12.0, 6.0)] * n_correlation
        + [(-12.0, 6.0)] * n_correlation
        + [(-20.0, 20.0)] * n_correlation
    )
    results = []
    for name, initial in starts.items():
        initial_wss = float(objective_and_gradient(initial)[0])
        result = minimize(
            objective_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 2000, "maxfun": 150000, "ftol": 1.0e-9, "gtol": 1.0e-5, "maxls": 60},
        )
        results.append((name, result, initial_wss))

    selected_name, selected, initial_wss = min(results, key=lambda item: float(item[1].fun))
    model = _unpack_joint_parameters(selected.x, marginal, spatial_dimension)
    numerical_jitter = 1.0e-10
    model["r_a"] = (1.0 - numerical_jitter) * model["r_a"] + numerical_jitter * np.eye(p)
    model["r_b"] = (1.0 - numerical_jitter) * model["r_b"] + numerical_jitter * np.eye(p)
    model["r_v"] = (1.0 - numerical_jitter) * model["r_v"] + numerical_jitter * np.eye(p)

    # Rebuild all derived matrices after the numerical PD jitter.
    average_nu = 0.5 * (marginal.nu_diag[:, None] + marginal.nu_diag[None, :])
    average_alpha_sq = 0.5 * (
        marginal.alpha_diag[:, None] ** 2 + marginal.alpha_diag[None, :] ** 2
    )
    model["nu"] = average_nu + model["delta_a"] * (1.0 - model["r_a"])
    model["alpha"] = np.sqrt(
        np.clip(average_alpha_sq + model["delta_b"] * (1.0 - model["r_b"]), 1.0e-15, None)
    )
    np.fill_diagonal(model["nu"], marginal.nu_diag)
    np.fill_diagonal(model["alpha"], marginal.alpha_diag)
    multiplier = _ags_v_multiplier(
        model["nu"], model["alpha"], model["delta_a"], spatial_dimension
    )
    v_diag = marginal.sigma_diag * np.diag(multiplier)
    model["v_matrix"] = np.sqrt(np.outer(v_diag, v_diag)) * model["r_v"]
    model["sigma"] = np.divide(
        model["v_matrix"], multiplier, out=np.zeros_like(multiplier), where=multiplier > 0
    )
    model["sigma"] = 0.5 * (model["sigma"] + model["sigma"].T)
    np.fill_diagonal(model["sigma"], marginal.sigma_diag)
    model["rho"] = model["sigma"] / np.sqrt(
        np.outer(marginal.sigma_diag, marginal.sigma_diag)
    )
    model["rho"] = 0.5 * (model["rho"] + model["rho"].T)
    np.fill_diagonal(model["rho"], 1.0)
    final_cross_wss = _cross_wss(empirical, model)

    diagnostics = {
        "parameterization": "joint full-factor correlation matrices",
        "free_parameter_count": int(2 + 3 * n_correlation),
        "free_parameters": {
            "delta_A": 1,
            "delta_B": 1,
            "R_A_offdiagonal_factor_parameters": int(n_correlation),
            "R_B_offdiagonal_factor_parameters": int(n_correlation),
            "R_V_offdiagonal_factor_parameters": int(n_correlation),
        },
        "selected_start": selected_name,
        "selected_start_cross_wss": initial_wss,
        "final_cross_wss": final_cross_wss,
        "optimizer_objective_before_jitter": float(selected.fun),
        "numerical_positive_definite_jitter": numerical_jitter,
        "success": bool(selected.success),
        "message": str(selected.message),
        "iterations": int(selected.nit),
        "function_evaluations": int(selected.nfev),
        "final_gradient_inf_norm": float(np.max(np.abs(selected.jac))),
        "all_starts": [
            {
                "start": name,
                "initial_cross_wss": initial,
                "final_cross_wss": float(result.fun),
                "success": bool(result.success),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "final_gradient_inf_norm": float(np.max(np.abs(result.jac))),
            }
            for name, result, initial in results
        ],
    }
    return model, diagnostics


def fit_flexible_matern(
    empirical: EmpiricalVariograms,
    tail_bins: int = 10,
    spatial_dimension: int = 2,
) -> FlexibleMaternFit:
    """Run the requested marginal-first, joint-cross-second estimation."""
    marginal = fit_marginal_models(empirical, tail_bins=tail_bins)
    model, joint_optimizer = fit_joint_cross_model(
        empirical, marginal, spatial_dimension=spatial_dimension
    )
    cross_wss = float(joint_optimizer["final_cross_wss"])
    return FlexibleMaternFit(
        periods=empirical.periods.copy(),
        h_bins=empirical.h_bins.copy(),
        sigma=model["sigma"],
        rho=model["rho"],
        rho_empirical=empirical.rho_empirical.copy(),
        alpha=model["alpha"],
        nu=model["nu"],
        delta_a=float(model["delta_a"]),
        delta_b=float(model["delta_b"]),
        r_a=model["r_a"],
        r_b=model["r_b"],
        r_v=model["r_v"],
        v_matrix=model["v_matrix"],
        marginal=marginal,
        joint_optimizer=joint_optimizer,
        wss_marginal=float(marginal.wss),
        wss_cross=cross_wss,
        wss_total=float(marginal.wss + cross_wss),
    )


def empirical_long_frame(empirical: EmpiricalVariograms, fit: FlexibleMaternFit) -> pd.DataFrame:
    rows = []
    for i in range(len(empirical.periods)):
        for j in range(i + 1):
            fitted = theoretical_semivariogram(
                empirical.h_bins, fit.sigma[i, j], fit.nu[i, j], fit.alpha[i, j]
            )
            for b, distance in enumerate(empirical.h_bins):
                rows.append(
                    {
                        "i": i,
                        "j": j,
                        "period_i_s": empirical.periods[i],
                        "period_j_s": empirical.periods[j],
                        "distance_km": distance,
                        "gamma_empirical": empirical.gamma[i, j, b],
                        "gamma_fitted": fitted[b],
                        "station_pairs": int(empirical.counts[i, j, b]),
                        "w_distance": 1.0 / distance,
                        "common_records": int(empirical.record_counts[i, j]),
                        "eligible_events": int(empirical.event_counts[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def parameter_long_frame(empirical: EmpiricalVariograms, fit: FlexibleMaternFit) -> pd.DataFrame:
    rows = []
    for i in range(len(empirical.periods)):
        for j in range(i + 1):
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "period_i_s": empirical.periods[i],
                    "period_j_s": empirical.periods[j],
                    "parameter_stage": "marginal_stage_1" if i == j else "joint_cross_stage_2",
                    "sigma_ij": fit.sigma[i, j],
                    "rho_ij": fit.rho[i, j],
                    "rho_empirical_pairwise_common": fit.rho_empirical[i, j],
                    "alpha_ij_per_km": fit.alpha[i, j],
                    "nu_ij": fit.nu[i, j],
                    "curve_wss": _curve_wss(
                        empirical.gamma[i, j], empirical.counts[i, j], empirical.h_bins,
                        fit.sigma[i, j], fit.nu[i, j], fit.alpha[i, j],
                    ),
                    "common_records": int(empirical.record_counts[i, j]),
                    "eligible_events": int(empirical.event_counts[i, j]),
                }
            )
    return pd.DataFrame(rows)


def covariance_matrix_for_sites(
    lat: Iterable[float],
    lon: Iterable[float],
    fit: FlexibleMaternFit,
) -> np.ndarray:
    """Build a complete site-by-period covariance matrix for PSD checking."""
    lat = np.asarray(list(lat), dtype=float)
    lon = np.asarray(list(lon), dtype=float)
    distance = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    n_sites = len(lat)
    p = len(fit.periods)
    covariance = np.zeros((n_sites * p, n_sites * p), dtype=float)
    for a in range(n_sites):
        for b in range(n_sites):
            block = np.zeros((p, p), dtype=float)
            for i in range(p):
                for j in range(p):
                    block[i, j] = fit.sigma[i, j] * matern_correlation(
                        np.array([distance[a, b]]), fit.nu[i, j], fit.alpha[i, j]
                    )[0]
            covariance[a * p : (a + 1) * p, b * p : (b + 1) * p] = block
    return 0.5 * (covariance + covariance.T)


def validity_diagnostics(
    empirical: EmpiricalVariograms,
    fit: FlexibleMaternFit,
    frames: list[pd.DataFrame],
    max_sites: int = 18,
) -> dict:
    """Check the AGS matrices and representative full covariance blocks."""
    sigma_diag = np.diag(fit.sigma)
    formula_error = np.max(
        np.abs(fit.sigma - fit.rho * np.sqrt(np.outer(sigma_diag, sigma_diag)))
    )
    diag_nu = np.diag(fit.nu)
    diag_alpha_sq = np.diag(fit.alpha) ** 2
    nu_slack = fit.nu - 0.5 * (diag_nu[:, None] + diag_nu[None, :])
    alpha_slack = fit.alpha**2 - 0.5 * (
        diag_alpha_sq[:, None] + diag_alpha_sq[None, :]
    )
    offdiagonal = ~np.eye(len(fit.periods), dtype=bool)

    block_checks = []
    for eqid, event in frames[0].groupby("eqid", sort=False):
        if len(event) < max_sites:
            continue
        sample = event.iloc[:max_sites]
        covariance = covariance_matrix_for_sites(
            sample["station_latitude"], sample["station_longitude"], fit
        )
        eigenvalues = np.linalg.eigvalsh(covariance)
        block_checks.append(
            {
                "eqid": str(eqid),
                "sites": int(max_sites),
                "dimension": int(len(covariance)),
                "min_eigenvalue": float(eigenvalues[0]),
                "max_eigenvalue": float(eigenvalues[-1]),
            }
        )
        if len(block_checks) >= 3:
            break

    return {
        "estimation_order": [
            "stage_1_fit_sigma_ii_alpha_ii_nu_ii_from_auto_curves",
            "stage_2_jointly_fit_delta_A_delta_B_R_A_R_B_R_V_from_cross_curves",
        ],
        "number_of_periods": int(len(fit.periods)),
        "number_of_auto_curves": int(len(fit.periods)),
        "number_of_cross_curves": int(len(fit.periods) * (len(fit.periods) - 1) // 2),
        "number_of_total_curves": int(len(fit.periods) * (len(fit.periods) + 1) // 2),
        "weight_definition": "w_ijb = 1 / h_b for every supported bin",
        "anisotropy": False,
        "sigma_formula_max_abs_error": float(formula_error),
        "min_eigenvalue_R_A": float(np.linalg.eigvalsh(fit.r_a)[0]),
        "min_eigenvalue_R_B": float(np.linalg.eigvalsh(fit.r_b)[0]),
        "min_eigenvalue_R_V": float(np.linalg.eigvalsh(fit.r_v)[0]),
        "min_entry_R_A": float(np.min(fit.r_a)),
        "min_entry_R_B": float(np.min(fit.r_b)),
        "min_eigenvalue_V": float(np.linalg.eigvalsh(fit.v_matrix)[0]),
        "min_eigenvalue_sigma_at_zero": float(np.linalg.eigvalsh(fit.sigma)[0]),
        "min_cross_nu_minus_marginal_average": float(np.min(nu_slack[offdiagonal])),
        "min_cross_alpha_sq_minus_marginal_average": float(np.min(alpha_slack[offdiagonal])),
        "raw_empirical_rho_min_eigenvalue": float(np.linalg.eigvalsh(empirical.rho_empirical)[0]),
        "delta_A": float(fit.delta_a),
        "delta_B": float(fit.delta_b),
        "wss_marginal_stage_1": float(fit.wss_marginal),
        "wss_cross_stage_2": float(fit.wss_cross),
        "wss_total": float(fit.wss_total),
        "joint_optimizer": fit.joint_optimizer,
        "representative_full_block_checks": block_checks,
    }
