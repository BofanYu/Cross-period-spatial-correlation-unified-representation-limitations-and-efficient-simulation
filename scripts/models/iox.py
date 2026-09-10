"""Event-pooled WLS semivariogram estimator for the nine-period IOX model.

The empirical calculation is always performed inside an earthquake event.
Only the resulting bin sums and pair counts are pooled across events.  The
cross-period IOX basis uses an exact, event-specific Cholesky square root of
each fitted marginal correlation matrix.  Consequently, the nine auto
covariances are exactly the fitted Matérn models and a positive-definite
coregionalization matrix guarantees a valid covariance inside every event.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit, gammaln, kv


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
PERIOD_NAMES = tuple(code for code, _ in PERIOD_SPECS)
PERIODS = np.asarray([period for _, period in PERIOD_SPECS], dtype=float)
EARTH_RADIUS_KM = 6371.0088


@dataclass
class EventData:
    eqid: str
    recid: np.ndarray
    coordinates_km: np.ndarray
    values: np.ndarray


@dataclass
class EmpiricalVariograms:
    h: np.ndarray
    gamma: np.ndarray
    counts: np.ndarray
    common_records: np.ndarray
    eligible_events: np.ndarray


@dataclass
class MarginalFit:
    sill: np.ndarray
    phi: np.ndarray
    nu: np.ndarray
    nugget_fraction: np.ndarray
    wss_by_period: np.ndarray
    optimizer_rows: list[dict]


@dataclass
class IOXWLSFit:
    periods: np.ndarray
    marginal: MarginalFit
    basis: np.ndarray
    scaling: np.ndarray
    sigma: np.ndarray
    rho_core: np.ndarray
    rho_colocated: np.ndarray
    raw_pairwise_rho: np.ndarray
    fitted_gamma: np.ndarray
    wss_auto: float
    wss_cross: float
    wss_total: float
    optimizer: dict


def earth_chord_coordinates(longitude, latitude) -> np.ndarray:
    """Return 3-D Earth-chord coordinates in kilometres."""
    lon = np.radians(np.asarray(longitude, dtype=float))
    lat = np.radians(np.asarray(latitude, dtype=float))
    return EARTH_RADIUS_KM * np.column_stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
    )


def load_union_panel(input_dir: str | Path) -> tuple[pd.DataFrame, list[EventData], pd.DataFrame]:
    """Load the union of all records; do not impose a nine-period complete case."""
    input_dir = Path(input_dir)
    required = ("recid", "eqid", "station_longitude", "station_latitude", "scaled_deltaW")
    responses: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for code, period in PERIOD_SPECS:
        path = input_dir / f"data_ngawest2_{code}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise KeyError(f"{path.name} is missing columns: {missing}")
        frame = frame.loc[:, required].dropna().copy()
        if frame["recid"].duplicated().any():
            raise ValueError(f"{path.name} contains duplicate recid values")
        frame["eqid"] = frame["eqid"].astype(str)
        responses.append(frame[["recid", "scaled_deltaW"]].rename(columns={"scaled_deltaW": code}))
        metadata_parts.append(frame[["recid", "eqid", "station_longitude", "station_latitude"]])
        event_sizes = frame.groupby("eqid", sort=False).size()
        summaries.append(
            {
                "period_s": period,
                "file": path.name,
                "records": int(len(frame)),
                "events": int(frame["eqid"].nunique()),
                "min_records_per_event": int(event_sizes.min()),
                "median_records_per_event": float(event_sizes.median()),
                "max_records_per_event": int(event_sizes.max()),
            }
        )

    all_metadata = pd.concat(metadata_parts, ignore_index=True)
    first_metadata = all_metadata.drop_duplicates("recid", keep="first")
    check = all_metadata.merge(first_metadata, on="recid", suffixes=("", "_first"), validate="many_to_one")
    inconsistent = (
        (check["eqid"] != check["eqid_first"])
        | ((check["station_longitude"] - check["station_longitude_first"]).abs() > 1.0e-8)
        | ((check["station_latitude"] - check["station_latitude_first"]).abs() > 1.0e-8)
    )
    if inconsistent.any():
        raise ValueError("At least one recid has inconsistent event or coordinate metadata")

    panel = first_metadata.copy()
    for response in responses:
        panel = panel.merge(response, on="recid", how="outer", validate="one_to_one", sort=False)
    panel = panel.sort_values(["eqid", "recid"], kind="stable").reset_index(drop=True)
    if (panel.loc[:, PERIOD_NAMES].notna().sum(axis=1) == 0).any():
        raise ValueError("The union panel contains a row with no period response")

    events: list[EventData] = []
    for eqid, frame in panel.groupby("eqid", sort=False):
        events.append(
            EventData(
                eqid=str(eqid),
                recid=frame["recid"].to_numpy(),
                coordinates_km=earth_chord_coordinates(
                    frame["station_longitude"].to_numpy(), frame["station_latitude"].to_numpy()
                ),
                values=frame.loc[:, PERIOD_NAMES].to_numpy(dtype=float),
            )
        )
    return panel, events, pd.DataFrame(summaries)


def _pair_bins(coordinates_km: np.ndarray, bin_size_km: float, max_distance_km: float):
    """Match the established Matérn workflow: nearest-bin assignment, excluding bin zero."""
    n = len(coordinates_km)
    a, b = np.triu_indices(n, k=1)
    distance = np.linalg.norm(coordinates_km[a] - coordinates_km[b], axis=1)
    bin_number = np.rint(distance / bin_size_km).astype(np.int64)
    n_bins = int(round(max_distance_km / bin_size_km))
    spatial = (bin_number >= 1) & (bin_number <= n_bins)
    return a, b, distance, bin_number - 1, spatial


def empirical_event_semivariograms(
    events: list[EventData], bin_size_km: float = 2.0, max_distance_km: float = 100.0
) -> EmpiricalVariograms:
    """Compute every curve event-by-event, then pool pair sums and counts."""
    q = len(PERIODS)
    n_bins = int(round(max_distance_km / bin_size_km))
    sums = np.zeros((q, q, n_bins), dtype=float)
    counts = np.zeros((q, q, n_bins), dtype=np.int64)
    common_records = np.zeros((q, q), dtype=np.int64)
    eligible_events = np.zeros((q, q), dtype=np.int64)

    for event in events:
        a, b, _, bins0, spatial = _pair_bins(event.coordinates_km, bin_size_km, max_distance_km)
        observed = np.isfinite(event.values)
        for i in range(q):
            for j in range(i + 1):
                common = observed[:, i] & observed[:, j]
                common_records[i, j] += int(common.sum())
                if common.sum() < 2:
                    continue
                keep = spatial & common[a] & common[b]
                if not keep.any():
                    continue
                eligible_events[i, j] += 1
                delta_i = event.values[a[keep], i] - event.values[b[keep], i]
                delta_j = event.values[a[keep], j] - event.values[b[keep], j]
                contribution = 0.5 * delta_i * delta_j
                sums[i, j] += np.bincount(bins0[keep], weights=contribution, minlength=n_bins)
                counts[i, j] += np.bincount(bins0[keep], minlength=n_bins)

    gamma = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=gamma, where=counts > 0)
    for i in range(q):
        for j in range(i):
            gamma[j, i] = gamma[i, j]
            counts[j, i] = counts[i, j]
            common_records[j, i] = common_records[i, j]
            eligible_events[j, i] = eligible_events[i, j]
    h = np.arange(1, n_bins + 1, dtype=float) * bin_size_km
    return EmpiricalVariograms(h, gamma, counts, common_records, eligible_events)


def matern_correlation(distance, phi: float, nu: float) -> np.ndarray:
    """Unit-variance Matérn correlation with rate phi (per km)."""
    distance = np.asarray(distance, dtype=float)
    x = float(phi) * np.abs(distance)
    result = np.ones_like(x)
    positive = x > 0
    xp = x[positive]
    if xp.size:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            log_result = (1.0 - float(nu)) * np.log(2.0) - gammaln(float(nu))
            log_result = log_result + float(nu) * np.log(xp) + np.log(kv(float(nu), xp))
            value = np.exp(log_result)
        result[positive] = np.clip(np.where(np.isfinite(value), value, 0.0), 0.0, 1.0)
    return result


def marginal_semivariogram(distance, sill: float, phi: float, nu: float, nugget_fraction: float):
    """Marginal semivariogram; alpha is the nugget fraction of the total sill."""
    distance = np.asarray(distance, dtype=float)
    out = float(sill) * (
        1.0 - (1.0 - float(nugget_fraction)) * matern_correlation(distance, phi, nu)
    )
    out = np.asarray(out)
    out[distance == 0] = 0.0
    return out


def tail_auto_sills(empirical: EmpiricalVariograms, tail_bins: int = 10) -> np.ndarray:
    sills = np.empty(len(PERIODS), dtype=float)
    for i, period in enumerate(PERIODS):
        supported = (empirical.counts[i, i] > 0) & np.isfinite(empirical.gamma[i, i])
        values = empirical.gamma[i, i, supported]
        if len(values) < tail_bins:
            raise ValueError(f"T={period:g}s has fewer than {tail_bins} supported bins")
        sills[i] = float(values[-tail_bins:].mean())
        if not np.isfinite(sills[i]) or sills[i] <= 0:
            raise ValueError(f"T={period:g}s has an invalid tail sill")
    return sills


def _auto_wss(parameters, h, gamma, sill) -> float:
    log_phi, log_nu, alpha_logit = parameters
    phi = float(np.exp(log_phi))
    nu = float(np.exp(log_nu))
    alpha = float(expit(alpha_logit))
    prediction = marginal_semivariogram(h, sill, phi, nu, alpha)
    return float(np.sum((gamma - prediction) ** 2 / h))


def fit_marginals(empirical: EmpiricalVariograms, tail_bins: int = 10) -> MarginalFit:
    """Fit phi, nu and nugget fraction separately; keep each sill fixed."""
    q = len(PERIODS)
    sill = tail_auto_sills(empirical, tail_bins)
    phi = np.empty(q)
    nu = np.empty(q)
    nugget = np.empty(q)
    wss = np.empty(q)
    rows: list[dict] = []
    bounds = [
        (np.log(1.0e-4), np.log(2.0)),
        (np.log(0.05), np.log(5.0)),
        (-9.21024037, 2.94443898),  # alpha in [0.0001, 0.95]
    ]
    starts = (
        (0.014, 0.20, 0.20),
        (0.008, 0.50, 0.05),
        (0.030, 1.00, 0.35),
        (0.080, 2.00, 0.60),
    )

    for i, period in enumerate(PERIODS):
        supported = (empirical.counts[i, i] > 0) & np.isfinite(empirical.gamma[i, i])
        h = empirical.h[supported]
        gamma = empirical.gamma[i, i, supported]
        results = []
        for phi0, nu0, nugget0 in starts:
            x0 = np.asarray(
                [np.log(phi0), np.log(nu0), np.log(nugget0 / (1.0 - nugget0))]
            )
            results.append(
                minimize(
                    _auto_wss,
                    x0,
                    args=(h, gamma, sill[i]),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 2000, "ftol": 1.0e-14, "gtol": 1.0e-9},
                )
            )
        best = min(results, key=lambda result: float(result.fun))
        phi[i], nu[i], nugget[i] = (
            np.exp(best.x[0]),
            np.exp(best.x[1]),
            expit(best.x[2]),
        )
        wss[i] = float(best.fun)
        rows.append(
            {
                "period_s": period,
                "sigma_ii_tail10": sill[i],
                "phi_per_km": phi[i],
                "nu": nu[i],
                "nugget_fraction_alpha": nugget[i],
                "nugget_variance": sill[i] * nugget[i],
                "spatial_variance": sill[i] * (1.0 - nugget[i]),
                "wss": wss[i],
                "success": bool(best.success),
                "message": str(best.message),
                "iterations": int(best.nit),
                "function_evaluations": int(best.nfev),
            }
        )
    return MarginalFit(sill, phi, nu, nugget, wss, rows)


def _maxmin_order(coordinates: np.ndarray, recid: np.ndarray) -> np.ndarray:
    """Deterministic farthest-point order, applied separately inside each event."""
    n = len(coordinates)
    if n <= 2:
        return np.arange(n, dtype=np.int64)
    center = coordinates.mean(axis=0)
    from_center = np.sum((coordinates - center) ** 2, axis=1)
    first = int(np.argmax(from_center))
    selected = np.zeros(n, dtype=bool)
    selected[first] = True
    order = np.empty(n, dtype=np.int64)
    order[0] = first
    min_distance_sq = np.sum((coordinates - coordinates[first]) ** 2, axis=1)
    min_distance_sq[first] = -np.inf
    for k in range(1, n):
        next_index = int(np.argmax(min_distance_sq))
        order[k] = next_index
        selected[next_index] = True
        distance_sq = np.sum((coordinates - coordinates[next_index]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
        min_distance_sq[selected] = -np.inf
    return order


def iox_semivariogram_basis(
    events: list[EventData],
    marginal: MarginalFit,
    bin_size_km: float = 2.0,
    max_distance_km: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool exact event IOX unit-sigma semivariogram factors by distance bin."""
    q = len(PERIODS)
    n_bins = int(round(max_distance_km / bin_size_km))
    sums = np.zeros((q, q, n_bins), dtype=float)
    counts = np.zeros((q, q, n_bins), dtype=np.int64)
    scaling_sums = np.zeros((q, q), dtype=float)
    scaling_counts = np.zeros((q, q), dtype=np.int64)

    for event in events:
        order = _maxmin_order(event.coordinates_km, event.recid)
        coordinates = event.coordinates_km[order]
        values = event.values[order]
        observed = np.isfinite(values)
        distances = squareform(pdist(coordinates))
        a, b, _, bins0, spatial = _pair_bins(coordinates, bin_size_km, max_distance_km)

        correlation_matrices: list[np.ndarray] = []
        factors: list[np.ndarray] = []
        for j in range(q):
            correlation = (1.0 - marginal.nugget_fraction[j]) * matern_correlation(
                distances, marginal.phi[j], marginal.nu[j]
            )
            np.fill_diagonal(correlation, 1.0)
            correlation = 0.5 * (correlation + correlation.T)
            try:
                factor = cholesky(correlation, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                factor = cholesky(
                    correlation + 1.0e-10 * np.eye(len(correlation)),
                    lower=True,
                    check_finite=False,
                )
            correlation_matrices.append(correlation)
            factors.append(factor)

        for i in range(q):
            for j in range(i + 1):
                common = observed[:, i] & observed[:, j]
                if not common.any():
                    continue
                if i == j:
                    diagonal_factor = np.ones(len(values))
                    cross_factor = correlation_matrices[i]
                else:
                    cross_factor = factors[i] @ factors[j].T
                    diagonal_factor = np.diag(cross_factor)
                scaling_sums[i, j] += float(diagonal_factor[common].sum())
                scaling_counts[i, j] += int(common.sum())

                keep = spatial & common[a] & common[b]
                if not keep.any():
                    continue
                ak = a[keep]
                bk = b[keep]
                unit_semivariogram = 0.5 * (
                    diagonal_factor[ak]
                    + diagonal_factor[bk]
                    - cross_factor[ak, bk]
                    - cross_factor[bk, ak]
                )
                sums[i, j] += np.bincount(
                    bins0[keep], weights=unit_semivariogram, minlength=n_bins
                )
                counts[i, j] += np.bincount(bins0[keep], minlength=n_bins)

    basis = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=basis, where=counts > 0)
    scaling = np.full((q, q), np.nan)
    np.divide(scaling_sums, scaling_counts, out=scaling, where=scaling_counts > 0)
    for i in range(q):
        for j in range(i):
            basis[j, i] = basis[i, j]
            counts[j, i] = counts[i, j]
            scaling[j, i] = scaling[i, j]
    return basis, scaling, counts


def _nearest_correlation(matrix: np.ndarray, iterations: int = 200) -> np.ndarray:
    y = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    np.fill_diagonal(y, 1.0)
    correction = np.zeros_like(y)
    for _ in range(iterations):
        residual = y - correction
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (residual + residual.T))
        psd = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        correction = psd - residual
        d = np.sqrt(np.clip(np.diag(psd), 1.0e-15, None))
        updated = psd / np.outer(d, d)
        np.fill_diagonal(updated, 1.0)
        if np.max(np.abs(updated - y)) < 1.0e-12:
            y = updated
            break
        y = updated
    return 0.999 * y + 0.001 * np.eye(len(y))


def _pack_unit_lower_from_correlation(correlation: np.ndarray) -> np.ndarray:
    factor = np.linalg.cholesky(correlation)
    unit_lower = factor / np.diag(factor)[:, None]
    return unit_lower[np.tril_indices(len(correlation), k=-1)]


def _correlation_from_unit_lower(parameters: np.ndarray, q: int, eigen_floor: float = 1.0e-6):
    unit_lower = np.eye(q)
    unit_lower[np.tril_indices(q, k=-1)] = parameters
    covariance = unit_lower @ unit_lower.T
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    correlation = (1.0 - eigen_floor) * correlation + eigen_floor * np.eye(q)
    return 0.5 * (correlation + correlation.T)


def fit_cross_coregionalization(
    empirical: EmpiricalVariograms,
    marginal: MarginalFit,
    basis: np.ndarray,
    eigen_floor: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Jointly minimize the sum of all 36 cross WLS terms under Sigma > 0."""
    q = len(PERIODS)
    pair_indices = [(i, j) for i in range(q) for j in range(i)]
    quadratic_a = np.zeros(len(pair_indices))
    quadratic_b = np.zeros(len(pair_indices))
    quadratic_c = np.zeros(len(pair_indices))
    raw_rho = np.eye(q)

    for k, (i, j) in enumerate(pair_indices):
        valid = (
            (empirical.counts[i, j] > 0)
            & np.isfinite(empirical.gamma[i, j])
            & np.isfinite(basis[i, j])
        )
        h = empirical.h[valid]
        y = empirical.gamma[i, j, valid]
        x = np.sqrt(marginal.sill[i] * marginal.sill[j]) * basis[i, j, valid]
        w = 1.0 / h
        quadratic_a[k] = float(np.sum(w * x * x))
        quadratic_b[k] = float(np.sum(w * x * y))
        quadratic_c[k] = float(np.sum(w * y * y))
        estimate = quadratic_b[k] / quadratic_a[k] if quadratic_a[k] > 0 else 0.0
        raw_rho[i, j] = raw_rho[j, i] = float(estimate)

    clipped_raw = np.clip(raw_rho, -0.995, 0.995)
    np.fill_diagonal(clipped_raw, 1.0)
    start_correlation = _nearest_correlation(clipped_raw)
    starts = (
        _pack_unit_lower_from_correlation(start_correlation),
        np.zeros(q * (q - 1) // 2),
    )

    lower_indices = np.tril_indices(q, k=-1)

    def objective(parameters):
        correlation = _correlation_from_unit_lower(parameters, q, eigen_floor)
        rho = correlation[lower_indices]
        return float(np.sum(quadratic_a * rho * rho - 2.0 * quadratic_b * rho + quadratic_c))

    results = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(-5.0, 5.0)] * len(start),
            options={"maxiter": 2500, "maxls": 50, "ftol": 1.0e-14, "gtol": 1.0e-9},
        )
        for start in starts
    ]
    best = min(results, key=lambda result: float(result.fun))
    correlation = _correlation_from_unit_lower(best.x, q, eigen_floor)
    diagnostics = {
        "success": bool(best.success),
        "message": str(best.message),
        "iterations": int(best.nit),
        "function_evaluations": int(best.nfev),
        "cross_wss": float(best.fun),
        "number_of_cross_curves": len(pair_indices),
        "free_cross_parameters": len(best.x),
        "minimum_eigenvalue_rho_core": float(np.linalg.eigvalsh(correlation).min()),
        "start_results": [
            {
                "success": bool(result.success),
                "wss": float(result.fun),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
            }
            for result in results
        ],
    }
    return correlation, raw_rho, diagnostics


def fit_iox_wls(
    empirical: EmpiricalVariograms,
    events: list[EventData],
    tail_bins: int = 10,
    bin_size_km: float = 2.0,
    max_distance_km: float = 100.0,
) -> IOXWLSFit:
    marginal = fit_marginals(empirical, tail_bins)
    basis, scaling, basis_counts = iox_semivariogram_basis(
        events, marginal, bin_size_km=bin_size_km, max_distance_km=max_distance_km
    )
    if not np.array_equal(basis_counts, empirical.counts):
        raise RuntimeError("The empirical and IOX theoretical bases used different station pairs")

    rho_core, raw_rho, optimizer = fit_cross_coregionalization(empirical, marginal, basis)
    sigma = rho_core * np.sqrt(np.outer(marginal.sill, marginal.sill))
    fitted = sigma[:, :, None] * basis
    for i in range(len(PERIODS)):
        fitted[i, i] = marginal_semivariogram(
            empirical.h,
            marginal.sill[i],
            marginal.phi[i],
            marginal.nu[i],
            marginal.nugget_fraction[i],
        )

    rho_colocated = rho_core * scaling
    np.fill_diagonal(rho_colocated, 1.0)
    wss_auto = float(marginal.wss_by_period.sum())
    wss_cross = float(optimizer["cross_wss"])
    return IOXWLSFit(
        periods=PERIODS.copy(),
        marginal=marginal,
        basis=basis,
        scaling=scaling,
        sigma=sigma,
        rho_core=rho_core,
        rho_colocated=rho_colocated,
        raw_pairwise_rho=raw_rho,
        fitted_gamma=fitted,
        wss_auto=wss_auto,
        wss_cross=wss_cross,
        wss_total=wss_auto + wss_cross,
        optimizer=optimizer,
    )


def empirical_long_frame(empirical: EmpiricalVariograms, fit: IOXWLSFit) -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(len(PERIODS)):
        for j in range(i + 1):
            for b, h in enumerate(empirical.h):
                if empirical.counts[i, j, b] <= 0:
                    continue
                residual = empirical.gamma[i, j, b] - fit.fitted_gamma[i, j, b]
                rows.append(
                    {
                        "period_i_s": PERIODS[i],
                        "period_j_s": PERIODS[j],
                        "curve_type": "auto" if i == j else "cross",
                        "distance_km": h,
                        "empirical_semivariogram": empirical.gamma[i, j, b],
                        "iox_unit_sigma_basis": fit.basis[i, j, b],
                        "fitted_semivariogram": fit.fitted_gamma[i, j, b],
                        "residual": residual,
                        "weight_1_over_h": 1.0 / h,
                        "weighted_squared_residual": residual * residual / h,
                        "station_pair_count": int(empirical.counts[i, j, b]),
                        "common_record_count": int(empirical.common_records[i, j]),
                        "eligible_event_count": int(empirical.eligible_events[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def pair_parameter_frame(fit: IOXWLSFit) -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(len(PERIODS)):
        for j in range(i + 1):
            rows.append(
                {
                    "period_i_s": PERIODS[i],
                    "period_j_s": PERIODS[j],
                    "curve_type": "auto" if i == j else "cross",
                    "sigma_ij_core": fit.sigma[i, j],
                    "rho_ij_core": fit.rho_core[i, j],
                    "iox_colocated_scaling": fit.scaling[i, j],
                    "rho_ij_colocated_average": fit.rho_colocated[i, j],
                    "raw_pairwise_rho_before_pd_constraint": fit.raw_pairwise_rho[i, j],
                    "phi_i_per_km": fit.marginal.phi[i],
                    "phi_j_per_km": fit.marginal.phi[j],
                    "nu_i": fit.marginal.nu[i],
                    "nu_j": fit.marginal.nu[j],
                    "nugget_fraction_i": fit.marginal.nugget_fraction[i],
                    "nugget_fraction_j": fit.marginal.nugget_fraction[j],
                }
            )
    return pd.DataFrame(rows)

