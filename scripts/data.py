"""Load the processed residuals and compute empirical spatial statistics."""
from pathlib import Path

import numpy as np
import pandas as pd


def col_to_period(col):
    """Convert a scaled_deltaW...WT column name to oscillator period."""
    return float(col.split("WT")[-1]) / 100.0


def select_period_columns(df, periods):
    """Return residual columns matching the requested periods."""
    period_cols = [
        c for c in df.columns
        if c.startswith("scaled_deltaW") and "WT" in c
    ]
    period_pairs = [(col_to_period(c), c) for c in period_cols]

    selected = []
    for period in periods:
        matches = [
            col for value, col in period_pairs
            if np.isclose(value, float(period), rtol=0.0, atol=1e-10)
        ]
        if not matches:
            raise KeyError(f"No residual column found for period {period:g}s")
        selected.append(matches[0])
    return selected


def period_file_code(period):
    """Return the NGA-West2 filename period code, such as T001 or T300."""
    return f"T{int(round(float(period) * 100.0)):03d}"


def load_period_datasets(data_dir, periods):
    """Load one NGA-West2 residual CSV per period, aligned to periods order."""
    data_dir = Path(data_dir)
    period_dfs = []
    keep_cols = [
        "recid",
        "eqid",
        "station_latitude",
        "station_longitude",
        "scaled_deltaW",
    ]

    for period in periods:
        code = period_file_code(period)
        path = data_dir / f"data_ngawest2_{code}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        df_period = pd.read_csv(path)
        missing = [col for col in keep_cols if col not in df_period.columns]
        if missing:
            raise KeyError(f"{path} is missing columns: {missing}")

        period_dfs.append(df_period[keep_cols].dropna().copy())
    return period_dfs


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two points or arrays of points."""
    earth_radius_km = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2))
        * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


def compute_empirical_crossvariograms(
    df_event,
    period_cols,
    bin_size=2,
    max_distance=100,
):
    """Compute empirical cross-variograms for a single event."""
    nper = len(period_cols)
    nmax = round(max_distance / bin_size)

    lats = df_event["station_latitude"].values
    lons = df_event["station_longitude"].values
    eps = df_event[period_cols].values.astype(float)

    dist = haversine_km(
        lats[:, None],
        lons[:, None],
        lats[None, :],
        lons[None, :],
    )
    dist_bin = np.round(dist / bin_size).astype(int)

    gamma = [[np.zeros(nmax) for _ in range(nper)] for _ in range(nper)]
    counts = np.zeros(nmax)

    for b in range(1, nmax + 1):
        mask = dist_bin == b
        npairs = mask.sum()
        if npairs == 0:
            continue

        counts[b - 1] = npairs
        idx_i, idx_j = np.where(mask)

        for k in range(nper):
            diff_k = eps[idx_i, k] - eps[idx_j, k]
            for l in range(nper):
                diff_l = eps[idx_i, l] - eps[idx_j, l]
                gamma[k][l][b - 1] = np.nansum(diff_k * diff_l) / (2.0 * npairs)

    return gamma, counts


def compute_empirical_bivariate_variogram(
    df_event,
    value_col_i,
    value_col_j,
    bin_size=2,
    max_distance=100,
):
    """Compute one empirical cross-variogram for two residual columns."""
    nmax = round(max_distance / bin_size)
    lats = df_event["station_latitude"].values
    lons = df_event["station_longitude"].values
    eps_i = df_event[value_col_i].values.astype(float)
    eps_j = df_event[value_col_j].values.astype(float)

    dist = haversine_km(
        lats[:, None],
        lons[:, None],
        lats[None, :],
        lons[None, :],
    )
    dist_bin = np.round(dist / bin_size).astype(int)

    gamma = np.zeros(nmax)
    counts = np.zeros(nmax)

    for b in range(1, nmax + 1):
        mask = dist_bin == b
        npairs = mask.sum()
        if npairs == 0:
            continue

        counts[b - 1] = npairs
        idx_a, idx_b = np.where(mask)
        diff_i = eps_i[idx_a] - eps_i[idx_b]
        diff_j = eps_j[idx_a] - eps_j[idx_b]
        gamma[b - 1] = np.nansum(diff_i * diff_j) / (2.0 * npairs)

    return gamma, counts


def pairwise_common_frame(df_i, df_j):
    """Return records common to two period data frames, matched by recid."""
    left = df_i.rename(columns={"scaled_deltaW": "eps_i"})
    if df_i is df_j:
        out = left.copy()
        out["eps_j"] = out["eps_i"]
        return out

    right = df_j[["recid", "scaled_deltaW"]].rename(columns={"scaled_deltaW": "eps_j"})
    return left.merge(right, on="recid", how="inner")


def pool_bivariate_variogram(
    df_pair,
    value_col_i="eps_i",
    value_col_j="eps_j",
    bin_size=2,
    max_distance=100,
    event_col="eqid",
    min_stations=5,
):
    """Pool one pairwise empirical cross-variogram over events."""
    nmax = round(max_distance / bin_size)
    gamma_sum = np.zeros(nmax)
    count_sum = np.zeros(nmax)

    for _, df_event in df_pair.groupby(event_col):
        if len(df_event) < min_stations:
            continue

        gamma_event, counts_event = compute_empirical_bivariate_variogram(
            df_event,
            value_col_i,
            value_col_j,
            bin_size=bin_size,
            max_distance=max_distance,
        )
        gamma_sum += gamma_event * counts_event
        count_sum += counts_event

    gamma = np.zeros(nmax)
    np.divide(gamma_sum, count_sum, out=gamma, where=count_sum > 0)
    return gamma, count_sum


def pool_pairwise_crossvariograms(
    period_dfs,
    periods,
    bin_size=2,
    max_distance=100,
    min_stations=5,
):
    """Pool gamma_ij using all period data and pairwise-common cross data."""
    nper = len(periods)
    nmax = round(max_distance / bin_size)
    gamma_emp = [[np.zeros(nmax) for _ in range(nper)] for _ in range(nper)]
    count_emp = [[np.zeros(nmax) for _ in range(nper)] for _ in range(nper)]
    record_counts = np.zeros((nper, nper), dtype=int)

    for i in range(nper):
        for j in range(i + 1):
            df_pair = pairwise_common_frame(period_dfs[i], period_dfs[j])
            record_counts[i, j] = len(df_pair)
            record_counts[j, i] = len(df_pair)

            gamma_ij, counts_ij = pool_bivariate_variogram(
                df_pair,
                bin_size=bin_size,
                max_distance=max_distance,
                min_stations=min_stations,
            )
            gamma_emp[i][j] = gamma_ij
            count_emp[i][j] = counts_ij

            if i != j:
                gamma_emp[j][i] = gamma_ij.copy()
                count_emp[j][i] = counts_ij.copy()

    return gamma_emp, count_emp, record_counts


def pairwise_lag0_corr(period_dfs):
    """Compute lag-zero correlations from each two-period common dataset."""
    nper = len(period_dfs)
    rho = np.eye(nper)

    for i in range(nper):
        for j in range(i):
            df_pair = pairwise_common_frame(period_dfs[i], period_dfs[j])
            if len(df_pair) < 2:
                rho_ij = np.nan
            else:
                rho_ij = np.corrcoef(df_pair["eps_i"], df_pair["eps_j"])[0, 1]
            rho[i, j] = rho_ij
            rho[j, i] = rho_ij
    return rho


def pool_empirical_crossvariograms(
    df_use,
    period_cols,
    bin_size=2,
    max_distance=100,
    event_col="eqid",
    min_stations=5,
):
    """Pool event-wise empirical cross-variograms with pair-count weights."""
    nper = len(period_cols)
    nmax = round(max_distance / bin_size)
    events = df_use[event_col].unique()

    gamma_sum = [[np.zeros(nmax) for _ in range(nper)] for _ in range(nper)]
    count_sum = np.zeros(nmax)

    for eqid in events:
        df_event = df_use[df_use[event_col] == eqid]
        if len(df_event) < min_stations:
            continue

        gamma_event, counts_event = compute_empirical_crossvariograms(
            df_event,
            period_cols,
            bin_size=bin_size,
            max_distance=max_distance,
        )

        for k in range(nper):
            for l in range(nper):
                gamma_sum[k][l] += gamma_event[k][l] * counts_event
        count_sum += counts_event

    gamma_emp = [[np.zeros(nmax) for _ in range(nper)] for _ in range(nper)]
    for k in range(nper):
        for l in range(nper):
            np.divide(
                gamma_sum[k][l],
                count_sum,
                out=gamma_emp[k][l],
                where=count_sum > 0,
            )

    return gamma_emp, count_sum, events


def empirical_sills(gamma_emp, tail_bins=10):
    """Estimate per-period sills from the last lag bins."""
    nper = len(gamma_emp)
    return np.array([
        np.mean(gamma_emp[i][i][-tail_bins:])
        for i in range(nper)
    ])


def effective_ranges(rho, h_grid, level=np.exp(-1.0)):
    """Return the h value nearest to the target correlation level."""
    rho = np.asarray(rho, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    if rho.ndim == 1:
        return h_grid[np.argmin(np.abs(rho - level))]
    return np.array([
        h_grid[np.argmin(np.abs(row - level))]
        for row in rho
    ])


def powered_exponential_corr(distance, length_scale, exponent, jitter=0.0):
    """Powered-exponential correlation kernel."""
    distance = np.asarray(distance, dtype=float)
    length_scale = max(float(length_scale), 1e-6)
    exponent = float(np.clip(exponent, 1e-3, 2.0))
    corr = np.exp(-((distance / length_scale) ** exponent))
    if np.ndim(corr) == 2 and jitter > 0.0:
        corr = corr.copy()
        corr.flat[:: corr.shape[0] + 1] += float(jitter)
    return corr
