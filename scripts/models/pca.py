"""PCA correlation reconstruction with likelihood and semivariogram fits."""
import numpy as np
from scripts.data import effective_ranges, haversine_km, powered_exponential_corr
from scripts.fitting.mle import fit_pe_pairwise_composite_mle
from scripts.fitting.semivariogram import fit_pe_wls



# Likelihood fitting


def corr_matrix_mle(y):
    """Return covariance and correlation matrices for column data."""
    cov = (y.T @ y) / len(y)
    scale = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(scale, scale)
    return cov, corr


def pca_from_corr_mle(corr):
    """Eigen-decompose a correlation matrix with deterministic signs."""
    evals, evecs = np.linalg.eigh(corr)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    for k in range(evecs.shape[1]):
        j = np.argmax(np.abs(evecs[:, k]))
        if evecs[j, k] < 0:
            evecs[:, k] *= -1.0

    explained = evals / np.clip(evals.sum(), 1e-12, None)
    comps = evecs.T
    return explained, comps, evals


def compute_pc_variograms(
    df_use,
    period_cols,
    events,
    loadings,
    std_pool,
    bin_size,
    nmax,
    event_col="eqid",
    min_stations=5,
):
    """Compute empirical auto-variograms for PCA score fields."""
    n_components = loadings.shape[1]
    gamma_sum = [np.zeros(nmax) for _ in range(n_components)]
    count_sum = np.zeros(nmax)

    for eqid in events:
        df_event = df_use[df_use[event_col] == eqid]
        if len(df_event) < min_stations:
            continue

        y_event = df_event[period_cols].values.astype(float)
        z_event = (y_event / std_pool[None, :]) @ loadings

        lats = df_event["station_latitude"].values
        lons = df_event["station_longitude"].values
        dist = haversine_km(
            lats[:, None],
            lons[:, None],
            lats[None, :],
            lons[None, :],
        )
        dist_bin = np.round(dist / bin_size).astype(int)

        for b in range(1, nmax + 1):
            mask = dist_bin == b
            npairs = mask.sum()
            if npairs == 0:
                continue

            count_sum[b - 1] += npairs
            idx_i, idx_j = np.where(mask)
            for k in range(n_components):
                diff = z_event[idx_i, k] - z_event[idx_j, k]
                gamma_sum[k][b - 1] += np.sum(diff ** 2) / 2.0

    gamma_pc = []
    for values in gamma_sum:
        out = np.zeros(nmax)
        np.divide(values, count_sum, out=out, where=count_sum > 0)
        gamma_pc.append(out)
    return gamma_pc, count_sum


def prepare_pc_events(
    df_use,
    period_cols,
    events,
    loadings,
    std_pool,
    evals,
    event_col="eqid",
    min_stations=5,
):
    """Build per-component standardized score fields for MLE fitting."""
    n_components = loadings.shape[1]
    score_scales = np.sqrt(np.clip(np.asarray(evals, dtype=float), 1e-12, None))
    event_sets = [[] for _ in range(n_components)]

    for eqid in events:
        df_event = df_use[df_use[event_col] == eqid]
        if len(df_event) < min_stations:
            continue

        y_event = df_event[period_cols].values.astype(float)
        z_event = (y_event / std_pool[None, :]) @ loadings
        z_unit = z_event / score_scales[None, :]

        lats = df_event["station_latitude"].to_numpy(dtype=float)
        lons = df_event["station_longitude"].to_numpy(dtype=float)
        dist = haversine_km(
            lats[:, None],
            lons[:, None],
            lats[None, :],
            lons[None, :],
        )

        for k in range(n_components):
            event_sets[k].append({
                "eqid": eqid,
                "n": len(df_event),
                "y": z_unit[:, k].astype(float),
                "d": dist,
            })

    return event_sets


def fit_pca_workflow_mle(
    df_use,
    period_cols,
    events,
    h_bins,
    h_fine,
    bin_size,
    nmax,
    tail_bins=10,
):
    """Fit the PCA-based spatial correlation reconstruction with MLE on normalized PCs."""
    y_all = df_use[period_cols].values.astype(float)
    cov_pool, corr_pool = corr_matrix_mle(y_all)
    explained, comps, evals = pca_from_corr_mle(corr_pool)

    var_pool = np.diag(cov_pool)
    std_pool = np.sqrt(np.clip(var_pool, 1e-12, None))
    loadings = comps.T
    n_components = loadings.shape[1]

    gamma_pc, count_pc = compute_pc_variograms(
        df_use,
        period_cols,
        events,
        loadings,
        std_pool,
        bin_size,
        nmax,
    )
    pc_events = prepare_pc_events(
        df_use,
        period_cols,
        events,
        loadings,
        std_pool,
        evals,
    )

    pc_sills = np.array([np.mean(g[-tail_bins:]) for g in gamma_pc])
    length_scales = np.zeros(n_components)
    exponents = np.zeros(n_components)
    mle_nll = np.zeros(n_components)

    for k in range(n_components):
        if evals[k] <= 1e-10 or not pc_events[k]:
            length_scales[k], exponents[k], mle_nll[k] = 1.0, 1.0, np.nan
            continue

        fit = fit_pe_pairwise_composite_mle(pc_events[k], cutoff_km=100.0)
        length_scales[k] = fit["LE"]
        exponents[k] = fit["gammaE"]
        mle_nll[k] = fit["nll"]

    rho_pc = np.column_stack([
        powered_exponential_corr(h_fine, length_scales[k], exponents[k])
        for k in range(n_components)
    ])
    rho_pc[0, :] = 1.0

    weights = (loadings ** 2) * evals[None, :]
    cov_th = weights @ rho_pc.T
    rho = cov_th / np.clip(cov_th[:, [0]], 1e-12, None)
    le_eff = effective_ranges(rho, h_fine)

    return {
        "Y_all": y_all,
        "C_pool": cov_pool,
        "R_pool": corr_pool,
        "explained": explained,
        "components": comps,
        "evals": evals,
        "var_pool": var_pool,
        "std_pool": std_pool,
        "U": loadings,
        "K": n_components,
        "gamma_pc": gamma_pc,
        "count_pc": count_pc,
        "pc_sills": pc_sills,
        "LE": length_scales,
        "gammaE": exponents,
        "mle_nll": mle_nll,
        "rho_pc": rho_pc,
        "rho": rho,
        "le_eff": le_eff,
    }


def pca_cross_correlation_mle(i, j, h_arr, loadings, var_z, rho_pc):
    """Cross-period correlation from a PCA reconstruction."""
    del h_arr
    w_ij = loadings[i, :] * loadings[j, :] * var_z
    cov_h = rho_pc @ w_ij

    w_ii = (loadings[i, :] ** 2) * var_z
    w_jj = (loadings[j, :] ** 2) * var_z
    return cov_h / np.sqrt(np.clip(w_ii.sum() * w_jj.sum(), 1e-12, None))


def pca_cross_corr_at_h_mle(i, j, h, loadings, var_z, length_scales, exponents):
    """Cross-period PCA correlation at one distance."""
    rho_pc_h = np.array([
        powered_exponential_corr(h, length_scales[k], exponents[k])
        for k in range(len(length_scales))
    ])
    if h == 0:
        rho_pc_h[:] = 1.0

    w_ij = loadings[i, :] * loadings[j, :] * var_z
    w_ii = (loadings[i, :] ** 2) * var_z
    w_jj = (loadings[j, :] ** 2) * var_z
    return float(w_ij @ rho_pc_h) / np.sqrt(np.clip(w_ii.sum() * w_jj.sum(), 1e-12, None))



# Semivariogram fitting


def corr_matrix_semivariogram(y):
    """Return covariance and correlation matrices for column data."""
    cov = (y.T @ y) / len(y)
    scale = np.sqrt(np.diag(cov))
    corr = cov / np.outer(scale, scale)
    return cov, corr


def pca_from_corr_semivariogram(corr):
    """Eigen-decompose a correlation matrix with deterministic signs."""
    evals, evecs = np.linalg.eigh(corr)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    for k in range(evecs.shape[1]):
        j = np.argmax(np.abs(evecs[:, k]))
        if evecs[j, k] < 0:
            evecs[:, k] *= -1.0

    explained = evals / evals.sum()
    comps = evecs.T
    return explained, comps, evals


def fit_pca_workflow_semivariogram(
    df_use,
    period_cols,
    events,
    h_bins,
    h_fine,
    bin_size,
    nmax,
    tail_bins=10,
):
    """Fit the paper's PCA reconstruction with nugget-free PE latent kernels."""
    y_all = df_use[period_cols].values.astype(float)
    cov_pool, corr_pool = corr_matrix_semivariogram(y_all)
    explained, comps, evals = pca_from_corr_semivariogram(corr_pool)

    var_pool = np.diag(cov_pool)
    std_pool = np.sqrt(var_pool)
    loadings = comps.T
    n_components = loadings.shape[1]

    gamma_pc, count_pc = compute_pc_variograms(
        df_use,
        period_cols,
        events,
        loadings,
        std_pool,
        bin_size,
        nmax,
    )

    pc_sills = np.array([np.mean(g[-tail_bins:]) for g in gamma_pc])
    length_scales = np.zeros(n_components)
    exponents = np.zeros(n_components)
    for k in range(n_components):
        if pc_sills[k] > 0.01:
            length_scales[k], exponents[k] = fit_pe_wls(
                gamma_pc[k],
                h_bins,
                pc_sills[k],
            )
        else:
            length_scales[k], exponents[k] = 1.0, 1.0

    rho_pc = np.exp(
        -(
            np.maximum(h_fine[:, None], 1e-12)
            / length_scales[None, :]
        ) ** exponents[None, :]
    )
    rho_pc[0, :] = 1.0

    weights = (loadings ** 2) * evals[None, :]
    cov_th = weights @ rho_pc.T
    rho = cov_th / cov_th[:, [0]]
    le_eff = effective_ranges(rho, h_fine)

    return {
        "Y_all": y_all,
        "C_pool": cov_pool,
        "R_pool": corr_pool,
        "explained": explained,
        "components": comps,
        "evals": evals,
        "var_pool": var_pool,
        "std_pool": std_pool,
        "U": loadings,
        "K": n_components,
        "gamma_pc": gamma_pc,
        "count_pc": count_pc,
        "pc_sills": pc_sills,
        "LE": length_scales,
        "gammaE": exponents,
        "rho_pc": rho_pc,
        "rho": rho,
        "le_eff": le_eff,
    }


def pca_cross_correlation_semivariogram(i, j, h_arr, loadings, var_z, rho_pc):
    """Cross-period correlation from a PCA reconstruction."""
    del h_arr
    w_ij = loadings[i, :] * loadings[j, :] * var_z
    cov_h = rho_pc @ w_ij

    w_ii = (loadings[i, :] ** 2) * var_z
    w_jj = (loadings[j, :] ** 2) * var_z
    return cov_h / np.sqrt(w_ii.sum() * w_jj.sum())


def pca_cross_corr_at_h_semivariogram(
    i,
    j,
    h,
    loadings,
    var_z,
    length_scales,
    exponents,
    nugget_fraction=None,
):
    """Cross-period PCA correlation at one distance."""
    rho_pc_h = np.exp(
        -(
            np.maximum(h, 1e-12)
            / length_scales
        ) ** exponents
    )
    if nugget_fraction is not None:
        rho_pc_h *= 1.0 - np.asarray(nugget_fraction, dtype=float)
    if h == 0:
        rho_pc_h[:] = 1.0

    w_ij = loadings[i, :] * loadings[j, :] * var_z
    w_ii = (loadings[i, :] ** 2) * var_z
    w_jj = (loadings[j, :] ** 2) * var_z
    return float(w_ij @ rho_pc_h) / np.sqrt(w_ii.sum() * w_jj.sum())
