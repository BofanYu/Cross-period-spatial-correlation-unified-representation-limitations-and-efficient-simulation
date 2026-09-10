"""Separable cross-period correlation with one fitted spatial kernel."""
import numpy as np
from scipy.optimize import minimize
from scripts.data import effective_ranges, empirical_sills, powered_exponential_corr
from scripts.fitting.mle import fit_pe_pairwise_composite_mle, prepare_univariate_events
from scripts.fitting.semivariogram import fit_pe_wls



# Likelihood fitting


def fit_separable_kernel_mle(
    period_dfs,
    gamma_emp,
    h_bins,
    h_fine,
    tail_bins=10,
    initial=(20.0, 0.4),
    min_stations=5,
):
    """Fit one common powered-exponential kernel for all periods by MLE."""
    sills = empirical_sills(gamma_emp, tail_bins=tail_bins)

    all_events = []
    for df_period in period_dfs:
        events, _ = prepare_univariate_events(df_period, min_stations=min_stations)
        all_events.extend(events)

    fit = fit_pe_pairwise_composite_mle(all_events, initial=initial, cutoff_km=100.0)
    length_scale = fit["LE"]
    exponent = fit["gammaE"]

    kernel = 1.0 - np.exp(-((h_bins / length_scale) ** exponent))
    wss = 0.0
    for i in range(len(gamma_emp)):
        pred = sills[i] * kernel
        wss += float(np.sum((1.0 / h_bins) * (gamma_emp[i][i] - pred) ** 2))

    rho_common = powered_exponential_corr(h_fine, length_scale, exponent)
    rho_common[0] = 1.0
    rho = np.tile(rho_common, (len(gamma_emp), 1))
    le_eff = effective_ranges(rho_common, h_fine)

    return {
        "LE": float(length_scale),
        "gammaE": float(exponent),
        "nll": float(fit["nll"]),
        "rho_common": rho_common,
        "rho": rho,
        "le_eff": float(le_eff),
        "wss": float(wss),
        "emp_sills": sills,
    }


def separable_matrix_mle(h, length_scale, exponent, rho_0):
    """Pure separable correlation matrix rho(h) = rho_0 * c(h)."""
    kernel = powered_exponential_corr(h, length_scale, exponent)
    return kernel * rho_0


def separable_psd_check_mle(h_grid, length_scale, exponent, rho_0):
    """Check the pure separable model over a distance grid."""
    pd_ok = True
    min_eval = np.inf
    worst_h = 0.0
    worst_mat = None

    for h in h_grid:
        rho_mat = separable_matrix_mle(h, length_scale, exponent, rho_0)
        evals = np.linalg.eigvalsh(0.5 * (rho_mat + rho_mat.T))
        current_min = evals.min()
        if current_min < min_eval:
            min_eval = current_min
            worst_h = h
            worst_mat = rho_mat.copy()
        if current_min < -1e-10:
            pd_ok = False

    return {
        "pd_ok": pd_ok,
        "min_eval": min_eval,
        "worst_h": worst_h,
        "worst_mat": worst_mat,
    }


def separable_variogram_ij_mle(h, i, j, length_scale, exponent, sills, rho_0):
    """Cross-variogram under the pure separable kernel."""
    kernel = powered_exponential_corr(h, length_scale, exponent)
    corr_0 = 1.0 if i == j else rho_0[i, j]
    sill_ij = np.sqrt(sills[i] * sills[j]) * corr_0
    return sill_ij * (1.0 - kernel)


def separable_cross_corr_mle(i, j, h_arr, length_scale, exponent, rho_0):
    """Cross-period correlation under the pure separable kernel."""
    kernel = powered_exponential_corr(h_arr, length_scale, exponent)
    corr_0 = 1.0 if i == j else rho_0[i, j]
    return corr_0 * kernel



# Semivariogram fitting


def fit_separable_kernel_semivariogram(gamma_emp, h_bins, h_fine, tail_bins=10, initial=(20.0, 0.4)):
    """Fit one common powered-exponential kernel for all periods."""
    nper = len(gamma_emp)
    sills = empirical_sills(gamma_emp, tail_bins=tail_bins)

    def cost(params):
        length_scale, exponent = params
        kernel = 1.0 - np.exp(-((h_bins / length_scale) ** exponent))
        total = 0.0
        for i in range(nper):
            pred = sills[i] * kernel
            total += float(np.sum((1.0 / h_bins) * (gamma_emp[i][i] - pred) ** 2))
        return total

    result = minimize(
        cost,
        initial,
        bounds=[(1.0, 200.0), (0.05, 1.5)],
        method="L-BFGS-B",
    )
    length_scale, exponent = result.x

    rho_common = np.exp(-((h_fine / length_scale) ** exponent))
    rho_common[0] = 1.0
    rho = np.tile(rho_common, (nper, 1))
    le_eff = effective_ranges(rho_common, h_fine)

    return {
        "LE": float(length_scale),
        "gammaE": float(exponent),
        "rho_common": rho_common,
        "rho": rho,
        "le_eff": float(le_eff),
        "wss": float(cost((length_scale, exponent))),
        "emp_sills": sills,
    }


def separable_matrix_semivariogram(h, length_scale, exponent, rho_0):
    """Pure separable correlation matrix rho(h) = rho_0 * c(h)."""
    kernel = np.exp(-((h / length_scale) ** exponent))
    return kernel * rho_0


def separable_psd_check_semivariogram(h_grid, length_scale, exponent, rho_0):
    """Check the pure separable model over a distance grid."""
    pd_ok = True
    min_eval = np.inf
    worst_h = 0.0
    worst_mat = None

    for h in h_grid:
        rho_mat = separable_matrix_semivariogram(h, length_scale, exponent, rho_0)
        evals = np.linalg.eigvalsh(0.5 * (rho_mat + rho_mat.T))
        current_min = evals.min()
        if current_min < min_eval:
            min_eval = current_min
            worst_h = h
            worst_mat = rho_mat.copy()
        if current_min < -1e-10:
            pd_ok = False

    return {
        "pd_ok": pd_ok,
        "min_eval": min_eval,
        "worst_h": worst_h,
        "worst_mat": worst_mat,
    }


def separable_variogram_ij_semivariogram(h, i, j, length_scale, exponent, sills, rho_0):
    """Cross-variogram under the pure separable kernel."""
    kernel = np.exp(-((h / length_scale) ** exponent))
    corr_0 = 1.0 if i == j else rho_0[i, j]
    sill_ij = np.sqrt(sills[i] * sills[j]) * corr_0
    return sill_ij * (1.0 - kernel)


def separable_cross_corr_semivariogram(i, j, h_arr, length_scale, exponent, rho_0):
    """Cross-period correlation under the pure separable kernel."""
    kernel = np.exp(-((h_arr / length_scale) ** exponent))
    corr_0 = 1.0 if i == j else rho_0[i, j]
    return corr_0 * kernel
