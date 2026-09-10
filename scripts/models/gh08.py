"""GH08 correlation model and its marginal powered-exponential fits."""
import numpy as np
from scripts.data import effective_ranges, empirical_sills, powered_exponential_corr
from scripts.fitting.mle import fit_pe_pairwise_composite_mle, prepare_univariate_events
from scripts.fitting.semivariogram import fit_pe_wls



# Likelihood fitting


def fit_direct_auto_pe_mle(period_dfs, gamma_emp, h_bins, h_fine, tail_bins=10, min_stations=5):
    """Fit direct same-period powered-exponential correlations by MLE on standardized fields."""
    del h_bins
    nper = len(gamma_emp)
    sills = empirical_sills(gamma_emp, tail_bins=tail_bins)

    length_scales = np.zeros(nper)
    exponents = np.zeros(nper)
    nll = np.zeros(nper)

    for i, df_period in enumerate(period_dfs):
        events, _ = prepare_univariate_events(df_period, min_stations=min_stations)
        fit = fit_pe_pairwise_composite_mle(events, cutoff_km=100.0)
        length_scales[i] = fit["LE"]
        exponents[i] = fit["gammaE"]
        nll[i] = fit["nll"]

    rho = np.zeros((nper, len(h_fine)))
    for i in range(nper):
        rho[i, :] = powered_exponential_corr(h_fine, length_scales[i], exponents[i])
        rho[i, 0] = 1.0

    return {
        "emp_sills": sills,
        "LE": length_scales,
        "gammaE": exponents,
        "nll": nll,
        "rho": rho,
        "le_eff": effective_ranges(rho, h_fine),
    }


def goda_hong_matrix_mle(h, le_direct, ge_direct, rho_0):
    """GH08-style correlation matrix using the larger-period auto correlation."""
    nper = len(le_direct)
    rho_mat = np.zeros((nper, nper))

    for i in range(nper):
        for j in range(nper):
            kmax = max(i, j)
            rho_auto = powered_exponential_corr(h, le_direct[kmax], ge_direct[kmax])
            if i == j:
                rho_mat[i, j] = rho_auto
            else:
                rho_mat[i, j] = rho_0[i, j] * rho_auto

    return 0.5 * (rho_mat + rho_mat.T)


def goda_hong_psd_check_mle(h_grid, le_direct, ge_direct, rho_0):
    """Check the GH08-style matrix over a distance grid."""
    pd_ok = True
    min_eval = np.inf
    worst_h = 0.0
    worst_mat = None

    for h in h_grid:
        rho_mat = goda_hong_matrix_mle(h, le_direct, ge_direct, rho_0)
        evals = np.linalg.eigvalsh(rho_mat)
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


def goda_hong_variogram_ij_mle(h, i, j, le_direct, ge_direct, sills, rho_0):
    """GH08-style cross-variogram for one period pair and distance."""
    if i == j:
        rho_auto = powered_exponential_corr(h, le_direct[i], ge_direct[i])
        return sills[i] * (1.0 - rho_auto)

    kmax = max(i, j)
    rho_auto = powered_exponential_corr(h, le_direct[kmax], ge_direct[kmax])
    sill_ij = np.sqrt(sills[i] * sills[j]) * rho_0[i, j]
    return sill_ij * (1.0 - rho_auto)



# Semivariogram fitting


def fit_direct_auto_pe_semivariogram(gamma_emp, h_bins, h_fine, tail_bins=10):
    """Fit direct same-period powered-exponential variograms."""
    nper = len(gamma_emp)
    sills = empirical_sills(gamma_emp, tail_bins=tail_bins)

    length_scales = np.zeros(nper)
    exponents = np.zeros(nper)
    for i in range(nper):
        length_scales[i], exponents[i] = fit_pe_wls(
            gamma_emp[i][i],
            h_bins,
            sills[i],
        )

    rho = np.zeros((nper, len(h_fine)))
    for i in range(nper):
        rho[i, :] = np.exp(-((h_fine / length_scales[i]) ** exponents[i]))
        rho[i, 0] = 1.0

    return {
        "emp_sills": sills,
        "LE": length_scales,
        "gammaE": exponents,
        "rho": rho,
        "le_eff": effective_ranges(rho, h_fine),
    }


def goda_hong_matrix_semivariogram(h, le_direct, ge_direct, rho_0):
    """GH08-style correlation matrix using the larger-period auto correlation."""
    nper = len(le_direct)
    rho_mat = np.zeros((nper, nper))

    for i in range(nper):
        for j in range(nper):
            kmax = max(i, j)
            rho_auto = np.exp(-((h / le_direct[kmax]) ** ge_direct[kmax]))
            if i == j:
                rho_mat[i, j] = rho_auto
            else:
                rho_mat[i, j] = rho_0[i, j] * rho_auto

    return 0.5 * (rho_mat + rho_mat.T)


def goda_hong_psd_check_semivariogram(h_grid, le_direct, ge_direct, rho_0):
    """Check the GH08-style matrix over a distance grid."""
    pd_ok = True
    min_eval = np.inf
    worst_h = 0.0
    worst_mat = None

    for h in h_grid:
        rho_mat = goda_hong_matrix_semivariogram(h, le_direct, ge_direct, rho_0)
        evals = np.linalg.eigvalsh(rho_mat)
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


def goda_hong_variogram_ij_semivariogram(h, i, j, le_direct, ge_direct, sills, rho_0):
    """GH08-style cross-variogram for one period pair and distance."""
    if i == j:
        rho_auto = np.exp(-((h / le_direct[i]) ** ge_direct[i]))
        return sills[i] * (1.0 - rho_auto)

    kmax = max(i, j)
    rho_auto = np.exp(-((h / le_direct[kmax]) ** ge_direct[kmax]))
    sill_ij = np.sqrt(sills[i] * sills[j]) * rho_0[i, j]
    return sill_ij * (1.0 - rho_auto)
