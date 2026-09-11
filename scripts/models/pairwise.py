"""Direct pairwise powered-exponential fits for cross-variograms."""

import numpy as np

from scripts.fitting.semivariogram import fit_pe_wls


def fit_pairwise_pe(gamma_emp, h_bins, tail_bins=10):
    """Fit each gamma_ij(h) independently with a fixed empirical sill."""
    nper = len(gamma_emp)
    length_scales = np.zeros((nper, nper))
    exponents = np.zeros((nper, nper))
    sills = np.zeros((nper, nper))
    wss = 0.0

    for i in range(nper):
        for j in range(i, nper):
            sill = float(np.mean(gamma_emp[i][j][-tail_bins:]))
            le_fit, ge_fit = fit_pe_wls(gamma_emp[i][j], h_bins, sill)

            length_scales[i, j] = le_fit
            exponents[i, j] = ge_fit
            sills[i, j] = sill
            length_scales[j, i] = le_fit
            exponents[j, i] = ge_fit
            sills[j, i] = sill

            pred = sill * (1.0 - np.exp(-((h_bins / le_fit) ** ge_fit)))
            wss += float(np.sum((1.0 / h_bins) * (gamma_emp[i][j] - pred) ** 2))

    return {
        "LE": length_scales,
        "gammaE": exponents,
        "sills": sills,
        "wss": wss,
    }


def pairwise_variogram_ij(h, i, j, pairwise_fit):
    """Evaluate a fitted direct pairwise PE cross-variogram."""
    sill = pairwise_fit["sills"][i, j]
    le = pairwise_fit["LE"][i, j]
    ge = pairwise_fit["gammaE"][i, j]
    return sill * (1.0 - np.exp(-((h / le) ** ge)))
