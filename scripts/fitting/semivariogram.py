"""Weighted semivariogram fitting for the eight paper model families.

Run all: python -m scripts.fitting.semivariogram
Run one: python -m scripts.fitting.semivariogram --models gh08
"""
import os

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.optimize import minimize


def fit_pe_wls(gamma_data, h_bins, sill_est, initial=(20.0, 0.4)):
    """Fit a powered-exponential variogram by weighted least squares."""
    gamma_data = np.asarray(gamma_data, dtype=float)
    h_bins = np.asarray(h_bins, dtype=float)

    def cost(params):
        length_scale, exponent = params
        pred = sill_est * (1.0 - np.exp(-((h_bins / length_scale) ** exponent)))
        return np.sum((1.0 / h_bins) * (gamma_data - pred) ** 2)

    result = minimize(
        cost,
        initial,
        bounds=[(1.0, 200.0), (0.05, 1.5)],
        method="L-BFGS-B",
    )
    return result.x


def fit_pe_wls_nugget(
    gamma_data,
    h_bins,
    sill_est,
    initial=(20.0, 0.4, 0.1),
):
    """Fit a nugget + powered-exponential variogram by weighted least squares.

    The fitted model for positive lags is

        sill * [f + (1 - f) * (1 - exp(-(h / ell) ** p))],

    where ``f`` is the nugget fraction of the total sill.  The semivariogram is
    still exactly zero at lag zero; the discontinuity is represented later as
    an observation-level identity covariance component.
    """
    gamma_data = np.asarray(gamma_data, dtype=float)
    h_bins = np.asarray(h_bins, dtype=float)
    sill_est = float(sill_est)
    valid = (
        np.isfinite(gamma_data)
        & np.isfinite(h_bins)
        & (h_bins > 0.0)
    )
    if not bool(valid.any()) or not np.isfinite(sill_est) or sill_est <= 0.0:
        return np.asarray(initial, dtype=float)

    gamma_use = gamma_data[valid]
    h_use = h_bins[valid]

    def cost(params):
        length_scale, exponent, nugget_fraction = params
        structured = np.exp(-((h_use / length_scale) ** exponent))
        pred = sill_est * (1.0 - (1.0 - nugget_fraction) * structured)
        return np.sum((1.0 / h_use) * (gamma_use - pred) ** 2)

    first_fraction = float(np.clip(gamma_use[0] / sill_est, 0.0, 0.95))
    starts = [
        np.asarray(initial, dtype=float),
        np.asarray([initial[0], initial[1], first_fraction], dtype=float),
        np.asarray([10.0, 0.7, 0.0], dtype=float),
        np.asarray([40.0, 0.7, 0.35], dtype=float),
    ]
    bounds = [(1.0, 200.0), (0.05, 1.5), (0.0, 0.999)]
    results = [
        minimize(cost, start, bounds=bounds, method="L-BFGS-B")
        for start in starts
    ]
    finite = [result for result in results if np.isfinite(result.fun)]
    if not finite:
        return np.asarray(initial, dtype=float)
    return min(finite, key=lambda result: result.fun).x


def main():
    from scripts.refit import run_cli
    run_cli("semivariogram", ("pairwise", "gh08", "separable", "pca", "lmc", "iox", "sbss", "matern"))


if __name__ == "__main__":
    main()
