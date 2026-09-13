from __future__ import annotations

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

from scripts.common.core import comparison_data_from_state
from scripts.common.semivariograms import (
    build_curves,
    covariance_to_semivariogram,
    lmc_covariance_curves,
    pca_covariance_curves,
)






def _model_semivariograms(payload: dict, data) -> dict[str, np.ndarray]:
    """Evaluate every fitted semivariogram on the original 2--100 km bins."""
    h_bins = np.asarray(data.h_bins, dtype=float)
    h_with_zero = np.r_[0.0, h_bins]

    five_curves, _ = build_curves(payload, data, h_with_zero)
    curves = {
        method: np.asarray(values[:, :, 1:], dtype=float)
        for method, values in five_curves.items()
    }

    pca_mle_covariance = pca_covariance_curves(payload["pca"]["mle"], h_with_zero)
    curves["PCA MLE"] = covariance_to_semivariogram(pca_mle_covariance)[:, :, 1:]

    lmc_mle_covariance = lmc_covariance_curves(payload["lmc_block_mle"], h_with_zero)
    curves["LMC block MLE"] = covariance_to_semivariogram(lmc_mle_covariance)[:, :, 1:]

    # IOX is event/order dependent, so its native fit supplies the station-pair-
    # pooled value for every distance bin. SBSS and Matérn also retain their
    # native fitted semivariogram arrays, avoiding any likelihood-time rescaling.
    curves["IOX full semivariogram"] = np.asarray(
        payload["iox_full_semivariogram"]["fitted_gamma"], dtype=float
    )
    curves["SBSS semivariogram"] = np.asarray(
        payload["sbss_semivariogram"]["fitted_gamma"], dtype=float
    )
    curves["Multivariate Matern semivariogram"] = np.asarray(
        payload["multivariate_matern"]["fitted_gamma"], dtype=float
    )
    return curves


def calculate_wls(payload: dict) -> list[dict[str, float | str]]:
    """Sum 45 unique period-pair targets (9 direct + 36 cross), weighted by 1/h.

    The 81-entry sum is retained only as a diagnostic, not the paper objective.
    """
    data = comparison_data_from_state(payload["data_state"])
    empirical = np.asarray(data.gamma_emp, dtype=float)
    h_bins = np.asarray(data.h_bins, dtype=float)
    curves = _model_semivariograms(payload, data)

    q = len(data.periods)
    expected_shape = (q, q, len(h_bins))
    diagonal = np.eye(q, dtype=bool)
    off_diagonal = ~diagonal
    rows = []

    for method, fitted in curves.items():
        if fitted.shape != expected_shape:
            raise ValueError(
                f"{method} has shape {fitted.shape}; expected {expected_shape}"
            )
        contributions = (empirical - fitted) ** 2 / h_bins[None, None, :]
        auto_wls = float(contributions[diagonal].sum())
        ordered_cross_wls = float(contributions[off_diagonal].sum())
        rows.append(
            {
                "method": method,
                "wls_pairwise_definition_81": auto_wls + ordered_cross_wls,
                "auto_wls_9": auto_wls,
                "ordered_cross_wls_72": ordered_cross_wls,
                "unique_lower_triangle_wls_45": auto_wls + 0.5 * ordered_cross_wls,
            }
        )

    rows.sort(key=lambda row: float(row["unique_lower_triangle_wls_45"]))
    pairwise_wls = next(
        float(row["unique_lower_triangle_wls_45"])
        for row in rows
        if row["method"] == "Pairwise semivariogram"
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["ratio_to_pairwise"] = (
            float(row["unique_lower_triangle_wls_45"]) / pairwise_wls
        )

    cached_pairwise_wls = float(payload["pairwise_empirical"]["fit"]["wss"])
    if not np.isclose(pairwise_wls, cached_pairwise_wls, rtol=1e-12, atol=1e-14):
        raise RuntimeError(
            "Uniform WLS does not reproduce the original Pairwise objective: "
            f"{pairwise_wls} versus {cached_pairwise_wls}"
        )
    return rows
