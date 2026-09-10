from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

from scripts.analysis.core import comparison_data_from_state
from scripts.analysis.semivariograms import (
    build_curves,
    covariance_to_semivariogram,
    lmc_covariance_curves,
    pca_covariance_curves,
)


DEFAULT_CACHE = PROJECT_ROOT / "results/models.pkl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/analysis/wls.csv"


def _load_payload(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


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
    """Apply the original Pairwise objective to all ten fitted models.

    The objective is

        sum_{i=1}^9 sum_{j=1}^9 sum_b
            (1 / h_b) [gamma_emp,ij(h_b) - gamma_model,ij(h_b)]^2.

    Thus every off-diagonal period pair is counted twice, exactly as in
    ``scripts/models/pairwise.py``.
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

    rows.sort(key=lambda row: float(row["wls_pairwise_definition_81"]))
    pairwise_wls = next(
        float(row["wls_pairwise_definition_81"])
        for row in rows
        if row["method"] == "Pairwise semivariogram"
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["ratio_to_pairwise"] = (
            float(row["wls_pairwise_definition_81"]) / pairwise_wls
        )

    cached_pairwise_wls = float(payload["pairwise_empirical"]["fit"]["wss"])
    if not np.isclose(pairwise_wls, cached_pairwise_wls, rtol=1e-12, atol=1e-14):
        raise RuntimeError(
            "Uniform WLS does not reproduce the original Pairwise objective: "
            f"{pairwise_wls} versus {cached_pairwise_wls}"
        )
    return rows


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "method",
        "wls_pairwise_definition_81",
        "auto_wls_9",
        "ordered_cross_wls_72",
        "unique_lower_triangle_wls_45",
        "ratio_to_pairwise",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the original Pairwise empirical-semivariogram WLS to ten models."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = _load_payload(args.cache.resolve())
    rows = calculate_wls(payload)
    write_csv(rows, args.output.resolve())

    print(
        "rank  method                              WLS(81)       auto(9)       cross(72)    ratio"
    )
    for row in rows:
        print(
            f"{int(row['rank']):>4}  {str(row['method']):<35} "
            f"{float(row['wls_pairwise_definition_81']):>12.9f} "
            f"{float(row['auto_wls_9']):>12.9f} "
            f"{float(row['ordered_cross_wls_72']):>12.9f} "
            f"{float(row['ratio_to_pairwise']):>8.3f}"
        )
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
