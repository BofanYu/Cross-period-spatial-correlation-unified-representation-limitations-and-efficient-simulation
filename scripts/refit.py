"""Shared refit orchestration used by the two fitting entry points.

All model fits are stored together in results/models.pkl. Selecting a subset
updates those fits and retains the remaining archived models.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from scripts.analysis import core as mch
from scripts.models import iox, sbss, matern

ROOT = Path(__file__).resolve().parents[1]


def fit_iox(data):
    _, events, _ = iox.load_union_panel(ROOT / "data/periods")
    empirical = iox.empirical_event_semivariograms(events, bin_size_km=2.0, max_distance_km=100.0)
    fit = iox.fit_iox_wls(empirical, events, tail_bins=10, bin_size_km=2.0, max_distance_km=100.0)
    h = data.h_fine
    basis, scaling, counts = iox.iox_semivariogram_basis(
        events, fit.marginal, bin_size_km=float(h[1] - h[0]), max_distance_km=float(h[-1])
    )
    fitted_gamma = fit.sigma[:, :, None] * basis
    denominator = np.sqrt(np.outer(np.diag(fit.sigma), np.diag(fit.sigma)))
    curves = np.empty((len(data.periods), len(data.periods), len(h)))
    curves[:, :, 0] = fit.rho_colocated
    curves[:, :, 1:] = fit.rho_colocated[:, :, None] - fitted_gamma / denominator[:, :, None]
    curves = 0.5 * (curves + curves.swapaxes(0, 1))
    result = dict(
        kernel_family="powered_exponential",
        periods=fit.periods, h_bins=empirical.h, empirical_gamma=empirical.gamma,
        fitted_gamma=fit.fitted_gamma, station_pair_counts=empirical.counts,
        marginal_parameters=pd.DataFrame(fit.marginal.optimizer_rows),
        pair_parameters=iox.pair_parameter_frame(fit), sigma_core=fit.sigma,
        rho_core=fit.rho_core, iox_colocated_scaling=scaling,
        rho_colocated_average=fit.rho_colocated, raw_pairwise_rho=fit.raw_pairwise_rho,
        comparison_h_grid=h, comparison_unit_sigma_basis=basis,
        comparison_fitted_gamma=fitted_gamma, comparison_station_pair_counts=counts,
        correlation_curves=(h, curves), optimizer_diagnostics=fit.optimizer,
    )
    return result, (h, curves)


def fit_sbss(data):
    common = sbss.load_pca_common_data(ROOT / "data/common_records.csv")
    fit = sbss.fit_sbss(common, rings_km=sbss.RINGS_KM)
    h = data.h_fine
    curves = np.asarray([[np.interp(h, fit.h_fine, curve) for curve in row]
                         for row in fit.correlation_fine])
    # Native WLS arrays use the original 2--100 km empirical bins.
    fitted_gamma = np.asarray([[np.interp(fit.empirical_observed.h_bins, fit.h_fine, curve)
                               for curve in row] for row in fit.semivariogram_fine])
    result = dict(
        periods=fit.periods, h_bins=fit.empirical_observed.h_bins,
        empirical_gamma=fit.empirical_observed.gamma, fitted_gamma=fitted_gamma,
        ordered_station_pair_counts=fit.empirical_observed.counts,
        latent_parameters=pd.DataFrame(fit.latent.optimizer_rows),
        latent_scores=fit.latent.scores, mixing_raw=fit.mixing_raw,
        mixing_standardized=fit.mixing_standardized,
        unmixing_standardized=fit.unmixing_standardized,
        scatter_zero_correlation=fit.scatter_zero,
        joint_diagonalization_diagnostics=fit.joint_diagnostics,
        correlation_curves=(h, curves),
    )
    return result, (h, curves)


def fit_matern(data):
    periods, frames, summary = matern.load_period_datasets(ROOT / "data/periods")
    empirical = matern.pool_pairwise_event_semivariograms(periods, frames, summary)
    fit = matern.fit_flexible_matern(empirical, tail_bins=10, spatial_dimension=2)
    h = np.linspace(0.0, 100.0, 1001)
    curves = np.asarray([[fit.rho[i, j] * matern.matern_correlation(h, fit.nu[i, j], fit.alpha[i, j])
                          for j in range(len(periods))] for i in range(len(periods))])
    fitted_gamma = np.asarray([[matern.theoretical_semivariogram(empirical.h_bins, fit.sigma[i, j], fit.nu[i, j], fit.alpha[i, j])
                                for j in range(len(periods))] for i in range(len(periods))])
    result = dict(
        periods=periods, h_bins=empirical.h_bins, empirical_gamma=empirical.gamma,
        fitted_gamma=fitted_gamma, station_pair_counts=empirical.counts,
        parameters=matern.parameter_long_frame(empirical, fit), sigma=fit.sigma,
        rho_fitted=fit.rho, rho_empirical=fit.rho_empirical,
        alpha_per_km=fit.alpha, nu=fit.nu, R_A=fit.r_a, R_B=fit.r_b,
        R_V=fit.r_v, V_theorem_matrix=fit.v_matrix,
        marginal_optimizer_diagnostics=fit.marginal.optimizer_rows,
        joint_cross_optimizer_diagnostics=fit.joint_optimizer,
        correlation_curves=(h, curves),
    )
    return result, (h, curves)


def fit_lmc_mle(data, mle):
    """The paper's aligned cap80 -> full cap80 -> full all-sites workflow."""
    options = dict(ridge=1e-6, ftol=1e-7, gtol=1e-4, maxls=60, maxcor=20,
                   diag_lower=1e-5, diag_upper=10.0, offdiag_bound=10.0, random_state=123)
    wide = mch.predictive_full_data_wide(data)

    def report(stage, fit):
        print(f"LMC {stage}: objective={fit['nll']:.9g}; {fit['message']}", flush=True)
        return {key: value for key, value in fit.items() if key not in ("blocks", "result")}

    print("LMC stage 1/3: aligned cap80", flush=True)
    aligned = report("aligned cap80", mch.fit_lmc_mle_block(
        data, mle, max_stations_per_event=80, maxiter=250, init=None, **options))
    print("LMC stage 2/3: full cap80", flush=True)
    capped = report("full cap80", mch.fit_lmc_mle_block_full_data(
        data, mle, wide=wide, max_stations_per_event=80, maxiter=250,
        init=aligned, **options))
    print("LMC stage 3/3: full all-sites", flush=True)
    return report("full all-sites", mch.fit_lmc_mle_block_full_data(
        data, mle, wide=wide, max_stations_per_event=None, maxiter=150,
        init=capped, **options))


def fit_model(data, family, method, payload, mle, semi):
    """Update one model in the shared archive and return its plotted curves."""
    h = data.h_fine
    if family == "pairwise":
        result = mch.fit_pairwise_empirical(data, semi["pairwise"])
        payload["pairwise_empirical"] = result
        return "Pairwise empirical semivariogram", (h, mch.pairwise_fitted_corr_curves(data, result["fit"], h))
    if family == "gh08":
        result = semi["product"].fit_direct_auto_pe(data.gamma_emp, data.h_bins, h, tail_bins=10)
        curves = mch.goda_hong_curves(result, data.rho_0, h)
        payload["gh08"] = {"direct_semi": result, "gh_curves_semi": curves}
        return "GH08 semivariogram", (h, curves)
    if family == "separable":
        result = semi["product"].fit_separable_kernel(data.gamma_emp, data.h_bins, h, tail_bins=10)
        curves = mch.separable_curves(result, data.rho_0, h)
        payload["kronecker"] = {"semivariogram": result, "curves_semivariogram": curves,
                                "summary": pd.DataFrame()}
        return "Kronecker semivariogram", (h, curves)
    if family == "pca":
        if method == "mle":
            result = mch.fit_pca_full_mle(data, mle, mle["common"])
            model = mle["pca"]
            label = "PCA MLE"
        else:
            result = semi["pca"].fit_pca_workflow(
                data.df_pca, data.period_cols, data.events_pca, data.h_bins,
                h, data.bin_size, len(data.h_bins), tail_bins=10)
            model = semi["pca"]
            label = "PCA semivariogram"
        payload.setdefault("pca", {})[method] = result
        payload["pca"]["summary"] = pd.DataFrame()
        return label, (h, mch.pca_period_pair_curves(result, model, h))
    if family == "lmc":
        if method == "mle":
            result = fit_lmc_mle(data, mle)
            payload["lmc_block_mle"] = {
                key: value for key, value in result.items()
                if key not in ("result", "history", "events", "blocks")}
            label = "LMC block MLE"
        else:
            result = mch.fit_lmc_semivariogram(data, semi, n_iterations=30)
            payload["lmc_semivariogram"] = result
            label = "LMC semivariogram"
        return label, (h, mch.lmc_period_pair_curves(result, semi["lmc"], h))
    label, key, function = {
        "iox": ("IOX full semivariogram", "iox_full_semivariogram", fit_iox),
        "sbss": ("SBSS semivariogram", "sbss_semivariogram", fit_sbss),
        "matern": ("Multivariate Matern semivariogram", "multivariate_matern", fit_matern),
    }[family]
    payload[key], curves = function(data)
    return label, curves


def run_cli(method, families, default=None):
    parser = argparse.ArgumentParser(description=f"Refit paper models by {method}.")
    parser.add_argument("--models", nargs="+", choices=families,
                        default=list(default or families), metavar="MODEL")
    parser.add_argument("--output", type=Path, default=ROOT / "results/models.pkl",
                        help="Updated archive; unselected models retain their saved fits.")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/models.pkl",
                        help="Model archive to update and compare against.")
    args = parser.parse_args()

    payload = mch.load_pickle_cross_platform(args.reference)
    reference_curves = payload["cross_period_psd"]["curves"].copy()
    mle, semi = mch.load_all_modules(ROOT)
    data = mch.load_comparison_data(ROOT, mle["common"], ROOT / "data/common_records.csv",
                                    ROOT / "data/periods")
    payload["data_state"] = mch.comparison_data_to_state(data)
    payload["data_state"].update(project_root=".", pca_data_path="data/common_records.csv",
                                  full_data_dir="data/periods")
    rows = []
    for family in args.models:
        print(f"Fitting {family} ({method})...", flush=True)
        start = time.perf_counter()
        label, (h, curves) = fit_model(data, family, method, payload, mle, semi)
        elapsed = time.perf_counter() - start
        payload["cross_period_psd"]["curves"][label] = (h, curves)
        ref_h, ref_curves = reference_curves[label]
        aligned = np.asarray([[np.interp(ref_h, h, values) for values in row] for row in curves])
        difference = aligned - ref_curves
        rows.append(dict(method=label, seconds=elapsed,
                         curve_rmse=float(np.sqrt(np.mean(difference ** 2))),
                         curve_max_abs_difference=float(np.max(np.abs(difference)))))

    payload["cross_period_psd"]["summary"] = pd.DataFrame()
    payload["cross_period_psd"]["detail"] = pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report = args.output.parent / "analysis" / f"refit_{method}_comparison.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved models: {args.output}\nSaved comparison: {report}")
