"""Fit the full-data and complete-case models from the processed residual CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from scripts.common import core as mch
from scripts.models import iox, sbss, matern

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "results/analysis/model_fit"
FULL_MODELS = (
    ("pairwise", "semivariogram"), ("gh08", "semivariogram"),
    ("separable", "semivariogram"), ("pca", "semivariogram"), ("pca", "mle"),
    ("lmc", "semivariogram"), ("lmc", "mle"), ("iox", "semivariogram"),
    ("sbss", "semivariogram"), ("matern", "semivariogram"),
)
COMPLETE_MODELS = tuple(item for item in FULL_MODELS if item[0] in ("separable", "pca", "lmc"))


def prepare_data(dataset="full"):
    """Reconstruct empirical targets; no fitted-model file is read."""
    mle, _ = mch.load_all_modules(ROOT)
    data = mch.load_comparison_data(
        ROOT, mle["common"], ROOT / "data/common_records.csv", ROOT / "data/periods")
    if dataset == "complete":
        wide = mch.predictive_full_data_wide(data)
        wide = wide.dropna(subset=data.period_cols).reset_index(drop=True)
        data = mch._nonpsd_comparison_data_from_wide(
            data, mle["common"], wide, data.period_cols)
    elif dataset != "full":
        raise ValueError("dataset must be 'full' or 'complete'")
    return data


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


def fit_model(data, family, method, payload, mle, semi, dataset="full"):
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
            if dataset == "complete":
                from scripts.fitting.mle import fit_lmc_complete_mle
                result = fit_lmc_complete_mle(data)
            else:
                result = fit_lmc_mle(data, mle)
            payload["lmc_block_mle"] = {
                key: value for key, value in result.items()
                if key not in ("result", "history", "events", "blocks")}
            label = "LMC block MLE"
        else:
            result = mch.fit_lmc_semivariogram(data, semi, n_iterations=30)
            payload["lmc_semivariogram"] = result
            label = "LMC semivariogram"
        model_api = mle["lmc"] if method == "mle" else semi["lmc"]
        return label, (h, mch.lmc_period_pair_curves(result, model_api, h))
    label, key, function = {
        "iox": ("IOX full semivariogram", "iox_full_semivariogram", fit_iox),
        "sbss": ("SBSS semivariogram", "sbss_semivariogram", fit_sbss),
        "matern": ("Multivariate Matern semivariogram", "multivariate_matern", fit_matern),
    }[family]
    payload[key], curves = function(data)
    return label, curves


def fit_models(dataset="full", models=None, method=None, output=None, initial=None,
               include_lmc_mle=True, data=None):
    """Return (fitted payload, timing DataFrame), optionally writing one PKL.

    With ``initial=None`` every selected model is fitted from CSV input. Supply
    a previous payload explicitly to update selected fits while retaining others.
    ``data=prepare_data(dataset)`` excludes data preparation from fit timing.
    """
    import copy
    jobs = FULL_MODELS if dataset == "full" else COMPLETE_MODELS
    if models is not None:
        allowed = {family for family, _ in jobs}
        unknown = set(models) - allowed
        if unknown:
            raise ValueError(f"Models unavailable for {dataset}: {sorted(unknown)}")
        jobs = tuple(job for job in jobs if job[0] in models)
    if method is not None:
        jobs = tuple(job for job in jobs if job[1] == method)
    if not include_lmc_mle:
        jobs = tuple(job for job in jobs if job != ("lmc", "mle"))
    data = prepare_data(dataset) if data is None else data
    mle, semi = mch.load_all_modules(ROOT)
    payload = copy.deepcopy(initial) if initial is not None else {}
    payload["data_state"] = mch.comparison_data_to_state(data)
    payload["data_state"].update(project_root=".", pca_data_path="data/common_records.csv",
                                  full_data_dir="data/periods")
    payload.setdefault("cross_period_psd", {"curves": {}})
    rows = []
    for family, fit_method in jobs:
        print(f"Fitting {dataset}: {family} ({fit_method})...", flush=True)
        start = time.perf_counter()
        label, (h, curves) = fit_model(data, family, fit_method, payload, mle, semi, dataset)
        elapsed = time.perf_counter() - start
        payload["cross_period_psd"]["curves"][label] = (h, curves)
        rows.append(dict(dataset=dataset, method=label, family=family,
                         fitting_method=fit_method, fit_time_seconds=elapsed))
        # Save after each completed fit so the two fitting CLIs can be run in turn.
        if output is not None:
            save_models(payload, output)
    payload["cross_period_psd"]["summary"] = pd.DataFrame()
    payload["cross_period_psd"]["detail"] = pd.DataFrame()
    if output is not None:
        save_models(payload, output)
    return payload, pd.DataFrame(rows)


def save_models(payload, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_cli(method, families, default=None):
    parser = argparse.ArgumentParser(description=f"Fit article models by {method}.")
    parser.add_argument("--models", nargs="+", choices=families, metavar="MODEL")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", choices=["full", "complete"], default="full")
    parser.add_argument("--fresh", action="store_true", help="Start a new archive instead of updating existing fits.")
    args = parser.parse_args()
    output = args.output or MODEL_DIR / ("models.pkl" if args.dataset == "full" else "models_complete.pkl")
    selected = args.models
    if selected is None and default is not None:
        selected = default
    initial = mch.load_pickle_cross_platform(output) if output.exists() and not args.fresh else None
    _, timing = fit_models(args.dataset, selected, method, output, initial)
    print(timing.to_string(index=False))
    print(f"Saved models: {output}")
