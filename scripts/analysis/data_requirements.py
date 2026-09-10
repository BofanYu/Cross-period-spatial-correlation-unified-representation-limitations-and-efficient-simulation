"""Repeat the paper's saved event samples, with all stations retained.

--models fast runs nine models; --models all also reruns the expensive LMC MLE.
Use --event-grid 40 --max-reps 1 for a small executable check.
"""
from __future__ import annotations
import argparse
from scripts.analysis import write_metadata
import os
from pathlib import Path
import time
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
from scripts.analysis import core as mch
from scripts.models import iox as iox
from scripts.models import sbss as sbss
from scripts.models import matern as matern

def _sbss_subset(common, selected, module):
    frame = common.frame.loc[common.frame["eqid"].isin(selected)].copy()
    if frame.empty:
        raise ValueError("No common/aligned records remain in this event sample")
    frame = frame.sort_values("eqid", kind="stable").reset_index(drop=True)
    values = frame[common.period_columns].to_numpy(dtype=float)
    covariance = (values.T @ values) / len(values)
    period_rms = np.sqrt(np.clip(np.diag(covariance), 1.0e-15, None))
    correlation = covariance / np.outer(period_rms, period_rms)
    return module.CommonData(
        frame=frame,
        period_columns=list(common.period_columns),
        periods=np.asarray(common.periods, dtype=float).copy(),
        values=values,
        standardized_values=values / period_rms[None, :],
        covariance=covariance,
        correlation=correlation,
        period_rms=period_rms,
    )

def _iox_curves(module, fit, events, target_h):
    step = float(target_h[1] - target_h[0])
    basis, _, _ = module.iox_semivariogram_basis(
        events,
        fit.marginal,
        bin_size_km=step,
        max_distance_km=float(target_h[-1]),
    )
    fitted_gamma = fit.sigma[:, :, None] * basis
    denominator = np.sqrt(np.outer(np.diag(fit.sigma), np.diag(fit.sigma)))
    curves = np.empty((*fit.rho_colocated.shape, len(target_h)), dtype=float)
    curves[:, :, 0] = fit.rho_colocated
    curves[:, :, 1:] = (
        fit.rho_colocated[:, :, None]
        - fitted_gamma / denominator[:, :, None]
    )
    return 0.5 * (curves + np.swapaxes(curves, 0, 1))

def _matern_curves(module, fit, target_h):
    q = len(fit.periods)
    curves = np.empty((q, q, len(target_h)), dtype=float)
    for i in range(q):
        for j in range(q):
            curves[i, j] = fit.rho[i, j] * module.matern_correlation(
                target_h, fit.nu[i, j], fit.alpha[i, j]
            )
    curves[:, :, 0] = fit.rho
    return curves

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=("baseline", "fast", "all"), default="fast")
    parser.add_argument("--event-grid", type=int, nargs="+", default=[40, 60, 80, 100, 120, 134])
    parser.add_argument("--max-reps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/data_requirement")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(ROOT / "data/samples/sampled_events.csv")
    samples = samples[samples.n_events.isin(args.event_grid)]
    if args.max_reps is not None:
        samples = samples[samples.rep < args.max_reps]
    samples = samples.sort_values(["n_events", "rep"])
    config = dict(models=args.models, event_grid=args.event_grid,
                  repetitions=samples.groupby("n_events").size().to_dict(),
                  site_cap=None, lmc_iterations=30,
                  lmc_mle=dict(reference="results/analysis/data_requirement/lmc_full_reference.csv (original 134-event repetition 0)", center=False, cap80_maxiter=250, all_sites_maxiter=150,
                               cap80_gtol=1e-4, all_sites_gtol=1e-3,
                               note="Executable original sample-fitting settings; old run metadata incorrectly listed final gtol=1e-4."))
    write_metadata(args.output / "configuration.csv", config)
    payload = mch.load_pickle_cross_platform(ROOT / "results/models.pkl")
    data = mch.comparison_data_from_state(payload["data_state"])
    mle, semi = mch.load_all_modules(ROOT)
    context = mch.prepare_nonpsd_evaluation(payload, mch.NonPSDEvalConfig(seed=12345))
    repetitions = samples.groupby("n_events").size().to_dict()
    baseline_samples = samples[samples.n_events < 134]
    baseline_grid = tuple(n for n in args.event_grid if n < 134)
    rows = []
    if not baseline_samples.empty:
        baseline = mch.nonpsd_refit_stability_full_event_sample_evaluation(
            context, data, mle, semi,
            methods=("pairwise", "gh08", "kronecker", "pca_semivariogram", "pca_mle", "lmc"),
            n_event_grid=baseline_grid, site_caps=None, n_reps=repetitions,
            lmc_iterations=30, sampled_events=baseline_samples,
        )["refit_stability"]
        rows = baseline.to_dict("records")
    print(f"Completed {len(rows)} baseline fits.", flush=True)
    pd.DataFrame(rows).to_csv(args.output / "refit_detail.csv", index=False)
    if args.models != "baseline":
        _, events, _ = iox.load_union_panel(ROOT / "data/periods")
        common = sbss.load_pca_common_data(ROOT / "data/common_records.csv")
        periods, frames, input_summary = matern.load_period_datasets(ROOT / "data/periods")
        for sample in baseline_samples.itertuples(index=False):
            selected = set(str(sample.sampled_event_ids).split())
            selected_events = [event for event in events if str(event.eqid) in selected]
            selected_frames = [frame[frame.eqid.isin(selected)].copy() for frame in frames]
            for method in ("IOX full semivariogram", "SBSS semivariogram", "Multivariate Matern semivariogram"):
                print(f"{sample.n_events} events, repetition {sample.rep}: {method}", flush=True)
                ref_h, ref_curves = payload["cross_period_psd"]["curves"][method]
                start = time.perf_counter()
                row = dict(n_events=sample.n_events, rep=sample.rep, site_cap="all", display_name=method,
                           model={"IOX full semivariogram":"iox_full_semivariogram", "SBSS semivariogram":"sbss_semivariogram", "Multivariate Matern semivariogram":"multivariate_matern"}[method])
                try:
                    if method == "IOX full semivariogram":
                        empirical = iox.empirical_event_semivariograms(selected_events, bin_size_km=2.0, max_distance_km=100.0)
                        fit = iox.fit_iox_wls(empirical, selected_events, tail_bins=10, bin_size_km=2.0, max_distance_km=100.0)
                        fit_h, curves = ref_h, _iox_curves(iox, fit, selected_events, ref_h)
                    elif method == "SBSS semivariogram":
                        fit = sbss.fit_sbss(_sbss_subset(common, selected, sbss), rings_km=sbss.RINGS_KM)
                        fit_h, curves = fit.h_fine, fit.correlation_fine
                    else:
                        empirical = matern.pool_pairwise_event_semivariograms(periods, selected_frames, input_summary)
                        fit = matern.fit_flexible_matern(empirical, tail_bins=10, spatial_dimension=2)
                        fit_h, curves = ref_h, _matern_curves(matern, fit, ref_h)
                    row["curve_rmse_vs_full_fit"] = mch._nonpsd_curve_rmse(ref_h, ref_curves, fit_h, curves)
                    row["fit_success"] = bool(np.isfinite(row["curve_rmse_vs_full_fit"]))
                except Exception as exc:
                    row.update(fit_success=False, error=str(exc), curve_rmse_vs_full_fit=np.nan)
                row["fit_seconds"] = time.perf_counter() - start
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.output / "refit_detail.csv", index=False)
    if args.models == "all":
        selection = pd.read_csv(ROOT / "data/samples/lmc_cap80_selection.csv")
        wide = context["df"]
        reference_pack = pd.read_csv(ROOT / "results/analysis/data_requirement/lmc_full_reference.csv", float_precision="round_trip")
        reference_fit = dict(B_raw=[reference_pack.loc[reference_pack.component == f"B{i}"].pivot(index="row", columns="column", values="value").to_numpy() for i in (1, 2, 3)],
                             length_scales=(15.0, 70.0, 0.0), gamma_pe=(0.4, 1.0, 1.0))
        lmc_ref_h = data.h_fine
        lmc_ref_curves = mch._nonpsd_lmc_curves_from_fit(reference_fit, lmc_ref_h)
        options = dict(max_stations_per_event=None, ridge=1e-6, ftol=1e-7, maxls=60,
                       maxcor=20, diag_lower=1e-5, diag_upper=10.0, offdiag_bound=10.0,
                       random_state=123, center=False)
        for sample in samples.itertuples(index=False):
            selected = set(str(sample.sampled_event_ids).split())
            subset = wide[wide.eqid.astype(str).isin(selected)].copy()
            event_ids = subset.eqid.unique()
            capped = mch.apply_lmc_block_mle_station_selection(
                wide, selection, data.period_cols, event_ids=event_ids, require_complete=False)
            print(f"{sample.n_events} events, repetition {sample.rep}: LMC block MLE", flush=True)
            start = time.perf_counter()
            cap80 = mch.fit_lmc_mle_block_full_data(data, mle, wide=capped, maxiter=250, gtol=1e-4, **options)
            fit = mch.fit_lmc_mle_block_full_data(data, mle, wide=subset, init=cap80, maxiter=150, gtol=1e-3, **options)
            ref_h, ref_curves = lmc_ref_h, lmc_ref_curves
            curves = mch._nonpsd_lmc_curves_from_fit(fit, ref_h)
            rmse = mch._nonpsd_curve_rmse(ref_h, ref_curves, ref_h, curves)
            rows.append(dict(model="lmc_block_mle", display_name="LMC block MLE", n_events=sample.n_events,
                             rep=sample.rep, site_cap="all", curve_rmse_vs_full_fit=rmse,
                             fit_success=bool(np.isfinite(rmse)), optimizer_success=fit["success"],
                             fit_seconds=time.perf_counter()-start))
            pd.DataFrame(rows).to_csv(args.output / "refit_detail.csv", index=False)
    if 134 in args.event_grid:
        reference_models = ("pairwise", "gh08", "kronecker", "pca_semivariogram", "pca_mle", "lmc")
        if args.models != "baseline":
            reference_models += ("iox_full_semivariogram", "sbss_semivariogram", "multivariate_matern")
        for model in reference_models:
            rows.append(dict(model=model, display_name=mch.NONPSD_MODEL_NAME_MAP[model], n_events=134,
                             rep=0, site_cap="all", curve_rmse_vs_full_fit=0.0, fit_success=True,
                             sample_source="full-data reference fit"))
    expected_count = len(baseline_samples) * (6 if args.models == "baseline" else 9)
    if 134 in args.event_grid:
        expected_count += 6 if args.models == "baseline" else 9
    if args.models == "all":
        expected_count += len(samples)
    if len(rows) != expected_count:
        raise RuntimeError(f"Model coverage incomplete: {len(rows)} rows, expected {expected_count}.")
    detail = pd.DataFrame(rows)
    detail["display_name"] = detail["display_name"].replace({
        "Kronecker semivariogram": "Separable kernel semivariogram", "LMC block MLE": "LMC MLE"})
    detail.to_csv(args.output / "refit_detail.csv", index=False)
    success = detail[detail.fit_success & np.isfinite(detail.curve_rmse_vs_full_fit)]
    summary = success.groupby(["display_name", "model", "n_events"], as_index=False).agg(
        n_success=("curve_rmse_vs_full_fit", "size"), rmse_median=("curve_rmse_vs_full_fit", "median"),
        rmse_mean=("curve_rmse_vs_full_fit", "mean"), rmse_std=("curve_rmse_vs_full_fit", "std"))
    reference = (summary.n_events == 134) & (summary.n_success == 1)
    summary.loc[reference, "rmse_std"] = 0.0
    summary.to_csv(args.output / "data_requirement_summary.csv", index=False)
    print(summary.to_string(index=False))
    failed = detail[~detail.fit_success.astype(bool)]
    if len(failed):
        raise RuntimeError(f"{len(failed)} fits failed; inspect refit_detail.csv for their errors.")


if __name__ == "__main__":
    main()
