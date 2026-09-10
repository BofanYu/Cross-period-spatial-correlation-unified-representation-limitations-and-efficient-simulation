"""Check data, objectives, PSD diagnostics and structured covariance identities."""
from __future__ import annotations

import argparse
import importlib
from scripts.analysis import write_metadata
import os
from pathlib import Path
import time

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
from scripts.analysis import core as mch
from scripts.analysis.wls import calculate_wls
from scripts.analysis.tables import MAIN, EXTRA


def check_data(data):
    expected_periods = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0, 6.0]
    expected_records = [13898, 13898, 13898, 13897, 13891, 13846, 13342, 7997, 5558]
    expected_events = [134, 134, 134, 134, 134, 134, 128, 66, 41]
    expected_sites = [2941, 2941, 2941, 2941, 2941, 2941, 2937, 2780, 2451]
    expected_medians = [76.5, 76.5, 76.5, 76.5, 76.5, 76, 76, 78.5, 79]
    expected_pairs = [1169078, 1169078, 1169078, 1169004, 1168626, 1164864,
                      1140275, 843626, 678938]
    np.testing.assert_allclose(data.periods, expected_periods, rtol=0, atol=0)
    rows = []
    for i, (period, frame) in enumerate(zip(data.periods, data.period_dfs)):
        counts = frame.groupby("eqid").size()
        sites = len(frame[["station_latitude", "station_longitude"]].drop_duplicates())
        pairs = int((counts * (counts - 1) // 2).sum())
        assert (len(frame), len(counts), sites, pairs) == (
            expected_records[i], expected_events[i], expected_sites[i], expected_pairs[i])
        assert (counts.median(), counts.min(), counts.max()) == (
            expected_medians[i], 41 if i == 7 else 40, 539 if i == 8 else 542)
        rows.append(dict(period_s=float(period), records=len(frame), events=len(counts),
                         station_locations=sites, station_pairs=pairs,
                         median_stations_per_event=float(counts.median()),
                         min_stations_per_event=int(counts.min()),
                         max_stations_per_event=int(counts.max())))
    wide = mch.predictive_full_data_wide(data)
    observed = int(np.isfinite(wide[data.period_cols].to_numpy(dtype=float)).sum())
    totals = dict(events=int(wide.eqid.nunique()), observed_entries=observed,
                  union_records=len(wide), complete_records=len(data.df_pca),
                  complete_events=int(data.df_pca.eqid.nunique()))
    assert list(totals.values()) == [134, 110225, 13904, 5555, 41], totals
    totals["union_station_locations"] = len(
        wide[["station_latitude", "station_longitude"]].drop_duplicates())
    assert totals["union_station_locations"] == 2943
    return dict(passed=True, totals=totals, table_01=rows)


def check_wls(payload):
    expected = {
        "Pairwise semivariogram": 0.20447201804505266,
        "GH08 semivariogram": 0.37682357414597273,
        "Kronecker semivariogram": 0.5763133954529416,
        "PCA MLE": 3.1868133795906814,
        "PCA semivariogram": 1.9571537051163646,
        "LMC block MLE": 0.38004329319847746,
        "LMC semivariogram": 0.22635996269829806,
        "Multivariate Matern semivariogram": 0.17834701753232862,
        "SBSS semivariogram": 1.8146016272613397,
        "IOX full semivariogram": 0.1789621196335667,
    }
    rows = calculate_wls(payload)
    actual = {row["method"]: float(row["wls_pairwise_definition_81"]) for row in rows}
    assert set(actual) == set(expected), actual.keys()
    for method, value in actual.items():
        np.testing.assert_allclose(value, expected[method], rtol=1e-10, atol=1e-12,
                                   err_msg=method)
    return dict(passed=True, definition="81 ordered pairs, weight 1/h", values=actual)


def check_psd_diagnostic(payload, data):
    names = ["Pairwise empirical semivariogram", "GH08 semivariogram"]
    curves = {name: payload["cross_period_psd"]["curves"][name] for name in names}
    result = mch.cross_period_psd_analysis(
        data=data, mle=None, semi=None, pairwise_empirical=None, gh08=None,
        pca=None, lmc_semivariogram=None, kronecker=None, ggp_model_a=None,
        n_samples=20, n_periods=9, max_stations_per_sample=30, min_stations=8,
        random_state=0, curve_sets=curves,
    )
    summary = result["summary"].set_index("method")
    for method, minimum, failures in zip(names, [-0.212271, -0.102117], [18, 20]):
        row = summary.loc[method]
        np.testing.assert_allclose(row.min_eigenvalue, minimum, rtol=0, atol=1e-6)
        assert int(row.n_non_psd_samples) == failures
    assert set(result["detail"].n_periods) == {9}
    return dict(passed=True, seed=0, samples=20, periods=9, stations=30,
                note="Nine periods used for the archived diagnostic.",
                diagnostics=result["summary"].to_dict(orient="records"))


def check_covariances(payload, data, semi):
    event = data.df_pca.loc[data.df_pca.eqid == data.df_pca.eqid.iloc[0]]
    event = event.drop_duplicates(["station_latitude", "station_longitude"]).iloc[:6]
    lat = event.station_latitude.to_numpy(dtype=float)
    lon = event.station_longitude.to_numpy(dtype=float)
    distance = mch._haversine_between(lat, lon, lat, lon)
    n, q = len(lat), len(data.periods)
    diagnostics = {}
    for key in ("lmc_semivariogram", "lmc_block_mle"):
        fit = payload[key]
        dense = mch._event_covariance_lmc(lat, lon, fit, semi["lmc"])
        factor_covariance = np.zeros_like(dense)
        factors = []
        for b, length, exponent in zip(fit["B_raw"], fit["length_scales"], fit["gamma_pe"]):
            kernel = (np.exp(-(np.maximum(distance, 1e-12) / length) ** exponent)
                      if length > 0 else np.zeros_like(distance))
            kernel[np.isclose(distance, 0)] = 1.0
            lower_s = np.linalg.cholesky(kernel)
            lower_b = np.linalg.cholesky(0.5 * (b + b.T))
            # Row-major vec(Ls Z Lb.T) = (Ls kron Lb) vec(Z).
            factor = np.kron(lower_s, lower_b)
            factor_covariance += factor @ factor.T
            factors.append((lower_s, lower_b))
        np.testing.assert_allclose(factor_covariance, dense, rtol=1e-11, atol=1e-11)
        rng = np.random.default_rng(90123)
        expected_statistic = 0.0
        for _ in range(3):
            sample = sum(ls @ rng.standard_normal((n, q)) @ lb.T for ls, lb in factors)
            expected_statistic += float(np.mean(np.abs(sample.mean(axis=0)))) / 3
        native = mch._simulate_lmc_residuals_native_timed(
            lat, lon, fit, semi["lmc"], 3, np.random.default_rng(90123), nearest_psd=False)
        np.testing.assert_allclose(native["sample_mean_abs"], expected_statistic,
                                   rtol=1e-11, atol=1e-11)
        diagnostics[key] = dict(
            covariance_max_error=float(np.max(np.abs(factor_covariance - dense))),
            native_draw_error=float(abs(native["sample_mean_abs"] - expected_statistic)),
            minimum_eigenvalue=float(np.linalg.eigvalsh(dense).min()),
        )
    for name, function, key in (
        ("Multivariate Matern", mch._event_covariance_multivariate_matern, "multivariate_matern"),
        ("SBSS", mch._event_covariance_sbss, "sbss_semivariogram"),
        ("IOX", mch._event_covariance_iox, "iox_full_semivariogram"),
    ):
        covariance = function(lat, lon, payload[key], np.ones(q))
        np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.diag(covariance), np.ones(n * q), atol=1e-10)
        assert np.isfinite(covariance).all()
        minimum = float(np.linalg.eigvalsh(covariance).min())
        assert minimum > -1e-9, (name, minimum)
        diagnostics[name] = dict(minimum_eigenvalue=minimum, dimension=n * q)
    return dict(passed=True, stations=n, periods=q, diagnostics=diagnostics)


def check_archived_event_likelihood(payload, data, semi):
    """Use whole small events and global variances to check archived likelihoods."""
    wide = mch.predictive_full_data_wide(data)
    variances = mch._full_data_period_variances(wide, data.period_cols)
    counts = wide.groupby("eqid").size()
    eligible = counts[(counts < 80) & counts.index.isin(data.df_pca.eqid.unique())]
    event_ids = eligible.sort_values().index[:2]
    assert len(event_ids) == 2
    methods = MAIN + EXTRA
    curves = mch._prediction_curve_sets(data, payload["cross_period_psd"], methods=methods)
    archived = pd.read_csv(ROOT / "results/analysis/evaluation/full_event_likelihood.csv")
    archived = archived.set_index(["eqid", "method"])
    fits = {key: payload[key] for key in (
        "pairwise_empirical", "pca", "gh08", "lmc_semivariogram", "lmc_block_mle",
        "kronecker", "iox_full_semivariogram", "sbss_semivariogram", "multivariate_matern")}
    rows = []
    with threadpool_limits(limits=1):
        for eqid in event_ids:
            event = mch._full_data_event_frame(wide, eqid, data.period_cols, None, 0)
            y = event[data.period_cols].to_numpy(dtype=float).reshape(-1)
            observed = np.flatnonzero(np.isfinite(y))
            for method in methods:
                covariance = mch._full_data_covariance(
                    method, event, data, curves, variances, semi=semi, **fits)
                covariance = covariance[np.ix_(observed, observed)]
                covariance = 0.5 * (covariance + covariance.T)
                minimum = float(np.linalg.eigvalsh(covariance).min())
                repaired = minimum < -1e-8
                if repaired:
                    covariance = mch.nearest_psd_matrix(
                        covariance, eps=1e-7, preserve_diagonal=True)
                calculated = float(-mch._mvn_nlpd(y[observed], covariance))
                reference = archived.loc[(eqid, method)]
                assert reference.fit_success and bool(reference.nearest_psd_used) == repaired
                assert (len(event), len(observed)) == (
                    int(reference.n_stations), int(reference.n_observations))
                # PSD repair produces nearly singular matrices and is more sensitive
                # to eigensolver/BLAS roundoff than the native PSD representations.
                rtol, atol = (1e-7, 1e-6) if repaired else (1e-10, 1e-8)
                np.testing.assert_allclose(calculated, reference.log_likelihood,
                                           rtol=rtol, atol=atol, err_msg=f"{eqid}: {method}")
                rows.append(dict(eqid=int(eqid), method=method, stations=len(event),
                                 observed_entries=len(observed), nearest_psd_used=repaired,
                                 calculated=calculated, archived=float(reference.log_likelihood),
                                 absolute_error=abs(calculated - float(reference.log_likelihood))))
    return dict(passed=True, evaluations=len(rows), events=[int(x) for x in event_ids],
                geometry="Complete original event geometries; no station subsampling",
                variances="Estimated once using all 110225 observed entries",
                native_psd_tolerance=dict(rtol=1e-10, atol=1e-8),
                repaired_tolerance=dict(rtol=1e-7, atol=1e-6), diagnostics=rows)


def check_full_lmc_objective(payload, data, mle):
    """Evaluate the full fitting objective and gradient at the paper parameters."""
    start = time.perf_counter()
    fit = payload["lmc_block_mle"]
    wide = mch.predictive_full_data_wide(data)
    module = mle["lmc"]
    with threadpool_limits(limits=1):
        blocks = module.prepare_lmc_partial_block_events(
            wide, data.period_cols, fit["length_scales"], fit["gamma_pe"],
            min_stations=5, max_stations_per_event=None, random_state=123, center=True)
        totals = dict(events=len(blocks), rows=sum(block.n_stations for block in blocks),
                      observations=sum(block.y.size for block in blocks))
        assert list(totals.values()) == [134, 13904, 110225], totals
        objective = module.LMCPartialBlockObjective(
            blocks, q=len(data.periods), jitter=1e-8, ridge=1e-6)
        value, gradient = objective(np.asarray(fit["theta"]))
    calculated = dict(objective=float(value), gradient_norm=float(np.linalg.norm(gradient)),
                      gradient_inf_norm=float(np.max(np.abs(gradient))))
    reference = dict(objective=float(fit["nll"]),
                     gradient_norm=float(fit["final_gradient_norm"]),
                     gradient_inf_norm=float(fit["final_gradient_inf_norm"]))
    for key in calculated:
        np.testing.assert_allclose(calculated[key], reference[key], rtol=1e-8, atol=1e-12,
                                   err_msg=f"Full LMC fitting {key}")
    return dict(passed=True, seconds=time.perf_counter() - start, **totals,
                calculated=calculated, archived=reference,
                absolute_errors={key: abs(calculated[key] - reference[key]) for key in calculated},
                tolerance=dict(rtol=1e-8, atol=1e-12), center=True, jitter=1e-8, ridge=1e-6,
                optimizer_rerun=False, archived_optimizer_success=bool(fit["success"]),
                archived_optimizer_message=fit["message"])


def validate(cache, output, full_lmc_objective=False):
    start = time.perf_counter()
    payload = mch.load_pickle_cross_platform(cache)
    data = mch.comparison_data_from_state(payload["data_state"])
    data.project_root = ROOT
    mle, semi = mch.load_all_modules(ROOT)
    for module in ("scripts.models.iox", "scripts.models.sbss",
                   "scripts.models.matern"):
        importlib.import_module(module)
    sources = list((ROOT / "scripts").rglob("*.py"))
    for source in sources:
        compile(source.read_text(encoding="utf-8-sig"), str(source), "exec")
    checks = dict(imports_and_syntax=dict(passed=True, source_files=len(sources),
                                          fitting_modules=len(mle) + len(semi)))
    checks["data"] = check_data(data)
    checks["wls"] = check_wls(payload)
    checks["psd_diagnostic"] = check_psd_diagnostic(payload, data)
    checks["covariance_and_sampling"] = check_covariances(payload, data, semi)
    checks["archived_event_likelihood"] = check_archived_event_likelihood(payload, data, semi)
    if full_lmc_objective:
        print("Checking LMC objective and analytic gradient on all 134 events...", flush=True)
        checks["full_lmc_objective"] = check_full_lmc_objective(payload, data, mle)
    result = dict(passed=True, seconds=time.perf_counter() - start, checks=checks,
                  scope="Archived values, covariance identities, and 20 whole-event likelihoods; "
                        + ("full LMC fitting objective included. " if full_lmc_objective else "")
                        + "Optimizer reruns and full predictive evaluation are separate.")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_metadata(output, result)
    print(f"All {len(checks)} validation groups passed in {result['seconds']:.2f}s. {output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "results/models.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/checks.csv")
    parser.add_argument("--full-lmc-objective", action="store_true",
                        help="Also evaluate the full 134-event LMC objective and gradient at the archived parameters.")
    args = parser.parse_args()
    validate(args.cache, args.output, args.full_lmc_objective)
