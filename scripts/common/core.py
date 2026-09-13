"""Shared fitting, covariance, prediction, and resampling routines for the paper."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path, PosixPath, PureWindowsPath

import numpy as np
import pandas as pd




@dataclass
class ComparisonData:
    project_root: Path
    pca_data_path: Path
    full_data_dir: Path
    periods: np.ndarray
    bin_size: int
    max_distance: int
    tail_bins: int
    h_bins: np.ndarray
    h_fine: np.ndarray
    df_pca_source: pd.DataFrame
    df_pca: pd.DataFrame
    period_cols: list[str]
    events_pca: np.ndarray
    period_dfs: list[pd.DataFrame]
    gamma_emp: list[list[np.ndarray]]
    count_emp: list[list[np.ndarray]]
    pair_record_counts: np.ndarray
    rho_0: np.ndarray
    emp_sills: np.ndarray







def load_all_modules(project_root=None):
    """Select the model implementations for each fitting method."""
    from types import SimpleNamespace
    from scripts import data
    from scripts.fitting import mle, semivariogram
    from scripts.models import pairwise, pca, lmc, gh08, separable

    def model_api(module, method):
        suffix = "_" + method
        members = {name: value for name, value in vars(module).items()
                   if not name.startswith("_")}
        members.update({name[:-len(suffix)]: value for name, value in members.copy().items()
                        if name.endswith(suffix)})
        return SimpleNamespace(**members)

    def framework(method, fitter):
        common = {name: value for module in (data, fitter)
                  for name, value in vars(module).items() if not name.startswith("_")}
        product = vars(model_api(gh08, method)) | vars(model_api(separable, method))
        return dict(common=SimpleNamespace(**common), pairwise=pairwise,
                    pca=model_api(pca, method), lmc=model_api(lmc, method),
                    product=SimpleNamespace(**product))

    return framework("mle", mle), framework("semivariogram", semivariogram)

def load_comparison_data(
    project_root: Path,
    common,
    pca_data_path: Path,
    full_data_dir: Path,
    periods=(0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0, 6.0),
    bin_size=2,
    max_distance=100,
    tail_bins=10,
    h_fine_max=120.0,
    h_fine_n=400,
) -> ComparisonData:
    periods = np.asarray(periods, dtype=float)
    nmax = round(max_distance / bin_size)
    h_bins = np.arange(1, nmax + 1) * bin_size
    h_fine = np.linspace(0.0, h_fine_max, h_fine_n)

    df_pca_source = pd.read_csv(pca_data_path)
    period_cols = common.select_period_columns(df_pca_source, periods)
    df_pca = df_pca_source[
        ["eqid", "station_latitude", "station_longitude", *period_cols]
    ].dropna().copy()
    events_pca = df_pca["eqid"].unique()

    period_dfs = common.load_period_datasets(full_data_dir, periods)
    gamma_emp, count_emp, pair_record_counts = common.pool_pairwise_crossvariograms(
        period_dfs,
        periods,
        bin_size=bin_size,
        max_distance=max_distance,
    )
    rho_0 = common.pairwise_lag0_corr(period_dfs)
    emp_sills = common.empirical_sills(gamma_emp, tail_bins=tail_bins)

    return ComparisonData(
        project_root=Path(project_root),
        pca_data_path=Path(pca_data_path),
        full_data_dir=Path(full_data_dir),
        periods=periods,
        bin_size=bin_size,
        max_distance=max_distance,
        tail_bins=tail_bins,
        h_bins=h_bins,
        h_fine=h_fine,
        df_pca_source=df_pca_source,
        df_pca=df_pca,
        period_cols=period_cols,
        events_pca=events_pca,
        period_dfs=period_dfs,
        gamma_emp=gamma_emp,
        count_emp=count_emp,
        pair_record_counts=pair_record_counts,
        rho_0=rho_0,
        emp_sills=emp_sills,
    )


def powered_exponential(h_grid, length_scale, exponent):
    h_grid = np.asarray(h_grid, dtype=float)
    return np.exp(-((h_grid / float(length_scale)) ** float(exponent)))


def fit_pca_full_mle(data: ComparisonData, mle, common):
    y_all = data.df_pca[data.period_cols].values.astype(float)
    cov_pool, corr_pool = mle["pca"].corr_matrix(y_all)
    explained, comps, evals = mle["pca"].pca_from_corr(corr_pool)

    var_pool = np.diag(cov_pool)
    std_pool = np.sqrt(np.clip(var_pool, 1e-12, None))
    loadings = comps.T
    n_components = loadings.shape[1]

    gamma_pc, count_pc = mle["pca"].compute_pc_variograms(
        data.df_pca,
        data.period_cols,
        data.events_pca,
        loadings,
        std_pool,
        data.bin_size,
        len(data.h_bins),
    )
    pc_events = mle["pca"].prepare_pc_events(
        data.df_pca,
        data.period_cols,
        data.events_pca,
        loadings,
        std_pool,
        evals,
    )

    pc_sills = np.array([np.mean(g[-data.tail_bins:]) for g in gamma_pc])
    length_scales = np.zeros(n_components)
    exponents = np.zeros(n_components)
    mle_nll = np.zeros(n_components)

    for k in range(n_components):
        if evals[k] <= 1e-10 or not pc_events[k]:
            length_scales[k], exponents[k], mle_nll[k] = 1.0, 1.0, np.nan
            continue
        fit = common.fit_pe_mle(pc_events[k])
        length_scales[k] = fit["LE"]
        exponents[k] = fit["gammaE"]
        mle_nll[k] = fit["nll"]

    rho_pc = np.column_stack([
        common.powered_exponential_corr(data.h_fine, length_scales[k], exponents[k])
        for k in range(n_components)
    ])
    rho_pc[0, :] = 1.0

    weights = (loadings ** 2) * evals[None, :]
    cov_th = weights @ rho_pc.T
    rho = cov_th / np.clip(cov_th[:, [0]], 1e-12, None)

    return {
        "Y_all": y_all,
        "C_pool": cov_pool,
        "R_pool": corr_pool,
        "explained": explained,
        "components": comps,
        "evals": evals,
        "var_pool": var_pool,
        "std_pool": std_pool,
        "U": loadings,
        "K": n_components,
        "gamma_pc": gamma_pc,
        "count_pc": count_pc,
        "pc_sills": pc_sills,
        "LE": length_scales,
        "gammaE": exponents,
        "mle_nll": mle_nll,
        "rho_pc": rho_pc,
        "rho": rho,
        "le_eff": common.effective_ranges(rho, data.h_fine),
    }


def goda_hong_curves(direct_fit, rho0, h_grid):
    length_scales = np.asarray(direct_fit["LE"], dtype=float)
    exponents = np.asarray(direct_fit["gammaE"], dtype=float)
    nper = len(length_scales)
    curves = np.zeros((nper, nper, len(h_grid)))

    for ih, h in enumerate(h_grid):
        for i in range(nper):
            for j in range(nper):
                kmax = max(i, j)
                rho_auto = powered_exponential(h, length_scales[kmax], exponents[kmax])
                curves[i, j, ih] = rho_auto if i == j else rho0[i, j] * rho_auto
    curves[:, :, 0] = rho0
    np.fill_diagonal(curves[:, :, 0], 1.0)
    return curves


def separable_curves(separable_fit, rho0, h_grid):
    kernel = powered_exponential(h_grid, separable_fit["LE"], separable_fit["gammaE"])
    kernel[0] = 1.0
    nper = len(rho0)
    curves = np.zeros((nper, nper, len(h_grid)))
    for i in range(nper):
        for j in range(nper):
            rho0_ij = 1.0 if i == j else rho0[i, j]
            curves[i, j, :] = rho0_ij * kernel
    return curves


def fit_pairwise_empirical(data: ComparisonData, pairwise_module=None):
    out = {
        "gamma_emp": data.gamma_emp,
        "count_emp": data.count_emp,
        "record_counts": data.pair_record_counts,
        "rho_0": data.rho_0,
        "emp_sills": data.emp_sills,
    }
    if pairwise_module is not None:
        out["fit"] = pairwise_module.fit_pairwise_pe(
            data.gamma_emp,
            data.h_bins,
            tail_bins=data.tail_bins,
        )
    return out


def fit_lmc_semivariogram(data: ComparisonData, semi, n_iterations=30):
    return semi["lmc"].fit_lmc_c4(
        data.gamma_emp,
        data.periods,
        data.bin_size,
        data.rho_0,
        data.h_fine,
        n_iterations=n_iterations,
    )


def apply_lmc_block_mle_station_selection(
    df,
    station_selection,
    period_cols,
    *,
    event_ids=None,
    event_col="eqid",
    require_complete=True,
):
    """Filter a data frame to a pre-recorded LMC cap station selection."""
    period_cols = list(period_cols)
    selection = pd.read_csv(station_selection) if isinstance(station_selection, (str, Path)) else pd.DataFrame(station_selection).copy()
    if selection.empty:
        return df.iloc[0:0].copy()

    keep = [event_col, "station_latitude", "station_longitude", *period_cols]
    if "recid" in df.columns:
        keep.insert(1, "recid")

    df_model = df[[c for c in keep if c in df.columns]].copy()
    if require_complete:
        df_model = df_model.dropna(subset=period_cols)
    else:
        df_model = df_model.loc[df_model[period_cols].notna().any(axis=1)].copy()
    df_model = df_model.reset_index(drop=False).rename(columns={"index": "_source_index"})
    df_model["_lmc_source_pos"] = np.arange(len(df_model), dtype=int)

    if event_ids is not None:
        event_set = set(pd.Index(event_ids).dropna().tolist())
        selection = selection[selection[event_col].isin(event_set)].copy()
        df_model = df_model[df_model[event_col].isin(event_set)].copy()
    if selection.empty or df_model.empty:
        return df_model.iloc[0:0].drop(columns=[c for c in ["_source_index", "_lmc_source_pos"] if c in df_model], errors="ignore")

    if "_lmc_source_pos" in selection.columns:
        selected_pos = set(selection["_lmc_source_pos"].astype(int).tolist())
        out = df_model[df_model["_lmc_source_pos"].isin(selected_pos)].copy()
    elif "recid" in selection.columns and "recid" in df_model.columns:
        keys = selection[[event_col, "recid"]].drop_duplicates()
        out = df_model.merge(keys, on=[event_col, "recid"], how="inner")
    else:
        keys = selection[[event_col, "station_latitude", "station_longitude"]].drop_duplicates()
        out = df_model.merge(keys, on=[event_col, "station_latitude", "station_longitude"], how="inner")

    if "_lmc_selected_rank" in selection.columns:
        if "_lmc_source_pos" in selection.columns and "_lmc_source_pos" in out.columns:
            rank_key = [event_col, "_lmc_source_pos"]
        elif "recid" in selection.columns and "recid" in out.columns:
            rank_key = [event_col, "recid"]
        else:
            rank_key = [event_col, "station_latitude", "station_longitude"]
        out = out.merge(
            selection[rank_key + ["_lmc_selected_rank"]].drop_duplicates(),
            on=rank_key,
            how="left",
        )
        sort_cols = [event_col, "_lmc_selected_rank"]
    else:
        sort_cols = [event_col, "_lmc_source_pos"] if "_lmc_source_pos" in out.columns else [event_col]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out.drop(columns=[c for c in ["_source_index", "_lmc_source_pos", "_lmc_selected_rank"] if c in out.columns], errors="ignore")


def fit_lmc_mle_block(
    data: ComparisonData,
    mle,
    length_scales=(15.0, 70.0, 0.0),
    gamma_pe=(0.4, 1.0, 1.0),
    weights=(0.15, 0.35, 0.50),
    min_stations=5,
    max_stations_per_event=80,
    random_state=123,
    maxiter=250,
    jitter=1e-8,
    ridge=0.0,
    init=None,
    ftol=1e-7,
    gtol=1e-4,
    maxls=60,
    maxcor=20,
    diag_lower=1e-5,
    diag_upper=10.0,
    offdiag_bound=10.0,
    log_every=0,
):
    """Fit fixed-kernel LMC by exact block-diagonal Gaussian likelihood."""
    return mle["lmc"].fit_lmc_block_mle(
        data.df_pca,
        data.period_cols,
        data.h_fine,
        length_scales=length_scales,
        gamma_pe=gamma_pe,
        weights=weights,
        min_stations=min_stations,
        max_stations_per_event=max_stations_per_event,
        random_state=random_state,
        maxiter=maxiter,
        jitter=jitter,
        ridge=ridge,
        init=init,
        ftol=ftol,
        gtol=gtol,
        maxls=maxls,
        maxcor=maxcor,
        diag_lower=diag_lower,
        diag_upper=diag_upper,
        offdiag_bound=offdiag_bound,
        log_every=log_every,
        center=True,
    )


def fit_lmc_mle_block_full_data(
    data: ComparisonData,
    mle,
    length_scales=(15.0, 70.0, 0.0),
    gamma_pe=(0.4, 1.0, 1.0),
    weights=(0.15, 0.35, 0.50),
    min_stations=5,
    max_stations_per_event=80,
    random_state=123,
    maxiter=250,
    jitter=1e-8,
    ridge=0.0,
    init=None,
    ftol=1e-7,
    gtol=1e-4,
    maxls=60,
    maxcor=20,
    diag_lower=1e-5,
    diag_upper=10.0,
    offdiag_bound=10.0,
    log_every=0,
    wide=None,
    center=True,
):
    """Fit exact LMC MLE to the outer-joined full data without imputation."""
    if wide is None:
        wide = predictive_full_data_wide(data)
    required = {
        "eqid",
        "station_latitude",
        "station_longitude",
        *data.period_cols,
    }
    missing = sorted(required.difference(wide.columns))
    if missing:
        raise KeyError(f"Full-data wide table is missing columns: {missing}")

    return mle["lmc"].fit_lmc_partial_block_mle(
        wide,
        data.period_cols,
        data.h_fine,
        length_scales=length_scales,
        gamma_pe=gamma_pe,
        weights=weights,
        min_stations=min_stations,
        max_stations_per_event=max_stations_per_event,
        random_state=random_state,
        maxiter=maxiter,
        jitter=jitter,
        ridge=ridge,
        init=init,
        ftol=ftol,
        gtol=gtol,
        maxls=maxls,
        maxcor=maxcor,
        diag_lower=diag_lower,
        diag_upper=diag_upper,
        offdiag_bound=offdiag_bound,
        log_every=log_every,
        center=center,
    )


def comparison_data_to_state(data: ComparisonData):
    """Convert ComparisonData to a reload-safe plain dictionary for caching."""
    state = {field.name: getattr(data, field.name) for field in fields(ComparisonData)}
    for key in ("project_root", "pca_data_path", "full_data_dir"):
        if key in state:
            state[key] = str(state[key])
    return state


def comparison_data_from_state(state):
    """Rebuild ComparisonData from a cache dictionary."""
    state = dict(state)
    for key in ("project_root", "pca_data_path", "full_data_dir"):
        if key in state:
            state[key] = Path(__file__).resolve().parents[2] / Path(state[key])
    return ComparisonData(**state)


class _CrossPlatformPathUnpickler(pickle.Unpickler):
    """Load pickles containing pathlib WindowsPath objects on non-Windows hosts."""

    def find_class(self, module, name):
        if module == "pathlib" and name in {"WindowsPath", "PureWindowsPath"}:
            return PureWindowsPath
        if module == "pathlib" and name in {"PosixPath", "PurePosixPath"}:
            return PosixPath
        return super().find_class(module, name)


def load_pickle_cross_platform(path):
    with Path(path).open("rb") as handle:
        return _CrossPlatformPathUnpickler(handle).load()






def pairwise_empirical_corr_curves(data: ComparisonData):
    """Convert empirical pairwise cross-variograms to correlation matrices by lag."""
    nper = len(data.periods)
    curves = np.zeros((nper, nper, len(data.h_bins)))

    for i in range(nper):
        for j in range(nper):
            sill_ij = float(np.mean(data.gamma_emp[i][j][-data.tail_bins:]))
            if np.isclose(sill_ij, 0.0):
                curves[i, j, :] = np.nan
                continue
            rho0_ij = 1.0 if i == j else data.rho_0[i, j]
            curves[i, j, :] = rho0_ij * (
                1.0 - np.asarray(data.gamma_emp[i][j], dtype=float) / sill_ij
            )

    curves = 0.5 * (curves + np.swapaxes(curves, 0, 1))
    return curves


def pairwise_fitted_corr_curves(data: ComparisonData, pairwise_fit, h_grid):
    """Evaluate fitted pairwise PE correlations on a distance grid."""
    nper = len(data.periods)
    curves = np.zeros((nper, nper, len(h_grid)))
    length_scales = np.asarray(pairwise_fit["LE"], dtype=float)
    exponents = np.asarray(pairwise_fit["gammaE"], dtype=float)

    for i in range(nper):
        for j in range(nper):
            rho0_ij = 1.0 if i == j else data.rho_0[i, j]
            curves[i, j, :] = rho0_ij * np.exp(
                -((np.maximum(h_grid, 1e-12) / length_scales[i, j]) ** exponents[i, j])
            )
            curves[i, j, 0] = rho0_ij

    curves = 0.5 * (curves + np.swapaxes(curves, 0, 1))
    return curves


def pca_period_pair_curves(pca_fit, pca_module, h_grid):
    nper = len(pca_fit["rho"])
    curves = np.zeros((nper, nper, len(h_grid)))
    for i in range(nper):
        for j in range(nper):
            if i == j:
                curves[i, j, :] = pca_fit["rho"][i, :]
            else:
                curves[i, j, :] = pca_module.pca_cross_correlation(
                    i,
                    j,
                    h_grid,
                    pca_fit["U"],
                    pca_fit["evals"],
                    pca_fit["rho_pc"],
                )
    curves[:, :, 0] = pca_fit["R_pool"]
    np.fill_diagonal(curves[:, :, 0], 1.0)
    return curves


def lmc_period_pair_curves(lmc_fit, lmc_module, h_grid):
    nper = len(lmc_fit["rho"])
    curves = np.zeros((nper, nper, len(h_grid)))
    for i in range(nper):
        for j in range(nper):
            if i == j:
                curves[i, j, :] = lmc_fit["rho"][i, :]
            else:
                curves[i, j, :] = lmc_module.lmc_cross_corr(
                    i,
                    j,
                    h_grid,
                    lmc_fit["B_raw"],
                    lmc_fit["length_scales"],
                    lmc_fit["gamma_pe"],
                )
    return curves


def ggp_model_a_curves(ggp_model_a, h_grid):
    ggp = ggp_model_a["module"]
    params = ggp_model_a["params"]
    nper = len(ggp_model_a["rho_0"])
    curves = np.zeros((nper, nper, len(h_grid)))
    for i in range(nper):
        for j in range(nper):
            curves[i, j, :] = ggp.cressie_chain_corr_curve(h_grid, params, i, j)
    return curves


def summarize_sample_psd_details(psd_detail):
    """Summarize PSD diagnostics for sampled station-period simulation matrices."""
    rows = []
    for method, group in psd_detail.groupby("method", sort=False):
        finite = group.dropna(subset=["min_eigenvalue"])
        if finite.empty:
            rows.append({
                "method": method,
                "min_eigenvalue": np.nan,
                "worst_sample": np.nan,
                "worst_eqid": np.nan,
                "n_samples": int(len(group)),
                "n_non_psd_samples": int(len(group)),
                "max_negative_eigenvalues": np.nan,
                "matrix_dimension_mean": float(group["matrix_dimension"].mean())
                if "matrix_dimension" in group
                else np.nan,
                "psd_ok_all": False,
            })
            continue

        worst_idx = finite["min_eigenvalue"].idxmin()
        rows.append({
            "method": method,
            "min_eigenvalue": float(finite.loc[worst_idx, "min_eigenvalue"]),
            "worst_sample": int(finite.loc[worst_idx, "sample_id"]),
            "worst_eqid": finite.loc[worst_idx, "eqid"],
            "n_samples": int(len(group)),
            "n_non_psd_samples": int((~group["psd_ok"]).sum()),
            "max_negative_eigenvalues": int(finite["negative_eigenvalues"].max()),
            "matrix_dimension_mean": float(group["matrix_dimension"].mean()),
            "psd_ok_all": bool(group["psd_ok"].all()),
        })
    return pd.DataFrame(rows)


def cross_period_curve_sets(
    data: ComparisonData,
    mle,
    semi,
    pairwise_empirical,
    gh08,
    pca,
    lmc_semivariogram,
    kronecker,
    ggp_model_a=None,
    lmc_block_mle=None,
    include_gh_mle=False,
    include_kronecker_mle=False,
):
    """Correlation curves used to construct cross-period simulation matrices."""
    pairwise_curves = (
        pairwise_fitted_corr_curves(data, pairwise_empirical["fit"], data.h_fine)
        if pairwise_empirical is not None and "fit" in pairwise_empirical
        else pairwise_empirical_corr_curves(data)
    )
    pairwise_grid = data.h_fine if pairwise_empirical is not None and "fit" in pairwise_empirical else data.h_bins
    curve_sets = {
        "Pairwise empirical semivariogram": (pairwise_grid, pairwise_curves),
        "GH08 semivariogram": (data.h_fine, gh08["gh_curves_semi"]),
        "PCA semivariogram": (
            data.h_fine,
            pca_period_pair_curves(pca["semivariogram"], semi["pca"], data.h_fine),
        ),
        "PCA MLE": (
            data.h_fine,
            pca_period_pair_curves(pca["mle"], mle["pca"], data.h_fine),
        ),
        "LMC semivariogram": (
            data.h_fine,
            lmc_period_pair_curves(lmc_semivariogram, semi["lmc"], data.h_fine),
        ),
        "Kronecker semivariogram": (data.h_fine, kronecker["curves_semivariogram"]),
    }
    if ggp_model_a is not None and len(ggp_model_a.get("rho_0", [])) == len(data.periods):
        curve_sets["GGP Model A"] = (data.h_fine, ggp_model_a_curves(ggp_model_a, data.h_fine))
    if include_gh_mle:
        curve_sets["GH08 MLE"] = (data.h_fine, gh08["gh_curves_mle"])
    if include_kronecker_mle:
        curve_sets["Kronecker MLE"] = (data.h_fine, kronecker["curves_mle"])
    if lmc_block_mle is not None:
        curve_sets["LMC block MLE"] = (
            data.h_fine,
            lmc_period_pair_curves(lmc_block_mle, mle["lmc"], data.h_fine),
        )
    return curve_sets


def _simulation_correlation_from_curves(lat, lon, h_grid, curves, period_indices=None):
    q_all = curves.shape[0]
    period_indices = np.arange(q_all) if period_indices is None else np.asarray(period_indices, dtype=int)
    q = len(period_indices)
    n = len(lat)
    dist = _haversine_between(lat, lon, lat, lon)
    corr = np.zeros((n * q, n * q))
    h_grid, curves = _curves_with_zero_for_arrays(h_grid, curves)

    for a in range(n):
        for b in range(n):
            block_all = _corr_at_distance(dist[a, b], h_grid, curves)
            block = block_all[np.ix_(period_indices, period_indices)]
            corr[a * q:(a + 1) * q, b * q:(b + 1) * q] = block
    return 0.5 * (corr + corr.T)


def _curves_with_zero_for_arrays(h_grid, curves):
    h_grid = np.asarray(h_grid, dtype=float)
    curves = np.asarray(curves, dtype=float)
    if len(h_grid) and np.isclose(h_grid[0], 0.0):
        return h_grid, curves
    curves0 = curves[:, :, [0]]
    np.fill_diagonal(curves0[:, :, 0], 1.0)
    return np.r_[0.0, h_grid], np.concatenate([curves0, curves], axis=2)


def cross_period_psd_analysis(
    data: ComparisonData,
    mle,
    semi,
    pairwise_empirical,
    gh08,
    pca,
    lmc_semivariogram,
    kronecker,
    ggp_model_a,
    lmc_block_mle=None,
    n_samples=20,
    n_periods=6,
    max_stations_per_sample=30,
    min_stations=8,
    random_state=0,
    tol=1e-10,
    curve_sets=None,
    exclude_methods=(),
):
    """PSD diagnostics for sampled station-period simulation correlation matrices."""
    rng = np.random.default_rng(random_state)
    q_total = len(data.period_cols)
    n_periods = min(int(n_periods), q_total)
    if curve_sets is None:
        curve_sets = cross_period_curve_sets(
            data=data,
            mle=mle,
            semi=semi,
            pairwise_empirical=pairwise_empirical,
            gh08=gh08,
            pca=pca,
            lmc_semivariogram=lmc_semivariogram,
            kronecker=kronecker,
            ggp_model_a=ggp_model_a,
            lmc_block_mle=lmc_block_mle,
            include_gh_mle=False,
            include_kronecker_mle=False,
        )
    else:
        curve_sets = {
            method: (np.asarray(h_grid, dtype=float), np.asarray(curves, dtype=float))
            for method, (h_grid, curves) in curve_sets.items()
        }

    excluded = set(exclude_methods or ())
    curve_sets = {
        method: value for method, value in curve_sets.items()
        if method not in excluded
    }
    if not curve_sets:
        raise ValueError("No cross-period curve sets remain after method filtering.")

    event_counts = data.df_pca.groupby("eqid").size()
    eligible_events = event_counts[event_counts >= min_stations].index.to_numpy()
    if len(eligible_events) == 0:
        raise ValueError(f"No events have at least {min_stations} complete PCA records.")

    sample_specs = []
    for sample_id in range(n_samples):
        eqid = rng.choice(eligible_events)
        df_event = data.df_pca.loc[
            data.df_pca["eqid"] == eqid,
            ["station_latitude", "station_longitude", *data.period_cols],
        ].dropna().copy()
        n_stations = min(len(df_event), int(max_stations_per_sample))
        station_rows = rng.choice(len(df_event), size=n_stations, replace=False)
        period_indices = np.sort(rng.choice(q_total, size=n_periods, replace=False))
        sample_specs.append({
            "sample_id": sample_id,
            "eqid": eqid,
            "lat": df_event.iloc[station_rows]["station_latitude"].to_numpy(dtype=float),
            "lon": df_event.iloc[station_rows]["station_longitude"].to_numpy(dtype=float),
            "period_indices": period_indices,
            "periods": ",".join(data.period_cols[i] for i in period_indices),
        })

    records = []
    for spec in sample_specs:
        for method, (h_grid, curves) in curve_sets.items():
            corr = _simulation_correlation_from_curves(
                spec["lat"],
                spec["lon"],
                h_grid,
                curves,
                period_indices=spec["period_indices"],
            )
            if np.isnan(corr).any():
                min_eval = np.nan
                negative = np.nan
                psd_ok = False
            else:
                evals = np.linalg.eigvalsh(corr)
                min_eval = float(evals.min())
                negative = int(np.sum(evals < -tol))
                psd_ok = bool(min_eval >= -tol)
            records.append({
                "method": method,
                "sample_id": int(spec["sample_id"]),
                "eqid": spec["eqid"],
                "n_stations": int(len(spec["lat"])),
                "n_periods": int(len(spec["period_indices"])),
                "matrix_dimension": int(len(spec["lat"]) * len(spec["period_indices"])),
                "periods": spec["periods"],
                "min_eigenvalue": min_eval,
                "negative_eigenvalues": negative,
                "psd_ok": psd_ok,
            })

    detail = pd.DataFrame(records)
    summary = summarize_sample_psd_details(detail)
    return {"summary": summary, "detail": detail, "curves": curve_sets, "samples": sample_specs}


def _psd_positive_part(evals, evecs):
    pos = np.maximum(np.asarray(evals, dtype=float), 0.0)
    out = (evecs * pos) @ evecs.T
    return 0.5 * (out + out.T), pos


def _positive_part_divided_difference(evals, zero_tol=1e-12):
    evals = np.asarray(evals, dtype=float)
    pos = np.maximum(evals, 0.0)
    diff = evals[:, None] - evals[None, :]
    pos_diff = pos[:, None] - pos[None, :]
    omega = np.zeros_like(diff)
    ok = np.abs(diff) > float(zero_tol)
    omega[ok] = pos_diff[ok] / diff[ok]
    same = ~ok
    local_slope = np.broadcast_to((evals > 0.0).astype(float)[:, None], diff.shape)
    omega[same] = local_slope[same]
    return np.clip(0.5 * (omega + omega.T), 0.0, 1.0)


def _cg_solve_qi_sun(operator, rhs, tol, maxiter):
    from scipy.sparse.linalg import cg

    try:
        return cg(operator, rhs, rtol=float(tol), atol=0.0, maxiter=int(maxiter))
    except TypeError:
        return cg(operator, rhs, tol=float(tol), maxiter=int(maxiter))


def _nearest_correlation_qi_sun(
    mat,
    tol=1e-8,
    max_iter=50,
    cg_max_iter=None,
    eig_floor=0.0,
    line_search_beta=0.5,
    line_search_sigma=1e-4,
):
    """Nearest correlation matrix via the Qi-Sun (2006) semismooth Newton-CG route."""
    from scipy.sparse.linalg import LinearOperator

    corr = np.asarray(mat, dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    n = corr.shape[0]
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("nearest correlation input must be a square matrix")
    if n == 0:
        return corr.copy()
    target = np.ones(n, dtype=float)
    y = 1.0 - np.diag(corr)
    cg_max_iter = int(cg_max_iter or max(20, min(200, 2 * n)))

    def project_and_state(y_vec):
        shifted = corr + np.diag(y_vec)
        evals, evecs = np.linalg.eigh(0.5 * (shifted + shifted.T))
        projected, pos_evals = _psd_positive_part(evals, evecs)
        grad = np.diag(projected) - target
        objective = 0.5 * float(pos_evals @ pos_evals) - float(target @ y_vec)
        return projected, evals, evecs, grad, objective

    projected, evals, evecs, grad, objective = project_and_state(y)
    for _ in range(int(max_iter)):
        grad_norm = float(np.linalg.norm(grad, ord=np.inf))
        if grad_norm <= float(tol):
            break

        omega = _positive_part_divided_difference(evals)

        def hessian_vec(vec):
            vec = np.asarray(vec, dtype=float)
            h_hat = evecs.T @ (vec[:, None] * evecs)
            z_hat = omega * h_hat
            uz = evecs @ z_hat
            return np.einsum("ij,ij->i", uz, evecs)

        operator = LinearOperator((n, n), matvec=hessian_vec, dtype=float)
        cg_tol = min(0.5, max(1e-6, np.sqrt(grad_norm)))
        step, info = _cg_solve_qi_sun(operator, -grad, tol=cg_tol, maxiter=cg_max_iter)
        if info != 0 or not np.all(np.isfinite(step)):
            step = -grad

        directional_derivative = float(grad @ step)
        if directional_derivative >= 0.0:
            step = -grad
            directional_derivative = float(grad @ step)

        alpha = 1.0
        accepted = False
        for _ls in range(30):
            trial_y = y + alpha * step
            trial_projected, trial_evals, trial_evecs, trial_grad, trial_objective = project_and_state(trial_y)
            if trial_objective <= objective + float(line_search_sigma) * alpha * directional_derivative:
                y = trial_y
                projected, evals, evecs, grad, objective = (
                    trial_projected,
                    trial_evals,
                    trial_evecs,
                    trial_grad,
                    trial_objective,
                )
                accepted = True
                break
            alpha *= float(line_search_beta)
        if not accepted:
            y = trial_y
            projected, evals, evecs, grad, objective = (
                trial_projected,
                trial_evals,
                trial_evecs,
                trial_grad,
                trial_objective,
            )

    out = 0.5 * (projected + projected.T)
    np.fill_diagonal(out, 1.0)
    if eig_floor and eig_floor > 0.0:
        delta = min(max(float(eig_floor), 0.0), 1.0)
        out = (1.0 - delta) * out + delta * np.eye(n)
        np.fill_diagonal(out, 1.0)
    return 0.5 * (out + out.T)


def _eigenvalue_floor_psd_matrix(mat, eps=1e-8):
    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = 0.5 * (mat + mat.T)
    evals, evecs = np.linalg.eigh(mat)
    evals = np.clip(evals, eps, None)
    out = (evecs * evals) @ evecs.T
    return 0.5 * (out + out.T)


def nearest_psd_matrix(mat, eps=1e-8, preserve_diagonal=True):
    """Project a symmetric covariance matrix to PSD, optionally preserving its diagonal.

    With preserve_diagonal=True, this uses the Qi-Sun (2006) nearest correlation
    matrix Newton-CG repair on the standardized correlation matrix, then rescales
    back to the original covariance diagonal.
    """
    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = 0.5 * (mat + mat.T)

    if not preserve_diagonal:
        return _eigenvalue_floor_psd_matrix(mat, eps=eps)

    diag = np.clip(np.diag(mat).copy(), eps, None)
    scale = np.sqrt(diag)
    corr = mat / np.outer(scale, scale)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    corr_psd = _nearest_correlation_qi_sun(corr, eig_floor=eps)
    out = corr_psd * np.outer(scale, scale)
    np.fill_diagonal(out, diag)
    return 0.5 * (out + out.T)


def _haversine_between(lat_a, lon_a, lat_b, lon_b):
    lat_a = np.radians(np.asarray(lat_a, dtype=float))
    lon_a = np.radians(np.asarray(lon_a, dtype=float))
    lat_b = np.radians(np.asarray(lat_b, dtype=float))
    lon_b = np.radians(np.asarray(lon_b, dtype=float))
    dlat = lat_a[:, None] - lat_b[None, :]
    dlon = lon_a[:, None] - lon_b[None, :]
    aa = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_a)[:, None] * np.cos(lat_b)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * 6371.0 * np.arcsin(np.minimum(1.0, np.sqrt(aa)))


def _curves_with_zero(data: ComparisonData, h_grid, curves):
    h_grid = np.asarray(h_grid, dtype=float)
    curves = np.asarray(curves, dtype=float)
    if len(h_grid) and np.isclose(h_grid[0], 0.0):
        return h_grid, curves

    rho0 = np.asarray(data.rho_0, dtype=float)
    curves0 = np.concatenate([rho0[:, :, None], curves], axis=2)
    return np.r_[0.0, h_grid], curves0


def _corr_at_distance(h, h_grid, curves):
    nper = curves.shape[0]
    mat = np.zeros((nper, nper))
    for i in range(nper):
        for j in range(nper):
            y = np.asarray(curves[i, j, :], dtype=float)
            ok = np.isfinite(y)
            if ok.sum() == 0:
                mat[i, j] = 1.0 if i == j else 0.0
            else:
                mat[i, j] = np.interp(h, h_grid[ok], y[ok])
    mat = 0.5 * (mat + mat.T)
    return mat


def _event_covariance_from_corr_curves(lat, lon, h_grid, curves, period_variances):
    q = curves.shape[0]
    n = len(lat)
    std_outer = np.sqrt(np.outer(period_variances, period_variances))
    dist = _haversine_between(lat, lon, lat, lon)
    cov = np.zeros((n * q, n * q))
    h_grid = np.asarray(h_grid, dtype=float)
    curves = np.asarray(curves, dtype=float)
    for i in range(q):
        for j in range(q):
            y = np.asarray(curves[i, j, :], dtype=float)
            ok = np.isfinite(y)
            if ok.sum() == 0:
                block = np.full_like(dist, 1.0 if i == j else 0.0, dtype=float)
            else:
                block = np.interp(dist.ravel(), h_grid[ok], y[ok]).reshape(n, n)
            cov[i::q, j::q] = block * std_outer[i, j]
    return 0.5 * (cov + cov.T)


def _event_covariance_separable(lat, lon, separable_fit, rho0, period_variances):
    q = len(rho0)
    n = len(lat)
    std_outer = np.sqrt(np.outer(period_variances, period_variances))
    period_cov = np.asarray(rho0, dtype=float) * std_outer
    dist = _haversine_between(lat, lon, lat, lon)
    length_scale = float(separable_fit["LE"])
    exponent = float(separable_fit["gammaE"])
    cov = np.zeros((n * q, n * q))
    spatial = np.exp(-((np.maximum(dist, 1e-12) / length_scale) ** exponent))
    spatial[np.isclose(dist, 0.0)] = 1.0
    for i in range(q):
        for j in range(q):
            cov[i::q, j::q] = spatial * period_cov[i, j]
    return 0.5 * (cov + cov.T)


def _event_covariance_gh08(lat, lon, direct_fit, rho0, period_variances):
    q = len(rho0)
    n = len(lat)
    std_outer = np.sqrt(np.outer(period_variances, period_variances))
    dist = _haversine_between(lat, lon, lat, lon)
    length_scales = np.asarray(direct_fit["LE"], dtype=float)
    exponents = np.asarray(direct_fit["gammaE"], dtype=float)
    cov = np.zeros((n * q, n * q))

    dist_safe = np.maximum(dist, 1e-12)
    zero_mask = np.isclose(dist, 0.0)
    for i in range(q):
        for j in range(q):
            kmax = max(i, j)
            rho_auto = np.exp(-((dist_safe / length_scales[kmax]) ** exponents[kmax]))
            rho_auto[zero_mask] = 1.0
            rho_ij = rho_auto if i == j else rho0[i, j] * rho_auto
            cov[i::q, j::q] = rho_ij * std_outer[i, j]
    return 0.5 * (cov + cov.T)


def _event_covariance_pairwise_pe(lat, lon, pairwise_fit, rho0, period_variances):
    q = len(rho0)
    n = len(lat)
    std_outer = np.sqrt(np.outer(period_variances, period_variances))
    dist = _haversine_between(lat, lon, lat, lon)
    length_scales = np.asarray(pairwise_fit["LE"], dtype=float)
    exponents = np.asarray(pairwise_fit["gammaE"], dtype=float)
    cov = np.zeros((n * q, n * q))

    dist_safe = np.maximum(dist, 1e-12)
    zero_mask = np.isclose(dist, 0.0)
    for i in range(q):
        for j in range(q):
            rho0_ij = 1.0 if i == j else rho0[i, j]
            rho_ij = rho0_ij * np.exp(
                -((dist_safe / length_scales[i, j]) ** exponents[i, j])
            )
            rho_ij[zero_mask] = rho0_ij
            cov[i::q, j::q] = rho_ij * std_outer[i, j]
    return 0.5 * (cov + cov.T)


def _event_covariance_pca(lat, lon, pca_fit):
    q = len(pca_fit["var_pool"])
    n = len(lat)
    loadings = np.asarray(pca_fit["U"], dtype=float)
    evals = np.asarray(pca_fit["evals"], dtype=float)
    std_pool = np.asarray(pca_fit["std_pool"], dtype=float)
    length_scales = np.asarray(pca_fit["LE"], dtype=float)
    exponents = np.asarray(pca_fit["gammaE"], dtype=float)
    nugget_fraction = np.asarray(
        pca_fit.get("nugget_fraction", np.zeros_like(length_scales)),
        dtype=float,
    )
    if nugget_fraction.shape != length_scales.shape:
        raise ValueError("PCA nugget fractions do not match the fitted PC kernels")
    dist = _haversine_between(lat, lon, lat, lon)
    cov = np.zeros((n * q, n * q))
    std_outer = np.outer(std_pool, std_pool)
    dist_safe = np.maximum(dist, 1e-12)
    for pc, (length_scale, exponent, nugget) in enumerate(
        zip(length_scales, exponents, nugget_fraction)
    ):
        spatial = (1.0 - nugget) * np.exp(
            -((dist_safe / length_scale) ** exponent)
        )
        # The nugget is an observation-level identity component.  It restores
        # the total PC variance only for the same station observation, not for
        # two distinct records that happen to share coordinates.
        np.fill_diagonal(spatial, 1.0)
        period_component = evals[pc] * np.outer(loadings[:, pc], loadings[:, pc]) * std_outer
        for i in range(q):
            for j in range(q):
                cov[i::q, j::q] += spatial * period_component[i, j]
    return 0.5 * (cov + cov.T)


def _simulate_pca_residuals_native_timed(lat, lon, pca_fit, n_simulations, rng, nearest_psd=True, jitter=1e-8):
    """Time native PCA residual simulation: independent PC fields, then rotate to periods."""
    import time

    n = len(lat)
    q = len(pca_fit["var_pool"])
    loadings = np.asarray(pca_fit["U"], dtype=float)
    evals = np.asarray(pca_fit["evals"], dtype=float)
    std_pool = np.asarray(pca_fit["std_pool"], dtype=float)
    length_scales = np.asarray(pca_fit["LE"], dtype=float)
    exponents = np.asarray(pca_fit["gammaE"], dtype=float)
    nugget_fraction = np.asarray(
        pca_fit.get("nugget_fraction", np.zeros_like(length_scales)),
        dtype=float,
    )
    k = min(
        loadings.shape[1],
        len(evals),
        len(length_scales),
        len(exponents),
        len(nugget_fraction),
    )

    cov_time = 0.0
    repair_time = 0.0
    factor_time = 0.0
    sample_time = 0.0
    repaired = False
    max_jitter_used = 0.0

    t0 = time.perf_counter()
    dist = _haversine_between(lat, lon, lat, lon)
    cov_time += time.perf_counter() - t0
    dist_safe = np.maximum(dist, 1e-12)

    pc_scores = np.empty((n, int(n_simulations), k), dtype=float)
    for pc in range(k):
        t0 = time.perf_counter()
        cov_pc = evals[pc] * (1.0 - nugget_fraction[pc]) * np.exp(
            -((dist_safe / length_scales[pc]) ** exponents[pc])
        )
        np.fill_diagonal(cov_pc, evals[pc])
        cov_pc = 0.5 * (cov_pc + cov_pc.T)
        cov_time += time.perf_counter() - t0

        lower = None
        last_error = None
        jitter_used = 0.0
        for repair_pass in (False, True):
            if repair_pass:
                if not nearest_psd:
                    break
                repair_t0 = time.perf_counter()
                cov_pc = nearest_psd_matrix(cov_pc, eps=1e-7, preserve_diagonal=True)
                repair_time += time.perf_counter() - repair_t0
                repaired = True

            factor_t0 = time.perf_counter()
            scale = max(float(np.mean(np.diag(cov_pc))), 1.0)
            for attempt in range(8):
                jitter_used = 0.0 if attempt == 0 else float(jitter) * scale * (10 ** (attempt - 1))
                try:
                    lower = np.linalg.cholesky(cov_pc + jitter_used * np.eye(n))
                    break
                except np.linalg.LinAlgError as exc:
                    last_error = exc
            factor_time += time.perf_counter() - factor_t0
            if lower is not None:
                break
        if lower is None:
            raise np.linalg.LinAlgError(f"PCA native Cholesky failed: {last_error}")
        max_jitter_used = max(max_jitter_used, float(jitter_used))

        sample_t0 = time.perf_counter()
        z = rng.standard_normal((n, int(n_simulations)))
        pc_scores[:, :, pc] = lower @ z
        sample_time += time.perf_counter() - sample_t0

    sample_t0 = time.perf_counter()
    sims = np.einsum("nsk,qk,q->nqs", pc_scores, loadings[:, :k], std_pool)
    sample_time += time.perf_counter() - sample_t0

    return {
        "covariance_time_seconds": cov_time,
        "psd_check_time_seconds": 0.0,
        "nearest_psd_repair_time_seconds": repair_time,
        "factorization_time_seconds": factor_time,
        "sampling_time_seconds": sample_time,
        "total_time_seconds": cov_time + repair_time + factor_time + sample_time,
        "nearest_psd_used": repaired,
        "cholesky_jitter_used": max_jitter_used,
        "sample_mean_abs": float(np.mean(np.abs(np.mean(sims.reshape(n * q, int(n_simulations)), axis=1)))),
        "success": True,
        "native_simulation": True,
        "native_component_count": int(k),
        "matrix_dimension": int(n),
        "full_dense_equivalent_dimension": int(n * q),
        "estimated_covariance_memory_mb": float(k * n ** 2 * 8 / 1e6),
    }


def _cholesky_with_optional_repair(mat, jitter=1e-8, nearest_psd=True, eps=1e-7):
    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = 0.5 * (mat + mat.T)
    lower = None
    last_error = None
    repaired = False
    repair_time = 0.0
    factor_time = 0.0
    jitter_used = 0.0

    for repair_pass in (False, True):
        if repair_pass:
            if not nearest_psd:
                break
            import time

            repair_t0 = time.perf_counter()
            mat = nearest_psd_matrix(mat, eps=eps, preserve_diagonal=True)
            repair_time += time.perf_counter() - repair_t0
            repaired = True

        import time

        factor_t0 = time.perf_counter()
        scale = max(float(np.mean(np.diag(mat))), 1.0)
        for attempt in range(8):
            jitter_used = 0.0 if attempt == 0 else float(jitter) * scale * (10 ** (attempt - 1))
            try:
                lower = np.linalg.cholesky(mat + jitter_used * np.eye(mat.shape[0]))
                break
            except np.linalg.LinAlgError as exc:
                last_error = exc
        factor_time += time.perf_counter() - factor_t0
        if lower is not None:
            break

    if lower is None:
        raise np.linalg.LinAlgError(f"Cholesky failed: {last_error}")
    return lower, repaired, repair_time, factor_time, float(jitter_used)


def _simulate_kronecker_residuals_native_timed(
    lat,
    lon,
    separable_fit,
    rho0,
    period_variances,
    n_simulations,
    rng,
    nearest_psd=True,
    jitter=1e-8,
):
    """Time separable/Kronecker simulation without building the full nq by nq matrix."""
    import time

    n = len(lat)
    q = len(rho0)
    cov_time = 0.0
    repair_time = 0.0
    factor_time = 0.0
    sample_time = 0.0
    repaired = False
    max_jitter_used = 0.0

    t0 = time.perf_counter()
    dist = _haversine_between(lat, lon, lat, lon)
    dist_safe = np.maximum(dist, 1e-12)
    spatial = np.exp(-((dist_safe / float(separable_fit["LE"])) ** float(separable_fit["gammaE"])))
    spatial[np.isclose(dist, 0.0)] = 1.0
    std_outer = np.sqrt(np.outer(period_variances, period_variances))
    period_cov = np.asarray(rho0, dtype=float) * std_outer
    cov_time += time.perf_counter() - t0

    lower_s, s_repaired, s_repair, s_factor, s_jitter = _cholesky_with_optional_repair(
        spatial,
        jitter=jitter,
        nearest_psd=nearest_psd,
    )
    lower_t, t_repaired, t_repair, t_factor, t_jitter = _cholesky_with_optional_repair(
        period_cov,
        jitter=jitter,
        nearest_psd=nearest_psd,
    )
    repaired = bool(s_repaired or t_repaired)
    repair_time += s_repair + t_repair
    factor_time += s_factor + t_factor
    max_jitter_used = max(s_jitter, t_jitter)

    sample_t0 = time.perf_counter()
    sample_mean_abs = 0.0
    for _ in range(int(n_simulations)):
        z = rng.standard_normal((n, q))
        sims = lower_s @ z @ lower_t.T
        sample_mean_abs += float(np.mean(np.abs(np.mean(sims, axis=0))))
    sample_time += time.perf_counter() - sample_t0
    sample_mean_abs /= max(int(n_simulations), 1)

    return {
        "covariance_time_seconds": cov_time,
        "psd_check_time_seconds": 0.0,
        "nearest_psd_repair_time_seconds": repair_time,
        "factorization_time_seconds": factor_time,
        "sampling_time_seconds": sample_time,
        "total_time_seconds": cov_time + repair_time + factor_time + sample_time,
        "nearest_psd_used": repaired,
        "cholesky_jitter_used": max_jitter_used,
        "sample_mean_abs": sample_mean_abs,
        "success": True,
        "native_simulation": True,
        "native_component_count": 1,
        "matrix_dimension": int(n),
        "full_dense_equivalent_dimension": int(n * q),
        "estimated_covariance_memory_mb": float((n ** 2 + q ** 2) * 8 / 1e6),
    }


def _event_covariance_lmc(lat, lon, lmc_fit, lmc_module):
    q = len(lmc_fit["rho"])
    n = len(lat)
    dist = _haversine_between(lat, lon, lat, lon)
    b_list = lmc_fit["B_raw"]
    length_scales = np.asarray(lmc_fit["length_scales"], dtype=float)
    gamma_pe = np.atleast_1d(np.asarray(lmc_fit["gamma_pe"], dtype=float))
    if len(gamma_pe) == 1:
        gamma_pe = np.repeat(gamma_pe, len(length_scales))

    cov = np.zeros((n * q, n * q))
    dist_safe = np.maximum(dist, 1e-12)
    zero_mask = np.isclose(dist, 0.0)
    for b_mat, length_scale, exponent in zip(b_list, length_scales, gamma_pe):
        if hasattr(lmc_module, "corr_kernel"):
            try:
                kernel = np.asarray(lmc_module.corr_kernel(dist, length_scale, exponent), dtype=float)
            except Exception:
                kernel = np.exp(-((dist_safe / length_scale) ** exponent))
        elif float(length_scale) <= 0.0:
            kernel = np.zeros_like(dist)
        else:
            kernel = np.exp(-((dist_safe / length_scale) ** exponent))
        kernel[zero_mask] = 1.0
        for i in range(q):
            for j in range(q):
                cov[i::q, j::q] += kernel * b_mat[i, j]
    return 0.5 * (cov + cov.T)


def _matern_correlation_native(distance, nu, alpha):
    """Rate-parameterized Matérn correlation evaluated at arbitrary distances."""
    from scipy.special import gammaln, kv

    distance = np.asarray(distance, dtype=float)
    x = float(alpha) * np.abs(distance)
    out = np.ones_like(x, dtype=float)
    positive = x > 0.0
    xp = x[positive]
    if xp.size:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            log_value = (1.0 - float(nu)) * np.log(2.0) - gammaln(float(nu))
            log_value += float(nu) * np.log(xp) + np.log(kv(float(nu), xp))
            value = np.exp(log_value)
        out[positive] = np.clip(np.where(np.isfinite(value), value, 0.0), 0.0, 1.0)
    return out


def _event_covariance_multivariate_matern(
    lat,
    lon,
    multivariate_matern,
    period_variances,
):
    """Build the fitted AGS Matérn covariance directly on an event distance matrix."""
    rho = np.asarray(multivariate_matern["rho_fitted"], dtype=float)
    alpha = np.asarray(multivariate_matern["alpha_per_km"], dtype=float)
    nu = np.asarray(multivariate_matern["nu"], dtype=float)
    q = len(period_variances)
    n = len(lat)
    expected = (q, q)
    if rho.shape != expected or alpha.shape != expected or nu.shape != expected:
        raise ValueError(
            "Multivariate Matérn parameter matrices do not match the prediction periods: "
            f"rho={rho.shape}, alpha={alpha.shape}, nu={nu.shape}, expected={expected}"
        )

    distance = _haversine_between(lat, lon, lat, lon)
    std_outer = np.sqrt(np.outer(period_variances, period_variances))
    cov = np.zeros((n * q, n * q), dtype=float)
    for i in range(q):
        for j in range(q):
            correlation = rho[i, j] * _matern_correlation_native(
                distance,
                nu=nu[i, j],
                alpha=alpha[i, j],
            )
            cov[i::q, j::q] = correlation * std_outer[i, j]
    return 0.5 * (cov + cov.T)


def _event_covariance_sbss(lat, lon, sbss_semivariogram, period_variances):
    """Reconstruct the SBSS covariance from its mixing matrix and latent PE kernels."""
    mixing = np.asarray(sbss_semivariogram["mixing_standardized"], dtype=float)
    parameters = sbss_semivariogram["latent_parameters"].copy()
    if "latent_component" in parameters:
        parameters = parameters.sort_values("latent_component")
    length_scales = parameters["length_scale_km"].to_numpy(dtype=float)
    exponents = parameters["powered_exponent"].to_numpy(dtype=float)

    q = len(period_variances)
    n = len(lat)
    if mixing.shape != (q, q) or len(length_scales) != q or len(exponents) != q:
        raise ValueError(
            "SBSS mixing/kernel dimensions do not match the prediction periods: "
            f"mixing={mixing.shape}, kernels={len(length_scales)}, expected={(q, q)}"
        )

    distance = _haversine_between(lat, lon, lat, lon)
    latent_correlation = np.exp(
        -(
            np.maximum(distance[:, :, None], 0.0)
            / length_scales[None, None, :]
        ) ** exponents[None, None, :]
    )
    covariance_zero = mixing @ mixing.T
    scale_zero = np.sqrt(np.clip(np.diag(covariance_zero), 1.0e-15, None))
    std_outer = np.sqrt(np.outer(period_variances, period_variances))

    cov = np.zeros((n * q, n * q), dtype=float)
    for i in range(q):
        for j in range(q):
            latent_weights = mixing[i] * mixing[j]
            block = np.tensordot(latent_correlation, latent_weights, axes=([2], [0]))
            block /= scale_zero[i] * scale_zero[j]
            cov[i::q, j::q] = block * std_outer[i, j]
    return 0.5 * (cov + cov.T)


def _earth_chord_coordinates_native(longitude, latitude):
    """Return the same 3-D Earth-chord coordinates used by the IOX WLS fit."""
    earth_radius_km = 6371.0088
    lon = np.radians(np.asarray(longitude, dtype=float))
    lat = np.radians(np.asarray(latitude, dtype=float))
    return earth_radius_km * np.column_stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
    )


def _iox_maxmin_order(coordinates):
    """Replicate the IOX fit's deterministic farthest-point event ordering."""
    coordinates = np.asarray(coordinates, dtype=float)
    n = len(coordinates)
    if n <= 2:
        return np.arange(n, dtype=np.int64)

    center = coordinates.mean(axis=0)
    from_center = np.sum((coordinates - center) ** 2, axis=1)
    first = int(np.argmax(from_center))
    selected = np.zeros(n, dtype=bool)
    selected[first] = True
    order = np.empty(n, dtype=np.int64)
    order[0] = first
    min_distance_sq = np.sum((coordinates - coordinates[first]) ** 2, axis=1)
    min_distance_sq[first] = -np.inf
    for k in range(1, n):
        next_index = int(np.argmax(min_distance_sq))
        order[k] = next_index
        selected[next_index] = True
        distance_sq = np.sum((coordinates - coordinates[next_index]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
        min_distance_sq[selected] = -np.inf
    return order


def _event_covariance_iox(
    lat,
    lon,
    iox_full_semivariogram,
    period_variances,
):
    """Build the exact event-specific Cholesky IOX covariance in max-min order."""
    marginal = iox_full_semivariogram["marginal_parameters"]
    periods = np.asarray(iox_full_semivariogram["periods"], dtype=float)
    sigma = np.asarray(iox_full_semivariogram["sigma_core"], dtype=float)
    q = len(period_variances)
    n = len(lat)
    if len(periods) != q or sigma.shape != (q, q):
        raise ValueError(
            "IOX parameter dimensions do not match the prediction periods: "
            f"periods={len(periods)}, sigma={sigma.shape}, expected={(q, q)}"
        )

    marginal_periods = marginal["period_s"].to_numpy(dtype=float)
    row_indices = []
    for period in periods:
        matches = np.flatnonzero(np.isclose(marginal_periods, period, rtol=0.0, atol=1.0e-10))
        if len(matches) != 1:
            raise ValueError(f"Expected one IOX marginal row for period {period:g}s; found {len(matches)}")
        row_indices.append(int(matches[0]))
    marginal = marginal.iloc[row_indices]
    pe = "length_scale_km" in marginal
    phi = (1.0 / marginal["length_scale_km"].to_numpy(float) if pe
           else marginal["phi_per_km"].to_numpy(float))
    nu = marginal["gamma_pe" if pe else "nu"].to_numpy(float)
    nugget_fraction = marginal["nugget_fraction" if pe else "nugget_fraction_alpha"].to_numpy(float)

    coordinates = _earth_chord_coordinates_native(lon, lat)
    order = _iox_maxmin_order(coordinates)
    coordinates_ordered = coordinates[order]
    distance = np.linalg.norm(
        coordinates_ordered[:, None, :] - coordinates_ordered[None, :, :],
        axis=2,
    )

    factors = []
    for i in range(q):
        correlation = (1.0 - nugget_fraction[i]) * (
            np.exp(-(distance * phi[i])**nu[i]) if pe else
            _matern_correlation_native(distance, nu=nu[i], alpha=phi[i]))
        np.fill_diagonal(correlation, 1.0)
        correlation = 0.5 * (correlation + correlation.T)
        factor = np.linalg.cholesky(correlation)
        factors.append(factor)

    inverse_order = np.empty(n, dtype=np.int64)
    inverse_order[order] = np.arange(n, dtype=np.int64)
    sigma_diagonal = np.clip(np.diag(sigma), 1.0e-15, None)
    shared_scale = np.sqrt(np.asarray(period_variances, dtype=float) / sigma_diagonal)
    cov = np.zeros((n * q, n * q), dtype=float)
    for i in range(q):
        for j in range(q):
            block_ordered = sigma[i, j] * (factors[i] @ factors[j].T)
            block = block_ordered[np.ix_(inverse_order, inverse_order)]
            block *= shared_scale[i] * shared_scale[j]
            cov[i::q, j::q] = block
    return 0.5 * (cov + cov.T)


def _simulate_lmc_residuals_native_timed(
    lat,
    lon,
    lmc_fit,
    lmc_module,
    n_simulations,
    rng,
    nearest_psd=True,
    jitter=1e-8,
):
    """Time LMC simulation through component spatial fields and B-matrix mixing."""
    import time

    n = len(lat)
    q = len(lmc_fit["rho"])
    b_list = lmc_fit["B_raw"]
    length_scales = np.asarray(lmc_fit["length_scales"], dtype=float)
    gamma_pe = np.atleast_1d(np.asarray(lmc_fit["gamma_pe"], dtype=float))
    if len(gamma_pe) == 1:
        gamma_pe = np.repeat(gamma_pe, len(length_scales))

    cov_time = 0.0
    repair_time = 0.0
    factor_time = 0.0
    sample_time = 0.0
    repaired = False
    max_jitter_used = 0.0
    spatial_lowers = []
    period_lowers = []

    t0 = time.perf_counter()
    dist = _haversine_between(lat, lon, lat, lon)
    dist_safe = np.maximum(dist, 1e-12)
    zero_mask = np.isclose(dist, 0.0)
    cov_time += time.perf_counter() - t0

    for b_mat, length_scale, exponent in zip(b_list, length_scales, gamma_pe):
        t0 = time.perf_counter()
        if hasattr(lmc_module, "corr_kernel"):
            try:
                kernel = np.asarray(lmc_module.corr_kernel(dist, length_scale, exponent), dtype=float)
            except Exception:
                kernel = np.exp(-((dist_safe / length_scale) ** exponent)) if float(length_scale) > 0.0 else np.zeros_like(dist)
        elif float(length_scale) <= 0.0:
            kernel = np.zeros_like(dist)
        else:
            kernel = np.exp(-((dist_safe / length_scale) ** exponent))
        kernel[zero_mask] = 1.0
        cov_time += time.perf_counter() - t0

        lower_s, s_repaired, s_repair, s_factor, s_jitter = _cholesky_with_optional_repair(
            kernel,
            jitter=jitter,
            nearest_psd=nearest_psd,
        )
        lower_b, b_repaired, b_repair, b_factor, b_jitter = _cholesky_with_optional_repair(
            b_mat,
            jitter=jitter,
            nearest_psd=nearest_psd,
        )
        spatial_lowers.append(lower_s)
        period_lowers.append(lower_b)
        repaired = bool(repaired or s_repaired or b_repaired)
        repair_time += s_repair + b_repair
        factor_time += s_factor + b_factor
        max_jitter_used = max(max_jitter_used, s_jitter, b_jitter)

    sample_t0 = time.perf_counter()
    sample_mean_abs = 0.0
    for _ in range(int(n_simulations)):
        sims = np.zeros((n, q), dtype=float)
        for lower_s, lower_b in zip(spatial_lowers, period_lowers):
            z = rng.standard_normal((n, q))
            sims += lower_s @ z @ lower_b.T
        sample_mean_abs += float(np.mean(np.abs(np.mean(sims, axis=0))))
    sample_time += time.perf_counter() - sample_t0
    sample_mean_abs /= max(int(n_simulations), 1)

    return {
        "covariance_time_seconds": cov_time,
        "psd_check_time_seconds": 0.0,
        "nearest_psd_repair_time_seconds": repair_time,
        "factorization_time_seconds": factor_time,
        "sampling_time_seconds": sample_time,
        "total_time_seconds": cov_time + repair_time + factor_time + sample_time,
        "nearest_psd_used": repaired,
        "cholesky_jitter_used": max_jitter_used,
        "sample_mean_abs": sample_mean_abs,
        "success": True,
        "native_simulation": True,
        "native_component_count": int(len(spatial_lowers)),
        "matrix_dimension": int(n),
        "full_dense_equivalent_dimension": int(n * q),
        "estimated_covariance_memory_mb": float((len(spatial_lowers) * n ** 2 + len(period_lowers) * q ** 2) * 8 / 1e6),
    }


def _event_covariance_ggp_model_a(lat, lon, ggp_model_a):
    ggp = ggp_model_a["module"]
    params = ggp_model_a["params"]
    q = len(ggp_model_a["rho_0"])
    n = len(lat)
    dist = _haversine_between(lat, lon, lat, lon)
    cov = np.zeros((n * q, n * q))
    for a in range(n):
        for b in range(n):
            h = dist[a, b]
            block = np.zeros((q, q))
            for i in range(q):
                for j in range(q):
                    block[i, j] = ggp.cressie_chain_cov_curve(np.array([h]), params, i, j)[0]
            cov[a * q:(a + 1) * q, b * q:(b + 1) * q] = block
    return 0.5 * (cov + cov.T)


def _mvn_nlpd(residual, cov, jitter=1e-6):
    cov = 0.5 * (cov + cov.T)
    cov = cov + jitter * np.eye(cov.shape[0])
    try:
        lower = np.linalg.cholesky(cov)
        alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, residual))
        logdet = 2.0 * np.sum(np.log(np.diag(lower)))
    except np.linalg.LinAlgError:
        cov = nearest_psd_matrix(cov, eps=jitter, preserve_diagonal=True)
        lower = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
        alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, residual))
        logdet = 2.0 * np.sum(np.log(np.diag(lower)))
    return float(0.5 * (len(residual) * np.log(2.0 * np.pi) + logdet + residual @ alpha))


def _covariance_for_prediction_method(
    method,
    lat,
    lon,
    data,
    curve_sets,
    period_variances,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    lmc_semivariogram=None,
    lmc_mle=None,
    lmc_block_mle=None,
    gh08=None,
    kronecker=None,
    ggp_model_a=None,
    iox_full_semivariogram=None,
    sbss_semivariogram=None,
    multivariate_matern=None,
):
    if (
        method == "Pairwise empirical semivariogram"
        and pairwise_empirical is not None
        and "fit" in pairwise_empirical
    ):
        return _event_covariance_pairwise_pe(
            lat,
            lon,
            pairwise_empirical["fit"],
            data.rho_0,
            period_variances,
        )
    if method == "PCA semivariogram" and pca is not None:
        return _event_covariance_pca(lat, lon, pca["semivariogram"])
    if method == "PCA MLE" and pca is not None:
        return _event_covariance_pca(lat, lon, pca["mle"])
    if method == "LMC semivariogram" and lmc_semivariogram is not None and semi is not None:
        return _event_covariance_lmc(lat, lon, lmc_semivariogram, semi["lmc"])
    if method == "LMC MLE" and lmc_mle is not None and semi is not None:
        return _event_covariance_lmc(lat, lon, lmc_mle, semi["lmc"])
    if method == "LMC block MLE" and lmc_block_mle is not None and semi is not None:
        return _event_covariance_lmc(lat, lon, lmc_block_mle, semi["lmc"])
    if method == "GH08 semivariogram" and gh08 is not None:
        return _event_covariance_gh08(
            lat,
            lon,
            gh08["direct_semi"],
            data.rho_0,
            period_variances,
        )
    if method == "GH08 MLE" and gh08 is not None:
        return _event_covariance_gh08(
            lat,
            lon,
            gh08["direct_mle"],
            data.rho_0,
            period_variances,
        )
    if method == "Kronecker semivariogram" and kronecker is not None:
        return _event_covariance_separable(
            lat,
            lon,
            kronecker["semivariogram"],
            data.rho_0,
            period_variances,
        )
    if method == "Kronecker MLE" and kronecker is not None:
        return _event_covariance_separable(
            lat,
            lon,
            kronecker["mle"],
            data.rho_0,
            period_variances,
        )
    if method == "GGP Model A" and ggp_model_a is not None:
        return _event_covariance_ggp_model_a(lat, lon, ggp_model_a)
    if method == "IOX full semivariogram" and iox_full_semivariogram is not None:
        return _event_covariance_iox(
            lat,
            lon,
            iox_full_semivariogram,
            period_variances,
        )
    if method == "SBSS semivariogram" and sbss_semivariogram is not None:
        return _event_covariance_sbss(
            lat,
            lon,
            sbss_semivariogram,
            period_variances,
        )
    if method == "Multivariate Matern semivariogram" and multivariate_matern is not None:
        return _event_covariance_multivariate_matern(
            lat,
            lon,
            multivariate_matern,
            period_variances,
        )

    h_grid, curves = curve_sets[method]
    return _event_covariance_from_corr_curves(
        lat,
        lon,
        h_grid,
        curves,
        period_variances,
    )


def predictive_accuracy_analysis(
    data: ComparisonData,
    cross_period_psd,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    gh08=None,
    lmc_semivariogram=None,
    lmc_mle=None,
    lmc_block_mle=None,
    kronecker=None,
    ggp_model_a=None,
    max_events=20,
    max_stations_per_event=50,
    test_fraction=0.25,
    min_stations=8,
    random_state=0,
    nearest_psd=True,
    methods=None,
    iox_full_semivariogram=None,
    sbss_semivariogram=None,
    multivariate_matern=None,
):
    """Hold out stations within events and compare conditional Gaussian predictions."""
    rng = np.random.default_rng(random_state)
    period_variances = np.var(data.df_pca[data.period_cols].to_numpy(dtype=float), axis=0)
    curve_sets = {
        name: _curves_with_zero(data, h_grid, curves)
        for name, (h_grid, curves) in cross_period_psd["curves"].items()
    }
    if methods is not None:
        methods = set(methods)
        curve_sets = {name: value for name, value in curve_sets.items() if name in methods}

    event_counts = data.df_pca.groupby("eqid").size().sort_values(ascending=False)
    event_ids = event_counts.index[:max_events]
    records = []

    for eqid in event_ids:
        df_event = data.df_pca.loc[
            data.df_pca["eqid"] == eqid,
            ["station_latitude", "station_longitude", *data.period_cols],
        ].dropna().copy()
        if len(df_event) < min_stations:
            continue

        if max_stations_per_event is not None and int(max_stations_per_event) > 0 and len(df_event) > int(max_stations_per_event):
            sample_seed = int(random_state + int(eqid)) if np.issubdtype(type(eqid), np.integer) else random_state
            df_event = df_event.sample(int(max_stations_per_event), random_state=sample_seed)

        n = len(df_event)
        n_test = max(1, int(round(test_fraction * n)))
        n_test = min(n_test, n - 1)
        perm = rng.permutation(n)
        test_idx = np.sort(perm[:n_test])
        train_idx = np.sort(perm[n_test:])

        y_all = df_event[data.period_cols].to_numpy(dtype=float)
        y_train = y_all[train_idx, :].reshape(-1)
        y_test = y_all[test_idx, :].reshape(-1)
        lat = df_event["station_latitude"].to_numpy(dtype=float)
        lon = df_event["station_longitude"].to_numpy(dtype=float)
        q = len(data.period_cols)

        train_vars = np.concatenate([np.arange(i * q, (i + 1) * q) for i in train_idx])
        test_vars = np.concatenate([np.arange(i * q, (i + 1) * q) for i in test_idx])

        for method, (h_grid, curves) in curve_sets.items():
            cov_joint = _covariance_for_prediction_method(
                method,
                lat,
                lon,
                data,
                curve_sets,
                period_variances,
                pairwise_empirical=pairwise_empirical,
                pca=pca,
                semi=semi,
                gh08=gh08,
                lmc_semivariogram=lmc_semivariogram,
                lmc_mle=lmc_mle,
                lmc_block_mle=lmc_block_mle,
                kronecker=kronecker,
                ggp_model_a=ggp_model_a,
                iox_full_semivariogram=iox_full_semivariogram,
                sbss_semivariogram=sbss_semivariogram,
                multivariate_matern=multivariate_matern,
            )
            min_eval_before = float(np.linalg.eigvalsh(cov_joint).min())
            repaired = bool(min_eval_before < -1e-8 or np.isnan(min_eval_before))
            if nearest_psd and repaired:
                cov_joint = nearest_psd_matrix(cov_joint, eps=1e-7, preserve_diagonal=True)

            k_tt = cov_joint[np.ix_(train_vars, train_vars)]
            k_st = cov_joint[np.ix_(test_vars, train_vars)]
            k_ss = cov_joint[np.ix_(test_vars, test_vars)]
            k_tt = nearest_psd_matrix(k_tt, eps=1e-7, preserve_diagonal=True) if nearest_psd else k_tt
            k_tt_j = k_tt + 1e-6 * np.eye(k_tt.shape[0])

            try:
                alpha = np.linalg.solve(k_tt_j, y_train)
                pred = k_st @ alpha
                solved = np.linalg.solve(k_tt_j, k_st.T)
            except np.linalg.LinAlgError:
                alpha = np.linalg.pinv(k_tt_j) @ y_train
                pred = k_st @ alpha
                solved = np.linalg.pinv(k_tt_j) @ k_st.T
                repaired = True

            pred_cov = k_ss - k_st @ solved
            if nearest_psd:
                pred_cov = nearest_psd_matrix(pred_cov, eps=1e-7, preserve_diagonal=True)

            err = y_test - pred
            pred_sd = np.sqrt(np.clip(np.diag(pred_cov), 0.0, None))
            crps_values = _normal_crps(err, pred_sd)
            cover = {}
            for label, z in [("50", 0.67448975), ("80", 1.28155157), ("90", 1.64485363), ("95", 1.95996398)]:
                cover[f"coverage_{label}_count"] = int(np.sum(np.abs(err) <= z * pred_sd))
            n_obs = len(err)
            records.append({
                "method": method,
                "eqid": eqid,
                "n_train_stations": len(train_idx),
                "n_test_stations": len(test_idx),
                "n_observations": n_obs,
                "sse": float(np.sum(err ** 2)),
                "sae": float(np.sum(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "nlpd": _mvn_nlpd(err, pred_cov),
                "mean_crps": float(np.mean(crps_values)),
                "crps_sum": float(np.sum(crps_values)),
                "mean_pred_sd": float(np.mean(pred_sd)),
                "pred_sd_sum": float(np.sum(pred_sd)),
                "mean_interval95_width": float(np.mean(2.0 * 1.95996398 * pred_sd)),
                "interval95_width_sum": float(np.sum(2.0 * 1.95996398 * pred_sd)),
                "nearest_psd_used": repaired,
                "min_eigenvalue_before_repair": min_eval_before,
                **cover,
            })

    detail = pd.DataFrame(records)
    summary_rows = []
    for method, group in detail.groupby("method", sort=False):
        n_obs = group["n_observations"].sum()
        summary_rows.append({
            "method": method,
            "rmse": float(np.sqrt(group["sse"].sum() / n_obs)),
            "mae": float(group["sae"].sum() / n_obs),
            "mean_nlpd": float(group["nlpd"].sum() / n_obs),
            "mean_crps": float(group["crps_sum"].sum() / n_obs) if "crps_sum" in group else np.nan,
            "n_events": int(group["eqid"].nunique()),
            "n_observations": int(n_obs),
            "nearest_psd_repairs": int(group["nearest_psd_used"].sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values("rmse").reset_index(drop=True)
    return {"summary": summary, "detail": detail}


def predictive_station_holdout_repeated_analysis(
    data: ComparisonData,
    cross_period_psd,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    gh08=None,
    lmc_semivariogram=None,
    lmc_mle=None,
    lmc_block_mle=None,
    kronecker=None,
    ggp_model_a=None,
    methods=None,
    n_repeats=20,
    random_state=0,
    iox_full_semivariogram=None,
    sbss_semivariogram=None,
    multivariate_matern=None,
    **kwargs,
):
    """Repeat station holdout without refitting model parameters."""
    outputs = []
    for rep in range(int(n_repeats)):
        out = predictive_accuracy_analysis(
            data=data,
            cross_period_psd=cross_period_psd,
            pairwise_empirical=pairwise_empirical,
            pca=pca,
            semi=semi,
            gh08=gh08,
            lmc_semivariogram=lmc_semivariogram,
            lmc_mle=lmc_mle,
            lmc_block_mle=lmc_block_mle,
            kronecker=kronecker,
            ggp_model_a=ggp_model_a,
            iox_full_semivariogram=iox_full_semivariogram,
            sbss_semivariogram=sbss_semivariogram,
            multivariate_matern=multivariate_matern,
            methods=methods,
            random_state=int(random_state) + rep,
            **kwargs,
        )
        detail = out["detail"].copy()
        detail["repeat"] = rep
        detail["task"] = "station_holdout"
        outputs.append(detail)
    detail = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    good = detail.loc[detail["n_observations"] > 0].copy() if not detail.empty else detail
    summary = _summarize_predictive_detail(good, ["method"]) if not good.empty else pd.DataFrame()
    return {"summary": summary, "detail": detail}


def predictive_random_station_period_holdout_analysis(
    data: ComparisonData,
    cross_period_psd,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    gh08=None,
    lmc_semivariogram=None,
    lmc_mle=None,
    lmc_block_mle=None,
    kronecker=None,
    ggp_model_a=None,
    methods=None,
    mask_fraction=0.10,
    n_repeats=20,
    max_events=5,
    max_stations_per_event=60,
    min_stations=8,
    random_state=0,
    nearest_psd=True,
    event_ids=None,
    iox_full_semivariogram=None,
    sbss_semivariogram=None,
    multivariate_matern=None,
):
    """Mask random station-period observations within each event and predict from the rest."""
    rng = np.random.default_rng(random_state)
    period_variances = np.var(data.df_pca[data.period_cols].to_numpy(dtype=float), axis=0)
    curve_sets = _prediction_curve_sets(data, cross_period_psd, methods=methods)
    event_ids = _prediction_event_ids(data.df_pca, max_events) if event_ids is None else list(event_ids)
    q = len(data.period_cols)
    records = []

    for rep in range(int(n_repeats)):
        for eqid in event_ids:
            df_event = data.df_pca.loc[
                data.df_pca["eqid"] == eqid,
                ["station_latitude", "station_longitude", *data.period_cols],
            ].dropna(subset=data.period_cols).copy()
            if len(df_event) < min_stations:
                continue
            df_event = _sample_event_frame(df_event, max_stations_per_event, random_state + rep, eqid)
            n = len(df_event)
            y_flat = df_event[data.period_cols].to_numpy(dtype=float).reshape(-1)
            lat = df_event["station_latitude"].to_numpy(dtype=float)
            lon = df_event["station_longitude"].to_numpy(dtype=float)
            n_total = len(y_flat)
            n_test = max(1, int(round(float(mask_fraction) * n_total)))
            n_test = min(n_test, n_total - 1)
            test_vars = np.sort(rng.choice(n_total, size=n_test, replace=False))
            train_vars = np.setdiff1d(np.arange(n_total), test_vars)

            for method in curve_sets:
                base = {
                    "task": "random_station_period_holdout",
                    "method": method,
                    "eqid": eqid,
                    "repeat": rep,
                    "n_stations": int(n),
                    "mask_fraction": float(mask_fraction),
                }
                try:
                    cov_joint = _covariance_for_prediction_method(
                        method,
                        lat,
                        lon,
                        data,
                        curve_sets,
                        period_variances,
                        pairwise_empirical=pairwise_empirical,
                        pca=pca,
                        semi=semi,
                        gh08=gh08,
                        lmc_semivariogram=lmc_semivariogram,
                        lmc_mle=lmc_mle,
                        lmc_block_mle=lmc_block_mle,
                        kronecker=kronecker,
                        ggp_model_a=ggp_model_a,
                        iox_full_semivariogram=iox_full_semivariogram,
                        sbss_semivariogram=sbss_semivariogram,
                        multivariate_matern=multivariate_matern,
                    )
                    records.append({
                        **base,
                        "fit_success": True,
                        "error": "",
                        **_conditional_prediction_metrics(
                            y_flat,
                            cov_joint,
                            train_vars,
                            test_vars,
                            nearest_psd=nearest_psd,
                        ),
                    })
                except Exception as exc:
                    records.append({
                        **base,
                        "fit_success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "n_observations": 0,
                        "sse": np.nan,
                        "sae": np.nan,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "bias": np.nan,
                        "nlpd": np.nan,
                        "mean_crps": np.nan,
                        "crps_sum": np.nan,
                        "mean_pred_sd": np.nan,
                        "pred_sd_sum": np.nan,
                        "mean_interval95_width": np.nan,
                        "interval95_width_sum": np.nan,
                        "nearest_psd_used": False,
                        "min_eigenvalue_before_repair": np.nan,
                        "coverage_50_count": 0,
                        "coverage_80_count": 0,
                        "coverage_90_count": 0,
                        "coverage_95_count": 0,
                    })

    detail = pd.DataFrame(records)
    good = detail.loc[detail["fit_success"] & (detail["n_observations"] > 0)].copy()
    summary = _summarize_predictive_detail(good, ["method"]) if not good.empty else pd.DataFrame()
    return {"summary": summary, "detail": detail}


def _prediction_curve_sets(data, cross_period_psd, methods=None):
    curve_sets = {
        name: _curves_with_zero(data, h_grid, curves)
        for name, (h_grid, curves) in cross_period_psd["curves"].items()
    }
    if methods is None:
        return curve_sets
    return {name: curve_sets[name] for name in methods if name in curve_sets}


def _prediction_event_ids(df, max_events):
    event_counts = df.groupby("eqid").size().sort_values(ascending=False)
    if max_events is None or int(max_events) <= 0:
        return event_counts.index
    return event_counts.index[: int(max_events)]


def _sample_event_frame(df_event, max_stations_per_event, random_state, eqid):
    if max_stations_per_event is None or int(max_stations_per_event) <= 0:
        return df_event
    if len(df_event) <= int(max_stations_per_event):
        return df_event
    try:
        sample_seed = int(random_state) + int(eqid)
    except Exception:
        sample_seed = int(random_state)
    return df_event.sample(int(max_stations_per_event), random_state=sample_seed)


def _normalize_period_holdout_sets(periods, holdout_period_sets):
    periods = np.asarray(periods, dtype=float)
    if holdout_period_sets is None:
        holdout_period_sets = [periods[-1]]
    if isinstance(holdout_period_sets, str):
        if holdout_period_sets == "all_single":
            holdout_period_sets = list(periods)
        elif holdout_period_sets == "interior_single":
            holdout_period_sets = list(periods[1:-1])
        elif holdout_period_sets == "high_end":
            holdout_period_sets = [periods[-1]]
        else:
            raise ValueError(f"Unknown holdout_period_sets shortcut: {holdout_period_sets}")

    scenarios = []
    for spec in holdout_period_sets:
        raw_values = [spec] if np.isscalar(spec) else list(spec)
        indices = []
        for value in raw_values:
            if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(periods):
                idx = int(value)
            else:
                matches = np.where(np.isclose(periods, float(value), rtol=1e-5, atol=1e-8))[0]
                if len(matches) == 0:
                    raise ValueError(f"Cannot find held-out period {value} in {periods.tolist()}")
                idx = int(matches[0])
            indices.append(idx)
        indices = sorted(set(indices))
        scenarios.append({
            "scenario": "T=" + ",".join(f"{periods[i]:g}" for i in indices),
            "heldout_period_indices": np.asarray(indices, dtype=int),
            "heldout_period_secs": ",".join(f"{periods[i]:g}" for i in indices),
        })
    return scenarios


def _normal_crps(error, sd):
    """Univariate Gaussian CRPS for errors y - mean and predictive standard deviations."""
    from scipy.special import erf

    error = np.asarray(error, dtype=float)
    sd = np.asarray(sd, dtype=float)
    sd = np.maximum(sd, 1e-12)
    z = error / sd
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    return sd * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))


def _conditional_prediction_metrics(y_flat, cov_joint, train_vars, test_vars, nearest_psd=True, jitter=1e-6):
    train_vars = np.asarray(train_vars, dtype=int)
    test_vars = np.asarray(test_vars, dtype=int)
    cov_joint = 0.5 * (cov_joint + cov_joint.T)
    try:
        min_eval_before = float(np.linalg.eigvalsh(cov_joint).min())
    except Exception:
        min_eval_before = np.nan
    repaired = bool((not np.isfinite(min_eval_before)) or min_eval_before < -1e-8)
    if nearest_psd and repaired:
        cov_joint = nearest_psd_matrix(cov_joint, eps=1e-7, preserve_diagonal=True)

    k_tt = cov_joint[np.ix_(train_vars, train_vars)]
    k_st = cov_joint[np.ix_(test_vars, train_vars)]
    k_ss = cov_joint[np.ix_(test_vars, test_vars)]
    if nearest_psd:
        k_tt = nearest_psd_matrix(k_tt, eps=1e-7, preserve_diagonal=True)
    k_tt_j = k_tt + jitter * np.eye(k_tt.shape[0])

    y_train = y_flat[train_vars]
    y_test = y_flat[test_vars]
    try:
        alpha = np.linalg.solve(k_tt_j, y_train)
        solved = np.linalg.solve(k_tt_j, k_st.T)
    except np.linalg.LinAlgError:
        k_tt_pinv = np.linalg.pinv(k_tt_j)
        alpha = k_tt_pinv @ y_train
        solved = k_tt_pinv @ k_st.T
        repaired = True

    pred = k_st @ alpha
    pred_cov = 0.5 * (k_ss - k_st @ solved + (k_ss - k_st @ solved).T)
    if nearest_psd:
        pred_cov = nearest_psd_matrix(pred_cov, eps=1e-7, preserve_diagonal=True)

    err = y_test - pred
    pred_sd = np.sqrt(np.clip(np.diag(pred_cov), 0.0, None))
    crps_values = _normal_crps(err, pred_sd)
    cover = {}
    for label, z in [("50", 0.67448975), ("80", 1.28155157), ("90", 1.64485363), ("95", 1.95996398)]:
        cover[f"coverage_{label}_count"] = int(np.sum(np.abs(err) <= z * pred_sd))
    n_obs = len(err)
    return {
        "n_observations": int(n_obs),
        "sse": float(np.sum(err ** 2)),
        "sae": float(np.sum(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "nlpd": _mvn_nlpd(err, pred_cov, jitter=jitter),
        "mean_crps": float(np.mean(crps_values)),
        "crps_sum": float(np.sum(crps_values)),
        "mean_pred_sd": float(np.mean(pred_sd)),
        "pred_sd_sum": float(np.sum(pred_sd)),
        "mean_interval95_width": float(np.mean(2.0 * 1.95996398 * pred_sd)),
        "interval95_width_sum": float(np.sum(2.0 * 1.95996398 * pred_sd)),
        "nearest_psd_used": repaired,
        "min_eigenvalue_before_repair": min_eval_before,
        **cover,
    }


def _summarize_predictive_detail(detail, group_cols):
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in detail.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n_obs = int(group["n_observations"].sum())
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            "rmse": float(np.sqrt(group["sse"].sum() / n_obs)),
            "mae": float(group["sae"].sum() / n_obs),
            "mean_bias": float(np.average(group["bias"], weights=group["n_observations"])),
            "mean_nlpd": float(group["nlpd"].sum() / n_obs),
            "mean_crps": float(group["crps_sum"].sum() / n_obs) if "crps_sum" in group else np.nan,
            "n_events": int(group["eqid"].nunique()) if "eqid" in group else int(len(group)),
            "n_observations": n_obs,
            "nearest_psd_repairs": int(group["nearest_psd_used"].sum()),
            "min_eigenvalue_before_repair": float(group["min_eigenvalue_before_repair"].min()),
            "mean_pred_sd": float(group["pred_sd_sum"].sum() / n_obs),
            "mean_interval95_width": float(group["interval95_width_sum"].sum() / n_obs),
        })
        for label in ["50", "80", "90", "95"]:
            row[f"coverage_{label}"] = float(group[f"coverage_{label}_count"].sum() / n_obs)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rmse"]).reset_index(drop=True)


def predictive_period_holdout_analysis(
    data: ComparisonData,
    cross_period_psd,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    gh08=None,
    lmc_semivariogram=None,
    lmc_mle=None,
    lmc_block_mle=None,
    kronecker=None,
    ggp_model_a=None,
    methods=None,
    holdout_period_sets=None,
    max_events=5,
    max_stations_per_event=60,
    min_stations=8,
    random_state=0,
    nearest_psd=True,
    iox_full_semivariogram=None,
    sbss_semivariogram=None,
    multivariate_matern=None,
):
    """Predict held-out periods at observed sites using the other periods in each event."""
    period_variances = np.var(data.df_pca[data.period_cols].to_numpy(dtype=float), axis=0)
    curve_sets = _prediction_curve_sets(data, cross_period_psd, methods=methods)
    scenarios = _normalize_period_holdout_sets(data.periods, holdout_period_sets)
    event_ids = _prediction_event_ids(data.df_pca, max_events)
    q = len(data.period_cols)
    records = []

    for eqid in event_ids:
        df_event = data.df_pca.loc[
            data.df_pca["eqid"] == eqid,
            ["station_latitude", "station_longitude", *data.period_cols],
        ].dropna(subset=data.period_cols).copy()
        if len(df_event) < min_stations:
            continue
        df_event = _sample_event_frame(df_event, max_stations_per_event, random_state, eqid)
        n = len(df_event)
        y_flat = df_event[data.period_cols].to_numpy(dtype=float).reshape(-1)
        lat = df_event["station_latitude"].to_numpy(dtype=float)
        lon = df_event["station_longitude"].to_numpy(dtype=float)

        for method in curve_sets:
            cov_joint = _covariance_for_prediction_method(
                method,
                lat,
                lon,
                data,
                curve_sets,
                period_variances,
                pairwise_empirical=pairwise_empirical,
                pca=pca,
                semi=semi,
                gh08=gh08,
                lmc_semivariogram=lmc_semivariogram,
                lmc_mle=lmc_mle,
                lmc_block_mle=lmc_block_mle,
                kronecker=kronecker,
                ggp_model_a=ggp_model_a,
                iox_full_semivariogram=iox_full_semivariogram,
                sbss_semivariogram=sbss_semivariogram,
                multivariate_matern=multivariate_matern,
            )
            all_periods = np.arange(q)
            for scenario in scenarios:
                heldout = scenario["heldout_period_indices"]
                known = np.asarray([i for i in all_periods if i not in set(heldout)], dtype=int)
                if len(known) == 0:
                    continue
                train_vars = np.asarray([site * q + p for site in range(n) for p in known], dtype=int)
                test_vars = np.asarray([site * q + p for site in range(n) for p in heldout], dtype=int)
                try:
                    metrics = _conditional_prediction_metrics(
                        y_flat,
                        cov_joint,
                        train_vars,
                        test_vars,
                        nearest_psd=nearest_psd,
                    )
                    records.append({
                        "task": "period_holdout",
                        "method": method,
                        "eqid": eqid,
                        "n_stations": int(n),
                        "scenario": scenario["scenario"],
                        "heldout_period_secs": scenario["heldout_period_secs"],
                        "n_heldout_periods": int(len(heldout)),
                        "n_known_periods": int(len(known)),
                        "fit_success": True,
                        "error": "",
                        **metrics,
                    })
                except Exception as exc:
                    records.append({
                        "task": "period_holdout",
                        "method": method,
                        "eqid": eqid,
                        "n_stations": int(n),
                        "scenario": scenario["scenario"],
                        "heldout_period_secs": scenario["heldout_period_secs"],
                        "n_heldout_periods": int(len(heldout)),
                        "n_known_periods": int(len(known)),
                        "fit_success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "n_observations": 0,
                        "sse": np.nan,
                        "sae": np.nan,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "bias": np.nan,
                        "nlpd": np.nan,
                        "mean_crps": np.nan,
                        "crps_sum": np.nan,
                        "mean_pred_sd": np.nan,
                        "pred_sd_sum": np.nan,
                        "mean_interval95_width": np.nan,
                        "interval95_width_sum": np.nan,
                        "nearest_psd_used": False,
                        "min_eigenvalue_before_repair": np.nan,
                        "coverage_50_count": 0,
                        "coverage_80_count": 0,
                        "coverage_90_count": 0,
                        "coverage_95_count": 0,
                    })

    detail = pd.DataFrame(records)
    good = detail.loc[detail["fit_success"] & (detail["n_observations"] > 0)].copy()
    summary = _summarize_predictive_detail(good, ["method", "scenario", "heldout_period_secs"])
    return {"summary": summary, "detail": detail}


def predictive_event_loglikelihood_analysis(
    data: ComparisonData,
    cross_period_psd,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    gh08=None,
    lmc_semivariogram=None,
    lmc_mle=None,
    lmc_block_mle=None,
    kronecker=None,
    ggp_model_a=None,
    methods=None,
    max_events=5,
    max_stations_per_event=60,
    min_stations=8,
    random_state=0,
    nearest_psd=True,
    event_ids=None,
    iox_full_semivariogram=None,
    sbss_semivariogram=None,
    multivariate_matern=None,
):
    """Compute independent event-wise Gaussian log likelihoods under each method."""
    period_variances = np.var(data.df_pca[data.period_cols].to_numpy(dtype=float), axis=0)
    curve_sets = _prediction_curve_sets(data, cross_period_psd, methods=methods)
    event_ids = _prediction_event_ids(data.df_pca, max_events) if event_ids is None else list(event_ids)
    records = []

    for eqid in event_ids:
        df_event = data.df_pca.loc[
            data.df_pca["eqid"] == eqid,
            ["station_latitude", "station_longitude", *data.period_cols],
        ].dropna(subset=data.period_cols).copy()
        if len(df_event) < min_stations:
            continue
        df_event = _sample_event_frame(df_event, max_stations_per_event, random_state, eqid)
        y_flat = df_event[data.period_cols].to_numpy(dtype=float).reshape(-1)
        lat = df_event["station_latitude"].to_numpy(dtype=float)
        lon = df_event["station_longitude"].to_numpy(dtype=float)

        for method in curve_sets:
            try:
                cov_joint = _covariance_for_prediction_method(
                    method,
                    lat,
                    lon,
                    data,
                    curve_sets,
                    period_variances,
                    pairwise_empirical=pairwise_empirical,
                    pca=pca,
                    semi=semi,
                    gh08=gh08,
                    lmc_semivariogram=lmc_semivariogram,
                    lmc_mle=lmc_mle,
                    lmc_block_mle=lmc_block_mle,
                    kronecker=kronecker,
                    ggp_model_a=ggp_model_a,
                    iox_full_semivariogram=iox_full_semivariogram,
                    sbss_semivariogram=sbss_semivariogram,
                    multivariate_matern=multivariate_matern,
                )
                cov_joint = 0.5 * (cov_joint + cov_joint.T)
                try:
                    min_eval_before = float(np.linalg.eigvalsh(cov_joint).min())
                except Exception:
                    min_eval_before = np.nan
                repaired = bool((not np.isfinite(min_eval_before)) or min_eval_before < -1e-8)
                if nearest_psd and repaired:
                    cov_joint = nearest_psd_matrix(cov_joint, eps=1e-7, preserve_diagonal=True)
                nlpd = _mvn_nlpd(y_flat, cov_joint, jitter=1e-6)
                records.append({
                    "method": method,
                    "eqid": eqid,
                    "n_stations": int(len(df_event)),
                    "n_observations": int(len(y_flat)),
                    "log_likelihood": float(-nlpd),
                    "nll": float(nlpd),
                    "nll_per_observation": float(nlpd / len(y_flat)),
                    "nearest_psd_used": repaired,
                    "min_eigenvalue_before_repair": min_eval_before,
                    "fit_success": True,
                    "error": "",
                })
            except Exception as exc:
                records.append({
                    "method": method,
                    "eqid": eqid,
                    "n_stations": int(len(df_event)),
                    "n_observations": int(len(y_flat)),
                    "log_likelihood": np.nan,
                    "nll": np.nan,
                    "nll_per_observation": np.nan,
                    "nearest_psd_used": False,
                    "min_eigenvalue_before_repair": np.nan,
                    "fit_success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    detail = pd.DataFrame(records)
    good = detail.loc[detail["fit_success"]].copy()
    if good.empty:
        summary = pd.DataFrame()
    else:
        rows = []
        for method, group in good.groupby("method", sort=False):
            n_obs = int(group["n_observations"].sum())
            total_nll = float(group["nll"].sum())
            rows.append({
                "method": method,
                "total_log_likelihood": float(group["log_likelihood"].sum()),
                "total_nll": total_nll,
                "nll_per_observation": float(total_nll / n_obs),
                "n_events": int(group["eqid"].nunique()),
                "n_observations": n_obs,
                "nearest_psd_repairs": int(group["nearest_psd_used"].sum()),
                "min_eigenvalue_before_repair": float(group["min_eigenvalue_before_repair"].min()),
            })
        summary = pd.DataFrame(rows).sort_values("nll_per_observation").reset_index(drop=True)
    return {"summary": summary, "detail": detail}


def residual_simulation_timing_benchmark(
    data,
    cross_period_psd,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    gh08=None,
    lmc_semivariogram=None,
    lmc_block_mle=None,
    kronecker=None,
    ggp_model_a=None,
    methods=None,
    max_events=5,
    max_stations_per_event=120,
    min_stations=8,
    n_simulations=200,
    random_state=123,
    nearest_psd=True,
    jitter=1e-8,
):
    """Benchmark unconditional within-event residual simulation from fitted curves."""
    import time

    rng = np.random.default_rng(int(random_state))
    period_variances = np.var(data.df_pca[data.period_cols].to_numpy(dtype=float), axis=0)
    curve_sets = _prediction_curve_sets(data, cross_period_psd, methods=methods)
    event_ids = _prediction_event_ids(data.df_pca, max_events)
    q = len(data.period_cols)
    records = []

    for eqid in event_ids:
        df_event = data.df_pca.loc[
            data.df_pca["eqid"] == eqid,
            ["station_latitude", "station_longitude", *data.period_cols],
        ].dropna(subset=data.period_cols).copy()
        if len(df_event) < int(min_stations):
            continue
        df_event = _sample_event_frame(df_event, max_stations_per_event, random_state, eqid)
        if len(df_event) < int(min_stations):
            continue

        lat = df_event["station_latitude"].to_numpy(dtype=float)
        lon = df_event["station_longitude"].to_numpy(dtype=float)
        n = len(df_event)
        dim = int(n * q)

        for method in curve_sets:
            row = {
                "method": method,
                "eqid": eqid,
                "n_stations": int(n),
                "n_periods": int(q),
                "matrix_dimension": dim,
                "n_simulations": int(n_simulations),
                "success": False,
                "error": "",
                "nearest_psd_used": False,
                "min_eigenvalue_before_repair": np.nan,
                "covariance_time_seconds": np.nan,
                "psd_check_time_seconds": np.nan,
                "nearest_psd_repair_time_seconds": 0.0,
                "factorization_time_seconds": np.nan,
                "sampling_time_seconds": np.nan,
                "total_time_seconds": np.nan,
                "estimated_covariance_memory_mb": float(dim ** 2 * 8 / 1e6),
                "native_simulation": False,
                "native_component_count": 1,
                "full_dense_equivalent_dimension": dim,
            }
            try:
                if method in {"PCA semivariogram", "PCA MLE"} and pca is not None:
                    pca_fit = pca["semivariogram"] if method == "PCA semivariogram" else pca["mle"]
                    row.update(_simulate_pca_residuals_native_timed(
                        lat,
                        lon,
                        pca_fit,
                        n_simulations,
                        rng,
                        nearest_psd=nearest_psd,
                        jitter=jitter,
                    ))
                    timing_cols = [
                        "covariance_time_seconds",
                        "psd_check_time_seconds",
                        "nearest_psd_repair_time_seconds",
                        "factorization_time_seconds",
                        "sampling_time_seconds",
                    ]
                    row["total_time_seconds"] = float(np.nansum([row[col] for col in timing_cols]))
                    records.append(row)
                    continue

                t0 = time.perf_counter()
                cov = _covariance_for_prediction_method(
                    method,
                    lat,
                    lon,
                    data,
                    curve_sets,
                    period_variances,
                    pairwise_empirical=pairwise_empirical,
                    pca=pca,
                    semi=semi,
                    gh08=gh08,
                    lmc_semivariogram=lmc_semivariogram,
                    lmc_block_mle=lmc_block_mle,
                    kronecker=kronecker,
                    ggp_model_a=ggp_model_a,
                )
                row["covariance_time_seconds"] = time.perf_counter() - t0

                row["psd_check_time_seconds"] = 0.0
                cov = 0.5 * (cov + cov.T)
                factor_time = 0.0
                last_error = None
                lower = None
                jitter_used = 0.0

                for repair_pass in (False, True):
                    if repair_pass:
                        if not nearest_psd:
                            break
                        repair_t0 = time.perf_counter()
                        cov = nearest_psd_matrix(cov, eps=1e-7, preserve_diagonal=True)
                        row["nearest_psd_repair_time_seconds"] = time.perf_counter() - repair_t0
                        row["nearest_psd_used"] = True

                    factor_t0 = time.perf_counter()
                    scale = max(float(np.mean(np.diag(cov))), 1.0)
                    for attempt in range(8):
                        jitter_used = 0.0 if attempt == 0 else float(jitter) * scale * (10 ** (attempt - 1))
                        try:
                            lower = np.linalg.cholesky(cov + jitter_used * np.eye(dim))
                            break
                        except np.linalg.LinAlgError as exc:
                            last_error = exc
                    factor_time += time.perf_counter() - factor_t0
                    if lower is not None:
                        break

                if lower is None:
                    raise np.linalg.LinAlgError(f"Cholesky failed: {last_error}")
                row["factorization_time_seconds"] = factor_time
                row["cholesky_jitter_used"] = float(jitter_used)

                sample_t0 = time.perf_counter()
                z = rng.standard_normal((dim, int(n_simulations)))
                sims = lower @ z
                row["sampling_time_seconds"] = time.perf_counter() - sample_t0
                row["sample_mean_abs"] = float(np.mean(np.abs(np.mean(sims, axis=1))))
                row["success"] = True
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"

            timing_cols = [
                "covariance_time_seconds",
                "psd_check_time_seconds",
                "nearest_psd_repair_time_seconds",
                "factorization_time_seconds",
                "sampling_time_seconds",
            ]
            row["total_time_seconds"] = float(np.nansum([row[col] for col in timing_cols]))
            records.append(row)

    detail = pd.DataFrame(records)
    if detail.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            detail.groupby("method", sort=False)
            .agg(
                n_events=("eqid", "nunique"),
                n_success=("success", "sum"),
                covariance_time_total=("covariance_time_seconds", "sum"),
                psd_check_time_total=("psd_check_time_seconds", "sum"),
                repair_time_total=("nearest_psd_repair_time_seconds", "sum"),
                factorization_time_total=("factorization_time_seconds", "sum"),
                sampling_time_total=("sampling_time_seconds", "sum"),
                total_time_seconds=("total_time_seconds", "sum"),
                nearest_psd_repairs=("nearest_psd_used", "sum"),
                matrix_dimension_mean=("matrix_dimension", "mean"),
                matrix_dimension_max=("matrix_dimension", "max"),
                full_dense_equivalent_dimension_mean=("full_dense_equivalent_dimension", "mean"),
                native_component_count_mean=("native_component_count", "mean"),
                covariance_memory_mb_mean=("estimated_covariance_memory_mb", "mean"),
            )
            .reset_index()
            .sort_values("total_time_seconds")
        )
    return {"summary": summary, "detail": detail}


def _period_indices_for_count(data, n_periods):
    q_total = len(data.period_cols)
    n_periods = max(1, min(int(n_periods), q_total))
    if n_periods == q_total:
        return np.arange(q_total, dtype=int)
    idx = np.rint(np.linspace(0, q_total - 1, n_periods)).astype(int)
    idx = np.unique(np.clip(idx, 0, q_total - 1))
    if len(idx) < n_periods:
        missing = [i for i in range(q_total) if i not in set(idx)]
        idx = np.sort(np.r_[idx, missing[: n_periods - len(idx)]])
    return idx.astype(int)


def _subset_square_by_periods(value, period_idx):
    arr = np.asarray(value)
    if arr.ndim >= 2 and arr.shape[0] > np.max(period_idx) and arr.shape[1] > np.max(period_idx):
        return arr[np.ix_(period_idx, period_idx)]
    return value


def _subset_vector_by_periods(value, period_idx):
    arr = np.asarray(value)
    if arr.ndim == 1 and arr.shape[0] > np.max(period_idx):
        return arr[period_idx]
    return value


def _subset_comparison_data_periods(data, period_idx):
    period_idx = np.asarray(period_idx, dtype=int)
    period_cols = [data.period_cols[i] for i in period_idx]
    kwargs = {field.name: getattr(data, field.name) for field in fields(ComparisonData)}
    kwargs["periods"] = np.asarray(data.periods, dtype=float)[period_idx]
    kwargs["period_cols"] = period_cols
    kwargs["df_pca"] = data.df_pca[
        ["eqid", "station_latitude", "station_longitude", *period_cols]
    ].dropna(subset=period_cols).copy()
    kwargs["events_pca"] = kwargs["df_pca"]["eqid"].unique()
    if len(data.period_dfs) > int(np.max(period_idx)):
        kwargs["period_dfs"] = [data.period_dfs[i] for i in period_idx]
    if len(data.gamma_emp) > int(np.max(period_idx)):
        kwargs["gamma_emp"] = [[data.gamma_emp[i][j] for j in period_idx] for i in period_idx]
    if len(data.count_emp) > int(np.max(period_idx)):
        kwargs["count_emp"] = [[data.count_emp[i][j] for j in period_idx] for i in period_idx]
    kwargs["pair_record_counts"] = _subset_square_by_periods(data.pair_record_counts, period_idx)
    kwargs["rho_0"] = _subset_square_by_periods(data.rho_0, period_idx)
    kwargs["emp_sills"] = _subset_square_by_periods(data.emp_sills, period_idx)
    return ComparisonData(**kwargs)


def _subset_cross_period_psd(cross_period_psd, period_idx):
    if cross_period_psd is None:
        return None
    period_idx = np.asarray(period_idx, dtype=int)
    out = dict(cross_period_psd)
    curves_out = {}
    for name, (h_grid, curves) in cross_period_psd.get("curves", {}).items():
        curves_arr = np.asarray(curves)
        if (
            curves_arr.ndim == 3
            and curves_arr.shape[0] > np.max(period_idx)
            and curves_arr.shape[1] > np.max(period_idx)
        ):
            curves_arr = curves_arr[period_idx][:, period_idx, :]
        curves_out[name] = (h_grid, curves_arr)
    out["curves"] = curves_out
    return out


def _subset_pairwise_empirical(pairwise_empirical, period_idx):
    if pairwise_empirical is None:
        return None
    out = dict(pairwise_empirical)
    if isinstance(out.get("fit"), dict):
        fit = dict(out["fit"])
        for key in ("LE", "gammaE"):
            if key in fit:
                fit[key] = _subset_square_by_periods(fit[key], period_idx)
        out["fit"] = fit
    return out


def _subset_direct_fit_by_periods(fit, period_idx):
    if fit is None:
        return None
    out = dict(fit)
    for key in ("LE", "gammaE", "nll"):
        if key in out:
            out[key] = _subset_vector_by_periods(out[key], period_idx)
    return out


def _subset_gh08_fit_by_periods(gh08, period_idx):
    if gh08 is None:
        return None
    out = dict(gh08)
    for key in ("direct_semi", "direct_mle"):
        if isinstance(out.get(key), dict):
            out[key] = _subset_direct_fit_by_periods(out[key], period_idx)
    return out


def _subset_pca_fit_by_periods(pca_fit, period_idx):
    if pca_fit is None:
        return None
    out = dict(pca_fit)
    period_idx = np.asarray(period_idx, dtype=int)
    n_periods = len(period_idx)
    n_components = n_periods
    if "U" in out:
        loadings = np.asarray(out["U"], dtype=float)[period_idx, :]
        n_components = min(n_periods, loadings.shape[1])
        out["U"] = loadings[:, :n_components]
    for key in ("var_pool", "std_pool", "periods"):
        if key in out:
            out[key] = _subset_vector_by_periods(out[key], period_idx)
    for key in ("evals", "LE", "gammaE", "explained", "pc_sills", "mle_nll"):
        if key in out:
            arr = np.asarray(out[key])
            if arr.ndim == 1 and arr.shape[0] >= n_components:
                out[key] = arr[:n_components]
    for key in ("gamma_pc", "count_pc"):
        if key in out and isinstance(out[key], (list, tuple)):
            out[key] = list(out[key])[:n_components]
    if "rho_pc" in out:
        arr = np.asarray(out["rho_pc"])
        if arr.ndim == 2 and arr.shape[1] >= n_components:
            out["rho_pc"] = arr[:, :n_components]
    if "K" in out:
        out["K"] = int(n_components)
    return out


def _subset_pca_by_periods(pca, period_idx):
    if pca is None:
        return None
    out = dict(pca)
    for key in ("semivariogram", "mle"):
        if isinstance(out.get(key), dict):
            out[key] = _subset_pca_fit_by_periods(out[key], period_idx)
    return out


def _subset_lmc_fit_by_periods(lmc_fit, period_idx):
    if lmc_fit is None:
        return None
    out = dict(lmc_fit)
    if "rho" in out:
        out["rho"] = _subset_square_by_periods(out["rho"], period_idx)
    if "B_raw" in out:
        out["B_raw"] = [_subset_square_by_periods(b_mat, period_idx) for b_mat in out["B_raw"]]
    if "periods" in out:
        out["periods"] = _subset_vector_by_periods(out["periods"], period_idx)
    return out


def _residual_sampling_method_specs(methods=None, pca_variant="semivariogram"):
    pca_internal = "PCA MLE" if str(pca_variant).lower() == "mle" else "PCA semivariogram"
    aliases = {
        "Pairwise empirical": ("Pairwise empirical", "Pairwise empirical semivariogram"),
        "Pairwise empirical semivariogram": ("Pairwise empirical", "Pairwise empirical semivariogram"),
        "GH08": ("GH08", "GH08 semivariogram"),
        "GH08 semivariogram": ("GH08", "GH08 semivariogram"),
        "GH08 MLE": ("GH08", "GH08 MLE"),
        "Kronecker": ("Kronecker", "Kronecker semivariogram"),
        "Kronecker semivariogram": ("Kronecker", "Kronecker semivariogram"),
        "Kronecker MLE": ("Kronecker", "Kronecker MLE"),
        "PCA": ("PCA", pca_internal),
        "PCA semivariogram": ("PCA", "PCA semivariogram"),
        "PCA MLE": ("PCA", "PCA MLE"),
        "LMC": ("LMC", "LMC semivariogram"),
        "LMC semivariogram": ("LMC", "LMC semivariogram"),
        "LMC MLE": ("LMC", "LMC MLE"),
        "LMC block MLE": ("LMC", "LMC block MLE"),
    }
    requested = methods or ("Pairwise empirical", "GH08", "Kronecker", "PCA", "LMC")
    specs = []
    seen = set()
    for method in requested:
        display, internal = aliases.get(method, (method, method))
        if display in seen:
            continue
        specs.append((display, internal))
        seen.add(display)
    return specs


def residual_sampling_scaling_benchmark(
    data,
    cross_period_psd,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    gh08=None,
    lmc_semivariogram=None,
    lmc_block_mle=None,
    kronecker=None,
    ggp_model_a=None,
    methods=None,
    record_counts=(40, 80, 120, 160),
    period_counts=(3, 6, 9),
    max_events=5,
    min_records=None,
    n_simulations=2,
    random_state=123,
    nearest_psd=True,
    jitter=1e-8,
    pca_variant="semivariogram",
):
    """Benchmark residual simulation scaling over selected record and period counts.

    Method labels are representation-level labels by default. For example, the
    PCA row uses one configured PCA fit variant but is displayed simply as PCA.
    """
    import time

    rng = np.random.default_rng(int(random_state))
    record_counts = sorted({int(n) for n in record_counts if int(n) > 0})
    period_counts = sorted({int(q) for q in period_counts if int(q) > 0})
    specs = _residual_sampling_method_specs(methods=methods, pca_variant=pca_variant)
    event_ids = _prediction_event_ids(data.df_pca, max_events)
    records = []

    for period_count in period_counts:
        period_idx = _period_indices_for_count(data, period_count)
        sub_data = _subset_comparison_data_periods(data, period_idx)
        sub_cross_period_psd = _subset_cross_period_psd(cross_period_psd, period_idx)
        sub_pairwise_empirical = _subset_pairwise_empirical(pairwise_empirical, period_idx)
        sub_pca = _subset_pca_by_periods(pca, period_idx)
        sub_gh08 = _subset_gh08_fit_by_periods(gh08, period_idx)
        sub_lmc_semivariogram = _subset_lmc_fit_by_periods(lmc_semivariogram, period_idx)
        sub_lmc_block_mle = _subset_lmc_fit_by_periods(lmc_block_mle, period_idx)
        q = len(sub_data.period_cols)
        period_variances = np.var(sub_data.df_pca[sub_data.period_cols].to_numpy(dtype=float), axis=0)
        curve_sets = _prediction_curve_sets(
            sub_data,
            sub_cross_period_psd,
            methods=[internal for _, internal in specs],
        )
        period_label = ", ".join(f"{p:g}" for p in np.asarray(sub_data.periods, dtype=float))

        for eqid in event_ids:
            df_event = sub_data.df_pca.loc[
                sub_data.df_pca["eqid"] == eqid,
                ["station_latitude", "station_longitude", *sub_data.period_cols],
            ].dropna(subset=sub_data.period_cols).copy()
            if min_records is not None and len(df_event) < int(min_records):
                continue
            max_requested_records = max((n for n in record_counts if len(df_event) >= n), default=0)
            if max_requested_records <= 0:
                continue
            nested_event_sample = _sample_event_frame(
                df_event,
                max_requested_records,
                random_state,
                eqid,
            )

            for n_records in record_counts:
                if len(df_event) < n_records:
                    continue
                df_sample = nested_event_sample.iloc[:n_records].copy()
                lat = df_sample["station_latitude"].to_numpy(dtype=float)
                lon = df_sample["station_longitude"].to_numpy(dtype=float)
                n = len(df_sample)
                dim = int(n * q)

                for display_method, internal_method in specs:
                    row = {
                        "method": display_method,
                        "internal_method": internal_method,
                        "eqid": eqid,
                        "n_records": int(n),
                        "n_stations": int(n),
                        "n_periods": int(q),
                        "periods": period_label,
                        "matrix_dimension": dim,
                        "n_simulations": int(n_simulations),
                        "success": False,
                        "error": "",
                        "nearest_psd_used": False,
                        "covariance_time_seconds": np.nan,
                        "psd_check_time_seconds": 0.0,
                        "nearest_psd_repair_time_seconds": 0.0,
                        "factorization_time_seconds": np.nan,
                        "sampling_time_seconds": np.nan,
                        "total_time_seconds": np.nan,
                        "estimated_covariance_memory_mb": float(dim ** 2 * 8 / 1e6),
                        "native_simulation": False,
                        "native_component_count": 1,
                        "full_dense_equivalent_dimension": dim,
                    }
                    try:
                        if internal_method in {"PCA semivariogram", "PCA MLE"} and sub_pca is not None:
                            pca_key = "semivariogram" if internal_method == "PCA semivariogram" else "mle"
                            row.update(_simulate_pca_residuals_native_timed(
                                lat,
                                lon,
                                sub_pca[pca_key],
                                n_simulations,
                                rng,
                                nearest_psd=nearest_psd,
                                jitter=jitter,
                            ))
                            timing_cols = [
                                "covariance_time_seconds",
                                "psd_check_time_seconds",
                                "nearest_psd_repair_time_seconds",
                                "factorization_time_seconds",
                                "sampling_time_seconds",
                            ]
                            row["total_time_seconds"] = float(np.nansum([row[col] for col in timing_cols]))
                            records.append(row)
                            continue

                        t0 = time.perf_counter()
                        cov = _covariance_for_prediction_method(
                            internal_method,
                            lat,
                            lon,
                            sub_data,
                            curve_sets,
                            period_variances,
                            pairwise_empirical=sub_pairwise_empirical,
                            pca=sub_pca,
                            semi=semi,
                            gh08=sub_gh08,
                            lmc_semivariogram=sub_lmc_semivariogram,
                            lmc_block_mle=sub_lmc_block_mle,
                            kronecker=kronecker,
                            ggp_model_a=ggp_model_a,
                        )
                        row["covariance_time_seconds"] = time.perf_counter() - t0

                        cov = 0.5 * (cov + cov.T)
                        lower = None
                        last_error = None
                        jitter_used = 0.0
                        factor_time = 0.0
                        for repair_pass in (False, True):
                            if repair_pass:
                                if not nearest_psd:
                                    break
                                repair_t0 = time.perf_counter()
                                cov = nearest_psd_matrix(cov, eps=1e-7, preserve_diagonal=True)
                                row["nearest_psd_repair_time_seconds"] = time.perf_counter() - repair_t0
                                row["nearest_psd_used"] = True

                            factor_t0 = time.perf_counter()
                            scale = max(float(np.mean(np.diag(cov))), 1.0)
                            for attempt in range(8):
                                jitter_used = 0.0 if attempt == 0 else float(jitter) * scale * (10 ** (attempt - 1))
                                try:
                                    lower = np.linalg.cholesky(cov + jitter_used * np.eye(dim))
                                    break
                                except np.linalg.LinAlgError as exc:
                                    last_error = exc
                            factor_time += time.perf_counter() - factor_t0
                            if lower is not None:
                                break

                        if lower is None:
                            raise np.linalg.LinAlgError(f"Cholesky failed: {last_error}")
                        row["factorization_time_seconds"] = factor_time
                        row["cholesky_jitter_used"] = float(jitter_used)

                        sample_t0 = time.perf_counter()
                        z = rng.standard_normal((dim, int(n_simulations)))
                        sims = lower @ z
                        row["sampling_time_seconds"] = time.perf_counter() - sample_t0
                        row["sample_mean_abs"] = float(np.mean(np.abs(np.mean(sims, axis=1))))
                        row["success"] = True
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"

                    timing_cols = [
                        "covariance_time_seconds",
                        "psd_check_time_seconds",
                        "nearest_psd_repair_time_seconds",
                        "factorization_time_seconds",
                        "sampling_time_seconds",
                    ]
                    row["total_time_seconds"] = float(np.nansum([row[col] for col in timing_cols]))
                    records.append(row)

    detail = pd.DataFrame(records)
    if detail.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            detail.groupby(["method", "n_records", "n_periods", "periods"], sort=False)
            .agg(
                n_events=("eqid", "nunique"),
                n_success=("success", "sum"),
                covariance_time_total=("covariance_time_seconds", "sum"),
                psd_check_time_total=("psd_check_time_seconds", "sum"),
                repair_time_total=("nearest_psd_repair_time_seconds", "sum"),
                factorization_time_total=("factorization_time_seconds", "sum"),
                sampling_time_total=("sampling_time_seconds", "sum"),
                total_time_seconds=("total_time_seconds", "sum"),
                mean_time_seconds=("total_time_seconds", "mean"),
                nearest_psd_repairs=("nearest_psd_used", "sum"),
                matrix_dimension_mean=("matrix_dimension", "mean"),
                matrix_dimension_max=("matrix_dimension", "max"),
                full_dense_equivalent_dimension_mean=("full_dense_equivalent_dimension", "mean"),
                native_component_count_mean=("native_component_count", "mean"),
                covariance_memory_mb_mean=("estimated_covariance_memory_mb", "mean"),
            )
            .reset_index()
            .sort_values(["n_periods", "n_records", "total_time_seconds"])
        )
    return {"summary": summary, "detail": detail}


def synthetic_station_lat_lon(
    n_stations,
    center_lat=34.05,
    center_lon=-118.25,
    region_km=120.0,
    random_state=123,
):
    """Generate reproducible synthetic station coordinates in a local square region."""
    rng = np.random.default_rng(int(random_state))
    east_km = rng.uniform(-0.5 * float(region_km), 0.5 * float(region_km), int(n_stations))
    north_km = rng.uniform(-0.5 * float(region_km), 0.5 * float(region_km), int(n_stations))
    lat = float(center_lat) + north_km / 111.0
    lon = float(center_lon) + east_km / (111.0 * np.cos(np.radians(float(center_lat))))
    return lat.astype(float), lon.astype(float)


def residual_sampling_synthetic_station_scaling_benchmark(
    data,
    pca=None,
    semi=None,
    lmc_semivariogram=None,
    kronecker=None,
    methods=("PCA", "Kronecker", "LMC"),
    station_counts=(20, 40, 80, 160, 320, 640, 1280, 2560, 5120),
    period_counts=(3, 6, 9),
    n_simulations=1,
    random_state=123,
    center_lat=34.05,
    center_lon=-118.25,
    region_km=120.0,
    nearest_psd=True,
    jitter=1e-8,
    pca_variant="semivariogram",
):
    """Benchmark native residual simulation using synthetic nested station coordinates."""
    rng = np.random.default_rng(int(random_state))
    station_counts = sorted({int(n) for n in station_counts if int(n) > 0})
    period_counts = sorted({int(q) for q in period_counts if int(q) > 0})
    specs = _residual_sampling_method_specs(methods=methods, pca_variant=pca_variant)
    max_stations = max(station_counts) if station_counts else 0
    lat_all, lon_all = synthetic_station_lat_lon(
        max_stations,
        center_lat=center_lat,
        center_lon=center_lon,
        region_km=region_km,
        random_state=random_state,
    )
    records = []

    for period_count in period_counts:
        period_idx = _period_indices_for_count(data, period_count)
        sub_data = _subset_comparison_data_periods(data, period_idx)
        sub_pca = _subset_pca_by_periods(pca, period_idx)
        sub_lmc_semivariogram = _subset_lmc_fit_by_periods(lmc_semivariogram, period_idx)
        q = len(sub_data.period_cols)
        period_variances = np.var(sub_data.df_pca[sub_data.period_cols].to_numpy(dtype=float), axis=0)
        period_label = ", ".join(f"{p:g}" for p in np.asarray(sub_data.periods, dtype=float))

        for n_stations in station_counts:
            lat = lat_all[:n_stations]
            lon = lon_all[:n_stations]
            dim = int(n_stations * q)

            for display_method, internal_method in specs:
                row = {
                    "method": display_method,
                    "internal_method": internal_method,
                    "eqid": "synthetic",
                    "n_records": int(n_stations),
                    "n_stations": int(n_stations),
                    "n_periods": int(q),
                    "periods": period_label,
                    "matrix_dimension": int(n_stations),
                    "n_simulations": int(n_simulations),
                    "success": False,
                    "error": "",
                    "nearest_psd_used": False,
                    "covariance_time_seconds": np.nan,
                    "psd_check_time_seconds": 0.0,
                    "nearest_psd_repair_time_seconds": 0.0,
                    "factorization_time_seconds": np.nan,
                    "sampling_time_seconds": np.nan,
                    "total_time_seconds": np.nan,
                    "estimated_covariance_memory_mb": np.nan,
                    "native_simulation": True,
                    "native_component_count": np.nan,
                    "full_dense_equivalent_dimension": dim,
                    "synthetic_region_km": float(region_km),
                }
                try:
                    if internal_method in {"PCA semivariogram", "PCA MLE"}:
                        if sub_pca is None:
                            raise ValueError("PCA fit is not available.")
                        pca_key = "semivariogram" if internal_method == "PCA semivariogram" else "mle"
                        row.update(_simulate_pca_residuals_native_timed(
                            lat,
                            lon,
                            sub_pca[pca_key],
                            n_simulations,
                            rng,
                            nearest_psd=nearest_psd,
                            jitter=jitter,
                        ))
                    elif internal_method in {"Kronecker semivariogram", "Kronecker MLE"}:
                        if kronecker is None:
                            raise ValueError("Kronecker fit is not available.")
                        kron_key = "mle" if internal_method == "Kronecker MLE" else "semivariogram"
                        row.update(_simulate_kronecker_residuals_native_timed(
                            lat,
                            lon,
                            kronecker[kron_key],
                            sub_data.rho_0,
                            period_variances,
                            n_simulations,
                            rng,
                            nearest_psd=nearest_psd,
                            jitter=jitter,
                        ))
                    elif internal_method in {"LMC semivariogram", "LMC MLE", "LMC block MLE"}:
                        if sub_lmc_semivariogram is None or semi is None:
                            raise ValueError("LMC fit is not available.")
                        row.update(_simulate_lmc_residuals_native_timed(
                            lat,
                            lon,
                            sub_lmc_semivariogram,
                            semi["lmc"],
                            n_simulations,
                            rng,
                            nearest_psd=nearest_psd,
                            jitter=jitter,
                        ))
                    else:
                        raise ValueError(f"Synthetic native timing does not support {internal_method!r}.")
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    timing_cols = [
                        "covariance_time_seconds",
                        "psd_check_time_seconds",
                        "nearest_psd_repair_time_seconds",
                        "factorization_time_seconds",
                        "sampling_time_seconds",
                    ]
                    row["total_time_seconds"] = float(np.nansum([row[col] for col in timing_cols]))
                records.append(row)

    detail = pd.DataFrame(records)
    if detail.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            detail.groupby(["method", "n_records", "n_periods", "periods"], sort=False)
            .agg(
                n_events=("eqid", "nunique"),
                n_success=("success", "sum"),
                covariance_time_total=("covariance_time_seconds", "sum"),
                psd_check_time_total=("psd_check_time_seconds", "sum"),
                repair_time_total=("nearest_psd_repair_time_seconds", "sum"),
                factorization_time_total=("factorization_time_seconds", "sum"),
                sampling_time_total=("sampling_time_seconds", "sum"),
                total_time_seconds=("total_time_seconds", "sum"),
                mean_time_seconds=("total_time_seconds", "mean"),
                nearest_psd_repairs=("nearest_psd_used", "sum"),
                matrix_dimension_mean=("matrix_dimension", "mean"),
                matrix_dimension_max=("matrix_dimension", "max"),
                full_dense_equivalent_dimension_mean=("full_dense_equivalent_dimension", "mean"),
                native_component_count_mean=("native_component_count", "mean"),
                covariance_memory_mb_mean=("estimated_covariance_memory_mb", "mean"),
            )
            .reset_index()
            .sort_values(["n_periods", "n_records", "total_time_seconds"])
        )
    return {
        "summary": summary,
        "detail": detail,
        "synthetic_coordinates": pd.DataFrame({
            "station_latitude": lat_all,
            "station_longitude": lon_all,
        }),
    }


# -----------------------------------------------------------------------------
# Full-data predictive evaluation with partially observed station-period vectors
# -----------------------------------------------------------------------------

def predictive_full_data_wide(data: ComparisonData):
    """Outer-join all period-specific records without imputing missing periods."""
    key_cols = ["recid", "eqid", "station_latitude", "station_longitude"]
    frames = []
    for period_col, frame in zip(data.period_cols, data.period_dfs):
        required = [*key_cols, "scaled_deltaW"]
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise KeyError(f"Period dataframe is missing columns: {missing}")
        part = frame[required].copy()
        if part.duplicated(key_cols).any():
            raise ValueError(f"Duplicate station-period rows found for {period_col}")
        part = part.rename(columns={"scaled_deltaW": period_col})
        frames.append(part)

    if not frames:
        raise ValueError("No period dataframes are available")
    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.merge(frame, on=key_cols, how="outer", validate="one_to_one")
    wide = wide.sort_values(["eqid", "recid"]).reset_index(drop=True)
    wide["n_observed_periods"] = wide[data.period_cols].notna().sum(axis=1).astype(int)
    return wide


def _full_data_period_variances(wide, period_cols):
    values = wide[period_cols].to_numpy(dtype=float)
    variances = np.nanvar(values, axis=0)
    return np.clip(variances, 1e-12, None)


def _full_data_event_frame(wide, eqid, period_cols, max_stations_per_event, random_state):
    cols = ["recid", "station_latitude", "station_longitude", *period_cols]
    event = wide.loc[wide["eqid"] == eqid, cols].copy()
    event = event.loc[event[period_cols].notna().any(axis=1)].copy()
    return _sample_event_frame(event, max_stations_per_event, random_state, eqid).sort_values("recid")


def _full_data_covariance(
    method,
    event,
    data,
    curve_sets,
    period_variances,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    gh08=None,
    lmc_semivariogram=None,
    lmc_mle=None,
    lmc_block_mle=None,
    kronecker=None,
    ggp_model_a=None,
    iox_full_semivariogram=None,
    sbss_semivariogram=None,
    multivariate_matern=None,
):
    return _covariance_for_prediction_method(
        method,
        event["station_latitude"].to_numpy(dtype=float),
        event["station_longitude"].to_numpy(dtype=float),
        data,
        curve_sets,
        period_variances,
        pairwise_empirical=pairwise_empirical,
        pca=pca,
        semi=semi,
        gh08=gh08,
        lmc_semivariogram=lmc_semivariogram,
        lmc_mle=lmc_mle,
        lmc_block_mle=lmc_block_mle,
        kronecker=kronecker,
        ggp_model_a=ggp_model_a,
        iox_full_semivariogram=iox_full_semivariogram,
        sbss_semivariogram=sbss_semivariogram,
        multivariate_matern=multivariate_matern,
    )


def _full_data_metric_payload(error, pred_cov, repaired, min_eval_before, jitter=1e-6):
    error = np.asarray(error, dtype=float)
    pred_cov = 0.5 * (np.asarray(pred_cov, dtype=float) + np.asarray(pred_cov, dtype=float).T)
    pred_sd = np.sqrt(np.clip(np.diag(pred_cov), 0.0, None))
    crps = _normal_crps(error, pred_sd)
    cover = {}
    for label, z in [("50", 0.67448975), ("80", 1.28155157), ("90", 1.64485363), ("95", 1.95996398)]:
        cover[f"coverage_{label}_count"] = int(np.sum(np.abs(error) <= z * pred_sd))
    n_obs = len(error)
    return {
        "n_observations": int(n_obs),
        "sse": float(np.sum(error ** 2)),
        "sae": float(np.sum(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "nlpd": _mvn_nlpd(error, pred_cov, jitter=jitter),
        "mean_crps": float(np.mean(crps)),
        "crps_sum": float(np.sum(crps)),
        "mean_pred_sd": float(np.mean(pred_sd)),
        "pred_sd_sum": float(np.sum(pred_sd)),
        "mean_interval95_width": float(np.mean(2.0 * 1.95996398 * pred_sd)),
        "interval95_width_sum": float(np.sum(2.0 * 1.95996398 * pred_sd)),
        "nearest_psd_used": bool(repaired),
        "min_eigenvalue_before_repair": float(min_eval_before),
        **cover,
    }


def _full_data_conditional_result(y_observed, cov_observed, train_pos, test_pos, nearest_psd=True, jitter=1e-6):
    """Condition within the observed-data marginal; original NaNs never enter."""
    y_observed = np.asarray(y_observed, dtype=float)
    cov_observed = 0.5 * (np.asarray(cov_observed, dtype=float) + np.asarray(cov_observed, dtype=float).T)
    train_pos = np.asarray(train_pos, dtype=int)
    test_pos = np.asarray(test_pos, dtype=int)
    try:
        min_eval_before = float(np.linalg.eigvalsh(cov_observed).min())
    except Exception:
        min_eval_before = np.nan
    repaired = bool((not np.isfinite(min_eval_before)) or min_eval_before < -1e-8)
    if nearest_psd and repaired:
        cov_observed = nearest_psd_matrix(cov_observed, eps=1e-7, preserve_diagonal=True)

    k_oo = cov_observed[np.ix_(train_pos, train_pos)]
    k_to = cov_observed[np.ix_(test_pos, train_pos)]
    k_tt = cov_observed[np.ix_(test_pos, test_pos)]
    if nearest_psd:
        k_oo = nearest_psd_matrix(k_oo, eps=1e-7, preserve_diagonal=True)
    k_oo_j = k_oo + jitter * np.eye(k_oo.shape[0])
    y_train = y_observed[train_pos]
    y_test = y_observed[test_pos]
    try:
        alpha = np.linalg.solve(k_oo_j, y_train)
        solved = np.linalg.solve(k_oo_j, k_to.T)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(k_oo_j)
        alpha = inverse @ y_train
        solved = inverse @ k_to.T
        repaired = True
    prediction = k_to @ alpha
    pred_cov = k_tt - k_to @ solved
    pred_cov = 0.5 * (pred_cov + pred_cov.T)
    if nearest_psd:
        pred_cov = nearest_psd_matrix(pred_cov, eps=1e-7, preserve_diagonal=True)
    error = y_test - prediction
    metrics = _full_data_metric_payload(error, pred_cov, repaired, min_eval_before, jitter=jitter)
    return {
        "metrics": metrics,
        "error": error,
        "prediction": prediction,
        "pred_cov": pred_cov,
    }


def _full_data_failure_row(base, exc):
    return {
        **base,
        "fit_success": False,
        "error": f"{type(exc).__name__}: {exc}",
        "n_observations": 0,
    }


def _full_data_period_rows(base, result, test_original_indices, periods):
    rows = []
    period_index = np.asarray(test_original_indices, dtype=int) % len(periods)
    for p in np.unique(period_index):
        use = np.flatnonzero(period_index == p)
        sub_cov = result["pred_cov"][np.ix_(use, use)]
        metrics = _full_data_metric_payload(
            result["error"][use],
            sub_cov,
            result["metrics"]["nearest_psd_used"],
            result["metrics"]["min_eigenvalue_before_repair"],
        )
        rows.append({
            **base,
            "period_index": int(p),
            "period_s": float(periods[p]),
            "fit_success": True,
            "error": "",
            **metrics,
        })
    return rows


def _full_data_summary(detail, group_cols):
    if detail.empty:
        return pd.DataFrame()
    good = detail.loc[detail["fit_success"] & (detail["n_observations"] > 0)].copy()
    return _summarize_predictive_detail(good, group_cols) if not good.empty else pd.DataFrame()


def predictive_event_loglikelihood_full_data(
    data: ComparisonData,
    cross_period_psd,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    gh08=None,
    lmc_semivariogram=None,
    lmc_mle=None,
    lmc_block_mle=None,
    kronecker=None,
    ggp_model_a=None,
    methods=None,
    max_events=None,
    max_stations_per_event=80,
    min_stations=8,
    random_state=0,
    nearest_psd=True,
    wide=None,
    iox_full_semivariogram=None,
    sbss_semivariogram=None,
    multivariate_matern=None,
):
    """Observed-data event likelihood using all available period records."""
    wide = predictive_full_data_wide(data) if wide is None else wide.copy()
    period_variances = _full_data_period_variances(wide, data.period_cols)
    curve_sets = _prediction_curve_sets(data, cross_period_psd, methods=methods)
    event_ids = _prediction_event_ids(wide, max_events)
    records = []
    q = len(data.period_cols)

    for eqid in event_ids:
        event = _full_data_event_frame(wide, eqid, data.period_cols, max_stations_per_event, random_state)
        if len(event) < min_stations:
            continue
        y_matrix = event[data.period_cols].to_numpy(dtype=float)
        y_flat = y_matrix.reshape(-1)
        observed = np.flatnonzero(np.isfinite(y_flat))
        if len(observed) < 2:
            continue
        for method in curve_sets:
            base = {
                "task": "full_partial_event_likelihood",
                "method": method,
                "eqid": eqid,
                "n_stations": int(len(event)),
                "n_observations": int(len(observed)),
            }
            try:
                cov_full = _full_data_covariance(
                    method, event, data, curve_sets, period_variances,
                    pairwise_empirical=pairwise_empirical,
                    pca=pca,
                    semi=semi,
                    gh08=gh08,
                    lmc_semivariogram=lmc_semivariogram,
                    lmc_mle=lmc_mle,
                    lmc_block_mle=lmc_block_mle,
                    kronecker=kronecker,
                    ggp_model_a=ggp_model_a,
                    iox_full_semivariogram=iox_full_semivariogram,
                    sbss_semivariogram=sbss_semivariogram,
                    multivariate_matern=multivariate_matern,
                )
                cov_obs = cov_full[np.ix_(observed, observed)]
                cov_obs = 0.5 * (cov_obs + cov_obs.T)
                try:
                    min_eval = float(np.linalg.eigvalsh(cov_obs).min())
                except Exception:
                    min_eval = np.nan
                repaired = bool((not np.isfinite(min_eval)) or min_eval < -1e-8)
                if nearest_psd and repaired:
                    cov_obs = nearest_psd_matrix(cov_obs, eps=1e-7, preserve_diagonal=True)
                nll = _mvn_nlpd(y_flat[observed], cov_obs)
                records.append({
                    **base,
                    "log_likelihood": float(-nll),
                    "nll": float(nll),
                    "nll_per_observation": float(nll / len(observed)),
                    "nearest_psd_used": repaired,
                    "min_eigenvalue_before_repair": min_eval,
                    "fit_success": True,
                    "error": "",
                })
            except Exception as exc:
                records.append({
                    **base,
                    "log_likelihood": np.nan,
                    "nll": np.nan,
                    "nll_per_observation": np.nan,
                    "nearest_psd_used": False,
                    "min_eigenvalue_before_repair": np.nan,
                    "fit_success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    detail = pd.DataFrame(records)
    good = detail.loc[detail["fit_success"]].copy() if not detail.empty else detail
    rows = []
    for method, group in good.groupby("method", sort=False):
        n_obs = int(group["n_observations"].sum())
        total_nll = float(group["nll"].sum())
        rows.append({
            "method": method,
            "total_log_likelihood": float(group["log_likelihood"].sum()),
            "total_nll": total_nll,
            "nll_per_observation": float(total_nll / n_obs),
            "n_events": int(group["eqid"].nunique()),
            "n_observations": n_obs,
            "nearest_psd_repairs": int(group["nearest_psd_used"].sum()),
            "min_eigenvalue_before_repair": float(group["min_eigenvalue_before_repair"].min()),
        })
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("nll_per_observation").reset_index(drop=True)
    return {"summary": summary, "detail": detail, "wide": wide, "period_variances": period_variances}


def predictive_random_station_period_holdout_full_data(
    data: ComparisonData,
    cross_period_psd,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    gh08=None,
    lmc_semivariogram=None,
    lmc_mle=None,
    lmc_block_mle=None,
    kronecker=None,
    ggp_model_a=None,
    methods=None,
    mask_fraction=0.10,
    n_repeats=5,
    max_events=None,
    max_stations_per_event=80,
    min_stations=8,
    random_state=0,
    nearest_psd=True,
    wide=None,
    iox_full_semivariogram=None,
    sbss_semivariogram=None,
    multivariate_matern=None,
):
    """Mask only genuinely observed station-period entries in the full partial data."""
    wide = predictive_full_data_wide(data) if wide is None else wide.copy()
    period_variances = _full_data_period_variances(wide, data.period_cols)
    curve_sets = _prediction_curve_sets(data, cross_period_psd, methods=methods)
    event_ids = _prediction_event_ids(wide, max_events)
    rng = np.random.default_rng(random_state)
    records, period_records = [], []

    for repeat in range(int(n_repeats)):
        for eqid in event_ids:
            event = _full_data_event_frame(wide, eqid, data.period_cols, max_stations_per_event, random_state + repeat)
            if len(event) < min_stations:
                continue
            y_flat = event[data.period_cols].to_numpy(dtype=float).reshape(-1)
            observed = np.flatnonzero(np.isfinite(y_flat))
            if len(observed) < 2:
                continue
            n_test = min(max(1, int(round(mask_fraction * len(observed)))), len(observed) - 1)
            test_original = np.sort(rng.choice(observed, size=n_test, replace=False))
            test_pos = np.flatnonzero(np.isin(observed, test_original))
            train_pos = np.flatnonzero(~np.isin(observed, test_original))
            y_observed = y_flat[observed]

            for method in curve_sets:
                base = {
                    "task": "full_partial_random_station_period_holdout",
                    "method": method,
                    "eqid": eqid,
                    "repeat": repeat,
                    "n_stations": int(len(event)),
                    "mask_fraction": float(mask_fraction),
                }
                try:
                    cov_full = _full_data_covariance(
                        method, event, data, curve_sets, period_variances,
                        pairwise_empirical=pairwise_empirical,
                        pca=pca,
                        semi=semi,
                        gh08=gh08,
                        lmc_semivariogram=lmc_semivariogram,
                        lmc_mle=lmc_mle,
                        lmc_block_mle=lmc_block_mle,
                        kronecker=kronecker,
                        ggp_model_a=ggp_model_a,
                        iox_full_semivariogram=iox_full_semivariogram,
                        sbss_semivariogram=sbss_semivariogram,
                        multivariate_matern=multivariate_matern,
                    )
                    cov_observed = cov_full[np.ix_(observed, observed)]
                    result = _full_data_conditional_result(
                        y_observed, cov_observed, train_pos, test_pos, nearest_psd=nearest_psd,
                    )
                    records.append({**base, "fit_success": True, "error": "", **result["metrics"]})
                    period_records.extend(_full_data_period_rows(base, result, test_original, data.periods))
                except Exception as exc:
                    records.append(_full_data_failure_row(base, exc))

    detail = pd.DataFrame(records)
    period_detail = pd.DataFrame(period_records)
    return {
        "summary": _full_data_summary(detail, ["method"]),
        "period_summary": _full_data_summary(period_detail, ["method", "period_s"]),
        "detail": detail,
        "period_detail": period_detail,
        "wide": wide,
        "period_variances": period_variances,
    }


def predictive_station_holdout_full_data(
    data: ComparisonData,
    cross_period_psd,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    gh08=None,
    lmc_semivariogram=None,
    lmc_mle=None,
    lmc_block_mle=None,
    kronecker=None,
    ggp_model_a=None,
    methods=None,
    test_fraction=0.10,
    n_repeats=5,
    max_events=None,
    max_stations_per_event=80,
    min_stations=8,
    random_state=0,
    nearest_psd=True,
    wide=None,
    iox_full_semivariogram=None,
    sbss_semivariogram=None,
    multivariate_matern=None,
):
    """Hold out stations; score only periods genuinely observed at test stations."""
    wide = predictive_full_data_wide(data) if wide is None else wide.copy()
    period_variances = _full_data_period_variances(wide, data.period_cols)
    curve_sets = _prediction_curve_sets(data, cross_period_psd, methods=methods)
    event_ids = _prediction_event_ids(wide, max_events)
    rng = np.random.default_rng(random_state)
    q = len(data.period_cols)
    records, period_records = [], []

    for repeat in range(int(n_repeats)):
        for eqid in event_ids:
            event = _full_data_event_frame(wide, eqid, data.period_cols, max_stations_per_event, random_state + repeat)
            if len(event) < min_stations:
                continue
            y_flat = event[data.period_cols].to_numpy(dtype=float).reshape(-1)
            observed = np.flatnonzero(np.isfinite(y_flat))
            n_test_stations = min(max(1, int(round(test_fraction * len(event)))), len(event) - 1)
            test_sites = np.sort(rng.choice(len(event), size=n_test_stations, replace=False))
            observed_sites = observed // q
            test_pos = np.flatnonzero(np.isin(observed_sites, test_sites))
            train_pos = np.flatnonzero(~np.isin(observed_sites, test_sites))
            if len(test_pos) == 0 or len(train_pos) == 0:
                continue
            test_original = observed[test_pos]
            y_observed = y_flat[observed]

            for method in curve_sets:
                base = {
                    "task": "full_partial_station_holdout",
                    "method": method,
                    "eqid": eqid,
                    "repeat": repeat,
                    "n_stations": int(len(event)),
                    "n_test_stations": int(n_test_stations),
                }
                try:
                    cov_full = _full_data_covariance(
                        method, event, data, curve_sets, period_variances,
                        pairwise_empirical=pairwise_empirical,
                        pca=pca,
                        semi=semi,
                        gh08=gh08,
                        lmc_semivariogram=lmc_semivariogram,
                        lmc_mle=lmc_mle,
                        lmc_block_mle=lmc_block_mle,
                        kronecker=kronecker,
                        ggp_model_a=ggp_model_a,
                        iox_full_semivariogram=iox_full_semivariogram,
                        sbss_semivariogram=sbss_semivariogram,
                        multivariate_matern=multivariate_matern,
                    )
                    cov_observed = cov_full[np.ix_(observed, observed)]
                    result = _full_data_conditional_result(
                        y_observed, cov_observed, train_pos, test_pos, nearest_psd=nearest_psd,
                    )
                    records.append({**base, "fit_success": True, "error": "", **result["metrics"]})
                    period_records.extend(_full_data_period_rows(base, result, test_original, data.periods))
                except Exception as exc:
                    records.append(_full_data_failure_row(base, exc))

    detail = pd.DataFrame(records)
    period_detail = pd.DataFrame(period_records)
    return {
        "summary": _full_data_summary(detail, ["method"]),
        "period_summary": _full_data_summary(period_detail, ["method", "period_s"]),
        "detail": detail,
        "period_detail": period_detail,
        "wide": wide,
        "period_variances": period_variances,
    }


def predictive_period_holdout_full_data(
    data: ComparisonData,
    cross_period_psd,
    pairwise_empirical=None,
    pca=None,
    semi=None,
    gh08=None,
    lmc_semivariogram=None,
    lmc_mle=None,
    lmc_block_mle=None,
    kronecker=None,
    ggp_model_a=None,
    methods=None,
    holdout_period_sets="all_single",
    max_events=None,
    max_stations_per_event=80,
    min_stations=8,
    random_state=0,
    nearest_psd=True,
    wide=None,
    iox_full_semivariogram=None,
    sbss_semivariogram=None,
    multivariate_matern=None,
):
    """Hold out observed target periods and condition on all other observed entries."""
    wide = predictive_full_data_wide(data) if wide is None else wide.copy()
    period_variances = _full_data_period_variances(wide, data.period_cols)
    curve_sets = _prediction_curve_sets(data, cross_period_psd, methods=methods)
    event_ids = _prediction_event_ids(wide, max_events)
    scenarios = _normalize_period_holdout_sets(data.periods, holdout_period_sets)
    q = len(data.period_cols)
    records, period_records = [], []

    for eqid in event_ids:
        event = _full_data_event_frame(wide, eqid, data.period_cols, max_stations_per_event, random_state)
        if len(event) < min_stations:
            continue
        y_flat = event[data.period_cols].to_numpy(dtype=float).reshape(-1)
        observed = np.flatnonzero(np.isfinite(y_flat))
        observed_periods = observed % q
        y_observed = y_flat[observed]
        covariance_cache = {}
        covariance_errors = {}
        for method in curve_sets:
            try:
                cov_full = _full_data_covariance(
                    method, event, data, curve_sets, period_variances,
                    pairwise_empirical=pairwise_empirical,
                    pca=pca,
                    semi=semi,
                    gh08=gh08,
                    lmc_semivariogram=lmc_semivariogram,
                    lmc_mle=lmc_mle,
                    lmc_block_mle=lmc_block_mle,
                    kronecker=kronecker,
                    ggp_model_a=ggp_model_a,
                    iox_full_semivariogram=iox_full_semivariogram,
                    sbss_semivariogram=sbss_semivariogram,
                    multivariate_matern=multivariate_matern,
                )
                covariance_cache[method] = cov_full[np.ix_(observed, observed)]
            except Exception as exc:
                covariance_errors[method] = exc
        for scenario in scenarios:
            heldout = scenario["heldout_period_indices"]
            test_pos = np.flatnonzero(np.isin(observed_periods, heldout))
            train_pos = np.flatnonzero(~np.isin(observed_periods, heldout))
            if len(test_pos) == 0 or len(train_pos) == 0:
                continue
            test_original = observed[test_pos]
            for method in curve_sets:
                base = {
                    "task": "full_partial_period_holdout",
                    "method": method,
                    "eqid": eqid,
                    "repeat": 0,
                    "n_stations": int(len(event)),
                    "scenario": scenario["scenario"],
                    "heldout_period_secs": scenario["heldout_period_secs"],
                }
                try:
                    if method in covariance_errors:
                        raise covariance_errors[method]
                    cov_observed = covariance_cache[method]
                    result = _full_data_conditional_result(
                        y_observed, cov_observed, train_pos, test_pos, nearest_psd=nearest_psd,
                    )
                    records.append({**base, "fit_success": True, "error": "", **result["metrics"]})
                    period_records.extend(_full_data_period_rows(base, result, test_original, data.periods))
                except Exception as exc:
                    records.append(_full_data_failure_row(base, exc))

    detail = pd.DataFrame(records)
    period_detail = pd.DataFrame(period_records)
    return {
        "summary": _full_data_summary(detail, ["method", "scenario", "heldout_period_secs"]),
        "period_summary": _full_data_summary(period_detail, ["method", "period_s"]),
        "detail": detail,
        "period_detail": period_detail,
        "wide": wide,
        "period_variances": period_variances,
    }


# ---------------------------------------------------------------------------
# Non-PSD / non-predictive evaluation helpers
# ---------------------------------------------------------------------------

NONPSD_MODEL_NAME_MAP = {
    "pairwise": "Pairwise empirical semivariogram",
    "gh08": "GH08 semivariogram",
    "kronecker": "Separable kernel semivariogram",
    "pca_semivariogram": "PCA semivariogram",
    "pca_mle": "PCA MLE",
    "lmc": "LMC semivariogram",
    "lmc_block_mle": "LMC MLE",
    "iox_full_semivariogram": "IOX full semivariogram",
    "sbss_semivariogram": "SBSS semivariogram",
    "multivariate_matern": "Multivariate Matern semivariogram",
    "ggp_model_a": "GGP Model A",
}

NONPSD_MODEL_ORDER = [
    "pairwise",
    "gh08",
    "kronecker",
    "pca_semivariogram",
    "pca_mle",
    "lmc",
    "lmc_block_mle",
    "iox_full_semivariogram",
    "sbss_semivariogram",
    "multivariate_matern",
    "ggp_model_a",
]


@dataclass
class NonPSDEvalConfig:
    """Configuration for evidence tables that exclude PSD and predictive accuracy."""

    data_source: str = "full"
    periods: object = "all"
    n_events: int = 0
    max_sites_per_event: int = 0
    min_sites_per_event: int = 4
    reference_n_events: int = 0
    reference_max_sites_per_event: int = 0
    bootstrap_reps: int = 50
    stability_reps: int = 10
    robustness_reps: int = 10
    seed: int = 12345
    bin_step: int = 5
    sim_sites: int = 35
    sim_test_sites: int = 8
    sim_n_sims: int = 20
    k_pca_max: int = 5


@dataclass
class NonPSDEventBlock:
    eqid: object
    coords_km: np.ndarray
    y: np.ndarray
    recids: object = None

    @property
    def n_sites(self):
        return int(self.y.shape[0])

    @property
    def n_periods(self):
        return int(self.y.shape[1])


def _nonpsd_period_code(period):
    mapping = {
        0.01: "T001",
        0.03: "T003",
        0.06: "T006",
        0.10: "T010",
        0.30: "T030",
        0.60: "T060",
        1.00: "T100",
        3.00: "T300",
        6.00: "T600",
    }
    for sec, code in mapping.items():
        if abs(float(period) - sec) < 1e-8:
            return code
    return f"T{int(round(float(period) * 100)):03d}"


def _nonpsd_normalize_period_selection(periods_available, periods="all"):
    available = np.asarray(periods_available, dtype=float)
    if periods is None or periods == "all":
        idx = list(range(len(available)))
    else:
        selected = []
        for p in periods:
            if isinstance(p, str):
                code = p.upper() if p.upper().startswith("T") else "T" + p.upper()
                selected.append(int(code[1:]) / 100.0)
            else:
                selected.append(float(p))
        idx = []
        for val in selected:
            j = int(np.argmin(np.abs(available - val)))
            if abs(float(available[j]) - float(val)) > 1e-8:
                raise ValueError(f"Selected period {val} not found in {available.tolist()}")
            idx.append(j)
    secs = [float(available[i]) for i in idx]
    return idx, secs, [_nonpsd_period_code(s) for s in secs]


def _nonpsd_powered_exponential(h, length_scale, gamma):
    h = np.asarray(h, dtype=float)
    if float(length_scale) <= 0.0:
        return np.where(np.isclose(h, 0.0), 1.0, 0.0)
    return np.exp(-((np.maximum(h, 0.0) / float(length_scale)) ** float(gamma)))


def _nonpsd_lmc_curves_from_fit(lmc_fit, h_grid):
    b_list = [np.asarray(b, dtype=float) for b in lmc_fit.get("B_raw", lmc_fit.get("B_norm"))]
    length_scales = np.asarray(lmc_fit["length_scales"], dtype=float)
    gamma_pe = np.asarray(lmc_fit["gamma_pe"], dtype=float)
    q = b_list[0].shape[0]
    sill = np.sum([np.diag(b) for b in b_list], axis=0)
    denom = np.sqrt(np.clip(np.outer(sill, sill), 1e-12, None))
    curves = np.zeros((q, q, len(h_grid)), dtype=float)
    for ih, h in enumerate(h_grid):
        cov = np.zeros((q, q), dtype=float)
        for b, ell, gam in zip(b_list, length_scales, gamma_pe):
            cov += b * float(_nonpsd_powered_exponential(h, ell, gam))
        curves[:, :, ih] = cov / denom
    return 0.5 * (curves + np.swapaxes(curves, 0, 1))


def nonpsd_model_curves_from_cache(results, include_lmc=True):
    """Return cached fitted correlation curves keyed by compact model id."""
    curve_store = results.get("cross_period_psd", {}).get("curves", {})
    reverse = {v: k for k, v in NONPSD_MODEL_NAME_MAP.items()}
    reverse.update({"Kronecker semivariogram": "kronecker", "LMC block MLE": "lmc_block_mle"})
    out = {}
    for display_name, value in curve_store.items():
        key = reverse.get(display_name, display_name.lower().replace(" ", "_"))
        h, curves = value
        out[key] = (np.asarray(h, dtype=float), np.asarray(curves, dtype=float))

    h_grid = np.asarray(results.get("data_state", {}).get("h_fine", []), dtype=float)
    if include_lmc and "lmc" not in out and results.get("lmc_semivariogram") is not None:
        out["lmc"] = (h_grid, _nonpsd_lmc_curves_from_fit(results["lmc_semivariogram"], h_grid))
    if "lmc_block_mle" not in out and results.get("lmc_block_mle") is not None:
        out["lmc_block_mle"] = (h_grid, _nonpsd_lmc_curves_from_fit(results["lmc_block_mle"], h_grid))
    return {k: out[k] for k in NONPSD_MODEL_ORDER if k in out}


def nonpsd_subset_curves(curves_by_model, period_idx):
    idx = np.asarray(period_idx, dtype=int)
    out = {}
    for model, (h, curves) in curves_by_model.items():
        curves = np.asarray(curves, dtype=float)
        out[model] = (
            np.asarray(h, dtype=float),
            curves[np.ix_(idx, idx, np.arange(curves.shape[2]))],
        )
    return out


def _nonpsd_full_wide_df_from_period_dfs(results, period_idx):
    ds = results["data_state"]
    period_dfs = ds.get("period_dfs")
    if not period_dfs:
        raise KeyError("data_state['period_dfs'] is missing; cannot build full-data evaluation table")
    all_cols = list(ds["period_cols"])
    period_cols = [all_cols[i] for i in period_idx]
    key_cols = ["recid", "eqid", "station_latitude", "station_longitude"]
    wide = None
    for out_col, df_period in zip(all_cols, period_dfs):
        if out_col not in period_cols:
            continue
        value_col = "scaled_deltaW" if "scaled_deltaW" in df_period.columns else out_col
        keep = [c for c in key_cols if c in df_period.columns] + [value_col]
        df_one = df_period[keep].copy().rename(columns={value_col: out_col})
        merge_keys = [c for c in key_cols if c in df_one.columns]
        wide = df_one if wide is None else wide.merge(df_one, on=merge_keys, how="outer")
    if wide is None:
        raise RuntimeError("No selected period data frames were available.")
    return wide, period_cols


def nonpsd_get_df_from_cache(results, period_idx, data_source="full"):
    ds = results["data_state"]
    all_cols = list(ds["period_cols"])
    period_cols = [all_cols[i] for i in period_idx]
    if str(data_source).lower() in {"full", "all", "period_dfs", "pairwise"}:
        df, period_cols = _nonpsd_full_wide_df_from_period_dfs(results, period_idx)
        df = df.dropna(subset=period_cols, how="all").copy()
        return df, period_cols

    df = ds.get("df_pca")
    if df is None:
        raise KeyError("data_state['df_pca'] is missing")
    keep = ["eqid", "station_latitude", "station_longitude"]
    if "recid" in df.columns:
        keep.append("recid")
    keep += period_cols
    df = df[[c for c in keep if c in df.columns]].dropna(subset=period_cols).copy()
    return df, period_cols


def _nonpsd_latlon_to_xy_km(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    r = 6371.0
    lat0 = np.nanmean(lat) * np.pi / 180.0
    x = r * np.cos(lat0) * (lon * np.pi / 180.0)
    y = r * (lat * np.pi / 180.0)
    coords = np.column_stack([x, y])
    coords -= np.nanmean(coords, axis=0, keepdims=True)
    return coords


def nonpsd_make_event_blocks(
    df,
    period_cols,
    n_events=0,
    max_sites_per_event=0,
    min_sites_per_event=4,
    seed=123,
    center_event_period=True,
    event_ids=None,
    require_complete=False,
):
    rng = np.random.default_rng(seed)
    if event_ids is None:
        ids = list(df.groupby("eqid").size().sort_values(ascending=False).index)
    else:
        ids = list(event_ids)
    if n_events and n_events > 0:
        ids = ids[:int(n_events)]

    blocks = []
    for eqid in ids:
        g = df[df["eqid"] == eqid].copy()
        if require_complete:
            g = g.dropna(subset=list(period_cols)).copy()
        else:
            g = g.dropna(subset=list(period_cols), how="all").copy()
        if len(g) < int(min_sites_per_event):
            continue
        if max_sites_per_event and max_sites_per_event > 0 and len(g) > int(max_sites_per_event):
            g = g.sample(int(max_sites_per_event), random_state=int(rng.integers(0, 2**31 - 1)))
        y = g[list(period_cols)].to_numpy(dtype=float)
        if center_event_period:
            col_mean = np.full(y.shape[1], np.nan, dtype=float)
            observed_cols = np.isfinite(y).any(axis=0)
            if np.any(observed_cols):
                col_mean[observed_cols] = np.nanmean(y[:, observed_cols], axis=0)
                y = y - col_mean[None, :]
        keep = np.isfinite(y).any(axis=1)
        if int(np.sum(keep)) < int(min_sites_per_event):
            continue
        if not np.all(keep):
            g = g.iloc[keep].copy()
            y = y[keep]
        coords = _nonpsd_latlon_to_xy_km(
            g["station_latitude"].to_numpy(dtype=float),
            g["station_longitude"].to_numpy(dtype=float),
        )
        recids = g["recid"].to_numpy() if "recid" in g.columns else None
        blocks.append(NonPSDEventBlock(eqid=eqid, coords_km=coords, y=y, recids=recids))
    return blocks


def _nonpsd_subset_wide_df(df, period_cols, n_events, site_cap, rng, min_sites_per_event=4):
    event_sizes = df.groupby("eqid").size().sort_values(ascending=False)
    event_ids = event_sizes.index.to_numpy()
    if n_events and n_events > 0:
        n_take = min(int(n_events), len(event_ids))
        event_ids = rng.choice(event_ids, size=n_take, replace=False)
    pieces = []
    for eqid in event_ids:
        g = df[df["eqid"] == eqid].dropna(subset=list(period_cols), how="all").copy()
        if len(g) < int(min_sites_per_event):
            continue
        if site_cap and site_cap > 0 and len(g) > int(site_cap):
            g = g.sample(int(site_cap), random_state=int(rng.integers(0, 2**31 - 1)))
        pieces.append(g)
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def _nonpsd_period_dfs_from_wide(df, period_cols):
    df = df.copy()
    if "recid" not in df.columns:
        df.insert(0, "recid", np.arange(len(df), dtype=int))
    key_cols = ["recid", "eqid", "station_latitude", "station_longitude"]
    out = []
    for col in period_cols:
        keep = [c for c in key_cols if c in df.columns] + [col]
        out.append(df[keep].dropna(subset=[col]).rename(columns={col: "scaled_deltaW"}).copy())
    return out


def _nonpsd_corr_stats(values_i, values_j):
    values_i = np.asarray(values_i, dtype=float)
    values_j = np.asarray(values_j, dtype=float)
    valid = np.isfinite(values_i) & np.isfinite(values_j)
    values_i = values_i[valid]
    values_j = values_j[valid]
    if values_i.size == 0:
        return np.zeros(6, dtype=float)
    return np.asarray([
        float(values_i.size),
        float(np.sum(values_i)),
        float(np.sum(values_j)),
        float(np.sum(values_i * values_i)),
        float(np.sum(values_j * values_j)),
        float(np.sum(values_i * values_j)),
    ], dtype=float)


def _nonpsd_corr_from_stats(stats):
    n, sx, sy, sx2, sy2, sxy = np.asarray(stats, dtype=float)
    if n < 2:
        return np.nan
    cov = sxy - sx * sy / n
    vx = sx2 - sx * sx / n
    vy = sy2 - sy * sy / n
    if vx <= 0.0 or vy <= 0.0:
        return np.nan
    return float(cov / np.sqrt(vx * vy))


def _nonpsd_make_event_variogram_cache(template, common, df, period_cols, min_stations=5):
    """Precompute event-level bivariate variogram contributions for fast resampling."""
    df = df.copy()
    if "recid" not in df.columns:
        df.insert(0, "recid", np.arange(len(df), dtype=int))
    period_cols = list(period_cols)
    nper = len(period_cols)
    nmax = round(template.max_distance / template.bin_size)
    event_cache = {}

    for eqid, group in df.groupby("eqid", sort=False):
        gamma_sum = np.zeros((nper, nper, nmax), dtype=float)
        count_sum = np.zeros((nper, nper, nmax), dtype=float)
        record_counts = np.zeros((nper, nper), dtype=int)
        corr_stats = np.zeros((nper, nper, 6), dtype=float)

        for i in range(nper):
            col_i = period_cols[i]
            for j in range(i + 1):
                col_j = period_cols[j]
                pair = group.dropna(subset=[col_i, col_j]).copy()
                record_counts[i, j] = len(pair)
                record_counts[j, i] = len(pair)
                if i != j:
                    stats = _nonpsd_corr_stats(pair[col_i].to_numpy(), pair[col_j].to_numpy())
                    corr_stats[i, j, :] = stats
                    corr_stats[j, i, :] = np.asarray([stats[0], stats[2], stats[1], stats[4], stats[3], stats[5]])
                if len(pair) < int(min_stations):
                    continue
                gamma_ij, counts_ij = common.compute_empirical_bivariate_variogram(
                    pair,
                    col_i,
                    col_j,
                    bin_size=template.bin_size,
                    max_distance=template.max_distance,
                )
                weighted = np.asarray(gamma_ij, dtype=float) * np.asarray(counts_ij, dtype=float)
                gamma_sum[i, j, :] = weighted
                count_sum[i, j, :] = counts_ij
                if i != j:
                    gamma_sum[j, i, :] = weighted
                    count_sum[j, i, :] = counts_ij

        event_cache[eqid] = {
            "gamma_sum": gamma_sum,
            "count_sum": count_sum,
            "record_counts": record_counts,
            "corr_stats": corr_stats,
        }

    return {
        "event_cache": event_cache,
        "period_cols": tuple(period_cols),
        "nper": nper,
        "nmax": nmax,
        "min_stations": int(min_stations),
    }


def _nonpsd_comparison_data_from_event_cache(template, common, df, period_cols, cache, selected_eqids):
    """Build ComparisonData by summing cached event-level variogram contributions."""
    df = df.copy()
    if "recid" not in df.columns:
        df.insert(0, "recid", np.arange(len(df), dtype=int))
    period_cols = list(period_cols)
    period_idx = [list(template.period_cols).index(col) for col in period_cols]
    periods = np.asarray(template.periods, dtype=float)[period_idx]
    nper = len(period_cols)
    nmax = cache["nmax"]

    gamma_sum = np.zeros((nper, nper, nmax), dtype=float)
    count_sum = np.zeros((nper, nper, nmax), dtype=float)
    record_counts = np.zeros((nper, nper), dtype=int)
    corr_stats = np.zeros((nper, nper, 6), dtype=float)

    selected_set = set(selected_eqids)
    for eqid in selected_eqids:
        contrib = cache["event_cache"].get(eqid)
        if contrib is None or eqid not in selected_set:
            continue
        gamma_sum += contrib["gamma_sum"]
        count_sum += contrib["count_sum"]
        record_counts += contrib["record_counts"]
        corr_stats += contrib["corr_stats"]

    gamma_emp = [[np.zeros(nmax) for _ in range(nper)] for _ in range(nper)]
    count_emp = [[np.zeros(nmax) for _ in range(nper)] for _ in range(nper)]
    for i in range(nper):
        for j in range(nper):
            np.divide(gamma_sum[i, j], count_sum[i, j], out=gamma_emp[i][j], where=count_sum[i, j] > 0)
            count_emp[i][j] = count_sum[i, j].copy()

    rho_0 = np.eye(nper)
    for i in range(nper):
        for j in range(i):
            rho = _nonpsd_corr_from_stats(corr_stats[i, j, :])
            rho_0[i, j] = rho
            rho_0[j, i] = rho

    period_dfs = _nonpsd_period_dfs_from_wide(df, period_cols)
    emp_sills = common.empirical_sills(gamma_emp, tail_bins=template.tail_bins)
    df_pca = df[["eqid", "station_latitude", "station_longitude", *period_cols]].dropna().copy()
    return ComparisonData(
        project_root=template.project_root,
        pca_data_path=template.pca_data_path,
        full_data_dir=template.full_data_dir,
        periods=periods,
        bin_size=template.bin_size,
        max_distance=template.max_distance,
        tail_bins=template.tail_bins,
        h_bins=template.h_bins,
        h_fine=template.h_fine,
        df_pca_source=df.copy(),
        df_pca=df_pca,
        period_cols=period_cols,
        events_pca=df_pca["eqid"].unique(),
        period_dfs=period_dfs,
        gamma_emp=gamma_emp,
        count_emp=count_emp,
        pair_record_counts=record_counts,
        rho_0=rho_0,
        emp_sills=emp_sills,
    )


def _nonpsd_comparison_data_from_wide(template, common, df, period_cols):
    df = df.copy()
    if "recid" not in df.columns:
        df.insert(0, "recid", np.arange(len(df), dtype=int))
    period_cols = list(period_cols)
    period_idx = [list(template.period_cols).index(col) for col in period_cols]
    periods = np.asarray(template.periods, dtype=float)[period_idx]
    period_dfs = _nonpsd_period_dfs_from_wide(df, period_cols)
    gamma_emp, count_emp, pair_record_counts = common.pool_pairwise_crossvariograms(
        period_dfs,
        periods,
        bin_size=template.bin_size,
        max_distance=template.max_distance,
    )
    rho_0 = common.pairwise_lag0_corr(period_dfs)
    emp_sills = common.empirical_sills(gamma_emp, tail_bins=template.tail_bins)
    df_pca = df[["eqid", "station_latitude", "station_longitude", *period_cols]].dropna().copy()
    return ComparisonData(
        project_root=template.project_root,
        pca_data_path=template.pca_data_path,
        full_data_dir=template.full_data_dir,
        periods=periods,
        bin_size=template.bin_size,
        max_distance=template.max_distance,
        tail_bins=template.tail_bins,
        h_bins=template.h_bins,
        h_fine=template.h_fine,
        df_pca_source=df.copy(),
        df_pca=df_pca,
        period_cols=period_cols,
        events_pca=df_pca["eqid"].unique(),
        period_dfs=period_dfs,
        gamma_emp=gamma_emp,
        count_emp=count_emp,
        pair_record_counts=pair_record_counts,
        rho_0=rho_0,
        emp_sills=emp_sills,
    )


def _nonpsd_curve_rmse(ref_h, ref_curves, fit_h, fit_curves):
    ref_h = np.asarray(ref_h, dtype=float)
    fit_h = np.asarray(fit_h, dtype=float)
    ref_curves = np.asarray(ref_curves, dtype=float)
    fit_curves = np.asarray(fit_curves, dtype=float)
    if ref_curves.shape[:2] != fit_curves.shape[:2]:
        return np.nan
    q = ref_curves.shape[0]
    errs = []
    for i in range(q):
        for j in range(i, q):
            ref = ref_curves[i, j, :]
            fit = np.interp(ref_h, fit_h, fit_curves[i, j, :], left=np.nan, right=np.nan)
            valid = np.isfinite(ref) & np.isfinite(fit)
            if np.any(valid):
                errs.append((fit[valid] - ref[valid]) ** 2)
    if not errs:
        return np.nan
    return float(np.sqrt(np.mean(np.concatenate(errs))))


def _nonpsd_refit_semivariogram_curves(sub_data, mle, semi, methods, lmc_iterations=10):
    import time

    out = {}
    errors = {}
    timings = {}
    if "pairwise" in methods:
        t0 = time.perf_counter()
        try:
            pairwise = fit_pairwise_empirical(sub_data, semi["pairwise"])
            grid = sub_data.h_fine if "fit" in pairwise else sub_data.h_bins
            curves = (
                pairwise_fitted_corr_curves(sub_data, pairwise["fit"], sub_data.h_fine)
                if "fit" in pairwise
                else pairwise_empirical_corr_curves(sub_data)
            )
            out["pairwise"] = (grid, curves)
        except Exception as exc:
            errors["pairwise"] = f"{type(exc).__name__}: {exc}"
        timings["pairwise"] = float(time.perf_counter() - t0)
    if "gh08" in methods:
        t0 = time.perf_counter()
        try:
            direct = semi["product"].fit_direct_auto_pe(
                sub_data.gamma_emp,
                sub_data.h_bins,
                sub_data.h_fine,
                tail_bins=sub_data.tail_bins,
            )
            out["gh08"] = (sub_data.h_fine, goda_hong_curves(direct, sub_data.rho_0, sub_data.h_fine))
        except Exception as exc:
            errors["gh08"] = f"{type(exc).__name__}: {exc}"
        timings["gh08"] = float(time.perf_counter() - t0)
    if "kronecker" in methods:
        t0 = time.perf_counter()
        try:
            sep = semi["product"].fit_separable_kernel(
                sub_data.gamma_emp,
                sub_data.h_bins,
                sub_data.h_fine,
                tail_bins=sub_data.tail_bins,
            )
            out["kronecker"] = (sub_data.h_fine, separable_curves(sep, sub_data.rho_0, sub_data.h_fine))
        except Exception as exc:
            errors["kronecker"] = f"{type(exc).__name__}: {exc}"
        timings["kronecker"] = float(time.perf_counter() - t0)
    if "pca_semivariogram" in methods and len(sub_data.df_pca) >= max(8, len(sub_data.period_cols) + 2):
        t0 = time.perf_counter()
        try:
            pca_fit = semi["pca"].fit_pca_workflow(
                sub_data.df_pca,
                sub_data.period_cols,
                sub_data.events_pca,
                sub_data.h_bins,
                sub_data.h_fine,
                sub_data.bin_size,
                len(sub_data.h_bins),
                tail_bins=sub_data.tail_bins,
            )
            out["pca_semivariogram"] = (
                sub_data.h_fine,
                pca_period_pair_curves(pca_fit, semi["pca"], sub_data.h_fine),
            )
        except Exception as exc:
            errors["pca_semivariogram"] = f"{type(exc).__name__}: {exc}"
        timings["pca_semivariogram"] = float(time.perf_counter() - t0)
    elif "pca_semivariogram" in methods:
        errors["pca_semivariogram"] = f"insufficient complete rows: {len(sub_data.df_pca)}"
        timings["pca_semivariogram"] = 0.0
    if "pca_mle" in methods and len(sub_data.df_pca) >= max(8, len(sub_data.period_cols) + 2):
        t0 = time.perf_counter()
        try:
            pca_fit = fit_pca_full_mle(sub_data, mle, mle["common"])
            out["pca_mle"] = (
                sub_data.h_fine,
                pca_period_pair_curves(pca_fit, mle["pca"], sub_data.h_fine),
            )
        except Exception as exc:
            errors["pca_mle"] = f"{type(exc).__name__}: {exc}"
        timings["pca_mle"] = float(time.perf_counter() - t0)
    elif "pca_mle" in methods:
        errors["pca_mle"] = f"insufficient complete rows: {len(sub_data.df_pca)}"
        timings["pca_mle"] = 0.0
    if "lmc" in methods:
        t0 = time.perf_counter()
        try:
            lmc_fit = fit_lmc_semivariogram(sub_data, semi, n_iterations=lmc_iterations)
            out["lmc"] = (
                sub_data.h_fine,
                lmc_period_pair_curves(lmc_fit, semi["lmc"], sub_data.h_fine),
            )
        except Exception as exc:
            errors["lmc"] = f"{type(exc).__name__}: {exc}"
        timings["lmc"] = float(time.perf_counter() - t0)
    return out, errors, timings


def _nonpsd_sample_full_event_ids_with_common(full_event_ids, common_event_ids, n_events, rng):
    """Sample full-universe event ids without replacement, forcing common-event overlap."""
    full_ids = list(pd.Index(full_event_ids).dropna().unique())
    common_ids = list(pd.Index(common_event_ids).dropna().unique())
    n_take = min(int(n_events), len(full_ids))
    if n_take <= 0:
        return np.asarray([], dtype=object)

    common_set = set(common_ids)
    common_pool = [eqid for eqid in full_ids if eqid in common_set]
    if not common_pool:
        return np.asarray(rng.choice(np.asarray(full_ids, dtype=object), size=n_take, replace=False), dtype=object)

    anchor = rng.choice(np.asarray(common_pool, dtype=object), size=1, replace=False)[0]
    remaining_pool = [eqid for eqid in full_ids if eqid != anchor]
    if n_take == 1:
        selected = np.asarray([anchor], dtype=object)
    else:
        rest = rng.choice(np.asarray(remaining_pool, dtype=object), size=n_take - 1, replace=False)
        selected = np.asarray([anchor, *list(rest)], dtype=object)
    rng.shuffle(selected)
    return selected


def _nonpsd_parse_event_id_field(value):
    """Parse whitespace/comma separated event ids from cached sample CSV fields."""
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        tokens = list(value)
    else:
        tokens = [value]

    parsed = []
    for token in tokens:
        if pd.isna(token):
            continue
        if isinstance(token, str):
            stripped = token.strip()
            if not stripped:
                continue
            try:
                parsed.append(int(stripped))
                continue
            except ValueError:
                try:
                    parsed.append(float(stripped))
                    continue
                except ValueError:
                    parsed.append(stripped)
                    continue
        parsed.append(token)
    return parsed


def _nonpsd_n_reps_for_event(n_reps, n_events):
    """Return a scalar repetition count from either an int or event-count mapping."""
    if isinstance(n_reps, Mapping):
        if int(n_events) in n_reps:
            return int(n_reps[int(n_events)])
        if str(int(n_events)) in n_reps:
            return int(n_reps[str(int(n_events))])
        raise KeyError(f"n_reps does not include n_events={int(n_events)}")
    return int(n_reps)


def _nonpsd_full_event_sample_lookup(sampled_events, full_event_ids, n_event_grid=None, n_reps=None):
    """Build a lookup of cached sampled full-event ids keyed by requested event count and replicate."""
    if sampled_events is None:
        return None
    sample_df = pd.read_csv(sampled_events) if isinstance(sampled_events, (str, Path)) else pd.DataFrame(sampled_events).copy()
    if sample_df.empty:
        raise ValueError("sampled_events is empty")
    required = {"n_events", "rep", "sampled_event_ids"}
    missing = required.difference(sample_df.columns)
    if missing:
        raise ValueError(f"sampled_events is missing required columns: {sorted(missing)}")

    full_id_by_str = {str(eqid): eqid for eqid in pd.Index(full_event_ids).dropna().unique()}
    lookup = {}
    for _, row in sample_df.iterrows():
        requested_n = int(row["requested_n_events"]) if "requested_n_events" in row and pd.notna(row["requested_n_events"]) else int(row["n_events"])
        n_events = int(row["n_events"])
        rep = int(row["rep"])
        parsed = _nonpsd_parse_event_id_field(row["sampled_event_ids"])
        selected = [full_id_by_str.get(str(eqid), eqid) for eqid in parsed]
        if not selected:
            raise ValueError(f"sampled_events has no sampled_event_ids for n_events={n_events}, rep={rep}")
        selected = np.asarray(selected, dtype=object)
        lookup[(requested_n, rep)] = selected
        lookup[(n_events, rep)] = selected

    if n_event_grid is not None and n_reps is not None:
        missing_keys = [
            (int(n_events), int(rep))
            for n_events in n_event_grid
            for rep in range(_nonpsd_n_reps_for_event(n_reps, n_events))
            if (int(n_events), int(rep)) not in lookup
        ]
        if missing_keys:
            preview = ", ".join(f"n={n}, rep={rep}" for n, rep in missing_keys[:8])
            raise ValueError(f"sampled_events does not cover requested samples: {preview}")
    return lookup


def _nonpsd_normalize_site_caps(site_caps):
    """Normalize site-cap settings; None/"all" means use every available site."""
    if site_caps is None:
        return ("all",)
    if isinstance(site_caps, str):
        site_caps = (site_caps,)
    elif np.isscalar(site_caps):
        site_caps = (site_caps,)
    normalized = []
    for cap in site_caps:
        if cap is None:
            normalized.append("all")
            continue
        if isinstance(cap, str):
            cap_clean = cap.strip().lower()
            if cap_clean in {"", "all", "none", "full", "no_cap", "unlimited"}:
                normalized.append("all")
                continue
            cap = int(cap_clean)
        cap = int(cap)
        normalized.append("all" if cap <= 0 else cap)
    return tuple(normalized)


def nonpsd_refit_stability_full_event_sample_evaluation(
    context,
    data,
    mle,
    semi,
    methods=("pairwise", "gh08", "kronecker", "pca_semivariogram", "lmc"),
    n_event_grid=(40, 60, 80, 100, 120),
    site_caps=(40, 160, 320),
    n_reps=10,
    seed=None,
    lmc_iterations=10,
    curve_rmse_threshold=0.10,
    sampled_events=None,
):
    """Refit after sampling events from the full event universe.

    Non-PCA methods use the sampled full-event rows. PCA methods keep the same
    full-event x-axis but fit only the sampled events that are also present in
    the aligned/common-event data.
    """
    import time

    config = context["config"]
    seed = config.seed if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    period_cols = context["period_cols"]
    reference_curves = context["curves"]
    pca_methods = {model for model in methods if str(model).startswith("pca")}
    full_methods = tuple(model for model in methods if model not in pca_methods)
    pca_methods = tuple(model for model in methods if model in pca_methods)

    keep = ["eqid", "station_latitude", "station_longitude"]
    if "recid" in context["df"].columns:
        keep.append("recid")
    keep += list(period_cols)
    full_df = context["df"][[c for c in keep if c in context["df"].columns]].dropna(
        subset=list(period_cols), how="all"
    ).copy()

    aligned_keep = ["eqid", "station_latitude", "station_longitude"]
    if "recid" in data.df_pca.columns:
        aligned_keep.append("recid")
    aligned_keep += list(period_cols)
    aligned_df = data.df_pca[[c for c in aligned_keep if c in data.df_pca.columns]].dropna(
        subset=list(period_cols), how="all"
    ).copy()

    full_event_ids = pd.Index(full_df["eqid"].dropna().unique())
    common_event_ids = pd.Index(aligned_df["eqid"].dropna().unique())
    common_set = set(common_event_ids)
    sample_lookup = _nonpsd_full_event_sample_lookup(
        sampled_events,
        full_event_ids,
        n_event_grid=n_event_grid,
        n_reps=n_reps,
    )
    site_caps = _nonpsd_normalize_site_caps(site_caps)
    full_event_variogram_cache = None
    if full_methods and any(site_cap == "all" for site_cap in site_caps):
        cache_meta = (tuple(period_cols), int(config.min_sites_per_event), int(len(full_df)), int(full_df["eqid"].nunique()))
        cached_pack = context.get("_full_event_variogram_cache_pack")
        if cached_pack is not None and cached_pack.get("meta") == cache_meta:
            full_event_variogram_cache = cached_pack["cache"]
        else:
            full_event_variogram_cache = _nonpsd_make_event_variogram_cache(
                data,
                mle["common"],
                full_df,
                period_cols,
                min_stations=config.min_sites_per_event,
            )
            context["_full_event_variogram_cache_pack"] = {
                "meta": cache_meta,
                "cache": full_event_variogram_cache,
            }

    detail_rows = []
    subset_cache = {}
    for requested_n_events in n_event_grid:
        full_n_events = min(int(requested_n_events), int(len(full_event_ids)))
        if full_n_events < 1:
            continue
        for site_cap in site_caps:
            site_cap_for_subset = None if site_cap == "all" else int(site_cap)
            site_cap_label = "all" if site_cap_for_subset is None else int(site_cap_for_subset)
            for rep in range(_nonpsd_n_reps_for_event(n_reps, requested_n_events)):
                if sample_lookup is None:
                    selected_eqids = _nonpsd_sample_full_event_ids_with_common(
                        full_event_ids, common_event_ids, full_n_events, rng
                    )
                else:
                    selected_eqids = sample_lookup[(int(requested_n_events), int(rep))]
                selected_set = set(selected_eqids)
                selected_common = [eqid for eqid in selected_eqids if eqid in common_set]

                source_specs = []
                if full_methods:
                    source_specs.append({
                        "methods": full_methods,
                        "df": full_df[full_df["eqid"].isin(selected_set)].copy(),
                        "source": "full",
                        "cache_tag": "full",
                    })
                if pca_methods:
                    source_specs.append({
                        "methods": pca_methods,
                        "df": aligned_df[aligned_df["eqid"].isin(set(selected_common))].copy(),
                        "source": "aligned_common_in_full_sample",
                        "cache_tag": "pca_common",
                    })

                for spec in source_specs:
                    if spec["df"].empty:
                        for model in spec["methods"]:
                            detail_rows.append({
                                "model": model,
                                "display_name": NONPSD_MODEL_NAME_MAP.get(model, model),
                                "n_events": int(full_n_events),
                                "requested_n_events": int(requested_n_events),
                                "site_cap": site_cap_label,
                                "rep": int(rep),
                                "total_rows": 0,
                                "complete_rows": 0,
                                "events_used": 0,
                                "full_sample_events": int(len(selected_eqids)),
                                "common_events_in_full_sample": int(len(selected_common)),
                                "sample_source": spec["source"],
                                "subsample_seconds": 0.0,
                                "sub_data_seconds": 0.0,
                                "fit_seconds": np.nan,
                                "fit_group_seconds": 0.0,
                                "curve_rmse_vs_full_fit": np.nan,
                                "fit_success": False,
                                "error": "no rows after applying full-event/common-event sample",
                            })
                        continue

                    cache_key = (spec["cache_tag"], tuple(selected_eqids), site_cap_label, int(rep))
                    if cache_key in subset_cache:
                        sub_df, sub_data, subsample_seconds, sub_data_seconds = subset_cache[cache_key]
                    else:
                        t_subsample = time.perf_counter()
                        sub_df = _nonpsd_subset_wide_df(
                            spec["df"],
                            period_cols,
                            n_events=0,
                            site_cap=site_cap_for_subset,
                            rng=rng,
                            min_sites_per_event=config.min_sites_per_event,
                        )
                        subsample_seconds = float(time.perf_counter() - t_subsample)
                        t_sub_data = time.perf_counter()
                        if sub_df.empty:
                            sub_data = None
                        elif (
                            spec["cache_tag"] == "full"
                            and site_cap_for_subset is None
                            and full_event_variogram_cache is not None
                        ):
                            sub_data = _nonpsd_comparison_data_from_event_cache(
                                data,
                                mle["common"],
                                sub_df,
                                period_cols,
                                full_event_variogram_cache,
                                sub_df["eqid"].dropna().unique(),
                            )
                        else:
                            sub_data = _nonpsd_comparison_data_from_wide(
                                data,
                                mle["common"],
                                sub_df,
                                period_cols,
                            )
                        sub_data_seconds = float(time.perf_counter() - t_sub_data) if sub_data is not None else 0.0
                        subset_cache[cache_key] = (sub_df, sub_data, subsample_seconds, sub_data_seconds)

                    if sub_df.empty or sub_data is None:
                        for model in spec["methods"]:
                            detail_rows.append({
                                "model": model,
                                "display_name": NONPSD_MODEL_NAME_MAP.get(model, model),
                                "n_events": int(full_n_events),
                                "requested_n_events": int(requested_n_events),
                                "site_cap": site_cap_label,
                                "rep": int(rep),
                                "total_rows": int(len(sub_df)),
                                "complete_rows": 0,
                                "events_used": int(sub_df["eqid"].nunique()) if "eqid" in sub_df else 0,
                                "full_sample_events": int(len(selected_eqids)),
                                "common_events_in_full_sample": int(len(selected_common)),
                                "sample_source": spec["source"],
                                "subsample_seconds": subsample_seconds,
                                "sub_data_seconds": sub_data_seconds,
                                "fit_seconds": np.nan,
                                "fit_group_seconds": 0.0,
                                "curve_rmse_vs_full_fit": np.nan,
                                "fit_success": False,
                                "error": "empty subsample",
                            })
                        continue

                    t_fit_group = time.perf_counter()
                    fitted, errors, fit_timings = _nonpsd_refit_semivariogram_curves(
                        sub_data,
                        mle=mle,
                        semi=semi,
                        methods=spec["methods"],
                        lmc_iterations=lmc_iterations,
                    )
                    fit_group_seconds = float(time.perf_counter() - t_fit_group)

                    base_row = {
                        "n_events": int(full_n_events),
                        "requested_n_events": int(requested_n_events),
                        "site_cap": site_cap_label,
                        "rep": int(rep),
                        "total_rows": int(len(sub_df)),
                        "complete_rows": int(len(sub_data.df_pca)),
                        "events_used": int(sub_df["eqid"].nunique()),
                        "full_sample_events": int(len(selected_eqids)),
                        "common_events_in_full_sample": int(len(selected_common)),
                        "sample_source": spec["source"],
                        "subsample_seconds": subsample_seconds,
                        "sub_data_seconds": sub_data_seconds,
                        "fit_group_seconds": fit_group_seconds,
                    }
                    for model, error in errors.items():
                        detail_rows.append({
                            "model": model,
                            "display_name": NONPSD_MODEL_NAME_MAP.get(model, model),
                            **base_row,
                            "fit_seconds": float(fit_timings.get(model, np.nan)),
                            "curve_rmse_vs_full_fit": np.nan,
                            "fit_success": False,
                            "error": error,
                        })
                    for model in spec["methods"]:
                        if model in errors:
                            continue
                        if model not in fitted:
                            detail_rows.append({
                                "model": model,
                                "display_name": NONPSD_MODEL_NAME_MAP.get(model, model),
                                **base_row,
                                "fit_seconds": float(fit_timings.get(model, np.nan)),
                                "curve_rmse_vs_full_fit": np.nan,
                                "fit_success": False,
                                "error": "fit did not return curves",
                            })
                            continue
                        if model not in reference_curves:
                            continue
                        fit_h, fit_curves = fitted[model]
                        ref_h, ref_curves = reference_curves[model]
                        detail_rows.append({
                            "model": model,
                            "display_name": NONPSD_MODEL_NAME_MAP.get(model, model),
                            **base_row,
                            "fit_seconds": float(fit_timings.get(model, np.nan)),
                            "curve_rmse_vs_full_fit": _nonpsd_curve_rmse(ref_h, ref_curves, fit_h, fit_curves),
                            "curve_rmse_pair_mode": "unique_period_pairs",
                            "fit_success": True,
                            "error": "",
                        })

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return {"refit_stability": detail, "refit_requirement": pd.DataFrame()}

    success = detail[detail.get("fit_success", True).astype(bool)].copy()
    if success.empty:
        return {"refit_stability": detail, "refit_requirement": pd.DataFrame()}

    med = success.groupby(["model", "display_name", "n_events", "site_cap"], as_index=False).median(numeric_only=True)
    req_rows = []
    for (model, display_name), group in med.groupby(["model", "display_name"], sort=False):
        ok = group[group["curve_rmse_vs_full_fit"] <= float(curve_rmse_threshold)].copy()
        if ok.empty:
            best = group.sort_values("curve_rmse_vs_full_fit").iloc[0]
            best_site = best["site_cap"]
            best_site_label = "all sites" if str(best_site).lower() == "all" else f"cap {int(best_site)}"
            req = f"not reached; best tested {int(best['n_events'])} full events x {best_site_label}"
            req_rmse = float(best["curve_rmse_vs_full_fit"])
            reached = False
        else:
            best = ok.sort_values(["n_events", "site_cap"]).iloc[0]
            best_site = best["site_cap"]
            best_site_label = "all sites" if str(best_site).lower() == "all" else f"cap {int(best_site)}"
            req = f"~{int(best['n_events'])} full events x {best_site_label}"
            req_rmse = float(best["curve_rmse_vs_full_fit"])
            reached = True
        req_rows.append({
            "model": model,
            "display_name": display_name,
            "curve_rmse_threshold": float(curve_rmse_threshold),
            "requirement": req,
            "median_curve_rmse_at_requirement": req_rmse,
            "threshold_reached": bool(reached),
            "note": "Events are sampled without replacement from the full event universe; PCA fits only the sampled events that are also common/aligned.",
        })
    return {"refit_stability": detail, "refit_requirement": pd.DataFrame(req_rows)}


def prepare_nonpsd_evaluation(cache_or_results, config=None):
    """Prepare data/curves once, then reuse the returned context across cells."""
    config = config or NonPSDEvalConfig()
    if isinstance(cache_or_results, (str, Path)):
        results = load_pickle_cross_platform(Path(cache_or_results))
        if "data_state" in results and "data" not in results:
            results["data"] = comparison_data_from_state(results["data_state"])
    else:
        results = cache_or_results

    periods_all = np.asarray(results["data_state"]["periods"], dtype=float)
    period_idx, periods_sec, period_codes = _nonpsd_normalize_period_selection(periods_all, config.periods)
    curves = nonpsd_subset_curves(nonpsd_model_curves_from_cache(results, include_lmc=True), period_idx)
    df, period_cols = nonpsd_get_df_from_cache(results, period_idx, data_source=config.data_source)
    blocks = nonpsd_make_event_blocks(
        df,
        period_cols,
        n_events=config.n_events,
        max_sites_per_event=config.max_sites_per_event,
        min_sites_per_event=config.min_sites_per_event,
        seed=config.seed,
        center_event_period=True,
        require_complete=False,
    )
    reference_blocks = nonpsd_make_event_blocks(
        df,
        period_cols,
        n_events=config.reference_n_events,
        max_sites_per_event=config.reference_max_sites_per_event,
        min_sites_per_event=config.min_sites_per_event,
        seed=config.seed,
        center_event_period=True,
        require_complete=False,
    )
    h_bins = np.asarray(results["data_state"].get("h_bins", np.arange(2, 102, 2)), dtype=float)
    h_bins_eval = h_bins[::max(1, int(config.bin_step))]
    if not blocks:
        raise RuntimeError("No non-PSD evaluation event blocks available after filtering.")
    return {
        "results": results,
        "config": config,
        "period_idx": period_idx,
        "periods_sec": periods_sec,
        "period_codes": period_codes,
        "curves": curves,
        "data_source": config.data_source,
        "df": df,
        "period_cols": period_cols,
        "blocks": blocks,
        "reference_blocks": reference_blocks,
        "h_bins_eval": h_bins_eval,
        "k_pca": min(config.k_pca_max, len(period_idx)),
    }


