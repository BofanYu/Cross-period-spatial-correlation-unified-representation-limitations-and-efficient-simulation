"""Figure builders used by plot_figures.py; values come from the paper snapshot."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, ScalarFormatter, LogLocator
import numpy as np
import pandas as pd
from scripts.analysis import core as mch
from scripts.analysis.core import _haversine_between
from scripts.analysis import styles as mcr




SELECTED_PERIODS = (0.1, 6.0)

def station_counts(frame) -> np.ndarray:
    return frame.groupby("eqid", sort=False).size().to_numpy(dtype=int)

def pair_separations(frame) -> np.ndarray:
    distances = []
    for _, event in frame.groupby("eqid", sort=False):
        latitude = event["station_latitude"].to_numpy(dtype=float)
        longitude = event["station_longitude"].to_numpy(dtype=float)
        distance_matrix = _haversine_between(latitude, longitude, latitude, longitude)
        distances.append(distance_matrix[np.triu_indices(len(event), k=1)])
    return np.concatenate(distances) if distances else np.empty(0, dtype=float)

def make_data_figure(frames):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 19,
            "axes.labelsize": 22,
            "axes.linewidth": 1.45,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "xtick.major.width": 1.35,
            "ytick.major.width": 1.35,
            "xtick.major.size": 6.0,
            "ytick.major.size": 6.0,
            "legend.fontsize": 20,
        }
    )

    colors = {0.1: "#4C92BF", 6.0: "#F39C53"}
    labels = {0.1: "0.1 s", 6.0: "6 s"}
    station_bin_edges = np.arange(35.0, 560.0 + 15.0, 15.0)
    distance_bin_edges = np.arange(0.0, 675.0 + 25.0, 25.0)

    station_data = {period: station_counts(frame) for period, frame in frames.items()}
    distance_data = {period: pair_separations(frame) for period, frame in frames.items()}

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.15))
    ax_station, ax_distance = axes

    for period in SELECTED_PERIODS:
        ax_station.hist(
            station_data[period],
            bins=station_bin_edges,
            color=colors[period],
            alpha=0.72,
            edgecolor="#4A4A4A",
            linewidth=1.3,
            label=labels[period],
        )
        ax_distance.hist(
            distance_data[period],
            bins=distance_bin_edges,
            color=colors[period],
            alpha=0.72,
            edgecolor="#4A4A4A",
            linewidth=1.3,
            label=labels[period],
        )

    ax_station.set_xlabel("Number of usable stations per event", labelpad=11)
    ax_station.set_ylabel("Count", labelpad=10)
    ax_station.set_xlim(27.0, 558.0)
    ax_station.set_ylim(bottom=0.0)
    ax_station.set_xticks([100, 200, 300, 400, 500])

    ax_distance.set_xlabel("Station-pair separation distance (km)", labelpad=11)
    ax_distance.set_ylabel("Count", labelpad=7)
    ax_distance.set_xlim(-8.0, 640.0)
    ax_distance.set_ylim(0.0, 149000.0)
    ax_distance.set_xticks([0, 100, 200, 300, 400, 500, 600])
    ax_distance.set_yticks(np.arange(0, 140001, 20000))
    scientific_formatter = ScalarFormatter(useMathText=True)
    scientific_formatter.set_scientific(True)
    scientific_formatter.set_powerlimits((0, 0))
    scientific_formatter.set_useOffset(False)
    ax_distance.yaxis.set_major_formatter(scientific_formatter)
    ax_distance.yaxis.get_offset_text().set_fontsize(18)

    for ax in axes:
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.9, alpha=0.85)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", pad=7)
        legend = ax.legend(
            loc="upper right",
            frameon=True,
            borderpad=0.45,
            labelspacing=0.35,
            handlelength=1.45,
            handletextpad=0.55,
            borderaxespad=0.45,
        )
        legend.get_frame().set_linewidth(1.2)
        legend.get_frame().set_edgecolor("#B8B8B8")
        legend.get_frame().set_alpha(0.96)

    # Panel identifiers are intentionally separate from the plot titles.
    fig.subplots_adjust(left=0.075, right=0.995, top=0.975, bottom=0.245, wspace=0.19)
    for label, ax in zip(("(a)", "(b)"), axes):
        position = ax.get_position()
        fig.text(
            0.5 * (position.x0 + position.x1),
            0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=22,
        )

    return fig




AXIS_LABEL_SIZE = 15

TICK_LABEL_SIZE = 13

LEGEND_SIZE = 13

LINE_WIDTH = 1.7

MARKER_SIZE = 2.5

def _period_label(period: float) -> str:
    return f"T={period:g}s"

def _style_axes(axes, *, y_limits: tuple[float, float], zero_line: bool) -> None:
    for ax in axes:
        if zero_line:
            ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.set_xlabel("Distance (km)", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylim(*y_limits)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Correlation", fontsize=AXIS_LABEL_SIZE)

def _add_empirical_style_legend(fig, source_axis) -> None:
    handles, labels = source_axis.get_legend_handles_labels()
    # Keep the empirical line-plus-point legend identity even when the fitted
    # correlation curves themselves are intentionally drawn without markers.
    handles = [
        Line2D(
            [],
            [],
            color=handle.get_color(),
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE * 1.35,
        )
        for handle in handles
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=5,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=2.5,
        columnspacing=1.3,
        markerscale=1.35,
    )

def _add_panel_labels(fig, axes) -> None:
    for label, ax in zip(("(a)", "(b)"), axes):
        position = ax.get_position()
        fig.text(
            0.5 * (position.x0 + position.x1),
            0.14,
            label,
            ha="center",
            va="center",
            fontsize=15,
        )

def _direct_pc_semivariogram_correlations(payload: dict):
    state = payload["data_state"]
    pca_fit = payload["pca"]["semivariogram"]
    frame = state["df_pca"]
    period_columns = list(state["period_cols"])
    events = np.asarray(state["events_pca"])
    h_bins = np.asarray(state["h_bins"], dtype=float)
    bin_size = float(state["bin_size"])

    loadings = np.asarray(pca_fit["U"], dtype=float)
    eigenvalues = np.asarray(pca_fit["evals"], dtype=float)
    pooled_sd = np.asarray(pca_fit["std_pool"], dtype=float)
    n_components = loadings.shape[1]
    n_bins = len(h_bins)

    gamma_sum = np.zeros((n_bins, n_components, n_components), dtype=float)
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    for event_id in events:
        event = frame.loc[frame["eqid"] == event_id]
        if len(event) < 5:
            continue

        values = event[period_columns].to_numpy(dtype=float)
        scores = (values / pooled_sd[None, :]) @ loadings
        latitude = event["station_latitude"].to_numpy(dtype=float)
        longitude = event["station_longitude"].to_numpy(dtype=float)
        distance_matrix = mch._haversine_between(latitude, longitude, latitude, longitude)
        site_i, site_j = np.triu_indices(len(event), k=1)
        distance_bins = np.rint(distance_matrix[site_i, site_j] / bin_size).astype(int)
        score_differences = scores[site_i] - scores[site_j]

        for bin_number in range(1, n_bins + 1):
            use = distance_bins == bin_number
            if not bool(use.any()):
                continue
            differences = score_differences[use]
            gamma_sum[bin_number - 1] += 0.5 * (differences.T @ differences)
            pair_counts[bin_number - 1] += differences.shape[0]

    valid = pair_counts > 0
    gamma = gamma_sum[valid] / pair_counts[valid, None, None]
    covariance_zero_pc = np.diag(eigenvalues)
    period_covariance_zero = loadings @ covariance_zero_pc @ loadings.T
    period_variance_zero = np.diag(period_covariance_zero)

    rho_non_cpc = np.empty((valid.sum(), len(period_columns)), dtype=float)
    rho_cpc = np.empty_like(rho_non_cpc)
    for row, gamma_matrix in enumerate(gamma):
        gamma_matrix = 0.5 * (gamma_matrix + gamma_matrix.T)
        covariance_full = covariance_zero_pc - gamma_matrix
        covariance_cpc = covariance_zero_pc - np.diag(np.diag(gamma_matrix))
        rho_non_cpc[row] = np.diag(loadings @ covariance_full @ loadings.T) / period_variance_zero
        rho_cpc[row] = np.diag(loadings @ covariance_cpc @ loadings.T) / period_variance_zero

    distance = np.r_[0.0, h_bins[valid]]
    rho_non_cpc = np.vstack((np.ones(len(period_columns)), rho_non_cpc))
    rho_cpc = np.vstack((np.ones(len(period_columns)), rho_cpc))
    return np.asarray(state["periods"], dtype=float), distance, rho_non_cpc, rho_cpc

def build_empirical_semivariogram_figure(payload: dict, colors: np.ndarray) -> plt.Figure:
    periods, distance, rho_non_cpc, rho_cpc = _direct_pc_semivariogram_correlations(payload)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True, sharey=True)
    for index, (period, color) in enumerate(zip(periods, colors)):
        label = _period_label(period)
        axes[0].plot(
            distance,
            rho_non_cpc[:, index],
            color=color,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label=label,
        )
        axes[1].plot(
            distance,
            rho_cpc[:, index],
            color=color,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label=label,
        )

    _style_axes(axes, y_limits=(-0.23, 1.06), zero_line=True)
    _add_panel_labels(fig, axes)
    _add_empirical_style_legend(fig, axes[1])
    fig.subplots_adjust(bottom=0.29, wspace=0.05)
    return fig




DISPLAY_PERIOD_TICKS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 6.0)

METHOD_ORDER = (
    "Pairwise empirical semivariogram",
    "GH08 semivariogram",
    "PCA semivariogram",
    "Kronecker semivariogram",
    "LMC semivariogram",
)

SHORT_LABELS = {
    "Pairwise empirical semivariogram": "Pairwise empirical",
    "GH08 semivariogram": "GH08",
    "PCA semivariogram": "PCA",
    "Kronecker semivariogram": "Separable kernel",
    "LMC semivariogram": "LMC",
}

MODEL_PLOT_STYLES = {
    "Pairwise empirical semivariogram": {
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "o",
    },
    "GH08 semivariogram": {
        "color": "#1f77b4",
        "linestyle": "--",
        "marker": "s",
    },
    "PCA semivariogram": {
        "color": "#d62728",
        "linestyle": ":",
        "marker": "D",
    },
    "Kronecker semivariogram": {
        "color": "#2ca02c",
        "linestyle": "-.",
        "marker": "^",
    },
    "LMC semivariogram": {
        "color": "#8c564b",
        "linestyle": (0, (3, 1, 1, 1)),
        "marker": "P",
    },
}

PAIRWISE_METHOD = "Pairwise empirical semivariogram"

GH08_METHOD = "GH08 semivariogram"

def _same_period_style(method: str) -> dict:
    """Use one visual identity for the identical Pairwise and GH08 diagonals."""
    style_method = PAIRWISE_METHOD if method == GH08_METHOD else method
    return MODEL_PLOT_STYLES[style_method]

def _same_period_legend_label(method: str) -> str:
    """Represent the identical Pairwise and GH08 diagonals by Pairwise only."""
    return "_nolegend_" if method == GH08_METHOD else SHORT_LABELS[method]

def _period_formatter(value: float, _position: int | None = None) -> str:
    return f"{value:g}"

def load_semivariogram_curves(cache_path: Path) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    payload = mch.load_pickle_cross_platform(cache_path)
    periods = np.asarray(payload["data_state"]["periods"], dtype=float)
    cached_curves = payload["cross_period_psd"]["curves"]

    selected = {
        method: (
            np.asarray(cached_curves[method][0], dtype=float),
            np.asarray(cached_curves[method][1], dtype=float),
        )
        for method in METHOD_ORDER
        if method in cached_curves and "semivariogram" in method.lower()
    }
    missing = [method for method in METHOD_ORDER if method not in selected]
    if missing:
        raise KeyError(f"Expected semivariogram curves are missing from the cache: {missing}")

    q = len(periods)
    for method, (distance_grid, curves) in selected.items():
        if curves.shape != (q, q, len(distance_grid)):
            raise ValueError(
                f"Unexpected curve shape for {method}: {curves.shape}; "
                f"expected {(q, q, len(distance_grid))}"
            )
        if not np.isfinite(curves).all():
            raise ValueError(f"Non-finite cached correlations found for {method}")
        symmetry_error = float(np.max(np.abs(curves - curves.swapaxes(0, 1))))
        if symmetry_error > 1.0e-8:
            raise ValueError(f"Cached correlation tensor is not symmetric for {method}: {symmetry_error:g}")
    return periods, selected

def interpolate_correlation_matrix(
    distance_grid: np.ndarray,
    curves: np.ndarray,
    distance_km: float,
) -> np.ndarray:
    if distance_km < float(distance_grid[0]) or distance_km > float(distance_grid[-1]):
        raise ValueError(
            f"Requested distance {distance_km:g} km is outside cached range "
            f"[{distance_grid[0]:g}, {distance_grid[-1]:g}] km"
        )
    q = curves.shape[0]
    result = np.empty((q, q), dtype=float)
    for i in range(q):
        for j in range(q):
            result[i, j] = np.interp(distance_km, distance_grid, curves[i, j])
    return 0.5 * (result + result.T)

def build_fixed_distance_matrices(
    curve_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    distances_km: tuple[float, ...],
) -> dict[str, dict[float, np.ndarray]]:
    return {
        method: {
            distance_km: interpolate_correlation_matrix(distance_grid, curves, distance_km)
            for distance_km in distances_km
        }
        for method, (distance_grid, curves) in curve_sets.items()
    }

def build_normalized_decay_matrices(
    curve_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    distances_km: tuple[float, ...],
    minimum_absolute_rho0: float = 1.0e-8,
) -> dict[str, dict[float, np.ndarray]]:
    """Return rhobar_rs(h) = rho_rs(h) / rho_rs(0) for every method and distance."""
    result: dict[str, dict[float, np.ndarray]] = {}
    for method, (distance_grid, curves) in curve_sets.items():
        rho0 = interpolate_correlation_matrix(distance_grid, curves, 0.0)
        minimum = float(np.min(np.abs(rho0)))
        if minimum <= minimum_absolute_rho0:
            raise ValueError(
                f"Cannot safely normalize {method}: minimum |rho_ij(0)|={minimum:.3e} "
                f"is not above {minimum_absolute_rho0:.3e}"
            )
        result[method] = {
            distance_km: interpolate_correlation_matrix(distance_grid, curves, distance_km) / rho0
            for distance_km in distances_km
        }
    return result

def _comparison_limits(
    matrices: dict[str, dict[float, np.ndarray]],
    distance_km: float,
) -> tuple[float, float]:
    all_values = np.concatenate([method_values[distance_km].ravel() for method_values in matrices.values()])
    lower = min(0.0, np.floor(float(np.min(all_values)) * 20.0) / 20.0)
    upper = max(0.05, np.ceil(float(np.max(all_values)) * 20.0) / 20.0)
    return lower, upper

def plot_combined_distance_figure(
    periods: np.ndarray,
    correlation_matrices: dict[str, dict[float, np.ndarray]],
    normalized_matrices: dict[str, dict[float, np.ndarray]],
    distance_km: float,
    levels: np.ndarray,
) -> plt.Figure:
    """Plot same-period curves first, followed by five rhobar_rs contour panels."""
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.5))
    axes_flat = axes.ravel()
    line_ax = axes_flat[0]
    contour_axes = axes_flat[1:]

    maximum = max(
        float(np.max(np.diag(correlation_matrices[method][distance_km])))
        for method in METHOD_ORDER
    )
    for method in METHOD_ORDER:
        line_ax.plot(
            periods,
            np.diag(correlation_matrices[method][distance_km]),
            markersize=7.5,
            linewidth=2.4,
            label=_same_period_legend_label(method),
            **_same_period_style(method),
        )
    line_ax.set_xscale("log")
    line_ax.set_xticks(DISPLAY_PERIOD_TICKS)
    line_ax.xaxis.set_major_formatter(FuncFormatter(_period_formatter))
    line_ax.set_xlim(periods[0], periods[-1])
    line_ax.set_ylim(0.0, min(1.0, max(0.05, maximum * 1.12)))
    line_ax.grid(alpha=0.22, linewidth=0.8)
    line_ax.set_title("(a) Same-period correlation", fontsize=23, pad=10)
    line_ax.set_xlabel("Period T (s)", fontsize=22, labelpad=8)
    line_ax.set_ylabel("Correlation", fontsize=22, labelpad=10)
    line_ax.tick_params(axis="both", which="major", labelsize=14, width=1.1, length=5)
    handles, labels = line_ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=22,
        ncol=len(labels),
        loc="upper center",
        bbox_to_anchor=(0.475, 0.972),
        borderaxespad=0.2,
        columnspacing=1.25,
        handlelength=2.6,
        handletextpad=0.6,
        markerscale=1.25,
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(3.0)

    contour = None
    for panel_letter, ax, method in zip("bcdef", contour_axes, METHOD_ORDER):
        matrix = normalized_matrices[method][distance_km]
        contour = ax.contourf(
            periods,
            periods,
            matrix,
            levels=levels,
            cmap="viridis",
            vmin=float(levels[0]),
            vmax=float(levels[-1]),
            extend="neither",
        )
        ax.contour(
            periods,
            periods,
            matrix,
            levels=levels[::2],
            colors="white",
            linewidths=0.45,
            alpha=0.62,
        )
        ax.plot(periods, periods, color="white", linewidth=1.0, alpha=0.88)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(DISPLAY_PERIOD_TICKS)
        ax.set_yticks(DISPLAY_PERIOD_TICKS)
        ax.xaxis.set_major_formatter(FuncFormatter(_period_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(_period_formatter))
        ax.set_xlim(periods[0], periods[-1])
        ax.set_ylim(periods[0], periods[-1])
        ax.set_title(
            f"({panel_letter}) {SHORT_LABELS[method]}",
            fontsize=23,
            pad=10,
        )
        ax.set_xlabel(r"Period $T_r$ (s)", fontsize=22, labelpad=8)
        ax.set_ylabel(r"Period $T_s$ (s)", fontsize=22, labelpad=9)
        ax.tick_params(axis="both", which="major", labelsize=14, width=1.0, length=4)

    if contour is None:
        raise RuntimeError("No normalized contour panels were created")
    fig.subplots_adjust(left=0.075, right=0.875, bottom=0.085, top=0.865, wspace=0.31, hspace=0.37)
    colorbar_axis = fig.add_axes([0.91, 0.12, 0.015, 0.72])
    colorbar = fig.colorbar(contour, cax=colorbar_axis, ticks=np.linspace(levels[0], levels[-1], 6))
    colorbar.set_label(
        "Normalized cross-period correlation",
        fontsize=22,
        labelpad=13,
    )
    colorbar.ax.tick_params(labelsize=14, width=1.0, length=4)
    return fig




BASE_METHODS = (
    "Pairwise empirical semivariogram",
    "GH08 semivariogram",
    "Kronecker semivariogram",
    "PCA semivariogram",
    "LMC semivariogram",
)

PLOTTED_BASE_METHODS = tuple(
    method for method in BASE_METHODS if method != "GH08 semivariogram"
)

NEW_METHODS = (
    "Multivariate Matern semivariogram",
    "SBSS semivariogram",
    "IOX full semivariogram",
)

LABELS = {
    BASE_METHODS[0]: "Pairwise empirical",
    BASE_METHODS[1]: "GH08",
    BASE_METHODS[2]: "Separable kernel",
    BASE_METHODS[3]: "PCA",
    BASE_METHODS[4]: "LMC",
    NEW_METHODS[0]: "Matérn",
    NEW_METHODS[1]: "SBSS",
    NEW_METHODS[2]: "IOX",
}

PERIOD_TICKS = (0.01, 0.03, 0.1, 0.3, 1, 3, 6)

def plot_comparison(project_root: Path, periods, matrices):
    from scripts.analysis import styles as mcr

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 11.7))
    fig.subplots_adjust(left=0.075, right=0.86, bottom=0.085, top=0.83, wspace=0.32, hspace=0.35)
    line_axis = axes.flat[0]
    styles = {}
    for method in (*PLOTTED_BASE_METHODS, *NEW_METHODS):
        canonical = "Separable kernel semivariogram" if method == "Kronecker semivariogram" else method
        highlighted = method in NEW_METHODS
        style = dict(mcr.MODEL_LINE_STYLES[canonical])
        style.update({
            "color": mcr.HIGHLIGHT_MODEL_COLORS.get(canonical, mcr.BASELINE_MODEL_COLOR),
            "linewidth": 2.7 if highlighted else 1.85,
            "markersize": 7.8 if highlighted else 6.3,
            "markerfacecolor": mcr.HIGHLIGHT_MODEL_COLORS[canonical] if highlighted else "none",
            "markeredgewidth": 1.1,
            "zorder": 4 if highlighted else 2,
        })
        # These two diagonal curves are identical. Alternate their hollow
        # marker locations without offsetting or changing any data values.
        if method == BASE_METHODS[0]:
            style["markevery"] = list(range(0, len(periods), 2))
        elif method == BASE_METHODS[1]:
            style["markevery"] = list(range(1, len(periods), 2))
        line_axis.plot(periods, np.diag(matrices[method]["rho_h"]), label=LABELS[method], **style)
        styles[method] = style
    formatter = FuncFormatter(lambda value, _pos: f"{value:g}")
    line_axis.set_xscale("log")
    line_axis.set_xlim(periods[0], periods[-1])
    line_axis.set_xticks(PERIOD_TICKS)
    line_axis.xaxis.set_major_formatter(formatter)
    ymax = max(np.diag(values["rho_h"]).max() for values in matrices.values())
    line_axis.set_ylim(0, min(1.0, np.ceil(ymax * 1.12 / 0.05) * 0.05))
    line_axis.set_title("(a) Same-period correlation", fontsize=22, pad=10)
    line_axis.set_xlabel("Period T (s)", fontsize=22, labelpad=7)
    line_axis.set_ylabel("Correlation", fontsize=22, labelpad=8)
    line_axis.tick_params(axis="both", labelsize=14, width=1.1, length=5)
    line_axis.grid(alpha=0.22, linewidth=0.8)

    all_values = np.concatenate([matrices[method]["rhobar"].ravel() for method in NEW_METHODS])
    # One shared scale, expanded upward when necessary rather than clipping
    # SBSS values above the old figure's 0.75 maximum.
    lower = min(0.0, np.floor(all_values.min() / 0.1) * 0.1)
    upper = max(0.05, np.ceil(all_values.max() / 0.1) * 0.1)
    levels = np.linspace(lower, upper, int(round((upper - lower) / 0.05)) + 1)
    for letter, axis, method in zip("bcd", list(axes.flat)[1:], NEW_METHODS):
        matrix = matrices[method]["rhobar"]
        assert matrix.min() >= levels[0] and matrix.max() <= levels[-1]
        contour = axis.contourf(periods, periods, matrix, levels=levels, cmap="viridis", extend="neither")
        interior_levels = levels[::2]
        interior_levels = interior_levels[(interior_levels > matrix.min()) & (interior_levels < matrix.max())]
        if len(interior_levels):
            axis.contour(periods, periods, matrix, levels=interior_levels, colors="white", linewidths=0.45, alpha=0.62)
        axis.plot(periods, periods, color="white", linewidth=1.0, alpha=0.88)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(periods[0], periods[-1])
        axis.set_ylim(periods[0], periods[-1])
        axis.set_xticks(PERIOD_TICKS)
        axis.set_yticks(PERIOD_TICKS)
        axis.xaxis.set_major_formatter(formatter)
        axis.yaxis.set_major_formatter(formatter)
        axis.set_title(f"({letter}) {LABELS[method]}", fontsize=22, pad=10)
        axis.set_xlabel(r"Period $T_r$ (s)", fontsize=22, labelpad=7)
        axis.set_ylabel(r"Period $T_s$ (s)", fontsize=22, labelpad=8)
        axis.tick_params(axis="both", labelsize=14, length=5, width=1.1)

    handles, labels = line_axis.get_legend_handles_labels()
    # Explicit row-major legend order across four columns.
    order = [0, 4, 1, 5, 2, 6, 3]
    legend = fig.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        ncol=4, loc="upper center", bbox_to_anchor=(0.48, 0.985),
        fontsize=18, frameon=False, handlelength=2.5, columnspacing=1.35,
        handletextpad=0.6, markerscale=1.2, labelspacing=0.55,
    )
    color_axis = fig.add_axes([0.90, 0.12, 0.018, 0.66])
    colorbar = fig.colorbar(contour, cax=color_axis, ticks=np.linspace(lower, upper, 7))
    colorbar.set_label("Normalized cross-period correlation", fontsize=21, labelpad=12)
    colorbar.ax.tick_params(labelsize=14, length=5, width=1.0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    for axis in axes.flat:
        assert not legend_box.overlaps(axis.get_tightbbox(renderer)), "Legend overlaps a panel"
    return fig, levels, styles




PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIGURE_DIR = PROJECT_ROOT / "results/figures"

def _style_axis(ax):
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

def _plot_data_requirement(
    data,
    methods,
    plot_colors,
    metric,
    ylabel,
    output_name,
    *,
    show_legend,
):
    fig, ax = plt.subplots(figsize=mcr.DATA_REQUIREMENT_FIGSIZE)
    subplot_adjust = dict(mcr.DATA_REQUIREMENT_SUBPLOT_ADJUST)
    legend_bottom = None
    if show_legend:
        legend_rows = int(np.ceil(len(methods) / mcr.DATA_REQUIREMENT_LEGEND_NCOL))
        # Pin the legend just above the axes instead of reserving a fixed,
        # overly tall legend band.  Five-row all-model legends need a little
        # more height than the four-row selected-model legend.
        legend_bottom = 0.735 if legend_rows >= 5 else 0.765
        subplot_adjust["top"] = legend_bottom - 0.018
    fig.subplots_adjust(**subplot_adjust)
    for method in methods:
        values = data.loc[data["display_name"].eq(method)].sort_values("n_events")
        if values.empty:
            continue
        line_kwargs = mcr.method_line_kwargs(method)
        if "MLE" in method:
            line_kwargs["linestyle"] = "-"
        elif "semivariogram" in method.lower():
            line_kwargs["linestyle"] = "--"
        if plot_colors is not None:
            line_kwargs.update({
                "color": plot_colors[method],
                "markersize": 6.5,
                "linewidth": 2.2,
                "markerfacecolor": plot_colors[method],
            })
        ax.plot(
            values["n_events"],
            values[metric],
            label=method,
            **line_kwargs,
        )
    ax.set_xlabel(
        "Number of events",
        fontsize=mcr.DATA_REQUIREMENT_AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=mcr.DATA_REQUIREMENT_AXIS_LABEL_FONTSIZE,
    )
    ax.set_xticks(sorted(data["n_events"].unique()))
    ax.tick_params(
        axis="both",
        labelsize=mcr.DATA_REQUIREMENT_TICK_FONTSIZE,
        length=6,
        width=1.1,
    )
    _style_axis(ax)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_bottom),
            ncol=mcr.DATA_REQUIREMENT_LEGEND_NCOL,
            frameon=False,
            fontsize=mcr.DATA_REQUIREMENT_LEGEND_FONTSIZE,
            handlelength=2.7,
            markerscale=1.15,
            columnspacing=1.35,
            labelspacing=0.55,
        )
    path = FIGURE_DIR / output_name
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path

def _plot_sampling_sweep(summary, x_col, x_label, log_x, output_name):
    methods = [
        method
        for method in mcr.ALL_MODELS
        if method in set(summary["method"])
    ]
    fig, axes = mcr._plot_residual_sweep(  # pylint: disable=protected-access
        summary,
        x_col=x_col,
        x_label=x_label,
        methods=methods,
        log_x=log_x,
    )
    fig.set_size_inches(15.0, 5.0)
    for panel_index, axis in enumerate(axes):
        panel_label = chr(ord("a") + panel_index)
        axis.set_title("")
        axis.set_xlabel(
            f"{x_label}\n({panel_label})",
            fontsize=17 if x_label == "Number of stations" else 18,
            labelpad=7,
            linespacing=1.3,
        )
        axis.tick_params(
            axis="both",
            which="major",
            labelsize=16,
            length=6,
            width=1.1,
        )
        axis.tick_params(axis="both", which="minor", length=3.5, width=0.9)
        for line in axis.lines:
            line.set_linewidth(2.5)
            line.set_markersize(8.0)
        if x_label == "Number of stations":
            for tick_label in axis.get_xticklabels():
                tick_label.set_rotation(32)
                tick_label.set_horizontalalignment("right")
                tick_label.set_rotation_mode("anchor")

    axes[0].set_ylabel("")
    fig.supylabel("Seconds", x=0.043, fontsize=19)
    for legend in list(fig.legends):
        legend.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=18,
        handlelength=2.4,
        markerscale=1.25,
        columnspacing=1.5,
        labelspacing=0.45,
    )
    layout_w_pad = 1.8 if x_label == "Number of stations" else 1.0
    fig.tight_layout(rect=(0.03, 0.01, 1.0, 0.79), w_pad=layout_w_pad)
    if x_col == "station_count":
        missing = summary.loc[summary.method.isin(("IOX full semivariogram", "Multivariate Matern semivariogram")) & summary.median_total_seconds.isna()]
        if not missing.empty:
            stations = ", ".join(str(int(v)) for v in sorted(missing.station_count.unique()))
            axes[0].text(0.98, 0.035,
                f"IOX and Multivariate Matérn: {stations} stations\nnot measured (insufficient physical memory)",
                transform=axes[0].transAxes, ha="right", va="bottom", fontsize=10.5,
                color="#333333", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2))
    path = FIGURE_DIR / output_name
    fig.savefig(
        path,
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(fig)
    return path




METHODS = {
    "Separable kernel semivariogram": {
        "label": "Separable",
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "^",
    },
    "PCA semivariogram": {
        "label": "PCA",
        "color": "#d62728",
        "linestyle": "--",
        "marker": "D",
    },
    "LMC semivariogram": {
        "label": "LMC",
        "color": "#8c564b",
        "linestyle": "-.",
        "marker": "P",
    },
    "LMC conventional method": {
        "label": "LMC conventional method",
        "color": "#8c564b",
        "linestyle": ":",
        "marker": "P",
    },
}

METRICS = (
    ("median_total_seconds", "(a) Total time"),
    ("median_first_realization_seconds", "(b) First realization"),
    ("median_sampling_seconds_per_realization", "(c) Sampling per realization"),
)

def plot_sweep(
    summary: pd.DataFrame,
    *,
    x_column: str,
    x_label: str,
    output_path: Path,
    log_x_base: int | None,
) -> None:
    selected = summary.loc[summary["method"].isin(METHODS)].copy()
    selected[x_column] = pd.to_numeric(selected[x_column], errors="coerce")
    for metric, _ in METRICS:
        selected[metric] = pd.to_numeric(selected[metric], errors="coerce")

    positive_values = np.concatenate(
        [
            selected[metric].to_numpy(dtype=float)
            for metric, _ in METRICS
        ]
    )
    positive_values = positive_values[
        np.isfinite(positive_values) & (positive_values > 0.0)
    ]
    if positive_values.size == 0:
        raise ValueError(f"No positive timing values available for {output_path.name}")
    y_min = 10.0 ** np.floor(np.log10(positive_values.min()))
    y_max = 10.0 ** np.ceil(np.log10(positive_values.max()))

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 5.0),
        sharex=True,
        sharey=True,
    )
    x_ticks = sorted(selected[x_column].dropna().unique())

    for panel_index, (axis, (metric, _title)) in enumerate(zip(axes, METRICS)):
        for method, style in METHODS.items():
            values = selected.loc[selected["method"].eq(method)].sort_values(x_column)
            if values.empty:
                continue
            axis.plot(
                values[x_column],
                values[metric],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.5,
                markersize=8.0,
            )
        panel_label = chr(ord("a") + panel_index)
        axis.set_xlabel(
            f"{x_label}\n({panel_label})",
            fontsize=17 if x_label == "Number of stations" else 18,
            labelpad=7,
            linespacing=1.3,
        )
        axis.set_yscale("log")
        axis.set_ylim(y_min, y_max)
        if log_x_base is not None:
            axis.set_xscale("log", base=log_x_base)
        axis.set_xticks(x_ticks)
        axis.set_xticklabels([str(int(value)) for value in x_ticks])
        if x_label == "Number of stations":
            for tick_label in axis.get_xticklabels():
                tick_label.set_rotation(32)
                tick_label.set_horizontalalignment("right")
                tick_label.set_rotation_mode("anchor")
        axis.grid(which="major", alpha=0.25)
        axis.grid(which="minor", alpha=0.08)
        axis.tick_params(
            axis="both",
            which="major",
            labelsize=16,
            length=6,
            width=1.1,
        )
        axis.tick_params(axis="both", which="minor", length=3.5, width=0.9)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].yaxis.set_major_locator(LogLocator(base=10))
    axes[0].yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
    )
    for axis in axes[1:]:
        axis.tick_params(axis="y", which="both", left=False, labelleft=False)

    figure.supylabel("Seconds", x=0.043, fontsize=19)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles),
        frameon=False,
        fontsize=18,
        handlelength=2.4,
        markerscale=1.25,
        columnspacing=1.25,
    )
    layout_w_pad = 1.8 if x_label == "Number of stations" else 1.0
    figure.tight_layout(rect=(0.03, 0.01, 1.0, 0.91), w_pad=layout_w_pad)
    # Leave unmeasured dense sizes blank; never extrapolate benchmark data.
    conventional = selected.loc[selected["method"].eq("LMC conventional method")]
    if x_column == "station_count" and conventional[METRICS[0][0]].isna().any():
        missing = conventional.loc[conventional[METRICS[0][0]].isna(), x_column]
        missing_text = ", ".join(str(int(value)) for value in missing)
        axes[0].text(
            0.97, 0.04,
            f"LMC conventional: {missing_text} stations\nnot measured (insufficient physical memory)",
            transform=axes[0].transAxes, ha="right", va="bottom",
            fontsize=10.5, color="#a32626",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2),
        )
    figure.canvas.draw()
    legend_box = legend.get_window_extent(figure.canvas.get_renderer())
    if legend_box.x0 < 0 or legend_box.x1 > figure.bbox.width:
        raise ValueError("Legend exceeds the figure width")
    figure.savefig(
        output_path,
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(figure)




def plot_fitting_time(summary: pd.DataFrame, output_path: Path) -> None:
    excluded_methods = {
        "IOX full semivariogram",
        "SBSS semivariogram",
        "Multivariate Matern semivariogram",
    }
    plotted = summary[
        summary["fit_time_median_seconds"].notna()
        & ~summary["method"].isin(excluded_methods)
    ].copy()
    plotted = plotted.sort_values("fit_time_median_seconds", ascending=True)
    display_labels = {
        "Pairwise empirical semivariogram": "Pairwise empirical",
        "GH08 semivariogram": "GH08",
        "PCA semivariogram": "PCA semivariogram",
        "Kronecker semivariogram": "Separable kernel",
        "LMC semivariogram": "LMC semivariogram",
        "PCA MLE": "PCA MLE",
        "LMC MLE": "LMC MLE",
    }
    plotted["plot_label"] = plotted["method"].map(display_labels).fillna(
        plotted["method"].str.replace(" semivariogram", "", regex=False)
    )
    # Only PCA and LMC have an explicit MLE-versus-semivariogram comparison.
    # Every other method uses the default semivariogram color.
    colors = np.where(
        plotted["method"].isin({"PCA MLE", "LMC MLE"}),
        "#E45756",
        "#4C78A8",
    )
    fig, ax = plt.subplots(figsize=(9.8, 5.1))
    ax.barh(
        plotted["plot_label"],
        plotted["fit_time_median_seconds"],
        color=colors,
        alpha=0.9,
        height=0.78,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Time (seconds)", fontsize=20, labelpad=7)
    ax.tick_params(axis="both", which="major", labelsize=18, length=6, width=1.1)
    ax.tick_params(axis="x", which="minor", length=3.5, width=0.9)
    ax.grid(axis="x", which="both", alpha=0.22)
    xmin = max(float(plotted["fit_time_median_seconds"].min()) * 0.55, 1.0e-4)
    xmax = float(plotted["fit_time_median_seconds"].max()) * 2.6
    ax.set_xlim(xmin, xmax)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#4C78A8", label="Semivariogram fit"),
            Patch(facecolor="#E45756", label="Maximum likelihood fit"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=17,
        handlelength=2.0,
        handleheight=1.15,
        labelspacing=0.55,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.29, right=0.985, bottom=0.18, top=0.97)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
