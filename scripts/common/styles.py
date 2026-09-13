"""Publication plot styles and shared timing axes."""
from __future__ import annotations
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.analysis.computation import samplers as ct


ALL_MODELS = (
    "Pairwise empirical semivariogram",
    "GH08 semivariogram",
    "Separable kernel semivariogram",
    "PCA semivariogram",
    "PCA MLE",
    "LMC semivariogram",
    "LMC MLE",
    "IOX full semivariogram",
    "SBSS semivariogram",
    "Multivariate Matern semivariogram",
)

HIGHLIGHT_MODEL_COLORS = {
    "IOX full semivariogram": "#E69F00",
    "SBSS semivariogram": "#009E73",
    "Multivariate Matern semivariogram": "#0072B2",
}

BASELINE_MODEL_COLOR = "#666666"

DATA_REQUIREMENT_FIGSIZE = (12.8, 6.6)

DATA_REQUIREMENT_AXIS_LABEL_FONTSIZE = 22

DATA_REQUIREMENT_TICK_FONTSIZE = 18

DATA_REQUIREMENT_LEGEND_FONTSIZE = 18

DATA_REQUIREMENT_LEGEND_NCOL = 2

DATA_REQUIREMENT_SUBPLOT_ADJUST = {
    "left": 0.13,
    "right": 0.985,
    "bottom": 0.16,
    "top": 0.97,
}

MODEL_LINE_STYLES = {
    "Pairwise empirical semivariogram": {"linestyle": "-", "marker": "o"},
    "GH08 semivariogram": {"linestyle": "--", "marker": "s"},
    "Separable kernel semivariogram": {"linestyle": "-.", "marker": "^"},
    "PCA semivariogram": {"linestyle": ":", "marker": "D"},
    "PCA MLE": {"linestyle": (0, (5, 1)), "marker": "v"},
    "LMC semivariogram": {"linestyle": (0, (3, 1, 1, 1)), "marker": "P"},
    "LMC MLE": {"linestyle": (0, (1, 1)), "marker": "X"},
    "IOX full semivariogram": {"linestyle": "-", "marker": "o"},
    "SBSS semivariogram": {"linestyle": "--", "marker": "s"},
    "Multivariate Matern semivariogram": {"linestyle": "-.", "marker": "^"},
}

METHOD_ALIASES = {
    # Compatibility aliases are accepted at cache/module boundaries only.
    "Kronecker": "Separable kernel semivariogram",
    "Kronecker semivariogram": "Separable kernel semivariogram",
    "Separable kernel": "Separable kernel semivariogram",
    "PCA": "PCA semivariogram",
    "LMC": "LMC semivariogram",
    "LMC block MLE": "LMC MLE",
}

def canonical_method_name(name: str) -> str:
    return METHOD_ALIASES.get(str(name), str(name))

def _style_axis(ax) -> None:
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

def method_line_kwargs(method: str) -> dict:
    """Return the stable visual identity used for a model's line plots."""
    method = canonical_method_name(method)
    highlighted = method in HIGHLIGHT_MODEL_COLORS
    style = MODEL_LINE_STYLES.get(method, {"linestyle": "-", "marker": "o"})
    return {
        **style,
        "markersize": 5.4 if highlighted else 4.8,
        "linewidth": 2.6 if highlighted else 1.55,
        "alpha": 1.0 if highlighted else 0.9,
        "zorder": 4 if highlighted else 2,
        "color": HIGHLIGHT_MODEL_COLORS.get(method, BASELINE_MODEL_COLOR),
        "markerfacecolor": None if highlighted else "none",
        "markeredgewidth": 1.0,
    }

def _plot_residual_sweep(
    summary: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    methods: Sequence[str],
    log_x: bool,
):
    metrics = (
        ("median_total_seconds", "Total time"),
        ("median_setup_seconds", "Setup"),
        ("median_sampling_seconds_per_realization", "Sampling per realization"),
    )
    x_values = sorted(summary[x_col].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        for method in methods:
            values = summary.loc[summary["method"].eq(method)].sort_values(x_col)
            plotted_metric = pd.to_numeric(values[metric], errors="coerce").where(
                pd.to_numeric(
                    values["median_total_seconds"], errors="coerce"
                ).notna()
            )
            ax.plot(
                values[x_col],
                plotted_metric,
                label=method,
                **method_line_kwargs(method),
            )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        if log_x:
            ax.set_xscale("log", base=2 if x_col == "station_count" else 10)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(value) for value in x_values])
        _style_axis(ax)
    axes[0].set_ylabel("Seconds")
    shared_values = pd.concat(
        [
            pd.to_numeric(summary[metric], errors="coerce").where(
                pd.to_numeric(
                    summary["median_total_seconds"], errors="coerce"
                ).notna()
            )
            for metric, _ in metrics
        ],
        ignore_index=True,
    )
    ct.format_log_time_axis(axes[0], shared_values)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    return fig, axes
