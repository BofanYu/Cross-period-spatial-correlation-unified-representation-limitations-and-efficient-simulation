"""Fitting timers and the supplementary timing plot."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.common import styles


def supplementary_fit_figure(summary):
    data = summary.loc[summary.method.ne("LMC MLE")].sort_values("fit_time_median_seconds")
    names = data.method.replace({"Pairwise empirical semivariogram": "Pairwise fit",
        "Kronecker semivariogram": "Separable kernel", "IOX full semivariogram": "IOX full",
        "SBSS semivariogram": "SBSS", "Multivariate Matern semivariogram": "Multivariate Matern"})
    colors = [styles.HIGHLIGHT_MODEL_COLORS.get(name, "#666666") for name in data.method]
    fig, ax = plt.subplots(figsize=(9.8, 5.1))
    ax.barh(names, data.fit_time_median_seconds, color=colors, height=.78)
    ax.set_xscale("log")
    ax.set_xlabel("Time (seconds)", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.grid(axis="x", which="both", alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.34, right=.985, bottom=.18, top=.97)
    return fig
