"""Recreate all thirteen article figures from the included fits and measurements."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from scripts.analysis import core as mch
from scripts.analysis import styles as style
from scripts.analysis import plotting as p


def save(fig, path, dpi=240, pad=0.04):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print(path.name, flush=True)


def data_requirements(data, methods, selected):
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 12.4), sharex=True)
    colors = {m: ("tab:red" if "PCA" in m else "tab:brown" if "LMC" in m
                  else "tab:green" if "Separable" in m else "tab:orange" if "GH08" in m
                  else "tab:blue") for m in methods}
    for ax, metric, label, letter in zip(axes, ("rmse_median", "rmse_std"),
        ("Median RMSE vs full-data reference", "RMSE standard deviation"), "ab"):
        for method in methods:
            values = data.loc[data.display_name.eq(method)].sort_values("n_events")
            kwargs = style.method_line_kwargs(method)
            kwargs["linestyle"] = "--" if "MLE" in method else "-"
            if selected:
                kwargs.update(color=colors[method], markerfacecolor=colors[method], linewidth=2.2, markersize=6.5)
            ax.plot(values.n_events, values[metric], label=method.replace("Pairwise empirical", "Pairwise fit"), **kwargs)
        ax.set_ylabel(label, fontsize=20)
        ax.set_xlabel(f"Number of events\n({letter})", fontsize=20)
        ax.set_xticks(sorted(data.n_events.unique()))
        ax.tick_params(labelsize=17, labelbottom=True)
        p._style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    by_name=dict(zip(methods,zip(handles,labels)))
    order=([methods[i] for i in [0,2,3,5,1,4,6]] if selected else
           [methods[i] for i in [0,2,3,5,8,1,7,4,6,9]])
    if selected:
        from matplotlib.lines import Line2D
        entries=[by_name[m] for m in order]
        entries.insert(5,(Line2D([],[],linestyle="none"),""))
    else: entries=[by_name[m] for m in order]
    handles,labels=zip(*entries)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=16,
        bbox_to_anchor=(0.54, 1.0), columnspacing=1.1)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.09, top=0.79, hspace=0.35)
    return fig


def move_panel_labels(fig,n):
    for ax in fig.axes[:n]:
        title=ax.get_title()
        ax.set_title("")
        ax.text(.5,-.215,title,transform=ax.transAxes,ha="center",va="top",fontsize=21,clip_on=False)
    fig.subplots_adjust(bottom=.12,hspace=.37)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "results/models.pkl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/figures")
    parser.add_argument("--benchmarks-dir", type=Path, default=ROOT / "results/analysis/benchmarks",
                        help="Measurement CSV collection in the same schema as results/analysis/benchmarks.")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    p.FIGURE_DIR = out
    reference = args.benchmarks_dir
    payload = mch.load_pickle_cross_platform(args.cache)
    periods = np.asarray(payload["data_state"]["periods"])
    data = mch.comparison_data_from_state(payload["data_state"])
    frames = {period: data.period_dfs[np.flatnonzero(np.isclose(periods,period))[0]] for period in (0.1,6.)}
    save(p.make_data_figure(frames), out / "fig01_data_characteristics.png", dpi=300, pad=0.035)
    plt.rcdefaults()

    periods, curves = p.load_semivariogram_curves(args.cache)
    matrices = p.build_fixed_distance_matrices(curves, (5.,))
    normalized = p.build_normalized_decay_matrices(curves, (5.,))
    levels = np.linspace(*p._comparison_limits(normalized, 5.), 16)
    fig=p.plot_combined_distance_figure(periods, matrices, normalized, 5., levels)
    move_panel_labels(fig,6)
    save(fig, out / "fig02_cross_period_flexibility.png")
    colors = plt.cm.Blues(np.linspace(0.05,0.95,len(periods)))
    save(p.build_empirical_semivariogram_figure(payload,colors), out / "fig03_pca_cpc.png")

    requirements = pd.read_csv(reference / "data_requirement_summary.csv")
    requirements["display_name"] = requirements.display_name.replace({"Kronecker semivariogram":"Separable kernel semivariogram"})
    save(data_requirements(requirements,list(style.ALL_MODELS[:7]),True), out / "fig04_data_requirements.png")
    p.plot_fitting_time(pd.read_csv(reference / "fit_time_summary.csv"), out / "fig05_fitting_time.png")
    print("fig05_fitting_time.png", flush=True)
    for number,sweep,column,label,base in ((6,"station","station_count","stations",2),
        (7,"period","period_count","periods",None),(8,"realization","n_realizations","realizations",10)):
        p.plot_sweep(pd.read_csv(reference/f"main_{sweep}_summary.csv"), x_column=column,
            x_label=f"Number of {label}", output_path=out/f"fig{number:02d}_sampling_{label}.png", log_x_base=base)
        print(f"fig{number:02d}_sampling_{label}.png", flush=True)

    extra = {}
    for method in (*p.BASE_METHODS,*p.NEW_METHODS):
        h,rho=payload["cross_period_psd"]["curves"][method]
        rho0=p.interpolate_correlation_matrix(np.asarray(h),np.asarray(rho),0.)
        rhoh=p.interpolate_correlation_matrix(np.asarray(h),np.asarray(rho),5.)
        extra[method] = {"rho_0":rho0,"rho_h":rhoh,"rhobar":rhoh/rho0}
    fig,_,_ = p.plot_comparison(ROOT,periods,extra)
    move_panel_labels(fig,4)
    save(fig,out/"fig09_additional_models.png", dpi=300, pad=0.05)
    save(data_requirements(requirements,list(style.ALL_MODELS),False),out/"fig10_all_data_requirements.png")
    for number,sweep,column,label,log in ((11,"station","station_count","stations",True),
        (12,"period","period_count","periods",False),(13,"realization","n_realizations","realizations",True)):
        table=pd.read_csv(reference/f"supplement_{sweep}_summary.csv")
        # Preserve the original plotting definition: the middle panel uses setup seconds.
        p._plot_sampling_sweep(table,column,f"Number of {label}",log,f"fig{number:02d}_all_sampling_{label}.png")
        print(f"fig{number:02d}_all_sampling_{label}.png", flush=True)


if __name__ == "__main__":
    main()
