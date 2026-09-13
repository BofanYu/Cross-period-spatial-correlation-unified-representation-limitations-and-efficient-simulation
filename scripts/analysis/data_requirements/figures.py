import matplotlib.pyplot as plt
from scripts.common import plotting as p
from scripts.common import styles as style

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
