import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from shared.plot_style import (
    FS_LEGEND, FS_SUPTITLE, FS_TITLE,
    apply_thesis_style, grid_size, save_fig,
)

# Distinguish series by shape and dash pattern, not colour alone: when several
# models sit at nearly the same value their lines overlap, and identical round
# markers become impossible to tell apart.
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "<", ">", "p"]
LINESTYLES = ["-", "--", "-.", ":"]


def _series_style(i):
    return MARKERS[i % len(MARKERS)], LINESTYLES[i % len(LINESTYLES)]


METRIC_TITLES = {
    "final_accuracy":              "Final Accuracy",
    "final_backward_transfer":     "Final BWT",
    "plasticity":                  "Plasticity",
    "stability":                   "Stability",
}

_ZERO_LINE_METRICS = ("final_backward_transfer", "forward_transfer")


def mean_std(entry, key):
    vals = [pc["summary"].get(key) for pc in entry["per_config"].values()]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def save_epoch_analysis_plots(results_by_epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    apply_thesis_style()
    epochs_sorted = sorted(results_by_epoch)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    datasets = sorted({r["dataset"] for r in results_by_epoch[epochs_sorted[0]]})
    for ds in datasets:
        # Gather each model's result for this dataset, keyed by epoch budget.
        per_model = {}
        for E in epochs_sorted:
            for r in results_by_epoch[E]:
                if r["dataset"] == ds:
                    per_model.setdefault(r["model"], {})[E] = r
        models = sorted(per_model)

        # Only chart metrics that were actually computed (e.g. forward transfer is
        # not calculated in the multi-head/TIL setting, so it is skipped here).
        metric_keys = [
            key for key in METRIC_TITLES
            if any(mean_std(entry, key)[0] is not None
                   for m in models for entry in per_model[m].values())
        ]

        n_panels = len(metric_keys)
        ncols = 2                        # 2 columns keep each panel legible at 16 cm
        nrows = -(-n_panels // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=grid_size(nrows, ncols),
                                 squeeze=False)
        fig.suptitle(f"Epoch-number analysis -- {ds}  (mean +/- std over configs)",
                     fontsize=FS_SUPTITLE, fontweight="bold")
        for ax in axes.flat[n_panels:]:
            ax.axis("off")
        # Markers sit exactly on the 5/10/20 epoch values; series are told apart by
        # marker shape and dash pattern (not colour alone) so overlapping lines in a
        # tight cluster stay distinguishable.
        n_models = len(models)
        for ax, key in zip(axes.flat, metric_keys):
            ax.set_title(METRIC_TITLES[key], fontsize=FS_TITLE)
            ax.set_xlabel("epochs")
            ax.set_xticks(epochs_sorted)
            for i, model in enumerate(models):
                marker, ls = _series_style(i)
                xs, ys, errs = [], [], []
                for E in epochs_sorted:
                    entry = per_model[model].get(E)
                    if entry is None:
                        continue
                    mu, sd = mean_std(entry, key)
                    if mu is None:
                        continue
                    xs.append(E)
                    ys.append(mu)
                    errs.append(sd)
                if xs:
                    ax.errorbar(xs, ys, yerr=errs, marker=marker, linestyle=ls,
                                capsize=2, markersize=4, elinewidth=0.7, alpha=0.85,
                                color=colors[i % len(colors)], label=model, linewidth=1.0)
            if key in _ZERO_LINE_METRICS:
                ax.axhline(0, color="gray", lw=0.6, ls="--")

        # One key for the whole figure, matching each model's colour/marker/dash so a
        # line can be identified even where it overlaps others. A panel may omit a
        # model whose metric is undefined, so the key uses the full model list.
        handles = [Line2D([], [], color=colors[i % len(colors)],
                          marker=_series_style(i)[0], linestyle=_series_style(i)[1],
                          markersize=4, linewidth=1.0, label=m)
                   for i, m in enumerate(models)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
                   ncol=min(5, n_models), fontsize=FS_LEGEND, frameon=False)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        ds_slug = ds.lower().replace("-", "_").replace(" ", "")
        save_fig(fig, os.path.join(out_dir, f"{ds_slug}_metric_vs_epochs.png"))
        plt.close(fig)
