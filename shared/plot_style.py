"""Print-sized matplotlib defaults for figures embedded in the thesis.

Matplotlib font sizes are absolute points, independent of ``figsize``. A figure
authored 20 in wide and scaled to a 16 cm text block shrinks its 10 pt labels to
3.2 pt. The only fix is to author the figure at its final printed width, so
``\\includegraphics[width=\\textwidth]`` applies a 1.0x scale and a 9 pt label
prints as 9 pt.

Everything here is expressed against TEXT_WIDTH_IN / TEXT_HEIGHT_IN; change those
two constants if the thesis geometry changes.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Thesis text block: 16 cm x 22 cm.
TEXT_WIDTH_IN = 16.0 / 2.54    # 6.30 in
TEXT_HEIGHT_IN = 22.0 / 2.54   # 8.66 in

# Point sizes as they appear on the printed page.
FS_SUPTITLE = 10
FS_TITLE = 9
FS_LABEL = 8
FS_TICK = 7
FS_LEGEND = 7
FS_ANNOT = 6
FS_FOOTER = 6

SAVE_DPI = 300


def apply_thesis_style():
    """Install print-sized rcParams. Call once before creating figures."""
    plt.rcParams.update({
        "figure.figsize":    (TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.618),
        "figure.titlesize":  FS_SUPTITLE,
        "font.size":         FS_LABEL,
        "axes.titlesize":    FS_TITLE,
        "axes.labelsize":    FS_LABEL,
        "xtick.labelsize":   FS_TICK,
        "ytick.labelsize":   FS_TICK,
        "legend.fontsize":   FS_LEGEND,
        "legend.title_fontsize": FS_LEGEND,
        "lines.linewidth":   1.0,
        "lines.markersize":  3.0,
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "savefig.dpi":       SAVE_DPI,
        "figure.dpi":        SAVE_DPI,
    })


def fig_size(width_frac=1.0, aspect=0.618, height=None):
    """Figure size in inches, clamped to the printable text block.

    width_frac: fraction of \\textwidth the figure will be included at.
    aspect:     height / width, ignored when `height` is given.
    height:     explicit height in inches, capped at TEXT_HEIGHT_IN.
    """
    w = TEXT_WIDTH_IN * width_frac
    h = height if height is not None else w * aspect
    return (w, min(h, TEXT_HEIGHT_IN))


def grid_size(nrows, ncols, panel_aspect=0.85, width_frac=1.0):
    """Size a nrows x ncols panel grid that fills the text width.

    Height follows from the per-panel aspect ratio and is capped at the text
    height so a full-page figure never overflows onto the next page.
    """
    w = TEXT_WIDTH_IN * width_frac
    h = (w / ncols) * panel_aspect * nrows
    return (w, min(h, TEXT_HEIGHT_IN))


def model_legend(fig, labels, colors, y=0.965, ncol=5, kind="patch"):
    """Figure-level colour key, placed under the suptitle.

    Panels in a grid share one colour per model, so a single figure legend replaces
    per-panel tick labels or a legend that eats a whole panel. Reserve room for it
    by passing ``rect=[0, 0, 1, 0.93]`` to tight_layout.

    kind: "patch" for bar charts, "line" for line/errorbar charts.
    """
    if kind == "line":
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=c, marker="o", markersize=3, label=lab)
                   for lab, c in zip(labels, colors)]
    else:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=c, edgecolor="white", label=lab)
                   for lab, c in zip(labels, colors)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=min(ncol, len(handles)), fontsize=FS_LEGEND, frameon=False)


def config_colors(n):
    """n visually distinct colours for per-task-config series.

    The default prop_cycle is tab10, so anything past 10 configs would silently
    reuse colour 1 and make two configs indistinguishable in the comparison
    legend. Past 10 we switch to tab20, reordered dark-shades-first: tab20's even
    entries *are* the tab10 hues, so configs 1-10 keep the same colour they had in
    10-config figures and configs 11+ take the lighter variants. Covers 20; past
    that we fall back to cycling.
    """
    if n <= 10:
        cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        tab20 = plt.get_cmap("tab20").colors
        cyc = list(tab20[0::2]) + list(tab20[1::2])   # 10 dark (= tab10), then 10 light
    return [matplotlib.colors.to_hex(cyc[i % len(cyc)]) for i in range(n)]


def zero_bar_stubs(ax, xs, values, colors, rng, width=0.8):
    """Draw a small colour stub centred on 0 for bars that round to ~0.

    A 0.00-valued bar has no height, so its colour (and thus which model it is)
    is invisible. The stub is a short mark on the zero line in the bar's colour;
    the printed value label still conveys the true magnitude.
    """
    thr = 0.006 * rng          # below this the real bar is <~2 px tall -> invisible
    h = 0.02 * rng
    for x, v, c in zip(xs, values, colors):
        if v is not None and abs(v) < thr:
            ax.bar(x, h, width=width, bottom=-h / 2.0, color=c,
                   edgecolor="white", linewidth=0.5, zorder=3)


def label_bars(ax, xs, values, errs=None, colors=None, width=0.8,
               fmt="{:.2f}", fontsize=FS_ANNOT):
    """Annotate bars with value labels and set a y-range that keeps every label
    clear of the axes frame.

    The bar-to-label gap and the headroom above the tallest label are proportional
    to the panel's own data range, so a panel spanning [0, 1] and one spanning
    [-0.6, 0.1] get visually identical spacing.

    If `colors` is given, near-zero bars get a colour stub so they stay visible.

    Returns (ymin, ymax), the final axis limits, so shaded regions can span them.
    """
    values = list(values)
    errs = [0.0] * len(values) if errs is None else list(errs)
    hi = max([v + e for v, e in zip(values, errs)] + [0.0])
    lo = min([v - e for v, e in zip(values, errs)] + [0.0])
    rng = (hi - lo) or 1.0
    gap = 0.045 * rng
    ymax = hi + 0.17 * rng
    ymin = lo - (0.17 * rng if lo < 0 else 0.02 * rng)
    ax.set_ylim(ymin, ymax)
    if colors is not None:
        zero_bar_stubs(ax, xs, values, colors, rng, width=width)
    for x, v, e in zip(xs, values, errs):
        if v >= 0:
            ax.text(x, v + e + gap, fmt.format(v), ha="center", va="bottom", fontsize=fontsize)
        else:
            ax.text(x, v - e - gap, fmt.format(v), ha="center", va="top", fontsize=fontsize)
    return ymin, ymax


def save_fig(fig, path, also_pdf=False):
    """Save at print dpi. Set also_pdf=True to additionally emit a vector PDF."""
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    if also_pdf:
        fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
