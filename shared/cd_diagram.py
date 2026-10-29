"""Critical Difference diagram for the Nemenyi post-hoc test (Demsar 2006, Fig. 1).

Models sit on an average-rank axis, best on the left. Any two models joined by a thick
horizontal bar are *not* significantly different -- their average ranks are closer
together than the Critical Difference. Reading it: bars group the models that the data
cannot separate; models in no common bar differ significantly.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.plot_style import FS_ANNOT, FS_TICK, apply_thesis_style, save_fig


def _cliques(names, ranks, cd):
    """Maximal runs of rank-adjacent models spanning no more than the CD.

    Only maximal runs are kept: a run fully contained in a longer one would draw a
    redundant bar on top of it.
    """
    k = len(names)
    spans = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] <= cd:
            j += 1
        if j > i:
            spans.append((i, j))
    return [s for s in spans
            if not any(o != s and o[0] <= s[0] and s[1] <= o[1] for o in spans)]


def plot_cd_diagram(names, ranks, cd, path, title=None, alpha=0.05, also_pdf=True):
    """Draw the diagram for models `names` with `ranks` (both already sorted best-first).

    `cd` is the Critical Difference from shared.stats_tests.nemenyi_cd.
    """
    apply_thesis_style()
    k = len(names)
    lo, hi = 1, k
    n_left = (k + 1) // 2                      # left column takes the better half
    row_gap = 0.34
    n_rows = max(n_left, k - n_left)
    top = 0.0                                  # axis line sits here, rows hang below

    # The non-significance bars live between the axis and the first label row, so the
    # rows have to start below however many bars end up stacked there.
    cliques = _cliques(names, ranks, cd)
    bar_gap = 0.085
    bar_zone = 0.13 + bar_gap * max(len(cliques) - 1, 0)
    first_row = top - bar_zone - 0.20
    bottom = first_row - row_gap * (n_rows - 1)

    fig, ax = plt.subplots(figsize=(5.5, 1.35 + 0.28 * n_rows))
    ax.set_xlim(lo - 0.55 * (hi - lo) / k, hi + 0.55 * (hi - lo) / k)
    ax.set_ylim(bottom - 0.42, top + 0.95)
    ax.axis("off")

    # rank axis
    ax.plot([lo, hi], [top, top], color="black", linewidth=0.8, zorder=3)
    for tick in range(lo, hi + 1):
        ax.plot([tick, tick], [top, top + 0.06], color="black", linewidth=0.8, zorder=3)
        ax.text(tick, top + 0.10, str(tick), ha="center", va="bottom", fontsize=FS_TICK)

    # CD ruler
    if cd <= hi - lo:
        ax.plot([lo, lo + cd], [top + 0.50, top + 0.50], color="black", linewidth=0.9)
        for end in (lo, lo + cd):
            ax.plot([end, end], [top + 0.46, top + 0.54], color="black", linewidth=0.9)
        ax.text(lo + cd / 2, top + 0.57, "CD = %.2f" % cd,
                ha="center", va="bottom", fontsize=FS_ANNOT)

    # model stems: elbow from the rank position out to a label in the left/right column
    for i, (name, rank) in enumerate(zip(names, ranks)):
        left = i < n_left
        row = i if left else k - 1 - i
        y = first_row - row_gap * row
        edge = ax.get_xlim()[0] if left else ax.get_xlim()[1]
        ax.plot([rank, rank], [top, y], color="0.45", linewidth=0.7, zorder=2)
        ax.plot([rank, edge], [y, y], color="0.45", linewidth=0.7, zorder=2)
        ax.text(edge, y + 0.05, "%s (%.2f)" % (name, rank),
                ha="left" if left else "right", va="bottom", fontsize=FS_ANNOT)

    # non-significance bars, stacked so overlapping cliques stay readable
    for depth, (i, j) in enumerate(cliques):
        y = top - 0.13 - bar_gap * depth
        ax.plot([ranks[i] - 0.04, ranks[j] + 0.04], [y, y],
                color="black", linewidth=2.2, solid_capstyle="round", zorder=4)

    ax.text((lo + hi) / 2, bottom - 0.34,
            "average rank (1 = best); models joined by a bar are not significantly "
            "different at $\\alpha$ = %g" % alpha,
            ha="center", va="center", fontsize=FS_ANNOT, color="0.3")
    if title:
        ax.set_title(title, pad=18)

    save_fig(fig, path, also_pdf=also_pdf)
    plt.close(fig)
