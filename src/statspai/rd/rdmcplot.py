"""RD plot for a design with several cutoffs (``rdmulti::rdmcplot``).

One :func:`statspai.rdplot` per cutoff, each on the units assigned to that
cutoff, drawn on a single axis. The numbers are those of ``rdplot`` on the
subsample; nothing is estimated here that ``rdplot`` does not estimate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._rdplot_core import rdplot_numbers

__all__ = ["rdmcplot"]

_PALETTE = (
    "#1F77B4",
    "#C0392B",
    "#2E8B57",
    "#8E44AD",
    "#E67E22",
    "#16A085",
    "#7F8C8D",
    "#D4AC0D",
)
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def _per_cutoff(value: Any, k: int, name: str) -> List[Any]:
    """Broadcast a scalar option, or check a per-cutoff sequence."""
    if value is None or np.isscalar(value):
        return [value] * k
    items = list(value)
    if len(items) != k:
        raise MethodIncompatibility(
            f"rdmcplot: {name}= lists {len(items)} values for {k} cutoffs."
        )
    return items


def rdmcplot(
    data: pd.DataFrame,
    y: str,
    x: str,
    cutoff_var: str,
    cutoffs: Optional[Sequence[float]] = None,
    *,
    p: Union[int, Sequence[int]] = 4,
    nbins: Union[None, int, Sequence[Optional[int]]] = None,
    binselect: Union[str, Sequence[str]] = "esmv",
    kernel: str = "uniform",
    h: Union[None, float, Sequence[Optional[float]]] = None,
    ci_level: float = 0.95,
    hide_ci: bool = True,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (10, 7),
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    RD plot with one binned scatter and polynomial fit per cutoff.

    For a design in which each unit faces its own cutoff (the design of
    :func:`statspai.rdmc`). Each cutoff's units are plotted as
    :func:`statspai.rdplot` would plot them alone, in their own colour,
    with a dashed line at the cutoff.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    x : str
        Running variable.
    cutoff_var : str
        Column holding each unit's cutoff.
    cutoffs : sequence of float, optional
        Cutoffs to plot. Defaults to every value of ``cutoff_var``.
    p : int or sequence of int, default 4
        Order of the global polynomial on each side. A sequence gives one
        order per cutoff.
    nbins : int or sequence, optional
        Bins on each side, the same for every cutoff or one per cutoff.
        Chosen by ``binselect`` when omitted.
    binselect : str or sequence of str, default 'esmv'
        Bin selection rule, as in :func:`statspai.rdplot`.
    kernel : str, default 'uniform'
    h : float or sequence, optional
        Bandwidth of the polynomial fit; the full support when omitted.
    ci_level : float, default 0.95
    hide_ci : bool, default True
        Draw the binned means without their intervals. Several overlaid
        series are hard to read with them.
    ax : matplotlib Axes, optional
    figsize : tuple, default (10, 7)
    title, x_label, y_label : str, optional

    Returns
    -------
    (fig, ax)
        ``fig.rdmcplot_data`` maps each cutoff to the dict that
        ``fig.rdplot_data`` holds for a single-cutoff plot (bin means,
        polynomial fit, number of bins).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 900
    >>> cut = rng.choice([30.0, 60.0], size=n)
    >>> score = cut + rng.uniform(-20, 20, n)
    >>> out = 1 + 0.5 * (score >= cut) + 0.01 * score + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({"y": out, "x": score, "cut": cut})
    >>> fig, ax = sp.rdmcplot(df, y="y", x="x", cutoff_var="cut", p=1)
    >>> sorted(fig.rdmcplot_data)
    [30.0, 60.0]

    References
    ----------
    [@cattaneo2024extensions]
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

    cvals = data[cutoff_var].to_numpy()
    if cutoffs is None:
        cutoffs = sorted(pd.unique(cvals[pd.notna(cvals)]).tolist())
    cutoffs = [float(c) for c in cutoffs]
    k = len(cutoffs)
    if k == 0:
        raise DataInsufficient(f"rdmcplot: {cutoff_var!r} has no observed cutoff.")
    p_list = _per_cutoff(p, k, "p")
    nbins_list = _per_cutoff(nbins, k, "nbins")
    bin_list = _per_cutoff(binselect, k, "binselect")
    h_list = _per_cutoff(h, k, "h")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    results: Dict[float, Dict[str, Any]] = {}
    for i, c in enumerate(cutoffs):
        sub = data.loc[cvals == c]
        Y = sub[y].to_numpy(dtype=float)
        X = sub[x].to_numpy(dtype=float)
        ok = np.isfinite(Y) & np.isfinite(X)
        if (X[ok] < c).sum() < 2 or (X[ok] >= c).sum() < 2:
            raise DataInsufficient(
                f"rdmcplot: cutoff {c:g} has fewer than two observations on "
                "one side."
            )
        res = rdplot_numbers(
            Y,
            X,
            c=c,
            p=int(p_list[i]),
            nbins=nbins_list[i],
            binselect=bin_list[i],
            kernel=kernel,
            h=h_list[i],
            ci=100.0 * ci_level,
        )
        results[c] = res
        colour = _PALETTE[i % len(_PALETTE)]
        vb, vp = res["vars_bins"], res["vars_poly"]
        bx, by = vb["rdplot_mean_bin"], vb["rdplot_mean_y"]
        if hide_ci:
            ax.scatter(
                bx,
                by,
                color=colour,
                s=26,
                alpha=0.85,
                marker=_MARKERS[i % len(_MARKERS)],
                label=f"cutoff {c:g}",
                zorder=3,
            )
        else:
            yerr = np.vstack([by - vb["rdplot_ci_l"], vb["rdplot_ci_r"] - by])
            ax.errorbar(
                bx,
                by,
                yerr=yerr,
                fmt=_MARKERS[i % len(_MARKERS)],
                color=colour,
                markersize=4,
                capsize=2,
                alpha=0.75,
                linewidth=0.8,
                label=f"cutoff {c:g}",
                zorder=3,
            )
        xs, ys = vp["rdplot_x"], vp["rdplot_y"]
        half = len(xs) // 2
        ax.plot(xs[:half], ys[:half], color=colour, linewidth=1.5, zorder=4)
        ax.plot(xs[half:], ys[half:], color=colour, linewidth=1.5, zorder=4)
        ax.axvline(x=c, color=colour, linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel(x_label or x, fontsize=11)
    ax.set_ylabel(y_label or y, fontsize=11)
    ax.set_title(title or "Multiple-cutoff RD plot", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc="best", frameon=False)
    fig.tight_layout()
    fig.rdmcplot_data = results  # type: ignore[attr-defined]
    return fig, ax
