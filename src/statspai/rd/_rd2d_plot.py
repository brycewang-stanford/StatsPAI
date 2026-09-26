"""Plotting for boundary discontinuity designs (:func:`sp.rd2d_plot`)."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility

_PLOT_TYPES = ("scatter", "heatmap", "boundary_effects")


@accepts_aliases(treat="treatment")
def rd2d_plot(
    data: pd.DataFrame,
    y: str,
    x1: str,
    x2: str,
    treatment: str,
    boundary: Optional[Callable[..., Any]] = None,
    result: Optional[CausalResult] = None,
    plot_type: str = "scatter",
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> Tuple[Any, Any]:
    """
    2D boundary RD visualization.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable name.
    x1, x2 : str
        Running variable names.
    treatment : str
        Binary assignment indicator.
    boundary : callable, optional
        Boundary function ``f(x1) -> x2``; ``None`` draws the line ``x1 = 0``.
    result : CausalResult, optional
        Result of :func:`sp.rd2d`; its evaluation points are marked on the
        scatter, and ``plot_type='boundary_effects'`` plots its pointwise
        bias-corrected effects with robust intervals.
    plot_type : {'scatter', 'heatmap', 'boundary_effects'}
    ax : matplotlib Axes, optional
    figsize : tuple, default (10, 8)

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> x1 = rng.uniform(-1, 1, n)
    >>> x2 = rng.uniform(-1, 1, n)
    >>> treat = (x1 >= 0).astype(int)
    >>> y = 1.5 * treat + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "treat": treat})
    >>> res = sp.rd2d(df, y="y", x1="x1", x2="x2", treatment="treat")
    >>> fig, ax = sp.rd2d_plot(
    ...     df, y="y", x1="x1", x2="x2", treatment="treat", result=res
    ... )
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:  # pragma: no cover
        raise ImportError(
            "matplotlib required. Install: pip install matplotlib"
        )  # pragma: no cover

    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "`data` must be a pandas DataFrame.",
            recovery_hint="Pass a DataFrame containing y, x1, x2, and treatment.",
            diagnostics={"type": type(data).__name__},
        )
    for col in (y, x1, x2, treatment):
        if col not in data.columns:
            raise MethodIncompatibility(
                f"Column '{col}' not found in data",
                recovery_hint="Check y/x1/x2/treatment names against data.columns.",
                diagnostics={"missing_columns": [col]},
            )
    if boundary is not None and not callable(boundary):
        raise MethodIncompatibility(
            "boundary must be callable or None",
            recovery_hint="Pass boundary=lambda x1: f(x1), or leave boundary=None.",
            diagnostics={"boundary_type": type(boundary).__name__},
        )
    if plot_type not in _PLOT_TYPES:
        raise MethodIncompatibility(
            "plot_type must be 'scatter', 'heatmap', or 'boundary_effects', got "
            f"'{plot_type}'",
            recovery_hint="Use plot_type='scatter', 'heatmap', or 'boundary_effects'.",
            diagnostics={"plot_type": plot_type},
        )

    Y = data[y].to_numpy(dtype=float)
    X1 = data[x1].to_numpy(dtype=float)
    X2 = data[x2].to_numpy(dtype=float)
    T = data[treatment].to_numpy(dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x1_range = np.linspace(np.nanmin(X1), np.nanmax(X1), 300)
    x2_boundary = (
        None if boundary is None else np.array([boundary(v) for v in x1_range])
    )

    def _draw_boundary(lw: float) -> None:
        if x2_boundary is not None:
            ax.plot(
                x1_range, x2_boundary, "k-", linewidth=lw, label="Boundary", zorder=3
            )
        else:
            ax.axvline(x=0, color="k", linewidth=lw, label="Boundary (x1=0)", zorder=3)

    detail = None if result is None else result.detail

    if plot_type == "scatter":
        ax.scatter(
            X1[T == 0],
            X2[T == 0],
            c="#3498DB",
            alpha=0.4,
            s=15,
            label="Control",
            zorder=2,
        )
        ax.scatter(
            X1[T == 1],
            X2[T == 1],
            c="#E74C3C",
            alpha=0.4,
            s=15,
            label="Treated",
            zorder=2,
        )
        _draw_boundary(2)
        if detail is not None and "b1" in detail.columns:
            ax.scatter(
                detail["b1"],
                detail["b2"],
                marker="D",
                c="k",
                s=40,
                zorder=4,
                label="Evaluation points",
            )
        ax.set_xlabel(x1, fontsize=11)
        ax.set_ylabel(x2, fontsize=11)
        ax.set_title("2D Boundary RD: Treatment Assignment", fontsize=13)
        ax.legend(fontsize=9, loc="best")

    elif plot_type == "heatmap":
        norm = Normalize(vmin=np.nanpercentile(Y, 2), vmax=np.nanpercentile(Y, 98))
        sc = ax.scatter(
            X1, X2, c=Y, cmap="RdYlBu_r", norm=norm, s=12, alpha=0.7, zorder=2
        )
        fig.colorbar(sc, ax=ax, label=y, shrink=0.8)
        _draw_boundary(2.5)
        ax.set_xlabel(x1, fontsize=11)
        ax.set_ylabel(x2, fontsize=11)
        ax.set_title("2D Boundary RD: Outcome Heatmap", fontsize=13)
        ax.legend(fontsize=9, loc="best")

    else:  # boundary_effects
        if detail is None or "estimate_q" not in detail.columns:
            raise MethodIncompatibility(
                "plot_type='boundary_effects' requires a result from rd2d() with "
                "boundary eval points (approach='location' or 'distance').",
                recovery_hint=(
                    "Re-estimate with rd2d(..., eval_points=...) or n_eval > 1."
                ),
                diagnostics={"plot_type": plot_type, "has_result": result is not None},
            )
        pos = np.arange(len(detail))
        est = detail["estimate_q"].to_numpy()
        ax.errorbar(
            pos,
            est,
            yerr=[est - detail["ci_lower"], detail["ci_upper"] - est],
            fmt="o-",
            color="#2C3E50",
            capsize=4,
            capthick=1.2,
            linewidth=1.5,
            markersize=6,
            zorder=3,
        )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xticks(pos)
        ax.set_xticklabels(
            [f"({a:.2g}, {b:.2g})" for a, b in zip(detail["b1"], detail["b2"])],
            rotation=30,
            fontsize=8,
        )
        ax.set_xlabel("Boundary point b = (b1, b2)", fontsize=11)
        ax.set_ylabel("Bias-corrected effect (robust CI)", fontsize=11)
        ax.set_title("2D Boundary RD: Effects Along Boundary", fontsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig, ax
