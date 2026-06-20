"""
Diagnostic dashboards, multi-method comparison, and robustness tables for RD.

Three publication-oriented utilities sit on top of the canonical estimators
in :mod:`statspai.rd`:

- :func:`rd_dashboard` — single-figure 4-panel diagnostic (RD plot, density,
  covariate balance, bandwidth sensitivity).  Adapted from the
  recommendations of Cattaneo, Idrobo & Titiunik (2020/2024) and the
  binned-scatter best-practices of Calonico, Cattaneo & Titiunik (2015,
  *Journal of the American Statistical Association* 110(512), 1753-1769).

- :func:`rd_compare` — side-by-side comparison of multiple RD estimators on
  the same data (e.g. ``rdrobust``, ``rd_honest``, ``rd_flex``,
  ``rdrandinf``).  Returns a tidy DataFrame with point estimates, SEs,
  and CIs in a ready-for-output layout.

- :func:`rd_robustness_table` — a sweep over kernels × bandwidth selectors
  × polynomial orders × donuts.  The returned :class:`pd.DataFrame` is
  ready to feed :func:`statspai.outreg2` or :meth:`pandas.DataFrame.to_latex`
  for publication.

References
----------
Calonico, S., Cattaneo, M.D. and Titiunik, R. (2015).
"Optimal Data-Driven Regression Discontinuity Plots." *Journal of the
American Statistical Association* 110(512), 1753-1769.
[@calonico2015optimal]

Cattaneo, M.D., Idrobo, N. and Titiunik, R. (2024).
"A Practical Introduction to Regression Discontinuity Designs:
Extensions." Cambridge University Press. doi:10.1017/9781009441896.
[@cattaneo2024extensions]
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Sequence, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# rd_dashboard
# =============================================================================


def rd_dashboard(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0.0,
    covs: Optional[List[str]] = None,
    fuzzy: Optional[str] = None,
    bw_grid: Optional[Sequence[float]] = None,
    h: Optional[float] = None,
    figsize: Tuple[float, float] = (12, 9),
    title: Optional[str] = None,
    save: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Four-panel RD diagnostic dashboard.

    Combines the four checks every RD analysis should report:

    1. **RD plot** (top-left): IMSE-binned scatter with polynomial fit.
    2. **Density discontinuity** (top-right): :func:`rdplotdensity`
       output for a McCrary-style manipulation test.
    3. **Covariate balance** (bottom-left): mean of each ``cov`` on
       each side of the cutoff with point ranges.
    4. **Bandwidth sensitivity** (bottom-right): point estimate and CI
       across a grid of bandwidths.

    Parameters
    ----------
    data : pd.DataFrame
    y, x : str
    c : float, default 0.0
        Cutoff.
    covs : list of str, optional
        Covariates for the balance panel.  If None, the panel shows the
        density's binomial near-cutoff test instead.
    fuzzy : str, optional
        Fuzzy treatment column (passed through to RD plot/sensitivity).
    bw_grid : sequence of float, optional
        Bandwidths to evaluate in the sensitivity panel.  If None,
        uses ``[0.5, 0.75, 1.0, 1.25, 1.5, 2.0] × h_mse``.
    h : float, optional
        Reference bandwidth used for plotting and as the basis for
        ``bw_grid``.  If None, MSE-optimal bandwidth from rdrobust.
    figsize : tuple, default (12, 9)
    title : str, optional
        Suptitle.
    save : str, optional
        If a path is given, also save the figure (extension determines
        format).

    Returns
    -------
    (fig, axes) where axes is a 2x2 numpy array.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 0.8 * (x >= 0) + 0.5 * x + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({"x": x, "y": y})
    >>> fig, axes = sp.rd_dashboard(df, y="y", x="x", c=0.0)
    >>> axes.shape
    (2, 2)
    >>> fig.savefig("rd_dashboard.png")  # doctest: +SKIP
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(  # pragma: no cover
            "rd_dashboard needs matplotlib. Install: pip install matplotlib"
        ) from e

    from .rdrobust import rdrobust, rdplot, rdplotdensity

    # Resolve reference bandwidth
    if h is None:
        try:
            h_ref = rdrobust(
                data,
                y=y,
                x=x,
                c=c,
                p=1,
                fuzzy=fuzzy,
                warn_mass_points=False,
                warn_weak_first_stage=False,
            )
            h_auto = h_ref.model_info.get("bandwidth_h", None)
            if isinstance(h_auto, tuple):
                h = float(h_auto[0])
            elif h_auto is not None:
                h = float(h_auto)
        except Exception:  # pragma: no cover
            h = None

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.35, wspace=0.30)

    # Panel 1: RD plot
    rdplot(
        data,
        y=y,
        x=x,
        c=c,
        ax=axes[0, 0],
        h=h,
        title="(a) RD plot",
        show_bw=h is not None,
    )

    # Panel 2: density discontinuity
    rdplotdensity(data, x=x, c=c, ax=axes[0, 1], title="(b) Density at cutoff")

    # Panel 3: covariate balance OR running-variable detail
    if covs:
        _plot_balance(axes[1, 0], data, x=x, c=c, covs=covs)
    else:
        _plot_running_var_summary(axes[1, 0], data, x=x, c=c)

    # Panel 4: bandwidth sensitivity
    _plot_bw_sensitivity(
        axes[1, 1],
        data,
        y=y,
        x=x,
        c=c,
        fuzzy=fuzzy,
        h_ref=h,
        bw_grid=bw_grid,
    )

    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.995)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")

    return fig, axes


def _plot_balance(
    ax: Any,
    data: pd.DataFrame,
    x: str,
    c: float,
    covs: List[str],
) -> None:
    df = data.dropna(subset=[x])
    left = df[df[x] < c]
    right = df[df[x] >= c]
    means = []
    for col in covs:
        if col not in df.columns:
            continue  # pragma: no cover
        ml, mr = float(left[col].mean()), float(right[col].mean())
        sl = float(left[col].std() / np.sqrt(max(left[col].count(), 1)))
        sr = float(right[col].std() / np.sqrt(max(right[col].count(), 1)))
        means.append((col, ml, mr, sl, sr))
    if not means:
        ax.text(
            0.5,
            0.5,
            "(no covariates supplied)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("(c) Covariate balance")
        ax.axis("off")
        return
    rows = list(range(len(means)))
    for i, (col, ml, mr, sl, sr) in enumerate(means):
        ax.errorbar(
            ml,
            i - 0.12,
            xerr=1.96 * sl,
            fmt="o",
            color="#E74C3C",
            capsize=3,
            label="Left" if i == 0 else None,
        )
        ax.errorbar(
            mr,
            i + 0.12,
            xerr=1.96 * sr,
            fmt="s",
            color="#3498DB",
            capsize=3,
            label="Right" if i == 0 else None,
        )
    ax.set_yticks(rows)
    ax.set_yticklabels([m[0] for m in means])
    ax.invert_yaxis()
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(c) Covariate balance at cutoff", fontsize=11)


def _plot_running_var_summary(
    ax: Any,
    data: pd.DataFrame,
    x: str,
    c: float,
) -> None:
    X = data[x].dropna().to_numpy(dtype=float)
    n_unique = int(np.unique(X).size)
    ax.hist(X[X < c], bins=30, alpha=0.5, color="#E74C3C", label="Left")
    ax.hist(X[X >= c], bins=30, alpha=0.5, color="#3498DB", label="Right")
    ax.axvline(c, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(x, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(
        f"(c) Running-variable distribution (n_unique={n_unique})", fontsize=11
    )
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_bw_sensitivity(
    ax: Any,
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float,
    fuzzy: Optional[str],
    h_ref: Optional[float],
    bw_grid: Optional[Sequence[float]],
) -> None:
    from .rdrobust import rdrobust

    if h_ref is None or h_ref <= 0:
        ax.text(
            0.5,
            0.5,
            "(bandwidth not available)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("(d) Bandwidth sensitivity")
        ax.axis("off")
        return
    if bw_grid is None:
        bw_values = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0]) * h_ref
    else:
        bw_values = np.atleast_1d(bw_grid).astype(float)

    estimates, lows, highs = [], [], []
    for h_val in bw_values:
        try:
            r = rdrobust(
                data,
                y=y,
                x=x,
                c=c,
                h=h_val,
                fuzzy=fuzzy,
                warn_mass_points=False,
                warn_weak_first_stage=False,
            )
            estimates.append(float(r.estimate))
            lo, hi = r.ci
            lows.append(float(lo))
            highs.append(float(hi))
        except Exception:  # pragma: no cover
            estimates.append(np.nan)
            lows.append(np.nan)
            highs.append(np.nan)
    est_values, low_values, high_values = map(np.array, (estimates, lows, highs))

    ax.fill_between(bw_values, low_values, high_values, color="#3498DB", alpha=0.20)
    ax.plot(bw_values, est_values, "-o", color="#2C3E50", linewidth=1.4)
    ax.axvline(
        h_ref, color="gray", linestyle="--", linewidth=0.8, label=f"h_MSE = {h_ref:.3f}"
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Bandwidth h", fontsize=10)
    ax.set_ylabel("τ̂ (95% CI)", fontsize=10)
    ax.set_title("(d) Bandwidth sensitivity", fontsize=11)
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# rd_compare
# =============================================================================

_DEFAULT_COMPARE_METHODS = (
    "rdrobust",
    "honest",
    "randinf",
)


def rd_compare(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0.0,
    methods: Sequence[str] = _DEFAULT_COMPARE_METHODS,
    fuzzy: Optional[str] = None,
    method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compare multiple RD estimators on the same data.

    Returns a tidy DataFrame: one row per method with point estimate,
    SE, p-value and CI.  Useful for robustness tables across estimation
    families (local-polynomial vs honest vs local-randomisation vs
    flexible-covariate-adjusted).

    Parameters
    ----------
    data : pd.DataFrame
    y, x : str
    c : float, default 0.0
    methods : sequence of str
        Method aliases recognised by the :data:`sp.rd._RD_METHOD_ALIASES`
        dispatcher.  Default: ``('rdrobust', 'honest', 'randinf')``.
    fuzzy : str, optional
        Fuzzy treatment column passed through to all methods that
        accept it.
    method_kwargs : dict of dict, optional
        Per-method extra kwargs, e.g. ``{'rdrobust': {'kernel': 'uniform'}}``.
    alpha : float, default 0.05
        Confidence level forwarded to each estimator that accepts ``alpha``.

    Returns
    -------
    pd.DataFrame
        Columns: ``method``, ``estimate``, ``se``, ``pvalue``,
        ``ci_lower``, ``ci_upper``, ``n_obs``, ``status``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 0.8 * (x >= 0) + 0.5 * x + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({"x": x, "y": y})
    >>> tab = sp.rd_compare(df, y="y", x="x", c=0.0,
    ...                     methods=["rdrobust"])
    >>> tab.shape
    (1, 8)
    >>> tab.loc[0, "method"], tab.loc[0, "status"]
    ('rdrobust', 'ok')
    >>> round(float(tab.loc[0, "estimate"]), 3)
    0.813
    """
    method_kwargs = method_kwargs or {}
    rows = []
    for m in methods:
        kw = dict(method_kwargs.get(m, {}))
        kw.setdefault("alpha", alpha)
        if fuzzy is not None and m in (
            "rdrobust",
            "forest",
            "boost",
            "lasso",
            "extrapolate",
        ):
            kw.setdefault("fuzzy", fuzzy)
        try:
            from . import _rd_dispatch  # circular-safe import

            r = _rd_dispatch(data=data, y=y, x=x, c=c, method=m, **kw)
            est = float(getattr(r, "estimate", float("nan")))
            se = (
                float(getattr(r, "se", float("nan")))
                if getattr(r, "se", None) is not None
                else float("nan")
            )
            pv = (
                float(getattr(r, "pvalue", float("nan")))
                if getattr(r, "pvalue", None) is not None
                else float("nan")
            )
            ci = getattr(r, "ci", None)
            lo, hi = (
                (float(ci[0]), float(ci[1]))
                if ci is not None
                else (float("nan"), float("nan"))
            )
            n = int(getattr(r, "n_obs", 0) or 0)
            rows.append(
                {
                    "method": m,
                    "estimate": est,
                    "se": se,
                    "pvalue": pv,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n_obs": n,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "method": m,
                    "estimate": float("nan"),
                    "se": float("nan"),
                    "pvalue": float("nan"),
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                    "n_obs": 0,
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# rd_robustness_table
# =============================================================================


def rd_robustness_table(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0.0,
    fuzzy: Optional[str] = None,
    kernels: Sequence[str] = ("triangular", "epanechnikov", "uniform"),
    bwselects: Sequence[str] = ("mserd", "cerrd", "msetwo"),
    polynomials: Sequence[int] = (1, 2),
    donuts: Sequence[float] = (0.0,),
    covs: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Sweep over (kernel, bwselect, polynomial, donut) and return a
    robustness table for reporting.

    Each row is one specification.  The DataFrame contains both
    "Conventional" and "Robust" point estimates and CIs from
    :func:`rdrobust`, plus the bandwidth used.  Suitable for
    direct ``.to_latex(...)`` / ``.to_excel(...)`` export, or feed
    into :func:`statspai.outreg2` for native multi-column tables.

    Parameters
    ----------
    data : pd.DataFrame
    y, x : str
    c : float
    fuzzy, covs, cluster : optional, forwarded to rdrobust
    kernels, bwselects, polynomials, donuts : sequences
        Specification grid.
    alpha : float

    Returns
    -------
    pd.DataFrame with one row per specification and columns:
        kernel, bwselect, p, donut, h, b,
        estimate_conv, se_conv, ci_conv_lo, ci_conv_hi, pvalue_conv,
        estimate_rbc,  se_rbc,  ci_rbc_lo,  ci_rbc_hi,  pvalue_rbc,
        n_left, n_right, status

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 0.8 * (x >= 0) + 0.5 * x + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({"x": x, "y": y})
    >>> tab = sp.rd_robustness_table(
    ...     df, y="y", x="x", c=0.0,
    ...     kernels=["triangular", "uniform"],
    ...     bwselects=["mserd"], polynomials=[1],
    ... )
    >>> tab.shape
    (2, 19)
    >>> tab["kernel"].tolist()
    ['triangular', 'uniform']
    >>> round(float(tab.loc[0, "estimate_conv"]), 3)
    0.765
    """
    from .rdrobust import rdrobust

    rows: List[Dict[str, Any]] = []
    for kernel in kernels:
        for bwselect in bwselects:
            for p in polynomials:
                for donut in donuts:
                    spec = {
                        "kernel": kernel,
                        "bwselect": bwselect,
                        "p": int(p),
                        "donut": float(donut),
                    }
                    try:
                        r = rdrobust(
                            data=data,
                            y=y,
                            x=x,
                            c=c,
                            fuzzy=fuzzy,
                            p=int(p),
                            kernel=kernel,
                            bwselect=bwselect,
                            donut=float(donut),
                            covs=covs,
                            cluster=cluster,
                            alpha=alpha,
                            warn_mass_points=False,
                            warn_weak_first_stage=False,
                        )
                        info = r.model_info or {}
                        conv = info.get("conventional", {})
                        rbc_ = info.get("robust", {})
                        h_used = info.get("bandwidth_h", float("nan"))
                        b_used = info.get("bandwidth_b", float("nan"))
                        rows.append(
                            {
                                **spec,
                                "h": h_used,
                                "b": b_used,
                                "estimate_conv": conv.get("estimate", float("nan")),
                                "se_conv": conv.get("se", float("nan")),
                                "ci_conv_lo": conv.get("ci", (float("nan"),) * 2)[0],
                                "ci_conv_hi": conv.get("ci", (float("nan"),) * 2)[1],
                                "pvalue_conv": conv.get("pvalue", float("nan")),
                                "estimate_rbc": rbc_.get("estimate", float("nan")),
                                "se_rbc": rbc_.get("se", float("nan")),
                                "ci_rbc_lo": rbc_.get("ci", (float("nan"),) * 2)[0],
                                "ci_rbc_hi": rbc_.get("ci", (float("nan"),) * 2)[1],
                                "pvalue_rbc": rbc_.get("pvalue", float("nan")),
                                "n_left": int(info.get("n_left", 0)),
                                "n_right": int(info.get("n_right", 0)),
                                "status": "ok",
                            }
                        )
                    except Exception as exc:  # pragma: no cover
                        rows.append(
                            {
                                **spec,
                                "h": float("nan"),
                                "b": float("nan"),
                                "estimate_conv": float("nan"),
                                "se_conv": float("nan"),
                                "ci_conv_lo": float("nan"),
                                "ci_conv_hi": float("nan"),
                                "pvalue_conv": float("nan"),
                                "estimate_rbc": float("nan"),
                                "se_rbc": float("nan"),
                                "ci_rbc_lo": float("nan"),
                                "ci_rbc_hi": float("nan"),
                                "pvalue_rbc": float("nan"),
                                "n_left": 0,
                                "n_right": 0,
                                "status": f"error: {type(exc).__name__}: {exc}",
                            }
                        )
    return pd.DataFrame(rows)
