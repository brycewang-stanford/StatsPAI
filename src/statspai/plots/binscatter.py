"""
Binned scatter plots with residualization.

Creates publication-quality binned scatter plots that visualize the
conditional expectation E[Y|X] after partialling out control variables
and/or fixed effects — the workhorse visualization of empirical
economics.

The implementation follows two traditions:

1. **Classic binscatter** (Stepner 2013): Residualize Y and X on
   controls, bin X into equal-sized bins, plot bin means of Y against
   bin means of X, overlay a linear/polynomial fit.

2. **binsreg** (Cattaneo, Crump, Farrell & Feng 2024): Data-driven
   optimal bin selection based on IMSE criteria with pointwise
   confidence intervals.

References
----------
Cattaneo, M.D., Crump, R.K., Farrell, M.H. and Feng, Y. (2024).
"On Binscatter."
*American Economic Review*, 114(5), 1488-1514.
doi:10.1257/aer.20221576
[@cattaneo2024binscatter]

Stepner, M. (2013).
"binscatter: Stata module to generate binned scatterplots."
Statistical Software Components, Boston College.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ..exceptions import MethodIncompatibility


@accepts_aliases(covariates="controls")
def binscatter(
    data: pd.DataFrame,
    y: str,
    x: str,
    controls: Optional[List[str]] = None,
    absorb: Optional[List[str]] = None,
    n_bins: Optional[int] = None,
    ci: bool = False,
    ci_level: float = 0.95,
    fit: str = "linear",
    fit_on_raw: bool = False,
    by: Optional[str] = None,
    weights: Optional[str] = None,
    quantiles: bool = True,
    ax: Any = None,
    figsize: Tuple[float, float] = (8, 6),
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    legend: bool = True,
    scatter_kw: Optional[dict] = None,
    line_kw: Optional[dict] = None,
    method: str = "classic",
    cb: bool = False,
    nsims: int = 500,
    simsgrid: int = 20,
    seed: Optional[int] = None,
) -> Tuple:
    """
    Binned scatter plot with optional residualization.

    Visualizes E[Y|X] after controlling for covariates and/or fixed
    effects. Standard tool in empirical economics for showing
    conditional relationships.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable (Y-axis).
    x : str
        Running variable (X-axis).
    controls : list of str, optional
        Control variables to partial out from both Y and X before
        binning. Uses OLS residualization.
    absorb : list of str, optional
        Fixed-effect variables to absorb (demeaned within groups)
        before binning. More efficient than including as dummies in
        ``controls``.
    n_bins : int, optional
        Number of bins. Default: min(20, ceil(n^(1/3))) following the
        IMSE-optimal rate from Cattaneo et al. (2024).
    ci : bool, default False
        Show pointwise confidence intervals within each bin.
    ci_level : float, default 0.95
        Confidence level for CIs.
    fit : str, default 'linear'
        Overlay fit line: ``'linear'``, ``'quadratic'``, ``'cubic'``,
        ``'poly4'``, or ``'none'``.
    fit_on_raw : bool, default False
        If True, fit the overlay line on the raw (residualized) micro
        data, not the bin means. More accurate for nonlinear fits.
    by : str, optional
        Grouping variable — produces separate series on the same plot
        (e.g., ``by='female'`` overlays male and female).
    weights : str, optional
        Analytic weight variable (Stata ``[aw=]``). With ``'classic'`` it
        weights the absorption and residualization, the means the
        residuals are put back on, the bin means and their intervals, the
        quantile bins (weighted quantiles, as Stata's binscatter), and the
        fit line with ``fit_on_raw=True``.
    quantiles : bool, default True
        If True, bins are quantile-spaced (equal number of obs per
        bin; equal weight with ``weights``). If False, bins are evenly
        spaced on the X-axis.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Creates new figure if None.
    figsize : tuple, default (8, 6)
    colors : list of str, optional
        Colors for each ``by`` group.
    markers : list of str, optional
        Marker styles for each ``by`` group.
    title, x_label, y_label : str, optional
    legend : bool, default True
    scatter_kw : dict, optional
        Extra kwargs for ``ax.scatter()``.
    line_kw : dict, optional
        Extra kwargs for ``ax.plot()`` (the fit line).
    method : {'classic', 'binsreg'}, default 'classic'
        ``'classic'`` residualizes ``y`` and ``x`` on the controls and
        plots bin means (Stepner's ``binscatter``). ``'binsreg'`` is the
        binscatter of Cattaneo, Crump, Farrell and Feng (2024), as R / Stata
        ``binsreg`` computes it: the controls enter one regression with the
        bin indicators and each dot is that bin's coefficient at the mean of
        the controls (residualizing estimates a different function when
        ``x`` and the controls are related); bins at the quantiles of ``x``
        (``quantiles`` is ignored); ``n_bins=None`` selects the IMSE-optimal
        number (direct plug-in, as ``binsregselect``); ``ci=True`` gives
        robust bias-corrected intervals from a continuous piecewise-linear
        fit on the same bins. ``absorb`` and ``weights`` are not available
        with ``'binsreg'``.
        With ``controls`` and ``'classic'`` an ``AssumptionWarning`` says
        that residualize-then-bin is not consistent for ``E[y | x, w]``
        and ``bin_data.attrs['diagnostics']`` records it; the default stays
        ``'classic'``.
    cb : bool, default False
        With ``method='binsreg'``: add the uniform confidence band (same
        piecewise-linear fit, simulated critical value of the sup-t
        statistic at level ``ci_level``).
    nsims : int, default 500
        Simulation draws for the band's critical value.
    simsgrid : int, default 20
        Evaluation points inside each bin for the band.
    seed : int, optional
        Seed of the band's simulation.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    bin_data : pd.DataFrame
        The binned data (columns: x_mean, y_mean, n, [ci_lower, ci_upper]).
        With ``method='binsreg'``, ``bin_data.attrs['binsreg']`` holds the
        number of bins, the selection (``rot_poly`` / ``rot`` / ``dpi``),
        the knots, and with ``cb=True`` the band (``x``, ``fit``,
        ``cb_lower``, ``cb_upper``) and its critical value; per group when
        ``by`` is given.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> education = rng.integers(8, 20, n).astype(float)
    >>> age = rng.normal(40, 10, n)
    >>> experience = np.clip(age - education - 6, 0, None)
    >>> female = rng.integers(0, 2, n)
    >>> firm_id = rng.integers(0, 10, n)
    >>> year = rng.integers(2010, 2015, n)
    >>> wage = (5 + 0.4 * education + 0.05 * age + 0.03 * experience
    ...         - 0.5 * female + rng.normal(0, 1, n))
    >>> df = pd.DataFrame({'wage': wage, 'education': education, 'age': age,
    ...                    'experience': experience, 'female': female,
    ...                    'firm_id': firm_id, 'year': year})

    Basic usage:

    >>> fig, ax, bins = sp.binscatter(df, y='wage', x='education')

    With controls (partial out age and experience):

    >>> fig, ax, bins = sp.binscatter(df, y='wage', x='education',
    ...                                controls=['age', 'experience'])

    With fixed effects:

    >>> fig, ax, bins = sp.binscatter(df, y='wage', x='education',
    ...                                absorb=['firm_id', 'year'])

    By gender, with confidence intervals:

    >>> fig, ax, bins = sp.binscatter(df, y='wage', x='education',
    ...                                controls=['age'],
    ...                                by='female', ci=True)

    Notes
    -----
    The classic binscatter algorithm:

    1. Residualize: Regress Y on controls → get ỹ. Regress X on
       controls → get x̃. Add back means for interpretability.
    2. Bin: Sort x̃ into ``n_bins`` equal-frequency (quantile) bins.
    3. Plot: For each bin, plot (mean(x̃), mean(ỹ)).
    4. Fit: Overlay a linear/polynomial fit.

    When ``absorb`` is specified, ``y``, ``x`` and the controls are first
    within-transformed on the fixed effects (weighted with ``weights``), so
    the residuals are those of one regression on controls and fixed
    effects; the points are then put back on the means of ``y`` and ``x``.

    See Cattaneo et al. (2024, *AER*) for the theory behind optimal
    bin selection and the pitfalls of naive binscatter.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib and scipy required. " "Install: pip install matplotlib scipy"
        )

    # --- Validate inputs ---
    df = data.copy()
    required_cols = [y, x]
    if controls:
        required_cols += controls
    if absorb:
        required_cols += absorb
    if by:
        required_cols.append(by)
    if weights:
        required_cols.append(weights)

    check_columns(
        df,
        {
            "y": y,
            "x": x,
            "controls": controls or None,
            "absorb": absorb or None,
            "by": by or None,
            "weights": weights or None,
        },
        function="binscatter",
    )

    # Drop NaN in relevant columns
    df = df[required_cols].dropna()

    if method not in ("classic", "binsreg"):
        raise MethodIncompatibility(
            f"binscatter: method must be 'classic' or 'binsreg', not {method!r}",
            recovery_hint="Pass method='binsreg' for the CCFF (2024) binscatter.",
        )
    if method == "binsreg" and absorb:
        raise MethodIncompatibility(
            "binscatter(method='binsreg') does not absorb fixed effects",
            recovery_hint="Add the fixed effects to controls as indicator "
            "columns, or use method='classic'.",
        )
    if method == "binsreg" and weights:
        raise MethodIncompatibility(
            "binscatter(method='binsreg') does not take weights",
            recovery_hint="Use method='classic' for weighted bin means.",
        )

    if len(df) < 10:
        raise ValueError("Need at least 10 observations after dropping NaN")

    # Residualize-then-bin (Stepner's binscatter) does not estimate
    # E[y | x, w] unless x and the controls are unrelated; binsreg puts the
    # controls in one regression with the bin indicators (CCFF 2024).
    classic_controls_note: Optional[str] = None
    if method == "classic" and controls:
        classic_controls_note = (
            "binscatter(method='classic') residualizes y and x on the "
            "controls before binning; that is not a consistent estimate of "
            "E[y | x, controls] when x is related to the controls."
        )
        from ..exceptions import AssumptionWarning
        from ..exceptions import warn as _warn

        _warn(
            AssumptionWarning,
            classic_controls_note
            + " Pass method='binsreg' (Cattaneo, Crump, Farrell and Feng "
            "2024), which keeps the controls in one regression with the bin "
            "indicators.",
            stacklevel=3,
            recovery_hint="Pass method='binsreg' to sp.binscatter.",
            diagnostics={"method": "classic", "controls": list(controls)},
        )

    # --- Default bins ---
    if n_bins is None and method == "classic":
        n_bins = min(20, max(5, int(np.ceil(len(df) ** (1 / 3)))))
    binsreg_info: dict = {}

    # --- Defaults for colors/markers ---
    if colors is None:
        colors = [
            "#2C3E50",
            "#E74C3C",
            "#3498DB",
            "#2ECC71",
            "#9B59B6",
            "#F39C12",
            "#1ABC9C",
            "#E67E22",
        ]
    if markers is None:
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    # --- Determine groups ---
    if by is not None:
        groups = sorted(df[by].unique())
    else:
        groups = [None]

    # --- Create figure ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    all_bin_data = []

    for g_idx, group_val in enumerate(groups):
        # Subset for this group
        if group_val is not None:
            mask = df[by] == group_val
            df_g = df[mask].copy()
            label = f"{by}={group_val}"
        else:
            df_g = df.copy()
            label = None

        Y_raw = df_g[y].values.astype(float)
        X_raw = df_g[x].values.astype(float)
        W = df_g[weights].values.astype(float) if weights else None

        if method == "binsreg":
            from ._binsreg import binsreg_series

            series = binsreg_series(
                X_raw,
                Y_raw,
                df_g[controls].to_numpy(dtype=float) if controls else None,
                n_bins,
                ci,
                cb,
                ci_level,
                nsims,
                simsgrid,
                seed,
            )
            bin_df = series["dots"]
            binsreg_info[group_val] = {
                k: series[k]
                for k in (
                    "n_bins",
                    "selection",
                    "knots",
                    "band",
                    "cb_critical_value",
                )
            }
            X_plot, Y_plot = X_raw, Y_raw

        if method == "classic":
            # The levels the residuals are put back on: the (weighted) means
            # of the variables themselves, as Stata's binscatter does. Taken
            # before absorbing, whose within-transform has mean zero; until
            # 1.41 an ``absorb`` plot was centred on (0, 0).
            Y_level = _wmean(Y_raw, W)
            X_level = _wmean(X_raw, W)
            Z = df_g[controls].values.astype(float) if controls else None

            # --- Step 1: Absorb fixed effects (weighted, exact) ---
            if absorb:
                Y_raw, X_raw, Z = _absorb_fe(Y_raw, X_raw, Z, df_g, absorb, W)

            # --- Step 2: Residualize on controls ---
            # With ``absorb`` the controls are within-transformed too, so
            # the residuals are those of one regression on controls + FE
            # (Frisch-Waugh-Lovell; Stata's ``areg``).
            if Z is not None:
                Y_resid, X_resid = _residualize(Y_raw, X_raw, Z, W)
            else:
                Y_resid = Y_raw
                X_resid = X_raw

            # Add back means for interpretability
            Y_plot = Y_resid - _wmean(Y_resid, W) + Y_level
            X_plot = X_resid - _wmean(X_resid, W) + X_level

            # --- Step 3: Bin ---
            bin_df = _compute_bins(X_plot, Y_plot, n_bins, quantiles, W, ci, ci_level)

        if group_val is not None:
            bin_df["group"] = group_val
        all_bin_data.append(bin_df)

        # --- Step 4: Plot scatter ---
        color = colors[g_idx % len(colors)]
        marker = markers[g_idx % len(markers)]
        skw = dict(s=50, zorder=5, alpha=0.85, edgecolors="white", linewidths=0.5)
        if scatter_kw:
            skw.update(scatter_kw)

        ax.scatter(
            bin_df["x_mean"],
            bin_df["y_mean"],
            color=color,
            marker=marker,
            label=label,
            **skw,
        )

        # CI error bars
        if ci and "ci_lower" in bin_df.columns:
            # Interval segments, not errorbar(yerr=): the binsreg intervals
            # are bias-corrected and need not contain the dot, and a
            # negative yerr made matplotlib raise (about 1 dataset in 10 at
            # n_bins=10, 2026-10-20 probe).
            ax.vlines(
                bin_df["x_mean"],
                bin_df["ci_lower"],
                bin_df["ci_upper"],
                color=color,
                linewidth=0.8,
                alpha=0.6,
                zorder=3,
            )

        band = binsreg_info.get(group_val, {}).get("band")
        if band is not None:
            ax.fill_between(
                band["x"],
                band["cb_lower"],
                band["cb_upper"],
                color=color,
                alpha=0.15,
                linewidth=0,
                zorder=2,
            )

        # --- Step 5: Overlay fit line ---
        if fit != "none":
            fit_order = {"linear": 1, "quadratic": 2, "cubic": 3, "poly4": 4}.get(
                fit, 1
            )

            if fit_on_raw:
                x_fit_data = X_plot
                y_fit_data = Y_plot
            else:
                x_fit_data = bin_df["x_mean"].values
                y_fit_data = bin_df["y_mean"].values

            if len(x_fit_data) > fit_order:
                if not fit_on_raw:
                    fit_w = np.sqrt(bin_df["n"].values)
                elif W is not None:
                    fit_w = np.sqrt(W)  # weighted least squares on the micro data
                else:
                    fit_w = None
                coeffs = np.polyfit(x_fit_data, y_fit_data, fit_order, w=fit_w)
                x_grid = np.linspace(
                    bin_df["x_mean"].min(), bin_df["x_mean"].max(), 200
                )
                y_grid = np.polyval(coeffs, x_grid)

                lkw = dict(linewidth=1.5, alpha=0.8, zorder=4)
                if line_kw:
                    lkw.update(line_kw)
                ax.plot(x_grid, y_grid, color=color, **lkw)

    # --- Styling ---
    ax.set_xlabel(x_label or x, fontsize=11)
    ax.set_ylabel(y_label or y, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend and by is not None:
        ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()

    # Combine bin data
    result_df = pd.concat(all_bin_data, ignore_index=True)
    if method == "binsreg":
        result_df.attrs["binsreg"] = (
            binsreg_info[None] if by is None else dict(binsreg_info)
        )
    if classic_controls_note is not None:
        result_df.attrs["diagnostics"] = {
            "classic_with_controls": classic_controls_note,
            "suggest": "method='binsreg'",
        }

    return fig, ax, result_df


# ======================================================================
# Internal helpers
# ======================================================================


def _wmean(v: np.ndarray, w: Optional[np.ndarray]) -> float:
    return float(v.mean()) if w is None else float(np.average(v, weights=w))


def _absorb_fe(
    y: np.ndarray,
    x: np.ndarray,
    z: Optional[np.ndarray],
    df: pd.DataFrame,
    fe_vars: List[str],
    w: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Weighted within-transform of ``y``, ``x`` and the controls ``z``.

    Exact residuals on the fixed-effect dummies (sparse direct solve), with
    the analytic weights when given. Until 1.41 this was ten unweighted
    alternating-projection sweeps, which ignored ``weights=`` and could stop
    short of convergence with several effects.
    """
    from ..exceptions import ConvergenceWarning
    from ..fast._fe_direct import residualise_codes

    cols = [y, x] + ([z[:, j] for j in range(z.shape[1])] if z is not None else [])
    M = np.column_stack(cols).astype(np.float64)
    codes = [
        pd.factorize(df[fe].to_numpy(), sort=False)[0].astype(np.int64)
        for fe in fe_vars
    ]
    M, ok = residualise_codes(M, codes, w)
    if not ok:
        import warnings

        warnings.warn(
            ConvergenceWarning(
                "binscatter: the fixed-effect residualisation stopped at its "
                "rounding floor; the plotted points may be inaccurate.",
                recovery_hint="Check for very large or nearly collinear effects.",
            ),
            stacklevel=3,
        )
    z_out = M[:, 2:] if z is not None else None
    return M[:, 0], M[:, 1], z_out


def _residualize(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    W: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partial out controls Z from both Y and X via OLS."""
    n = len(Y)
    Z_c = np.column_stack([np.ones(n), Z])

    if W is not None:
        sqw = np.sqrt(W)
        Z_w = Z_c * sqw[:, np.newaxis]
        Y_w = Y * sqw
        X_w = X * sqw
    else:
        Z_w = Z_c
        Y_w = Y
        X_w = X

    try:
        beta_y = np.linalg.lstsq(Z_w, Y_w, rcond=None)[0]
        beta_x = np.linalg.lstsq(Z_w, X_w, rcond=None)[0]
        Y_resid = Y - Z_c @ beta_y
        X_resid = X - Z_c @ beta_x
    except np.linalg.LinAlgError:
        Y_resid = Y
        X_resid = X

    return Y_resid, X_resid


def _compute_bins(
    X: np.ndarray,
    Y: np.ndarray,
    n_bins: int,
    quantiles: bool,
    W: Optional[np.ndarray],
    ci: bool,
    ci_level: float,
) -> pd.DataFrame:
    """Compute bin means (and optionally CIs)."""
    from scipy import stats as sp_stats

    # Determine bin edges
    right_closed = False
    if quantiles and W is not None:
        # Equal-weight bins at the weighted quantiles, as Stata's
        # ``xtile ... [aw=]`` forms them for binscatter: cut points are
        # ``_pctile`` weighted percentiles and a bin is ``(c_{k-1}, c_k]``.
        from ..output.sumstats import weighted_percentile

        cuts = [weighted_percentile(X, W, 100.0 * k / n_bins) for k in range(1, n_bins)]
        edges = np.unique(np.r_[X.min(), cuts, X.max()])
        n_bins_actual = len(edges) - 1
        right_closed = True
    elif quantiles:
        # Equal-frequency bins
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(X, percentiles)
        edges = np.unique(edges)  # remove duplicates
        n_bins_actual = len(edges) - 1
    else:
        # Equal-width bins
        edges = np.linspace(X.min(), X.max(), n_bins + 1)
        n_bins_actual = n_bins

    # Assign to bins
    bin_idx = np.digitize(X, edges[1:-1], right=right_closed)  # 0-indexed bins
    bin_idx = np.clip(bin_idx, 0, n_bins_actual - 1)

    rows = []
    z = sp_stats.norm.ppf(1 - (1 - ci_level) / 2)

    for b in range(n_bins_actual):
        mask = bin_idx == b
        n_b = mask.sum()
        if n_b == 0:
            continue

        x_b = X[mask]
        y_b = Y[mask]

        if W is not None:
            w_b = W[mask]
            x_mean = float(np.average(x_b, weights=w_b))
            y_mean = float(np.average(y_b, weights=w_b))
        else:
            x_mean = float(x_b.mean())
            y_mean = float(y_b.mean())

        row = {
            "x_mean": x_mean,
            "y_mean": y_mean,
            "n": n_b,
        }

        if ci and n_b > 1:
            if W is not None:
                # SE of an analytic-weighted mean (Stata ``mean y [aw=w]``,
                # ``regress y [aw=w]``): weights rescaled to sum to n_b.
                w_n = w_b * (n_b / w_b.sum())
                se = float(
                    np.sqrt(np.sum(w_n * (y_b - y_mean) ** 2) / ((n_b - 1) * n_b))
                )
            else:
                se = float(np.std(y_b, ddof=1) / np.sqrt(n_b))
            row["ci_lower"] = y_mean - z * se
            row["ci_upper"] = y_mean + z * se

        rows.append(row)

    return pd.DataFrame(rows)
