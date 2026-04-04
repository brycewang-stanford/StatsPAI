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

Stepner, M. (2013).
"binscatter: Stata module to generate binned scatterplots."
Statistical Software Components, Boston College.
"""

from typing import Optional, List, Tuple, Union, Any

import numpy as np
import pandas as pd


def binscatter(
    data: pd.DataFrame,
    y: str,
    x: str,
    controls: Optional[List[str]] = None,
    absorb: Optional[List[str]] = None,
    n_bins: Optional[int] = None,
    ci: bool = False,
    ci_level: float = 0.95,
    fit: str = 'linear',
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
        Analytic weight variable.
    quantiles : bool, default True
        If True, bins are quantile-spaced (equal number of obs per
        bin). If False, bins are evenly spaced on the X-axis.
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

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    bin_data : pd.DataFrame
        The binned data (columns: x_mean, y_mean, n, [ci_lower, ci_upper]).

    Examples
    --------
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

    When ``absorb`` is specified, fixed effects are removed via within-
    group demeaning before step 1.

    See Cattaneo et al. (2024, *AER*) for the theory behind optimal
    bin selection and the pitfalls of naive binscatter.
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import stats as sp_stats
    except ImportError:
        raise ImportError(
            "matplotlib and scipy required. "
            "Install: pip install matplotlib scipy"
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

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data")

    # Drop NaN in relevant columns
    df = df[required_cols].dropna()

    if len(df) < 10:
        raise ValueError("Need at least 10 observations after dropping NaN")

    # --- Default bins ---
    if n_bins is None:
        n_bins = min(20, max(5, int(np.ceil(len(df) ** (1 / 3)))))

    # --- Defaults for colors/markers ---
    if colors is None:
        colors = ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71',
                  '#9B59B6', '#F39C12', '#1ABC9C', '#E67E22']
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

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

        # --- Step 1: Absorb fixed effects ---
        if absorb:
            Y_raw = _absorb_fe(Y_raw, df_g, absorb)
            X_raw = _absorb_fe(X_raw, df_g, absorb)

        # --- Step 2: Residualize on controls ---
        if controls:
            Z = df_g[controls].values.astype(float)
            Y_resid, X_resid = _residualize(Y_raw, X_raw, Z, W)
        else:
            Y_resid = Y_raw
            X_resid = X_raw

        # Add back means for interpretability
        Y_plot = Y_resid - Y_resid.mean() + Y_raw.mean()
        X_plot = X_resid - X_resid.mean() + X_raw.mean()

        # --- Step 3: Bin ---
        bin_df = _compute_bins(X_plot, Y_plot, n_bins, quantiles, W,
                               ci, ci_level)

        if group_val is not None:
            bin_df['group'] = group_val
        all_bin_data.append(bin_df)

        # --- Step 4: Plot scatter ---
        color = colors[g_idx % len(colors)]
        marker = markers[g_idx % len(markers)]
        skw = dict(s=50, zorder=5, alpha=0.85, edgecolors='white',
                   linewidths=0.5)
        if scatter_kw:
            skw.update(scatter_kw)

        ax.scatter(bin_df['x_mean'], bin_df['y_mean'],
                   color=color, marker=marker, label=label, **skw)

        # CI error bars
        if ci and 'ci_lower' in bin_df.columns:
            ax.errorbar(
                bin_df['x_mean'], bin_df['y_mean'],
                yerr=[bin_df['y_mean'] - bin_df['ci_lower'],
                      bin_df['ci_upper'] - bin_df['y_mean']],
                fmt='none', color=color, capsize=2, linewidth=0.8,
                alpha=0.6, zorder=3,
            )

        # --- Step 5: Overlay fit line ---
        if fit != 'none':
            fit_order = {'linear': 1, 'quadratic': 2, 'cubic': 3,
                         'poly4': 4}.get(fit, 1)

            if fit_on_raw:
                x_fit_data = X_plot
                y_fit_data = Y_plot
            else:
                x_fit_data = bin_df['x_mean'].values
                y_fit_data = bin_df['y_mean'].values

            if len(x_fit_data) > fit_order:
                coeffs = np.polyfit(x_fit_data, y_fit_data, fit_order,
                                    w=np.sqrt(bin_df['n'].values) if not fit_on_raw else None)
                x_grid = np.linspace(bin_df['x_mean'].min(),
                                     bin_df['x_mean'].max(), 200)
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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend and by is not None:
        ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()

    # Combine bin data
    result_df = pd.concat(all_bin_data, ignore_index=True)

    return fig, ax, result_df


# ======================================================================
# Internal helpers
# ======================================================================

def _absorb_fe(
    y: np.ndarray,
    df: pd.DataFrame,
    fe_vars: List[str],
) -> np.ndarray:
    """Demean Y within each FE group (iterative for multiple FEs)."""
    resid = y.copy()
    for _ in range(10):  # iterate for multiple FEs
        old = resid.copy()
        for fe in fe_vars:
            groups = df[fe].values
            for g in np.unique(groups):
                mask = groups == g
                resid[mask] -= resid[mask].mean()
        if np.max(np.abs(resid - old)) < 1e-10:
            break
    return resid


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

    n = len(X)

    # Determine bin edges
    if quantiles:
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
    bin_idx = np.digitize(X, edges[1:-1])  # 0-indexed bins
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
            'x_mean': x_mean,
            'y_mean': y_mean,
            'n': n_b,
        }

        if ci and n_b > 1:
            se = float(np.std(y_b, ddof=1) / np.sqrt(n_b))
            row['ci_lower'] = y_mean - z * se
            row['ci_upper'] = y_mean + z * se

        rows.append(row)

    return pd.DataFrame(rows)
