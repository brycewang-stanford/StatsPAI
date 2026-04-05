"""
Local polynomial regression (lpoly).

Non-parametric estimation of conditional mean E[Y|X=x] using
local polynomial fitting with kernel weights.

Equivalent to Stata's ``lpoly`` and R's ``locpol``.

References
----------
Fan, J. & Gijbels, I. (1996).
"Local Polynomial Modelling and Its Applications."
*Chapman & Hall/CRC Monographs on Statistics & Applied Probability*.
"""

from typing import Optional, Union, List
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import EconometricResults


class LPolyResult:
    """Results from local polynomial regression."""

    def __init__(self, grid, fitted, se, ci_lower, ci_upper, bandwidth,
                 degree, kernel, n, data_x, data_y):
        self.grid = grid
        self.fitted = fitted
        self.se = se
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.bandwidth = bandwidth
        self.degree = degree
        self.kernel = kernel
        self.n = n
        self._data_x = data_x
        self._data_y = data_y

    def summary(self) -> str:
        lines = [
            "Local Polynomial Regression",
            "=" * 50,
            f"Kernel: {self.kernel:<20s} Degree: {self.degree}",
            f"Bandwidth: {self.bandwidth:.4f}{' ':>10s} N: {self.n}",
            f"Grid points: {len(self.grid)}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def plot(self, ax=None, scatter=True, ci=True, **kwargs):
        """Plot the local polynomial fit with optional scatter and CI."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        if scatter:
            ax.scatter(self._data_x, self._data_y, alpha=0.3, s=10,
                       color='gray', label='Data')

        ax.plot(self.grid, self.fitted, color='steelblue', lw=2,
                label=f'lpoly (degree={self.degree})')

        if ci:
            ax.fill_between(self.grid, self.ci_lower, self.ci_upper,
                            alpha=0.2, color='steelblue', label='95% CI')

        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
        ax.set_title(kwargs.get('title', 'Local Polynomial Regression'))
        ax.legend()
        return ax


def _kernel_fn(u, kernel='epanechnikov'):
    """Evaluate kernel function at u."""
    if kernel == 'epanechnikov':
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
    elif kernel == 'gaussian':
        return stats.norm.pdf(u)
    elif kernel == 'uniform':
        return np.where(np.abs(u) <= 1, 0.5, 0.0)
    elif kernel == 'triangular':
        return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)
    elif kernel == 'biweight':
        return np.where(np.abs(u) <= 1, (15/16) * (1 - u**2)**2, 0.0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def _silverman_bandwidth(x):
    """Silverman's rule-of-thumb bandwidth."""
    n = len(x)
    sigma = min(np.std(x, ddof=1), stats.iqr(x) / 1.349)
    if sigma == 0:
        sigma = np.std(x, ddof=1)
    return 1.06 * sigma * n**(-1/5)


def _local_poly_fit(x0, x, y, h, degree, kernel):
    """
    Fit local polynomial at a single point x0.

    Returns (estimate, standard_error).
    """
    u = (x - x0) / h
    w = _kernel_fn(u, kernel) / h

    # Only use points with non-zero weight
    mask = w > 0
    if mask.sum() < degree + 1:
        return np.nan, np.nan

    x_local = x[mask] - x0
    y_local = y[mask]
    w_local = w[mask]

    # Design matrix: [1, (x-x0), (x-x0)^2, ..., (x-x0)^p]
    X = np.column_stack([x_local**j for j in range(degree + 1)])
    W = np.diag(w_local)

    try:
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y_local
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    # Estimate is beta[0] (the intercept = E[Y|X=x0])
    fitted = beta[0]

    # Standard error via sandwich
    resid = y_local - X @ beta
    sigma2 = np.sum(w_local * resid**2) / max(mask.sum() - degree - 1, 1)
    try:
        XtWX_inv = np.linalg.inv(XtWX)
        se = np.sqrt(sigma2 * XtWX_inv[0, 0])
    except np.linalg.LinAlgError:
        se = np.nan

    return fitted, se


def lpoly(
    data: pd.DataFrame = None,
    y: str = None,
    x: str = None,
    bandwidth: float = None,
    degree: int = 1,
    kernel: str = "epanechnikov",
    n_grid: int = 100,
    grid: np.ndarray = None,
    ci: bool = True,
    alpha: float = 0.05,
) -> LPolyResult:
    """
    Local polynomial regression.

    Equivalent to Stata's ``lpoly y x`` and R's ``locpol()``.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Dependent variable name.
    x : str
        Independent variable name.
    bandwidth : float, optional
        Kernel bandwidth. If None, uses Silverman's rule-of-thumb.
    degree : int, default 1
        Polynomial degree (0=Nadaraya-Watson, 1=local linear, 2=local quadratic).
    kernel : str, default 'epanechnikov'
        Kernel function: 'epanechnikov', 'gaussian', 'uniform', 'triangular', 'biweight'.
    n_grid : int, default 100
        Number of evaluation grid points.
    grid : np.ndarray, optional
        Custom evaluation grid. Overrides n_grid.
    ci : bool, default True
        Compute confidence intervals.
    alpha : float, default 0.05
        Significance level for CI.

    Returns
    -------
    LPolyResult
        Results with fitted values, SE, CI, and plot method.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.lpoly(df, y='wage', x='experience')
    >>> result.plot()
    >>> print(result.summary())
    """
    if data is None:
        raise ValueError("data is required")

    y_data = data[y].values.astype(float)
    x_data = data[x].values.astype(float)

    # Drop missing
    valid = np.isfinite(y_data) & np.isfinite(x_data)
    y_data = y_data[valid]
    x_data = x_data[valid]
    n = len(y_data)

    if bandwidth is None:
        bandwidth = _silverman_bandwidth(x_data)

    if grid is None:
        grid = np.linspace(x_data.min(), x_data.max(), n_grid)

    fitted = np.empty(len(grid))
    se = np.empty(len(grid))

    for i, x0 in enumerate(grid):
        fitted[i], se[i] = _local_poly_fit(x0, x_data, y_data, bandwidth,
                                           degree, kernel)

    # Confidence intervals
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = fitted - z_crit * se
    ci_upper = fitted + z_crit * se

    return LPolyResult(
        grid=grid, fitted=fitted, se=se,
        ci_lower=ci_lower, ci_upper=ci_upper,
        bandwidth=bandwidth, degree=degree, kernel=kernel,
        n=n, data_x=x_data, data_y=y_data,
    )
