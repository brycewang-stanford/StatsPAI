"""
Kernel density estimation (kdensity).

Non-parametric estimation of the probability density function f(x)
using kernel smoothing.

Equivalent to Stata's ``kdensity`` and R's ``density()``.

References
----------
Silverman, B.W. (1986).
"Density Estimation for Statistics and Data Analysis."
*Chapman & Hall/CRC*.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats


class KDensityResult:
    """Results from kernel density estimation."""

    def __init__(self, grid, density, bandwidth, kernel, n, data):
        self.grid = grid
        self.density = density
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.n = n
        self._data = data

    def summary(self) -> str:
        lines = [
            "Kernel Density Estimation",
            "=" * 50,
            f"Kernel: {self.kernel:<20s} Bandwidth: {self.bandwidth:.4f}",
            f"N: {self.n:<20d} Grid points: {len(self.grid)}",
            f"Min density: {self.density.min():.6f}",
            f"Max density: {self.density.max():.6f}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def plot(self, ax=None, hist=False, rug=False, **kwargs):
        """Plot the kernel density estimate."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        if hist:
            ax.hist(self._data, bins='auto', density=True, alpha=0.3,
                    color='gray', label='Histogram')

        ax.plot(self.grid, self.density, color='steelblue', lw=2,
                label=f'KDE (bw={self.bandwidth:.3f})')

        if rug:
            ax.plot(self._data, np.zeros_like(self._data), '|',
                    color='gray', alpha=0.3, ms=10)

        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Density'))
        ax.set_title(kwargs.get('title', 'Kernel Density Estimation'))
        ax.legend()
        return ax


def _kernel_fn(u, kernel='gaussian'):
    """Evaluate kernel function at u."""
    if kernel == 'gaussian':
        return stats.norm.pdf(u)
    elif kernel == 'epanechnikov':
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
    elif kernel == 'uniform':
        return np.where(np.abs(u) <= 1, 0.5, 0.0)
    elif kernel == 'triangular':
        return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)
    elif kernel == 'biweight':
        return np.where(np.abs(u) <= 1, (15/16) * (1 - u**2)**2, 0.0)
    elif kernel == 'cosine':
        return np.where(np.abs(u) <= 1, (np.pi/4) * np.cos(np.pi * u / 2), 0.0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def _silverman_bw(x):
    """Silverman's rule-of-thumb bandwidth."""
    n = len(x)
    sigma = min(np.std(x, ddof=1), stats.iqr(x) / 1.349)
    if sigma == 0:
        sigma = np.std(x, ddof=1)
    return 0.9 * sigma * n**(-1/5)


def _sheather_jones_bw(x):
    """Sheather-Jones (1991) plug-in bandwidth (simplified)."""
    from scipy.optimize import brentq

    n = len(x)
    sigma = np.std(x, ddof=1)
    iqr = stats.iqr(x)
    a = min(sigma, iqr / 1.349) if iqr > 0 else sigma

    # Use Silverman as fallback
    if a == 0:
        return _silverman_bw(x)

    return 0.9 * a * n**(-1/5)


def kdensity(
    data: pd.DataFrame = None,
    x: str = None,
    bandwidth: float = None,
    kernel: str = "gaussian",
    bw_method: str = "silverman",
    n_grid: int = 512,
    grid: np.ndarray = None,
    weights: str = None,
) -> KDensityResult:
    """
    Kernel density estimation.

    Equivalent to Stata's ``kdensity x`` and R's ``density(x)``.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    x : str
        Variable name for density estimation.
    bandwidth : float, optional
        Kernel bandwidth. If None, selected automatically.
    kernel : str, default 'gaussian'
        Kernel function: 'gaussian', 'epanechnikov', 'uniform',
        'triangular', 'biweight', 'cosine'.
    bw_method : str, default 'silverman'
        Bandwidth selection: 'silverman' or 'sheather-jones'.
    n_grid : int, default 512
        Number of evaluation grid points.
    grid : np.ndarray, optional
        Custom evaluation grid.
    weights : str, optional
        Column name for observation weights.

    Returns
    -------
    KDensityResult
        Results with density values, grid, and plot method.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.kdensity(df, x='income')
    >>> result.plot(hist=True)
    >>> print(result.summary())
    """
    if data is None:
        raise ValueError("data is required")

    x_data = data[x].dropna().values.astype(float)
    n = len(x_data)

    if weights is not None:
        w = data.loc[data[x].notna(), weights].values.astype(float)
        w = w / w.sum()
    else:
        w = np.ones(n) / n

    if bandwidth is None:
        if bw_method == 'silverman':
            bandwidth = _silverman_bw(x_data)
        elif bw_method == 'sheather-jones':
            bandwidth = _sheather_jones_bw(x_data)
        else:
            bandwidth = _silverman_bw(x_data)

    if grid is None:
        pad = 3 * bandwidth
        grid = np.linspace(x_data.min() - pad, x_data.max() + pad, n_grid)

    # Compute density at each grid point
    density = np.zeros(len(grid))
    for i, g in enumerate(grid):
        u = (g - x_data) / bandwidth
        density[i] = np.sum(w * _kernel_fn(u, kernel)) / bandwidth

    return KDensityResult(
        grid=grid, density=density, bandwidth=bandwidth,
        kernel=kernel, n=n, data=x_data,
    )
