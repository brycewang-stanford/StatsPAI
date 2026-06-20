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

from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility


class KDensityResult:
    """Results from kernel density estimation.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(3)
    >>> df = pd.DataFrame({"income": rng.normal(10.0, 2.0, 500)})
    >>> res = sp.kdensity(df, x="income", kernel="gaussian")
    >>> type(res).__name__
    'KDensityResult'
    >>> bool(res.bandwidth > 0)
    True
    >>> bool((res.density >= 0).all())
    True
    """

    def __init__(
        self,
        grid: np.ndarray,
        density: np.ndarray,
        bandwidth: float,
        kernel: str,
        n: int,
        data: np.ndarray,
    ) -> None:
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

    def plot(
        self,
        ax: Any = None,
        hist: bool = False,
        rug: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot the kernel density estimate."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        if hist:
            ax.hist(
                self._data,
                bins="auto",
                density=True,
                alpha=0.3,
                color="gray",
                label="Histogram",
            )

        ax.plot(
            self.grid,
            self.density,
            color="steelblue",
            lw=2,
            label=f"KDE (bw={self.bandwidth:.3f})",
        )

        if rug:
            ax.plot(
                self._data,
                np.zeros_like(self._data),
                "|",
                color="gray",
                alpha=0.3,
                ms=10,
            )

        ax.set_xlabel(kwargs.get("xlabel", "X"))
        ax.set_ylabel(kwargs.get("ylabel", "Density"))
        ax.set_title(kwargs.get("title", "Kernel Density Estimation"))
        ax.legend()
        return ax


def _kernel_fn(u: np.ndarray, kernel: str = "gaussian") -> np.ndarray:
    """Evaluate kernel function at u."""
    if kernel == "gaussian":
        return np.asarray(stats.norm.pdf(u), dtype=float)
    elif kernel == "epanechnikov":
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
    elif kernel == "uniform":
        return np.where(np.abs(u) <= 1, 0.5, 0.0)
    elif kernel == "triangular":
        return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)
    elif kernel == "biweight":
        return np.where(np.abs(u) <= 1, (15 / 16) * (1 - u**2) ** 2, 0.0)
    elif kernel == "cosine":
        return np.where(np.abs(u) <= 1, (np.pi / 4) * np.cos(np.pi * u / 2), 0.0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def _silverman_bw(x: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth."""
    n = len(x)
    iqr_val = float(stats.iqr(x))
    std_val = float(np.std(x, ddof=1))
    sigma = min(std_val, iqr_val / 1.349) if iqr_val > 0 else std_val
    if sigma == 0:
        sigma = 1.0  # constant data fallback
    return float(0.9 * sigma * n ** (-1 / 5))


def _sheather_jones_bw(x: np.ndarray) -> float:
    """Sheather-Jones (1991) plug-in bandwidth (simplified)."""
    n = len(x)
    sigma = float(np.std(x, ddof=1))
    iqr = float(stats.iqr(x))
    a = min(sigma, iqr / 1.349) if iqr > 0 else sigma

    # Use Silverman as fallback
    if a == 0:
        return _silverman_bw(x)

    return float(0.9 * a * n ** (-1 / 5))


def kdensity(
    data: Optional[pd.DataFrame] = None,
    x: Optional[str] = None,
    bandwidth: Optional[float] = None,
    kernel: str = "gaussian",
    bw_method: str = "silverman",
    n_grid: int = 512,
    grid: Optional[np.ndarray] = None,
    weights: Optional[str] = None,
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
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(
    ...     {'income': rng.lognormal(mean=10, sigma=0.5, size=500)})
    >>> result = sp.kdensity(df, x='income')
    >>> type(result).__name__
    'KDensityResult'
    >>> bool(result.bandwidth > 0)
    True
    >>> isinstance(result.summary(), str)
    True
    >>> ax = result.plot(hist=True)  # doctest: +SKIP
    """
    if data is None:
        raise ValueError("data is required")
    if x is None:
        raise MethodIncompatibility(
            "x is required",
            recovery_hint="Pass the column name to estimate with x='column'.",
        )
    if n_grid <= 0:
        raise ValueError("n_grid must be positive")
    if bandwidth is not None and (bandwidth <= 0 or not np.isfinite(bandwidth)):
        raise ValueError("bandwidth must be a positive finite number")

    x_data = data[x].dropna().values.astype(float)
    n = len(x_data)
    if n == 0:
        raise ValueError(
            "data has no finite observations after dropping missing values"
        )

    if weights is not None:
        w = data.loc[data[x].notna(), weights].values.astype(float)
        total_weight = w.sum()
        if (
            len(w) != n
            or not np.all(np.isfinite(w))
            or np.any(w < 0)
            or total_weight <= 0
        ):
            raise ValueError(
                "weights must be finite, non-negative, and sum to a positive value"
            )
        w = w / total_weight
    else:
        w = np.ones(n) / n

    if bandwidth is None:
        if bw_method == "silverman":
            bandwidth = _silverman_bw(x_data)
        elif bw_method == "sheather-jones":
            bandwidth = _sheather_jones_bw(x_data)
        else:
            raise ValueError("bw_method must be one of 'silverman' or 'sheather-jones'")
    assert bandwidth is not None

    if grid is None:
        pad = 3 * bandwidth
        grid = np.linspace(x_data.min() - pad, x_data.max() + pad, n_grid)
    else:
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 1 or len(grid) == 0 or not np.all(np.isfinite(grid)):
            raise ValueError("grid must be a non-empty one-dimensional finite array")

    # Compute density in grid blocks. This keeps the public result identical to
    # the direct kernel sum while avoiding one Python loop per grid point.
    density = np.empty(len(grid))
    block_size = max(1, min(len(grid), 1_000_000 // max(n, 1)))
    for start in range(0, len(grid), block_size):
        stop = min(start + block_size, len(grid))
        u = (grid[start:stop, None] - x_data[None, :]) / bandwidth
        density[start:stop] = (_kernel_fn(u, kernel) @ w) / bandwidth

    return KDensityResult(
        grid=grid,
        density=density,
        bandwidth=bandwidth,
        kernel=kernel,
        n=n,
        data=x_data,
    )
