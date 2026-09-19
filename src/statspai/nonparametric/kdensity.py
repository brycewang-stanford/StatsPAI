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

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility


class KDensityResult(ResultProtocolMixin):
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
    """Silverman's rule of thumb as R ``bw.nrd0``.

    ``0.9 * min(sd, IQR / 1.34) * n^(-1/5)`` with the type-7 (linear)
    sample quantiles and R's fallbacks when the scale is zero.
    """
    n = len(x)
    hi = float(np.std(x, ddof=1)) if n > 1 else 0.0
    q75, q25 = np.percentile(x, [75, 25])
    lo = min(hi, float(q75 - q25) / 1.34)
    if not lo > 0:
        lo = hi or abs(float(x[0])) or 1.0
    return float(0.9 * lo * n ** (-1 / 5))


def _stata_percentile(xs: np.ndarray, p: float) -> float:
    """Percentile as Stata's ``summarize, detail`` (unweighted).

    With ``P = n p / 100``: the average of the P-th and (P+1)-th order
    statistics when P is an integer, else the ceil(P)-th.
    """
    n = len(xs)
    P = n * p / 100.0
    k = int(np.floor(P))
    if P == k:
        return float((xs[k - 1] + xs[k]) / 2.0) if k >= 1 else float(xs[0])
    return float(xs[min(k, n - 1)])


def _stata_bw(x: np.ndarray) -> float:
    """Stata ``kdensity``'s default width (unweighted).

    ``0.9 * min(sd, (p75 - p25) / 1.349) / N^(1/5)`` with Stata's
    percentile definition; ``sd`` when that minimum is not positive.
    """
    xs = np.sort(x)
    sd = float(np.std(x, ddof=1))
    m = min(sd, (_stata_percentile(xs, 75) - _stata_percentile(xs, 25)) / 1.349)
    if m <= 0:
        m = sd
    return float(0.9 * m / len(x) ** 0.2)


def _sj_phi(x: np.ndarray, h: float, order: int) -> float:
    """Gaussian-kernel estimate of the density functional psi_order.

    ``sum_{i,j} phi^(order)((x_i - x_j)/h) / (n (n-1) h^(order+1))``, the
    diagonal included, as in Sheather & Jones (1991) and R ``bw.SJ``.
    """
    n = len(x)
    total = 0.0
    step = max(1, 2_000_000 // max(n, 1))
    for start in range(0, n, step):
        u = (x[start : start + step, None] - x[None, :]) / h
        u2 = u * u
        e = np.exp(-0.5 * u2)
        if order == 4:
            total += float(np.sum((u2 * u2 - 6.0 * u2 + 3.0) * e))
        else:
            total += float(np.sum((u2**3 - 15.0 * u2 * u2 + 45.0 * u2 - 15.0) * e))
    return float(total / (n * (n - 1) * h ** (order + 1) * np.sqrt(2.0 * np.pi)))


def _sheather_jones_bw(x: np.ndarray) -> float:
    """Sheather-Jones (1991) solve-the-equation bandwidth.

    The estimator R ``bw.SJ(method="ste")`` computes, evaluated on the
    exact pairwise differences (R bins them into ``nb`` classes) and with
    the fixed point solved to machine precision (R's ``uniroot`` stops at
    ``tol = 0.1 * lower``).
    """
    from scipy.optimize import brentq

    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("sheather-jones bandwidth needs at least 2 points")
    q75, q25 = np.percentile(x, [75, 25])
    scale = min(float(np.std(x, ddof=1)), float(q75 - q25) / 1.349)
    if not scale > 0:
        raise ValueError("sheather-jones bandwidth needs a positive scale")
    a = 1.24 * scale * n ** (-1 / 7)
    b = 1.23 * scale * n ** (-1 / 9)
    c1 = 1.0 / (2.0 * np.sqrt(np.pi) * n)
    td = -_sj_phi(x, b, 6)
    if not (np.isfinite(td) and td > 0):
        raise ValueError("sample is too sparse to find TD")
    alph2 = 1.357 * (_sj_phi(x, a, 4) / td) ** (1 / 7)

    def fsd(h: float) -> float:
        return float((c1 / _sj_phi(x, alph2 * h ** (5 / 7), 4)) ** 0.2 - h)

    hmax = 1.144 * scale * n ** (-1 / 5)
    lower, upper = 0.1 * hmax, hmax
    for itry in range(100):
        if fsd(lower) * fsd(upper) <= 0:
            break
        if itry % 2 == 0:
            upper *= 1.2
        else:
            lower /= 1.2
    else:
        raise ValueError("no Sheather-Jones solution in the search interval")
    return float(brentq(fsd, lower, upper, xtol=1e-14, rtol=4 * np.finfo(float).eps))


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
        Bandwidth selection when ``bandwidth`` is None:

        * ``'silverman'`` (alias ``'nrd0'``): R ``bw.nrd0``,
          ``0.9 min(sd, IQR/1.34) n^(-1/5)``.
        * ``'stata'``: Stata ``kdensity``'s default,
          ``0.9 min(sd, (p75-p25)/1.349) n^(-1/5)`` with Stata's
          percentiles (unweighted only).
        * ``'sheather-jones'``: Sheather-Jones solve-the-equation
          (R ``bw.SJ``, computed without binning).

        Each rule is applied as-is whatever the kernel. Note that the
        ``'epanechnikov'`` kernel here has support [-1, 1] (Stata's
        ``epan2``), not the unit-variance form of R ``density`` and Stata's
        default ``epanechnikov``.
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
        if bw_method in ("silverman", "nrd0"):
            bandwidth = _silverman_bw(x_data)
        elif bw_method == "stata":
            if weights is not None:
                raise NotImplementedError(
                    "bw_method='stata' is implemented for unweighted data only"
                )
            bandwidth = _stata_bw(x_data)
        elif bw_method == "sheather-jones":
            bandwidth = _sheather_jones_bw(x_data)
        else:
            raise ValueError(
                "bw_method must be one of 'silverman', 'nrd0', 'stata' or "
                "'sheather-jones'"
            )
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
