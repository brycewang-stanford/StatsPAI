"""
Power and sample size calculations for causal inference and econometric
designs.

Provides power analysis for designs that are missing from the Python ecosystem:
DID, RD, IV, cluster RCT, and standard RCT/OLS — with support for power curves,
minimum detectable effect (MDE), and sample size solving.

References:
- Burlig, Preonas & Woerman (2020): "Panel data and experimental design"
- Cattaneo, Titiunik & Vazquez-Bare (2019): "Power calculations for RD designs"
- Stock & Yogo (2005): "Testing for weak instruments"
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.stats import norm

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ConvergenceFailure, MethodIncompatibility

__all__ = [
    "power",
    "PowerResult",
    "power_rct",
    "power_did",
    "power_rd",
    "power_iv",
    "power_cluster_rct",
    "power_ols",
    "mde",
]

# ---------------------------------------------------------------------------
# Design registry
# ---------------------------------------------------------------------------

_DESIGN_FUNCTIONS: Dict[str, Callable[..., "PowerResult"]] = {}


# ---------------------------------------------------------------------------
# PowerResult
# ---------------------------------------------------------------------------


class PowerResult(ResultProtocolMixin):
    """Container for power analysis results.

    Attributes
    ----------
    power : float or np.ndarray
        Computed power value(s).
    n : int, float, or np.ndarray
        Sample size(s) used.
    effect_size : float or np.ndarray
        Effect size(s) used.
    design : str
        Name of the research design.
    params : dict
        All parameters passed to the power function.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.power_rct(n=500, effect_size=0.3)
    >>> isinstance(res, sp.PowerResult)
    True
    >>> res.design
    'rct'
    >>> round(float(res.power), 4)
    0.9184
    """

    def __init__(
        self,
        power_val: Any,
        n: Any,
        effect_size: Any,
        design: str,
        params: Dict[str, Any],
    ) -> None:
        self.power = power_val
        self.n = n
        self.effect_size = effect_size
        self.design = design
        self.params = params

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines: list[str] = []
        lines.append("=" * 56)
        lines.append(f"  Power Analysis — {self.design.upper()} design")
        lines.append("=" * 56)

        # Scalar result
        if np.ndim(self.power) == 0:
            lines.append(f"  Power        : {float(self.power):.4f}")
            lines.append(f"  Sample size  : {self._fmt_n(self.n)}")
            lines.append(f"  Effect size  : {float(self.effect_size):.4f}")
        else:
            arr = np.asarray(self.power)
            lines.append(f"  Power range  : [{arr.min():.4f}, {arr.max():.4f}]")
            n_arr = np.asarray(self.n)
            lines.append(
                f"  N range      : "
                f"[{self._fmt_n(n_arr.min())}, "
                f"{self._fmt_n(n_arr.max())}]"
            )
            es_arr = np.asarray(self.effect_size)
            if es_arr.ndim == 0:
                lines.append(f"  Effect size  : {float(es_arr):.4f}")
            else:
                lines.append(
                    f"  Effect range : " f"[{es_arr.min():.4f}, {es_arr.max():.4f}]"
                )

        lines.append(f"  Alpha        : {self.params.get('alpha', 0.05)}")

        # Design-specific extras
        extras = {
            k: v
            for k, v in self.params.items()
            if k not in ("n", "effect_size", "alpha", "power", "design")
        }
        if extras:
            lines.append("-" * 56)
            for k, v in extras.items():
                label = k.replace("_", " ").title()
                lines.append(f"  {label:20s}: {v}")

        lines.append("=" * 56)
        return "\n".join(lines)

    def __repr__(self) -> str:
        if np.ndim(self.power) == 0:
            return (
                f"PowerResult(design={self.design!r}, "
                f"power={float(self.power):.4f}, "
                f"n={self._fmt_n(self.n)}, "
                f"effect_size={float(self.effect_size):.4f})"
            )
        arr = np.asarray(self.power)
        return (
            f"PowerResult(design={self.design!r}, "
            f"power=[{arr.min():.4f}..{arr.max():.4f}], "
            f"len={len(arr)})"
        )

    def _repr_html_(self) -> str:
        """Rich display in Jupyter notebooks."""
        if np.ndim(self.power) == 0:
            rows = [
                ("Design", self.design.upper()),
                ("Power", f"{float(self.power):.4f}"),
                ("Sample size", self._fmt_n(self.n)),
                ("Effect size", f"{float(self.effect_size):.4f}"),
                ("Alpha", self.params.get("alpha", 0.05)),
            ]
        else:
            arr = np.asarray(self.power)
            n_arr = np.asarray(self.n)
            rows = [
                ("Design", self.design.upper()),
                ("Power range", f"[{arr.min():.4f}, {arr.max():.4f}]"),
                (
                    "N range",
                    (f"[{self._fmt_n(n_arr.min())}, " f"{self._fmt_n(n_arr.max())}]"),
                ),
                ("Alpha", self.params.get("alpha", 0.05)),
            ]

        html = (
            '<div style="font-family: monospace; padding:8px;">'
            '<table style="border-collapse:collapse;">'
            '<caption style="font-weight:bold; font-size:1.1em; '
            'padding-bottom:6px;">'
            f"Power Analysis &mdash; {self.design.upper()}</caption>"
        )
        for label, val in rows:
            html += (
                '<tr><td style="padding:2px 12px 2px 0; '
                f'font-weight:bold;">{label}</td>'
                f'<td style="padding:2px 0;">{val}</td></tr>'
            )
        html += "</table></div>"
        return html

    @staticmethod
    def _fmt_n(val: Any) -> str:
        """Format sample size as int when possible."""
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        fval = float(val)
        if fval == int(fval):
            return str(int(fval))
        return f"{fval:.1f}"

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        ax: Any = None,
        figsize: tuple[float, float] = (8, 5),
        **kwargs: Any,
    ) -> Any:
        """Plot power curve.

        Works when *n* or *effect_size* was supplied as an array / range.

        Parameters
        ----------
        ax : matplotlib Axes, optional
        figsize : tuple
        **kwargs : passed to ``ax.plot``

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if np.ndim(self.power) == 0:
            raise ValueError(
                "plot() requires power computed over a range of n or "
                "effect_size values. "
                "Pass n=range(...) or effect_size=np.linspace(...) to power()."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        power_arr = np.asarray(self.power)
        n_arr = np.asarray(self.n)
        es_arr = np.asarray(self.effect_size)

        # Determine x-axis: whichever was varied
        if n_arr.ndim > 0 and n_arr.size == power_arr.size:
            x = n_arr
            xlabel = "Sample size (N)"
        elif es_arr.ndim > 0 and es_arr.size == power_arr.size:
            x = es_arr
            xlabel = "Effect size"
        else:
            x = np.arange(len(power_arr))
            xlabel = "Index"

        plot_kwargs: Dict[str, Any] = dict(linewidth=2, color="#2563eb")
        plot_kwargs.update(kwargs)
        ax.plot(x, power_arr, **plot_kwargs)
        ax.axhline(
            0.8,
            linestyle="--",
            color="#9ca3af",
            linewidth=1,
            label="Power = 0.80",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Power")
        ax.set_title(f"Power Curve — {self.design.upper()} design")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return ax


# ---------------------------------------------------------------------------
# Helper: vectorise n / effect_size inputs
# ---------------------------------------------------------------------------


def _to_array(val: Any) -> np.ndarray:
    """Convert scalar, list, or range to numpy array."""
    if isinstance(val, range):
        return np.array(list(val), dtype=float)
    arr = np.asarray(val, dtype=float)
    return arr


# ---------------------------------------------------------------------------
# Design-specific power functions
# ---------------------------------------------------------------------------


def power_rct(
    n: Any,
    effect_size: Any,
    alpha: float = 0.05,
    ratio: float = 1.0,
    sigma: float = 1.0,
) -> PowerResult:
    """Power for a two-arm Randomised Controlled Trial.

    Parameters
    ----------
    n : int or array-like
        Total sample size (treatment + control).
    effect_size : float or array-like
        Standardised effect size (delta / sigma).
    alpha : float
        Significance level (two-sided).
    ratio : float
        Treatment / control allocation ratio (1 = equal allocation).
    sigma : float
        Outcome standard deviation (default 1 for standardised effect).

    Returns
    -------
    PowerResult

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.power_rct(n=500, effect_size=0.3)
    >>> round(float(res.power), 4)
    0.9184
    """
    n_arr = _to_array(n)
    es_arr = _to_array(effect_size)
    z_alpha = norm.ppf(1 - alpha / 2)

    # Proportion treated
    p = ratio / (1 + ratio)
    se = sigma / np.sqrt(n_arr * p * (1 - p))
    pwr = norm.cdf(np.abs(es_arr) * sigma / se - z_alpha)

    return PowerResult(
        power_val=float(pwr) if pwr.ndim == 0 else pwr,
        n=n,
        effect_size=effect_size,
        design="rct",
        params=dict(
            n=n, effect_size=effect_size, alpha=alpha, ratio=ratio, sigma=sigma
        ),
    )


def power_did(
    n: Any,
    effect_size: Any,
    n_periods: int,
    n_treated_periods: int,
    rho: float = 0.5,
    alpha: float = 0.05,
    sigma: float = 1.0,
    prop_treat: float = 0.5,
) -> PowerResult:
    r"""Power for Difference-in-Differences.

    Two groups observed for ``n_periods`` periods, the last
    ``n_treated_periods`` of them after treatment, with errors that follow a
    stationary AR(1) process within unit: the serially correlated setting of
    Burlig, Preonas & Woerman (2020). The estimator is the difference in
    post-minus-pre mean changes between the groups, which is the two-way
    fixed-effects coefficient on a balanced panel. Its variance is exact
    under the stated error process:

    .. math::

        \mathrm{Var}(\hat\tau) = \frac{\sigma^2\, c' R c}{p(1-p)\, n},

    where ``c`` puts ``1 / n_treated_periods`` on each post period and
    ``-1 / (n_periods - n_treated_periods)`` on each pre period, and ``R``
    is the AR(1) correlation matrix ``rho ** |s - t|``.

    Parameters
    ----------
    n : int or array-like
        Total number of units (treated + control).
    effect_size : float or array-like
        Effect in units of ``sigma``.
    n_periods : int
        Total number of time periods.
    n_treated_periods : int
        Number of post-treatment periods.
    rho : float
        First-order autocorrelation of the errors within a unit. Positive
        autocorrelation lowers the variance of a pre/post change when the
        panel is short and raises it as the panel lengthens.
    alpha : float
        Significance level (two-sided).
    sigma : float
        Standard deviation of the error in one unit-period.
    prop_treat : float, default 0.5
        Share of units that are treated.

    Returns
    -------
    PowerResult

    Notes
    -----
    Until the 2026-10 fix the variance was
    ``sigma^2 (1 + (T - 1) rho) T / (n T_pre T_post)``: a cluster design
    effect, which does not apply to a within-unit change, and without the
    ``1 / (p (1 - p))`` factor of a two-group comparison. With ``rho = 0``
    it overstated power (the standard error was half its value); with a
    large ``rho`` it understated it. For 200 units, ten periods of which
    five treated, ``rho = 0.5`` and an effect of 0.2 it returned 0.48 where
    simulation gives 0.38.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.power_did(n=1000, effect_size=0.1, n_periods=10,
    ...                    n_treated_periods=5)
    >>> round(float(res.power), 4)
    0.4516
    """
    n_arr = _to_array(n)
    es_arr = _to_array(effect_size)
    z_alpha = norm.ppf(1 - alpha / 2)

    T = int(n_periods)
    T_post = int(n_treated_periods)
    T_pre = T - T_post
    if T_pre < 1 or T_post < 1:
        raise MethodIncompatibility(
            "power_did needs at least one pre-treatment and one "
            f"post-treatment period (n_periods={n_periods}, "
            f"n_treated_periods={n_treated_periods})."
        )
    if not 0.0 < prop_treat < 1.0:
        raise MethodIncompatibility(f"prop_treat must be in (0, 1), got {prop_treat}.")
    if not -1.0 < rho < 1.0:
        raise MethodIncompatibility(f"rho must be in (-1, 1), got {rho}.")

    # Variance of (post mean - pre mean) for one unit under AR(1) errors.
    c = np.concatenate([np.full(T_pre, -1.0 / T_pre), np.full(T_post, 1.0 / T_post)])
    lags = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    change_var = float(c @ (float(rho) ** lags) @ c)
    se = sigma * np.sqrt(change_var / (prop_treat * (1.0 - prop_treat) * n_arr))
    pwr = norm.cdf(np.abs(es_arr) * sigma / se - z_alpha)

    return PowerResult(
        power_val=float(pwr) if pwr.ndim == 0 else pwr,
        n=n,
        effect_size=effect_size,
        design="did",
        params=dict(
            n=n,
            effect_size=effect_size,
            alpha=alpha,
            n_periods=n_periods,
            n_treated_periods=n_treated_periods,
            rho=rho,
            sigma=sigma,
            prop_treat=prop_treat,
        ),
    )


#: Variance constant of the local-linear estimator at a boundary,
#: ``(mu2^2 nu0 - 2 mu1 mu2 nu1 + mu1^2 nu2) / (mu0 mu2 - mu1^2)^2`` with
#: ``mu_j = int_0^1 u^j K(u) du`` and ``nu_j = int_0^1 u^j K(u)^2 du``.
_BOUNDARY_LL_CONSTANT = {
    "uniform": 4.0,
    "triangular": 24.0 / 5.0,
    "epanechnikov": 56832.0 / 12635.0,
}


def power_rd(
    n: Any,
    effect_size: Any,
    bandwidth: Optional[float] = None,
    kernel: str = "triangular",
    density_at_cutoff: float = 1.0,
    alpha: float = 0.05,
    sigma: float = 1.0,
) -> PowerResult:
    r"""Power for Regression Discontinuity designs.

    Power of the conventional local-linear t test at a fixed bandwidth.
    Each side's boundary intercept has asymptotic variance
    ``sigma^2 C_K / (n h f)``, where ``n h f`` is the expected number of
    observations within ``h`` of the cutoff on that side and ``C_K`` is the
    kernel's boundary constant (4 for uniform, 4.8 for triangular, about
    4.5 for Epanechnikov), so

    .. math::

        \mathrm{se}(\hat\tau) = \sigma \sqrt{\frac{2\, C_K}{n\, h\, f}}.

    The robust bias-corrected test that ``sp.rdrobust`` reports by default
    is less powerful than this; for a design calculation anchored on pilot
    data use ``sp.rdpower`` (Cattaneo, Titiunik & Vazquez-Bare 2019).

    Parameters
    ----------
    n : int or array-like
        Total sample size in the data.
    effect_size : float or array-like
        Effect at the cutoff in units of ``sigma``.
    bandwidth : float or None
        Bandwidth on each side of the cutoff, in units of the running
        variable. If *None*, defaults to 0.5 (the whole support of a
        running variable on [0, 1] with the cutoff at its midpoint).
    kernel : {'triangular', 'uniform', 'epanechnikov'}
        Kernel used for local weighting.
    density_at_cutoff : float
        Density of the running variable at the cutoff
        (default 1.0 for a uniform running variable on [0,1]).
    alpha : float
        Significance level.
    sigma : float
        Conditional outcome std dev near the cutoff.

    Returns
    -------
    PowerResult

    Notes
    -----
    Until the 2026-10 fix the standard error was that of a difference
    between two local means with an efficiency discount, about half the
    local-linear value: 2,000 observations and an effect of 0.25 returned
    0.93 where simulation of the local-linear test gives 0.74.

    References
    ----------
    cattaneo2019power

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.power_rd(n=2000, effect_size=0.25)
    >>> round(float(res.power), 4)
    0.7229
    """
    n_arr = _to_array(n)
    es_arr = _to_array(effect_size)
    z_alpha = norm.ppf(1 - alpha / 2)

    if bandwidth is None:
        bandwidth = 0.5
    if kernel not in _BOUNDARY_LL_CONSTANT:
        raise MethodIncompatibility(
            f"kernel must be one of {sorted(_BOUNDARY_LL_CONSTANT)}, got {kernel!r}."
        )
    c_k = _BOUNDARY_LL_CONSTANT[kernel]

    # Expected observations within the bandwidth on one side of the cutoff.
    n_side = np.maximum(n_arr * bandwidth * density_at_cutoff, 1.0)
    se = sigma * np.sqrt(2.0 * c_k / n_side)
    pwr = norm.cdf(np.abs(es_arr) * sigma / se - z_alpha)

    return PowerResult(
        power_val=float(pwr) if pwr.ndim == 0 else pwr,
        n=n,
        effect_size=effect_size,
        design="rd",
        params=dict(
            n=n,
            effect_size=effect_size,
            alpha=alpha,
            bandwidth=bandwidth,
            kernel=kernel,
            density_at_cutoff=density_at_cutoff,
            sigma=sigma,
        ),
    )


def power_iv(
    n: Any,
    effect_size: Any,
    first_stage_f: Optional[float] = None,
    r2_z: Optional[float] = None,
    alpha: float = 0.05,
    sigma: float = 1.0,
) -> PowerResult:
    """Power for Instrumental Variables / 2SLS estimation.

    The 2SLS estimator has asymptotic variance ``sigma^2 / (n R2_z)`` for a
    regressor with unit variance, where ``R2_z`` is the share of the
    endogenous regressor's variance the instruments explain: only that
    share identifies the effect. Power is therefore that of OLS on a sample
    of ``n R2_z`` observations.

    Parameters
    ----------
    n : int or array-like
        Sample size.
    effect_size : float or array-like
        Effect of a one-standard-deviation change in the endogenous
        variable, in units of ``sigma``.
    first_stage_f : float or None
        First-stage F-statistic of a single instrument, converted with
        ``R2_z = F / (F + n)``. With several instruments pass ``r2_z``.
    r2_z : float or None
        Partial R-squared of the first-stage regression.  Alternative to
        *first_stage_f*; if both are given, *first_stage_f* takes precedence.
    alpha : float
        Significance level.
    sigma : float
        Error standard deviation.

    Returns
    -------
    PowerResult

    Notes
    -----
    With neither ``first_stage_f`` nor ``r2_z`` the result is the OLS
    benchmark. The normal approximation is unreliable for a weak first
    stage (F below about 10), where 2SLS is biased and its t test
    over-rejects.

    Until the 2026-10 fix the result was the OLS power multiplied by
    ``F / (F + 1)``, which barely moves for any F above 10 however little
    of the regressor the instrument explains: 400 observations, an effect
    of 0.2 and a first-stage F of 37 returned 0.95 where simulation gives
    0.21.

    References
    ----------
    stock2005testing

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.power_iv(n=1000, effect_size=0.2, first_stage_f=20)
    >>> round(float(res.power), 4)
    0.1413
    """
    n_arr = _to_array(n)
    es_arr = _to_array(effect_size)
    z_alpha = norm.ppf(1 - alpha / 2)

    r2: Any
    if first_stage_f is not None:
        if first_stage_f <= 0:
            raise MethodIncompatibility(
                f"first_stage_f must be positive, got {first_stage_f}."
            )
        r2 = first_stage_f / (first_stage_f + n_arr)
    elif r2_z is not None:
        if not 0.0 < r2_z <= 1.0:
            raise MethodIncompatibility(f"r2_z must be in (0, 1], got {r2_z}.")
        r2 = r2_z
    else:
        r2 = 1.0  # no first-stage information: the OLS benchmark

    se = sigma / np.sqrt(n_arr * r2)
    pwr = norm.cdf(np.abs(es_arr) * sigma / se - z_alpha)

    return PowerResult(
        power_val=float(pwr) if pwr.ndim == 0 else pwr,
        n=n,
        effect_size=effect_size,
        design="iv",
        params=dict(
            n=n,
            effect_size=effect_size,
            alpha=alpha,
            first_stage_f=first_stage_f,
            r2_z=r2_z,
            sigma=sigma,
        ),
    )


def power_cluster_rct(
    n_clusters: Any,
    cluster_size: float,
    effect_size: Any,
    icc: float,
    alpha: float = 0.05,
    sigma: float = 1.0,
) -> PowerResult:
    """Power for a Cluster-Randomised Controlled Trial.

    Parameters
    ----------
    n_clusters : int or array-like
        Total number of clusters (treatment + control).
    cluster_size : int or float
        Average number of individuals per cluster.
    effect_size : float or array-like
        Standardised effect size.
    icc : float
        Intra-cluster correlation coefficient.
    alpha : float
        Significance level.
    sigma : float
        Individual-level outcome standard deviation.

    Returns
    -------
    PowerResult

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.power_cluster_rct(n_clusters=40, cluster_size=30,
    ...                            effect_size=0.3, icc=0.05)
    >>> round(float(res.power), 4)
    0.913
    """
    nc_arr = _to_array(n_clusters)
    es_arr = _to_array(effect_size)
    z_alpha = norm.ppf(1 - alpha / 2)

    m = float(cluster_size)
    design_effect = 1 + (m - 1) * icc
    n_total = nc_arr * m
    n_eff = n_total / design_effect

    # Equal allocation: p = 0.5
    se = sigma / np.sqrt(n_eff * 0.25)
    pwr = norm.cdf(np.abs(es_arr) * sigma / se - z_alpha)

    return PowerResult(
        power_val=float(pwr) if pwr.ndim == 0 else pwr,
        n=n_clusters,  # report cluster count as the "n"
        effect_size=effect_size,
        design="cluster_rct",
        params=dict(
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            effect_size=effect_size,
            icc=icc,
            alpha=alpha,
            sigma=sigma,
        ),
    )


def power_ols(
    n: Any,
    effect_size: Any,
    n_covariates: int = 0,
    r2_other: float = 0.0,
    alpha: float = 0.05,
    sigma: float = 1.0,
) -> PowerResult:
    """Power for OLS regression (single coefficient of interest).

    Parameters
    ----------
    n : int or array-like
        Sample size.
    effect_size : float or array-like
        Standardised effect of the variable of interest.
    n_covariates : int
        Number of other covariates in the model.
    r2_other : float
        R-squared attributable to other covariates (reduces residual
        variance and thus improves power).
    alpha : float
        Significance level.
    sigma : float
        Outcome standard deviation.

    Returns
    -------
    PowerResult

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.power_ols(n=500, effect_size=0.2)
    >>> round(float(res.power), 4)
    0.9939
    """
    n_arr = _to_array(n)
    es_arr = _to_array(effect_size)
    z_alpha = norm.ppf(1 - alpha / 2)

    # Residual variance after partialling out other covariates
    residual_factor = np.sqrt(1 - r2_other)
    # Degrees-of-freedom adjustment
    df_adj = np.maximum(n_arr - n_covariates - 1, 1)
    se = sigma * residual_factor / np.sqrt(df_adj)

    pwr = norm.cdf(np.abs(es_arr) * sigma / se - z_alpha)

    return PowerResult(
        power_val=float(pwr) if pwr.ndim == 0 else pwr,
        n=n,
        effect_size=effect_size,
        design="ols",
        params=dict(
            n=n,
            effect_size=effect_size,
            alpha=alpha,
            n_covariates=n_covariates,
            r2_other=r2_other,
            sigma=sigma,
        ),
    )


# ---------------------------------------------------------------------------
# Design registry
# ---------------------------------------------------------------------------

_DESIGN_FUNCTIONS = {
    "rct": power_rct,
    "did": power_did,
    "rd": power_rd,
    "iv": power_iv,
    "cluster_rct": power_cluster_rct,
    "ols": power_ols,
}


# ---------------------------------------------------------------------------
# Main dispatcher: power()
# ---------------------------------------------------------------------------


def power(
    design: str,
    *,
    n: Any = None,
    effect_size: Any = None,
    power_target: Optional[float] = None,
    **kwargs: Any,
) -> PowerResult:
    """Compute statistical power (or solve for sample size) for a causal
    inference / econometric design.

    Parameters
    ----------
    design : str
        One of ``'rct'``, ``'did'``, ``'rd'``, ``'iv'``,
        ``'cluster_rct'``, ``'ols'``.
    n : int, array-like, range, or None
        Sample size.  For cluster_rct this is *n_clusters*.
        Pass *None* to solve for the minimum n that achieves
        *power_target*.
    effect_size : float, array-like, or None
        Standardised effect size.  Pass *None* to solve for MDE
        (use :func:`mde` for a cleaner interface).
    power_target : float or None
        Target power (e.g. 0.80).  When *n* is None, the function
        performs a binary search for the minimum n that achieves this
        power.
    **kwargs
        Design-specific parameters forwarded to the underlying
        power function (e.g. ``n_periods``, ``icc``, ``bandwidth``).

    Returns
    -------
    PowerResult

    Examples
    --------
    >>> import statspai as sp
    >>> sp.power("did", n=1000, effect_size=0.1,
    ...          n_periods=10, n_treated_periods=5)
    PowerResult(design='did', power=0.4516, n=1000, effect_size=0.1000)
    >>> sp.power("did", power=0.8, effect_size=0.1,
    ...          n_periods=10, n_treated_periods=5)
    PowerResult(design='did', power=0.8000, n=2323, effect_size=0.1000)
    >>> result = sp.power("did", n=range(100, 2000, 100), effect_size=0.1,
    ...                   n_periods=10, n_treated_periods=5)
    >>> result
    PowerResult(design='did', power=[0.0840..0.7170], len=19)
    >>> ax = result.plot()
    """
    # Accept 'power' kwarg as alias for power_target (convenience API)
    if power_target is None and "power" in kwargs:
        power_target = kwargs.pop("power")

    design = design.lower().strip()
    if design not in _DESIGN_FUNCTIONS:
        supported = ", ".join(sorted(_DESIGN_FUNCTIONS))
        raise ValueError(f"Unknown design {design!r}. Supported designs: {supported}")

    func = _DESIGN_FUNCTIONS[design]

    # ------------------------------------------------------------------
    # Case 1: Solve for sample size
    # ------------------------------------------------------------------
    if n is None:
        if power_target is None:
            raise ValueError(
                "When n is None you must specify power_target (or power=) "
                "so that the required sample size can be solved."
            )
        if effect_size is None:
            raise ValueError("When solving for n, effect_size must be specified.")
        return _solve_for_n(func, effect_size, power_target, design, kwargs)

    # ------------------------------------------------------------------
    # Case 2: Solve for MDE
    # ------------------------------------------------------------------
    if effect_size is None:
        if power_target is None:
            raise ValueError(
                "When effect_size is None you must specify "
                "power_target (or power=) "
                "so that the MDE can be solved."
            )
        return _solve_for_mde(func, n, power_target, design, kwargs)

    # ------------------------------------------------------------------
    # Case 3: Compute power directly
    # ------------------------------------------------------------------
    # For cluster_rct, the first positional arg is n_clusters
    if design == "cluster_rct":
        return func(n_clusters=n, effect_size=effect_size, **kwargs)
    return func(n=n, effect_size=effect_size, **kwargs)


# ---------------------------------------------------------------------------
# MDE convenience function
# ---------------------------------------------------------------------------


def mde(
    design: str,
    *,
    n: Any = None,
    power_target: float = 0.8,
    **kwargs: Any,
) -> PowerResult:
    """Compute the Minimum Detectable Effect (MDE) for a given design.

    Inverts the power function to find the smallest effect size that
    achieves *power_target* at the given sample size.

    Parameters
    ----------
    design : str
        Research design (see :func:`power`).
    n : int
        Sample size.  For ``'cluster_rct'`` this is *n_clusters*.
    power_target : float
        Desired power (default 0.80).
    **kwargs
        Design-specific parameters.

    Returns
    -------
    PowerResult
        With ``.effect_size`` set to the MDE.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.mde("did", n=1000, n_periods=10, n_treated_periods=5)
    PowerResult(design='did', power=0.8000, n=1000, effect_size=0.1524)
    """
    if n is None:
        raise ValueError("n must be specified for MDE calculation.")
    return power(
        design,
        n=n,
        effect_size=None,
        power_target=power_target,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Internal solvers
# ---------------------------------------------------------------------------


def _solve_for_n(
    func: Callable[..., PowerResult],
    effect_size: Any,
    power_target: float,
    design: str,
    extra_kwargs: Dict[str, Any],
) -> PowerResult:
    """Binary search for the minimum n achieving *power_target*."""

    def _power_at_n(n_val: int) -> float:
        if design == "cluster_rct":
            res = func(
                n_clusters=n_val,
                effect_size=effect_size,
                **extra_kwargs,
            )
        else:
            res = func(n=n_val, effect_size=effect_size, **extra_kwargs)
        return float(np.asarray(res.power).item())

    # Expand search range until we exceed power_target
    lo, hi = 10, 100
    while _power_at_n(hi) < power_target:
        hi *= 2
        if hi > 1e9:
            raise ConvergenceFailure(
                "Could not find a sample size achieving the target power. "
                "Check that your effect size and design parameters are "
                "reasonable.",
                recovery_hint=(
                    "Increase the effect size, lower power_target, or "
                    "revise design-specific parameters before solving for "
                    "sample size."
                ),
                diagnostics={
                    "design": design,
                    "effect_size": effect_size,
                    "power_target": power_target,
                    "max_n": hi,
                },
            )

    # Binary search
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _power_at_n(mid) >= power_target:
            hi = mid
        else:
            lo = mid

    n_required = int(hi)
    # Compute final result at solved n
    if design == "cluster_rct":
        result = func(
            n_clusters=n_required,
            effect_size=effect_size,
            **extra_kwargs,
        )
    else:
        result = func(n=n_required, effect_size=effect_size, **extra_kwargs)
    return result


def _solve_for_mde(
    func: Callable[..., PowerResult],
    n: Any,
    power_target: float,
    design: str,
    extra_kwargs: Dict[str, Any],
) -> PowerResult:
    """Binary search for the minimum effect size achieving *power_target*."""

    def _power_at_es(es_val: float) -> float:
        if design == "cluster_rct":
            res = func(n_clusters=n, effect_size=es_val, **extra_kwargs)
        else:
            res = func(n=n, effect_size=es_val, **extra_kwargs)
        return float(np.asarray(res.power).item())

    # Search in [0, hi] for the MDE
    lo, hi = 0.0, 1.0
    while _power_at_es(hi) < power_target:
        hi *= 2
        if hi > 100:
            raise ConvergenceFailure(
                "Could not find a detectable effect size. "
                "The sample size may be too small for this design.",
                recovery_hint=(
                    "Increase n, lower power_target, or revisit the design "
                    "parameters before solving for an MDE."
                ),
                diagnostics={
                    "design": design,
                    "n": n,
                    "power_target": power_target,
                    "max_effect_size": hi,
                },
            )

    # Binary search with precision 1e-6
    for _ in range(80):
        mid = (lo + hi) / 2
        if _power_at_es(mid) >= power_target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break

    mde_val = round(hi, 6)
    if design == "cluster_rct":
        result = func(n_clusters=n, effect_size=mde_val, **extra_kwargs)
    else:
        result = func(n=n, effect_size=mde_val, **extra_kwargs)
    # Override effect_size to report the MDE cleanly
    result.effect_size = mde_val
    return result
