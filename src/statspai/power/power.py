"""
Power and sample size calculations for causal inference and econometric designs.

Provides power analysis for designs that are missing from the Python ecosystem:
DID, RD, IV, cluster RCT, and standard RCT/OLS — with support for power curves,
minimum detectable effect (MDE), and sample size solving.

References:
- Burlig, Preonas & Woerman (2020): "Panel data and experimental design"
- Cattaneo, Titiunik & Vazquez-Bare (2019): "Power calculations for RD designs"
- Stock & Yogo (2005): "Testing for weak instruments"
"""

import numpy as np
from scipy.stats import norm

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

_DESIGN_FUNCTIONS = {}  # populated after function definitions


# ---------------------------------------------------------------------------
# PowerResult
# ---------------------------------------------------------------------------


class PowerResult:
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
    """

    def __init__(self, power_val, n, effect_size, design, params):
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
        lines = []
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
            lines.append(f"  N range      : [{self._fmt_n(n_arr.min())}, {self._fmt_n(n_arr.max())}]")
            es_arr = np.asarray(self.effect_size)
            if es_arr.ndim == 0:
                lines.append(f"  Effect size  : {float(es_arr):.4f}")
            else:
                lines.append(f"  Effect range : [{es_arr.min():.4f}, {es_arr.max():.4f}]")

        lines.append(f"  Alpha        : {self.params.get('alpha', 0.05)}")

        # Design-specific extras
        extras = {k: v for k, v in self.params.items()
                  if k not in ("n", "effect_size", "alpha", "power", "design")}
        if extras:
            lines.append("-" * 56)
            for k, v in extras.items():
                label = k.replace("_", " ").title()
                lines.append(f"  {label:20s}: {v}")

        lines.append("=" * 56)
        return "\n".join(lines)

    def __repr__(self):
        if np.ndim(self.power) == 0:
            return (
                f"PowerResult(design={self.design!r}, power={float(self.power):.4f}, "
                f"n={self._fmt_n(self.n)}, effect_size={float(self.effect_size):.4f})"
            )
        arr = np.asarray(self.power)
        return (
            f"PowerResult(design={self.design!r}, power=[{arr.min():.4f}..{arr.max():.4f}], "
            f"len={len(arr)})"
        )

    def _repr_html_(self):
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
                ("N range", f"[{self._fmt_n(n_arr.min())}, {self._fmt_n(n_arr.max())}]"),
                ("Alpha", self.params.get("alpha", 0.05)),
            ]

        html = (
            '<div style="font-family: monospace; padding:8px;">'
            '<table style="border-collapse:collapse;">'
            f'<caption style="font-weight:bold; font-size:1.1em; padding-bottom:6px;">'
            f'Power Analysis &mdash; {self.design.upper()}</caption>'
        )
        for label, val in rows:
            html += (
                f'<tr><td style="padding:2px 12px 2px 0; font-weight:bold;">{label}</td>'
                f'<td style="padding:2px 0;">{val}</td></tr>'
            )
        html += "</table></div>"
        return html

    @staticmethod
    def _fmt_n(val):
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

    def plot(self, ax=None, figsize=(8, 5), **kwargs):
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
                "plot() requires power computed over a range of n or effect_size values. "
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

        plot_kwargs = dict(linewidth=2, color="#2563eb")
        plot_kwargs.update(kwargs)
        ax.plot(x, power_arr, **plot_kwargs)
        ax.axhline(0.8, linestyle="--", color="#9ca3af", linewidth=1, label="Power = 0.80")
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


def _to_array(val):
    """Convert scalar, list, or range to numpy array."""
    if isinstance(val, range):
        return np.array(list(val), dtype=float)
    arr = np.asarray(val, dtype=float)
    return arr


# ---------------------------------------------------------------------------
# Design-specific power functions
# ---------------------------------------------------------------------------


def power_rct(n, effect_size, alpha=0.05, ratio=1.0, sigma=1.0):
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
        n=n, effect_size=effect_size, design="rct",
        params=dict(n=n, effect_size=effect_size, alpha=alpha,
                    ratio=ratio, sigma=sigma),
    )


def power_did(n, effect_size, n_periods, n_treated_periods, rho=0.5,
              alpha=0.05, sigma=1.0):
    """Power for Difference-in-Differences.

    Follows Burlig, Preonas & Woerman (2020) accounting for serial
    correlation and the number of pre/post periods.

    Parameters
    ----------
    n : int or array-like
        Total number of units (treated + control).
    effect_size : float or array-like
        Standardised effect size.
    n_periods : int
        Total number of time periods.
    n_treated_periods : int
        Number of post-treatment periods.
    rho : float
        First-order autocorrelation of errors (0–1).
    alpha : float
        Significance level (two-sided).
    sigma : float
        Error standard deviation.

    Returns
    -------
    PowerResult
    """
    n_arr = _to_array(n)
    es_arr = _to_array(effect_size)
    z_alpha = norm.ppf(1 - alpha / 2)

    T = n_periods
    T_post = n_treated_periods
    T_pre = T - T_post

    # SE(DID) ~ sigma / sqrt(N) * sqrt((1 + (T-1)*rho) / (T_post * (1 - T_post/T)))
    # Denominator captures information gain from pre/post split
    numerator = 1 + (T - 1) * rho
    denominator = T_post * (1 - T_post / T)
    # Guard against degenerate cases
    denominator = np.maximum(denominator, 1e-12)

    se = sigma / np.sqrt(n_arr) * np.sqrt(numerator / denominator)
    pwr = norm.cdf(np.abs(es_arr) * sigma / se - z_alpha)

    return PowerResult(
        power_val=float(pwr) if pwr.ndim == 0 else pwr,
        n=n, effect_size=effect_size, design="did",
        params=dict(n=n, effect_size=effect_size, alpha=alpha,
                    n_periods=n_periods, n_treated_periods=n_treated_periods,
                    rho=rho, sigma=sigma),
    )


def power_rd(n, effect_size, bandwidth=None, kernel="triangular",
             density_at_cutoff=1.0, alpha=0.05, sigma=1.0):
    """Power for Regression Discontinuity designs.

    Following Cattaneo, Titiunik & Vazquez-Bare (2019).

    Parameters
    ----------
    n : int or array-like
        Total sample size in the data.
    effect_size : float or array-like
        Standardised effect size at the cutoff.
    bandwidth : float or None
        Bandwidth around the cutoff.  If *None*, defaults to 0.5
        (half the running-variable range on each side).
    kernel : {'triangular', 'uniform', 'epanechnikov'}
        Kernel used for local weighting.
    density_at_cutoff : float
        Estimated density of the running variable at the cutoff
        (default 1.0 for a uniform running variable on [0,1]).
    alpha : float
        Significance level.
    sigma : float
        Conditional outcome std dev near the cutoff.

    Returns
    -------
    PowerResult
    """
    n_arr = _to_array(n)
    es_arr = _to_array(effect_size)
    z_alpha = norm.ppf(1 - alpha / 2)

    if bandwidth is None:
        bandwidth = 0.5

    # Kernel efficiency relative to uniform
    kernel_eff = {
        "triangular": 0.75,
        "uniform": 1.0,
        "epanechnikov": 0.85,
    }
    eff = kernel_eff.get(kernel, 0.75)

    # Effective sample near cutoff
    n_eff = n_arr * bandwidth * density_at_cutoff * eff
    n_eff = np.maximum(n_eff, 1.0)

    # Each side of cutoff gets ~half the effective sample
    n_side = n_eff / 2.0
    se = sigma * np.sqrt(2.0 / n_side)  # comparing two local means

    pwr = norm.cdf(np.abs(es_arr) * sigma / se - z_alpha)

    return PowerResult(
        power_val=float(pwr) if pwr.ndim == 0 else pwr,
        n=n, effect_size=effect_size, design="rd",
        params=dict(n=n, effect_size=effect_size, alpha=alpha,
                    bandwidth=bandwidth, kernel=kernel,
                    density_at_cutoff=density_at_cutoff, sigma=sigma),
    )


def power_iv(n, effect_size, first_stage_f=None, r2_z=None,
             alpha=0.05, sigma=1.0):
    """Power for Instrumental Variables / 2SLS estimation.

    Accounts for the power penalty from a weak first stage.

    Parameters
    ----------
    n : int or array-like
        Sample size.
    effect_size : float or array-like
        Standardised effect size of the endogenous variable.
    first_stage_f : float or None
        First-stage F-statistic.  If provided, power is adjusted for
        instrument weakness: effective_power ~ power_ols * F / (F + 1).
    r2_z : float or None
        R-squared of the first-stage regression.  Alternative to
        *first_stage_f*; if both are given, *first_stage_f* takes precedence.
    alpha : float
        Significance level.
    sigma : float
        Error standard deviation.

    Returns
    -------
    PowerResult
    """
    n_arr = _to_array(n)
    es_arr = _to_array(effect_size)
    z_alpha = norm.ppf(1 - alpha / 2)

    # Baseline: OLS-like power
    se_ols = sigma / np.sqrt(n_arr)
    pwr_ols = norm.cdf(np.abs(es_arr) * sigma / se_ols - z_alpha)

    # Adjustment for first-stage weakness
    if first_stage_f is not None:
        adjustment = first_stage_f / (first_stage_f + 1)
    elif r2_z is not None:
        # F ~ n * r2_z / (1 - r2_z)  for single instrument
        f_approx = n_arr * r2_z / (1 - r2_z + 1e-12)
        adjustment = f_approx / (f_approx + 1)
    else:
        adjustment = 1.0  # no first-stage penalty

    # IV power = OLS power * adjustment
    # More precisely: IV variance is sigma^2 / (n * R^2_z), but the
    # F-based shortcut captures the intuition cleanly.
    pwr = pwr_ols * adjustment

    return PowerResult(
        power_val=float(pwr) if pwr.ndim == 0 else pwr,
        n=n, effect_size=effect_size, design="iv",
        params=dict(n=n, effect_size=effect_size, alpha=alpha,
                    first_stage_f=first_stage_f, r2_z=r2_z, sigma=sigma),
    )


def power_cluster_rct(n_clusters, cluster_size, effect_size, icc,
                      alpha=0.05, sigma=1.0):
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
        effect_size=effect_size, design="cluster_rct",
        params=dict(n_clusters=n_clusters, cluster_size=cluster_size,
                    effect_size=effect_size, icc=icc, alpha=alpha,
                    sigma=sigma),
    )


def power_ols(n, effect_size, n_covariates=0, r2_other=0.0,
              alpha=0.05, sigma=1.0):
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
        n=n, effect_size=effect_size, design="ols",
        params=dict(n=n, effect_size=effect_size, alpha=alpha,
                    n_covariates=n_covariates, r2_other=r2_other,
                    sigma=sigma),
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


def power(design, *, n=None, effect_size=None, power_target=None, **kwargs):
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
    >>> sp.power("did", n=1000, effect_size=0.1, n_periods=10, n_treated_periods=5)
    >>> sp.power("did", power=0.8, effect_size=0.1, n_periods=10, n_treated_periods=5)
    >>> result = sp.power("did", n=range(100, 2000, 100), effect_size=0.1,
    ...                   n_periods=10, n_treated_periods=5)
    >>> result.plot()
    """
    # Accept 'power' kwarg as alias for power_target (convenience API)
    if power_target is None and "power" in kwargs:
        power_target = kwargs.pop("power")

    design = design.lower().strip()
    if design not in _DESIGN_FUNCTIONS:
        supported = ", ".join(sorted(_DESIGN_FUNCTIONS))
        raise ValueError(
            f"Unknown design {design!r}. Supported designs: {supported}"
        )

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
            raise ValueError(
                "When solving for n, effect_size must be specified."
            )
        n_solved = _solve_for_n(func, effect_size, power_target, design, kwargs)
        return n_solved

    # ------------------------------------------------------------------
    # Case 2: Solve for MDE
    # ------------------------------------------------------------------
    if effect_size is None:
        if power_target is None:
            raise ValueError(
                "When effect_size is None you must specify power_target (or power=) "
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


def mde(design, *, n=None, power_target=0.8, **kwargs):
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
    """
    if n is None:
        raise ValueError("n must be specified for MDE calculation.")
    return power(design, n=n, effect_size=None, power_target=power_target, **kwargs)


# ---------------------------------------------------------------------------
# Internal solvers
# ---------------------------------------------------------------------------


def _solve_for_n(func, effect_size, power_target, design, extra_kwargs):
    """Binary search for the minimum n achieving *power_target*."""
    alpha = extra_kwargs.get("alpha", 0.05)

    def _power_at_n(n_val):
        if design == "cluster_rct":
            res = func(n_clusters=n_val, effect_size=effect_size, **extra_kwargs)
        else:
            res = func(n=n_val, effect_size=effect_size, **extra_kwargs)
        return float(np.asarray(res.power).item())

    # Expand search range until we exceed power_target
    lo, hi = 10, 100
    while _power_at_n(hi) < power_target:
        hi *= 2
        if hi > 1e9:
            raise RuntimeError(
                "Could not find a sample size achieving the target power. "
                "Check that your effect size and design parameters are reasonable."
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
        result = func(n_clusters=n_required, effect_size=effect_size, **extra_kwargs)
    else:
        result = func(n=n_required, effect_size=effect_size, **extra_kwargs)
    return result


def _solve_for_mde(func, n, power_target, design, extra_kwargs):
    """Binary search for the minimum effect size achieving *power_target*."""

    def _power_at_es(es_val):
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
            raise RuntimeError(
                "Could not find a detectable effect size. "
                "The sample size may be too small for this design."
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
