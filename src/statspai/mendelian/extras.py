"""
Additional MR estimators and instrument-strength diagnostics.

Rounds out the MR suite with:

- :func:`mr_mode` — weighted mode estimator (Hartwig et al. 2017).
  Consistent when the plurality of SNPs are valid instruments (the
  "ZEMPA" assumption), more permissive than the median's 50%.
- :func:`mr_f_statistic` — first-stage Cragg-Donald / mean F-statistic
  across SNPs. Flags weak-instrument bias when mean F < 10.
- :func:`mr_funnel_plot` — visualization of SNP-specific Wald ratios
  against their precision, to detect directional pleiotropy.
- :func:`mr_scatter_plot` — classic beta_Y vs beta_X scatter with IVW
  and Egger lines.

References
----------
Hartwig, F.P., Davey Smith, G. & Bowden, J. (2017).
"Robust inference in summary data Mendelian randomization via the
zero modal pleiotropy assumption." *IJE*, 46(6), 1985-1998. [@hartwig2017robust]

Staiger, D. & Stock, J.H. (1997).
"Instrumental variables regression with weak instruments."
*Econometrica*, 65(3), 557-586. [@staiger1997instrumental]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy import stats

from .._result_serialize import ResultProtocolMixin

__all__ = [
    "ModeBasedResult",
    "FStatisticResult",
    "mr_mode",
    "mr_f_statistic",
    "mr_funnel_plot",
    "mr_scatter_plot",
]


# --------------------------------------------------------------------------- #
#  Mode-based estimator (Hartwig 2017)
# --------------------------------------------------------------------------- #


@dataclass
class ModeBasedResult(ResultProtocolMixin):
    """Result container returned by :func:`mr_mode`.

    Holds the mode-based MR point estimate, bootstrap standard error,
    confidence interval, p-value, the number of SNPs, and the Silverman
    bandwidth used for the kernel density. Call :meth:`summary` for a
    formatted text report.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> bx = rng.uniform(0.2, 0.5, 25)
    >>> by = 0.3 * bx + rng.normal(0, 0.01, 25)
    >>> sx = rng.uniform(0.02, 0.04, 25)
    >>> sy = rng.uniform(0.02, 0.04, 25)
    >>> res = sp.mr_mode(bx, by, sx, sy, n_boot=200, seed=0)
    >>> isinstance(res, sp.ModeBasedResult)
    True
    >>> res.method
    'weighted-mode'
    """

    _citation_keys = ("hartwig2017robust", "staiger1997instrumental")

    estimate: float
    se: float
    ci: tuple[float, float]
    p_value: float
    n_snps: int
    bandwidth: float
    method: str = "weighted-mode"

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"MR {self.method} (Hartwig et al. 2017)\n"
            f"  Estimate = {self.estimate:.4f}   SE = {self.se:.4f}   "
            f"95% CI [{lo:.4f}, {hi:.4f}]   p = {self.p_value:.4g}\n"
            f"  Bandwidth (Silverman) = {self.bandwidth:.4f}   "
            f"n SNPs = {self.n_snps}"
        )


def _mad(x: np.ndarray) -> float:
    """R's ``mad()``: 1.4826 * median absolute deviation from the median."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _mbe_bandwidth(ratios: np.ndarray, phi: float) -> float:
    """Hartwig et al.'s bandwidth, as ``MendelianRandomization:::mbe_est``.

    ``0.9 * min(sd, mad) / n^(1/5) * phi`` on the UNWEIGHTED ratios. The
    helper this replaces used a weighted SD and IQR / 1.34, a different
    rule, which on the package's LDL-C / CHD example moved the estimate 7%.
    """
    n = len(ratios)
    s = 0.9 * min(float(np.std(ratios, ddof=1)), _mad(ratios)) / n ** (1 / 5)
    return float(s * phi)


def _mbe_estimate(
    ratios: np.ndarray,
    weights: np.ndarray,
    phi: float = 1.0,
    refine: bool = False,
) -> Tuple[float, float]:
    """Mode of the weighted Gaussian KDE of the ratios; returns (mode, h).

    The reference takes the abscissa of ``density(ratios, weights, bw = h)``
    with the largest ordinate: a 512-point grid over ``[min - 3h, max +
    3h]``. Evaluating the exact kernel on that same grid reproduces its
    estimate to machine precision, because the estimate IS a grid point. The
    price is quantisation: the grid step is ``(range + 6h) / 511``, which is
    coarse when some ratios are extreme (0.54 on the LDL-C example).
    ``refine=True`` climbs from the best grid point to the continuous mode.
    """
    h = _mbe_bandwidth(ratios, phi)
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    if not np.isfinite(h) or h <= 0:
        return float(ratios[np.argmax(w)]), float(h)

    def _dens(x):
        x = np.atleast_1d(x)
        return (
            w[None, :] * np.exp(-0.5 * ((x[:, None] - ratios[None, :]) / h) ** 2)
        ).sum(1)

    grid = np.linspace(ratios.min() - 3 * h, ratios.max() + 3 * h, 512)
    i = int(np.argmax(_dens(grid)))
    mode = float(grid[i])
    if refine:
        from scipy.optimize import minimize_scalar

        lo = grid[max(i - 1, 0)]
        hi = grid[min(i + 1, len(grid) - 1)]
        res = minimize_scalar(
            lambda x: -float(_dens(x)[0]),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-12 * max(1.0, abs(mode))},
        )
        if res.success and -res.fun >= float(_dens(mode)[0]):
            mode = float(res.x)
    return mode, float(h)


def mr_mode(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    method: str = "weighted",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    phi: float = 1.0,
    refine: bool = False,
) -> ModeBasedResult:
    """Mode-based MR estimator (Hartwig, Davey Smith & Bowden 2017).

    Matches ``MendelianRandomization::mr_mbe(stderror = "simple")`` to
    machine precision in both weightings: same bandwidth rule, same
    512-point density grid, same bootstrap construction.

    .. versionchanged:: 1.27.0
       The bandwidth was a weighted-SD / IQR rule rather than Hartwig et
       al.'s ``0.9 min(sd, mad) n^(-1/5)`` on the unweighted ratios, the
       mode was searched on a different grid, and the bootstrap redrew the
       exposure associations and recomputed the weights instead of
       resampling the ratios with the weights held fixed. On the reference
       package's LDL-C / CHD example the estimate moved 7%. The standard
       error is now the MAD of the bootstrap draws, as in the reference,
       rather than their SD.

    Parameters
    ----------
    method : {"weighted", "simple"}, default "weighted"
        ``weighted`` uses IVW weights when finding the mode; ``simple``
        uses unit weights (robust but less efficient).
    phi : float, default 1.0
        Bandwidth multiplier, the reference's ``phi``.
    refine : bool, default False
        The reference estimate is a point of its 512-point density grid,
        whose step is ``(range + 6h) / 511`` -- coarse when a few ratios
        are extreme. ``True`` climbs to the continuous mode of the same
        density. Leave ``False`` to reproduce ``mr_mbe``.
    n_boot : int, default 1000
        Bootstrap replicates for SE.

    References
    ----------
    hartwig2017robust

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> bx = rng.uniform(0.2, 0.5, 25)
    >>> by = 0.3 * bx + rng.normal(0, 0.01, 25)
    >>> sx = rng.uniform(0.02, 0.04, 25)
    >>> sy = rng.uniform(0.02, 0.04, 25)
    >>> res = sp.mr_mode(bx, by, sx, sy, n_boot=200, seed=0)
    >>> res.n_snps
    25
    >>> bool(0.0 < res.estimate < 1.0)  # near the true 0.3
    True
    """
    if method not in ("weighted", "simple"):
        raise ValueError("method must be 'weighted' or 'simple'")
    rng = np.random.default_rng(seed)
    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    # se_exposure is not used: mr_mbe's default stderror = "simple" takes the
    # first-order ratio SE se_y / |bx|, which ignores it. It stays in the
    # signature for symmetry with the other estimators.
    del se_exposure
    sy = np.asarray(se_outcome, dtype=float)

    ratios = by / bx
    ratio_se = sy / np.abs(bx)
    w = 1.0 / ratio_se**2 if method == "weighted" else np.ones_like(ratios)

    estimate, bandwidth = _mbe_estimate(ratios, w, phi=phi, refine=refine)

    # Reference bootstrap (mbe_boot): resample the RATIOS from their
    # first-order sampling distributions, keep the weights fixed, and
    # re-estimate -- bandwidth included -- on each draw.
    boot = np.empty(n_boot)
    for b in range(n_boot):
        r_b = rng.normal(ratios, ratio_se)
        boot[b] = _mbe_estimate(r_b, w, phi=phi, refine=refine)[0]

    se = _mad(boot) if n_boot > 1 else float("nan")
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (estimate - z_crit * se, estimate + z_crit * se)
    z = estimate / se if se > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))

    _result = ModeBasedResult(
        estimate=float(estimate),
        se=se,
        ci=(float(ci[0]), float(ci[1])),
        p_value=p,
        n_snps=len(bx),
        bandwidth=bandwidth,
        method=f"{method}-mode",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.mendelian.mr_mode",
            params={
                "method": method,
                "bandwidth": bandwidth,
                "n_snps": len(bx),
            },
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# --------------------------------------------------------------------------- #
#  F-statistic: instrument strength
# --------------------------------------------------------------------------- #


@dataclass
class FStatisticResult(ResultProtocolMixin):
    """Result container returned by :func:`mr_f_statistic`.

    Summarises instrument strength across SNPs: the mean / min / max
    per-SNP F-statistic, mean explained variance ``r2_mean``, the full
    ``per_snp_F`` array, and a ``weak_instrument_risk`` flag that is
    ``True`` when any SNP's F falls below 10 (Staiger-Stock 1997).

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> bx = rng.uniform(0.2, 0.5, 25)
    >>> sx = rng.uniform(0.02, 0.04, 25)
    >>> f = sp.mr_f_statistic(bx, sx)
    >>> isinstance(f, sp.FStatisticResult)
    True
    >>> f.per_snp_F.shape
    (25,)
    """

    _citation_keys = ("hartwig2017robust", "staiger1997instrumental")

    f_mean: float
    f_min: float
    f_max: float
    weak_instrument_risk: bool
    r2_mean: float
    per_snp_F: np.ndarray

    def summary(self) -> str:
        flag = (
            "⚠ WEAK INSTRUMENT RISK"
            if self.weak_instrument_risk
            else "OK — instruments strong"
        )
        return (
            "MR Instrument-Strength Diagnostic\n"
            f"  Mean per-SNP F       = {self.f_mean:.2f}\n"
            f"  Min  per-SNP F       = {self.f_min:.2f}\n"
            f"  Max  per-SNP F       = {self.f_max:.2f}\n"
            f"  Mean per-SNP R^2     = {self.r2_mean:.4f}\n"
            f"  Staiger-Stock rule : F >= 10 required for each SNP; {flag}"
        )


def mr_f_statistic(
    beta_exposure: np.ndarray,
    se_exposure: np.ndarray,
    *,
    n_samples: Optional[int] = None,
) -> FStatisticResult:
    """Per-SNP F-statistic as an instrument-strength summary.

    Uses the standard summary-stat approximation
    ``F_i = (beta_i / se_i)^2``, which is valid for large GWAS where
    the first-stage regression is asymptotically Normal. Flags weak-IV
    risk when any SNP's F falls below 10 (Staiger-Stock 1997).

    Parameters
    ----------
    n_samples : int, optional
        GWAS sample size. If provided, R^2 is reported via
        R^2 = F / (F + n - 2); otherwise a small-sample approximation
        ``R^2 ≈ F / (F + n_snps - 2)`` is used.

    References
    ----------
    staiger1997instrumental

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> bx = rng.uniform(0.2, 0.5, 25)
    >>> sx = rng.uniform(0.02, 0.04, 25)  # strong instruments
    >>> f = sp.mr_f_statistic(bx, sx)
    >>> bool(f.f_mean > 10)  # Staiger-Stock rule of thumb
    True
    >>> f.weak_instrument_risk
    False
    """
    bx = np.asarray(beta_exposure, dtype=float)
    sx = np.asarray(se_exposure, dtype=float)
    f_per = (bx / sx) ** 2
    n = n_samples if n_samples is not None else len(bx)
    if n > 2:
        r2_per = f_per / (f_per + n - 2)
    else:
        r2_per = f_per * 0.0
    return FStatisticResult(
        f_mean=float(np.mean(f_per)),
        f_min=float(np.min(f_per)),
        f_max=float(np.max(f_per)),
        weak_instrument_risk=bool(np.min(f_per) < 10.0),
        r2_mean=float(np.mean(r2_per)),
        per_snp_F=f_per,
    )


# --------------------------------------------------------------------------- #
#  Funnel + scatter plots
# --------------------------------------------------------------------------- #


def mr_funnel_plot(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_outcome: np.ndarray,
    *,
    snp_ids: Optional[List[str]] = None,
    ax: Any = None,
) -> Any:
    """Funnel plot of SNP-specific Wald ratios vs. precision.

    An asymmetric funnel around the IVW estimate suggests directional
    horizontal pleiotropy. Returns the matplotlib axis.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> bx = rng.uniform(0.2, 0.5, 25)
    >>> by = 0.3 * bx + rng.normal(0, 0.01, 25)
    >>> sy = rng.uniform(0.02, 0.04, 25)
    >>> ax = sp.mr_funnel_plot(bx, by, sy)
    >>> ax is not None
    True
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required for plotting") from exc

    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    ratio = by / bx
    precision = np.abs(bx) / sy

    w = 1.0 / (sy / np.abs(bx)) ** 2
    ivw_est = float(np.sum(w * ratio) / np.sum(w))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ratio, precision, s=40, alpha=0.7, edgecolors="black")
    ax.axvline(ivw_est, color="red", ls="--", lw=1.5, label=f"IVW = {ivw_est:.3f}")
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    if snp_ids is not None:
        for xi, yi, name in zip(ratio, precision, snp_ids):
            ax.annotate(
                name,
                (xi, yi),
                fontsize=8,
                alpha=0.6,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xlabel("SNP-specific Wald ratio (β_Y / β_X)")
    ax.set_ylabel("Precision  |β_X| / SE(β_Y)")
    ax.set_title("MR Funnel Plot (asymmetry → directional pleiotropy)")
    ax.legend()
    return ax


def mr_scatter_plot(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    ax: Any = None,
) -> Any:
    """Classic MR scatter plot with IVW and MR-Egger lines.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> bx = rng.uniform(0.2, 0.5, 25)
    >>> by = 0.3 * bx + rng.normal(0, 0.01, 25)
    >>> sx = rng.uniform(0.02, 0.04, 25)
    >>> sy = rng.uniform(0.02, 0.04, 25)
    >>> ax = sp.mr_scatter_plot(bx, by, sx, sy)
    >>> ax is not None
    True
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required for plotting") from exc

    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sx = np.asarray(se_exposure, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)

    w = 1.0 / sy**2
    ivw = float(np.sum(w * bx * by) / np.sum(w * bx**2))

    X = np.column_stack([np.ones(len(bx)), bx])
    W = np.diag(w)
    try:
        coef = np.linalg.solve(X.T @ W @ X, X.T @ W @ by)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ by
    egger_intercept, egger_slope = float(coef[0]), float(coef[1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(
        bx, by, xerr=sx, yerr=sy, fmt="o", capsize=2, alpha=0.7, color="tab:blue"
    )
    xs = np.linspace(min(0, float(bx.min())), float(bx.max()), 50)
    ax.plot(xs, ivw * xs, "r--", lw=1.5, label=f"IVW slope = {ivw:.3f}")
    ax.plot(
        xs,
        egger_intercept + egger_slope * xs,
        "g-.",
        lw=1.5,
        label=f"MR-Egger: {egger_intercept:+.3f} + {egger_slope:.3f}β_X",
    )
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("SNP effect on exposure (β_X)")
    ax.set_ylabel("SNP effect on outcome (β_Y)")
    ax.set_title("MR Scatter Plot with IVW and MR-Egger Lines")
    ax.legend()
    return ax
