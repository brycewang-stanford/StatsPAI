"""
MR-PRESSO: global test for horizontal pleiotropy, outlier test and the
outlier-corrected IVW estimate (Verbanck et al. 2018).

Split out of :mod:`statspai.mendelian.diagnostics` (2026-10-20) to keep each
file to one concern; the old import path still works. Clean-room
implementation, specification in ``docs/dev/cleanroom/mr_presso.md``.

References
----------
Verbanck, M. et al. (2018). "Detection of widespread horizontal
pleiotropy in causal relationships inferred from Mendelian
randomization between complex traits and diseases." *Nature Genetics*,
50(5), 693-698. [@verbanck2018detection]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import stats

from .._result_serialize import ResultProtocolMixin

__all__ = ["MRPressoResult", "mr_presso"]


# --------------------------------------------------------------------------- #
#  MR-PRESSO (outlier detection + correction)
# --------------------------------------------------------------------------- #


@dataclass
class MRPressoResult(ResultProtocolMixin):
    """Container for MR-PRESSO global test and outlier-corrected estimates.

    Returned by :func:`mr_presso`.  Holds the raw IVW estimate, the
    global-test RSS and p-value, the list of detected ``outliers``, and
    (when outliers are present) the outlier-corrected estimate and
    distortion-test p-value.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 12
    >>> bx = rng.uniform(0.2, 0.5, n)
    >>> by = 0.3 * bx + rng.normal(0, 0.02, n)
    >>> sx = rng.uniform(0.02, 0.05, n)
    >>> sy = rng.uniform(0.02, 0.05, n)
    >>> res = sp.mr_presso(bx, by, sx, sy, n_boot=500, seed=0)
    >>> isinstance(res, sp.MRPressoResult)
    True
    >>> bool(0.0 <= res.global_test_pvalue <= 1.0)
    True
    """

    _citation_keys = (
        "bowden2016assessing",
        "hemani2017orienting",
        "bowden2018improving",
        "verbanck2018detection",
    )

    raw_estimate: float
    raw_se: float
    raw_p: float
    outlier_corrected_estimate: Optional[float]
    outlier_corrected_se: Optional[float]
    outlier_corrected_p: Optional[float]
    outliers: List[int] = field(default_factory=list)
    global_test_rss_obs: float = 0.0
    global_test_pvalue: float = 1.0
    distortion_p: Optional[float] = None
    outlier_pvalues: Optional[List[float]] = None
    distortion_coefficient: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "MR-PRESSO Global Test",
            f"  Observed RSS = {self.global_test_rss_obs:.3f}",
            f"  Global test p-value = {self.global_test_pvalue:.4g}",
            "",
            f"Raw IVW:            beta = {self.raw_estimate:.4f}  "
            f"SE = {self.raw_se:.4f}  p = {self.raw_p:.4g}",
        ]
        if self.outlier_corrected_estimate is not None:
            lines.append(
                f"Outlier-corrected:  beta = {self.outlier_corrected_estimate:.4f}  "
                f"SE = {self.outlier_corrected_se:.4f}  "
                f"p = {self.outlier_corrected_p:.4g}"
            )
            if self.outliers:
                lines.append(f"  Outlier SNP indices: {self.outliers}")
            if self.distortion_p is not None:
                lines.append(
                    f"  Distortion test p = {self.distortion_p:.4g} "
                    f"(H0: raw == corrected)"
                )
        else:
            lines.append("No outliers detected.")
        return "\n".join(lines)


def _ivw_regression(bx: np.ndarray, by: np.ndarray, w: np.ndarray) -> tuple:
    """Weighted no-intercept regression of ``by`` on ``bx`` (weights ``w``).

    Returns ``(slope, se, t, p)`` with the residual-scaled standard error and
    a t reference on ``n - 1`` degrees of freedom (an ordinary weighted
    least-squares fit; Verbanck et al. 2018 build every MR-PRESSO estimate on
    this IVW regression).
    """
    sxx = float(np.sum(w * bx**2))
    slope = float(np.sum(w * bx * by) / sxx)
    df = bx.size - 1
    resid = by - slope * bx
    se = float(np.sqrt(np.sum(w * resid**2) / df / sxx))
    t = slope / se if se > 0 else float("inf") * np.sign(slope)
    p = float(2 * stats.t.sf(abs(t), df))
    return slope, se, float(t), p


def _loo_slopes(bx: np.ndarray, by: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Leave-one-out IVW slopes along the last axis (works row-wise on 2-D)."""
    sxy = np.sum(w * bx * by, axis=-1, keepdims=True)
    sxx = np.sum(w * bx**2, axis=-1, keepdims=True)
    _ret: np.ndarray = (sxy - w * bx * by) / (sxx - w * bx**2)
    return _ret


def _presso_rss(bx: np.ndarray, by: np.ndarray, sy: np.ndarray) -> tuple:
    """Observed MR-PRESSO residuals (Verbanck et al. 2018, global test step 2).

    Returns ``(rss_obs, resid2)``: ``resid2[j] = (by_j - b_{-j} bx_j)^2`` with
    ``b_{-j}`` the IVW slope without variant ``j``, and the global statistic
    ``rss_obs = sum_j resid2[j] / se_outcome_j^2``.
    """
    w = 1.0 / sy**2
    resid2 = (by - _loo_slopes(bx, by, w) * bx) ** 2
    return float(np.sum(w * resid2)), resid2


def mr_presso(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    n_boot: int = 1000,
    sig_threshold: float = 0.05,
    seed: Optional[int] = None,
) -> MRPressoResult:
    """MR-PRESSO global test + outlier detection + corrected estimate.

    The Mendelian Randomization Pleiotropy RESidual Sum and Outlier method
    of Verbanck et al. (2018), single-exposure version:

    1. **Global test.** With ``b_{-j}`` the IVW slope fitted without variant
       ``j``, the observed statistic is
       ``RSS_obs = sum_j (by_j - b_{-j} bx_j)^2 / se_outcome_j^2``. Its null
       distribution is simulated ``n_boot`` times by drawing
       ``bx*_j ~ N(bx_j, se_exposure_j^2)`` and
       ``by*_j ~ N(b_{-j} bx_j, se_outcome_j^2)`` and recomputing the same
       statistic (leave-one-out slopes included) on the draws. The p-value is
       the share of draws with ``RSS* > RSS_obs``.
    2. **Outlier test** (only when the global test rejects at
       ``sig_threshold``). Variant ``j``'s p-value is the share of draws in
       which its own term exceeds the observed one, multiplied by the number
       of variants (Bonferroni) and capped at 1. Variants below
       ``sig_threshold`` are outliers.
    3. **Outlier-corrected estimate**: the IVW regression without the
       outliers.
    4. **Distortion test**: ``D = 100 (b_raw - b_corrected) / |b_corrected|``
       against a null in which variants are resampled from the non-outliers.

    Parameters
    ----------
    beta_exposure, beta_outcome : array-like
        Variant-exposure and variant-outcome associations.
    se_exposure, se_outcome : array-like
        Their standard errors (strictly positive).
    n_boot : int, default 1000
        Number of simulated null draws. The global p-value resolves ``1 /
        n_boot`` and the Bonferroni-adjusted outlier p-values ``J / n_boot``;
        a warning fires when ``n_boot < J / sig_threshold``.
    sig_threshold : float, default 0.05
        Level of the global and outlier tests.
    seed : int, optional
        Seed for the simulated null draws.

    Returns
    -------
    MRPressoResult
        IVW estimates use the residual-scaled standard error and a t test on
        ``n - 1`` degrees of freedom. Simulated p-values are ``k / n_boot``
        (0 means "below the Monte Carlo resolution"). ``outlier_pvalues`` is
        ``None`` when the outlier test was not run; the corrected estimate and
        the distortion test are ``None`` when no outlier was found.

    Raises
    ------
    ValueError
        On mismatched lengths, fewer than 3 variants, non-finite input,
        non-positive standard errors, ``n_boot < 1`` or ``sig_threshold``
        outside ``(0, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> bx = rng.uniform(0.2, 0.5, 15)
    >>> sx = np.full(15, 0.02)
    >>> sy = np.full(15, 0.02)
    >>> by = 0.3 * bx + rng.normal(0, 0.02, 15)
    >>> by[0] += 0.5
    >>> res = sp.mr_presso(bx, by, sx, sy, n_boot=500, seed=0)
    >>> 0 in res.outliers
    True

    Notes
    -----
    Clean-room implementation from Verbanck et al. (2018); the reference
    (R ``MRPRESSO`` 1.0, GPL-3) was used only as a black box. Its
    deterministic outputs (IVW fits, observed RSS, outlier-corrected fit,
    distortion coefficient) are matched exactly; its simulated p-values are
    compared as distributions (``docs/dev/cleanroom/mr_presso.md``).

    References
    ----------
    verbanck2018detection
    """
    bx, by, sx, sy = (
        np.asarray(a, dtype=float).ravel()
        for a in (beta_exposure, beta_outcome, se_exposure, se_outcome)
    )
    n = bx.size
    if not (by.size == sx.size == sy.size == n):
        raise ValueError("mr_presso: all four inputs must have the same length")
    if n < 3:
        raise ValueError(f"mr_presso needs at least 3 variants; got {n}")
    if not all(np.all(np.isfinite(a)) for a in (bx, by, sx, sy)):
        raise ValueError("mr_presso: inputs must be finite")
    if np.any(sx <= 0) or np.any(sy <= 0):
        raise ValueError("mr_presso: standard errors must be strictly positive")
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError(f"mr_presso: n_boot must be >= 1; got {n_boot}")
    if not 0 < sig_threshold < 1:
        raise ValueError(
            f"mr_presso: sig_threshold must lie in (0, 1); got {sig_threshold}"
        )
    if n_boot * sig_threshold < n:
        warnings.warn(
            f"mr_presso: n_boot={n_boot} cannot resolve sig_threshold="
            f"{sig_threshold:g} after the Bonferroni adjustment for {n} variants: "
            f"one exceeding draw already gives an adjusted p of {n}/{n_boot}, so "
            f"only variants that no draw exceeds can be flagged. Use n_boot >= "
            f"{int(np.ceil(n / sig_threshold))}.",
            UserWarning,
            stacklevel=2,
        )

    w = 1.0 / sy**2
    raw_b, raw_se, _, raw_p = _ivw_regression(bx, by, w)
    rss_obs, resid2 = _presso_rss(bx, by, sy)
    rss_obs_j = w * resid2

    # Null draws (global test step 3): exposure around its estimate, outcome
    # around the leave-one-out prediction; the statistic is recomputed on the
    # draws, leave-one-out slopes included.
    rng = np.random.default_rng(seed)
    pred = _loo_slopes(bx, by, w) * bx
    draw_x = bx + sx * rng.standard_normal((n_boot, n))
    draw_y = pred + sy * rng.standard_normal((n_boot, n))
    rss_null_j = w * (draw_y - _loo_slopes(draw_x, draw_y, w) * draw_x) ** 2
    global_p = float(np.mean(rss_null_j.sum(axis=1) > rss_obs))

    outlier_p: Optional[np.ndarray] = None
    outliers: List[int] = []
    if global_p < sig_threshold:
        outlier_p = np.minimum(1.0, n * np.mean(rss_null_j > rss_obs_j, axis=0))
        outliers = [int(i) for i in np.flatnonzero(outlier_p < sig_threshold)]

    cor_b = cor_se = cor_p = dist_p = dist_coef = None
    if outliers:
        keep = np.ones(n, dtype=bool)
        keep[outliers] = False
        if keep.sum() < 2:
            warnings.warn(
                f"mr_presso: {len(outliers)} of {n} variants flagged; fewer than 2 "
                "remain, so no outlier-corrected estimate is reported.",
                UserWarning,
                stacklevel=2,
            )
        else:
            cor_b, cor_se, _, cor_p = _ivw_regression(bx[keep], by[keep], w[keep])
            dist_coef = 100.0 * (raw_b - cor_b) / abs(cor_b)
            dist_p = _distortion_pvalue(bx, by, w, keep, raw_b, dist_coef, n_boot, rng)

    return MRPressoResult(
        raw_estimate=raw_b,
        raw_se=raw_se,
        raw_p=raw_p,
        outlier_corrected_estimate=cor_b,
        outlier_corrected_se=cor_se,
        outlier_corrected_p=cor_p,
        outliers=outliers,
        global_test_rss_obs=rss_obs,
        global_test_pvalue=global_p,
        distortion_p=dist_p,
        outlier_pvalues=None if outlier_p is None else [float(v) for v in outlier_p],
        distortion_coefficient=dist_coef,
    )


def _distortion_pvalue(
    bx: np.ndarray,
    by: np.ndarray,
    w: np.ndarray,
    keep: np.ndarray,
    raw_b: float,
    dist_coef: float,
    n_boot: int,
    rng: np.random.Generator,
) -> float:
    """Simulated p-value of the MR-PRESSO distortion test.

    Verbanck et al. (2018): the null replaces the ``n_out`` outliers by
    ``n_out`` variants drawn with replacement from the non-outliers, keeping
    ``J`` variants. For each draw the IVW slope ``b_null`` of that set plays
    the role of the corrected estimate, ``D_null = 100 (b_raw - b_null) /
    |b_null|``, and the p-value is the share of draws with ``|D_null| >
    |D_obs|``.
    """
    non = np.flatnonzero(keep)
    n_out = bx.size - non.size
    extra = rng.choice(non, size=(n_boot, n_out), replace=True)
    sxy_non = float(np.sum(w[non] * bx[non] * by[non]))
    sxx_non = float(np.sum(w[non] * bx[non] ** 2))
    sxy = sxy_non + np.sum(w[extra] * bx[extra] * by[extra], axis=1)
    sxx = sxx_non + np.sum(w[extra] * bx[extra] ** 2, axis=1)
    b_null = sxy / sxx
    d_null = 100.0 * (raw_b - b_null) / np.abs(b_null)
    return float(np.mean(np.abs(d_null) > abs(dist_coef)))
