"""
Distribution-Valued RDD (arXiv 2504.03992, 2025).

Estimates the RDD effect on the entire conditional distribution of Y
at the cutoff, returning the effect on each quantile of Y rather
than the mean. Equivalent to running the standard local-linear
estimator with the indicator 1{Y ≤ y} as the dependent variable for
a grid of y values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient
from ._core import _kernel_fn


@dataclass
class DistRDResult(ResultProtocolMixin):
    """RDD effect on the distribution of the outcome at the cutoff.

    Attributes
    ----------
    quantiles : np.ndarray
        Quantile levels at which the effects are reported.
    qte : np.ndarray
        Quantile treatment effect at the cutoff: the ``q``-quantile of the
        outcome just right of the cutoff minus the one just left of it, in
        units of the outcome.
    se : np.ndarray
        Bootstrap standard error of each ``qte``.
    bandwidth : float
        Bandwidth used in the local-linear fits.
    n_obs : int
        Number of complete observations.
    ci_lower, ci_upper : np.ndarray
        Percentile-bootstrap interval for each ``qte``.
    cdf_effect : np.ndarray
        Jump at the cutoff in ``P(Y <= y_q)``, where ``y_q`` is the pooled
        ``q``-quantile of the outcome (``y_at_quantile``). A probability,
        and negative where the treatment raises the outcome.
    cdf_se : np.ndarray
        Standard error of each ``cdf_effect``.
    y_at_quantile : np.ndarray
        The thresholds ``y_q`` behind ``cdf_effect``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 1.0 + 0.5 * x + (x >= 0) * 0.8 + rng.normal(0, 1, n)
    >>> data = pd.DataFrame({"y": y, "x": x})
    >>> res = sp.rd_distribution(data, y="y", running="x")
    >>> bool(res.qte.shape == res.quantiles.shape)
    True
    """

    quantiles: np.ndarray
    qte: np.ndarray
    se: np.ndarray
    bandwidth: float
    n_obs: int
    ci_lower: Optional[np.ndarray] = None
    ci_upper: Optional[np.ndarray] = None
    cdf_effect: Optional[np.ndarray] = None
    cdf_se: Optional[np.ndarray] = None
    y_at_quantile: Optional[np.ndarray] = None

    def summary(self) -> str:
        rows = [
            "Distribution-Valued RDD",
            "=" * 58,
            "  Quantile   QTE        SE       CDF jump   SE",
        ]
        cdf = (
            self.cdf_effect
            if self.cdf_effect is not None
            else [np.nan] * len(self.quantiles)
        )
        cdf_se = (
            self.cdf_se if self.cdf_se is not None else [np.nan] * len(self.quantiles)
        )
        for q, e, s_, c_, cs in zip(self.quantiles, self.qte, self.se, cdf, cdf_se):
            rows.append(f"  {q:.2f}     {e:+.4f}   {s_:.4f}   {c_:+.4f}    {cs:.4f}")
        return "\n".join(rows)


def _boundary_weights(r: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Equivalent-kernel weights of the local-linear intercept at ``r = 0``.

    The intercept of a weighted regression of ``v`` on ``(1, r)`` is
    ``sum_i l_i v_i`` with ``l`` returned here, so the local-linear estimate
    of ``P(Y <= y | R = 0)`` on one side is a weighted empirical CDF.
    """
    s0, s1, s2 = w.sum(), (w * r).sum(), (w * r * r).sum()
    det = s0 * s2 - s1 * s1
    if not np.isfinite(det) or abs(det) < 1e-12 * max(s0 * s2, 1e-300):
        return w / s0
    return w * (s2 - s1 * r) / det


def _side_quantiles(
    y: np.ndarray, r: np.ndarray, w: np.ndarray, levels: np.ndarray
) -> np.ndarray:
    """Quantiles of the outcome at the cutoff from one side.

    Sorts the outcomes, accumulates the boundary weights into the
    local-linear CDF, makes it monotone (the weights can be negative near
    the far edge of the window) and inverts it by linear interpolation.
    """
    order = np.argsort(y, kind="mergesort")
    ys = y[order]
    cdf = np.cumsum(_boundary_weights(r, w)[order])
    cdf = np.clip(np.maximum.accumulate(cdf), 0.0, 1.0)
    # First outcome value at which the CDF reaches each level, interpolated.
    out = np.empty(len(levels))
    for j, q in enumerate(levels):
        k = int(np.searchsorted(cdf, q, side="left"))
        if k <= 0:
            out[j] = ys[0]
        elif k >= len(ys):
            out[j] = ys[-1]
        else:
            lo, hi = cdf[k - 1], cdf[k]
            frac = 0.0 if hi <= lo else (q - lo) / (hi - lo)
            out[j] = ys[k - 1] + frac * (ys[k] - ys[k - 1])
    return out


def rd_distribution(
    data: pd.DataFrame,
    y: str,
    running: str,
    cutoff: float = 0.0,
    quantiles: Optional[np.ndarray] = None,
    bandwidth: Optional[float] = None,
    kernel: str = "triangular",
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> DistRDResult:
    """
    Distribution-valued sharp RDD.

    Estimates the distribution of the outcome just left and just right of
    the cutoff by local-linear regression of ``1{Y <= y}`` on the running
    variable, and reports the discontinuity two ways: as a quantile
    treatment effect (``qte``, the horizontal distance between the two
    distributions, in units of the outcome) and as the jump in the CDF at
    the pooled quantiles (``cdf_effect``, the vertical distance, a
    probability).

    Parameters
    ----------
    data : pd.DataFrame
    y, running : str
    cutoff : float
    quantiles : array-like, optional
        Defaults to (0.1, 0.25, 0.5, 0.75, 0.9).
    bandwidth : float, optional
        Defaults to the interquartile range of the running variable.
    kernel : str
    alpha : float
        Level of the percentile-bootstrap interval for ``qte``.
    n_boot : int, default 200
        Bootstrap replications for the ``qte`` standard errors.
    seed : int, default 0

    Returns
    -------
    DistRDResult
        ``qte`` with bootstrap ``se`` / ``ci_lower`` / ``ci_upper``, and
        ``cdf_effect`` with ``cdf_se``, at each requested quantile.

    Notes
    -----
    Until the 2026-10 fix ``qte`` held what is now ``cdf_effect``: a change
    in probability, negative where the treatment raises the outcome. On a
    design that shifts the outcome up by 2.0 it reported -0.48 at the
    median.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 1.0 + 0.5 * x + (x >= 0) * 0.8 + rng.normal(0, 1, n)
    >>> data = pd.DataFrame({"y": y, "x": x})
    >>> res = sp.rd_distribution(data, y="y", running="x", cutoff=0.0)
    >>> res.n_obs
    400
    >>> bool(res.qte.shape == res.se.shape == res.quantiles.shape)
    True
    >>> bool(res.ci_lower[2] < 0.8 < res.ci_upper[2])  # a 0.8 shift
    True
    >>> bool((res.cdf_effect < 0).all())  # mass moves to higher outcomes
    True
    """
    if quantiles is None:
        quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    quantiles = np.asarray(quantiles, dtype=float)
    df = data[[y, running]].dropna().reset_index(drop=True)
    R = df[running].to_numpy(float) - cutoff
    Y = df[y].to_numpy(float)
    n = len(df)
    if bandwidth is None:
        bandwidth = float(np.subtract(*np.percentile(R, [75, 25])))
    treat = (R >= 0).astype(int)
    weights = _kernel_fn(R / bandwidth, kernel)
    mask = weights > 0

    # --- Vertical distance: jump in the CDF at the pooled quantiles ---
    y_at_q = np.array([float(np.quantile(Y, q)) for q in quantiles])
    cdf_effect = np.zeros(len(quantiles))
    cdf_se = np.zeros(len(quantiles))
    for j, y_q in enumerate(y_at_q):
        ind = (Y <= y_q).astype(float)
        try:
            Xb = np.column_stack(
                [
                    np.ones(mask.sum()),
                    R[mask],
                    treat[mask],
                    R[mask] * treat[mask],
                ]
            )
            Wd = np.diag(weights[mask])
            beta = np.linalg.solve(Xb.T @ Wd @ Xb, Xb.T @ Wd @ ind[mask])
            resid = ind[mask] - Xb @ beta
            sigma2 = float(
                (weights[mask] * resid**2).sum()
                / max(weights[mask].sum() - Xb.shape[1], 1)
            )
            cov = sigma2 * np.linalg.pinv(Xb.T @ Wd @ Xb)
            cdf_effect[j] = float(beta[2])
            cdf_se[j] = float(np.sqrt(max(cov[2, 2], 0.0)))
        except np.linalg.LinAlgError:  # pragma: no cover
            cdf_effect[j] = np.nan  # pragma: no cover
            cdf_se[j] = np.nan  # pragma: no cover

    # --- Horizontal distance: quantile treatment effects at the cutoff ---
    Ym, Rm, Wm, Tm = Y[mask], R[mask], weights[mask], treat[mask].astype(bool)
    if Tm.sum() < 5 or (~Tm).sum() < 5:
        raise DataInsufficient(
            "rd_distribution needs at least 5 observations inside the "
            f"bandwidth on each side of the cutoff (got {int((~Tm).sum())} "
            f"left, {int(Tm.sum())} right)."
        )

    def _qte(idx: np.ndarray) -> np.ndarray:
        yy, rr, ww, tt = Ym[idx], Rm[idx], Wm[idx], Tm[idx]
        if tt.sum() < 5 or (~tt).sum() < 5:
            return np.full(len(quantiles), np.nan)
        right = _side_quantiles(yy[tt], rr[tt], ww[tt], quantiles)
        left = _side_quantiles(yy[~tt], rr[~tt], ww[~tt], quantiles)
        return right - left

    m = len(Ym)
    qte = _qte(np.arange(m))
    rng = np.random.default_rng(seed)
    boot = np.full((int(n_boot), len(quantiles)), np.nan)
    for b in range(int(n_boot)):
        boot[b] = _qte(rng.integers(0, m, size=m))

    from ..core._bootstrap import bootstrap_se

    se = np.array(
        [
            bootstrap_se(boot[:, j], label="rd.rd_distribution", warn=(j == 0))
            for j in range(len(quantiles))
        ]
    )
    with np.errstate(all="ignore"):
        ci_lower = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
        ci_upper = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)

    return DistRDResult(
        quantiles=quantiles,
        qte=qte,
        se=se,
        bandwidth=float(bandwidth),
        n_obs=n,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        cdf_effect=cdf_effect,
        cdf_se=cdf_se,
        y_at_quantile=y_at_q,
    )
