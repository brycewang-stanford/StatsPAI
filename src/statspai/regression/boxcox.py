"""
Box-Cox transformation of the outcome (Box and Cox 1964).

``sp.boxcox`` estimates the power ``lambda`` for which ``(y^lambda - 1) /
lambda`` (``log y`` at zero) is best described by a linear model with
normal, homoskedastic errors. It reports the maximum likelihood estimate,
a profile-likelihood interval and likelihood-ratio tests of the three
textbook transformations (reciprocal, log, none). The profile is that of R
``MASS::boxcox``; the estimate, log likelihood and tests are those of Stata
``boxcox y x, model(lhsonly)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._result_serialize import ResultProtocolMixin
from ..core.utils import create_design_matrices
from ..exceptions import DataInsufficient, MethodIncompatibility


def boxcox_transform(y: Any, lam: float) -> np.ndarray:
    """``(y^lam - 1) / lam``, and ``log(y)`` in the limit ``lam -> 0``."""
    yv = np.asarray(y, dtype=float)
    if abs(lam) < 1e-8:
        return np.asarray(np.log(yv), dtype=float)
    return np.asarray(np.expm1(lam * np.log(yv)) / lam, dtype=float)


@dataclass
class BoxCoxResult(ResultProtocolMixin):
    """Result of :func:`boxcox`.

    Attributes
    ----------
    lambda_ : float
        Maximum likelihood estimate of the power.
    ci : tuple of float
        Profile-likelihood interval at level ``1 - alpha``.
    loglik : float
        Log likelihood at ``lambda_`` (with the Jacobian, so it is
        comparable across powers and with Stata's).
    tests : pd.DataFrame
        Likelihood-ratio tests of ``lambda = -1``, ``0`` and ``1``.
    profile : pd.DataFrame
        ``lambda`` and ``loglik`` over the requested grid.
    params : pd.Series
        Regression coefficients of the transformed outcome at ``lambda_``.
    sigma2 : float
        Maximum likelihood residual variance at ``lambda_`` (divisor n).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(2)
    >>> x = rng.normal(size=300)
    >>> df = pd.DataFrame({"x": x, "y": (2 + 0.3 * x
    ...                                  + 0.2 * rng.normal(size=300)) ** 2})
    >>> res = sp.boxcox("y ~ x", df)
    >>> isinstance(res, sp.BoxCoxResult)
    True
    >>> bool(res.ci[0] < 0.5 < res.ci[1])   # the square root is the right scale
    True
    >>> list(res.tests["lambda"])
    [-1.0, 0.0, 1.0]
    """

    _citation_keys = ("box1964analysis",)
    lambda_: float
    ci: tuple
    loglik: float
    tests: pd.DataFrame
    profile: pd.DataFrame
    params: pd.Series
    sigma2: float
    n_obs: int
    alpha: float = 0.05
    outcome: str = "y"
    formula: Optional[str] = None
    _names: List[str] = field(default_factory=list, repr=False)

    def transform(self, y: Any, lam: Optional[float] = None) -> np.ndarray:
        """Apply the transformation at ``lam`` (the estimate by default)."""
        return boxcox_transform(y, self.lambda_ if lam is None else lam)

    def summary(self) -> str:
        level = 100 * (1 - self.alpha)
        lines = [
            "=" * 60,
            f"Box-Cox transformation of {self.outcome}",
            "=" * 60,
            f"  Observations : {self.n_obs}",
            f"  lambda       : {self.lambda_:.6f}",
            f"  {level:g}% interval: [{self.ci[0]:.6f}, {self.ci[1]:.6f}]"
            "  (profile likelihood)",
            f"  Log likelihood: {self.loglik:.4f}",
            "",
            "  H0            LR chi2(1)    p-value",
        ]
        for _, row in self.tests.iterrows():
            lines.append(
                f"  lambda = {row['lambda']:>4g}  {row['chi2']:>10.3f}"
                f"    {row['pvalue']:.4f}"
            )
        lines.append("=" * 60)
        text = "\n".join(lines)
        print(text)
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lambda": float(self.lambda_),
            "ci": [float(self.ci[0]), float(self.ci[1])],
            "loglik": float(self.loglik),
            "sigma2": float(self.sigma2),
            "n_obs": int(self.n_obs),
            "tests": self.tests.to_dict(orient="records"),
            "params": {str(k): float(v) for k, v in self.params.items()},
        }


def boxcox(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[Sequence[str]] = None,
    lambdas: Optional[Sequence[float]] = None,
    bounds: tuple = (-3.0, 3.0),
    alpha: float = 0.05,
) -> BoxCoxResult:
    """
    Box-Cox power transformation of the outcome of a linear regression.

    Finds the ``lambda`` that maximises the normal profile log likelihood of
    ``y^(lambda) = x'b + e`` with ``y^(lambda) = (y^lambda - 1) / lambda``
    (``log y`` at 0). Equivalent to R ``MASS::boxcox(lm(...))`` and Stata
    ``boxcox y x, model(lhsonly)``.

    Parameters
    ----------
    formula : str, optional
        Model for the untransformed outcome, e.g. ``"duration ~ treat + age"``.
    data : pd.DataFrame
    y, x : str and list of str, optional
        Outcome and regressors, as an alternative to ``formula``. An empty
        ``x`` fits the intercept-only model (the transformation toward
        normality of ``y`` itself).
    lambdas : sequence of float, optional
        Grid on which to report the profile log likelihood (default: 121
        points across ``bounds``). The estimate itself is found by a
        bounded scalar optimiser, not on the grid.
    bounds : (float, float), default (-3, 3)
        Search interval for ``lambda``.
    alpha : float, default 0.05
        One minus the level of the profile-likelihood interval.

    Returns
    -------
    BoxCoxResult
        ``lambda_``, ``ci``, ``loglik``, the likelihood-ratio ``tests`` of
        ``lambda = -1, 0, 1``, the ``profile`` and the coefficients of the
        transformed regression.

    Notes
    -----
    The profile log likelihood is ``-(n/2) log(RSS(lambda) / n) +
    (lambda - 1) sum(log y)`` up to a constant; the second term is the
    Jacobian that makes fits at different powers comparable. An interval
    endpoint equal to a bound means the likelihood had not dropped far
    enough there; widen ``bounds``.

    The outcome must be strictly positive. The coefficients in ``params``
    are conditional on ``lambda_``: standard errors from refitting
    ``sp.regress`` on the transformed outcome ignore that it was estimated.
    A transformation chosen to make the errors look normal changes what
    the coefficients mean; for a causal estimand on the original scale,
    transform back or model the mean directly (``sp.glm``, ``sp.poisson``).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=400)
    >>> df = pd.DataFrame({"x": x, "y": np.exp(1 + 0.3 * x
    ...                                        + 0.3 * rng.normal(size=400))})
    >>> res = sp.boxcox("y ~ x", df)
    >>> bool(res.ci[0] < 0 < res.ci[1])   # the log is the right scale
    True

    References
    ----------
    [@box1964analysis]
    """
    if data is None:
        raise MethodIncompatibility("boxcox: data is required.")
    if formula is not None:
        y_df, X_df = create_design_matrices(formula, data)
        names = [str(c) for c in X_df.columns]
        yv = np.asarray(y_df, dtype=float).reshape(len(X_df), -1)[:, -1]
        X = np.asarray(X_df, dtype=float)
        y_name = str(y_df.columns[-1]) if hasattr(y_df, "columns") else "y"
    elif y is not None:
        cols = [y] + list(x or [])
        frame = data[cols].dropna()
        yv = frame[y].to_numpy(dtype=float)
        X = np.column_stack(
            [np.ones(len(frame))] + [frame[c].to_numpy(float) for c in (x or [])]
        )
        names = ["Intercept"] + list(x or [])
        y_name = y
    else:
        raise MethodIncompatibility("boxcox: provide a formula or y (and x).")
    n, k = X.shape
    if n <= k + 1:
        raise DataInsufficient(
            f"boxcox: {n} observations for {k} coefficients.",
        )
    if np.any(~np.isfinite(yv)) or np.any(yv <= 0):
        raise MethodIncompatibility(
            "boxcox: the outcome must be strictly positive.",
            recovery_hint=(
                "Shift the outcome by a known constant, or use "
                "sp.glm(family='poisson') / sp.poisson for outcomes with zeros."
            ),
            diagnostics={"n_nonpositive": int(np.sum(yv <= 0))},
        )
    lo, hi = float(bounds[0]), float(bounds[1])
    if not lo < hi:
        raise MethodIncompatibility("boxcox: bounds must be increasing.")

    Q, _ = np.linalg.qr(X)
    logy = np.log(yv)
    sum_logy = float(logy.sum())

    def rss(lam: float) -> float:
        z = boxcox_transform(yv, lam)
        r = z - Q @ (Q.T @ z)
        return float(r @ r)

    def loglik(lam: float) -> float:
        return float(
            -0.5 * n * (np.log(2 * np.pi) + 1.0 + np.log(rss(lam) / n))
            + (lam - 1.0) * sum_logy
        )

    # coarse scan, then polish inside the bracket around the best point
    scan = np.linspace(lo, hi, 61)
    vals = np.array([loglik(v) for v in scan])
    j = int(np.argmax(vals))
    a, b = scan[max(j - 1, 0)], scan[min(j + 1, len(scan) - 1)]
    opt = optimize.minimize_scalar(
        lambda v: -loglik(v), bounds=(a, b), method="bounded", options={"xatol": 1e-12}
    )
    lam_hat = float(opt.x)
    ll_hat = float(-opt.fun)
    if vals[j] > ll_hat:
        lam_hat, ll_hat = float(scan[j]), float(vals[j])

    cut = ll_hat - 0.5 * stats.chi2.ppf(1 - alpha, 1)

    def endpoint(lo_: float, hi_: float) -> float:
        f_lo, f_hi = loglik(lo_) - cut, loglik(hi_) - cut
        if f_lo * f_hi > 0:  # no crossing inside the search interval
            return lo_ if abs(f_lo) < abs(f_hi) else hi_
        return float(optimize.brentq(lambda v: loglik(v) - cut, lo_, hi_, xtol=1e-12))

    ci = (
        endpoint(lo, lam_hat) if lam_hat > lo else lo,
        endpoint(lam_hat, hi) if lam_hat < hi else hi,
    )
    rows = []
    for null in (-1.0, 0.0, 1.0):
        chi2 = max(0.0, 2.0 * (ll_hat - loglik(null)))
        rows.append(
            {
                "lambda": null,
                "loglik": loglik(null),
                "chi2": chi2,
                "pvalue": float(stats.chi2.sf(chi2, 1)),
            }
        )
    grid = (
        np.linspace(lo, hi, 121)
        if lambdas is None
        else np.asarray(lambdas, dtype=float)
    )
    profile = pd.DataFrame({"lambda": grid, "loglik": [loglik(v) for v in grid]})
    z = boxcox_transform(yv, lam_hat)
    beta = np.linalg.lstsq(X, z, rcond=None)[0]
    return BoxCoxResult(
        lambda_=lam_hat,
        ci=ci,
        loglik=ll_hat,
        tests=pd.DataFrame(rows),
        profile=profile,
        params=pd.Series(beta, index=names),
        sigma2=rss(lam_hat) / n,
        n_obs=int(n),
        alpha=alpha,
        outcome=y_name,
        formula=formula,
        _names=names,
    )
