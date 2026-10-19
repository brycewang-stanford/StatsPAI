"""
Worst-case (Manski) bounds on the ATE with cross-fitted plug-in nuisances.

With an outcome known to lie in ``[y_min, y_max]`` and no other
assumption, the sharp identified set for :math:`E[Y(1) - Y(0)]` is the
Manski (1990) interval. Writing it conditionally on covariates,

.. math::

   L(x) &= π(x) μ_1(x) + (1-π(x)) y_{\\min}
          - [π(x) y_{\\max} + (1-π(x)) μ_0(x)], \\\\
   U(x) &= π(x) μ_1(x) + (1-π(x)) y_{\\max}
          - [π(x) y_{\\min} + (1-π(x)) μ_0(x)],

with :math:`π(x) = P(A=1 | X=x)` and :math:`μ_a(x) = E[Y | A=a, X=x]`,
and averaging over X gives back exactly the unconditional Manski bounds:
:math:`E[π(X) μ_1(X)] = E[A Y]` by iterated expectations. Covariates
therefore do **not** tighten worst-case bounds, and the width is
``y_max - y_min`` for every data set (pointwise and on average).
This function estimates the same interval through cross-fitted
:math:`\\hat π, \\hat μ_a` (random forest or gradient boosting by default);
the difference from the plain-means Manski estimate (``center_shift``) is
finite-sample estimation noise that vanishes as n grows, not
identifying information. Narrower bounds need narrower ``y_min`` /
``y_max`` or an additional assumption (monotone treatment response,
monotone selection, an instrument; see ``sp.manski_bounds``,
``sp.lee_bounds``, ``sp.iv_bounds``).

Before 2026-10-19 this docstring described the interval as "ML-enhanced",
adaptive and tightening as the outcome models improve, and cited two
papers that do not support that claim; none of it was true of the
computation.

The reported band takes the ``alpha/2`` quantile of the bootstrapped lower
bound and the ``1 - alpha/2`` quantile of the upper one, resampling the
cross-fitted pointwise bounds with the nuisance fits held fixed.

References
----------
[@manski1990nonparametric]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin


@dataclass
class MLBoundsResult(ResultProtocolMixin):
    """ATE bounds produced by :func:`ml_bounds`.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=300), "d": rng.integers(0, 2, 300)})
    >>> df["y"] = df.x + 0.5 * df.d + rng.normal(size=300)
    >>> res = sp.ml_bounds(df, y="y", treat="d", covariates=["x"],
    ...                    n_splits=2, n_bootstrap=20, random_state=0)
    >>> isinstance(res, sp.MLBoundsResult)
    True
    """

    lower: float
    upper: float
    lower_ci: Tuple[float, float]
    upper_ci: Tuple[float, float]
    y_range: Tuple[float, float]
    manski_lower: float
    manski_upper: float
    adaptive_lower: float  # raw plug-in (no CI inflation)
    adaptive_upper: float
    n_obs: int
    alpha: float
    learner: str
    method: str = "Worst-case (Manski) ATE bounds, cross-fitted plug-in"

    def center_shift(self) -> float:
        """Midpoint difference from the plain-means Manski estimate.

        Both estimate the same interval (by iterated expectations the
        covariate-averaged bounds are the Manski bounds), so this is
        finite-sample estimation noise of the cross-fitted nuisances and
        tends to zero as n grows. It is not identifying information.
        """
        mid_manski = 0.5 * (self.manski_lower + self.manski_upper)
        mid_ml = 0.5 * (self.adaptive_lower + self.adaptive_upper)
        return float(mid_ml - mid_manski)

    def summary(self) -> str:
        return (
            f"{self.method}\n"
            f"  N                : {self.n_obs}\n"
            f"  Y bounds         : [{self.y_range[0]:.4f}, {self.y_range[1]:.4f}]\n"
            f"  ML learner       : {self.learner}\n"
            f"  Classical Manski : [{self.manski_lower:.4f}, "
            f"{self.manski_upper:.4f}]   width="
            f"{self.manski_upper - self.manski_lower:.4f}\n"
            f"  Cross-fit plug-in: [{self.adaptive_lower:.4f}, "
            f"{self.adaptive_upper:.4f}]   width="
            f"{self.adaptive_upper - self.adaptive_lower:.4f}\n"
            f"  Bootstrap-inflated {100 * (1 - self.alpha):.0f}% band :\n"
            f"      [{self.lower:.4f}, {self.upper:.4f}]\n"
            f"  Midpoint difference (plug-in - plain means, estimation "
            f"noise) : {self.center_shift():+.4f}\n"
            f"  Note: both rows estimate the same worst-case interval, whose "
            f"width is\n"
            f"        y_max - y_min; covariates do not narrow it. Tighter "
            f"`y_min` / `y_max`\n"
            f"        or an extra assumption (sp.manski_bounds, sp.lee_bounds) "
            f"does."
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════
#  Core estimator
# ═══════════════════════════════════════════════════════════════════════


def _cross_fit_mu(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    a: int,
    base: BaseEstimator,
    n_splits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Cross-fit μ_a(x) = E[Y|A=a, X=x] on the full covariate matrix.
    Returns a length-n vector of out-of-fold predictions for every unit.
    """
    n = len(Y)
    pred: np.ndarray = np.full(n, np.nan)
    kf = KFold(
        n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1))
    )
    idx_all = np.arange(n)
    for train_idx, test_idx in kf.split(idx_all):
        mask = T[train_idx] == a
        if mask.sum() < 5:
            # Too few arm-a units IN THIS FOLD — fall back to the
            # in-fold arm-a mean (preserving cross-fitting honesty
            # by never using out-of-fold information).
            if mask.sum() >= 1:
                fill = float(np.mean(Y[train_idx][mask]))
            else:
                fill = float(np.mean(Y[train_idx]))
            pred[test_idx] = fill
            continue
        m = clone(base)
        m.fit(X[train_idx][mask], Y[train_idx][mask])
        pred[test_idx] = m.predict(X[test_idx])
    if np.any(np.isnan(pred)):
        # Should only happen if the KFold splitter leaves a point
        # uncovered (e.g. n < n_splits). Conservative fallback:
        # replicate the last observed prediction.
        last = pred[~np.isnan(pred)]
        fill = float(last[-1]) if last.size else 0.0
        pred = np.where(np.isnan(pred), fill, pred)
    return pred


def _cross_fit_pi(
    X: np.ndarray,
    T: np.ndarray,
    n_splits: int,
    rng: np.random.Generator,
    clip: Tuple[float, float] = (0.01, 0.99),
) -> np.ndarray:
    """Cross-fit propensity π(x) = P(A=1 | X=x)."""
    n = len(T)
    pred: np.ndarray = np.full(n, np.nan)
    kf = KFold(
        n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1))
    )
    for train_idx, test_idx in kf.split(np.arange(n)):
        if np.unique(T[train_idx]).size < 2:
            pred[test_idx] = float(np.mean(T[train_idx]))
            continue
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        m.fit(X[train_idx], T[train_idx])
        pred[test_idx] = m.predict_proba(X[test_idx])[:, 1]
    if np.any(np.isnan(pred)):
        pred = np.where(np.isnan(pred), float(np.mean(T)), pred)
    return np.clip(pred, *clip)


@accepts_aliases(seed="random_state")
def ml_bounds(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    *,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    learner: str = "random_forest",
    custom_learner: Optional[BaseEstimator] = None,
    n_splits: int = 5,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> MLBoundsResult:
    """
    Worst-case (Manski) bounds on the ATE, estimated with cross-fitted ML.

    The unobserved counterfactual of each arm is filled with the
    ``[y_min, y_max]`` support, conditionally on covariates, and the
    pointwise bounds are averaged. By iterated expectations the result
    estimates exactly the unconditional Manski (1990) interval: its width
    is ``y_max - y_min`` whatever the learner, and covariates do not
    tighten it. Use it when a covariate-conditional version of the
    worst-case bounds is wanted for reporting; to narrow the interval,
    narrow ``y_min`` / ``y_max`` or add an assumption
    (``sp.manski_bounds`` with monotonicity, ``sp.lee_bounds``).

    Parameters
    ----------
    data : DataFrame
    y, treat : str
        Outcome and 0/1 treatment column names.
    covariates : list of str
        Covariates X used for the outcome / propensity regressions.
    y_min, y_max : float, optional
        A priori bounds on Y. Defaults to the empirical min/max. The
        identified interval has width ``y_max - y_min``, so these, not the
        learner, set how informative it is.
    learner : {"random_forest", "gradient_boosting"}, default
        "random_forest"
        Outcome-regression learner. Ignored if ``custom_learner`` is set.
    custom_learner : sklearn-like estimator, optional
        Any ``.fit()`` / ``.predict()``-compatible regressor.
    n_splits : int, default 5
        Number of cross-fitting folds.
    n_bootstrap : int, default 200
        Non-parametric bootstrap replicates for the 2-sided frequentist
        band. Set to 0 to return the raw plug-in bounds only.
    alpha : float, default 0.05
        Significance level of the band.
    random_state : int, optional

    Returns
    -------
    MLBoundsResult

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> res = sp.ml_bounds(df, y='log_wage', treat='union',
    ...                    covariates=['education', 'experience', 'tenure'],
    ...                    n_bootstrap=50, random_state=0)
    >>> # the identified width is y_max - y_min, learner or not
    >>> w = res.adaptive_upper - res.adaptive_lower
    >>> bool(abs(w - (res.y_range[1] - res.y_range[0])) < 1e-9)
    True

    References
    ----------
    [@manski1990nonparametric]

    See also
    --------
    sp.lee_bounds, sp.manski_bounds, sp.horowitz_manski : classical
        Manski-type bounds without ML.
    """
    df = data[[y, treat] + list(covariates)].dropna().copy()
    Y = df[y].to_numpy(dtype=float)
    T = df[treat].to_numpy(dtype=int)
    X = df[list(covariates)].to_numpy(dtype=float)
    n = len(Y)
    if set(np.unique(T).tolist()) - {0, 1}:
        raise ValueError(f"`{treat}` must be binary 0/1.")

    y_lo = float(y_min) if y_min is not None else float(Y.min())
    y_hi = float(y_max) if y_max is not None else float(Y.max())
    if y_hi <= y_lo:
        raise ValueError("y_max must be strictly greater than y_min.")

    # Pick learner
    if custom_learner is not None:
        base = custom_learner
        lname = type(base).__name__
    elif learner == "random_forest":
        base = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=random_state,
        )
        lname = "RandomForestRegressor"
    elif learner == "gradient_boosting":
        base = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            random_state=random_state,
        )
        lname = "GradientBoostingRegressor"
    else:
        raise ValueError(f"Unknown learner: {learner!r}")

    rng = np.random.default_rng(random_state)

    # ── Cross-fitted nuisance functions ───────────────────────────────
    mu1 = _cross_fit_mu(X, Y, T, a=1, base=base, n_splits=n_splits, rng=rng)
    mu0 = _cross_fit_mu(X, Y, T, a=0, base=base, n_splits=n_splits, rng=rng)
    pi_x = _cross_fit_pi(X, T, n_splits=n_splits, rng=rng)

    # ── Pointwise worst-case bounds on τ(x) ───────────────────────────
    # Observed-arm prediction is plugged in; the counterfactual is filled
    # with the worst-case support. Averaged over X these estimate the
    # unconditional Manski bounds (iterated expectations).
    L_x = pi_x * mu1 + (1 - pi_x) * y_lo - (pi_x * y_hi + (1 - pi_x) * mu0)
    U_x = pi_x * mu1 + (1 - pi_x) * y_hi - (pi_x * y_lo + (1 - pi_x) * mu0)

    adaptive_lower = float(np.mean(L_x))
    adaptive_upper = float(np.mean(U_x))

    # Classical (no-ML) Manski bounds
    mu1_marg = float(np.mean(Y[T == 1])) if (T == 1).any() else y_lo
    mu0_marg = float(np.mean(Y[T == 0])) if (T == 0).any() else y_hi
    p_treated = float(np.mean(T))
    manski_lower = (
        p_treated * mu1_marg
        + (1 - p_treated) * y_lo
        - (p_treated * y_hi + (1 - p_treated) * mu0_marg)
    )
    manski_upper = (
        p_treated * mu1_marg
        + (1 - p_treated) * y_hi
        - (p_treated * y_lo + (1 - p_treated) * mu0_marg)
    )

    # ── Bootstrap inflation for finite-sample coverage ────────────────
    if n_bootstrap > 0:
        L_boot = np.empty(n_bootstrap)
        U_boot = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            L_boot[b] = float(np.mean(L_x[idx]))
            U_boot[b] = float(np.mean(U_x[idx]))
        # Two-sided band: lower endpoint uses the α/2 quantile of L,
        # upper uses the 1-α/2 quantile of U.
        lower_ci_lo = float(np.quantile(L_boot, alpha / 2))
        lower_ci_hi = float(np.quantile(L_boot, 1 - alpha / 2))
        upper_ci_lo = float(np.quantile(U_boot, alpha / 2))
        upper_ci_hi = float(np.quantile(U_boot, 1 - alpha / 2))
        lower_band = lower_ci_lo
        upper_band = upper_ci_hi
    else:
        lower_ci_lo = lower_ci_hi = adaptive_lower
        upper_ci_lo = upper_ci_hi = adaptive_upper
        lower_band = adaptive_lower
        upper_band = adaptive_upper

    return MLBoundsResult(
        lower=lower_band,
        upper=upper_band,
        lower_ci=(lower_ci_lo, lower_ci_hi),
        upper_ci=(upper_ci_lo, upper_ci_hi),
        y_range=(y_lo, y_hi),
        manski_lower=float(manski_lower),
        manski_upper=float(manski_upper),
        adaptive_lower=adaptive_lower,
        adaptive_upper=adaptive_upper,
        n_obs=n,
        alpha=alpha,
        learner=lname,
    )


__all__ = ["ml_bounds", "MLBoundsResult"]
