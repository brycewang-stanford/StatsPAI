"""
Overlap weights (Li, Morgan, Zaslavsky 2018).

Overlap weights target the "average treatment effect among the overlap
population" (ATO) — those observations with genuine equipoise. The
weight function is

    w_i = 1 - e(X_i)    for treated  i   (T_i=1)
    w_i =     e(X_i)    for control  i   (T_i=0)

which is proportional to the "tilting" that minimises the variance of
the resulting weighted treatment-effect estimator subject to exact
covariate balance on the moments used to fit ``e(·)`` (Li et al. 2018,
Theorem 1 & 3). Overlap weights:

- are bounded in [0, 1] — no extreme weights from small propensity
  scores, so results are stable even with poor overlap;
- exactly balance the log-odds covariates when ``e(X)`` is a logit fit;
- target ATO, not ATE, and should be interpreted accordingly.

References
----------
Li, F., Morgan, K.L., Zaslavsky, A.M. (2018). "Balancing Covariates via
Propensity Score Weighting." JASA, 113(521), 390-400.

Li, F., Thomas, L.E., Li, F. (2019). "Addressing Extreme Propensity
Scores via the Overlap Weights." American Journal of Epidemiology,
188(1), 250-257.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility
from .ps_diagnostics import balance_diagnostics


@_expands_categorical("covariates")
def overlap_weights(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    estimand: str = "ATO",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    trim: float = 0.0,
    *,
    se_method: str = "bootstrap",
) -> CausalResult:
    """Overlap-weight (ATO) treatment effect estimator.

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome column.
    treat : str
        Binary 0/1 treatment column.
    covariates : list of str
        Covariates for the logistic propensity-score model.
    estimand : {'ATO', 'ATE', 'ATT', 'ATC', 'matching', 'entropy'}
        Which generalized-weight scheme to use. All follow Li-Li-Li
        (2019) Table 1; 'ATO' uses the overlap weights; 'matching'
        uses the ``min(e, 1-e)`` weight; 'entropy' uses
        ``-e·log(e) - (1-e)·log(1-e)``; 'ATE/ATT/ATC' reduce to
        standard IPW for comparison.
    n_bootstrap : int
        Paired-sample bootstrap replications for SE.
    alpha : float
    seed : int, optional
    trim : float
        Optional clip of pscore to ``[trim, 1-trim]``. For overlap weights
        this is rarely needed — set to 0 by default.
    se_method : {'bootstrap', 'sandwich'}, default 'bootstrap'
        ``'sandwich'`` is the M-estimation standard error: the logit score
        equations and the two weighted means are stacked, so the sampling
        error of the estimated propensity score is carried into the
        effect. It is deterministic and is the standard error R
        ``PSweight::PSweight`` reports (no bootstrap). Available for every
        ``estimand`` except ``'matching'``, whose tilting function has a
        kink, and without ``trim``. ``'bootstrap'`` resamples rows
        ``n_bootstrap`` times.

    Returns
    -------
    CausalResult
        ``.estimate`` targets the named estimand; ``.model_info`` stores
        the weight summary, effective sample size, and pscore diagnostics.
        ``ess_treated`` and ``ess_control`` are the Kish effective sample
        sizes ``(sum w)^2 / sum w^2`` of the weights of each arm (what R
        ``WeightIt`` / ``cobalt`` report) and ``effective_sample_size``
        their sum; ``tilt_ess`` is the same formula applied to the
        tilting function ``h(e)`` over all rows, which measures the size
        of the target population and not the precision of the weighted
        means.

        .. versionchanged:: 1.44
           ``effective_sample_size`` used to hold ``tilt_ess``, which
           equals ``n`` for ``estimand='ATE'`` whatever the weights.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> result = sp.overlap_weights(df, y='log_wage', treat='union',
    ...                             covariates=['education', 'experience',
    ...                                         'tenure'],
    ...                             n_bootstrap=100, seed=42)
    >>> result.estimand, round(result.estimate, 3)
    ('ATO', 0.188)
    >>> report = result.summary()
    >>> ess = result.model_info['effective_sample_size']
    >>> bool(0 < ess <= len(df))
    True

    >>> # min(e, 1-e) matching weights instead of overlap weights
    >>> result = sp.overlap_weights(df, y='log_wage', treat='union',
    ...                             covariates=['education', 'experience',
    ...                                         'tenure'],
    ...                             estimand='matching',
    ...                             n_bootstrap=100, seed=42)
    """
    estimand = estimand.upper()
    valid = {"ATO", "ATE", "ATT", "ATC", "MATCHING", "ENTROPY"}
    if estimand not in valid:
        raise ValueError(f"estimand must be one of {valid}, got {estimand!r}")
    if se_method not in ("bootstrap", "sandwich"):
        raise MethodIncompatibility(
            f"overlap_weights: se_method must be 'bootstrap' or 'sandwich', "
            f"got {se_method!r}.",
            recovery_hint="Use se_method='sandwich' for the analytic "
            "(M-estimation) standard error.",
        )
    if se_method == "sandwich" and (estimand == "MATCHING" or trim > 0):
        raise MethodIncompatibility(
            "overlap_weights: the sandwich standard error differentiates the "
            "weights with respect to the propensity score; "
            + (
                "the matching weight min(e, 1 - e) has a kink at 0.5."
                if estimand == "MATCHING"
                else "a clipped score (trim > 0) is not differentiable at the "
                "bounds."
            ),
            recovery_hint="Use se_method='bootstrap' here.",
        )

    rng = np.random.default_rng(seed)

    df = data[[y, treat] + list(covariates)].dropna().copy()
    Y = df[y].to_numpy(dtype=np.float64)
    T = df[treat].to_numpy(dtype=np.float64)
    X = df[covariates].to_numpy(dtype=np.float64)
    n = len(Y)
    if not set(np.unique(T)).issubset({0, 1}):
        raise ValueError(f"Treatment '{treat}' must be binary.")

    def _estimate(
        X_b: np.ndarray,
        T_b: np.ndarray,
        Y_b: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        ps = _logit_pscore(X_b, T_b)
        if trim > 0:
            ps = np.clip(ps, trim, 1 - trim)
        h = _tilt(ps, estimand)
        w1 = T_b * h / ps
        w0 = (1 - T_b) * h / (1 - ps)
        w1 /= w1.sum() + 1e-30
        w0 /= w0.sum() + 1e-30
        return float(np.sum(w1 * Y_b) - np.sum(w0 * Y_b)), ps, h

    est, ps, h = _estimate(X, T, Y)

    if se_method == "sandwich":
        se = _sandwich_se(X, T, Y, ps, h, estimand)
    else:
        boot = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot[b] = _estimate(X[idx], T[idx], Y[idx])[0]
        se = float(np.std(boot, ddof=1))
    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (est - z * se, est + z * se)
    pval = float(2 * sp_stats.norm.sf(abs(est) / se)) if se > 0 else 1.0

    full_weights = T * h / ps + (1 - T) * h / (1 - ps)

    def _kish(w: np.ndarray) -> float:
        return float(w.sum() ** 2 / (w**2).sum()) if (w**2).sum() > 0 else np.nan

    # Kish effective sample size of the weights each arm's mean is taken
    # with (the "Weighted" row of WeightIt / cobalt); the overall figure is
    # their sum.
    ess_treated = _kish(full_weights[T == 1])
    ess_control = _kish(full_weights[T == 0])
    ess = ess_treated + ess_control
    balance = balance_diagnostics(
        df,
        treatment=treat,
        covariates=covariates,
        weights=full_weights,
        ps=pd.Series(ps, index=df.index),
    )
    model_info = {
        "model_type": (
            "OverlapWeights" if estimand == "ATO" else f"PSWeights_{estimand}"
        ),
        "estimand": estimand,
        "n_treated": int(T.sum()),
        "n_control": int((1 - T).sum()),
        "pscore_mean": float(ps.mean()),
        "pscore_min": float(ps.min()),
        "pscore_max": float(ps.max()),
        "effective_sample_size": float(ess),
        "ess_treated": ess_treated,
        "ess_control": ess_control,
        "tilt_sum": float(h.sum()),
        "tilt_ess": _kish(h),
        "n_bootstrap": n_bootstrap,
        "balance_summary": balance.summary_stats,
    }
    if se_method == "sandwich":
        model_info["n_bootstrap"] = None
        model_info["se_method"] = "sandwich"
    return CausalResult(
        method=f"Overlap Weights ({estimand})",
        estimand=estimand,
        estimate=est,
        se=se,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        # the covariate names are the index of the balance table; as a
        # column they are printed by ``summary()`` and kept by the exports
        detail=balance.table.reset_index(),
        model_info=model_info,
    )


def _logit_pscore(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Unpenalised logistic propensity score, shared with :func:`sp.match`.

    This used to fit ``sklearn.LogisticRegression(C=1e6)``, which is a
    *penalised* likelihood however large ``C`` is.  Two consequences, both
    undesirable: the same data got a different propensity score from
    ``sp.overlap_weights`` than from ``sp.match`` / ``sp.psmatch2`` (which
    have always used the Newton-Raphson MLE), and the overlap-weight theory
    being implemented (Li, Morgan & Zaslavsky 2018) is derived for the
    maximum-likelihood fit -- its exact-balance property holds at the score
    equations of the *unpenalised* logit.

    Measured against R's ``glm(family = binomial)`` on the committed
    fixture: the penalised fit was off by 8.5e-06 on the propensity score
    and drove the ATO/ATE/ATT/ATC estimates ~1e-6 away from
    ``WeightIt::weightit``; the MLE agrees to 2.6e-14 and 1e-14
    respectively.
    """
    from .match import MatchEstimator

    p = MatchEstimator._logit_propensity(X, T, poly=1)
    return np.asarray(np.clip(p, 1e-8, 1 - 1e-8), dtype=float)


def _tilt_derivative(e: np.ndarray, estimand: str) -> np.ndarray:
    """``dh / de`` of the tilting function."""
    if estimand == "ATE":
        return np.zeros_like(e)
    if estimand == "ATT":
        return np.ones_like(e)
    if estimand == "ATC":
        return -np.ones_like(e)
    if estimand == "ATO":
        return np.asarray(1.0 - 2.0 * e, dtype=float)
    if estimand == "ENTROPY":
        return np.asarray(np.log((1.0 - e) / e), dtype=float)
    raise ValueError(estimand)


def _sandwich_se(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    e: np.ndarray,
    h: np.ndarray,
    estimand: str,
) -> float:
    """M-estimation standard error of the weighted difference in means.

    The stacked estimating equations are the logit score ``z (T - e)`` and
    ``T (h/e) (Y - mu1)``, ``(1 - T) (h/(1 - e)) (Y - mu0)``. Solving the
    sandwich for the last two gives the influence function

        psi_i = [w1_i (Y_i - mu1) + G1' l_i] / mean(w1)
              - [w0_i (Y_i - mu0) + G0' l_i] / mean(w0),

    with ``l_i`` the influence function of the logit coefficients and
    ``G`` the mean derivative of each weighted residual with respect to
    them. The variance is ``mean(psi^2) / n``.
    """
    n = len(Y)
    Z = np.column_stack([np.ones(n), X])
    dh = _tilt_derivative(e, estimand)
    w1 = T * h / e
    w0 = (1 - T) * h / (1 - e)
    mu1 = np.sum(w1 * Y) / np.sum(w1)
    mu0 = np.sum(w0 * Y) / np.sum(w0)
    # d w / d(z'gamma) = (dw / de) e (1 - e)
    dw1 = T * (dh * e - h) / e**2 * e * (1 - e)
    dw0 = (1 - T) * (dh * (1 - e) + h) / (1 - e) ** 2 * e * (1 - e)
    info = (Z * (e * (1 - e))[:, None]).T @ Z / n
    l_gamma = (Z * (T - e)[:, None]) @ np.linalg.inv(info)
    G1 = (Z * (dw1 * (Y - mu1))[:, None]).mean(axis=0)
    G0 = (Z * (dw0 * (Y - mu0))[:, None]).mean(axis=0)
    psi = (w1 * (Y - mu1) + l_gamma @ G1) / w1.mean() - (
        w0 * (Y - mu0) + l_gamma @ G0
    ) / w0.mean()
    return float(np.sqrt(np.mean(psi**2) / n))


def _tilt(e: np.ndarray, estimand: str) -> np.ndarray:
    """Tilting function h(e) implementing the target population
    (Li-Morgan-Zaslavsky 2018, Table 1).
    """
    if estimand == "ATE":
        return np.asarray(np.ones_like(e), dtype=float)
    if estimand == "ATT":
        return np.asarray(e, dtype=float)
    if estimand == "ATC":
        return np.asarray(1.0 - e, dtype=float)
    if estimand == "ATO":
        return np.asarray(e * (1.0 - e), dtype=float)  # overlap weight
    if estimand == "MATCHING":
        return np.asarray(np.minimum(e, 1.0 - e), dtype=float)
    if estimand == "ENTROPY":
        entropy = e * np.log(np.clip(e, 1e-12, 1)) + (1 - e) * np.log(
            np.clip(1 - e, 1e-12, 1)
        )
        return np.asarray(-entropy, dtype=float)
    raise ValueError(estimand)


__all__ = ["overlap_weights"]
