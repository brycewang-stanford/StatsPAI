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

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


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

    Returns
    -------
    CausalResult
        ``.estimate`` targets the named estimand; ``.model_info`` stores
        the weight summary, effective sample size, and pscore diagnostics.
    """
    estimand = estimand.upper()
    valid = {"ATO", "ATE", "ATT", "ATC", "MATCHING", "ENTROPY"}
    if estimand not in valid:
        raise ValueError(f"estimand must be one of {valid}, got {estimand!r}")

    rng = np.random.default_rng(seed)

    df = data[[y, treat] + list(covariates)].dropna().copy()
    Y = df[y].to_numpy(dtype=np.float64)
    T = df[treat].to_numpy(dtype=np.float64)
    X = df[covariates].to_numpy(dtype=np.float64)
    n = len(Y)
    if not set(np.unique(T)).issubset({0, 1}):
        raise ValueError(f"Treatment '{treat}' must be binary.")

    def _estimate(X_b, T_b, Y_b):
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

    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b] = _estimate(X[idx], T[idx], Y[idx])[0]
    se = float(np.std(boot, ddof=1))
    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (est - z * se, est + z * se)
    pval = float(2 * (1 - sp_stats.norm.cdf(abs(est) / se))) if se > 0 else 1.0

    ess = (h.sum()) ** 2 / (h ** 2).sum() if (h ** 2).sum() > 0 else np.nan
    model_info = {
        "model_type": "OverlapWeights" if estimand == "ATO" else f"PSWeights_{estimand}",
        "estimand": estimand,
        "n_treated": int(T.sum()),
        "n_control": int((1 - T).sum()),
        "pscore_mean": float(ps.mean()),
        "pscore_min": float(ps.min()),
        "pscore_max": float(ps.max()),
        "effective_sample_size": float(ess),
        "tilt_sum": float(h.sum()),
        "n_bootstrap": n_bootstrap,
    }
    return CausalResult(
        method=f"Overlap Weights ({estimand})",
        estimand=estimand,
        estimate=est,
        se=se,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
    )


def _logit_pscore(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=1000, solver="lbfgs", C=1e6)
    m.fit(X, T)
    p = m.predict_proba(X)[:, 1]
    return np.clip(p, 1e-8, 1 - 1e-8)


def _tilt(e: np.ndarray, estimand: str) -> np.ndarray:
    """Tilting function h(e) implementing the target population
    (Li-Morgan-Zaslavsky 2018, Table 1).
    """
    if estimand == "ATE":
        return np.ones_like(e)
    if estimand == "ATT":
        return e
    if estimand == "ATC":
        return 1.0 - e
    if estimand == "ATO":
        return e * (1.0 - e)  # overlap weight
    if estimand == "MATCHING":
        return np.minimum(e, 1.0 - e)
    if estimand == "ENTROPY":
        return -(e * np.log(np.clip(e, 1e-12, 1)) + (1 - e) * np.log(np.clip(1 - e, 1e-12, 1)))
    raise ValueError(estimand)


__all__ = ["overlap_weights"]
