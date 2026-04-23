"""
Multi-arm causal forest (multi-arm CATE via doubly-robust random
forest regression, Athey-Wager 2021 style).

For :math:`K` discrete treatments :math:`\\{0, 1, ..., K-1\\}`, we
estimate the CATE of arm :math:`k` vs. baseline 0:

.. math::

    \\tau_k(x) = E[Y(k) - Y(0) \\mid X = x].

The estimator is a stacked extension of the standard binary causal
forest: for each arm :math:`k \\ge 1` we build AIPW pseudo-outcomes
using the multinomial propensity :math:`\\pi_k(x) = P(W = k \\mid X)`,
fit a random forest on :math:`(X, \\tilde Y_k)`, and read off
:math:`\\hat \\tau_k(x)`.

The implementation uses scikit-learn's ``RandomForestRegressor`` +
``LogisticRegression(multi_class='multinomial')`` — self-contained and
fast enough for typical observational datasets.

References
----------
Athey, S., Tibshirani, J., & Wager, S. (2019). "Generalized random
forests." *Annals of Statistics*, 47(2), 1148-1178. [@athey2019surrogate]

Nie, X., Brunskill, E., & Wager, S. (2021). "Learning when-to-treat
policies." *JASA*, 116(533), 392-409. [@nie2021learning]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression


@dataclass
class MultiArmForestResult:
    arms: Sequence[int]
    ate: Dict[int, float]
    ate_se: Dict[int, float]
    ci: Dict[int, tuple]
    cate: Dict[int, np.ndarray]
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        rows = []
        for k in self.arms:
            if k == 0:
                continue
            lo, hi = self.ci[k]
            rows.append(
                f"  arm {k} vs 0 : ATE={self.ate[k]:+.4f}  SE={self.ate_se[k]:.4f}  CI=[{lo:+.4f}, {hi:+.4f}]"
            )
        return "Multi-arm Causal Forest\n" + "\n".join(rows)

    def __repr__(self) -> str:  # pragma: no cover
        return f"MultiArmForestResult(K={len(self.arms)}, n={self.n_obs})"


def multi_arm_forest(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str],
    n_trees: int = 200,
    min_leaf: int = 5,
    max_depth: Optional[int] = None,
    propensity_bounds: tuple = (0.02, 0.98),
    random_state: int = 42,
    alpha: float = 0.05,
) -> MultiArmForestResult:
    """
    Fit a multi-arm causal forest and report the ATE against arm 0.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treat : str
        Treatment column with integer values 0, 1, ..., K-1. Arm 0 is
        the implicit reference.
    covariates : sequence of str
    n_trees : int, default 200
    min_leaf : int, default 5
    max_depth : int, optional
    propensity_bounds : (float, float)
    random_state : int, default 42
    alpha : float, default 0.05

    Returns
    -------
    MultiArmForestResult
    """
    X_cols = list(covariates)
    df = data[[y, treat] + X_cols].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(dtype=float)
    W = df[treat].to_numpy(dtype=int)
    X = df[X_cols].to_numpy(dtype=float)
    n = len(df)
    arms = sorted(set(W.tolist()))
    if 0 not in arms:
        raise ValueError("Treatment must include 0 as the reference arm")

    # Multinomial propensity
    try:
        lr = LogisticRegression(solver="lbfgs", max_iter=500)
        lr.fit(X, W)
        probs = lr.predict_proba(X)
    except Exception:
        # fall back to empirical marginals
        probs = np.tile(
            np.bincount(W, minlength=max(arms) + 1) / n,
            (n, 1),
        )
    # map column index to class label
    class_index = {lab: i for i, lab in enumerate(lr.classes_)}
    pi = {k: np.clip(probs[:, class_index[k]], *propensity_bounds) for k in arms}

    # Outcome regressions per arm
    mu = {}
    for k in arms:
        mask = W == k
        rf = RandomForestRegressor(
            n_estimators=n_trees, min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=random_state,
            bootstrap=True, n_jobs=-1,
        )
        rf.fit(X[mask], Y[mask])
        mu[k] = rf.predict(X)

    ate = {}
    ate_se = {}
    ci = {}
    cate = {}
    for k in arms:
        if k == 0:
            continue
        # AIPW pseudo-outcome for tau_k
        ind_k = (W == k).astype(float)
        ind_0 = (W == 0).astype(float)
        pseudo = (mu[k] - mu[0]) + ind_k * (Y - mu[k]) / pi[k] - ind_0 * (Y - mu[0]) / pi[0]
        # CATE forest on pseudo
        rf = RandomForestRegressor(
            n_estimators=n_trees, min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=random_state,
            bootstrap=True, n_jobs=-1,
        )
        rf.fit(X, pseudo)
        cate[k] = rf.predict(X)
        ate[k] = float(np.mean(pseudo))
        se = float(np.std(pseudo, ddof=1) / np.sqrt(n))
        ate_se[k] = se
        crit = float(stats.norm.ppf(1 - alpha / 2))
        ci[k] = (ate[k] - crit * se, ate[k] + crit * se)

    return MultiArmForestResult(
        arms=arms,
        ate=ate,
        ate_se=ate_se,
        ci=ci,
        cate=cate,
        n_obs=n,
        detail={"propensity_ranges": {k: (float(pi[k].min()), float(pi[k].max())) for k in arms}},
    )


__all__ = ["multi_arm_forest", "MultiArmForestResult"]
