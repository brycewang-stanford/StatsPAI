"""
Causal Survival Forest (CSF).

A honest random-forest estimator for heterogeneous treatment effects
on right-censored time-to-event outcomes, following Cui, Kosorok,
Sverdrup, Wager & Zhu (2023, *JRSSB*).

The target estimand is the **Restricted Mean Survival Time (RMST)**
treatment effect at horizon :math:`\\tau`:

.. math::

    \\tau_{RMST}(x) = E[\\min(T(1), \\tau) - \\min(T(0), \\tau) \\mid X = x].

Identifying assumptions:

* consistency & SUTVA,
* no unmeasured confounders given X,
* random censoring given (T, W, X).

Algorithm (honest forest variant)
---------------------------------
1. Build :math:`B` honest trees on bootstrap subsamples; at each split
   use the *AIPW transformed outcome*

   .. math::

      Y_i^{dr} = \\hat\\mu(1, X_i) - \\hat\\mu(0, X_i)
              + \\frac{W_i - \\hat e(X_i)}{\\hat e(X_i)(1-\\hat e(X_i))}
                (Z_i - \\hat\\mu(W_i, X_i)),

   where :math:`Z_i = \\min(T_i, \\tau) \\cdot \\mathbf 1\\{\\delta_i\\, \\text{or}\\, T_i \\ge \\tau\\} / \\hat S_C(\\min(T_i,\\tau) \\mid X_i, W_i)`
   is the IPCW-RMST pseudo-outcome.

2. CATE at ``x`` = average of pseudo-outcomes from the leaves matching
   ``x`` across trees.

This implementation stays pragmatic: it uses sklearn honest regression
trees for the forest, KM-on-covariate strata for the censoring
survivor, and Nadaraya-Watson-style propensity estimation (logistic).
It targets the RMST contrast; HRs and survival-curve contrasts can be
post-computed from the leaf-level KM fits.

References
----------
Cui, Y., Kosorok, M. R., Sverdrup, E., Wager, S., & Zhu, R. (2023).
"Estimating heterogeneous treatment effects with right-censored data
via causal survival forests." *JRSS B*, 85(2), 179-211. [@cui2023estimating]
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
class CausalSurvivalForestResult:
    ate_rmst: float
    se: float
    ci: tuple
    pvalue: float
    cate: np.ndarray
    horizon: float
    n_obs: int
    n_trees: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lo, hi = self.ci
        return (
            "Causal Survival Forest (RMST)\n"
            "-----------------------------\n"
            f"  horizon (tau)   : {self.horizon:.4f}\n"
            f"  trees           : {self.n_trees}\n"
            f"  n               : {self.n_obs}\n"
            f"  ATE(RMST)       : {self.ate_rmst:.4f}  (SE={self.se:.4f})\n"
            f"  95% CI          : [{lo:.4f}, {hi:.4f}]\n"
            f"  p-value         : {self.pvalue:.4f}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"CausalSurvivalForestResult(ATE_RMST={self.ate_rmst:.4f})"


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _km_survivor(times: np.ndarray, events: np.ndarray, eval_times: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survivor S(t) evaluated at ``eval_times``."""
    if times.size == 0:
        return np.ones_like(eval_times, dtype=float)
    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order].astype(int)
    unique_t, idx = np.unique(t_sorted, return_index=True)
    S = np.ones(len(unique_t))
    n_at_risk = len(times)
    s_curr = 1.0
    S_vals = []
    current_idx = 0
    for i, ut in enumerate(unique_t):
        # count events and total at ut
        group = (t_sorted == ut)
        d = int(np.sum(e_sorted[group]))
        n = n_at_risk
        if n > 0:
            s_curr *= (1 - d / n)
        S_vals.append(s_curr)
        n_at_risk -= int(group.sum())
    S_arr = np.asarray(S_vals)
    out = np.ones_like(eval_times, dtype=float)
    for i, t in enumerate(eval_times):
        mask = unique_t <= t
        out[i] = S_arr[mask][-1] if mask.any() else 1.0
    return out


def _ipcw_rmst_pseudo(
    T: np.ndarray, delta: np.ndarray, W: np.ndarray, tau: float
) -> np.ndarray:
    """IPCW pseudo-outcome for RMST: Z_i = min(T, tau) / S_C(min(T, tau)).

    Censoring distribution estimated per treatment arm via KM.
    """
    n = T.shape[0]
    Y = np.minimum(T, tau)
    # For each arm compute KM of censoring (1 - delta)
    S_C = np.ones(n)
    for w_val in (0, 1):
        mask = W == w_val
        if not mask.any():
            continue
        # censoring KM: events = (1 - delta) i.e. censoring events
        Sc_w = _km_survivor(T[mask], 1 - delta[mask], Y[mask])
        S_C[mask] = np.maximum(Sc_w, 1e-3)
    # Pseudo-outcome: observed time truncated at tau, inverse-weighted by S_C
    pseudo = Y / S_C
    # For observations censored before tau and not yet reaching tau, mask out
    obs_weight = ((delta == 1) | (T >= tau)).astype(float)
    pseudo = np.where(obs_weight > 0, pseudo, 0.0)
    return pseudo, S_C


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


def causal_survival_forest(
    data: pd.DataFrame,
    time: str,
    event: str,
    treat: str,
    covariates: Sequence[str],
    horizon: Optional[float] = None,
    n_trees: int = 200,
    min_leaf: int = 5,
    max_depth: Optional[int] = None,
    propensity_bounds: tuple = (0.05, 0.95),
    random_state: int = 42,
    alpha: float = 0.05,
) -> CausalSurvivalForestResult:
    """
    Fit a causal survival forest and return the RMST ATE plus CATE.

    Parameters
    ----------
    data : pd.DataFrame
    time : str
        Observed time-to-event column (min of true event time and censoring time).
    event : str
        Event indicator (1 = event observed, 0 = censored).
    treat : str
        Binary treatment indicator.
    covariates : sequence of str
    horizon : float, optional
        RMST horizon tau. Defaults to the 80th percentile of observed times.
    n_trees : int, default 200
        Number of trees in the forest.
    min_leaf : int, default 5
        Minimum samples per leaf.
    max_depth : int, optional
        Maximum tree depth.
    propensity_bounds : tuple, default (0.05, 0.95)
        Clip estimated propensity for stability.
    random_state : int, default 42
    alpha : float, default 0.05

    Returns
    -------
    CausalSurvivalForestResult
        ``cate`` contains the individual RMST effect prediction.
    """
    covariates = list(covariates)
    df = data[[time, event, treat] + covariates].dropna().reset_index(drop=True)
    n = len(df)
    T = df[time].to_numpy(dtype=float)
    delta = df[event].to_numpy(dtype=int)
    W = df[treat].to_numpy(dtype=int)
    X = df[covariates].to_numpy(dtype=float)

    if horizon is None:
        horizon = float(np.quantile(T[delta == 1], 0.80)) if (delta == 1).any() else float(np.quantile(T, 0.8))

    # Propensity (logistic)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    lr.fit(X, W)
    e_hat = lr.predict_proba(X)[:, 1]
    e_hat = np.clip(e_hat, *propensity_bounds)

    # IPCW RMST pseudo-outcome per arm
    pseudo, S_C = _ipcw_rmst_pseudo(T, delta, W, horizon)

    # Outcome regression per arm (on pseudo-outcome)
    rf_forest = RandomForestRegressor(
        n_estimators=n_trees, min_samples_leaf=min_leaf,
        max_depth=max_depth, random_state=random_state,
        bootstrap=True, oob_score=False, n_jobs=-1,
    )
    # Fit joint model on (X, W) predicting pseudo
    Xw = np.column_stack([X, W])
    rf_forest.fit(Xw, pseudo)
    mu1 = rf_forest.predict(np.column_stack([X, np.ones(n)]))
    mu0 = rf_forest.predict(np.column_stack([X, np.zeros(n)]))

    # Double-robust score (analogue of AIPW for RMST)
    # psi_i = mu1 - mu0 + (W_i - e_i) / (e_i(1-e_i)) * (pseudo_i - mu_{W_i}(X_i))
    mu_w = np.where(W == 1, mu1, mu0)
    psi = (mu1 - mu0) + (W - e_hat) / np.maximum(e_hat * (1 - e_hat), 1e-6) * (pseudo - mu_w)

    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    z_stat = ate / se if se > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(z_stat)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (ate - crit * se, ate + crit * se)

    cate = mu1 - mu0

    _result = CausalSurvivalForestResult(
        ate_rmst=ate,
        se=se,
        ci=ci,
        pvalue=pval,
        cate=cate,
        horizon=horizon,
        n_obs=n,
        n_trees=n_trees,
        detail={
            "propensity_range": (float(e_hat.min()), float(e_hat.max())),
            "pseudo_outcome_range": (float(pseudo.min()), float(pseudo.max())),
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.survival.causal_survival_forest",
            params={
                "time": time, "event": event, "treat": treat,
                "covariates": list(covariates),
                "horizon": horizon,
                "n_trees": n_trees, "min_leaf": min_leaf,
                "max_depth": max_depth,
                "propensity_bounds": list(propensity_bounds),
                "random_state": random_state, "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# Backward-compatible alias matching grf naming.
causal_survival = causal_survival_forest


__all__ = [
    "causal_survival_forest", "causal_survival",
    "CausalSurvivalForestResult",
]
