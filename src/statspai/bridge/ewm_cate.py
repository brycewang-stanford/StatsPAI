"""
Bridge: EWM ≡ CATE → policy (Ferman et al. 2025, arXiv 2510.26723).

Empirical Welfare Maximisation directly maximises a policy's utility
in a reparameterised CATE space; CATE-then-threshold is the same
optimisation under a specific reparameterisation. Both paths target
the same optimal policy value V*; reporting both gives a check on
threshold-vs-EWM agreement.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .core import BridgeResult, _agreement_test, _dr_combine, _register


@_register("ewm_cate")
def ewm_cate_bridge(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    policy_class: str = "linear",
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> BridgeResult:
    """
    Compare EWM (path A: direct policy search) against CATE-then-
    threshold (path B: predict effect, treat if positive) on the
    optimal policy value.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome (higher better).
    treat : str
        Binary treatment indicator (0/1).
    covariates : list of str
        Pre-treatment features used by both paths.
    policy_class : {'linear'}, default 'linear'
        Policy class for EWM. Only linear thresholds shipped.
    alpha : float
    n_boot : int, default 200
        Bootstrap reps for SE on each path.
    seed : int
    """
    df = data[[y, treat] + list(covariates)].dropna().reset_index(drop=True)
    if df[treat].nunique() != 2:
        raise ValueError(
            f"EWM bridge requires binary treat; got {df[treat].nunique()} values."
        )
    n = len(df)
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(int)
    X = df[covariates].to_numpy(float)
    rng = np.random.default_rng(seed)

    # ---------- Path B: CATE → threshold ---------- #
    # Use simple T-learner with linear regression for both arms (fast,
    # transparent; user can swap to sp.metalearner upstream if desired).
    def _t_learner_cate(Yi, Di, Xi):
        from sklearn.linear_model import LinearRegression
        m1 = LinearRegression().fit(Xi[Di == 1], Yi[Di == 1])
        m0 = LinearRegression().fit(Xi[Di == 0], Yi[Di == 0])
        return m1.predict(Xi) - m0.predict(Xi)

    cate = _t_learner_cate(Y, D, X)
    pi_cate = (cate > 0).astype(int)
    # Estimate value of pi_cate via doubly-robust IPW score
    # (Athey-Wager 2021 score).
    p_treat = float(np.mean(D))
    weights = np.where(D == 1, 1.0 / max(p_treat, 1e-3),
                       1.0 / max(1 - p_treat, 1e-3))
    score_cate = (
        cate
        + (Y - (D * cate)) * (2 * pi_cate - 1) * weights * (D == pi_cate)
    )
    value_cate = float(np.mean(score_cate))

    # ---------- Path A: EWM (linear threshold over X) ---------- #
    # Brute-force linear policy: π(x) = 1{x'β > 0}, β on the unit sphere.
    # Pick a small grid of random β's; keep the one that maximises
    # in-sample DR-policy value.
    p = X.shape[1]
    n_dirs = 200 if p <= 10 else 100
    dirs = rng.standard_normal((n_dirs, p))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    best_value = -np.inf
    best_pi = pi_cate
    for beta in dirs:
        pi = (X @ beta > 0).astype(int)
        score = (
            cate
            + (Y - (D * cate)) * (2 * pi - 1) * weights * (D == pi)
        )
        v = float(np.mean(score))
        if v > best_value:
            best_value = v
            best_pi = pi
    value_ewm = best_value

    # ---------- Bootstrap SEs ---------- #
    boot_cate = np.full(n_boot, np.nan)
    boot_ewm = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            cb = _t_learner_cate(Y[idx], D[idx], X[idx])
            pi_b = (cb > 0).astype(int)
            wb = np.where(D[idx] == 1, 1.0 / max(p_treat, 1e-3),
                          1.0 / max(1 - p_treat, 1e-3))
            sc_b = (
                cb + (Y[idx] - (D[idx] * cb)) * (2 * pi_b - 1)
                * wb * (D[idx] == pi_b)
            )
            boot_cate[b] = np.mean(sc_b)
            # Use same best β found above for EWM path bootstrap
            pi_e = best_pi[idx]
            sc_e = (
                cb + (Y[idx] - (D[idx] * cb)) * (2 * pi_e - 1)
                * wb * (D[idx] == pi_e)
            )
            boot_ewm[b] = np.mean(sc_e)
        except Exception:
            continue
    se_cate = float(np.nanstd(boot_cate, ddof=1)) or 1e-6
    se_ewm = float(np.nanstd(boot_ewm, ddof=1)) or 1e-6

    diff, diff_se, diff_p = _agreement_test(
        value_ewm, se_ewm, value_cate, se_cate
    )
    est_dr, se_dr = _dr_combine(
        value_ewm, se_ewm, value_cate, se_cate, diff_p
    )

    return BridgeResult(
        kind="ewm_cate",
        path_a_name="EWM (direct policy search)",
        path_b_name="CATE → threshold",
        estimate_a=float(value_ewm),
        estimate_b=float(value_cate),
        se_a=se_ewm,
        se_b=se_cate,
        diff=diff,
        diff_se=diff_se,
        diff_p=diff_p,
        estimate_dr=est_dr,
        se_dr=se_dr,
        n_obs=n,
        detail={"policy_class": policy_class, "n_directions": n_dirs},
        reference="Ferman et al. (2025), arXiv 2510.26723",
    )
