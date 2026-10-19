"""Doubly robust dose-response curve (Kennedy, Ma, McHugh and Small 2017).

``theta(t) = E[Y(t)]`` for a continuous treatment, estimated by regressing
the pseudo-outcome

    xi = (Y - mu(X, T)) / pi(T | X) * m(T) + mubar(T),
    m(t) = mean_j pi(t | X_j),   mubar(t) = mean_j mu(X_j, t),

on ``T`` with a local linear kernel smoother [@kennedy2017parametric].
``pi`` is the conditional density of the treatment (normal, with mean and
variance regressions), ``mu`` the outcome regression; the curve is
consistent if either is. Nuisances are cross-fitted. The bandwidth
minimises the leave-one-out risk of the smoother; pointwise standard
errors are the heteroskedasticity-robust local-linear sandwich on the
pseudo-outcomes, which is first-order valid when the nuisance estimators
converge fast enough (the paper's Theorem 2). They do not include the
smoothing bias: the bands are for the smoothed curve.

R ``npcausal::ctseff`` (GPL, used as a black box) implements the same
estimator with SuperLearner nuisances, a kernel density for the
standardized treatment residual and a binned smoother; the curves agree
in shape on shared fixtures but not to parity (see
``tests/test_dose_response_dr.py``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability


def _clone(model: Any) -> Any:
    from sklearn.base import clone

    return clone(model)


def _pseudo_outcomes(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    treatment_model: Any,
    outcome_model: Any,
    n_folds: int,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold

    n = len(Y)
    tm = treatment_model if treatment_model is not None else LinearRegression()
    om = outcome_model
    if om is None:
        from sklearn.ensemble import GradientBoostingRegressor

        om = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=random_state
        )
    mean_t = np.empty(n)
    var_t = np.empty(n)
    mu_obs = np.empty(n)
    # cross-fitted marginal pieces: for unit i, m(T_i) and mubar(T_i) average
    # over the units of the other folds, with that fold's nuisance fits
    exact = n <= 1000
    tq = T.copy() if exact else np.quantile(T, np.linspace(0, 1, 400))
    m_q = np.zeros(len(tq))
    mubar_q = np.zeros(len(tq))
    counts = 0
    folds = (
        [(np.arange(n), np.arange(n))]
        if n_folds <= 1
        else list(
            KFold(n_splits=n_folds, shuffle=True, random_state=random_state).split(X)
        )
    )
    for tr, te in folds:
        fm = _clone(tm).fit(X[tr], T[tr])
        r2 = (T[tr] - fm.predict(X[tr])) ** 2
        fv = LinearRegression().fit(X[tr], r2)
        fo = _clone(om).fit(np.column_stack([X[tr], T[tr]]), Y[tr])
        mt = fm.predict(X[te])
        vt = fv.predict(X[te])
        floor = 1e-3 * float(np.mean(r2))
        vt = np.maximum(vt, floor)
        mean_t[te] = mt
        var_t[te] = vt
        mu_obs[te] = fo.predict(np.column_stack([X[te], T[te]]))
        # averages over the test fold's covariates of pi(t | X_j) and
        # mu(X_j, t), at every unit's T (n <= 1000) or on 400 quantiles of
        # T, interpolated (larger n: the exact sum is O(n^2) predictions)
        sd = np.sqrt(vt)
        for start in range(0, len(tq), 512):
            blk = slice(start, min(start + 512, len(tq)))
            dens = (
                stats.norm.pdf((tq[blk, None] - mt[None, :]) / sd[None, :])
                / sd[None, :]
            )
            m_q[blk] += dens.sum(axis=1)
        for start in range(0, len(tq), 256):
            blk = np.arange(start, min(start + 256, len(tq)))
            Xrep = np.repeat(X[te], len(blk), axis=0)
            Trep = np.tile(tq[blk], len(te))
            pred = fo.predict(np.column_stack([Xrep, Trep])).reshape(len(te), len(blk))
            mubar_q[blk] += pred.sum(axis=0)
        counts += len(te)
    m_q /= counts
    mubar_q /= counts
    if exact:
        m_hat, mubar = m_q, mubar_q
    else:
        m_hat = np.interp(T, tq, m_q)
        mubar = np.interp(T, tq, mubar_q)
    pi_obs = stats.norm.pdf((T - mean_t) / np.sqrt(var_t)) / np.sqrt(var_t)
    small = pi_obs < 1e-8 * np.max(pi_obs)
    if small.any():
        raise NumericalInstability(
            "dose_response(method='dr'): the treatment density is numerically "
            f"zero at the observed dose for {int(small.sum())} units "
            "(no overlap).",
            recovery_hint="Restrict the sample to doses that are plausible "
            "for every covariate value, or use a better treatment model.",
        )
    xi = (Y - mu_obs) / pi_obs * m_hat + mubar
    ratio = m_hat / pi_obs
    info = {
        "max_density_ratio": float(ratio.max()),
        "n_folds": int(max(n_folds, 1)),
        "treatment_model": type(tm).__name__,
        "outcome_model": type(om).__name__,
        "marginal_averages": "exact" if exact else "400-point grid, interpolated",
    }
    return xi, info


def _local_linear_weights(
    T: np.ndarray, t0: float, h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear-smoother weights ``w`` with ``theta(t0) = w @ xi`` and the
    slope weights, for a Gaussian-kernel local linear fit."""
    u = (T - t0) / h
    k = stats.norm.pdf(u)
    D = np.column_stack([np.ones_like(T), T - t0])
    A = (D * k[:, None]).T @ D
    if np.linalg.cond(A) > 1e12:
        raise NumericalInstability(
            f"dose_response(method='dr'): too few observations near dose {t0:.4g} "
            f"for bandwidth {h:.4g}.",
            recovery_hint="Increase the bandwidth or narrow dose_range.",
        )
    W = np.linalg.solve(A, (D * k[:, None]).T)  # 2 x n
    return W[0], W[1]


def loo_risk(T: np.ndarray, xi: np.ndarray, h: float) -> float:
    """Leave-one-out squared error of the local linear smoother; ``inf``
    when the bandwidth leaves some observed dose without neighbours."""
    n = len(T)
    err = np.empty(n)
    for i in range(n):
        try:
            w0, _ = _local_linear_weights(T, T[i], h)
        except NumericalInstability:
            return float("inf")  # the bandwidth cannot fit every point
        fit = float(w0 @ xi)
        err[i] = (xi[i] - fit) / (1.0 - w0[i])
    return float(np.mean(err**2))


def dr_curve(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    dose_grid: np.ndarray,
    *,
    treatment_model: Any,
    outcome_model: Any,
    bandwidth: Optional[float],
    n_folds: int,
    random_state: int,
    alpha: float,
) -> Dict[str, Any]:
    n = len(Y)
    if n < 30:
        raise DataInsufficient(
            f"dose_response(method='dr'): {n} complete rows are too few.",
            recovery_hint="Use more data.",
        )
    xi, info = _pseudo_outcomes(
        Y, T, X, treatment_model, outcome_model, n_folds, random_state
    )
    sd_t = float(np.std(T, ddof=1))
    if bandwidth is None:
        grid = sd_t * np.geomspace(0.05, 1.5, 25)
        risks = np.array([loo_risk(T, xi, h) for h in grid])
        h = float(grid[int(np.argmin(risks))])
        bw_risk = {"bandwidth": grid.tolist(), "risk": risks.tolist()}
        if h in (grid[0], grid[-1]):
            info["bandwidth_at_grid_edge"] = True
    else:
        h = float(bandwidth)
        if not h > 0:
            raise MethodIncompatibility("dose_response: bandwidth must be positive.")
        bw_risk = None
    z = stats.norm.ppf(1 - alpha / 2)
    est = np.empty(len(dose_grid))
    se = np.empty(len(dose_grid))
    infl = np.empty((len(dose_grid), n))
    for g, t0 in enumerate(dose_grid):
        w0, w1 = _local_linear_weights(T, float(t0), h)
        b0, b1 = float(w0 @ xi), float(w1 @ xi)
        resid = xi - b0 - b1 * (T - t0)
        infl[g] = w0 * resid
        est[g] = b0
        se[g] = float(np.sqrt(np.sum(infl[g] ** 2)))
    return {
        "est": est,
        "se": se,
        "ci_lower": est - z * se,
        "ci_upper": est + z * se,
        "bandwidth": h,
        "bw_risk": bw_risk,
        "influence": infl,
        "pseudo_outcomes": xi,
        "info": info,
    }
