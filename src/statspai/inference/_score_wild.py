"""Score wild cluster bootstrap for GLM coefficients (Kline and Santos 2012).

Clean-room implementation (2026-10-20) from Kline and Santos (2012) as
presented in Roodman, MacKinnon, Nielsen and Webb (2019), Section 6.2 and
Appendix A.3 ("The score bootstrap": the bootstrap numerator perturbs the
cluster score contributions mapped through the inverse Hessian; the
bootstrapped scores are recentred by each cluster's share of the
observations before they enter the cluster-robust variance). Stata
``boottest`` (GPL-3) was used only as a black box (its outputs on our
data). Specification: ``docs/dev/cleanroom/score_wild.md``.

For a null ``beta_j = 0``:

1. fit the GLM with ``beta_j`` restricted to zero; at that fit the score of
   observation ``i`` is ``s_i = x_i (y_i - mu_i) mu'_i / V(mu_i)`` and the
   (expected) Hessian is ``X' diag(mu'^2 / V) X``, ``X`` the full design;
2. ``q_g = (H^{-1} sum_{i in g} s_i)_j``, the cluster's one-step
   contribution to the coefficient;
3. for multipliers ``w`` (one per cluster), the numerator is
   ``N(w) = sum_g w_g q_g`` and the recentred contributions
   ``J_g(w) = w_g q_g - c_g N(w)``, ``c_g = n_g / n``; the statistic is
   ``t(w) = N(w) / sqrt(sum_g J_g(w)^2)``;
4. the observed statistic is ``t(1)`` (all multipliers one); the p-value
   is the share of draws with ``|t(w)| > |t(1)|``.

References
----------
[@roodman2019fast] (Section 6.2 and Appendix A.3, which present the score
bootstrap of Kline and Santos 2012).
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..exceptions import MethodIncompatibility

__all__ = ["glm_score_wild_boot", "score_wild_from_q"]

#: Webb (2014) six-point distribution: +-sqrt(1/2), +-1, +-sqrt(3/2).
_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])

#: Relative margin below which a bootstrap statistic counts as a tie with
#: the observed one (a draw proportional to the all-ones vector gives the
#: observed statistic exactly in exact arithmetic).
_TIE = 1e-10


def _studentized(draws: np.ndarray, q: np.ndarray, share: np.ndarray) -> np.ndarray:
    """``t(w)`` for each row ``w`` of ``draws``."""
    numer = draws @ q
    recentred = draws * q[None, :] - numer[:, None] * share[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        _ret: np.ndarray = numer / np.sqrt(np.sum(recentred**2, axis=1))
        return _ret


def _multipliers(
    weight_type: str, n_clusters: int, n_boot: int, seed: Any
) -> Tuple[np.ndarray, bool]:
    """Bootstrap multipliers, and whether the Rademacher grid is enumerated.

    Rademacher draws enumerate all ``2**G`` sign vectors when that is no
    more than ``n_boot``; otherwise ``n_boot`` rows are drawn with
    ``numpy.random.default_rng(seed).choice`` (one call, shape
    ``(n_boot, G)``). Webb draws are never enumerated.
    """
    if weight_type == "rademacher":
        if 2**n_clusters <= n_boot:
            grid = itertools.product((1.0, -1.0), repeat=n_clusters)
            return np.array(list(grid), dtype=float), True
        rng = np.random.default_rng(seed)
        return rng.choice(np.array([-1.0, 1.0]), size=(n_boot, n_clusters)), False
    if weight_type == "webb":
        rng = np.random.default_rng(seed)
        return rng.choice(_WEBB, size=(n_boot, n_clusters)), False
    raise MethodIncompatibility(
        f"Unknown wild weight_type {weight_type!r}; use 'rademacher' or 'webb'.",
        recovery_hint="Pass weight_type='rademacher' or 'webb'.",
        diagnostics={"weight_type": weight_type},
    )


def score_wild_from_q(
    q: np.ndarray,
    c_share: np.ndarray,
    *,
    n_boot: int = 9999,
    weight_type: str = "rademacher",
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Score wild bootstrap test from per-cluster contributions.

    Parameters
    ----------
    q : ndarray
        ``(G,)`` restricted one-step contributions of each cluster to the
        tested coefficient (step 2 of the module docstring).
    c_share : ndarray
        ``(G,)`` each cluster's share of the observations.
    n_boot : int, default 9999
        Bootstrap draws (Rademacher grids with ``2**G <= n_boot`` are
        enumerated instead).
    weight_type : {'rademacher', 'webb'}, default 'rademacher'
    seed : int, optional

    Returns
    -------
    dict
        ``p_boot``, ``t_obs``, ``n_reps`` (rows used) and ``enumerated``
        (1.0 / 0.0). With fewer than two clusters, or contributions that
        are all zero, there is nothing to resample: ``p_boot = 1``,
        ``t_obs = 0``, ``n_reps = 0``.
    """
    q = np.asarray(q, dtype=float).ravel()
    share = np.asarray(c_share, dtype=float).ravel()
    if weight_type not in ("rademacher", "webb"):
        _multipliers(weight_type, 0, 0, None)
    if q.size < 2 or not np.any(q != 0.0):
        return {"p_boot": 1.0, "t_obs": 0.0, "n_reps": 0, "enumerated": 0.0}
    t_obs = float(_studentized(np.ones((1, q.size)), q, share)[0])
    draws, enumerated = _multipliers(weight_type, q.size, int(n_boot), seed)
    t_star = _studentized(draws, q, share)
    margin = _TIE * max(1.0, abs(t_obs))
    exceed = np.abs(t_star) - abs(t_obs) > margin
    return {
        "p_boot": float(np.mean(exceed)),
        "t_obs": t_obs,
        "n_reps": float(draws.shape[0]),
        "enumerated": 1.0 if enumerated else 0.0,
    }


def glm_score_wild_boot(
    X: np.ndarray,
    y: np.ndarray,
    family: Any,
    cluster_codes: np.ndarray,
    test_idx: int,
    *,
    n_boot: int = 9999,
    weight_type: str = "rademacher",
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Score wild cluster bootstrap test of ``beta[test_idx] = 0`` in a GLM.

    Parameters
    ----------
    X : ndarray
        ``(n, k)`` design, including any intercept and fixed-effect dummies.
    y : ndarray
        Outcome.
    family : statsmodels family
        GLM family with its link (Poisson / binomial logit or probit).
    cluster_codes : ndarray
        Integer cluster code per row, ``0..G-1``.
    test_idx : int
        Column of ``X`` under test.
    n_boot, weight_type, seed
        As in :func:`score_wild_from_q`.

    Returns
    -------
    dict
        As :func:`score_wild_from_q`.

    References
    ----------
    [@roodman2019fast] (Section 6.2 and Appendix A.3, which present the score
    bootstrap of Kline and Santos 2012).
    """
    import statsmodels.api as sm

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    codes = np.asarray(cluster_codes).ravel()
    n, k = X.shape
    keep = [c for c in range(k) if c != test_idx]
    if keep:
        restricted = sm.GLM(y, X[:, keep], family=family).fit()
        eta = X[:, keep] @ np.asarray(restricted.params, dtype=float)
    else:
        eta = np.zeros(n)
    mu = family.link.inverse(eta)
    dmu = family.link.inverse_deriv(eta)
    var = family.variance(mu)
    score = X * ((y - mu) * dmu / var)[:, None]
    info = (X * (dmu**2 / var)[:, None]).T @ X
    G = int(codes.max()) + 1
    sums = np.zeros((G, k))
    np.add.at(sums, codes, score)
    row = np.linalg.solve(info, np.eye(k)[:, test_idx])
    q = sums @ row
    share = np.bincount(codes, minlength=G) / float(n)
    return score_wild_from_q(
        q, share, n_boot=n_boot, weight_type=weight_type, seed=seed
    )
