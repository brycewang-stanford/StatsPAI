"""Firpo (2007) efficient unconditional quantile treatment effects.

Estimands
---------
QTE (population)
    ``QTE(tau) = F^{-1}_{Y(1)}(tau) - F^{-1}_{Y(0)}(tau)``
    weights ``D / p(X)`` and ``(1 - D) / (1 - p(X))``.
QTT (on the treated)
    ``QTT(tau) = F^{-1}_{Y(1)|D=1}(tau) - F^{-1}_{Y(0)|D=1}(tau)``
    weights ``D`` and ``(1 - D) p(X) / (1 - p(X))``.

Each marginal quantile is the minimiser of a *weighted check function*

.. math::

    q_j(\\tau) = \\arg\\min_q \\sum_i w_{ji}\\,\\rho_\\tau(Y_i - q),
    \\qquad \\rho_\\tau(u) = u(\\tau - \\mathbf{1}\\{u < 0\\})

which for a scalar ``q`` is exactly the weighted empirical quantile, so we
solve it in closed form via :func:`statspai.qte._core.weighted_quantile`
rather than numerically.

Relation to R's ``qte::ci.qte`` / ``ci.qtet``
---------------------------------------------
R solves the same check-function problem, but with ``stats::optimize``
(golden-section search, default tolerance ``.Machine$double.eps^0.25``)
over ``[min(Y), max(Y)]``.  The check function is piecewise linear, so
between order statistics it has *plateaus* on which every point is a
minimiser; golden section returns an arbitrary interior point of the
plateau, and its answer additionally carries the optimiser's tolerance.
On ``qte::lalonde.exp`` this makes R's reported quantiles differ from the
exact minimiser by up to ~2% at some tau.

The consequence for parity testing: **bit-agreement with R is not a
well-posed target here**, because R's own number is not a well-defined
functional of the data on a plateau.  The parity suite therefore checks the
statistically meaningful thing -- that our solution attains a weighted
check-function value no worse than R's at every tau -- rather than chasing
a golden-section artifact.  See
``tests/reference_parity/test_firpo_qte_parity.py``.

References
----------
firpo2007efficient, koenker2005quantile
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ._core import kernel_density_at, weighted_quantile

__all__ = [
    "firpo_weights",
    "firpo_quantiles",
    "firpo_influence_se",
    "weighted_checkfun",
]


def weighted_checkfun(
    q: float,
    y: np.ndarray,
    tau: float,
    weights: np.ndarray,
) -> float:
    """``sum_i w_i rho_tau(y_i - q)``, the objective both we and R minimise."""
    u = np.asarray(y, dtype=float) - float(q)
    return float(np.sum(np.asarray(weights, dtype=float) * u * (tau - (u < 0))))


def firpo_weights(
    D: np.ndarray,
    pscore: np.ndarray,
    estimand: str = "qte",
) -> Tuple[np.ndarray, np.ndarray]:
    """Firpo (2007) reweighting weights for the two potential-outcome arms.

    Parameters
    ----------
    D : ndarray
        Binary treatment.
    pscore : ndarray
        ``P(D = 1 | X)`` per observation.
    estimand : {'qte', 'qtt'}

    Returns
    -------
    (w1, w0) : tuple of ndarray
        Weights putting mass on the treated arm and the control arm
        respectively. Not normalised; callers use the Hajek (self-normalised)
        form, which is what the weighted-quantile solver does implicitly.
    """
    D = np.asarray(D, dtype=float)
    p = np.asarray(pscore, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError(
            "Firpo weights require 0 < p(X) < 1 for every observation; "
            f"{int(np.sum((p <= 0) | (p >= 1)))} observation(s) violate this. "
            "Trim the sample or re-specify the propensity model."
        )
    estimand = estimand.lower()
    if estimand == "qte":
        return D / p, (1.0 - D) / (1.0 - p)
    if estimand == "qtt":
        return D, (1.0 - D) * p / (1.0 - p)
    raise ValueError(f"estimand must be 'qte' or 'qtt', got {estimand!r}")


def firpo_quantiles(
    Y: np.ndarray,
    D: np.ndarray,
    pscore: np.ndarray,
    taus: np.ndarray,
    estimand: str = "qte",
) -> Tuple[np.ndarray, np.ndarray]:
    """Reweighted marginal quantiles ``(q1, q0)`` at each level in ``taus``."""
    w1, w0 = firpo_weights(D, pscore, estimand)
    Y = np.asarray(Y, dtype=float)
    q1 = weighted_quantile(Y, taus, w1)
    q0 = weighted_quantile(Y, taus, w0)
    return q1, q0


def firpo_influence_se(
    Y: np.ndarray,
    D: np.ndarray,
    pscore: np.ndarray,
    taus: np.ndarray,
    q1: np.ndarray,
    q0: np.ndarray,
    estimand: str = "qte",
) -> np.ndarray:
    """Influence-function standard errors for the Firpo QTE/QTT.

    For a weighted quantile ``q_j(tau)`` solving the weighted check-function
    problem, the influence function is

        psi_j(tau) = -w_j (1{Y <= q_j(tau)} - tau) / (E[w_j] f_j(q_j(tau)))

    where ``f_j`` is the density of the reweighted arm-``j`` distribution, and
    ``QTE(tau) = q_1(tau) - q_0(tau)`` gives ``psi_1 - psi_0``.
    ``SE = sd(psi_1 - psi_0) / sqrt(n)``.

    .. note::

        This treats ``p(X)`` as **known**.  When ``p(X)`` is estimated, the
        efficient estimator's asymptotic variance is *smaller* than the
        known-propensity variance (Firpo 2007, Theorem 3), so ignoring the
        estimation correction makes these SEs **conservative** -- confidence
        intervals are too wide, not too narrow.  Callers that want the
        efficiency gain reflected should use the bootstrap, which is why the
        public estimator defaults to ``se='bootstrap'`` when covariates are
        supplied.
    """
    Y = np.asarray(Y, dtype=float)
    n = len(Y)
    w1, w0 = firpo_weights(D, pscore, estimand)
    ew1, ew0 = float(np.mean(w1)), float(np.mean(w0))
    if ew1 <= 0 or ew0 <= 0:
        return np.full(len(taus), np.nan)

    f1 = kernel_density_at(Y, q1, weights=w1)
    f0 = kernel_density_at(Y, q0, weights=w0)

    se = np.empty(len(taus))
    for j, tau in enumerate(taus):
        if not (np.isfinite(f1[j]) and np.isfinite(f0[j])):
            se[j] = np.nan
            continue
        psi1 = -w1 * ((Y <= q1[j]).astype(float) - tau) / (ew1 * f1[j])
        psi0 = -w0 * ((Y <= q0[j]).astype(float) - tau) / (ew0 * f0[j])
        se[j] = float(np.std(psi1 - psi0, ddof=1) / np.sqrt(n))
    return se


def logit_pscore(
    X: Optional[np.ndarray],
    D: np.ndarray,
    trim: float = 0.001,
) -> np.ndarray:
    """``P(D=1|X)`` by logistic regression; constant share when ``X`` is None.

    With no covariates the propensity score is the constant sample share, and
    the Firpo weights collapse to plain within-group quantiles -- which is
    exactly what ``qte::ci.qte`` does when ``xformla`` is ``NULL``.
    """
    D = np.asarray(D, dtype=float)
    if X is None:
        return np.full(len(D), float(np.mean(D)))
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)
    clf.fit(X, D.astype(int))
    return np.clip(np.asarray(clf.predict_proba(X)[:, 1]), trim, 1.0 - trim)
