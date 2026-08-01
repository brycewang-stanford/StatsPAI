"""Doubly-robust triple-differences primitives (Ortiz-Villavicencio & Sant'Anna).

The DDD parameter is built from three two-by-two comparisons against the
treated-and-eligible cell. Writing the four cells as

    4 = treated,   eligible          (the group the effect is about)
    3 = treated,   ineligible
    2 = untreated, eligible
    1 = untreated, ineligible

the estimand is

    ATT_DDD = DiD(4 vs 3) + DiD(4 vs 2) - DiD(4 vs 1)

and each ``DiD(4 vs a)`` is an ordinary two-period doubly-robust DiD run on
the units in cells ``{a, 4}``: a propensity score for being in cell 4 rather
than cell ``a``, an outcome regression for the change in outcome fitted on
cell ``a``, and the Sant'Anna-Zhao doubly-robust combination of the two.

That decomposition is what makes covariates tractable here. Each piece is a
standard DR-DiD, so each comes with a known influence function, and the
influence function of the DDD is the same signed combination -- reweighted,
because the three comparisons rest on different subsamples:

    psi = (n/n_3) psi_3 + (n/n_2) psi_2 - (n/n_1) psi_1

with ``n_a`` the number of units in cells ``{a, 4}``. Standard errors follow
analytically, without a bootstrap.

Ported formula-by-formula from ``triplediff`` 0.2.4 (``compute_did``,
``compute_pscore``, ``compute_outcome_regression``, ``att_dr``) so the three
``est_method`` variants agree with the reference rather than merely
resembling it.

References
----------
ortiz2025better
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..exceptions import DataInsufficient, NumericalInstability

__all__ = ["DDDCellFit", "ddd_dr_cell"]

_TRIM_LEVEL = 0.995
_PS_CAP = 1.0 - 1e-6

# Cell codes, following the reference implementation's `subgroup`.
CELL_TREATED_ELIGIBLE = 4
CELL_TREATED_INELIGIBLE = 3
CELL_UNTREATED_ELIGIBLE = 2
CELL_UNTREATED_INELIGIBLE = 1
_COMPARISONS = (
    CELL_TREATED_INELIGIBLE,
    CELL_UNTREATED_ELIGIBLE,
    CELL_UNTREATED_INELIGIBLE,
)
_COMPARISON_LABEL = {
    CELL_TREATED_INELIGIBLE: "treated-eligible vs treated-ineligible",
    CELL_UNTREATED_ELIGIBLE: "treated-eligible vs untreated-eligible",
    CELL_UNTREATED_INELIGIBLE: "treated-eligible vs untreated-ineligible",
}


@dataclass
class DDDCellFit:
    """One ``ATT_DDD(g, t)`` cell with its influence function."""

    att: float
    se: float
    influence: np.ndarray
    components: Dict[int, float]
    n_units: int
    est_method: str
    diagnostics: Dict[str, object]


def _logit_pscore(
    X: np.ndarray,
    d: np.ndarray,
    w: np.ndarray,
    *,
    label: str,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted logistic fit of ``d`` on ``X``; returns (ps, hessian, keep).

    ``hessian`` is ``n * inv(X' W X)``, the scaling the influence-function
    correction below expects, and ``keep`` is the reference's asymmetric
    trimming: control units with a propensity above ``_TRIM_LEVEL`` drop out,
    treated units keep everything short of 1.01.
    """
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -500, 500)
        p = 1.0 / (1.0 + np.exp(-eta))
        wt = w * p * (1.0 - p)
        grad = X.T @ (w * (d - p))
        hess = X.T @ (wt[:, None] * X)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError as exc:
            raise NumericalInstability(
                f"ddd(est_method='dr'): the propensity-score design matrix is "
                f"singular for the {label} comparison. Drop a collinear "
                "covariate.",
                diagnostics={"comparison": label},
            ) from exc
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break

    eta = np.clip(X @ beta, -500, 500)
    ps = 1.0 / (1.0 + np.exp(-eta))
    ps = np.minimum(ps, _PS_CAP)

    wt = ps * (1.0 - ps) * w
    xwx = X.T @ (wt[:, None] * X)
    if np.linalg.cond(xwx) > 1.0 / np.finfo(float).eps:
        raise NumericalInstability(
            f"ddd(est_method='dr'): the propensity-score information matrix is "
            f"singular for the {label} comparison, so its estimation effect on "
            "the standard error is not identified.",
            diagnostics={"comparison": label},
        )
    hessian = np.linalg.inv(xwx) * n

    keep = np.ones(n, dtype=float)
    control = d == 0
    keep[control] = (ps[control] < _TRIM_LEVEL).astype(float)
    return ps, hessian, keep


def _outcome_regression(
    dy: np.ndarray,
    X: np.ndarray,
    control: np.ndarray,
    w: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """WLS of the outcome change on ``X`` fitted on the control arm."""
    Xc, yc, wc = X[control], dy[control], w[control]
    if Xc.shape[0] < Xc.shape[1]:
        raise DataInsufficient(
            f"ddd: the {label} comparison has fewer control units than "
            "covariates, so the outcome regression is not estimable.",
            diagnostics={"comparison": label, "n_control": int(Xc.shape[0])},
        )
    sw = np.sqrt(wc)
    beta, *_ = np.linalg.lstsq(Xc * sw[:, None], yc * sw, rcond=None)
    return X @ beta


def _one_comparison(
    *,
    cell: np.ndarray,
    a: int,
    dy: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
    est_method: str,
) -> Tuple[float, np.ndarray]:
    """DR / IPW / REG DiD for cells ``{a, 4}``; returns (att, influence).

    The influence function is returned on the FULL unit vector, zero outside
    the pair -- that is what lets the three comparisons be combined by simple
    addition even though each rests on a different subsample.
    """
    label = _COMPARISON_LABEL[a]
    pair = (cell == a) | (cell == CELL_TREATED_ELIGIBLE)
    if not pair.any():
        raise DataInsufficient(
            f"ddd: no units in the {label} comparison.",
            diagnostics={"comparison": label},
        )
    Xp, dyp, wp = X[pair], dy[pair], w[pair]
    pa4 = (cell[pair] == CELL_TREATED_ELIGIBLE).astype(float)
    paa = (cell[pair] == a).astype(float)
    n_pair = int(pair.sum())
    if pa4.sum() == 0 or paa.sum() == 0:
        raise DataInsufficient(
            f"ddd: the {label} comparison is missing one of its two arms.",
            diagnostics={
                "comparison": label,
                "n_treated_eligible": int(pa4.sum()),
                "n_comparison": int(paa.sum()),
            },
        )

    if est_method == "reg":
        ps = np.ones(n_pair)
        hessian = None
        keep = np.ones(n_pair)
    else:
        ps, hessian, keep = _logit_pscore(Xp, pa4, wp, label=label)

    if est_method == "ipw":
        or_delta = np.zeros(n_pair)
    else:
        or_delta = _outcome_regression(dyp, Xp, paa > 0, wp, label=label)

    w_treat = keep * wp * pa4
    if est_method == "reg":
        w_control = keep * wp * paa
    else:
        w_control = keep * wp * ps * paa / (1.0 - ps)

    mean_wt = float(np.mean(w_treat))
    mean_wc = float(np.mean(w_control))
    if mean_wt <= 0 or mean_wc <= 0:
        raise DataInsufficient(
            f"ddd: the {label} comparison has no effective weight on one arm "
            "after propensity trimming.",
            diagnostics={
                "comparison": label,
                "mean_w_treat": mean_wt,
                "mean_w_control": mean_wc,
            },
        )

    resid = dyp - or_delta
    riesz_treat = w_treat * resid
    riesz_control = w_control * resid
    att_treat = float(np.mean(riesz_treat)) / mean_wt
    att_control = float(np.mean(riesz_control)) / mean_wc
    att = att_treat - att_control

    # Estimation effect of the propensity score.
    if est_method == "reg":
        inf_control_ps = np.zeros(n_pair)
    else:
        m2 = np.mean((w_control * (resid - att_control))[:, None] * Xp, axis=0)
        score_ps = (wp * (pa4 - ps))[:, None] * Xp
        inf_control_ps = (score_ps @ hessian) @ m2

    # Estimation effect of the outcome regression.
    if est_method == "ipw":
        inf_treat_or = np.zeros(n_pair)
        inf_control_or = np.zeros(n_pair)
    else:
        m1 = np.mean(w_treat[:, None] * Xp, axis=0)
        m3 = np.mean(w_control[:, None] * Xp, axis=0)
        or_x = (wp * paa)[:, None] * Xp
        or_ex = (wp * paa * resid)[:, None] * Xp
        xpx = or_x.T @ Xp / n_pair
        if np.linalg.cond(xpx) > 1.0 / np.finfo(float).eps:
            raise NumericalInstability(
                f"ddd: the outcome-regression design matrix is singular for "
                f"the {label} comparison. Consider removing a covariate.",
                diagnostics={"comparison": label},
            )
        asy_or = np.linalg.solve(xpx, or_ex.T).T
        inf_treat_or = -(asy_or @ m1)
        inf_control_or = -(asy_or @ m3)

    inf_treat = (riesz_treat - w_treat * att_treat + inf_treat_or) / mean_wt
    inf_control = (
        riesz_control - w_control * att_control + inf_control_ps + inf_control_or
    ) / mean_wc

    influence = np.zeros(len(cell), dtype=float)
    influence[pair] = inf_treat - inf_control
    return att, influence


def ddd_dr_cell(
    *,
    cell: np.ndarray,
    dy: np.ndarray,
    X: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    est_method: str = "dr",
) -> DDDCellFit:
    """One ``ATT_DDD`` cell from unit-level cell codes and outcome changes.

    Parameters
    ----------
    cell : ndarray of int
        Per-unit cell code: 4 treated-eligible, 3 treated-ineligible,
        2 untreated-eligible, 1 untreated-ineligible.
    dy : ndarray
        Per-unit change in the outcome between the base and the comparison
        period.
    X : ndarray, optional
        Per-unit covariate matrix INCLUDING the intercept column. Defaults to
        an intercept only, which reproduces the unconditional DDD.
    weights : ndarray, optional
        Per-unit sampling weights; defaults to 1.
    est_method : {"dr", "ipw", "reg"}
        Doubly robust, inverse-probability weighting, or outcome regression.

    Returns
    -------
    DDDCellFit
    """
    cell = np.asarray(cell, dtype=int)
    dy = np.asarray(dy, dtype=float)
    n = len(cell)
    if len(dy) != n:
        raise ValueError("cell and dy must have the same length")
    if est_method not in {"dr", "ipw", "reg"}:
        raise ValueError(f"est_method must be 'dr', 'ipw' or 'reg', got {est_method!r}")
    if X is None:
        X = np.ones((n, 1), dtype=float)
    else:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] != n:
            raise ValueError("X must be (n_units, n_covariates)")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    atts: Dict[int, float] = {}
    infs: Dict[int, np.ndarray] = {}
    for a in _COMPARISONS:
        atts[a], infs[a] = _one_comparison(
            cell=cell, a=a, dy=dy, X=X, w=w, est_method=est_method
        )

    att = (
        atts[CELL_TREATED_INELIGIBLE]
        + atts[CELL_UNTREATED_ELIGIBLE]
        - atts[CELL_UNTREATED_INELIGIBLE]
    )

    # Each comparison's influence function lives on its own subsample, so it
    # is rescaled by that subsample's share before the three are combined.
    influence = np.zeros(n, dtype=float)
    sizes: Dict[int, int] = {}
    for a, sign in (
        (CELL_TREATED_INELIGIBLE, 1.0),
        (CELL_UNTREATED_ELIGIBLE, 1.0),
        (CELL_UNTREATED_INELIGIBLE, -1.0),
    ):
        n_a = int(((cell == a) | (cell == CELL_TREATED_ELIGIBLE)).sum())
        sizes[a] = n_a
        influence += sign * (n / n_a) * infs[a]

    # The reference uses sd(psi)/sqrt(n) -- the (n-1) denominator and the
    # mean subtraction, not sqrt(mean(psi^2)/n). The influence function is
    # mean-zero in population so the two agree asymptotically, but on a
    # finite sample they differ by sqrt(n/(n-1)) and that shows up in the
    # fourth digit. Match the reference.
    se = float(np.std(influence, ddof=1) / np.sqrt(n))
    return DDDCellFit(
        att=float(att),
        se=se,
        influence=influence,
        components={int(a): float(v) for a, v in atts.items()},
        n_units=n,
        est_method=est_method,
        diagnostics={
            "comparison_sizes": {int(a): int(v) for a, v in sizes.items()},
            "n_treated_eligible": int((cell == CELL_TREATED_ELIGIBLE).sum()),
            "n_covariates": int(X.shape[1]),
        },
    )
