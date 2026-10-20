"""Two-period DiD estimators for repeated cross-sections (Sant'Anna & Zhao 2020).

Clean-room rewrite (2026-10-20); the reference implementation (R ``DRDID``,
GPL-3) was used only as a black box. Specification and evidence:
``docs/dev/cleanroom/did_drdid.md``.

Repeated cross-sections observe each row once, in the pre-period
(``post = 0``) or the post-period (``post = 1``), so the ATT is identified
from the four (group x period) cells under the stationarity assumption of
Sant'Anna and Zhao (2020, Assumption 1(b)). Every estimator returns the ATT,
its analytic standard error and the per-row influence function, so
``sp.aggte`` and the multiplier bootstrap can combine cells.

==========================  ===========================================
function                    estimator (Sant'Anna and Zhao 2020)
==========================  ===========================================
:func:`drdid_rc`            DR, eq. (3.4); ``locally_efficient=False``
                            gives eq. (3.3)
:func:`drdid_imp_rc`        improved DR, eqs. (3.9)-(3.10): inverse
                            probability tilting + odds-weighted least
                            squares, no first-step estimation effect
:func:`std_ipw_did_rc`      IPW with Hajek-normalised arm weights
:func:`ipw_did_rc`          IPW of Abadie type, eq. (2.3)
:func:`reg_did_rc`          outcome regression, eq. (2.2)
:func:`twfe_did_rc`         two-way fixed-effects regression, eq. (2.5)
==========================  ===========================================

Conventions (each pinned by black-box runs of the reference on our data):
weights are divided by their mean; the propensity score is a logit of the
group dummy on the covariates over the pooled sample, capped at
``1 - 1e-6``; comparison rows with a score at or above ``trim_level`` get
zero weight in the comparison arm; the four outcome regressions are fit
separately per cell; the standard error is ``sqrt(mean(IF**2) / n)``
(``sd(IF) / sqrt(n)`` for the TWFE regression).

References
----------
[@santanna2020doubly]
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np

from .._result_serialize import attach_result_protocol
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._dr_common import (
    HajekMean,
    design_matrix,
    ipt_fit,
    logit_fit,
    logit_influence,
    normalized_weights,
    se_from_influence,
    spanning_columns,
    wls_coef,
    wls_influence,
)

__all__ = [
    "RCSResult",
    "drdid_imp_rc",
    "drdid_rc",
    "ipw_did_rc",
    "reg_did_rc",
    "std_ipw_did_rc",
    "twfe_did_rc",
]

#: Comparison rows with a propensity score at or above this level are given
#: zero weight (the reference's documented ``trim.level`` default).
_TRIM_LEVEL = 0.995

#: Fitted propensity scores are capped here so the odds stay finite.
_PSCORE_CAP = 1.0 - 1e-6

_VARIANCE_CONVENTIONS = ("delta", "reference")


@attach_result_protocol
class RCSResult(NamedTuple):
    """ATT, analytic SE, and the per-observation influence function."""

    att: float
    se: float
    influence: np.ndarray


class _Sample(NamedTuple):
    y: np.ndarray
    post: np.ndarray
    d: np.ndarray
    x: np.ndarray  # covariates with a leading intercept column
    w: np.ndarray  # weights normalised to mean one


def _as_vector(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise MethodIncompatibility(
            f"`{name}` contains missing or non-finite values.",
            recovery_hint=f"Drop or impute the rows with non-finite `{name}`.",
            diagnostics={"argument": name},
        )
    return arr


def _require_binary(arr: np.ndarray, name: str) -> None:
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise MethodIncompatibility(
            f"`{name}` must be a 0/1 indicator.",
            recovery_hint=f"Recode `{name}` as 0/1.",
            diagnostics={"argument": name, "values": sorted(set(np.unique(arr)))[:5]},
        )


def _prepare(
    y: Any,
    post: Any,
    d: Any,
    covariates: Any,
    weights: Any,
) -> _Sample:
    """Validate the inputs and return them as arrays (weights mean one)."""
    y_arr = _as_vector(y, "y")
    post_arr = _as_vector(post, "post")
    d_arr = _as_vector(d, "d")
    n = y_arr.shape[0]
    if post_arr.shape[0] != n or d_arr.shape[0] != n:
        raise MethodIncompatibility(
            "`y`, `post` and `d` must have the same length.",
            recovery_hint="Pass one value per observation for each argument.",
            diagnostics={
                "len_y": n,
                "len_post": int(post_arr.shape[0]),
                "len_d": int(d_arr.shape[0]),
            },
        )
    _require_binary(post_arr, "post")
    _require_binary(d_arr, "d")

    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov[:, None]
        if cov.shape[0] != n:
            raise MethodIncompatibility(
                "`covariates` must have one row per observation.",
                recovery_hint="Align the covariate matrix with `y`.",
                diagnostics={"rows": int(cov.shape[0]), "expected": n},
            )
        if not np.all(np.isfinite(cov)):
            raise MethodIncompatibility(
                "`covariates` contains missing or non-finite values.",
                recovery_hint="Drop or impute the rows with non-finite covariates.",
                diagnostics={"argument": "covariates"},
            )
        covariates = cov
    x = design_matrix(covariates, n)

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        raw = _as_vector(weights, "weights")
        if raw.shape[0] != n:
            raise MethodIncompatibility(
                "`weights` must have one entry per observation.",
                recovery_hint="Align the weight vector with `y`.",
                diagnostics={"len_weights": int(raw.shape[0]), "expected": n},
            )
        if np.any(raw < 0):
            raise MethodIncompatibility(
                "`weights` must be non-negative.",
                recovery_hint="Pass non-negative observation weights.",
                diagnostics={"min_weight": float(raw.min())},
            )
        if raw.sum() <= 0:
            raise DataInsufficient(
                "`weights` sum to zero.",
                recovery_hint="Give at least one observation a positive weight.",
                diagnostics={"n": n},
            )
        w = normalized_weights(raw, n)

    cells = {
        (dd, tt): float(np.sum(w[(d_arr == dd) & (post_arr == tt)]))
        for dd in (0, 1)
        for tt in (0, 1)
    }
    empty = [k for k, mass in cells.items() if mass <= 0]
    if empty:
        raise DataInsufficient(
            "Repeated cross-section DiD needs all four treatment x period "
            "cells to be populated; empty (d, post) cells: "
            f"{sorted(empty)}.",
            recovery_hint=(
                "Make sure treated and comparison observations exist in both "
                "the pre- and the post-period."
            ),
            diagnostics={"empty_cells": [list(k) for k in sorted(empty)]},
        )
    return _Sample(y_arr, post_arr, d_arr, x, w)


def _check_convention(variance_convention: str) -> None:
    if variance_convention not in _VARIANCE_CONVENTIONS:
        raise MethodIncompatibility(
            f"variance_convention must be one of {_VARIANCE_CONVENTIONS}, "
            f"got {variance_convention!r}.",
            recovery_hint="Use 'delta' (the default) or 'reference'.",
            diagnostics={"variance_convention": variance_convention},
        )


def _comparison_odds(
    s: _Sample, trim_level: float, tilting: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propensity score, its odds, and the trimming indicator.

    ``tilting=False`` fits the logit by maximum likelihood; ``True`` uses
    inverse probability tilting. Comparison rows with a score at or above
    ``trim_level`` get ``keep = 0``; treated rows are always kept.
    """
    fit = ipt_fit if tilting else logit_fit
    _, p = fit(s.d, s.x, s.w)
    p = np.minimum(p, _PSCORE_CAP)
    keep = np.where(s.d == 1.0, 1.0, (p < trim_level).astype(float))
    return p, p / (1.0 - p), keep


def _cell(s: _Sample, dd: int, tt: int) -> np.ndarray:
    return (s.d == dd) & (s.post == tt)


class _CellFit(NamedTuple):
    """Least-squares fit of ``y`` on one (group, period) cell."""

    x: np.ndarray  # the design columns kept for this cell (all rows)
    coef: np.ndarray
    rows: np.ndarray
    w: np.ndarray

    @property
    def fitted(self) -> np.ndarray:
        return self.x @ self.coef

    def effect(self, y: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Estimation effect ``l_i' grad`` of this fit's coefficients."""
        return wls_influence(self.x, y, self.w, self.rows, self.coef) @ grad


def _require_identified(
    s: _Sample, cells: Tuple[Tuple[int, int], ...], label: str
) -> None:
    """Each fitted cell needs at least as many rows as regressors."""
    k = s.x.shape[1]
    short = [
        (dd, tt) for dd, tt in cells if int(np.sum(_cell(s, dd, tt) & (s.w > 0))) < k
    ]
    if short:
        raise DataInsufficient(
            f"The outcome regressions are not identified: {label} need at "
            f"least {k} observations each (one per regressor); too few in "
            f"(d, post) cells {sorted(short)}.",
            recovery_hint=(
                "Use fewer covariates, or a design with more observations in "
                "each treatment x period cell."
            ),
            diagnostics={
                "regressors": k,
                "short_cells": [list(c) for c in sorted(short)],
            },
        )


def _fit_cell(
    s: _Sample, dd: int, tt: int, w: np.ndarray, *, drop_collinear: bool
) -> _CellFit:
    """Fit the regression of ``y`` on the covariates within one cell.

    A covariate that is collinear with earlier columns within the cell (for
    example constant there) leaves the regression unidentified. With
    ``drop_collinear`` it is dropped and contributes nothing to the cell's
    predictions; otherwise the fit is refused. The reference behaves the
    same way (black box): its improved estimators drop such columns, the
    others stop with a singular-design error.
    """
    rows = _cell(s, dd, tt)
    cols = spanning_columns(s.x, rows, w)
    if cols.size < s.x.shape[1] and not drop_collinear:
        raise MethodIncompatibility(
            f"The outcome regression on the (d={dd}, post={tt}) cell is "
            "singular: a covariate is collinear with the others (or constant) "
            "within that cell.",
            recovery_hint=(
                "Remove the covariate, or use method='imp', which drops "
                "covariates that are collinear within a cell."
            ),
            diagnostics={
                "cell": [dd, tt],
                "rank": int(cols.size),
                "regressors": int(s.x.shape[1]),
            },
        )
    x = s.x[:, cols]
    return _CellFit(x, wls_coef(x, s.y, w, rows), rows, w)


def _doubly_robust(
    s: _Sample,
    *,
    trim_level: float,
    locally_efficient: bool,
    tilting: bool,
    variance_convention: str,
) -> RCSResult:
    """Shared body of :func:`drdid_rc` and :func:`drdid_imp_rc`."""
    p, odds, keep = _comparison_odds(s, trim_level, tilting)

    # Outcome regressions per (group, period) cell. Under tilting the
    # comparison cells are weighted by the odds (Sant'Anna-Zhao eq. 3.10).
    _require_identified(
        s, ((0, 0), (0, 1), (1, 0), (1, 1)), "all four treatment x period cells"
    )
    fits: Dict[Tuple[int, int], _CellFit] = {
        (dd, tt): _fit_cell(
            s,
            dd,
            tt,
            s.w * odds if (tilting and dd == 0) else s.w,
            drop_collinear=tilting,
        )
        for dd in (0, 1)
        for tt in (0, 1)
    }
    mu = {key: fit.fitted for key, fit in fits.items()}
    xs = {key: fit.x for key, fit in fits.items()}

    a11 = s.w * s.d * s.post
    a10 = s.w * s.d * (1.0 - s.post)
    comp = keep * s.w * odds * (1.0 - s.d)
    a01 = comp * s.post
    a00 = comp * (1.0 - s.post)

    t11 = HajekMean(a11, s.y - mu[0, 1])
    t10 = HajekMean(a10, s.y - mu[0, 0])
    c01 = HajekMean(a01, s.y - mu[0, 1])
    c00 = HajekMean(a00, s.y - mu[0, 0])
    att = (t11.theta - t10.theta) - (c01.theta - c00.theta)
    influence = (t11.influence - t10.influence) - (c01.influence - c00.influence)

    grad = {key: np.zeros(fit.x.shape[1]) for key, fit in fits.items()}
    grad[0, 1] = c01.grad_through_z(xs[0, 1]) - t11.grad_through_z(xs[0, 1])
    pre_term = t10.grad_through_z(xs[0, 0]) - c00.grad_through_z(xs[0, 0])
    if variance_convention == "reference":
        # The reference carries this pre-period comparison-regression term
        # with the opposite sign; see "The pre-period comparison regression"
        # in the specification.
        pre_term = -pre_term
    grad[0, 0] = pre_term

    if locally_efficient:
        a_d = s.w * s.d
        post_gap = mu[1, 1] - mu[0, 1]
        pre_gap = mu[1, 0] - mu[0, 0]
        e_post_all = HajekMean(a_d, post_gap)
        e_post_cell = HajekMean(a11, post_gap)
        e_pre_all = HajekMean(a_d, pre_gap)
        e_pre_cell = HajekMean(a10, pre_gap)
        att += (e_post_all.theta - e_post_cell.theta) - (
            e_pre_all.theta - e_pre_cell.theta
        )
        influence = influence + (e_post_all.influence - e_post_cell.influence)
        influence = influence - (e_pre_all.influence - e_pre_cell.influence)
        for key, sign in (((1, 1), 1.0), ((0, 1), -1.0)):
            g = e_post_all.grad_through_z(xs[key]) - e_post_cell.grad_through_z(xs[key])
            grad[key] = grad[key] + sign * g
        for key, sign in (((1, 0), -1.0), ((0, 0), 1.0)):
            g = e_pre_all.grad_through_z(xs[key]) - e_pre_cell.grad_through_z(xs[key])
            grad[key] = grad[key] + sign * g

    if not tilting:
        # Estimation effects of the outcome regressions and of the logit.
        for key, fit in fits.items():
            if np.any(grad[key] != 0.0):
                influence = influence + fit.effect(s.y, grad[key])
        g_ps = c00.grad_through_a(a00[:, None] * s.x) - c01.grad_through_a(
            a01[:, None] * s.x
        )
        influence = influence + logit_influence(s.x, s.d, s.w, p) @ g_ps

    return RCSResult(float(att), se_from_influence(influence), influence)


def drdid_rc(
    y: Any,
    post: Any,
    d: Any,
    covariates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    trim_level: float = _TRIM_LEVEL,
    locally_efficient: bool = True,
    *,
    variance_convention: str = "delta",
) -> RCSResult:
    """Doubly robust DiD for repeated cross-sections.

    Logit propensity score by maximum likelihood, outcome regressions by
    least squares on each (group, period) cell. ``locally_efficient=True``
    is Sant'Anna and Zhao (2020) eq. (3.4), which also uses the treated
    regressions; ``False`` is eq. (3.3). The influence function carries the
    estimation effects of the logit and of the outcome regressions.

    Parameters
    ----------
    y, post, d : array-like
        Outcome, post-period dummy and treated-group dummy, one per row.
    covariates : ndarray, optional
        Covariates without an intercept (one is added).
    weights : array-like, optional
        Non-negative observation weights (normalised to mean one).
    trim_level : float, default 0.995
        Comparison rows with a propensity score at or above this level get
        zero weight.
    locally_efficient : bool, default True
        Use eq. (3.4) rather than (3.3).
    variance_convention : {'delta', 'reference'}, default 'delta'
        Sign of the pre-period comparison-regression estimation effect.
        ``'delta'`` is the delta-method influence function of the paper's
        Appendix B. ``'reference'`` carries that term with the opposite
        sign, as R ``DRDID`` 1.3.0 does (black box); it reproduces R's
        standard error but under-covers when the propensity model is wrong
        (Monte Carlo coverage 0.81 against 0.95 for ``'delta'``,
        ``tests/reference_parity/_fixtures/_drdid_rc_pre_regression_mc.py``).
        The two agree asymptotically when the propensity model is correct;
        see ``docs/dev/cleanroom/did_drdid.md``.

        .. versionchanged:: 1.40.0
           The default was ``'reference'``.

    Returns
    -------
    RCSResult
        ``att``, ``se`` and the per-row ``influence``.

    References
    ----------
    [@santanna2020doubly]
    """
    _check_convention(variance_convention)
    s = _prepare(y, post, d, covariates, weights)
    return _doubly_robust(
        s,
        trim_level=trim_level,
        locally_efficient=locally_efficient,
        tilting=False,
        variance_convention=variance_convention,
    )


def drdid_imp_rc(
    y: Any,
    post: Any,
    d: Any,
    covariates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    trim_level: float = _TRIM_LEVEL,
    locally_efficient: bool = True,
) -> RCSResult:
    """Improved doubly robust DiD for repeated cross-sections.

    Sant'Anna and Zhao (2020) Section 3.2: the propensity score by inverse
    probability tilting, the comparison-cell regressions by least squares
    weighted by the fitted odds, the treated-cell regressions by ordinary
    least squares. Those first steps leave no estimation effect, so the
    influence function is the sum of the Hajek terms.
    ``locally_efficient=False`` gives eq. (3.9), ``True`` eq. (3.10).

    Parameters are as in :func:`drdid_rc`.

    Returns
    -------
    RCSResult

    References
    ----------
    [@santanna2020doubly]
    """
    s = _prepare(y, post, d, covariates, weights)
    return _doubly_robust(
        s,
        trim_level=trim_level,
        locally_efficient=locally_efficient,
        tilting=True,
        variance_convention="delta",
    )


def reg_did_rc(
    y: Any,
    post: Any,
    d: Any,
    covariates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> RCSResult:
    """Outcome-regression DiD for repeated cross-sections.

    Sant'Anna and Zhao (2020) eq. (2.2): the treated pre/post mean change
    minus the average, over treated rows, of the predicted comparison-group
    change from two cell regressions (comparison/pre, comparison/post).

    Parameters are as in :func:`drdid_rc` (no propensity score).

    Returns
    -------
    RCSResult

    References
    ----------
    [@santanna2020doubly]
    """
    s = _prepare(y, post, d, covariates, weights)
    _require_identified(s, ((0, 0), (0, 1)), "the two comparison-group cells")
    post_fit = _fit_cell(s, 0, 1, s.w, drop_collinear=False)
    pre_fit = _fit_cell(s, 0, 0, s.w, drop_collinear=False)
    t11 = HajekMean(s.w * s.d * s.post, s.y)
    t10 = HajekMean(s.w * s.d * (1.0 - s.post), s.y)
    gap = HajekMean(s.w * s.d, post_fit.fitted - pre_fit.fitted)
    att = t11.theta - t10.theta - gap.theta
    influence = (
        t11.influence
        - t10.influence
        - gap.influence
        - post_fit.effect(s.y, gap.grad_through_z(post_fit.x))
        + pre_fit.effect(s.y, gap.grad_through_z(pre_fit.x))
    )
    return RCSResult(float(att), se_from_influence(influence), influence)


def std_ipw_did_rc(
    y: Any,
    post: Any,
    d: Any,
    covariates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    trim_level: float = _TRIM_LEVEL,
) -> RCSResult:
    """IPW DiD for repeated cross-sections with Hajek-normalised weights.

    Each of the four cells is averaged with weights that sum to one: the
    treated cells with the observation weights, the comparison cells with
    the propensity odds. The influence function carries the logit
    estimation effect.

    Parameters are as in :func:`drdid_rc`.

    Returns
    -------
    RCSResult

    References
    ----------
    [@santanna2020doubly]
    """
    s = _prepare(y, post, d, covariates, weights)
    p, odds, keep = _comparison_odds(s, trim_level, tilting=False)
    comp = keep * s.w * odds * (1.0 - s.d)
    a01, a00 = comp * s.post, comp * (1.0 - s.post)
    t11 = HajekMean(s.w * s.d * s.post, s.y)
    t10 = HajekMean(s.w * s.d * (1.0 - s.post), s.y)
    c01, c00 = HajekMean(a01, s.y), HajekMean(a00, s.y)
    att = (t11.theta - t10.theta) - (c01.theta - c00.theta)
    g_ps = c00.grad_through_a(a00[:, None] * s.x) - c01.grad_through_a(
        a01[:, None] * s.x
    )
    influence = (
        (t11.influence - t10.influence)
        - (c01.influence - c00.influence)
        + logit_influence(s.x, s.d, s.w, p) @ g_ps
    )
    return RCSResult(float(att), se_from_influence(influence), influence)


def ipw_did_rc(
    y: Any,
    post: Any,
    d: Any,
    covariates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    trim_level: float = _TRIM_LEVEL,
) -> RCSResult:
    """IPW DiD for repeated cross-sections of Abadie type (not normalised).

    Sant'Anna and Zhao (2020) eq. (2.3):
    ``tau = E[(a_1 - a_0) (T - lambda) Y / (lambda (1 - lambda))] / E[a_1]``
    with ``a_1 = w D`` and ``a_0`` the trimmed odds weights of the
    comparison rows. Written as ``(A / L1 - B / L0) / P`` in the sample
    moments ``A = E_n[(a_1 - a_0) T Y]``, ``B = E_n[(a_1 - a_0) (1 - T) Y]``,
    ``P = E_n[a_1]``, ``L1 = E_n[w T]`` and ``L0 = E_n[w (1 - T)]``; the
    influence function is the delta method in those five moments plus the
    logit estimation effect.

    Parameters are as in :func:`drdid_rc`.

    Returns
    -------
    RCSResult

    References
    ----------
    [@santanna2020doubly]
    """
    s = _prepare(y, post, d, covariates, weights)
    p, odds, keep = _comparison_odds(s, trim_level, tilting=False)
    a1 = s.w * s.d
    a0 = keep * s.w * odds * (1.0 - s.d)
    contrast = a1 - a0
    mom_a = np.mean(contrast * s.post * s.y)
    mom_b = np.mean(contrast * (1.0 - s.post) * s.y)
    mass = float(a1.mean())
    l1 = float(np.mean(s.w * s.post))
    l0 = float(np.mean(s.w * (1.0 - s.post)))
    att = (mom_a / l1 - mom_b / l0) / mass

    per_row = contrast * s.y * (s.post / l1 - (1.0 - s.post) / l0)
    influence = (per_row - att * a1) / mass
    influence = influence - (mom_a / (l1**2 * mass)) * (s.w * s.post - l1)
    influence = influence + (mom_b / (l0**2 * mass)) * (s.w * (1.0 - s.post) - l0)
    # d tau / d gamma: a_0 moves by a_0 x, entering A and B with a minus sign.
    shifted = a0 * s.y * (s.post / l1 - (1.0 - s.post) / l0)
    g_ps = -(shifted[:, None] * s.x).mean(axis=0) / mass
    influence = influence + logit_influence(s.x, s.d, s.w, p) @ g_ps
    return RCSResult(float(att), se_from_influence(influence), influence)


def twfe_did_rc(
    y: Any,
    post: Any,
    d: Any,
    covariates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> RCSResult:
    """Two-way fixed-effects regression DiD for repeated cross-sections.

    Weighted least squares of ``y`` on ``(1, d, post, d * post, covariates)``;
    the ATT is the interaction coefficient (Sant'Anna and Zhao 2020,
    eq. 2.5). That coefficient equals the ATT only under homogeneous effects
    and no covariate-specific trends. The influence function is the
    least-squares influence row for the interaction and the standard error
    ``sd(IF) / sqrt(n)``.

    Parameters
    ----------
    y, post, d : array-like
    covariates : ndarray, optional
    weights : array-like, optional

    Returns
    -------
    RCSResult

    References
    ----------
    [@santanna2020doubly]
    """
    s = _prepare(y, post, d, covariates, weights)
    n = s.y.shape[0]
    z = np.column_stack([s.x[:, :1], s.d, s.post, s.d * s.post, s.x[:, 1:]])
    every = np.ones(n, dtype=bool)
    coef = wls_coef(z, s.y, s.w, every)
    influence = wls_influence(z, s.y, s.w, every, coef)[:, 3]
    se = float(np.std(influence, ddof=1) / np.sqrt(n))
    return RCSResult(float(coef[3]), se, influence)
