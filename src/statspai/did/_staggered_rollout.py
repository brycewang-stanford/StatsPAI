"""Efficient estimation for staggered rollout designs (Roth & Sant'Anna 2023).

Every other DiD estimator in StatsPAI identifies off a *parallel-trends*
assumption. When treatment timing is genuinely randomised — a policy lottery,
a phased platform launch, an RCT rolled out in waves — that assumption is not
what the design delivers, and conditioning on it throws away the randomisation.
Roth & Sant'Anna show that under random adoption timing there is an efficient
estimator in the class of linear combinations of cohort-period means, and it
strictly dominates the plug-in.

Construction
------------
Collapse the panel to cohort-level means ``Ybar_g`` (a vector over periods),
their within-cohort covariance ``S_g``, and cohort sizes ``N_g``. Two weight
matrices index the same cohorts:

* ``A_theta`` places weight on the *outcome* period of each identified
  ``(g, t)`` cell — this is the estimand.
* ``A_0`` places the same cell weights on the cohort's last *pre*-treatment
  period ``g - 1``. Under random timing those cells have expectation zero, so
  they are valid controls and carry no bias.

The estimator is then ``theta(beta) = theta_0 - Xhat' beta`` with
``theta_0 = sum_g A_theta[g] Ybar_g`` and ``Xhat = sum_g A_0[g] Ybar_g``.
``beta = 1`` reproduces the usual plug-in; the efficient choice solves
``beta* = Xvar^-1 X_theta_cov``, i.e. it uses the pre-period moments as
optimal controls.

Two standard errors
-------------------
The conservative (Neyman) variance treats the estimated ``beta`` as fixed. The
*adjusted* variance subtracts the term the randomisation lets you recover —
under random timing the pre-period covariance is partly known, so the
conservative bound can be tightened. ``se_type='neyman'`` (the StatsPAI
default) reports the former; ``se_type='adjusted'`` reports the latter, which
is what R ``staggered`` prints as its primary ``se``. The adjusted SE is never
larger, so the default is the cautious one.

Randomisation inference
-----------------------
``fisher=True`` runs a Fisher randomisation test: adoption dates are permuted
across units — which is exactly the null the design licenses — and the
studentised statistic is recomputed on each draw. The p-value is the share of
draws whose ``|t|`` exceeds the observed one. This needs no asymptotics and is
the natural companion to a randomised rollout.

.. warning::
   Never-treated units must be coded ``g = inf``, not ``0``. With ``g = 0``
   they are read as a cohort treated before the sample and the estimate is
   badly wrong (on ``did::mpdta``: −0.370 instead of −0.047). This function
   accepts ``0``/``NaN``/``None`` and recodes them, so the trap cannot be
   reached through the public API.

Verified against R ``staggered`` 1.2.2 on canonical ``did::mpdta``: every
estimand × ``beta`` × ``use_last_treated_only`` combination matches to ~1e-10
for the estimate and both standard errors, as do the event-study path and the
``staggered_cs`` / ``staggered_sa`` wrappers. See
``tests/reference_parity/test_staggered_rollout_parity.py``.

References
----------
Roth, J. and Sant'Anna, P.H.C. (2023). "Efficient Estimation for Staggered
Rollout Designs." *Journal of Political Economy Microeconomics*, 1(4),
669-709. [@roth2023efficient]
"""

from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.results import CausalResult

__all__ = [
    "StaggeredRolloutResult",
    "staggered_rollout",
    "staggered_rollout_core",
    "staggered_cs",
    "staggered_sa",
]

_ESTIMANDS = ("simple", "cohort", "calendar", "eventstudy")
_SE_TYPES = ("neyman", "adjusted")

# MASS::ginv keeps singular values above sqrt(.Machine$double.eps) * d[1];
# numpy's rcond is the same relative cutoff, so this reproduces R exactly.
_GINV_RCOND = float(np.sqrt(np.finfo(float).eps))


class StaggeredRolloutResult(NamedTuple):
    """Point estimate, both standard errors, and the fitted control weights."""

    estimate: float
    se: float
    se_neyman: float
    se_adjusted: float
    beta: np.ndarray
    estimand: str
    efficient: bool
    cohorts: np.ndarray
    cohort_sizes: np.ndarray
    n_units: int
    se_type: str = "neyman"
    event_time: Optional[float] = None
    fisher_pvalue: Optional[float] = None
    fisher_pvalue_neyman: Optional[float] = None
    fisher_pvalue_adjusted: Optional[float] = None
    n_fisher: Optional[int] = None


class _Panel(NamedTuple):
    """A balanced panel in wide form, plus each unit's adoption date."""

    values: np.ndarray  # (n_units, n_periods)
    g_of_unit: np.ndarray  # (n_units,)
    t_list: np.ndarray  # (n_periods,)


class _Summaries(NamedTuple):
    """Per-cohort mean paths, within-cohort covariances and sizes."""

    g_list: np.ndarray
    ybar: List[np.ndarray]
    cov: List[np.ndarray]
    sizes: np.ndarray


def _wide_panel(data: pd.DataFrame, i: str, t: str, g: str, y: str) -> _Panel:
    """Pivot to units x periods, recoding never-treated and dropping singletons."""
    for col in (i, t, g, y):
        if col not in data.columns:
            raise MethodIncompatibility(
                f"staggered_rollout: column {col!r} not in data.",
                recovery_hint="Check the y / i / t / g arguments.",
                diagnostics={"columns": list(data.columns)},
            )

    df = data[[i, t, g, y]].copy()
    gv = pd.to_numeric(df[g], errors="coerce")
    # Never-treated may arrive as 0, NaN or None; all three mean "never".
    df["_g"] = np.where(gv.isna() | (gv == 0), np.inf, gv.astype(float))
    df["_y"] = pd.to_numeric(df[y], errors="coerce").astype(float)

    wide = df.pivot_table(index=[i, "_g"], columns=t, values="_y", aggfunc="first")
    if wide.isna().to_numpy().any():
        raise DataInsufficient(
            "the staggered-rollout estimator needs a balanced panel; some "
            "unit-period cells are missing.",
            recovery_hint="Balance the panel first (e.g. sp.balance_panel).",
            diagnostics={"n_missing": int(wide.isna().to_numpy().sum())},
        )

    t_list = np.asarray(wide.columns, dtype=float)
    g_of_unit = wide.index.get_level_values("_g").to_numpy(dtype=float)
    values = wide.to_numpy(dtype=float)

    # R drops cohorts with a single cross-sectional unit (their within-cohort
    # covariance is not estimable) and warns. Match that rather than failing:
    # a lone unit in one cohort should not sink an otherwise usable design.
    uniq, counts = np.unique(g_of_unit, return_counts=True)
    singleton = uniq[counts == 1]
    if singleton.size:
        keep = ~np.isin(g_of_unit, singleton)
        warnings.warn(
            "staggered_rollout: treatment cohort(s) g = "
            + ", ".join(f"{s:g}" for s in singleton)
            + " have a single cross-sectional unit each, so their "
            "within-cohort covariance is not estimable. Dropping them "
            f"({int((~keep).sum())} unit(s)).",
            UserWarning,
            stacklevel=3,
        )
        values, g_of_unit = values[keep], g_of_unit[keep]

    return _Panel(values=values, g_of_unit=g_of_unit, t_list=t_list)


def _summarize(panel: _Panel) -> _Summaries:
    """Collapse a wide panel to per-cohort means, covariances and sizes."""
    g_list = np.array(sorted(set(panel.g_of_unit.tolist())), dtype=float)
    if g_list.size < 2:
        raise DataInsufficient(
            "the staggered-rollout estimator needs at least two treatment "
            "cohorts (including never-treated).",
            recovery_hint="Check the treatment-timing column.",
            diagnostics={"n_cohorts": int(g_list.size)},
        )

    ybar, cov, sizes = [], [], []
    for gg in g_list:
        block = panel.values[panel.g_of_unit == gg]
        ybar.append(block.mean(axis=0))
        cov.append(np.atleast_2d(np.cov(block, rowvar=False)))
        sizes.append(block.shape[0])
    return _Summaries(
        g_list=g_list,
        ybar=ybar,
        cov=cov,
        sizes=np.asarray(sizes, dtype=float),
    )


def _cell_weights(
    t_val: float,
    g_val: float,
    g_list: np.ndarray,
    sizes: np.ndarray,
    use_last_treated_only: bool = False,
) -> np.ndarray:
    """Cohort weights for the ATE(g, t) contrast.

    The treated cohort enters with +1. Controls share −1 in proportion to
    size: every cohort not yet treated by ``max(t, g)``, or — when
    ``use_last_treated_only`` — only the last-treated cohort, which is what
    Sun & Abraham's estimator uses.
    """
    v = np.zeros(g_list.size)
    if use_last_treated_only:
        control = np.where((g_list > t_val) & (g_list == g_list.max()))[0]
    else:
        control = np.where(g_list > max(t_val, g_val))[0]
    if control.size == 0:
        raise DataInsufficient(
            f"ATE(g={g_val}, t={t_val}) has no not-yet-treated control "
            "cohort, so it is not identified.",
            recovery_hint="Restrict the horizon, or add never-treated units "
            "(coded g=inf).",
            diagnostics={"g": float(g_val), "t": float(t_val)},
        )
    v[control] = -sizes[control] / sizes[control].sum()
    v[int(np.where(g_list == g_val)[0][0])] += 1.0
    return np.asarray(v, dtype=float)


def _rows_to_list(a_zero: np.ndarray) -> List[np.ndarray]:
    """One ``(1, T)`` control block per cohort — the DiD form of ``A_0``."""
    return [a_zero[k : k + 1, :] for k in range(a_zero.shape[0])]


def _general_a0_list(g_list: np.ndarray, t_list: np.ndarray) -> List[np.ndarray]:
    """``A_0`` using **every** pre-period as a control, not just ``g - 1``.

    This is the general form of the estimator (R's ``use_DiD_A0 = FALSE``).
    Each cohort contributes an identity block over its own pre-periods, the
    blocks are stacked, and the last cohort carries minus their sum so the
    controls have expectation zero. The result is a ``K``-dimensional control
    vector rather than the single DiD contrast, so ``beta`` becomes a vector
    and the estimator can only get more efficient — at the cost of estimating
    more nuisance weights, which is why it is not the default.
    """
    n_t = t_list.size
    blocks: List[Tuple[int, np.ndarray]] = []
    for k in range(g_list.size - 1):
        n_pre = int(np.sum(t_list < g_list[k]))
        if n_pre == 0:
            continue
        block = np.zeros((n_pre, n_t))
        np.fill_diagonal(block, 1.0)
        blocks.append((k, block))

    if not blocks:
        raise DataInsufficient(
            "no cohort has an observed pre-treatment period, so the general "
            "control set is empty.",
            recovery_hint="Use use_did_a0=True, or extend the sample window "
            "backwards.",
            diagnostics={"n_cohorts": int(g_list.size)},
        )

    total = sum(block.shape[0] for _, block in blocks)
    a_zero = [np.zeros((total, n_t)) for _ in range(g_list.size)]
    row = 0
    for k, block in blocks:
        a_zero[k][row : row + block.shape[0], :] = block
        row += block.shape[0]
    a_zero[-1] = -sum(a_zero[:-1])
    return a_zero


def _weight_matrices(
    estimand: str,
    g_list: np.ndarray,
    t_list: np.ndarray,
    sizes: np.ndarray,
    event_time: float = 0.0,
    use_last_treated_only: bool = False,
    use_did_a0: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Build ``A_theta`` (cohorts x periods) and the per-cohort ``A_0`` blocks."""
    n_g, n_t = g_list.size, t_list.size
    g_max = g_list.max()

    if estimand == "eventstudy":
        a_theta, a_zero = _event_study_weights(
            g_list, t_list, sizes, event_time, use_last_treated_only
        )
        return a_theta, (
            _rows_to_list(a_zero) if use_did_a0 else _general_a0_list(g_list, t_list)
        )

    pairs = [(gg, tt) for gg in g_list for tt in t_list if tt >= gg and tt < g_max]
    if not pairs:
        raise DataInsufficient(
            "no identified (cohort, period) cells: every cohort is treated "
            "at or after the last period.",
            recovery_hint="Check the treatment-timing column and the sample " "window.",
            diagnostics={"n_cohorts": int(n_g), "n_periods": int(n_t)},
        )

    idx_of_g = {float(gg): k for k, gg in enumerate(g_list)}
    # 'simple' weights every identified (g, t) cell by cohort size, so a
    # cohort with more post-periods counts more. 'cohort' first averages
    # within a cohort (equal weight per own treated period) and only then
    # weights cohorts by size. 'calendar' averages within a period.
    n_total = sum(sizes[idx_of_g[float(gg)]] for gg, _ in pairs)
    eligible = sorted({float(gg) for gg, _ in pairs})
    n_total_eligible = sum(sizes[idx_of_g[gg]] for gg in eligible)
    n_treated_periods = {
        gg: sum(1 for g2, _ in pairs if float(g2) == gg) for gg in eligible
    }

    a_theta = np.zeros((n_g, n_t))
    a_zero = np.zeros((n_g, n_t))
    for gg, tt in pairs:
        if estimand == "cohort":
            w_cell = (sizes[idx_of_g[float(gg)]] / n_total_eligible) / (
                n_treated_periods[float(gg)]
            )
        elif estimand == "calendar":
            n_at_t = sum(sizes[idx_of_g[float(g2)]] for g2, t2 in pairs if t2 == tt)
            n_periods = len({t2 for _, t2 in pairs})
            w_cell = sizes[idx_of_g[float(gg)]] / (n_at_t * n_periods)
        else:  # simple
            w_cell = sizes[idx_of_g[float(gg)]] / n_total

        contrast = _cell_weights(tt, gg, g_list, sizes, use_last_treated_only)
        a_theta[:, int(np.where(t_list == tt)[0][0])] += w_cell * contrast
        pre = np.where(t_list == gg - 1)[0]
        if pre.size:
            a_zero[:, int(pre[0])] += w_cell * contrast
    return a_theta, (
        _rows_to_list(a_zero) if use_did_a0 else _general_a0_list(g_list, t_list)
    )


def _event_study_weights(
    g_list: np.ndarray,
    t_list: np.ndarray,
    sizes: np.ndarray,
    event_time: float,
    use_last_treated_only: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """``A_theta`` / ``A_0`` for the ATT ``event_time`` periods after adoption.

    A cohort contributes only if the cell it would supply is identified:
    ``max(g + e, g)`` must fall strictly before the last adoption date and
    within the observed window. Eligible cohorts are weighted by size, the
    outcome weight lands on period ``g + e`` and the control weight on the
    cohort's last pre-period ``g - 1``.
    """
    g_max = g_list.max()
    horizon = np.maximum(g_list + event_time, g_list)
    eligible = np.where((horizon < g_max) & (horizon <= t_list.max()))[0]
    if eligible.size == 0:
        raise DataInsufficient(
            f"no cohort supplies an identified cell at event time "
            f"{event_time:g}: every cohort would need a comparison cohort "
            "treated later than the horizon.",
            recovery_hint="Ask for an event time nearer zero, or extend the "
            "sample window.",
            diagnostics={"event_time": float(event_time)},
        )

    n_eligible = sizes[eligible].sum()
    a_theta = np.zeros((g_list.size, t_list.size))
    a_zero = np.zeros((g_list.size, t_list.size))
    for k in eligible:
        gg = float(g_list[k])
        w_cell = sizes[k] / n_eligible
        contrast = _cell_weights(
            gg + event_time, gg, g_list, sizes, use_last_treated_only
        )
        post = np.where(t_list == gg + event_time)[0]
        if post.size:
            a_theta[:, int(post[0])] += w_cell * contrast
        pre = np.where(t_list == gg - 1)[0]
        if pre.size:
            a_zero[:, int(pre[0])] += w_cell * contrast

    if not np.any(a_theta):
        raise DataInsufficient(
            f"event time {event_time:g} lands outside the observed periods "
            "for every eligible cohort.",
            recovery_hint="Ask for an event time inside the sample window.",
            diagnostics={"event_time": float(event_time)},
        )
    return a_theta, a_zero


class _Moments(NamedTuple):
    """The five sums the estimator and both variances are built from."""

    theta0: float
    xhat: np.ndarray
    x_var: np.ndarray
    x_theta: np.ndarray
    theta_var: np.ndarray


def _moments(a_theta: np.ndarray, a_zero: List[np.ndarray], s: _Summaries) -> _Moments:
    """Cohort-summed moments of ``A_theta`` and ``A_0`` against the data.

    ``a_zero[k]`` is ``(K, T)``: one row under the DiD control set, ``K`` rows
    under the general one. Everything downstream is already matrix-valued, so
    the two cases differ only in ``K``.
    """
    n_g = s.g_list.size
    at = [a_theta[k : k + 1, :] for k in range(n_g)]
    a0 = a_zero
    return _Moments(
        theta0=float(sum(float((at[k] @ s.ybar[k]).item()) for k in range(n_g))),
        xhat=sum(a0[k] @ s.ybar[k] for k in range(n_g)).reshape(-1, 1),
        x_var=sum(a0[k] @ s.cov[k] @ a0[k].T / s.sizes[k] for k in range(n_g)),
        x_theta=sum(a0[k] @ s.cov[k] @ at[k].T / s.sizes[k] for k in range(n_g)),
        theta_var=sum(at[k] @ s.cov[k] @ at[k].T / s.sizes[k] for k in range(n_g)),
    )


def _beta_star(m: _Moments) -> np.ndarray:
    """The efficient control weights ``Xvar^-1 X_theta_cov``."""
    try:
        return np.asarray(np.linalg.solve(m.x_var, m.x_theta), dtype=float)
    except np.linalg.LinAlgError as exc:
        raise DataInsufficient(
            "the pre-period control moments are collinear, so the "
            "efficient weights are not identified.",
            recovery_hint="Use efficient=False for the plug-in estimator.",
            diagnostics={"n_controls": int(m.x_var.shape[0])},
        ) from exc


def _n_pre_periods(g_min: float, t_list: np.ndarray) -> int:
    """How many observed periods precede ``g_min``.

    The reference implementation writes this as ``g_min - t_min``, which is
    the same count whenever periods are consecutive integers and silently
    wrong when they are not. Count directly and refuse the mismatch.
    """
    n_pre = int(np.sum(t_list < g_min))
    implied = g_min - float(t_list.min())
    if not np.isclose(implied, n_pre):
        raise MethodIncompatibility(
            "the adjusted (non-conservative) standard error assumes "
            "consecutive integer periods: with periods "
            f"{t_list.min():g}..{t_list.max():g} and first treated cohort "
            f"{g_min:g}, the pre-period count is ambiguous.",
            recovery_hint="Re-index periods as consecutive integers, or use "
            "se_type='neyman'.",
            diagnostics={"g_min": float(g_min), "n_pre_counted": n_pre},
        )
    return n_pre


def _adjustment_terms(
    a_theta: np.ndarray,
    s: _Summaries,
    t_list: np.ndarray,
    g_min: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Pieces of the variance the randomisation lets you subtract.

    Returns ``(betahat_g_sum, avg_MSM, N)``. The first two are ``None`` when
    no adjustment applies — that happens when the first cohort with any
    outcome weight adopts at or before the first observed period, so there is
    no pre-period covariance to exploit.
    """
    n_total = float(s.sizes.sum())
    nonzero = np.where(np.max(np.abs(a_theta), axis=1) != 0)[0]
    if nonzero.size == 0:
        return None, None, n_total
    if g_min is None:
        g_min = float(s.g_list[nonzero.min()])
    if g_min <= float(t_list.min()):
        return None, None, n_total

    n_pre = _n_pre_periods(g_min, t_list)
    if n_pre == 0:
        return None, None, n_total

    idx = np.where(s.g_list >= g_min)[0]
    # M selects the pre-gMin periods; M S_g M' is the leading block of S_g.
    betahat_sum = np.zeros((n_pre, 1))
    msm_sum = np.zeros((n_pre, n_pre))
    for k in idx:
        s_g = s.cov[k]
        msm = s_g[:n_pre, :n_pre]
        m_s_atheta = (s_g[:n_pre, :] @ a_theta[k : k + 1, :].T).reshape(n_pre, 1)
        betahat_sum += np.linalg.pinv(msm, rcond=_GINV_RCOND) @ m_s_atheta
        msm_sum += msm
    return betahat_sum, msm_sum / float(idx.size), n_total


def _variances(
    beta: np.ndarray,
    m: _Moments,
    a_theta: np.ndarray,
    s: _Summaries,
    t_list: np.ndarray,
) -> Tuple[float, float]:
    """Conservative (Neyman) and adjusted standard errors."""
    var_cons = float(
        (m.theta_var + beta.T @ m.x_var @ beta - 2 * m.x_theta.T @ beta).item()
    )
    if var_cons < 0:
        # Neyman variance is conservative but can go negative in tiny samples.
        warnings.warn(
            "staggered_rollout: the estimated conservative variance is "
            f"negative ({var_cons:.3g}); reporting a zero standard error. "
            "This is a small-sample artefact, not a precise estimate.",
            UserWarning,
            stacklevel=3,
        )
        var_cons = 0.0

    betahat_sum, avg_msm, n_total = _adjustment_terms(a_theta, s, t_list)
    if betahat_sum is None:
        return float(np.sqrt(var_cons)), float(np.sqrt(var_cons))

    adjustment = float((betahat_sum.T @ avg_msm @ betahat_sum).item()) / n_total
    var_adj = var_cons - adjustment
    if var_adj < 0:
        if var_cons != 0:
            warnings.warn(
                "staggered_rollout: the conservative variance is smaller "
                "than the randomisation adjustment, so the adjusted standard "
                "error is reported as zero. Use se_type='neyman' for the "
                "conservative bound.",
                UserWarning,
                stacklevel=3,
            )
        var_adj = 0.0
    return float(np.sqrt(var_cons)), float(np.sqrt(var_adj))


def _fit(
    s: _Summaries,
    a_theta: np.ndarray,
    a_zero: List[np.ndarray],
    t_list: np.ndarray,
    efficient: bool,
) -> Tuple[float, float, float, np.ndarray]:
    """Estimate, conservative SE, adjusted SE, control weights."""
    m = _moments(a_theta, a_zero, s)
    beta = _beta_star(m) if efficient else np.ones((m.x_var.shape[0], 1))
    estimate = m.theta0 - float((m.xhat.T @ beta).item())
    se_neyman, se_adjusted = _variances(beta, m, a_theta, s, t_list)
    return estimate, se_neyman, se_adjusted, np.asarray(beta, dtype=float).ravel()


def _fisher_pvalues(
    panel: _Panel,
    a_theta: np.ndarray,
    a_zero: List[np.ndarray],
    efficient: bool,
    estimate: float,
    se_neyman: float,
    se_adjusted: float,
    n_fisher: int,
    random_state: Optional[int],
) -> Tuple[float, float, int]:
    """Randomisation-test p-values from permuting adoption dates across units.

    Permuting the vector of adoption dates leaves every cohort size — and so
    both weight matrices — untouched; only which unit sits in which cohort
    moves. That is exactly the null the design licenses, and it lets the
    weights be built once and reused across draws.
    """

    def _tstat(est: float, se: float) -> float:
        return float(np.inf) if se == 0 else abs(est / se)

    obs_adj, obs_ney = _tstat(estimate, se_adjusted), _tstat(estimate, se_neyman)
    rng = np.random.default_rng(random_state)
    exceed_adj = exceed_ney = 0
    n_ok = 0
    n_failed = 0
    for _ in range(int(n_fisher)):
        permuted = panel._replace(g_of_unit=rng.permutation(panel.g_of_unit))
        try:
            est_p, ney_p, adj_p, _ = _fit(
                _summarize(permuted), a_theta, a_zero, panel.t_list, efficient
            )
        except (DataInsufficient, MethodIncompatibility, np.linalg.LinAlgError):
            # A draw can be degenerate (e.g. collinear control moments). The
            # reference drops those and shrinks the denominator; so do we.
            n_failed += 1
            continue
        n_ok += 1
        exceed_adj += _tstat(est_p, adj_p) > obs_adj
        exceed_ney += _tstat(est_p, ney_p) > obs_ney

    if n_ok == 0:
        raise DataInsufficient(
            "every Fisher permutation draw failed, so the randomisation "
            "p-value is not computable.",
            recovery_hint="Check for near-collinear pre-period moments, or "
            "use fisher=False.",
            diagnostics={"n_fisher": int(n_fisher)},
        )
    if n_failed:
        warnings.warn(
            f"staggered_rollout: {n_failed} of {int(n_fisher)} Fisher "
            "permutation draws failed and were dropped; the p-value uses "
            f"the remaining {n_ok}.",
            UserWarning,
            stacklevel=3,
        )
    return exceed_adj / n_ok, exceed_ney / n_ok, n_ok


def _event_time_grid(
    event_time: Union[float, Sequence[float]],
) -> Tuple[np.ndarray, bool]:
    """Normalise ``event_time`` to an array, remembering if it was scalar."""
    if isinstance(event_time, (int, float, np.integer, np.floating)):
        return np.asarray([float(event_time)], dtype=float), True
    grid = np.asarray(list(event_time), dtype=float)
    if grid.size == 0:
        raise MethodIncompatibility(
            "staggered_rollout: `event_time` is empty.",
            recovery_hint="Pass a scalar event time or a non-empty sequence.",
            diagnostics={"event_time": list(np.atleast_1d(event_time))},
        )
    return grid, False


def staggered_rollout_core(
    data: pd.DataFrame,
    i: str,
    t: str,
    g: str,
    y: str,
    estimand: str = "simple",
    efficient: bool = True,
    event_time: float = 0.0,
    use_last_treated_only: bool = False,
    use_did_a0: bool = True,
    se_type: str = "neyman",
    fisher: bool = False,
    n_fisher: int = 500,
    random_state: Optional[int] = None,
) -> StaggeredRolloutResult:
    """Roth-Sant'Anna estimator for a randomised staggered rollout."""
    if estimand not in _ESTIMANDS:
        raise MethodIncompatibility(
            f"estimand must be one of {list(_ESTIMANDS)}; got {estimand!r}.",
            recovery_hint="Pass estimand='simple', 'cohort', 'calendar' or "
            "'eventstudy'.",
            diagnostics={"estimand": estimand},
        )
    if se_type not in _SE_TYPES:
        raise MethodIncompatibility(
            f"se_type must be one of {list(_SE_TYPES)}; got {se_type!r}.",
            recovery_hint="Pass se_type='neyman' (conservative) or "
            "'adjusted' (R staggered's primary SE).",
            diagnostics={"se_type": se_type},
        )

    if not use_did_a0 and not efficient:
        raise MethodIncompatibility(
            "the plug-in estimator (efficient=False) is only defined for the "
            "DiD control set: it fixes beta = 1, which is a single contrast, "
            "while use_did_a0=False supplies a vector of pre-period controls. "
            "Subtracting their unweighted sum is not a recognised estimator "
            "and has no reference implementation.",
            recovery_hint="Use efficient=True with use_did_a0=False, or "
            "efficient=False with use_did_a0=True.",
            diagnostics={"efficient": efficient, "use_did_a0": use_did_a0},
        )

    panel = _wide_panel(data, i=i, t=t, g=g, y=y)
    s = _summarize(panel)
    a_theta, a_zero = _weight_matrices(
        estimand,
        s.g_list,
        panel.t_list,
        s.sizes,
        event_time=float(event_time),
        use_last_treated_only=use_last_treated_only,
        use_did_a0=use_did_a0,
    )
    estimate, se_neyman, se_adjusted, beta = _fit(
        s, a_theta, a_zero, panel.t_list, efficient
    )

    p_adj = p_ney = None
    n_ok = None
    if fisher:
        p_adj, p_ney, n_ok = _fisher_pvalues(
            panel,
            a_theta,
            a_zero,
            efficient,
            estimate,
            se_neyman,
            se_adjusted,
            n_fisher,
            random_state,
        )

    return StaggeredRolloutResult(
        estimate=estimate,
        se=se_adjusted if se_type == "adjusted" else se_neyman,
        se_neyman=se_neyman,
        se_adjusted=se_adjusted,
        beta=beta,
        estimand=estimand,
        efficient=bool(efficient),
        cohorts=s.g_list,
        cohort_sizes=s.sizes,
        n_units=int(s.sizes.sum()),
        se_type=se_type,
        event_time=float(event_time) if estimand == "eventstudy" else None,
        fisher_pvalue=p_adj if se_type == "adjusted" else p_ney,
        fisher_pvalue_neyman=p_ney,
        fisher_pvalue_adjusted=p_adj,
        n_fisher=n_ok,
    )


def _event_study_vcv(
    s: _Summaries,
    weights: List[Tuple[np.ndarray, List[np.ndarray]]],
    betas: List[np.ndarray],
    t_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Joint (conservative, adjusted) covariance across event times.

    Stacking ``A_theta - beta' A_0`` per event time turns each estimate into
    one linear functional of the same cohort means, so their covariance is
    just the cohort-summed sandwich of the stacked matrix.
    """
    n_e, n_g = len(weights), s.g_list.size
    combined = [np.zeros((n_e, t_list.size)) for _ in range(n_g)]
    for e, ((a_theta, a_zero), beta) in enumerate(zip(weights, betas)):
        b = np.asarray(beta, dtype=float).reshape(1, -1)
        for k in range(n_g):
            combined[k][e, :] = a_theta[k, :] - (b @ a_zero[k]).ravel()

    vcv_neyman = np.zeros((n_e, n_e))
    for k in range(n_g):
        vcv_neyman += combined[k] @ s.cov[k] @ combined[k].T / s.sizes[k]

    # The adjustment uses one common gMin across event times, as in R.
    g_mins = []
    for a_theta, _ in weights:
        nonzero = np.where(np.max(np.abs(a_theta), axis=1) != 0)[0]
        if nonzero.size:
            g_mins.append(float(s.g_list[nonzero.min()]))
    if not g_mins:
        return vcv_neyman, vcv_neyman
    g_min = min(g_mins)

    # Every event time shares one gMin, so ``avg_MSM`` is the same matrix for
    # all of them; the reference takes the first and so do we.
    stacked, avg_msm, n_total = [], None, float(s.sizes.sum())
    for a_theta, _ in weights:
        betahat_sum, msm, _ = _adjustment_terms(a_theta, s, t_list, g_min=g_min)
        if betahat_sum is None:
            return vcv_neyman, vcv_neyman
        stacked.append(betahat_sum.ravel())
        if avg_msm is None:
            avg_msm = msm
    stacked_arr = np.asarray(stacked, dtype=float)
    vcv = vcv_neyman - stacked_arr @ avg_msm @ stacked_arr.T / n_total
    neg = np.where(np.diag(vcv) < 0)[0]
    if neg.size:
        vcv[np.ix_(neg, neg)] = 0.0
    return vcv_neyman, vcv


def staggered_rollout(
    data: pd.DataFrame,
    y: str,
    i: str,
    t: str,
    g: str,
    estimand: str = "simple",
    efficient: bool = True,
    event_time: Union[float, Sequence[float]] = 0.0,
    use_last_treated_only: bool = False,
    use_did_a0: bool = True,
    se_type: str = "neyman",
    fisher: bool = False,
    n_fisher: int = 500,
    random_state: Optional[int] = None,
    alpha: float = 0.05,
) -> "CausalResult":
    """Efficient DiD for a **randomised** staggered rollout (Roth-Sant'Anna 2023).

    Use this when treatment *timing* was randomly assigned — a policy lottery,
    a phased platform launch, an RCT rolled out in waves. Every other DiD
    estimator in StatsPAI identifies off parallel trends, which under random
    timing is both unnecessary and wasteful: this estimator uses the
    randomisation directly and is efficient in the class of linear
    combinations of cohort-period means.

    .. warning::
       This is a **different estimand and identifying assumption** from
       ``sp.callaway_santanna``. On canonical ``did::mpdta`` (where timing is
       *not* randomised) this returns −0.0471 against CS's −0.0400; the gap is
       not a discrepancy, it is what happens when you apply a design-based
       estimator to an observational rollout. If timing was not randomised,
       use a parallel-trends estimator instead.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel in long form.
    y, i, t, g : str
        Outcome, unit id, period, and first-treatment period. Never-treated
        units may be coded ``0``, ``NaN`` or ``inf``; all three are accepted.
    estimand : {'simple', 'cohort', 'calendar', 'eventstudy'}, default 'simple'
        ``'simple'`` weights each identified cohort-period cell by cohort
        size; ``'cohort'`` averages within a cohort first, then weights
        cohorts by size; ``'calendar'`` averages within a calendar period;
        ``'eventstudy'`` reports the ATT ``event_time`` periods after
        adoption.
    efficient : bool, default True
        ``True`` uses the optimal pre-period control weights; ``False`` gives
        the plug-in estimator (``beta = 1``), which is the simple
        difference-in-means analogue and matches R's ``beta = 1``.
    event_time : float or sequence of float, default 0
        Only read when ``estimand='eventstudy'``. A sequence returns one row
        per event time in ``.detail``, with the joint covariance in
        ``model_info['vcov']``; ``.estimate`` then averages the requested
        non-negative event times (all of them if none is non-negative), with
        the SE taken from that covariance.
    use_last_treated_only : bool, default False
        Restrict the comparison group to the last-treated cohort, which is
        what Sun & Abraham's estimator does. The default uses every
        not-yet-treated cohort.
    use_did_a0 : bool, default True
        Which controls the efficient weights are chosen over. ``True`` uses
        the single DiD contrast at each cohort's last pre-period ``g - 1``.
        ``False`` uses **every** pre-period as a separate control, so ``beta``
        becomes a vector — the general form of the estimator, weakly more
        efficient at the cost of estimating more nuisance weights. Requires
        ``efficient=True``: the plug-in fixes ``beta = 1``, which is a single
        contrast and has no meaning against a vector control set.
    se_type : {'neyman', 'adjusted'}, default 'neyman'
        Which standard error lands in ``.se``. ``'neyman'`` is the
        conservative bound that treats the fitted control weights as fixed;
        ``'adjusted'`` subtracts the part of the variance the randomisation
        identifies and is what R ``staggered`` prints. Both are always
        available in ``model_info``. The adjusted SE is never larger, so the
        default is the cautious one.
    fisher : bool, default False
        Run a Fisher randomisation test: permute adoption dates across units
        and compare studentised statistics. The p-value lands in
        ``model_info['fisher_pvalue']``.
    n_fisher : int, default 500
        Permutation draws for the randomisation test.
    random_state : int, optional
        Seed for the permutation draws.
    alpha : float, default 0.05
        Level for the reported confidence interval.

    Returns
    -------
    CausalResult
        ``.estimate`` / ``.se`` carry the ATT and its standard error;
        ``model_info`` holds ``beta`` (the fitted control weights), both
        standard errors, and any randomisation p-values.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=6, staggered=True, seed=7)
    >>> res = sp.staggered_rollout(df, y='y', i='unit', t='time',
    ...                            g='first_treat')
    >>> res.se > 0
    True

    See Also
    --------
    staggered_cs : the Callaway-Sant'Anna estimand with design-based inference.
    staggered_sa : the Sun-Abraham estimand with design-based inference.

    References
    ----------
    roth2023efficient
    """
    from scipy import stats as _stats

    from ..core.results import CausalResult

    grid, is_scalar = _event_time_grid(event_time)
    if estimand != "eventstudy":
        grid, is_scalar = np.asarray([0.0]), True

    cores = [
        staggered_rollout_core(
            data,
            i=i,
            t=t,
            g=g,
            y=y,
            estimand=estimand,
            efficient=efficient,
            event_time=float(e),
            use_last_treated_only=use_last_treated_only,
            use_did_a0=use_did_a0,
            se_type=se_type,
            fisher=fisher,
            n_fisher=n_fisher,
            random_state=random_state,
        )
        for e in grid
    ]

    z = float(_stats.norm.ppf(1 - alpha / 2))
    model_info: Dict[str, Any] = {
        "estimator": "staggered_rollout",
        "estimand_type": estimand,
        "efficient": cores[0].efficient,
        "beta": cores[0].beta,
        "se_type": se_type,
        "use_last_treated_only": bool(use_last_treated_only),
        "use_did_a0": bool(use_did_a0),
        "identifying_assumption": "random adoption timing",
        "n_cohorts": int(cores[0].cohorts.size),
    }
    if fisher:
        model_info["n_fisher"] = cores[0].n_fisher

    if is_scalar:
        core = cores[0]
        detail = pd.DataFrame({"cohort": core.cohorts, "n_units": core.cohort_sizes})
        estimate, se = core.estimate, core.se
        model_info["se_neyman"] = core.se_neyman
        model_info["se_adjusted"] = core.se_adjusted
        # 1x1 for a scalar estimand, so `model_info['vcov']` can be read the
        # same way whichever branch produced the result (R's return_full_vcv).
        model_info["vcov"] = np.array([[core.se**2]])
        model_info["vcov_neyman"] = np.array([[core.se_neyman**2]])
        model_info["vcov_adjusted"] = np.array([[core.se_adjusted**2]])
        if fisher:
            model_info["fisher_pvalue"] = core.fisher_pvalue
            model_info["fisher_pvalue_neyman"] = core.fisher_pvalue_neyman
            model_info["fisher_pvalue_adjusted"] = core.fisher_pvalue_adjusted
        if estimand == "eventstudy":
            model_info["event_time"] = float(grid[0])
    else:
        # Rebuild the shared pieces once so the event times get a joint
        # covariance rather than a diagonal approximation.
        panel = _wide_panel(data, i=i, t=t, g=g, y=y)
        s = _summarize(panel)
        weights, betas = [], []
        for e in grid:
            a_theta, a_zero = _weight_matrices(
                "eventstudy",
                s.g_list,
                panel.t_list,
                s.sizes,
                event_time=float(e),
                use_last_treated_only=use_last_treated_only,
                use_did_a0=use_did_a0,
            )
            m = _moments(a_theta, a_zero, s)
            weights.append((a_theta, a_zero))
            betas.append(_beta_star(m) if efficient else np.ones((m.x_var.shape[0], 1)))
        vcv_neyman, vcv_adj = _event_study_vcv(s, weights, betas, panel.t_list)
        vcov = vcv_adj if se_type == "adjusted" else vcv_neyman

        detail = pd.DataFrame(
            {
                "event_time": grid,
                "estimate": [c.estimate for c in cores],
                "se": [c.se for c in cores],
                "se_neyman": [c.se_neyman for c in cores],
                "se_adjusted": [c.se_adjusted for c in cores],
            }
        )
        detail["ci_lower"] = detail["estimate"] - z * detail["se"]
        detail["ci_upper"] = detail["estimate"] + z * detail["se"]
        if fisher:
            detail["fisher_pvalue"] = [c.fisher_pvalue for c in cores]

        # ``.estimate`` summarises the post-adoption event times, matching the
        # convention in sp.event_study; the SE comes from the joint covariance
        # rather than pretending the event times are independent.
        post = np.where(grid >= 0)[0]
        pick = post if post.size else np.arange(grid.size)
        w = np.zeros(grid.size)
        w[pick] = 1.0 / pick.size
        estimate = float(w @ detail["estimate"].to_numpy())
        var = float(w @ vcov @ w)
        se = float(np.sqrt(var)) if var > 0 else 0.0
        # Both aggregate SEs come from the matching joint covariance. Reporting
        # a single event time's SE here would contradict `.estimate`, which is
        # an average across event times.
        var_neyman = float(w @ vcv_neyman @ w)
        var_adjusted = float(w @ vcv_adj @ w)
        model_info["se_neyman"] = float(np.sqrt(max(var_neyman, 0.0)))
        model_info["se_adjusted"] = float(np.sqrt(max(var_adjusted, 0.0)))
        model_info["vcov"] = vcov
        model_info["vcov_neyman"] = vcv_neyman
        model_info["vcov_adjusted"] = vcv_adj
        model_info["event_time"] = grid
        model_info["aggregation"] = (
            "equal-weighted average of the requested "
            f"{'non-negative ' if post.size else ''}event times"
        )
        if fisher:
            # A randomisation p-value is defined per event time, not for the
            # average, so the per-event-time values stay in `.detail` and no
            # scalar is invented here.
            model_info["fisher_pvalue_by_event_time"] = dict(
                zip(grid.tolist(), [c.fisher_pvalue for c in cores])
            )

    zstat = estimate / se if se > 0 else 0.0
    label = f"Roth & Sant'Anna (2023) staggered rollout — {estimand}"
    if estimand == "eventstudy" and is_scalar:
        label += f" (e={grid[0]:g})"
    label += f", {'efficient' if efficient else 'plug-in'}"
    if use_last_treated_only:
        label += ", last-treated controls"

    return CausalResult(
        method=label,
        estimand="ATT (design-based, random adoption timing)",
        estimate=estimate,
        se=se,
        pvalue=float(2 * (1 - _stats.norm.cdf(abs(zstat)))),
        ci=(estimate - z * se, estimate + z * se),
        alpha=alpha,
        n_obs=cores[0].n_units,
        detail=detail,
        model_info=model_info,
        _citation_key="roth2023efficient",
    )


def _drop_treated_at_start(data: pd.DataFrame, t: str, g: str) -> pd.DataFrame:
    """Drop units already treated in the first period.

    ATT(g, t) is not identified for them under parallel trends, so both
    reference wrappers remove them before estimating.
    """
    gv = pd.to_numeric(data[g], errors="coerce")
    g_clean = np.where(gv.isna() | (gv == 0), np.inf, gv.astype(float))
    t_min = float(pd.to_numeric(data[t], errors="coerce").min())
    early = g_clean <= t_min
    if not early.any():
        return data
    warnings.warn(
        "Dropping units treated in the first period or earlier: ATT(g, t) is "
        f"not identified for them ({int(early.sum())} row(s) removed).",
        UserWarning,
        stacklevel=3,
    )
    return data.loc[~early, :]


def staggered_cs(
    data: pd.DataFrame,
    y: str,
    i: str,
    t: str,
    g: str,
    estimand: str = "simple",
    event_time: Union[float, Sequence[float]] = 0.0,
    se_type: str = "neyman",
    fisher: bool = False,
    n_fisher: int = 500,
    random_state: Optional[int] = None,
    alpha: float = 0.05,
) -> "CausalResult":
    """Callaway-Sant'Anna's estimand with **design-based** inference.

    Same weights as ``sp.callaway_santanna`` — every not-yet-treated cohort
    serves as control — but the standard error comes from random adoption
    timing rather than parallel trends. Use it when timing was randomised and
    you want the familiar CS estimand; use ``sp.callaway_santanna`` when it
    was not.

    This is R ``staggered::staggered_cs``: the plug-in weights (``beta = 1``)
    with every not-yet-treated cohort as control, after dropping units already
    treated in the first period.

    Parameters
    ----------
    data, y, i, t, g, estimand, event_time, se_type, fisher, n_fisher
    random_state, alpha
        As in :func:`staggered_rollout`.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=6, staggered=True, seed=7)
    >>> res = sp.staggered_cs(df, y='y', i='unit', t='time', g='first_treat')
    >>> res.se > 0
    True

    References
    ----------
    roth2023efficient
    callaway2021difference
    """
    res = staggered_rollout(
        _drop_treated_at_start(data, t=t, g=g),
        y=y,
        i=i,
        t=t,
        g=g,
        estimand=estimand,
        efficient=False,
        event_time=event_time,
        use_last_treated_only=False,
        se_type=se_type,
        fisher=fisher,
        n_fisher=n_fisher,
        random_state=random_state,
        alpha=alpha,
    )
    res.method = f"Callaway-Sant'Anna estimand, design-based inference — {estimand}"
    res.model_info["estimator"] = "staggered_cs"
    return res


def staggered_sa(
    data: pd.DataFrame,
    y: str,
    i: str,
    t: str,
    g: str,
    estimand: str = "simple",
    event_time: Union[float, Sequence[float]] = 0.0,
    se_type: str = "neyman",
    fisher: bool = False,
    n_fisher: int = 500,
    random_state: Optional[int] = None,
    alpha: float = 0.05,
) -> "CausalResult":
    """Sun-Abraham's estimand with **design-based** inference.

    Identical to :func:`staggered_cs` except that only the *last-treated*
    cohort serves as control, which is what Sun & Abraham's interaction-
    weighted estimator does.

    This is R ``staggered::staggered_sa``.

    Parameters
    ----------
    data, y, i, t, g, estimand, event_time, se_type, fisher, n_fisher
    random_state, alpha
        As in :func:`staggered_rollout`.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=6, staggered=True, seed=7)
    >>> res = sp.staggered_sa(df, y='y', i='unit', t='time', g='first_treat')
    >>> res.se > 0
    True

    References
    ----------
    roth2023efficient
    sun2021estimating
    """
    res = staggered_rollout(
        _drop_treated_at_start(data, t=t, g=g),
        y=y,
        i=i,
        t=t,
        g=g,
        estimand=estimand,
        efficient=False,
        event_time=event_time,
        use_last_treated_only=True,
        se_type=se_type,
        fisher=fisher,
        n_fisher=n_fisher,
        random_state=random_state,
        alpha=alpha,
    )
    res.method = f"Sun-Abraham estimand, design-based inference — {estimand}"
    res.model_info["estimator"] = "staggered_sa"
    return res
