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
optimal controls. Inference is the conservative (Neyman) variance.

.. warning::
   Never-treated units must be coded ``g = inf``, not ``0``. With ``g = 0``
   they are read as a cohort treated before the sample and the estimate is
   badly wrong (on ``did::mpdta``: −0.370 instead of −0.047). This function
   accepts ``0``/``NaN``/``None`` and recodes them, so the trap cannot be
   reached through the public API.

Verified against R ``staggered`` 1.2.2 on canonical ``did::mpdta``: the
``simple`` estimand matches to ~1e-10 for both ``beta`` choices, as does the
conservative standard error. Cross-check on the same data: the package's own
``staggered_sa`` returns −0.0399512752, the Sun-Abraham value independently
pinned against ``fixest::sunab``.

References
----------
Roth, J. and Sant'Anna, P.H.C. (2023). "Efficient Estimation for Staggered
Rollout Designs." *Journal of Political Economy Microeconomics*, 1(4),
669-709. [@roth2023efficient]
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = [
    "StaggeredRolloutResult",
    "staggered_rollout",
    "staggered_rollout_core",
]

_ESTIMANDS = ("simple", "cohort", "calendar")


class StaggeredRolloutResult(NamedTuple):
    """Point estimate, conservative SE, and the fitted control weights."""

    estimate: float
    se: float
    beta: np.ndarray
    estimand: str
    efficient: bool
    cohorts: np.ndarray
    cohort_sizes: np.ndarray
    n_units: int


def _cohort_summaries(data: pd.DataFrame, i: str, t: str, g: str, y: str) -> tuple:
    """Collapse to per-cohort mean paths, covariances and sizes."""
    df = data[[i, t, g, y]].copy()
    gv = pd.to_numeric(df[g], errors="coerce")
    # Never-treated may arrive as 0, NaN or None; all mean "never".
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
    g_of_row = wide.index.get_level_values("_g").to_numpy(dtype=float)
    g_list = np.array(sorted(set(g_of_row)), dtype=float)
    values = wide.to_numpy(dtype=float)

    ybar, cov, sizes = [], [], []
    for gg in g_list:
        block = values[g_of_row == gg]
        if block.shape[0] < 2:
            raise DataInsufficient(
                f"cohort g={gg} has {block.shape[0]} unit(s); the "
                "within-cohort covariance is not estimable.",
                recovery_hint="Merge or drop singleton cohorts.",
                diagnostics={"cohort": float(gg), "n_units": int(block.shape[0])},
            )
        ybar.append(block.mean(axis=0))
        cov.append(np.cov(block, rowvar=False))
        sizes.append(block.shape[0])
    return g_list, t_list, ybar, cov, np.asarray(sizes, dtype=float)


def _cell_weights(
    t_val: float, g_val: float, g_list: np.ndarray, sizes: np.ndarray
) -> np.ndarray:
    """Cohort weights for the ATE(g, t) contrast.

    The treated cohort enters with +1; cohorts not yet treated by
    ``max(t, g)`` share −1 in proportion to their size.
    """
    v = np.zeros(g_list.size)
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
    return v


def _weight_matrices(
    estimand: str, g_list: np.ndarray, t_list: np.ndarray, sizes: np.ndarray
) -> tuple:
    """Build ``A_theta`` and ``A_0`` (both cohorts x periods)."""
    n_g, n_t = g_list.size, t_list.size
    g_max = g_list.max()
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

        contrast = _cell_weights(tt, gg, g_list, sizes)
        a_theta[:, int(np.where(t_list == tt)[0][0])] += w_cell * contrast
        pre = np.where(t_list == gg - 1)[0]
        if pre.size:
            a_zero[:, int(pre[0])] += w_cell * contrast
    return a_theta, a_zero


def staggered_rollout_core(
    data: pd.DataFrame,
    i: str,
    t: str,
    g: str,
    y: str,
    estimand: str = "simple",
    efficient: bool = True,
) -> StaggeredRolloutResult:
    """Roth-Sant'Anna estimator for a randomised staggered rollout."""
    if estimand not in _ESTIMANDS:
        raise MethodIncompatibility(
            f"estimand must be one of {list(_ESTIMANDS)}; got {estimand!r}.",
            recovery_hint="Pass estimand='simple', 'cohort' or 'calendar'.",
            diagnostics={"estimand": estimand},
        )

    g_list, t_list, ybar, cov, sizes = _cohort_summaries(data, i, t, g, y)
    if g_list.size < 2:
        raise DataInsufficient(
            "the staggered-rollout estimator needs at least two treatment "
            "cohorts (including never-treated).",
            recovery_hint="Check the treatment-timing column.",
            diagnostics={"n_cohorts": int(g_list.size)},
        )

    a_theta, a_zero = _weight_matrices(estimand, g_list, t_list, sizes)
    at = [a_theta[k : k + 1, :] for k in range(g_list.size)]
    a0 = [a_zero[k : k + 1, :] for k in range(g_list.size)]

    theta0 = float(sum(float((at[k] @ ybar[k]).item()) for k in range(g_list.size)))
    xhat = sum(a0[k] @ ybar[k] for k in range(g_list.size)).reshape(-1, 1)

    x_var = sum(a0[k] @ cov[k] @ a0[k].T / sizes[k] for k in range(g_list.size))
    x_theta = sum(a0[k] @ cov[k] @ at[k].T / sizes[k] for k in range(g_list.size))
    theta_var = sum(at[k] @ cov[k] @ at[k].T / sizes[k] for k in range(g_list.size))

    if efficient:
        try:
            beta = np.linalg.solve(x_var, x_theta)
        except np.linalg.LinAlgError as exc:
            raise DataInsufficient(
                "the pre-period control moments are collinear, so the "
                "efficient weights are not identified.",
                recovery_hint="Use efficient=False for the plug-in " "estimator.",
                diagnostics={"n_controls": int(x_var.shape[0])},
            ) from exc
    else:
        beta = np.ones((x_var.shape[0], 1))

    estimate = theta0 - float((xhat.T @ beta).item())
    var = float((theta_var + beta.T @ x_var @ beta - 2 * x_theta.T @ beta).item())
    if var < 0:
        # Neyman variance is conservative but can go negative in tiny samples.
        var = 0.0

    return StaggeredRolloutResult(
        estimate=estimate,
        se=float(np.sqrt(var)),
        beta=np.asarray(beta, dtype=float).ravel(),
        estimand=estimand,
        efficient=bool(efficient),
        cohorts=g_list,
        cohort_sizes=sizes,
        n_units=int(sizes.sum()),
    )


def staggered_rollout(
    data: pd.DataFrame,
    y: str,
    i: str,
    t: str,
    g: str,
    estimand: str = "simple",
    efficient: bool = True,
    alpha: float = 0.05,
):
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
    estimand : {'simple', 'cohort', 'calendar'}, default 'simple'
        ``'simple'`` weights each identified cohort-period cell by cohort
        size; ``'cohort'`` averages within a cohort first, then weights
        cohorts by size; ``'calendar'`` averages within a calendar period.
    efficient : bool, default True
        ``True`` uses the optimal pre-period control weights; ``False`` gives
        the plug-in estimator (``beta = 1``), which is the simple
        difference-in-means analogue.
    alpha : float, default 0.05
        Level for the reported confidence interval.

    Returns
    -------
    CausalResult
        ``.estimate`` / ``.se`` carry the ATT and its conservative (Neyman)
        standard error; ``model_info['beta']`` holds the fitted control
        weights.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=6, staggered=True, seed=7)
    >>> res = sp.staggered_rollout(df, y='y', i='unit', t='time',
    ...                            g='first_treat')
    >>> res.se > 0
    True

    References
    ----------
    roth2023efficient
    """
    from scipy import stats as _stats

    from ..core.results import CausalResult

    core = staggered_rollout_core(
        data, i=i, t=t, g=g, y=y, estimand=estimand, efficient=efficient
    )
    z = float(_stats.norm.ppf(1 - alpha / 2))
    zstat = core.estimate / core.se if core.se > 0 else 0.0
    return CausalResult(
        method=(
            "Roth & Sant'Anna (2023) staggered rollout — "
            f"{estimand}, {'efficient' if efficient else 'plug-in'}"
        ),
        estimand="ATT (design-based, random adoption timing)",
        estimate=core.estimate,
        se=core.se,
        pvalue=float(2 * (1 - _stats.norm.cdf(abs(zstat)))),
        ci=(core.estimate - z * core.se, core.estimate + z * core.se),
        alpha=alpha,
        n_obs=core.n_units,
        detail=pd.DataFrame({"cohort": core.cohorts, "n_units": core.cohort_sizes}),
        model_info={
            "estimator": "staggered_rollout",
            "estimand_type": estimand,
            "efficient": core.efficient,
            "beta": core.beta,
            "se_type": "conservative (Neyman)",
            "identifying_assumption": "random adoption timing",
            "n_cohorts": int(core.cohorts.size),
        },
        _citation_key="roth2023efficient",
    )
