"""Inference for difference-in-differences with few treated groups.

The cluster-robust variance of a DiD coefficient is an average over clusters
of the *treated* side's variance as much as the control side's, so with one
or a handful of treated groups it is estimated from one or a handful of
draws and over-rejects badly -- 30 to 50 percent at a nominal 5 percent in
the designs of Conley and Taber (2011) [@conley2011inference] and Ferman and
Pinto (2019) [@ferman2019inference], whatever the total number of clusters.
The multiplier bootstrap and the wild cluster bootstrap inherit the problem;
the cluster jackknife (:func:`statspai.did.cs_jackknife`) mitigates it only
when the treated side itself has several clusters.

Both methods here take the opposite route: they estimate the sampling
distribution of the treatment coefficient from the *control* groups, which
are many, and read the treated groups' contribution off it.

Conley--Taber
    Residualise the outcome and the treatment on group and period fixed
    effects (and covariates), so that ``alphahat = sum_jt dtilde_jt
    ytilde_jt / sum_jt dtilde_jt^2``. Impose the null, and apply the
    *treated* groups' residualised treatment path to each control group's
    residual path. The resulting scalars are draws from the distribution of
    ``alphahat - alpha`` under the null; the test compares ``alphahat``
    with them, and the confidence interval inverts that test. Arbitrary
    serial correlation within a group is allowed -- the path enters as one
    linear combination -- at the price of assuming the group-level errors
    are identically distributed across groups.

Ferman--Pinto
    Relaxes that last assumption in the direction that matters in practice.
    When groups are aggregates of individuals, a group's error variance
    falls with its size, so control groups that are larger than the treated
    one produce too narrow a placebo distribution (and vice versa). They
    model ``Var(W_j) = A + B / M_j`` with ``M_j`` the group's number of
    underlying observations, estimate ``A`` and ``B`` by regressing the
    squared control-group statistics on ``1 / M_j``, and rescale every
    control draw to the treated group's variance before inverting.

Neither method estimates the treatment effect: the point estimate is the
usual two-way fixed-effects coefficient, which is *not* consistent with a
fixed number of treated groups. What they deliver is a test and an interval
for it.

References
----------
[@conley2011inference] Conley and Taber (2011), "Inference with
    'Difference in Differences' with a Small Number of Policy Changes",
    *Review of Economics and Statistics*.
[@ferman2019inference] Ferman and Pinto (2019), "Inference in
    Differences-in-Differences with Few Treated Groups and
    Heteroskedasticity", *Review of Economics and Statistics*.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["did_few_treated"]

_METHODS = ("conley_taber", "ferman_pinto")


def _demean(
    values: np.ndarray, unit_codes: np.ndarray, time_codes: np.ndarray
) -> np.ndarray:
    """Two-way within transformation (alternating projections)."""
    out = np.asarray(values, dtype=float).copy()
    n_u = int(unit_codes.max()) + 1
    n_t = int(time_codes.max()) + 1
    u_count = np.bincount(unit_codes, minlength=n_u).astype(float)
    t_count = np.bincount(time_codes, minlength=n_t).astype(float)
    for _ in range(200):
        prev = out.copy()
        out -= (np.bincount(unit_codes, weights=out, minlength=n_u) / u_count)[
            unit_codes
        ]
        out -= (np.bincount(time_codes, weights=out, minlength=n_t) / t_count)[
            time_codes
        ]
        if np.max(np.abs(out - prev)) < 1e-12:
            break
    return out


def _residualise_covariates(
    y: np.ndarray, d: np.ndarray, X: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    if X is None or X.size == 0:
        return y, d
    coef, *_ = np.linalg.lstsq(X, np.column_stack([y, d]), rcond=None)
    fitted = X @ coef
    return y - fitted[:, 0], d - fitted[:, 1]


@accepts_aliases(_strict=True, unit="id", group="id", controls="covariates")
@_expands_categorical("covariates")
def did_few_treated(
    data: pd.DataFrame,
    y: str,
    id: str,
    time: str,
    treat: str,
    *,
    method: str = "conley_taber",
    covariates: Optional[List[str]] = None,
    group_size: Optional[str] = None,
    alpha: float = 0.05,
    null_value: float = 0.0,
    max_draws: int = 10000,
    seed: Optional[int] = 0,
) -> CausalResult:
    """Conley--Taber / Ferman--Pinto inference with few treated groups.

    .. versionadded:: 1.30.0

    Parameters
    ----------
    data : pandas.DataFrame
        Group-by-period panel (one row per ``unit`` x ``time``). Individual
        level data must be collapsed to group means first; pass the cell
        counts as ``group_size`` so ``method='ferman_pinto'`` can use them.
    y : str
        Outcome column.
    id : str
        Group identifier (the level treatment switches at, e.g. the state).
        ``unit=`` and ``group=`` are accepted aliases.
    time : str
        Period column.
    treat : str
        Treatment indicator, 0/1 by group and period. Groups whose indicator
        is ever 1 are the treated groups; the rest are controls.
    method : {'conley_taber', 'ferman_pinto'}, default 'conley_taber'
        ``'ferman_pinto'`` additionally rescales the control-group draws for
        the heteroskedasticity implied by unequal group sizes; it requires
        ``group_size``.
    covariates : list of str, optional
        Partialled out together with the fixed effects (alias ``controls=``).
    group_size : str, optional
        Number of underlying observations behind each group-period cell
        (``M_j``). Required by ``method='ferman_pinto'``; a group's value is
        averaged over its periods.
    alpha : float, default 0.05
        One minus the confidence level.
    null_value : float, default 0.0
        The null the reported ``pvalue`` tests.
    max_draws : int, default 10000
        With more than one treated group the placebo distribution is built
        from combinations of that many control groups; all of them are used
        when there are at most ``max_draws``, otherwise that many random
        combinations are drawn.
    seed : int or None, default 0
        Seed for the random combinations.

    Returns
    -------
    CausalResult
        ``estimate`` is the two-way fixed-effects coefficient (unchanged by
        the method), ``se`` is the standard deviation of the placebo
        distribution -- reported for scale only, since the interval is not
        ``estimate +/- z * se`` -- ``ci`` is the set of nulls the placebo
        test does not reject at ``alpha``, and ``pvalue`` tests
        ``null_value``.
        ``detail`` holds one row per placebo draw; ``model_info`` records
        the treated groups, the counts, and, for Ferman--Pinto, the fitted
        variance function.

    Notes
    -----
    ``pvalue`` is the two-sided placebo p-value,
    ``(1 + #{|W| >= |alphahat - null|}) / (1 + n_draws)``, and ``ci`` is
    its exact inversion: ``alphahat`` plus or minus the ``m``-th largest
    ``|W|`` with ``m = floor(alpha * (n_draws + 1))``. A null is inside the
    interval if and only if its p-value exceeds ``alpha``, and under the
    exchangeability of group-level errors the method assumes, coverage is
    at least ``1 - alpha`` for any number of control groups. With fewer
    than ``1 / alpha - 1`` draws no bounded interval exists and the
    interval is infinite, with a warning.

    ``model_info['quantile_interval']`` keeps ``alphahat`` minus the
    interpolated ``alpha / 2`` and ``1 - alpha / 2`` quantiles of the
    draws, which was the reported interval before the 2026-10 fix. It is
    asymmetric when the placebo distribution is skewed, and it
    under-covers when control groups are few (88% at a nominal 95% with
    29 control groups, 91% with 60).

    Both methods require many control groups, since the placebo distribution
    is estimated from them; fewer than ten raises.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for j in range(30):
    ...     eff = rng.normal(0, 1)          # group-level shock
    ...     for t in range(8):
    ...         d = 1.0 if (j == 0 and t >= 4) else 0.0
    ...         rows.append({"g": j, "t": t, "d": d,
    ...                      "y": eff + 0.1 * t + 2.0 * d + rng.normal(0, 0.5)})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.did_few_treated(df, y="y", id="g", time="t", treat="d")
    >>> bool(res.ci[0] < res.estimate < res.ci[1])
    True
    """
    if method not in _METHODS:
        raise MethodIncompatibility(
            f"method must be one of {list(_METHODS)}; got {method!r}.",
            recovery_hint="Use method='conley_taber' or 'ferman_pinto'.",
            diagnostics={"method": method},
        )
    if not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility(
            f"alpha must be in (0, 1); got {alpha!r}.",
            recovery_hint="Pass e.g. alpha=0.05.",
            diagnostics={"alpha": alpha},
        )
    covariates = list(covariates or [])
    needed = [y, id, time, treat] + covariates + ([group_size] if group_size else [])
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"columns not found in data: {missing}",
            recovery_hint="Check the column names.",
            diagnostics={"missing": missing},
        )
    if method == "ferman_pinto" and group_size is None:
        raise MethodIncompatibility(
            "method='ferman_pinto' needs group_size=, the number of "
            "observations behind each group-period cell: its correction is "
            "for the heteroskedasticity that unequal group sizes generate.",
            recovery_hint=(
                "Pass the cell counts, or use method='conley_taber' if the "
                "groups are of equal size."
            ),
            diagnostics={"method": method},
        )

    df = data.loc[:, needed].dropna().reset_index(drop=True)
    if df.duplicated([id, time]).any():
        raise MethodIncompatibility(
            "data has more than one row per (unit, time); these methods work "
            "on the group-by-period panel.",
            recovery_hint=(
                "Collapse to group means first and pass the cell counts as "
                "group_size."
            ),
            diagnostics={"n_duplicated": int(df.duplicated([id, time]).sum())},
        )

    unit_codes, unit_levels = pd.factorize(df[id], sort=True)
    time_codes, _ = pd.factorize(df[time], sort=True)
    y_arr = df[y].to_numpy(dtype=float)
    d_arr = df[treat].to_numpy(dtype=float)
    if not np.all(np.isin(np.unique(d_arr), (0.0, 1.0))):
        raise MethodIncompatibility(
            f"{treat!r} must be a 0/1 indicator by group and period.",
            recovery_hint="Recode the treatment indicator.",
            diagnostics={"values": np.unique(d_arr).tolist()[:5]},
        )

    treated_mask_group = (
        pd.Series(d_arr).groupby(unit_codes).max().reindex(range(len(unit_levels)))
    )
    treated_groups = np.flatnonzero(treated_mask_group.to_numpy() > 0)
    control_groups = np.flatnonzero(treated_mask_group.to_numpy() == 0)
    n1, n0 = len(treated_groups), len(control_groups)
    if n1 == 0:
        raise DataInsufficient(
            "no treated group: every group's treatment indicator is zero.",
            recovery_hint="Check the treatment column.",
        )
    if n0 < 10:
        raise DataInsufficient(
            f"only {n0} control groups. Both methods estimate the placebo "
            "distribution from the control groups and need many of them "
            "(the papers' asymptotics are in the number of controls).",
            recovery_hint=(
                "Use sp.wild_cluster_bootstrap or sp.cs_jackknife on a design "
                "with more treated clusters."
            ),
            diagnostics={"n_control_groups": n0, "n_treated_groups": n1},
        )
    if n1 > n0:
        raise DataInsufficient(
            f"{n1} treated groups but only {n0} control groups: the placebo "
            "distribution draws one distinct control group per treated group, "
            "which is impossible here.",
            recovery_hint=(
                "These methods are for a small number of treated groups; with "
                "this many use the cluster-robust or jackknife variance."
            ),
            diagnostics={"n_treated_groups": n1, "n_control_groups": n0},
        )
    if 10 * n1 > n0:
        # The placebo distribution is built from the treated groups' paths
        # alone; the control groups' own contribution to the coefficient is
        # negligible only when the treated groups are a small share. In
        # tests/reliability/few_clusters.py (40 groups, a true effect of
        # zero) the 5% test rejected 2.8% with one treated group, 7.0% with
        # two, 9.7% with five and 13.9% with ten, where the cluster-robust
        # test was back at 5.9%.
        warnings.warn(
            f"did_few_treated: {n1} treated and {n0} control groups. These "
            "methods are for one or two treated groups among many controls; "
            "with more than a tenth as many treated as controls the placebo "
            "test over-rejects (9.7% at 5 of 40, 13.9% at 10 of 40 in "
            "tests/reliability/few_clusters.py). With ten or more treated "
            "groups the cluster-robust variance is preferable.",
            UserWarning,
            stacklevel=2,
        )

    X = df[covariates].to_numpy(dtype=float) if covariates else None
    if X is not None:
        X = np.column_stack(
            [_demean(X[:, k], unit_codes, time_codes) for k in range(X.shape[1])]
        )
    y_t = _demean(y_arr, unit_codes, time_codes)
    d_t = _demean(d_arr, unit_codes, time_codes)
    y_t, d_t = _residualise_covariates(y_t, d_t, X)

    denom = float(d_t @ d_t)
    if denom <= 0:
        raise DataInsufficient(
            "the treatment has no within variation left after the fixed "
            "effects; the coefficient is not identified.",
            recovery_hint="Check that treatment timing varies across groups.",
        )
    estimate = float(d_t @ y_t / denom)

    # Residuals with the null imposed, and each group's residual path.
    resid = y_t - null_value * d_t
    by_group_resid = {j: resid[unit_codes == j] for j in range(len(unit_levels))}
    by_group_time = {j: time_codes[unit_codes == j] for j in range(len(unit_levels))}

    # The treated groups' residualised treatment path, indexed by period.
    n_periods = int(time_codes.max()) + 1
    treated_paths = []
    for j in treated_groups:
        path = np.zeros(n_periods)
        path[by_group_time[j]] = d_t[unit_codes == j]
        treated_paths.append(path)

    def _w(path: np.ndarray, j: int) -> float:
        """Group ``j``'s statistic under the treated path (CT's W)."""
        e = np.zeros(n_periods)
        e[by_group_time[j]] = by_group_resid[j]
        return float(path @ e / denom)

    # One statistic per control group per treated path.
    w_by_path = np.array([[_w(p, j) for j in control_groups] for p in treated_paths])

    scale = np.ones((len(treated_paths), n0))
    var_fit: Dict[str, Any] = {}
    if method == "ferman_pinto":
        sizes = df.groupby(unit_codes)[group_size].mean().to_numpy(dtype=float)
        if np.any(sizes <= 0):
            raise MethodIncompatibility(
                f"{group_size!r} must be positive for every group.",
                recovery_hint="Check the cell counts.",
                diagnostics={"min_group_size": float(np.min(sizes))},
            )
        inv_control = 1.0 / sizes[control_groups]
        for r in range(len(treated_paths)):
            # Var(W_j) = A + B / M_j, fitted on the control draws.
            design = np.column_stack([np.ones(n0), inv_control])
            coef, *_ = np.linalg.lstsq(design, w_by_path[r] ** 2, rcond=None)
            g_control = design @ coef
            g_treated = float(coef[0] + coef[1] / sizes[treated_groups[r]])
            fit_kind = "ols"
            if g_treated <= 0 or np.any(g_control <= 0):
                # W^2 is a one-draw estimate of Var(W_j), so the OLS fit of
                # a variance function can go negative on a finite sample.
                # The function is non-negative by construction; refit under
                # that constraint rather than rescaling by a negative
                # variance.
                from scipy.optimize import nnls

                coef = nnls(design, w_by_path[r] ** 2)[0]
                g_control = design @ coef
                g_treated = float(coef[0] + coef[1] / sizes[treated_groups[r]])
                fit_kind = "nnls"
                warnings.warn(
                    "did_few_treated(method='ferman_pinto'): the unrestricted "
                    "fit of Var(W) = A + B / M was negative somewhere, so it "
                    "was refitted with A, B >= 0. Check group_size for "
                    "outliers, and compare with method='conley_taber'.",
                    UserWarning,
                    stacklevel=2,
                )
            if g_treated <= 0 or np.any(g_control <= 0):
                raise DataInsufficient(
                    "the fitted variance function A + B / M is zero for some "
                    "group even under the non-negativity constraint, so the "
                    "heteroskedasticity correction is undefined here.",
                    recovery_hint=(
                        "Use method='conley_taber', or check group_size for "
                        "outliers."
                    ),
                    diagnostics={
                        "A": float(coef[0]),
                        "B": float(coef[1]),
                        "g_treated": g_treated,
                    },
                )
            scale[r] = np.sqrt(g_treated / g_control)
            var_fit[f"treated_{unit_levels[treated_groups[r]]}"] = {
                "A": float(coef[0]),
                "B": float(coef[1]),
                "var_treated": g_treated,
                "fit": fit_kind,
            }
    w_scaled = w_by_path * scale

    # Placebo draws of (alphahat - alpha): one control group per treated
    # group, without replacement, summed as the estimator sums them.
    rng = np.random.default_rng(seed)
    if n1 == 1:
        draws = w_scaled[0]
        combos: List[Tuple[int, ...]] = [(int(j),) for j in range(n0)]
    else:
        all_combos = list(combinations(range(n0), n1))
        if len(all_combos) > max_draws:
            idx = rng.choice(len(all_combos), size=max_draws, replace=False)
            combos = [all_combos[i] for i in idx]
        else:
            combos = all_combos
        draws = np.array(
            [sum(w_scaled[r, c[r]] for r in range(n1)) for c in combos], dtype=float
        )

    n_draws = int(draws.size)
    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    quantile_ci = (estimate - hi, estimate - lo)
    gap = abs(estimate - float(null_value))
    pvalue = float((1 + np.sum(np.abs(draws) >= gap)) / (1 + n_draws))

    # The interval is the set of nulls the test above does not reject:
    # p(a) > alpha  <=>  at least m of the |W| are >= |alphahat - a|, with
    # m = floor(alpha * (n_draws + 1)), i.e. |alphahat - a| <= the m-th
    # largest |W|. Under the exchangeability the method assumes, this covers
    # with probability at least 1 - alpha for any number of control groups.
    #
    # correctness fix (2026-10): the interval used to be alphahat minus the
    # interpolated alpha/2 and 1 - alpha/2 quantiles of the draws. With 29
    # control groups those sit between the two most extreme order
    # statistics, the interval covered 88% at a nominal 95%, and a null
    # could fall outside it while the reported p-value was above alpha.
    m_reject = int(np.floor(alpha * (n_draws + 1) + 1e-12))
    if m_reject >= 1:
        half_width = float(np.sort(np.abs(draws))[::-1][m_reject - 1])
        ci = (estimate - half_width, estimate + half_width)
    else:
        half_width = float("inf")
        ci = (float("-inf"), float("inf"))
        warnings.warn(
            f"did_few_treated: {n_draws} placebo draws cannot support a "
            f"{100 * (1 - alpha):g}% interval (the smallest attainable "
            f"p-value is 1/{n_draws + 1}); the interval is unbounded. Use a "
            "larger alpha or more control groups.",
            UserWarning,
            stacklevel=2,
        )

    detail = pd.DataFrame(
        {
            "draw": np.arange(n_draws),
            "groups": [
                ", ".join(str(unit_levels[control_groups[i]]) for i in c)
                for c in combos
            ],
            "w": draws,
        }
    )
    model_info = {
        "method": (
            "Conley-Taber (2011)"
            if method == "conley_taber"
            else "Ferman-Pinto (2019), size-corrected"
        ),
        "inference": "placebo distribution from control groups; interval inverted",
        "n_treated_groups": n1,
        "n_control_groups": n0,
        "treated_groups": [unit_levels[j] for j in treated_groups],
        "n_draws": n_draws,
        "null_value": float(null_value),
        "alpha": float(alpha),
        "quantiles": {"lower": lo, "upper": hi},
        "quantile_interval": quantile_ci,
        "interval": "inversion of the two-sided placebo test",
        "half_width": half_width,
        "covariates": covariates,
        "group_size": group_size,
        "variance_function": var_fit or None,
        "citation": (
            "Conley & Taber (2011), REStat"
            if method == "conley_taber"
            else "Ferman & Pinto (2019), REStat"
        ),
    }
    return CausalResult(
        method=f"DiD with few treated groups — {model_info['method']}",
        estimand="ATT (two-way fixed-effects coefficient)",
        estimate=estimate,
        se=float(np.std(draws, ddof=1)),
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=int(len(df)),
        detail=detail,
        model_info=model_info,
        _citation_key=(
            "conley2011inference" if method == "conley_taber" else "ferman2019inference"
        ),
    )
