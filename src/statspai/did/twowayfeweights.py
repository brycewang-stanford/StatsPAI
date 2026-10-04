"""Weights attached to two-way fixed effects regressions (de Chaisemartin and
D'Haultfoeuille 2020, 2023).

Under parallel trends the coefficient of a fixed effects regression is a
weighted sum of the treatment effects of the treated ``(g, t)`` cells, and
the weights can be negative. This module computes them for

* ``type='feTR'``: ``Y`` on group and period fixed effects and ``D``
  (Theorem 1 of the 2020 paper), with controls and with other treatments in
  the regression (the 2023 paper on several treatments);
* ``type='fdTR'``: the first-difference regression of ``dY`` on period fixed
  effects and ``dD`` (Theorem 2 of the 2020 paper).

Every convention is the one of the authors' Stata command
``twowayfeweights`` (and of R ``TwoWayFEWeights``, which
``statspai.did._twfe_weights`` already follows for the plain ``feTR`` case):

* a treatment, control or other treatment that varies within a cell is
  replaced by its unweighted cell mean;
* weights below ``1e-10`` in absolute value are set to zero before they are
  counted;
* in the first-difference regression the residual of a cell that is not in
  the regression (a group's first period) is zero, and the lead term is
  dropped when the next period of the group is missing;
* ``test_random_weights`` regresses each variable on the ratio ``W`` at the
  cell level, weighted by the cell's share of the treated, with standard
  errors clustered by group.

The numbers are compared with Stata on the four applications of the
authors' textbook in ``tests/reference_parity/test_twowayfeweights_stata.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["twowayfeweights"]

_ZERO_BELOW = 1e-10
# Largest dense fixed-effect design solved exactly; beyond it the projection
# is iterated.
_EXACT_CELLS = 60_000_000


def _fe_residuals(
    V: np.ndarray, fe_codes: Sequence[np.ndarray], w: np.ndarray
) -> np.ndarray:
    """Residuals of the columns of ``V`` on the fixed effects, weighted.

    Exact least squares on the dummy design when it is small enough to
    hold, otherwise alternating projections on the weighted group means
    (which is exact after one sweep on a balanced panel with equal
    weights and converges to 1e-13 elsewhere).
    """
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V[:, None]
    n = V.shape[0]
    levels = [int(c.max()) + 1 for c in fe_codes]
    if len(fe_codes) == 1:
        c = fe_codes[0]
        tot = np.bincount(c, weights=w, minlength=levels[0])
        S = sparse.csr_matrix((w, (c, np.arange(n))), shape=(levels[0], n))
        return np.asarray(V - ((S @ V) / tot[:, None])[c], dtype=float)
    n_cols = sum(levels) - (len(levels) - 1)
    if n * n_cols <= _EXACT_CELLS:
        X = np.zeros((n, n_cols))
        X[np.arange(n), fe_codes[0]] = 1.0
        offset = levels[0]
        for c, k in zip(fe_codes[1:], levels[1:]):
            keep = c > 0
            X[np.where(keep)[0], offset + c[keep] - 1] = 1.0
            offset += k - 1
        sw = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(X * sw, V * sw, rcond=None)
        return np.asarray(V - X @ coef, dtype=float)
    projectors = []
    for c, k in zip(fe_codes, levels):
        tot = np.bincount(c, weights=w, minlength=k)
        S = sparse.csr_matrix((w, (c, np.arange(n))), shape=(k, n))
        projectors.append((c, S, tot))
    R = V.copy()
    scale = np.maximum(np.abs(V).max(axis=0), 1e-300)
    for _ in range(100_000):
        change = 0.0
        for c, S, tot in projectors:
            means = (S @ R) / tot[:, None]
            change = max(change, float((np.abs(means).max(axis=0) / scale).max()))
            R -= means[c]
        if change < 1e-13:
            break
    else:  # pragma: no cover - 100k sweeps is not reached in practice
        raise DataInsufficient(
            "twowayfeweights: the fixed-effect projection did not converge."
        )
    return R


def _order_key(df: pd.DataFrame, group: str, time: str) -> np.ndarray:
    """Rank of each row in (group, time) order, for a stable sort."""
    g = pd.factorize(df[group], sort=True)[0].astype(np.int64)
    t = pd.factorize(df[time], sort=True)[0].astype(np.int64)
    return np.asarray(g * (int(t.max()) + 1) + t)


def _partial_out(
    v: np.ndarray, X: Optional[np.ndarray], w: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Residual of ``v`` on the columns of ``X`` (weighted), and rank of ``X``."""
    if X is None or X.shape[1] == 0:
        return v, 0
    sw = np.sqrt(w)
    coef, _, rank, _ = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)
    return v - X @ coef, int(rank)


def _summarise(weight: np.ndarray) -> Dict[str, Any]:
    plus = weight[weight > 0]
    minus = weight[weight < 0]
    return {
        "n_positive": int(plus.size),
        "n_negative": int(minus.size),
        "sum_positive": float(plus.sum()),
        "sum_negative": float(minus.sum()),
    }


def _sensitivity(
    cells: pd.DataFrame, beta: float, any_negative: bool
) -> Tuple[float, float]:
    """The two summary measures of dCDH (2020), Corollary 1.

    ``sigma_fe``: the smallest standard deviation of the cell effects under
    which ``beta`` and an average effect of zero are compatible.
    ``sigma_fe_2``: the smallest one under which ``beta`` and cell effects
    all of the opposite sign are; undefined without negative weights.
    """
    nat = cells["nat_weight"].to_numpy(dtype=float)
    W = cells["W"].to_numpy(dtype=float)
    m = int(np.sum(nat != 0))
    total = float(nat.sum())
    if m < 2 or not total > 0:
        return float("nan"), float("nan")
    mean = float(np.sum(nat * W) / total)
    sd = float(np.sqrt(np.sum(nat * (W - mean) ** 2) / total * m / (m - 1)))
    first = abs(beta) / sd if sd > 0 else float("nan")
    second = float("nan")
    if any_negative:
        s = cells.loc[cells["weight"] != 0].copy()
        s = s.sort_values(
            ["W", "group", "time"], ascending=[True, False, False], kind="mergesort"
        )
        s["P_k"] = s["nat_weight"].cumsum()
        s["S_k"] = s["weight"].cumsum()
        s["T_k"] = (s["nat_weight"] * s["W"] ** 2).cumsum()
        s = s.sort_values(
            ["W", "group", "time"], ascending=[False, True, True], kind="mergesort"
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            one_minus_p = 1.0 - s["P_k"].to_numpy()
            s_k = s["S_k"].to_numpy()
            sens2 = abs(beta) / np.sqrt(s["T_k"].to_numpy() + s_k**2 / one_minus_p)
            ind = (s["W"].to_numpy() < -s_k / one_minus_p).astype(float)
        ind[0] = 0.0
        ind = np.maximum.accumulate(ind)
        second = float(sens2[len(s) - int(ind.sum())])
    return float(first), second


def _random_weights_test(cells: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    """Regress each variable on ``W`` across cells, weighted by ``nat_weight``.

    Stata: ``regress var W [pweight = nat_weight], cluster(group)``. The
    correlation is the signed square root of that regression's R-squared.
    """
    rows = []
    for name in variables:
        sub = cells.loc[(cells["nat_weight"] > 0) & cells[name].notna()]
        n = len(sub)
        if n < 3:
            rows.append((name, np.nan, np.nan, np.nan, np.nan))
            continue
        pw = sub["nat_weight"].to_numpy(dtype=float)
        x = sub["W"].to_numpy(dtype=float)
        v = sub[name].to_numpy(dtype=float)
        X = np.column_stack([np.ones(n), x])
        XtWX_inv = np.linalg.inv(X.T @ (X * pw[:, None]))
        b = XtWX_inv @ (X.T @ (pw * v))
        resid = v - X @ b
        codes = pd.factorize(sub["group"])[0]
        g = int(codes.max()) + 1
        scores = np.zeros((g, 2))
        np.add.at(scores, codes, X * (pw * resid)[:, None])
        meat = scores.T @ scores
        factor = (g / (g - 1.0)) * ((n - 1.0) / (n - 2.0)) if g > 1 else np.nan
        se = float(np.sqrt(factor * (XtWX_inv @ meat @ XtWX_inv)[1, 1]))
        mean_v = float(np.sum(pw * v) / pw.sum())
        tss = float(np.sum(pw * (v - mean_v) ** 2))
        r2 = 1.0 - float(np.sum(pw * resid**2)) / tss if tss > 0 else np.nan
        corr = (1.0 if b[1] >= 0 else -1.0) * float(np.sqrt(max(r2, 0.0)))
        rows.append((name, float(b[1]), se, float(b[1]) / se, corr))
    return pd.DataFrame(
        rows, columns=["variable", "coef", "se", "t", "correlation"]
    ).set_index("variable")


@accepts_aliases(
    _strict=True,
    controls="covariates",
    weight="weights",
    id="group",
    unit="group",
    treatment="treat",
)
def twowayfeweights(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treat: str,
    *,
    type: str = "feTR",
    treat_level: Optional[str] = None,
    covariates: Optional[Sequence[str]] = None,
    other_treatments: Optional[Sequence[str]] = None,
    test_random_weights: Optional[Sequence[str]] = None,
    weights: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """Weights a two-way fixed effects regression puts on treatment effects.

    Under parallel trends the coefficient of ``treat`` in a regression with
    group and period fixed effects is a weighted sum of the average
    treatment effects of the treated ``(g, t)`` cells. The weights sum to
    one but some can be negative, in which case the coefficient can have
    the opposite sign of every one of those effects. This function returns
    the weights, how many are negative and what they add up to, and the
    two measures of how much effect heterogeneity it takes for the
    coefficient to be misleading. It is the diagnostic to run before
    reading a fixed effects coefficient as an average effect, and it
    applies to any treatment: binary or not, staggered or not.

    Parameters
    ----------
    data : pd.DataFrame
        Long panel, one or several rows per group and period.
    y : str
        Outcome of the regression. With ``type='fdTR'``, the first
        difference of the outcome.
    group, time : str
        Group and period identifiers (the fixed effects of the regression).
    treat : str
        Treatment of the regression. With ``type='fdTR'``, the first
        difference of the treatment.
    type : {'feTR', 'fdTR'}, default 'feTR'
        ``'feTR'``: the regression of ``y`` on group and period fixed
        effects and ``treat``. ``'fdTR'``: the regression of the first
        difference of the outcome on period fixed effects and the first
        difference of the treatment; pass the differences in ``y`` and
        ``treat`` and the treatment itself in ``treat_level``.
    treat_level : str, optional
        The treatment in levels. Required with ``type='fdTR'``, where the
        weights are attached to the cells with a non-zero treatment, not to
        those with a non-zero change.
    covariates : sequence of str, optional
        Control variables of the regression (Stata ``controls()``, accepted
        here as ``controls=`` too). The decomposition then holds under
        parallel trends conditional on a linear model in the controls.
    other_treatments : sequence of str, optional
        Other treatments in the regression (``type='feTR'`` only). The
        coefficient of ``treat`` is then the weighted sum of its own
        effects plus, for each other treatment, a weighted sum of that
        treatment's effects with weights that add up to zero: the
        coefficient is contaminated by the other treatments' effects
        unless these are homogeneous. The result reports both sets.
    test_random_weights : sequence of str, optional
        Variables to regress on the weights. If the weights are
        uncorrelated with the treatment effects the coefficient is still
        the average effect; a variable likely to move with the effects
        (calendar time, length of exposure) that is correlated with the
        weights is evidence against that.
    weights : str, optional
        Observation weights of the regression.
    alpha : float, default 0.05
        Level of the confidence interval of the coefficient.

    Returns
    -------
    CausalResult
        ``estimate`` is the coefficient of ``treat`` and ``se`` its
        standard error clustered by group. ``detail`` has one row per
        ``(group, time)`` cell: ``D`` (the treatment), ``weight``, the
        ratio ``W`` of the weight to the cell's share of the treated, and
        that share ``nat_weight``; with other treatments, a
        ``weight_<name>`` column each. ``model_info`` has ``n_positive``,
        ``n_negative``, ``sum_positive``, ``sum_negative``,
        ``n_treated_cells``, ``sigma_fe`` and ``sigma_fe_2`` (``nan`` when
        no weight is negative; both ``nan`` with other treatments, for
        which the measures are not defined), ``other_treatments`` (the
        same four counts per other treatment) and ``random_weights``
        (``coef``, ``se``, ``t``, ``correlation`` per variable).

    Notes
    -----
    A treatment, control or other treatment that varies within a cell is
    replaced by its cell mean, as in the reference command; the results
    are those of the regression with the cell-level variable.

    The standard error uses the convention of Stata ``xtreg, fe
    vce(cluster group)`` and ``reghdfe``: the group effects are not
    counted in the small-sample factor.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=8, staggered=True, seed=0)
    >>> on = (df["first_treat"] > 0) & (df["time"] >= df["first_treat"])
    >>> df["d"] = on.astype(float)
    >>> w = sp.twowayfeweights(df, y="y", group="unit", time="time", treat="d")
    >>> info = w.model_info
    >>> bool(abs(info["sum_positive"] + info["sum_negative"] - 1) < 1e-8)
    True
    >>> info["n_negative"] > 0  # some treated cells get a negative weight
    True

    References
    ----------
    dechaisemartin2020two, dechaisemartin2023several
    """
    if type not in ("feTR", "fdTR"):
        raise MethodIncompatibility(
            f"twowayfeweights: type={type!r} is not available; use 'feTR' "
            "(fixed effects regression) or 'fdTR' (first-difference "
            "regression). The 'feS' and 'fdS' decompositions of the "
            "reference command are not implemented.",
            diagnostics={"type": type},
        )
    covariates = list(covariates) if covariates else []
    others = list(other_treatments) if other_treatments else []
    random_vars = list(test_random_weights) if test_random_weights else []
    if type == "fdTR":
        if treat_level is None:
            raise MethodIncompatibility(
                "twowayfeweights(type='fdTR') needs treat_level=, the "
                "treatment in levels: y and treat are first differences, and "
                "the weights are attached to the cells with a non-zero "
                "treatment."
            )
        if others:
            raise MethodIncompatibility(
                "twowayfeweights: other_treatments= is only available with "
                "type='feTR'."
            )
    reserved = {"group", "time", "D", "weight", "W", "nat_weight"}
    clash = sorted(reserved.intersection(random_vars) - {group, time})
    if clash:
        raise MethodIncompatibility(
            f"twowayfeweights: test_random_weights variable(s) {clash} have the "
            "name of a column of the returned weights table; rename them.",
            diagnostics={"clash": clash},
        )
    needed = [y, group, time, treat] + covariates + others + random_vars
    needed += [c for c in (treat_level, weights) if c]
    missing = [c for c in dict.fromkeys(needed) if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"twowayfeweights: columns not found in data: {missing}",
            diagnostics={"missing": missing},
        )
    df = data[list(dict.fromkeys(needed))].copy()
    df["_w"] = df[weights].astype(float) if weights else 1.0

    if type == "feTR":
        df = df.dropna(subset=[y, group, time, treat] + covariates + others)
        df = df[df["_w"].notna()]
        in_reg: np.ndarray = np.ones(len(df), dtype=bool)
    else:
        reg_ok = df[[time, y, treat]].notna().all(axis=1)
        df = df[(reg_ok | df[treat_level].notna()) & df["_w"].notna()]
        df = df[df[group].notna() & df[time].notna()]
        reg_ok = df[[time, y, treat]].notna().all(axis=1)
        if covariates:
            df = df[~(reg_ok & df[covariates].isna().any(axis=1))]
            reg_ok = df[[time, y, treat]].notna().all(axis=1)
        in_reg = reg_ok.to_numpy()
    if len(df) == 0:
        raise DataInsufficient("twowayfeweights: no complete observation.")
    in_reg = in_reg[np.argsort(_order_key(df, group, time), kind="mergesort")]
    df = df.sort_values([group, time], kind="mergesort").reset_index(drop=True)

    # group x period level variables: unweighted cell means
    cell_key = [df[group], df[time]]
    for col in [treat] + covariates + others + ([treat_level] if treat_level else []):
        df[col] = df[col].astype(float).groupby(cell_key).transform("mean")

    w = df["_w"].to_numpy(dtype=float)
    gi = pd.factorize(df[group], sort=True)[0]
    ti = pd.factorize(df[time], sort=True)[0]
    n_groups = int(gi.max()) + 1
    obs = float(w.sum())
    level_col = treat if type == "feTR" else treat_level
    D_level = df[level_col].to_numpy(dtype=float)
    ok_level = np.isfinite(D_level)
    mean_D = float(np.sum(w[ok_level] * D_level[ok_level]) / np.sum(w[ok_level]))
    if not np.isfinite(mean_D) or mean_D == 0:
        raise DataInsufficient(
            "twowayfeweights: the treatment is zero in every cell, so there "
            "is no treated cell to put a weight on."
        )

    # residual of the treatment on the fixed effects, controls and other
    # treatments, and the coefficient by Frisch-Waugh-Lovell
    rhs = covariates + others
    r = in_reg
    fes = [gi[r], ti[r]] if type == "feTR" else [pd.factorize(ti[r], sort=True)[0]]
    block = df.loc[r, [treat, y] + rhs].to_numpy(dtype=float)
    res = _fe_residuals(block, fes, w[r])
    eps_r, rank_x = _partial_out(res[:, 0], res[:, 2:] if rhs else None, w[r])
    y_r, _ = _partial_out(res[:, 1], res[:, 2:] if rhs else None, w[r])
    denom = float(np.sum(w[r] * eps_r * eps_r))
    if not denom > 0:
        raise DataInsufficient(
            "twowayfeweights: the treatment has no variation left after the "
            "fixed effects"
            + (" and the other regressors" if rhs else "")
            + ", so the coefficient is not identified."
        )
    beta = float(np.sum(w[r] * eps_r * y_r) / denom)

    # standard error clustered by group (xtreg, fe / reghdfe convention)
    u = y_r - beta * eps_r
    score = np.bincount(gi[r], weights=w[r] * eps_r * u, minlength=n_groups)
    n_reg = int(r.sum())
    n_clusters = int(np.unique(gi[r]).size)
    k = 1 + rank_x + int(np.unique(ti[r]).size)
    if n_clusters > 1 and n_reg > k:
        factor = (n_clusters / (n_clusters - 1.0)) * ((n_reg - 1.0) / (n_reg - k))
        se = float(np.sqrt(factor * float(score @ score)) / denom)
    else:
        se = float("nan")

    eps = np.zeros(len(df))
    eps[r] = eps_r
    df["_eps"] = eps
    df["_P"] = df["_w"].groupby(cell_key).transform("sum") / obs
    first = df.groupby([group, time], sort=True).head(1).copy()
    P = first["_P"].to_numpy(dtype=float)
    D_cell = first[level_col].to_numpy(dtype=float)
    D_cell = np.where(np.isfinite(D_cell), D_cell, 0.0)
    e_cell = first["_eps"].to_numpy(dtype=float)
    nat = P * D_cell / mean_D

    if type == "feTR":
        denom_w = float(np.sum(w * eps * np.where(ok_level, D_level, 0.0)) / obs)
        ratio = e_cell * mean_D / denom_w
    else:
        # w_tilde_gt = eps_gt - eps_{g,t+1} P_{g,t+1} / P_gt, the lead taken
        # only when the group's next period is the next period of the panel
        g_cell = pd.factorize(first[group], sort=True)[0]
        t_cell = pd.factorize(first[time], sort=True)[0]
        nxt = np.r_[np.arange(1, len(first)), -1]
        has_lead = (nxt >= 0) & (g_cell[nxt] == g_cell) & (t_cell[nxt] == t_cell + 1)
        lead = np.where(has_lead, e_cell[nxt] * P[nxt] / P, 0.0)
        w_tilde = e_cell - lead
        denom_w = float(np.sum(P * w_tilde * D_cell) / np.sum(P))
        ratio = w_tilde * mean_D / denom_w
    if not np.isfinite(denom_w) or denom_w == 0:
        raise DataInsufficient(
            "twowayfeweights: the weights are not defined (their "
            "normalisation is zero)."
        )
    weight = ratio * nat
    weight = np.where(np.abs(weight) < _ZERO_BELOW, 0.0, weight)

    cells = pd.DataFrame(
        {
            "group": first[group].to_numpy(),
            "time": first[time].to_numpy(),
            "D": D_cell,
            "weight": weight,
            "W": ratio,
            "nat_weight": nat,
        }
    )
    summary = _summarise(weight)
    other_info: Dict[str, Dict[str, Any]] = {}
    for name in others:
        d_k = first[name].to_numpy(dtype=float)
        w_k = ratio * P * d_k / mean_D
        w_k = np.where(np.abs(w_k) < _ZERO_BELOW, 0.0, w_k)
        cells[f"weight_{name}"] = w_k
        other_info[name] = dict(_summarise(w_k), n_treated_cells=int(np.sum(d_k != 0)))

    if others:
        sigma, sigma2 = float("nan"), float("nan")
    else:
        sigma, sigma2 = _sensitivity(cells, beta, summary["n_negative"] > 0)

    random_table = None
    if random_vars:
        for name in random_vars:
            if name not in cells.columns:
                cells[name] = pd.to_numeric(first[name], errors="coerce").to_numpy()
        random_table = _random_weights_test(cells, random_vars)

    if np.isfinite(se) and se > 0:
        pvalue = float(2 * stats.t.sf(abs(beta / se), n_clusters - 1))
        crit = float(stats.t.ppf(1 - alpha / 2, n_clusters - 1))
        ci = (beta - crit * se, beta + crit * se)
    else:
        pvalue, ci = float("nan"), (float("nan"), float("nan"))

    model_info: Dict[str, Any] = {
        "type": type,
        "beta": beta,
        **summary,
        "n_treated_cells": int(np.sum(nat != 0)),
        "sigma_fe": sigma,
        "sigma_fe_2": sigma2,
        "other_treatments": other_info,
        "random_weights": random_table,
        "covariates": covariates,
        "weights": weights,
        "n_groups": n_groups,
        "n_clusters": n_clusters,
        "se_type": "cluster-robust by group (xtreg, fe / reghdfe convention)",
    }
    return CausalResult(
        method=(
            "Two-way fixed effects weights (de Chaisemartin and "
            f"D'Haultfoeuille 2020), type {type}"
        ),
        estimand=(
            "Coefficient of the "
            + ("fixed effects" if type == "feTR" else "first-difference")
            + " regression: a weighted sum of the treated cells' effects"
        ),
        estimate=beta,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n_reg,
        detail=cells,
        model_info=model_info,
        _citation_key="dechaisemartin2020two",
    )
