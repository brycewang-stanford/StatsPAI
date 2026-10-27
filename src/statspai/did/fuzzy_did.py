"""Fuzzy difference-in-differences (Wald-DID, Wald-TC, Wald-CIC).

de Chaisemartin and D'Haultfoeuille (2018) study designs where the
treatment rate rises more in one group than in another but nobody is
switched fully on or off: a "fuzzy" DiD. The usual estimator is the
Wald-DID, the DiD of the outcome over the DiD of the treatment (equal to
the 2SLS coefficient of the treatment instrumented by group x period).
It identifies a local average treatment effect only if the treatment
effect is stable over time and equal in the two groups. Two alternatives
drop that requirement:

* **Wald-TC** ("time corrected"): the counterfactual evolution of the
  switching group is the control group's evolution *among units with the
  same treatment*, ``delta_d = E(Y | G=0, T=1, D=d) - E(Y | G=0, T=0, D=d)``;
* **Wald-CIC** ("changes in changes"): the same idea on whole
  distributions, mapping each period-0 outcome through the control
  group's quantile-quantile transform at its treatment value.

This module follows the definitions in the paper and the conventions of
the authors' Stata command ``fuzzydid`` (de Chaisemartin, D'Haultfoeuille
and Guyonvarch 2019), which it reproduces to machine precision on the
fixtures in ``tests/reference_parity/_fixtures`` (black-box comparison:
the Stata code was not read):

* several groups and periods are coded with two variables, the change of
  the group's treatment between ``t-1`` and ``t`` on the period-``t`` rows
  (``group_change``) and between ``t`` and ``t+1`` on the same rows
  (``group_change_next``), each in {-1, 0, 1};
* each pair of consecutive periods and each direction (increase /
  decrease) gives one comparison against the stable groups; the Wald-DID
  pools them as ``sum_k s_k n_k DID_Y,k / sum_k s_k n_k DID_D,k`` with
  ``n_k`` the number of switching-group rows in the later period and
  ``s_k`` the direction; the Wald-TC and Wald-CIC are the averages of the
  per-comparison estimators with the same weights ``s_k n_k DID_D,k``;
* covariates enter through parametric models of the conditional means
  (OLS, logit or probit), integrated over the switching group's covariate
  distribution in the later period.

Analytic standard errors (``se_method='analytic'``) are the delta-method
(influence-function) variance of the Wald-DID / Wald-TC as smooth
functions of cell means; for the 2x2 Wald-DID this is exactly the HC0
(or CR0) variance of the 2SLS coefficient. ``fuzzydid`` itself reports
bootstrap standard errors only.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core._call_record import records_call
from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core.results import CausalResult
from ..exceptions import (
    ConvergenceFailure,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
    StatsPAIWarning,
)
from ._fuzzy_did_core import _MODELS, _analytic_if, _Design, _estimate

__all__ = ["fuzzy_did", "FuzzyDiDResult"]

_ORDER = ("did", "tc", "cic")
_LABEL = {"did": "W_DID", "tc": "W_TC", "cic": "W_CIC"}
_NUM_LABEL = {"did": "DID_num", "tc": "TC_num", "cic": "CIC_num"}

_BIBTEX = (
    "@article{dechaisemartin2018fuzzy,\n"
    "  title={Fuzzy Differences-in-Differences},\n"
    "  author={de Chaisemartin, C and D'Haultf{\\oe}uille, X},\n"
    "  journal={The Review of Economic Studies},\n"
    "  year={2018},\n"
    "  doi={10.1093/restud/rdx049}\n"
    "}"
)


class FuzzyDiDWarning(StatsPAIWarning):
    """Bootstrap replications that could not be estimated, and similar."""


class FuzzyDiDResult(CausalResult):
    """:class:`CausalResult` of :func:`fuzzy_did`.

    ``estimate`` / ``se`` are those of the first requested estimator
    (order ``did``, ``tc``, ``cic``); ``detail`` has one row per estimator
    and ``model_info['comparisons']`` one row per (period pair, direction)
    comparison with its weight and its own Wald ratios.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.FuzzyDiDResult.__name__
    'FuzzyDiDResult'
    """

    _CITATIONS = {**CausalResult._CITATIONS, "fuzzy_did": _BIBTEX}


# ----------------------------------------------------------------------
# Public function
# ----------------------------------------------------------------------


def _categories(d: np.ndarray, newcateg: Optional[Sequence[float]]):
    if newcateg is None:
        vals, cat = np.unique(d, return_inverse=True)
        return cat.astype(int), len(vals)
    bounds = np.sort(np.asarray(newcateg, dtype=float))
    if np.any(d > bounds[-1]):
        raise MethodIncompatibility(
            "fuzzy_did: newcateg's last upper bound is below the largest "
            "treatment value, so some rows belong to no category.",
            recovery_hint=f"End newcateg with a value >= {float(d.max())}.",
            diagnostics={"newcateg": bounds.tolist(), "max_treat": float(d.max())},
        )
    cat = np.searchsorted(bounds, d, side="left")
    used, cat = np.unique(cat, return_inverse=True)
    return cat.astype(int), len(used)


def _code_change(col: pd.Series, name: str, allowed) -> np.ndarray:
    v = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
    bad = ~np.isnan(v) & ~np.isin(v, allowed)
    if bad.any():
        raise MethodIncompatibility(
            f"fuzzy_did: {name} must take values in {sorted(allowed)} (or be "
            f"missing); found {sorted(set(v[bad].tolist()))[:5]}.",
            recovery_hint="With one group variable it is the 0/1 treatment-group "
            "indicator of a two-period design. With several groups or periods "
            "pass group_change (change into the period: 1 up, -1 down, 0 "
            "stable) and group_change_next (change out of the period).",
            diagnostics={"column": name},
        )
    out = np.where(np.isnan(v), 9, v).astype(int)
    return out


@records_call("fuzzy_did")
@accepts_aliases(_strict=True, controls="covariates", outcome="y", treatment="treat")
@_expands_categorical("covariates")
def fuzzy_did(
    data: pd.DataFrame,
    y: str,
    group_change: str,
    time: str,
    treat: str,
    *,
    group_change_next: Optional[str] = None,
    estimators: Sequence[str] = ("did", "tc", "cic"),
    covariates: Optional[List[str]] = None,
    modelx: Sequence[str] = ("ols", "ols", "ols"),
    newcateg: Optional[Sequence[float]] = None,
    numerator: bool = False,
    se_method: Literal["bootstrap", "analytic", "none"] = "bootstrap",
    n_boot: int = 50,
    cluster: Optional[str] = None,
    eqtest: bool = False,
    cic_quantile: Literal["exact", "stata"] = "exact",
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> FuzzyDiDResult:
    """Fuzzy difference-in-differences: Wald-DID, Wald-TC and Wald-CIC.

    Estimators of the local average treatment effect of de Chaisemartin and
    D'Haultfoeuille (2018) when the treatment rate changes more in some
    groups than in others, with the conventions of Stata ``fuzzydid``
    (``fuzzydid Y G T D`` or ``fuzzydid Y G_T G_T+1 T D``).

    Parameters
    ----------
    data : pd.DataFrame
        One row per individual observation (repeated cross-sections or a
        panel; rows are not linked over time).
    y : str
        Outcome.
    group_change : str
        Stata's ``G``. Without ``group_change_next``: the 0/1 indicator of
        the group whose treatment rate increases, in a design with exactly
        two periods. With ``group_change_next``: on the period-``t`` rows,
        the change of the group's treatment between ``t-1`` and ``t``
        (1 increase, -1 decrease, 0 stable, missing in the first period).
    time : str
        Period. Consecutive sorted values form the comparisons.
    treat : str
        Treatment ``D``: binary or any ordered numeric variable.
    group_change_next : str, optional
        Stata's second group variable ``G_F``: on the period-``t`` rows, the
        change of the group's treatment between ``t`` and ``t+1`` (missing
        in the last period). Required with more than two periods or when
        some groups' treatment decreases.
    estimators : sequence of {'did', 'tc', 'cic'}, default all three
        Wald-DID, Wald-TC (time-corrected) and Wald-CIC (changes in
        changes). Wald-CIC is not available with covariates.
    covariates : list of str, optional
        Covariates (fuzzydid ``continuous()``); ``'C(x)'`` / ``'i.x'`` or a
        categorical column enters as indicators (fuzzydid
        ``qualitative()``). Conditional means are estimated in each
        group x period cell with the models in ``modelx`` and averaged over
        the switching group's covariates in the later period.
    modelx : sequence of {'ols', 'logit', 'probit'}, default ('ols', 'ols', 'ols')
        fuzzydid ``modelx(reg1 reg2 reg3)``: models for E(Y|X), E(D|X) and,
        when D is not binary, P(D = d|X) (one model per treatment
        category). logit / probit need a 0/1 variable.
    newcateg : sequence of float, optional
        Upper bounds of sets of treatment values pooled when conditioning
        on D in the Wald-TC / Wald-CIC (fuzzydid ``newcateg()``): with
        ``[2, 4.5, 8]`` the sets are D <= 2, 2 < D <= 4.5, 4.5 < D <= 8.
    numerator : bool, default False
        Report only the numerators (placebo tests of each estimator's
        assumptions on pre-periods). Requires a single comparison (one
        increasing group, two periods), as fuzzydid.
    se_method : {'bootstrap', 'analytic', 'none'}, default 'bootstrap'
        ``'bootstrap'`` resamples rows (clusters with ``cluster``) and
        reports the bootstrap standard deviation and percentile intervals,
        as fuzzydid. ``'analytic'`` is the influence-function (delta-method)
        variance of the Wald-DID / Wald-TC without covariates, with normal
        intervals; for the 2x2 Wald-DID it equals the HC0 / CR0 variance of
        the 2SLS coefficient (not available for the Wald-CIC or with
        covariates). ``'none'`` is fuzzydid's ``nose``.
    n_boot : int, default 50
        Bootstrap replications (fuzzydid ``breps()``; same default).
    cluster : str, optional
        Block-bootstrap cluster (or CR0 cluster for ``'analytic'``).
    eqtest : bool, default False
        Test the equality of each pair of requested estimators (fuzzydid
        ``eqtest``): differences in ``model_info['eqtest']``.
    cic_quantile : {'exact', 'stata'}, default 'exact'
        How the Wald-CIC picks ``F^{-1}_{01}(u)``, the smallest control
        outcome whose empirical cdf reaches ``u = F_{00}(y)``. ``'exact'``
        decides it in integer arithmetic. ``'stata'`` reproduces fuzzydid,
        which computes ``ceil(u * n)`` in floating point: when ``u * n`` is
        an integer the rounding can move it to the next order statistic
        (always so when the two control cells of a treatment value have the
        same size). The two differ by a few 1e-5 on the fixtures where it
        happens and agree otherwise.
    alpha : float, default 0.05
        Level of the intervals.
    seed : int, optional
        Bootstrap seed.

    Returns
    -------
    FuzzyDiDResult
        A :class:`CausalResult`. ``detail`` has one row per estimator
        (estimate, se, z, p-value, interval); ``model_info['comparisons']``
        lists each (period pair, direction) comparison, its weight and its
        own estimates; ``model_info['bootstrap_failures']`` counts
        replications that could not be estimated (excluded).

    Notes
    -----
    Identification (de Chaisemartin and D'Haultfoeuille 2018): the
    Wald-DID needs the treatment effect to be stable over time and the
    same in both groups; the Wald-TC and Wald-CIC need instead that, in
    the stable group, units with the same treatment follow the same
    outcome trend (in mean for TC, in distribution for CIC) as those of
    the switching group would have without the change. When the stable
    group's treatment rate does not move and effects are homogeneous all
    three agree.

    p-values use the normal distribution of ``estimate / se``; bootstrap
    intervals are percentile intervals, as fuzzydid prints them. The
    bootstrap is not stratified by cell.

    Not implemented (refused by the Stata translator): fuzzydid's
    ``lqte`` (quantile effects), ``partial`` (bounds without a stable
    group) and ``sieves``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 4000
    >>> g = rng.integers(0, 2, n); t = rng.integers(0, 2, n)
    >>> d = rng.binomial(1, 0.2 + 0.1 * t + 0.4 * g * t)
    >>> df = pd.DataFrame({'g': g, 't': t, 'd': d,
    ...                    'y': 1 + 0.5 * t + 2.0 * d + rng.normal(size=n)})
    >>> r = sp.fuzzy_did(df, 'y', 'g', 't', 'd', se_method='analytic',
    ...                  estimators=('did', 'tc'))
    >>> bool(abs(r.estimate - 2.0) < 0.5)
    True

    References
    ----------
    [@dechaisemartin2018fuzzy]; [@dechaisemartin2019fuzzy] (the Stata
    command, multiple groups and periods, covariates).
    """
    want = [k for k in _ORDER if k in {str(e).lower() for e in estimators}]
    unknown = {str(e).lower() for e in estimators} - set(_ORDER)
    if unknown or not want:
        raise MethodIncompatibility(
            f"fuzzy_did: estimators must be among {list(_ORDER)}; got "
            f"{list(estimators)}.",
            recovery_hint="Pass e.g. estimators=('did', 'tc').",
            diagnostics={"estimators": list(estimators)},
        )
    if se_method not in ("bootstrap", "analytic", "none"):
        raise MethodIncompatibility(
            f"fuzzy_did: se_method must be 'bootstrap', 'analytic' or 'none'; "
            f"got {se_method!r}.",
            recovery_hint="Use se_method='bootstrap' (fuzzydid's default).",
            diagnostics={"se_method": se_method},
        )
    if cic_quantile not in ("exact", "stata"):
        raise MethodIncompatibility(
            f"fuzzy_did: cic_quantile must be 'exact' or 'stata'; got "
            f"{cic_quantile!r}.",
            recovery_hint="Use 'exact' (default) or 'stata' to reproduce fuzzydid.",
            diagnostics={"cic_quantile": cic_quantile},
        )
    modelx = tuple(str(m).lower() for m in modelx)
    if len(modelx) == 2:
        modelx = modelx + (modelx[1],)
    if len(modelx) != 3 or any(m not in _MODELS for m in modelx):
        raise MethodIncompatibility(
            f"fuzzy_did: modelx takes two or three of {list(_MODELS)}; got "
            f"{list(modelx)}.",
            recovery_hint="E.g. modelx=('ols', 'logit').",
            diagnostics={"modelx": list(modelx)},
        )
    covs = list(covariates or [])
    if covs and "cic" in want:
        raise MethodIncompatibility(
            "fuzzy_did: the Wald-CIC is not defined with covariates (as in "
            "fuzzydid).",
            recovery_hint="Drop 'cic' from estimators, or drop covariates.",
            diagnostics={"covariates": covs},
        )
    if se_method == "analytic" and ("cic" in want or covs):
        raise MethodIncompatibility(
            "fuzzy_did: analytic standard errors cover the Wald-DID and "
            "Wald-TC without covariates.",
            recovery_hint="Use se_method='bootstrap' for the Wald-CIC and for "
            "estimators with covariates.",
            diagnostics={"estimators": want, "covariates": covs},
        )
    if eqtest and (numerator or se_method == "none" or len(want) < 2):
        raise MethodIncompatibility(
            "fuzzy_did: eqtest needs at least two estimators and standard "
            "errors, and cannot be combined with numerator (as fuzzydid).",
            recovery_hint="Request two or more estimators with se_method "
            "'bootstrap' or 'analytic'.",
        )
    if se_method == "bootstrap" and int(n_boot) < 2:
        raise MethodIncompatibility(
            "fuzzy_did: n_boot must be at least 2.",
            recovery_hint="Use n_boot=50 (fuzzydid's default) or more.",
            diagnostics={"n_boot": n_boot},
        )

    cols = [y, group_change, time, treat] + covs
    if group_change_next is not None:
        cols.append(group_change_next)
    if cluster is not None:
        cols.append(cluster)
    missing_cols = [c for c in cols if c not in data.columns]
    if missing_cols:
        raise MethodIncompatibility(
            f"fuzzy_did: columns not found: {missing_cols}.",
            recovery_hint="Check the column names.",
            diagnostics={"missing": missing_cols},
        )

    df = data
    if group_change_next is None:
        gcol = _code_change(df[group_change], group_change, (0.0, 1.0))
        tvals = df.loc[gcol != 9, time].dropna().unique()
        if len(tvals) != 2:
            raise MethodIncompatibility(
                "fuzzy_did: with one group variable the design must have "
                f"exactly two periods; found {len(tvals)}.",
                recovery_hint="Pass group_change (change into each period) and "
                "group_change_next (change out of it), as fuzzydid's two G "
                "variables for designs with more periods.",
                diagnostics={"n_periods": int(len(tvals))},
            )
        t_sorted = np.sort(tvals)
        tv = df[time].to_numpy()
        gt = np.where(tv == t_sorted[1], gcol, 9)
        gf = np.where(tv == t_sorted[0], gcol, 9)
    else:
        allowed = (-1.0, 0.0, 1.0)
        gt = _code_change(df[group_change], group_change, allowed)
        gf = _code_change(df[group_change_next], group_change_next, allowed)

    yv = pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=float)
    dv = pd.to_numeric(df[treat], errors="coerce").to_numpy(dtype=float)
    tser = df[time]
    ok = ~np.isnan(yv) & ~np.isnan(dv) & tser.notna().to_numpy()
    ok &= (gt != 9) | (gf != 9)
    xmat = None
    if covs:
        xmat = df[covs].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        ok &= ~np.isnan(xmat).any(axis=1)
    cl = None
    if cluster is not None:
        ok &= df[cluster].notna().to_numpy()
    rows_all = np.flatnonzero(ok)
    if len(rows_all) == 0:
        raise DataInsufficient(
            "fuzzy_did: no row has the outcome, treatment, period and a group code.",
            recovery_hint="Check for missing values in the inputs.",
        )
    periods = sorted(pd.unique(tser.to_numpy()[rows_all]))
    tcode_all = np.full(len(df), -1)
    tcode_all[rows_all] = np.searchsorted(
        np.asarray(periods), tser.to_numpy()[rows_all]
    )
    sub = rows_all
    cat, n_cat = _categories(dv[sub], newcateg)
    if cluster is not None:
        cl = pd.factorize(df[cluster].to_numpy()[sub])[0]
    des = _Design(
        y=yv[sub],
        d=dv[sub],
        cat=cat,
        tcode=tcode_all[sub],
        gt=gt[sub],
        gf=gf[sub],
        x=None if xmat is None else xmat[sub],
        cluster=cl,
        periods=list(periods),
        n_cat=n_cat,
    )
    nobs = len(sub)
    rows = np.arange(nobs)

    est, comps, parts = _estimate(des, rows, want, numerator, modelx, cic_quantile)
    if numerator and len(comps) != 1:
        raise MethodIncompatibility(
            "fuzzy_did: numerator needs a single comparison (two periods, one "
            f"increasing group); this design has {len(comps)}.",
            recovery_hint="Restrict the data to one period pair and one "
            "direction of change, or drop numerator.",
            diagnostics={"n_comparisons": len(comps)},
        )
    if numerator and comps[0].sign != 1:
        raise MethodIncompatibility(
            "fuzzy_did: numerator needs the single switching group to be an "
            "increasing one.",
            recovery_hint="Recode the decreasing group as increasing by "
            "reversing time.",
        )

    point = np.array([est[k] for k in want])
    se = np.full(len(want), np.nan)
    ci = np.full((len(want), 2), np.nan)
    draws = None
    failures = 0
    zq = stats.norm.ppf(1 - alpha / 2)

    if se_method == "analytic":
        g = _analytic_if(des, rows, want, numerator)
        if cl is not None:
            g = np.stack(
                [np.bincount(cl, weights=g[:, j]) for j in range(len(want))], 1
            )
        cov = g.T @ g
        se = np.sqrt(np.diag(cov))
        ci = np.column_stack([point - zq * se, point + zq * se])
        draws_cov = cov
    elif se_method == "bootstrap":
        rng = np.random.default_rng(seed)
        out = []
        if cl is not None:
            members = [np.flatnonzero(cl == k) for k in range(cl.max() + 1)]
        for _ in range(int(n_boot)):
            if cl is None:
                idx = rng.integers(0, nobs, nobs)
            else:
                pick = rng.integers(0, len(members), len(members))
                idx = np.concatenate([members[k] for k in pick])
            try:
                e, _, _ = _estimate(des, idx, want, numerator, modelx, cic_quantile)
                out.append([e[k] for k in want])
            except (IdentificationFailure, ConvergenceFailure, ZeroDivisionError):
                failures += 1
                continue
        if failures:
            warnings.warn(
                f"fuzzy_did: {failures} of {n_boot} bootstrap replications "
                "could not be estimated (a cell or treatment value missing in "
                "the resample) and were dropped.",
                FuzzyDiDWarning,
                stacklevel=2,
            )
        draws = np.asarray(out, dtype=float)
        if len(draws) < 2:
            raise DataInsufficient(
                "fuzzy_did: fewer than two bootstrap replications succeeded.",
                recovery_hint="Use se_method='analytic' or pool treatment "
                "values with newcateg.",
                diagnostics={"failures": failures},
            )
        with np.errstate(invalid="ignore"):
            se = draws.std(axis=0, ddof=1)
        ci = np.column_stack(
            [
                np.quantile(draws, alpha / 2, axis=0),
                np.quantile(draws, 1 - alpha / 2, axis=0),
            ]
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = point / se
    pval = 2 * stats.norm.sf(np.abs(zstat))
    labels = [(_NUM_LABEL if numerator else _LABEL)[k] for k in want]
    detail = pd.DataFrame(
        {
            "estimator": labels,
            "estimate": point,
            "se": se,
            "z": zstat,
            "pvalue": pval,
            "ci_lower": ci[:, 0],
            "ci_upper": ci[:, 1],
        }
    )

    comp_rows = []
    pdt = parts.get("did_tc", [{}] * len(comps))
    pc = parts.get("cic", [{}] * len(comps))
    for c, a, b in zip(comps, pdt, pc):
        r = {
            "period_from": des.periods[c.pair],
            "period_to": des.periods[c.pair + 1],
            "direction": c.sign,
            "n_switchers_after": int(
                np.sum((des.tcode == c.pair + 1) & (des.gt == c.sign))
            ),
        }
        for k in ("did_d", "did", "tc"):
            if k in a:
                r[k] = float(np.real(a[k]))
        if "cic" in b:
            r["cic"] = float(b["cic"])
        comp_rows.append(r)

    eq = None
    if eqtest:
        eq_rows = []
        for i, j in combinations(range(len(want)), 2):
            diff = point[i] - point[j]
            if draws is not None:
                dd = draws[:, i] - draws[:, j]
                sd_ = dd.std(ddof=1)
                lo, hi = np.quantile(dd, [alpha / 2, 1 - alpha / 2])
            else:
                sd_ = float(
                    np.sqrt(draws_cov[i, i] + draws_cov[j, j] - 2 * draws_cov[i, j])
                )
                lo, hi = diff - zq * sd_, diff + zq * sd_
            eq_rows.append(
                {
                    "comparison": f"{labels[i]}-{labels[j]}",
                    "difference": diff,
                    "se": sd_,
                    "pvalue": float(2 * stats.norm.sf(abs(diff / sd_))),
                    "ci_lower": lo,
                    "ci_upper": hi,
                }
            )
        eq = pd.DataFrame(eq_rows)

    model_info = {
        "citation_key": "fuzzy_did",
        "estimators": labels,
        "numerator": bool(numerator),
        "design": "two_g" if group_change_next is not None else "2x2",
        "periods": des.periods,
        "comparisons": pd.DataFrame(comp_rows),
        "covariates": covs,
        "modelx": list(modelx) if covs else None,
        "newcateg": None if newcateg is None else list(newcateg),
        "n_treat_categories": n_cat,
        "se_method": se_method,
        "n_boot": int(n_boot) if se_method == "bootstrap" else None,
        "bootstrap_failures": failures if se_method == "bootstrap" else None,
        "cluster": cluster,
        "eqtest": eq,
        "ci_type": {"bootstrap": "percentile", "analytic": "normal"}.get(se_method),
        "cic_quantile": cic_quantile if "cic" in want else None,
    }
    if draws is not None:
        model_info["bootstrap_draws"] = pd.DataFrame(draws, columns=labels)

    first = 0
    res = FuzzyDiDResult(
        method=f"Fuzzy DiD ({labels[first]})",
        estimand="LATE",
        estimate=float(point[first]),
        se=float(se[first]),
        pvalue=float(pval[first]),
        ci=(float(ci[first, 0]), float(ci[first, 1])),
        alpha=float(alpha),
        n_obs=int(nobs),
        detail=detail,
        model_info=model_info,
        _citation_key="fuzzy_did",
    )
    return res
