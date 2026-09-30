"""Bunching summaries of a bin-by-event-time minimum-wage event study.

Cengiz, Dube, Lindner and Zipperer (2019) [@cengiz2019effect] regress jobs
per capita in each wage bin on indicators ``I^{tau k}``: ``tau`` years since
a minimum-wage increase and ``k`` dollars from the new minimum. The
coefficients are summarised as missing jobs below the new minimum
(``Delta b``), excess jobs at and just above it (``Delta a``), the
percentage change in affected employment and in affected workers' wage,
and two elasticities. ``sp.cdlz_bunching`` computes those summaries -- with
analytic delta-method standard errors -- from any fitted regression whose
coefficient names are listed in ``terms``; the regression itself (fixed
effects, controls, clean-control stacking) is the user's, typically
``sp.hdfe_ols``.

With ``E`` the pre-period employment-to-population ratio, ``B`` the share of
jobs below the new minimum, ``EWB`` the pre-period wage bill per capita,
``%dMW`` the mean percentage increase and ``MW`` the mean new minimum (in
the bin units), ``s = bins_per_unit / n_post`` and ``P`` the post terms::

    Delta b   = s * sum_{P, k < 0} alpha / E
    Delta a   = s * sum_{P, k >= 0} alpha / E
    %De       = (Delta a + Delta b) / B
    e_MW      = (Delta a + Delta b) / %dMW
    %Dwb      = s * sum_P (MW + k) alpha / EWB
    %Dw       = (%Dwb - %De) / (1 + %De)
    e_w       = %De / %Dw

``%Dw`` subtracts the employment change from the wage-bill change and
divides by ``1 + %De``, the paper's equation for the wage of affected
workers.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility


def _params_vcov(result: Any) -> "tuple[pd.Series, pd.DataFrame]":
    params = pd.Series(result.params).astype(float)
    if hasattr(result, "cov_params"):
        V = result.cov_params()
    else:
        V = result.vcov() if callable(getattr(result, "vcov", None)) else result.vcov
    if not isinstance(V, pd.DataFrame):
        V = pd.DataFrame(np.asarray(V), index=params.index, columns=params.index)
    return params, V


def cdlz_bunching(
    result: Any = None,
    *,
    terms: pd.DataFrame,
    epop: float,
    below_share: float,
    wage_bill: float,
    pct_mw: float,
    mw_level: float,
    post_years: Optional[Sequence[int]] = None,
    bins_per_unit: float = 4.0,
    params: Optional[pd.Series] = None,
    covariance: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """Missing / excess jobs, affected employment and wage, and elasticities.

    Parameters
    ----------
    result : fitted regression, optional
        Any result with ``params`` and a coefficient covariance
        (``sp.hdfe_ols``, ``sp.feols``, ``sp.regress``). Alternatively pass
        ``params`` and ``covariance``.
    terms : pd.DataFrame
        One row per treatment coefficient: ``term`` (the coefficient name),
        ``year`` (event time ``tau``, e.g. 0..4 after and -3..-2 before the
        increase, -1 being the normalised reference) and ``bin`` (``k``:
        -4..-1 below the new minimum, 0..4 at and above it, in the same
        units as ``mw_level``).
    epop : float
        ``E``: pre-period employment per capita (the normaliser).
    below_share : float
        ``B``: share of jobs below the new minimum before the increase.
    wage_bill : float
        ``EWB``: pre-period wage bill per capita.
    pct_mw : float
        Mean percentage increase of the minimum wage (0.101 = 10.1%).
    mw_level : float
        Mean new minimum, in the units of ``bin`` (dollars in CDLZ).
    post_years : sequence of int, optional
        Event years averaged in the summaries; default every ``year >= 0``.
    bins_per_unit : float, default 4
        Fine bins per unit of ``bin`` whose coefficient each term carries
        (CDLZ: a $1 indicator spans four $0.25 bins).
    params, covariance : optional
        Coefficients and their covariance matrix when ``result`` is not given.
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        Headline: the percentage change in affected employment ``%De``.
        ``model_info['summary']`` has every statistic with its SE, z-based
        interval and p-value; ``['event_path']`` the excess / missing jobs
        by event year (``s`` without the averaging); ``['bin_profile']`` the
        post-period change by bin and its running sum.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> terms = pd.DataFrame(
    ...     [(f"t{y}_{k}", y, k) for y in range(5) for k in range(-4, 5)],
    ...     columns=["term", "year", "bin"],
    ... )
    >>> # jobs move from just below the new minimum to just above it
    >>> b = pd.Series(np.where(terms["bin"] == -1, -0.002,
    ...               np.where(terms["bin"] == 0, 0.0021, 0.0)),
    ...               index=terms["term"])
    >>> V = pd.DataFrame(np.eye(len(b)) * 1e-8, index=b.index, columns=b.index)
    >>> r = sp.cdlz_bunching(terms=terms, params=b, covariance=V, epop=0.57,
    ...                      below_share=0.086, wage_bill=0.35, pct_mw=0.10,
    ...                      mw_level=8.8)
    >>> s = r.model_info["summary"]
    >>> bool(s.loc["missing_jobs_below", "estimate"] < 0
    ...      < s.loc["excess_jobs_above", "estimate"])
    True

    References
    ----------
    [@cengiz2019effect]
    """
    if result is not None:
        b_all, V_all = _params_vcov(result)
    elif params is not None and covariance is not None:
        b_all = pd.Series(params).astype(float)
        V_all = pd.DataFrame(covariance, index=b_all.index, columns=b_all.index)
    else:
        raise MethodIncompatibility(
            "cdlz_bunching: pass a fitted result, or params= and covariance=."
        )
    need = {"term", "year", "bin"}
    if not need <= set(terms.columns):
        raise MethodIncompatibility(
            f"cdlz_bunching: terms needs columns {sorted(need)}."
        )
    T = terms.copy()
    T["term"] = T["term"].astype(str)
    missing = [t for t in T["term"] if t not in b_all.index]
    if missing:
        raise MethodIncompatibility(
            f"cdlz_bunching: terms not in the regression: {missing[:5]}"
        )
    for name, v in dict(
        epop=epop, below_share=below_share, wage_bill=wage_bill, pct_mw=pct_mw
    ).items():
        if not np.isfinite(v) or v == 0:
            raise MethodIncompatibility(
                f"cdlz_bunching: {name} must be finite and nonzero."
            )
    years = sorted(T["year"].unique())
    post = (
        sorted(post_years) if post_years is not None else [y for y in years if y >= 0]
    )
    P = T[T["year"].isin(post)]
    if P.empty:
        raise MethodIncompatibility("cdlz_bunching: no post-period terms.")
    names = list(P["term"])
    b = b_all.loc[names].to_numpy()
    V = V_all.loc[names, names].to_numpy()
    k = P["bin"].to_numpy(dtype=float)
    s = bins_per_unit / len(post)
    below = (k < 0).astype(float)
    above = (k >= 0).astype(float)
    wage = mw_level + k
    E, B, EWB = float(epop), float(below_share), float(wage_bill)

    # Linear pieces: value = g'b, gradient g.
    g_b = s * below / E
    g_a = s * above / E
    g_e = (g_b + g_a) / B
    g_elas = (g_b + g_a) / float(pct_mw)
    g_wb = s * wage / EWB
    de = float(g_e @ b)
    dwb = float(g_wb @ b)
    dw = (dwb - de) / (1.0 + de)
    # Nonlinear pieces: chain rule.
    g_dw = (g_wb - g_e) / (1.0 + de) - (dwb - de) / (1.0 + de) ** 2 * g_e
    ew = de / dw
    g_ew = g_e / dw - de / dw**2 * g_dw

    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for label, est, g in (
        ("missing_jobs_below", float(g_b @ b), g_b),
        ("excess_jobs_above", float(g_a @ b), g_a),
        ("pct_change_affected_wage", dw, g_dw),
        ("pct_change_affected_employment", de, g_e),
        ("elasticity_wrt_mw", float(g_elas @ b), g_elas),
        ("elasticity_wrt_affected_wage", ew, g_ew),
    ):
        se = float(np.sqrt(max(g @ V @ g, 0.0)))
        rows.append(
            dict(
                statistic=label,
                estimate=est,
                se=se,
                ci_lower=est - z * se,
                ci_upper=est + z * se,
                pvalue=float(2 * stats.norm.sf(abs(est / se))) if se > 0 else np.nan,
            )
        )
    summary = pd.DataFrame(rows).set_index("statistic")

    path_rows = []
    for yr in years:
        Ty = T[T["year"] == yr]
        for side, mask in (("excess", Ty["bin"] >= 0), ("missing", Ty["bin"] < 0)):
            nm = list(Ty.loc[mask, "term"])
            if not nm:
                continue
            g = np.full(len(nm), bins_per_unit / E)
            est = float(g @ b_all.loc[nm].to_numpy())
            se = float(np.sqrt(g @ V_all.loc[nm, nm].to_numpy() @ g))
            path_rows.append(dict(year=yr, side=side, estimate=est, se=se))
    event_path = pd.DataFrame(path_rows)

    prof = []
    for kk in sorted(P["bin"].unique()):
        nm = list(P.loc[P["bin"] == kk, "term"])
        g = np.full(len(nm), s / E)
        est = float(g @ b_all.loc[nm].to_numpy())
        se = float(np.sqrt(g @ V_all.loc[nm, nm].to_numpy() @ g))
        prof.append(dict(bin=kk, estimate=est, se=se))
    bin_profile = pd.DataFrame(prof)
    bin_profile["running_sum"] = bin_profile["estimate"].cumsum()

    head = summary.loc["pct_change_affected_employment"]
    return CausalResult(
        method="Minimum-wage bunching estimator (Cengiz et al. 2019)",
        estimand="% change in affected employment",
        estimate=float(head["estimate"]),
        se=float(head["se"]),
        pvalue=float(head["pvalue"]),
        ci=(float(head["ci_lower"]), float(head["ci_upper"])),
        alpha=alpha,
        n_obs=int(getattr(result, "nobs", 0) or 0) if result is not None else 0,
        detail=summary.reset_index(),
        model_info=dict(
            summary=summary,
            event_path=event_path,
            bin_profile=bin_profile,
            post_years=post,
            constants=dict(
                epop=E,
                below_share=B,
                wage_bill=EWB,
                pct_mw=float(pct_mw),
                mw_level=float(mw_level),
                bins_per_unit=float(bins_per_unit),
            ),
        ),
        _citation_key="cdlz_bunching",
    )


# Mirrors paper.bib ``cengiz2019effect``.
CausalResult._CITATIONS["cdlz_bunching"] = (
    "@article{cengiz2019effect,\n"
    "  title={The Effect of Minimum Wages on Low-Wage Jobs*},\n"
    "  author={Cengiz, Doruk and Dube, Arindrajit and Lindner, Attila and "
    "Zipperer, Ben},\n"
    "  journal={The Quarterly Journal of Economics},\n"
    "  volume={134},\n"
    "  number={3},\n"
    "  pages={1405--1454},\n"
    "  year={2019},\n"
    "  doi={10.1093/qje/qjz014}\n"
    "}"
)
