"""
Arellano-Bond (1991) and Blundell-Bond (1998) dynamic panel GMM.

Estimates dynamic panel models of the form:

    Y_{it} = ρ Y_{i,t-1} + X_{it}'β + α_i + ε_{it}

using GMM with lagged levels (Arellano-Bond) or lagged levels and
differences (Blundell-Bond system GMM) as instruments.

The Arellano-Bond first-differenced estimator removes the fixed effect
α_i by first-differencing and then exploits the **block-diagonal**
GMM moment conditions E[Y_{i,s} ΔU_{it}] = 0 for s ≤ t-2: every lagged
level that is available for period t is a separate instrument for that
period's differenced equation. The one-step weight matrix is
(Σ_i Z_i' H Z_i)⁻¹ where H encodes the MA(1) structure of the
first-differenced i.i.d. errors (2 on the diagonal, -1 on the first
off-diagonals). This matches Stata's ``xtabond`` and ``xtdpd``.

All standard errors, the Sargan/Hansen over-identification tests, and the
Arellano-Bond AR(1)/AR(2) serial-correlation tests are validated to
machine precision against Stata 18 ``xtabond ..., noconstant`` (one-step
robust and non-robust, two-step conventional, and two-step
Windmeijer-corrected) on the canonical ``abdata`` employment panel — see
``tests/reference_parity/test_dynpanel_abdata_parity.py``.

The numerics live in :mod:`statspai.gmm._dynpanel`; this module is the
documented user-facing surface.

References
----------
Arellano, M. and Bond, S. (1991).
"Some Tests of Specification for Panel Data: Monte Carlo Evidence
and an Application to Employment Equations."
*Review of Economic Studies*, 58(2), 277-297. [@arellano1991some]

Blundell, R. and Bond, S. (1998).
"Initial Conditions and Moment Restrictions in Dynamic Panel Data
Models."
*Journal of Econometrics*, 87(1), 115-143. [@blundell1998initial]

Roodman, D. (2009).
"How to Do xtabond2: An Introduction to Difference and System GMM
in Stata."
*Stata Journal*, 9(1), 86-136. [@roodman2009xtabond]

Windmeijer, F. (2005).
"A Finite Sample Correction for the Variance of Linear Efficient
Two-Step GMM Estimators."
*Journal of Econometrics*, 126(1), 25-51. [@windmeijer2005finite]
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ._dynpanel import fit_dynamic_panel
from ._dynpanel._moments import first_difference_H as _ab_H  # noqa: F401  (re-export)

__all__ = ["xtabond", "xtdpdsys"]


def xtabond(
    data: pd.DataFrame,
    y: str,
    x: Optional[Sequence[str]] = None,
    id: str = "id",
    time: str = "time",
    lags: int = 1,
    gmm_lags: Tuple[int, Optional[int]] = (2, None),
    method: str = "difference",
    twostep: bool = False,
    robust: bool = True,
    alpha: float = 0.05,
    predetermined: Optional[Sequence[str]] = None,
    endogenous: Optional[Sequence[str]] = None,
    predetermined_lags: Optional[Tuple[Optional[int], Optional[int]]] = None,
    endogenous_lags: Optional[Tuple[Optional[int], Optional[int]]] = None,
    collapse: bool = False,
    time_dummies: bool = False,
    orthogonal: bool = False,
    constant: Optional[bool] = None,
) -> CausalResult:
    """
    Arellano-Bond / Blundell-Bond dynamic panel GMM estimator.

    Equivalent to Stata's ``xtabond`` / ``xtabond2``.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced or unbalanced panel in long format.
    y : str
        Dependent variable.
    x : list of str, optional
        Strictly exogenous regressors. Entered in first differences both
        as regressors and as their own (standard) instruments. Accepts
        Stata lag-operator syntax — ``"w"``, ``"L.w"``, ``"L2.k"``,
        ``"L(0/2).k"`` — so the canonical Arellano-Bond (1991) employment
        equation is a single call:
        ``x=["l(0/1).w", "l(0/2).k"], lags=2``.
    id : str, default 'id'
        Unit identifier.
    time : str, default 'time'
        Time period variable. Treated as an **ordinal** sequence: the
        sorted distinct values define consecutive periods, so a missing
        period (gap) is recognised, but non-integer / irregularly-spaced
        codes are collapsed to their rank order.
    lags : int, default 1
        Number of lags of Y to include (ρ₁ Y_{t-1} + ... + ρ_p Y_{t-p}).
    gmm_lags : tuple (min, max), default (2, None)
        Range of lags of Y (in levels) used as GMM instruments. ``min``
        must be ≥ 2 (deeper lags are orthogonal to the differenced error).
        ``max=None`` uses **all** available deeper lags, matching Stata's
        ``xtabond`` default. Setting ``max`` caps the instrument count
        (Stata's ``maxldep()``).
    method : str, default 'difference'
        ``'difference'`` — Arellano-Bond first-differenced GMM; parity with
        Stata's ``xtabond ..., noconstant``.

        ``'system'`` — Blundell-Bond (1998) system GMM: the level equation
        is stacked alongside the transformed one and instrumented with
        lagged *differences* (E[Δy_{i,t-1}(α_i + ε_{it})] = 0), and an
        intercept becomes identified. Prefer it when the series is
        persistent: as ρ approaches 1 the lagged levels become weak
        instruments for the differences and difference GMM is badly biased
        (on ``abdata`` it returns ρ̂ = 1.02; system GMM returns 0.69).
        Parity with ``xtabond2 ... robust`` — one-step, two-step
        Windmeijer, and collapsed — to machine precision.

        The extra level moments are an *additional* assumption (the
        deviations of the initial conditions from the long-run mean must be
        uncorrelated with α_i). Report the difference-in-Hansen test for
        the level instruments before relying on them.
    twostep : bool, default False
        Use two-step GMM with the efficient weight matrix. When
        ``robust=True`` the Windmeijer (2005) finite-sample correction is
        applied to the two-step standard errors; with ``robust=False`` the
        conventional (downward-biased) two-step SEs are returned and a
        warning is issued.
    robust : bool, default True
        Heteroskedasticity-robust standard errors (Windmeijer-corrected
        for two-step). When ``False``, the classical one/two-step VCE.
    alpha : float, default 0.05
        Significance level.
    predetermined : list of str, optional
        Regressors that are *predetermined* (weakly exogenous):
        E[x_{is} ε_{it}] = 0 for s ≤ t but not for s > t. Their own lags
        from ``predetermined_lags`` (default 1 and deeper) enter as
        block-diagonal GMM instruments rather than as a single Δx column.
        Accepts the same lag-operator syntax as ``x``. Equivalent to
        Stata's ``xtabond ..., pre()`` / ``xtabond2 ... gmm(x, lag(1 .))``.
    endogenous : list of str, optional
        Regressors correlated with the *contemporaneous* error. Their lags
        from ``endogenous_lags`` (default 2 and deeper) are used as GMM
        instruments. Equivalent to ``xtabond ..., endogenous()`` /
        ``xtabond2 ... gmm(x, lag(2 .))``.
    predetermined_lags, endogenous_lags : tuple (min, max), optional
        Instrument lag windows for the two classes above. ``None`` on
        either element means "class default" for the minimum and "all
        available deeper lags" for the maximum.

        Lags are **absolute**, i.e. counted from the equation period, which
        is ``xtabond2``'s ``gmm(x, lag(a b))`` convention and the one the
        modern literature states. Stata's older ``xtabond`` counts *further*
        lags beyond the deepest lag of the variable that appears as a
        regressor, so ``xtabond ..., pre(w, lagstruct(p, .))`` maps to
        ``predetermined=['l(0/p).w'], predetermined_lags=(p + 1, None)`` and
        ``endogenous(w, lagstruct(p, .))`` to
        ``endogenous_lags=(p + 2, None)``. Both mappings are pinned by
        reference-parity tests.
    collapse : bool, default False
        Collapse the block-diagonal instrument sets — one column per lag
        *distance* instead of one per (period, distance) pair. This is
        Roodman's (2009) remedy for instrument proliferation: the
        uncollapsed count grows as O(T²), which overfits the endogenous
        regressor, biases the estimate toward the within estimator, and
        pushes the Hansen p-value toward an uninformative 1.0. Matches
        ``xtabond2, collapse``.
    orthogonal : bool, default False
        Use Arellano-Bover (1995) forward orthogonal deviations instead of first
        differences for the transformed equation: subtract from each
        observation the mean of its *available future* ones, scaled so the
        transformed errors stay serially uncorrelated. Prefer it on gappy
        panels — first differencing destroys the equations on both sides of
        a hole, forward deviations only the one at the hole. Matches
        ``xtabond2, orthogonal`` (difference and system, one- and two-step).
    time_dummies : bool, default False
        Add period dummies (first period dropped) as regressors *and* as
        their own standard instruments. Roodman (2009) recommends these as
        a default: they absorb common shocks, which is what makes the
        no-cross-sectional-dependence assumption behind the moment
        conditions plausible.
    constant : bool, optional
        Include an intercept. Defaults to ``True`` for ``method='system'``
        (where the level equation identifies it, as in ``xtabond2``) and
        ``False`` for ``method='difference'`` (where it differences away);
        requesting it for difference GMM raises ``NotImplementedError``.

    Returns
    -------
    CausalResult
        ``estimate`` / ``se`` are the lagged-Y (ρ₁) coefficient. ``detail``
        carries the per-coefficient table (lagged Y first, then exogenous
        regressors, then predetermined, endogenous, time dummies and — for
        system GMM — ``_cons``).

        ``model_info`` holds the diagnostics: ``n_obs`` (transformed-equation
        rows for difference GMM, level-equation rows for system GMM — the
        counts ``xtabond`` and ``xtabond2`` respectively print, with
        ``n_obs_diff`` / ``n_obs_level`` / ``n_obs_total`` always available),
        ``n_instruments``, the AR(1)/AR(2) Arellano-Bond statistics, the
        Sargan test (valid under homoskedasticity) and the Hansen J
        (heteroskedasticity-robust, reported for one-step fits too).

        The Sargan statistic uses ``xtabond``'s scale
        ``σ̂² = ê*'ê* / (2 (N* − k))`` over the *transformed* rows only
        (level residuals still contain α_i and carry no information about
        σ²). ``xtabond2`` divides by ``2 N*`` instead, so its Sargan sits a
        factor ``N*/(N* − k)`` higher; the Hansen J, which has no such free
        scale, matches ``xtabond2`` exactly.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for firm in range(40):  # build a dynamic panel
    ...     alpha = rng.normal(0, 1)
    ...     y_prev = rng.normal(0, 1)
    ...     for t in range(8):
    ...         capital = rng.normal(0, 1)
    ...         labor = rng.normal(0, 1)
    ...         y = (0.5 * y_prev + 0.3 * capital + 0.2 * labor
    ...              + alpha + rng.normal(0, 1))
    ...         rows.append({'firm': firm, 'year': 2000 + t, 'output': y,
    ...                      'capital': capital, 'labor': labor})
    ...         y_prev = y
    >>> df = pd.DataFrame(rows)
    >>> # Arellano-Bond (difference GMM)
    >>> result = sp.xtabond(df, y='output', x=['capital', 'labor'],
    ...                     id='firm', time='year')
    >>> print(result.summary())  # doctest: +SKIP

    >>> # Two-step with Windmeijer-corrected SEs
    >>> result = sp.xtabond(df, y='output', x=['capital', 'labor'],
    ...                     id='firm', time='year', twostep=True)

    >>> # Blundell-Bond system GMM (use when the series is persistent)
    >>> result = sp.xtabond(df, y='output', x=['capital', 'labor'],
    ...                     id='firm', time='year', method='system',
    ...                     twostep=True, collapse=True)

    >>> # Lag operators, a predetermined regressor, and collapsed instruments
    >>> result = sp.xtabond(df, y='output', x=['l(0/1).capital'],
    ...                     predetermined=['labor'], id='firm', time='year',
    ...                     collapse=True)

    Notes
    -----
    **Arellano-Bond (1991)**: First-differences the equation to remove
    fixed effects α_i, then uses lagged levels Y_{i,t-2}, Y_{i,t-3}, ...
    as a block-diagonal set of GMM instruments for ΔY_{i,t-1}.

    No constant / time trend is included (unlike Stata's *default*
    ``xtabond``, which adds a ``_cons`` via a level moment). This matches
    Stata's ``xtabond ..., noconstant``; the reported ρ / β coefficients
    are identical to Stata's ``_cons`` run when the series has no drift.

    **Missing values are handled per variable, not per row.** A covariate
    that is unobserved at period *t* costs only the equations that need it;
    ``y_{i,t}`` remains available as a GMM instrument and as a lag source.
    (Before v1.21 a listwise ``dropna`` deleted the whole row, which
    amputated the instrument set whenever a lagged regressor was used — on
    ``abdata`` that moved ρ̂ from 0.849 to 0.660. See ``MIGRATION.md``.)

    **Balanced vs gapped panels.** All standard errors, the Sargan/Hansen
    tests, and the one-step AR(1)/AR(2) tests are validated to machine
    precision against Stata for balanced (and ragged-but-gap-free) panels.
    When a unit is missing an *interior* period, the estimator stays
    consistent but its finite-sample numbers can differ from Stata's
    ``xtabond`` by ~1% (Stata, ``xtabond2``, and R's ``plm`` each use a
    slightly different gap-weighting convention); a warning is emitted in
    that case.

    Key diagnostics:
    - **AR(1) test**: Should reject (expected in first differences).
    - **AR(2) test**: Should NOT reject (validates instrument exogeneity).
    - **Sargan / Hansen test**: Should NOT reject (overidentification).
      Sargan (one-step) is not robust to heteroskedasticity; prefer the
      two-step Hansen J when that is a concern.
    - **Instrument count**: reported in ``model_info['n_instruments']``;
      a warning fires when it reaches the number of units.

    See Roodman (2009, *Stata Journal*) for practical guidance.

    References
    ----------
    Arellano, M. and Bond, S. (1991). Some tests of specification for panel
    data: Monte Carlo evidence and an application to employment equations.
    *Review of Economic Studies*. [@arellano1991some]
    Roodman, D. (2009). How to do xtabond2: An introduction to difference and
    system GMM in Stata. *Stata Journal*. [@roodman2009xtabond]
    """
    if method not in ("difference", "system"):
        raise ValueError(f"method must be 'difference' or 'system', got {method!r}.")

    fit = fit_dynamic_panel(
        data,
        y=y,
        x=x,
        id=id,
        time=time,
        lags=lags,
        gmm_lags=gmm_lags,
        twostep=twostep,
        robust=robust,
        predetermined=predetermined,
        endogenous=endogenous,
        predetermined_lags=predetermined_lags,
        endogenous_lags=endogenous_lags,
        collapse=collapse,
        time_dummies=time_dummies,
        method=method,
        transform="fod" if orthogonal else "fd",
        constant=constant,
        stacklevel=3,
    )

    beta = np.asarray(fit["beta"], dtype=float)
    se = np.asarray(fit["se"], dtype=float)
    names = list(fit["names"])

    z_crit = stats.norm.ppf(1 - alpha / 2)
    rho = float(beta[0])
    rho_se = float(se[0])
    z_val = rho / rho_se if rho_se > 0 else np.nan
    pvalue = float(2 * stats.norm.sf(abs(z_val))) if np.isfinite(z_val) else np.nan
    ci = (rho - z_crit * rho_se, rho + z_crit * rho_se)

    with np.errstate(divide="ignore", invalid="ignore"):
        z_stats = np.where(se > 0, beta / se, np.nan)
    pvals = 2 * stats.norm.sf(np.abs(z_stats))
    detail = pd.DataFrame(
        {
            "variable": names,
            "coefficient": beta,
            "se": se,
            "z": z_stats,
            "pvalue": pvals,
        }
    )

    ar1, ar2 = fit["ar1"], fit["ar2"]
    sargan, hansen = fit["sargan"], fit["hansen"]
    model_info = {
        "method": ("SYSTEM GMM" if method == "system" else "DIFFERENCE GMM"),
        "n_obs_diff": fit["n_obs_diff"],
        "n_obs_level": fit["n_obs_level"],
        "n_obs_total": fit["n_obs_total"],
        "constant": fit["constant"],
        "twostep": twostep,
        "robust": robust,
        "windmeijer": bool(twostep and robust),
        "collapse": bool(collapse),
        "transform": "orthogonal deviations" if orthogonal else "first differences",
        "time_dummies": list(fit["time_dummies"]),
        "n_units": fit["n_units"],
        "n_obs": fit["n_obs"],
        "n_instruments": fit["n_instruments"],
        "n_regressors": fit["n_params"],
        "gmm_lags": fit["gmm_lags"],
        "ar1_z": ar1["z"],
        "ar1_p": ar1["pvalue"],
        "ar2_z": ar2["z"],
        "ar2_p": ar2["pvalue"],
        "sargan_stat": sargan["stat"],
        "sargan_df": sargan["df"],
        "sargan_p": sargan["pvalue"],
        "hansen_stat": hansen["stat"],
        "hansen_df": hansen["df"],
        "difference_in_hansen": fit["difference_in_hansen"],
        "hansen_p": hansen["pvalue"],
    }
    for key, note in (
        ("gap_warning", fit["gap_warning"]),
        ("instrument_warning", fit["instrument_warning"]),
        ("hansen_warning", fit["hansen_warning"]),
    ):
        if note:
            model_info[key] = note

    return CausalResult(
        method=(
            f"{'Blundell-Bond system' if method == 'system' else 'Arellano-Bond'} "
            f"({'Two-step' if twostep else 'One-step'} GMM)"
        ),
        estimand="rho (AR coefficient)",
        estimate=rho,
        se=rho_se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=fit["n_obs"],
        detail=detail,
        model_info=model_info,
        _citation_key="arellano_bond",
    )


def xtdpdsys(
    data: pd.DataFrame,
    y: str,
    x: Optional[Sequence[str]] = None,
    id: str = "id",
    time: str = "time",
    lags: int = 1,
    gmm_lags: Tuple[int, Optional[int]] = (2, None),
    twostep: bool = False,
    robust: bool = True,
    alpha: float = 0.05,
    **kwargs,
) -> CausalResult:
    """
    Blundell-Bond (1998) system GMM for dynamic panels.

    Thin alias for ``sp.xtabond(..., method='system')``, named after Stata's
    ``xtdpdsys`` so the estimator is discoverable under the name applied
    researchers already use. Every keyword of :func:`xtabond` is accepted.

    Parameters
    ----------
    data, y, x, id, time, lags, gmm_lags, twostep, robust, alpha
        As in :func:`xtabond`.
    **kwargs
        Forwarded to :func:`xtabond` — ``predetermined``, ``endogenous``,
        ``collapse``, ``time_dummies``, ``constant``, and the instrument
        lag windows.

    Returns
    -------
    CausalResult
        As :func:`xtabond`, including the ``_cons`` row the level equation
        identifies.

    Notes
    -----
    System GMM stacks the level equation alongside the first-differenced
    one and instruments it with lagged differences. It is the right default
    when the dependent variable is persistent — the case in which the
    lagged levels that difference GMM relies on are weak instruments and
    the estimate collapses toward (or past) unity.

    The extra moments buy that power at the price of an extra assumption:
    the deviation of each unit's initial condition from its long-run mean
    must be uncorrelated with the fixed effect. Test it — the level
    instruments have their own difference-in-Hansen statistic.

    Validated to machine precision against ``xtabond2 ..., robust``
    (one-step, two-step Windmeijer, and collapsed) on the ``abdata`` panel;
    see ``tests/reference_parity/test_dynpanel_abdata_parity.py``.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.xtdpdsys(df, y='n', x=['w', 'k'], id='id',      # doctest: +SKIP
    ...                   time='year', twostep=True, collapse=True)

    References
    ----------
    Blundell, R. and Bond, S. (1998). Initial conditions and moment
    restrictions in dynamic panel data models. *Journal of Econometrics*.
    [@blundell1998initial]
    Roodman, D. (2009). How to do xtabond2. *Stata Journal*.
    [@roodman2009xtabond]
    """
    return xtabond(
        data=data,
        y=y,
        x=x,
        id=id,
        time=time,
        lags=lags,
        gmm_lags=gmm_lags,
        method="system",
        twostep=twostep,
        robust=robust,
        alpha=alpha,
        **kwargs,
    )


# Citations
CausalResult._CITATIONS["arellano_bond"] = (
    "@article{arellano1991some,\n"
    "  title={Some Tests of Specification for Panel Data: Monte Carlo "
    "Evidence and an Application to Employment Equations},\n"
    "  author={Arellano, Manuel and Bond, Stephen},\n"
    "  journal={Review of Economic Studies},\n"
    "  volume={58},\n"
    "  number={2},\n"
    "  pages={277--297},\n"
    "  year={1991},\n"
    "  publisher={Oxford University Press}\n"
    "}"
)
