"""
One-call DID robustness summary across multiple staggered-DID estimators.

``did_summary()`` fits a common set of modern staggered-DID estimators to
the same data and returns a tidy comparison table with overall ATT, SE,
95 % CI, p-value, and a short note for each method. The goal is to make
method-robustness checks a single function call instead of hand-wiring
five different estimators.

Supported methods (all use the shared ``(y, group, time, first_treat)``
interface — ``group`` is the unit identifier, ``first_treat`` the cohort
variable):

================  =====================================================
Key               Underlying estimator
================  =====================================================
``'cs'``          Callaway & Sant'Anna (2021) + ``aggte(type='simple')``
``'sa'``          Sun & Abraham (2021) interaction-weighted
``'bjs'``         Borusyak, Jaravel & Spiess (2024) imputation
``'etwfe'``       Wooldridge (2021) extended TWFE
``'stacked'``     Cengiz et al. (2019) stacked event study
================  =====================================================

Example
-------
>>> import statspai as sp
>>> df = sp.dgp_did(n_units=200, n_periods=10, staggered=True, seed=0)
>>> summary = sp.did_summary(df, y='y', time='time',
...                          first_treat='first_treat', group='unit')
>>> print(summary.detail)
"""

from typing import Optional, List, Union

import numpy as np
import pandas as pd

from ..core.results import CausalResult


_DEFAULT_METHODS: List[str] = ["cs", "sa", "bjs", "etwfe", "stacked"]

_METHOD_LABELS = {
    "cs": "Callaway & Sant'Anna (2021)",
    "sa": "Sun & Abraham (2021)",
    "bjs": "Borusyak, Jaravel & Spiess (2024)",
    "etwfe": "Wooldridge (2021) ETWFE",
    "stacked": "Stacked DID (Cengiz et al. 2019)",
}


def _run_cs(data, y, group, time, first_treat, controls, cluster, alpha):
    from .callaway_santanna import callaway_santanna
    from .aggte import aggte

    cs = callaway_santanna(
        data, y=y, g=first_treat, t=time, i=group,
        x=controls, alpha=alpha,
    )
    return aggte(cs, type="simple", alpha=alpha, bstrap=False)


def _run_sa(data, y, group, time, first_treat, controls, cluster, alpha):
    from .sun_abraham import sun_abraham

    return sun_abraham(
        data, y=y, g=first_treat, t=time, i=group,
        covariates=controls, cluster=cluster, alpha=alpha,
    )


def _run_bjs(data, y, group, time, first_treat, controls, cluster, alpha):
    from .did_imputation import did_imputation

    return did_imputation(
        data, y=y, group=group, time=time, first_treat=first_treat,
        controls=controls, cluster=cluster, alpha=alpha,
    )


def _run_etwfe(data, y, group, time, first_treat, controls, cluster, alpha):
    from .wooldridge_did import etwfe

    return etwfe(
        data, y=y, group=group, time=time, first_treat=first_treat,
        controls=controls, cluster=cluster, alpha=alpha,
    )


def _run_stacked(data, y, group, time, first_treat, controls, cluster, alpha):
    from .stacked_did import stacked_did

    return stacked_did(
        data, y=y, group=group, time=time, first_treat=first_treat,
        controls=controls, cluster=cluster, alpha=alpha,
    )


_DISPATCH = {
    "cs": _run_cs,
    "sa": _run_sa,
    "bjs": _run_bjs,
    "etwfe": _run_etwfe,
    "stacked": _run_stacked,
}


def _extract(res: CausalResult) -> dict:
    """Pull (estimate, se, pvalue, ci_low, ci_high) from a CausalResult."""
    est = float(res.estimate) if res.estimate is not None else np.nan
    se = float(res.se) if res.se is not None else np.nan
    p = float(res.pvalue) if res.pvalue is not None else np.nan
    ci = res.ci if res.ci is not None else (np.nan, np.nan)
    ci_lo = float(ci[0]) if ci[0] is not None else np.nan
    ci_hi = float(ci[1]) if ci[1] is not None else np.nan
    n = int(res.n_obs) if getattr(res, "n_obs", None) else np.nan
    return dict(estimate=est, se=se, pvalue=p,
                ci_low=ci_lo, ci_high=ci_hi, n_obs=n)


def did_summary(
    data: pd.DataFrame,
    y: str,
    time: str,
    first_treat: str,
    group: str,
    methods: Union[str, List[str]] = "auto",
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    verbose: bool = False,
) -> CausalResult:
    """
    One-call method-robustness comparison for staggered DID.

    Fits every requested estimator to the same data and returns a single
    :class:`CausalResult` whose ``detail`` attribute is a tidy comparison
    table — one row per method, columns ``(method, estimator, estimate,
    se, pvalue, ci_low, ci_high, n_obs, note)``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset (long format).
    y : str
        Outcome variable.
    time : str
        Time / period variable (integer-valued).
    first_treat : str
        First-treatment period per unit; NaN (or 0) for never-treated.
    group : str
        Unit identifier.
    methods : str or list of str, default ``'auto'``
        Methods to run. Valid keys: ``'cs'``, ``'sa'``, ``'bjs'``,
        ``'etwfe'``, ``'stacked'``, or ``'all'`` / ``'auto'`` for all.
    controls : list of str, optional
        Time-varying covariates passed to methods that support them.
    cluster : str, optional
        Cluster variable for SE (defaults to ``group`` in each sub-method).
    alpha : float, default 0.05
        Significance level for confidence intervals.
    verbose : bool, default False
        Print progress for each method.

    Returns
    -------
    CausalResult
        ``estimate`` : mean of successfully-fit overall ATTs.
        ``se``       : standard deviation across methods (not a standard
                       error — a crude dispersion measure).
        ``detail``   : comparison DataFrame described above.
        ``model_info`` : ``{'methods_requested': [...], 'methods_fit':
                        [...], 'methods_failed': {name: error_msg, ...}}``.

    Notes
    -----
    Each method's overall ATT has slightly different interpretation:

    - CS ``aggte(type='simple')`` averages ATT(g, t) for post-treatment
      :math:`t \\geq g`, weighted by cohort size × exposure length.
    - SA / ETWFE / BJS / Stacked report cohort-size-weighted averages
      by construction.

    Differences across methods are informative about heterogeneity,
    model specification, and the sensitivity of conclusions to the
    estimator choice. Large disagreement is a red flag that deserves
    further investigation (e.g., via ``sp.bacon_decomposition`` or
    ``sp.honest_did``).

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=10, staggered=True, seed=0)
    >>> out = sp.did_summary(df, y='y', time='time',
    ...                      first_treat='first_treat', group='unit')
    >>> out.summary()
    >>> print(out.detail[['method', 'estimate', 'se', 'pvalue']])
    """
    if methods in ("auto", "all"):
        methods_list = list(_DEFAULT_METHODS)
    elif isinstance(methods, str):
        methods_list = [methods]
    else:
        methods_list = list(methods)

    unknown = [m for m in methods_list if m not in _DISPATCH]
    if unknown:
        raise ValueError(
            f"Unknown method(s): {unknown}. "
            f"Valid keys: {sorted(_DISPATCH)} (or 'auto' / 'all')."
        )

    rows: List[dict] = []
    failed: dict = {}
    fit: List[str] = []

    for name in methods_list:
        label = _METHOD_LABELS[name]
        if verbose:
            print(f"  running {name} ({label})...", flush=True)
        try:
            res = _DISPATCH[name](
                data, y=y, group=group, time=time,
                first_treat=first_treat, controls=controls,
                cluster=cluster, alpha=alpha,
            )
            vals = _extract(res)
            rows.append(dict(method=name, estimator=label, note="", **vals))
            fit.append(name)
        except Exception as exc:
            failed[name] = type(exc).__name__ + ": " + str(exc)[:160]
            rows.append(dict(
                method=name, estimator=label,
                estimate=np.nan, se=np.nan, pvalue=np.nan,
                ci_low=np.nan, ci_high=np.nan, n_obs=np.nan,
                note=f"FAILED: {type(exc).__name__}",
            ))

    detail = pd.DataFrame(rows, columns=[
        "method", "estimator", "estimate", "se", "pvalue",
        "ci_low", "ci_high", "n_obs", "note",
    ])

    ests = detail.loc[detail["estimate"].notna(), "estimate"].values
    if len(ests) > 0:
        avg_est = float(np.mean(ests))
        disp = float(np.std(ests, ddof=1)) if len(ests) > 1 else np.nan
    else:
        avg_est, disp = np.nan, np.nan

    return CausalResult(
        method="DID Method-Robustness Summary",
        estimand="Overall ATT (mean across methods)",
        estimate=avg_est,
        se=disp,
        pvalue=np.nan,
        ci=(np.nan, np.nan),
        alpha=alpha,
        n_obs=int(len(data)),
        detail=detail,
        model_info={
            "methods_requested": methods_list,
            "methods_fit": fit,
            "methods_failed": failed,
            "dispersion": disp,
        },
        _citation_key="did_summary",
    )
