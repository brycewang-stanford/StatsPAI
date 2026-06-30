"""
Thin wrappers around pyfixest estimation functions.

Each wrapper delegates to pyfixest and converts results into
StatsPAI's ``EconometricResults``, making them compatible with
``outreg2`` and the rest of the StatsPAI ecosystem.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..core.results import EconometricResults
from ..exceptions import MethodIncompatibility
from .adapter import _multi_fit_to_results, _pyfixest_to_econometric_results

#: vcov sentinels (case-insensitive) that select the wild cluster bootstrap.
_WILD_VCOV = frozenset({"wild", "wildbootstrap", "wild_cluster", "wcr", "boottest"})


def _check_pyfixest() -> Any:
    """Import pyfixest or raise a clear error."""
    try:
        import pyfixest as pf

        return pf
    except ImportError:
        raise ImportError(
            "pyfixest is required for high-dimensional fixed effects estimation.\n"
            "Install it with: pip install pyfixest\n"
            "Or install StatsPAI with the fixest extra: pip install statspai[fixest]"
        )


# --------------------------------------------------------------------------- #
#  feols — OLS / IV with high-dimensional fixed effects
# --------------------------------------------------------------------------- #


def _feols_wild(
    fml: str,
    data: pd.DataFrame,
    cluster: str,
    *,
    reps: int,
    weight_type: str,
    seed: Optional[int],
    weights: Optional[str],
    ssc: Optional[Any],
    fixef_rm: str,
    collin_tol: float,
    lean: bool,
    extra: Dict[str, Any],
) -> EconometricResults:
    """Native ``feols(..., vce="wild")`` — Stata ``boottest``-style inference.

    Fits the model once with cluster-robust (CRV1) SEs — which also stores the
    within-transformed design on the result (see ``fixest/adapter.py``) — then
    runs the WCR wild cluster bootstrap (Cameron-Gelbach-Miller 2008) on that
    partialled-out design for every non-absorbed coefficient.  Point estimates
    are the feols estimates; p-values and confidence intervals come from the
    bootstrap.  This is the verified within wild bootstrap (byte-identical to
    ``sp.regress`` on FE-demeaned data), promoted from the standalone
    ``sp.wild_cluster_boot`` to a first-class ``vce=`` option.
    """
    base = feols(
        fml,
        data,
        vcov={"CRV1": cluster},
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        collin_tol=collin_tol,
        lean=lean,
        **extra,
    )
    if isinstance(base, list):
        raise MethodIncompatibility(
            "feols(vce='wild') does not support multiple-estimation formulas "
            "(csw/sw/csw0). Fit each model separately."
        )

    from ..inference.jackknife import wild_cluster_boot

    se = base.std_errors.copy()
    pvals = pd.Series(np.nan, index=base.params.index, dtype=float)
    ci_lo = pd.Series(np.nan, index=base.params.index, dtype=float)
    ci_hi = pd.Series(np.nan, index=base.params.index, dtype=float)
    for var in base.params.index:
        out = wild_cluster_boot(
            base,
            data,
            cluster=cluster,
            variable=str(var),
            n_boot=reps,
            weight_type=weight_type,
            seed=seed,
        )
        pvals[var] = out["p_boot"]
        ci_lo[var], ci_hi[var] = out["ci_boot"]
        se[var] = out["se_cluster"]

    base.std_errors = se
    base.pvalues = pvals
    base.conf_int_lower = ci_lo
    base.conf_int_upper = ci_hi
    base.model_info = dict(base.model_info)
    base.model_info["vcov_type"] = (
        f"wild cluster bootstrap (Cameron-Gelbach-Miller 2008, {reps} reps, "
        f"{weight_type})"
    )
    base.model_info["n_boot"] = reps
    base.model_info["cluster"] = cluster
    return base


@accepts_aliases(vce="vcov")
def feols(
    fml: str,
    data: pd.DataFrame,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    *,
    weights: Optional[str] = None,
    ssc: Optional[Any] = None,
    fixef_rm: str = "none",
    collin_tol: float = 1e-6,
    lean: bool = False,
    cluster: Optional[str] = None,
    wild_reps: int = 999,
    wild_weight_type: str = "rademacher",
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Union[EconometricResults, List[EconometricResults]]:
    """
    Estimate OLS / IV with high-dimensional fixed effects via pyfixest.

    Uses the Frisch-Waugh-Lovell theorem for fast absorption of
    high-dimensional fixed effects.

    Parameters
    ----------
    fml : str
        A pyfixest-style formula. Examples:

        - ``"Y ~ X1 + X2"`` — plain OLS
        - ``"Y ~ X1 | firm + year"`` — two-way fixed effects
        - ``"Y ~ 1 | firm | X1 ~ Z1"`` — IV with fixed effects
        - ``"Y ~ X1 | csw0(firm, year)"`` — multiple estimations

    data : pd.DataFrame
        Input dataset.
    vcov : str or dict, optional
        Variance-covariance estimator.

        - ``"iid"`` — classical
        - ``"HC1"``, ``"HC2"``, ``"HC3"`` — heteroskedasticity-robust
        - ``{"CRV1": "firm"}`` — cluster-robust
        - ``{"CRV1": "firm + year"}`` — two-way clustering
    weights : str, optional
        Column name for regression weights.
    ssc : optional
        Small-sample correction via ``pyfixest.ssc()``.
    fixef_rm : str, default "none"
        How to handle singleton fixed effects:
        ``"none"`` (keep) or ``"singleton"`` (drop).
    collin_tol : float, default 1e-6
        Collinearity tolerance.
    lean : bool, default False
        If True, drop large intermediate arrays to save memory.
    **kwargs
        Additional arguments passed to ``pyfixest.feols()``.

    Returns
    -------
    EconometricResults or list of EconometricResults
        Single result for simple formulas, list for multiple estimations
        (e.g. ``csw0``/``sw``/``sw0`` syntax).

    Examples
    --------
    Two-way fixed effects with clustered SEs:

    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> firm = rng.integers(0, 8, n)
    >>> year = rng.integers(0, 5, n)
    >>> x1 = rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x1 + 0.1 * firm + 0.2 * year + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "firm": firm, "year": year})
    >>> res = sp.feols("y ~ x1 | firm + year", data=df, vcov={"CRV1": "firm"})  # doctest: +SKIP
    >>> "x1" in res.params.index  # doctest: +SKIP
    True

    Multiple estimation (``csw0`` returns a list, one fit per FE set):

    >>> results = sp.feols("y ~ x1 | csw0(firm, year)", data=df)  # doctest: +SKIP
    >>> summaries = [r.summary() for r in results]  # doctest: +SKIP

    Use with outreg2 to build a regression table:

    >>> r1 = sp.feols("y ~ x1", data=df)  # doctest: +SKIP
    >>> r2 = sp.feols("y ~ x1 | firm", data=df)  # doctest: +SKIP
    >>> sp.outreg2(r1, r2, filename="table.xlsx")  # doctest: +SKIP
    """
    # Wild cluster bootstrap path (Stata ``boottest`` / ``vce()``-style):
    #   sp.feols("y ~ x | firm", data=df, vce="wild", cluster="firm")
    if isinstance(vcov, str) and vcov.lower() in _WILD_VCOV:
        if cluster is None:
            raise MethodIncompatibility(
                "feols(vce='wild') requires cluster=... — the wild *cluster* "
                "bootstrap resamples residuals within clusters."
            )
        return _feols_wild(
            fml,
            data,
            cluster,
            reps=wild_reps,
            weight_type=wild_weight_type,
            seed=seed,
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            lean=lean,
            extra=kwargs,
        )

    # Grammar convenience: `cluster="firm"` is shorthand for one-way CRV1, so the
    # SE keyword reads the same as `sp.regress(..., cluster=...)`. An explicit
    # `vcov=` always wins if both are given.
    if cluster is not None and vcov is None:
        vcov = {"CRV1": cluster}

    pf = _check_pyfixest()

    pf_kwargs: Dict[str, Any] = {
        "fml": fml,
        "data": data,
        "fixef_rm": fixef_rm,
        "collin_tol": collin_tol,
        "lean": lean,
        **kwargs,
    }
    if vcov is not None:
        pf_kwargs["vcov"] = vcov
    if weights is not None:
        pf_kwargs["weights"] = weights
    if ssc is not None:
        pf_kwargs["ssc"] = ssc

    fit = pf.feols(**pf_kwargs)

    # Multiple estimation returns FixestMulti
    if hasattr(fit, "all_fitted_models"):
        return _multi_fit_to_results(fit, vcov=None)

    return _pyfixest_to_econometric_results(fit)


# --------------------------------------------------------------------------- #
#  fepois — Poisson regression with high-dimensional fixed effects
# --------------------------------------------------------------------------- #


def fepois(
    fml: str,
    data: pd.DataFrame,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    *,
    weights: Optional[str] = None,
    ssc: Optional[Any] = None,
    fixef_rm: str = "none",
    collin_tol: float = 1e-6,
    iwls_tol: float = 1e-8,
    iwls_maxiter: int = 25,
    **kwargs: Any,
) -> Union[EconometricResults, List[EconometricResults]]:
    """
    Estimate Poisson regression with high-dimensional fixed effects via pyfixest.

    Parameters
    ----------
    fml : str
        pyfixest formula. E.g. ``"Y ~ X1 | firm"``.
    data : pd.DataFrame
        Input dataset.
    vcov : str or dict, optional
        Variance-covariance estimator.
    weights : str, optional
        Column name for regression weights.
    ssc : optional
        Small-sample correction.
    fixef_rm : str, default "none"
        Singleton fixed effect handling.
    collin_tol : float, default 1e-6
        Collinearity tolerance.
    iwls_tol : float, default 1e-8
        IWLS convergence tolerance.
    iwls_maxiter : int, default 25
        Max IWLS iterations.
    **kwargs
        Additional arguments passed to ``pyfixest.fepois()``.

    Returns
    -------
    EconometricResults or list of EconometricResults

    Examples
    --------
    Poisson regression (PPML-style) with firm fixed effects:

    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> firm = rng.integers(0, 8, n)
    >>> x1 = rng.normal(size=n)
    >>> mu = np.exp(0.3 + 0.5 * x1 + 0.1 * firm)
    >>> df = pd.DataFrame({"y": rng.poisson(mu), "x1": x1, "firm": firm})
    >>> res = sp.fepois("y ~ x1 | firm", data=df)  # doctest: +SKIP
    >>> "x1" in res.params.index  # doctest: +SKIP
    True
    """
    pf = _check_pyfixest()

    pf_kwargs: Dict[str, Any] = {
        "fml": fml,
        "data": data,
        "fixef_rm": fixef_rm,
        "collin_tol": collin_tol,
        "iwls_tol": iwls_tol,
        "iwls_maxiter": iwls_maxiter,
        **kwargs,
    }
    if vcov is not None:
        pf_kwargs["vcov"] = vcov
    if weights is not None:
        pf_kwargs["weights"] = weights
    if ssc is not None:
        pf_kwargs["ssc"] = ssc

    fit = pf.fepois(**pf_kwargs)

    if hasattr(fit, "all_fitted_models"):
        return _multi_fit_to_results(fit, vcov=None)

    return _pyfixest_to_econometric_results(fit)


# --------------------------------------------------------------------------- #
#  feglm — GLM with high-dimensional fixed effects
# --------------------------------------------------------------------------- #


def feglm(
    fml: str,
    data: pd.DataFrame,
    family: str = "gaussian",
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    **kwargs: Any,
) -> Union[EconometricResults, List[EconometricResults]]:
    """
    Estimate GLM (logit, probit, Gaussian) with high-dimensional fixed effects.

    Parameters
    ----------
    fml : str
        pyfixest formula.
    data : pd.DataFrame
        Input dataset.
    family : str, default "gaussian"
        GLM family: ``"gaussian"``, ``"logit"``, ``"probit"``.
    vcov : str or dict, optional
        Variance-covariance estimator.
    **kwargs
        Additional arguments passed to ``pyfixest.feglm()``.

    Returns
    -------
    EconometricResults or list of EconometricResults

    Examples
    --------
    Logit GLM with firm fixed effects:

    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> firm = rng.integers(0, 8, n)
    >>> x1 = rng.normal(size=n)
    >>> p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x1)))
    >>> y = (rng.random(n) < p).astype(int)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "firm": firm})
    >>> res = sp.feglm("y ~ x1 | firm", data=df, family="logit")  # doctest: +SKIP
    >>> "x1" in res.params.index  # doctest: +SKIP
    True
    """
    pf = _check_pyfixest()

    pf_kwargs: Dict[str, Any] = {"fml": fml, "data": data, "family": family, **kwargs}
    if vcov is not None:
        pf_kwargs["vcov"] = vcov

    fit = pf.feglm(**pf_kwargs)

    if hasattr(fit, "all_fitted_models"):
        return _multi_fit_to_results(fit, vcov=None)

    return _pyfixest_to_econometric_results(fit)


# --------------------------------------------------------------------------- #
#  etable — convenience re-export
# --------------------------------------------------------------------------- #


def etable(
    *results: EconometricResults,
    **kwargs: Any,
) -> Any:
    """
    Display a pyfixest-style regression table for StatsPAI results.

    If the results carry a ``_pyfixest_fit`` reference, uses pyfixest's
    native ``etable``. Otherwise falls back to a simple pandas summary.

    Parameters
    ----------
    *results : EconometricResults
        One or more fitted results.
    **kwargs
        Passed to ``pyfixest.etable()``.

    Returns
    -------
    str or DataFrame

    Examples
    --------
    Side-by-side comparison of nested specifications:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 200
    >>> df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
    ...                    "firm": rng.integers(0, 10, n)})
    >>> df["wage"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(0, 0.5, n)
    >>> r1 = sp.regress("wage ~ x1", data=df)
    >>> r2 = sp.regress("wage ~ x1 + x2", data=df)
    >>> tab = sp.etable(r1, r2)  # DataFrame: one coefficient column per model
    >>> print(tab.round(2))

    With ``sp.feols`` fits (requires the ``fixest`` extra), pyfixest's
    native styled table is returned instead:

    >>> f1 = sp.feols("wage ~ x1 | firm", data=df)  # doctest: +SKIP
    >>> f2 = sp.feols("wage ~ x1 + x2 | firm", data=df)  # doctest: +SKIP
    >>> tab = sp.etable(f1, f2)  # doctest: +SKIP
    """
    pf_fits = [
        getattr(r, "_pyfixest_fit") for r in results if hasattr(r, "_pyfixest_fit")
    ]

    if pf_fits:
        pf = _check_pyfixest()
        return pf.etable(pf_fits, **kwargs)

    # Fallback: simple pandas table
    rows = {}
    for i, r in enumerate(results):
        col = f"({i + 1})"
        rows[col] = r.params
    return pd.DataFrame(rows)
