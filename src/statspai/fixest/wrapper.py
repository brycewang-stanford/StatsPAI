"""
Thin wrappers around pyfixest estimation functions.

Each wrapper delegates to pyfixest and converts results into
StatsPAI's ``EconometricResults``, making them compatible with
``outreg2`` and the rest of the StatsPAI ecosystem.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..core.results import EconometricResults
from .adapter import _pyfixest_to_econometric_results, _multi_fit_to_results


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

    >>> from statspai.fixest import feols
    >>> result = feols("wage ~ experience + tenure | firm + year",
    ...               data=df, vcov={"CRV1": "firm"})
    >>> print(result.summary())

    Multiple estimation:

    >>> results = feols("Y ~ X1 | csw0(firm, year)", data=df)
    >>> for r in results:
    ...     print(r.summary())

    Use with outreg2:

    >>> from statspai import outreg2
    >>> r1 = feols("Y ~ X1", data=df)
    >>> r2 = feols("Y ~ X1 | firm", data=df)
    >>> outreg2(r1, r2, filename="table.xlsx")
    """
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
    """
    pf_fits = [
        getattr(r, "_pyfixest_fit") for r in results
        if hasattr(r, "_pyfixest_fit")
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
