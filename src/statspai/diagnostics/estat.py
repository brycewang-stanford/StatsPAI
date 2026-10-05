"""
Unified post-estimation diagnostics -- Stata's ``estat`` command suite for Python.

This is the single-entry-point diagnostic dispatcher that operates on fitted
``EconometricResults`` objects.  Every test is implemented with numpy/scipy
only (no statsmodels dependency).

Usage
-----
>>> import numpy as np
>>> import pandas as pd
>>> import statspai as sp
>>> rng = np.random.default_rng(42)
>>> df = pd.DataFrame({"x1": rng.normal(size=200), "x2": rng.normal(size=200)})
>>> df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(size=200)
>>> result = sp.regress("y ~ x1 + x2", data=df)
>>> out = sp.estat(result, "hettest", print_results=False)   # Breusch-Pagan
>>> out["test"]
'Breusch-Pagan test for heteroskedasticity'
>>> for name in ["white",       # White's general test
...              "reset",       # Ramsey RESET
...              "bgodfrey",    # Breusch-Godfrey serial correlation
...              "dwatson",     # Durbin-Watson
...              "vif",         # Variance Inflation Factors
...              "ic",          # AIC / BIC / HQIC
...              "linktest",    # Specification link test
...              "normality",   # Jarque-Bera + Shapiro-Wilk
...              "leverage"]:   # Cook's D, DFBETAS
...     print(sp.estat(result, name, print_results=False)["test"])
White's test for heteroskedasticity
Ramsey RESET test
Breusch-Godfrey LM test (1 lag)
Durbin-Watson test
Variance Inflation Factors
Information Criteria
Specification link test
Normality of residuals
Leverage and influence diagnostics
>>> len(sp.estat(result, "all", print_results=False))  # all applicable tests
10

IV-specific tests (``"endogenous"`` for Durbin-Wu-Hausman, ``"overid"`` for
Sargan / Hansen J, ``"firststage"`` for the first-stage F) take an IV result:

>>> z = rng.normal(size=500)
>>> u = rng.normal(size=500)
>>> d = z + 0.5 * u + rng.normal(size=500)
>>> iv_df = pd.DataFrame({"y": 1 + 0.5 * d + u, "d": d, "z": z})
>>> iv_res = sp.ivreg("y ~ (d ~ z)", data=iv_df)
>>> fs = sp.estat(iv_res, "firststage", print_results=False)
>>> fs["statistic_label"], bool(fs["statistic"] > 10)
('F', True)

References
----------
Breusch, T.S. and Pagan, A.R. (1979). *Econometrica*, 47(5), 1287--1294.
White, H. (1980). *Econometrica*, 48(4), 817--838.
Ramsey, J.B. (1969). *JRSS-B*, 31(2), 350--371.
Breusch, T.S. (1978). "Testing for Autocorrelation in Dynamic Linear Models."
    *Australian Economic Papers*, 17(31), 334--355. [@breusch1978testing]
Godfrey, L.G. (1978). *Econometrica*, 46(6), 1293--1301.
Jarque, C.M. and Bera, A.K. (1987). *International Statistical Review*, 55, 163--172.
Cook, R.D. (1977). *Technometrics*, 19(1), 15--18. [@breusch1979simple]
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy import stats as sp_stats

from ..exceptions import MethodIncompatibility
from . import _estat_regression as _reg
from ._estat_did import DID_AGGREGATIONS, estat_did_aggregate

# ======================================================================
#  Line characters for pretty-printing
# ======================================================================
_HEAVY = "\u2501"  # ━
_LINE_WIDTH = 65


# ======================================================================
#  Public dispatcher
# ======================================================================


def estat(
    result: Any,
    test: str = "all",
    *,
    print_results: bool = True,
    lags: Optional[int] = None,
    powers: int = 3,
    alpha: float = 0.05,
    variables: Union[None, str, Sequence[str], np.ndarray] = None,
    version: str = "iid",
    rhs: bool = False,
    fill: str = "zero",
    threshold: Optional[float] = None,
    window: Optional[Sequence[float]] = None,
) -> Any:
    """
    Unified post-estimation diagnostics dispatcher.

    Parameters
    ----------
    result : EconometricResults
        A fitted result object with ``params``, ``data_info``, etc.
    test : str
        Name of the diagnostic test.  One of ``'hettest'``, ``'white'``,
        ``'imtest'`` (White's test with the skewness and kurtosis parts),
        ``'reset'``, ``'ovtest'``, ``'bgodfrey'``, ``'durbinalt'`` (Durbin's
        alternative test), ``'archlm'`` (Engle's LM test for ARCH effects),
        ``'dwatson'``, ``'vif'``, ``'beta'`` (standardized coefficients, Stata
        ``regress, beta``),
        ``'ic'``, ``'linktest'``, ``'normality'``, ``'leverage'``,
        ``'endogenous'``, ``'overid'``, ``'firststage'``,
        ``'classification'`` (after ``sp.logit`` / ``sp.probit``), ``'all'``.
        After ``sp.var``: ``'varlmar'`` (LM test for residual
        autocorrelation, at lag orders 1 .. ``lags``),
        ``'varwle'`` (lag exclusion), ``'varstable'`` (eigenvalues),
        ``'vargranger'`` (Granger causality); after ``sp.vec``:
        ``'veclmar'``, ``'vecstable'``. Each returns ``{'test', 'table'}``.
        After ``sp.didregress``: ``'ptrends'`` and ``'granger'``.
        After ``sp.callaway_santanna`` or ``sp.jwdid`` / ``sp.etwfe``:
        ``'simple'``, ``'group'``, ``'calendar'`` and ``'event'``, the
        aggregations of Stata's ``csdid`` / ``jwdid`` ``estat``. These
        return the aggregated result (``sp.aggte`` / ``sp.etwfe_emfx``),
        not a test dictionary.
    print_results : bool, default True
        If True, print a formatted table to stdout.
    lags : int, optional
        Number of lags: of the Breusch-Godfrey test, of Durbin's
        alternative test and of the ARCH LM test (default 1), or the highest lag order of
        ``'varlmar'`` (default 2).
    powers : int, default 3
        Highest power added by the RESET test (``3`` adds the square and the
        cube, as R's ``lmtest::resettest``; Stata's ``estat ovtest`` is
        ``powers=4``).
    alpha : float, default 0.05
        Significance level for interpretation strings.
    variables : {'rhs', 'fitted'}, list of str or ndarray, optional
        ``'hettest'``: what the error variance may depend on. Default
        ``'rhs'``, every regressor (R's ``lmtest::bptest``). ``'fitted'``
        is the default of Stata's ``estat hettest``; a list names
        regressors of the model. With ``'white'``, ``variables='fitted'``
        is the special form of White's test: the squared residuals on the
        fitted values and their squares, two restrictions whatever the
        number of regressors.
    version : {'iid', 'normal', 'fstat'}, default 'iid'
        ``'iid'`` is Koenker's N R-squared, valid without normal errors;
        ``'normal'`` (``'hettest'`` only) is the original score statistic,
        the default of Stata's ``estat hettest``; ``'fstat'`` is the F
        statistic of the auxiliary regression, for ``'hettest'``,
        ``'bgodfrey'``, ``'durbinalt'`` and ``'archlm'``.
    rhs : bool, default False
        ``'reset'`` only: add powers of the regressors instead of powers of
        the fitted values (Stata's ``estat ovtest, rhs``).
    fill : {'zero', 'drop'}, default 'zero'
        ``'bgodfrey'`` only: the lagged residual of the first ``lags``
        observations is set to zero (the default of Stata and of R's
        ``lmtest::bgtest``) or those observations are dropped (Stata's
        ``nomiss0``).
    threshold : float, optional
        ``'classification'`` only: the probability above which an
        observation is classified as a positive. Default 0.5.
    window : (int, int), optional
        ``'event'`` after ``sp.callaway_santanna`` only: the first and last
        event time kept, as in ``estat event, window(-4 5)``. The overall
        estimate then averages the post-treatment event times inside it.

    Returns
    -------
    dict or list of dict
        Test result(s).  Each dict has keys ``'test'``, ``'statistic'``
        (or equivalent), ``'pvalue'`` (when applicable), and
        ``'interpretation'``. The difference-in-differences aggregations
        return a ``CausalResult`` instead: the overall effect in
        ``.estimate`` / ``.se`` and one row per cohort, period or event
        time in ``.detail``.

    Notes
    -----
    ``sp.estat(result, 'group')`` after ``sp.callaway_santanna`` holds the
    cohort shares fixed, which is what ``csdid`` reports as ``GAverage``.
    ``sp.aggte(result, type='group')`` by default also carries the sampling
    error of the estimated shares, as R ``did`` does; the point estimates
    are the same and only the standard error of the overall row differs.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame({
    ...     "x1": rng.normal(size=200),
    ...     "x2": rng.normal(size=200),
    ... })
    >>> df["y"] = (1.0 + 0.5 * df["x1"] - 0.3 * df["x2"]
    ...            + rng.normal(size=200))
    >>> res = sp.regress("y ~ x1 + x2", data=df)
    >>> out = sp.estat(res, "hettest", print_results=False)
    >>> out["statistic_label"]
    'chi2(2)'
    >>> out["pvalue"] < 0.05   # homoskedastic: do not reject H0
    False
    """
    test = test.strip().lower()
    if test in DID_AGGREGATIONS:
        aggregated = estat_did_aggregate(result, test, window=window, alpha=alpha)
        if print_results:
            print(aggregated.summary())
        return aggregated
    var_lags = lags
    lags = 1 if lags is None else lags

    # Alias
    if test == "ovtest":
        test = "reset"
    if test in ("arellano_bond", "ar", "abond2"):
        test = "abond"
    if test in ("difference_in_hansen", "diffhansen", "dif"):
        test = "difhansen"

    # Dynamic-panel GMM fits have their own diagnostic vocabulary; Stata
    # splits it across `estat abond` / `estat sargan` and xtabond2's
    # difference-in-Hansen block. These read what sp.xtabond already
    # computed rather than re-deriving it, so the postestimation output can
    # never disagree with the fit it describes.
    from ._estat_dynpanel import (
        estat_abond,
        estat_difference_in_hansen,
        estat_sargan,
        is_dynamic_panel_result,
    )

    _dispatch = {
        "abond": lambda: estat_abond(result, alpha=alpha),
        "sargan": lambda: estat_sargan(result, alpha=alpha),
        "difhansen": lambda: estat_difference_in_hansen(result, alpha=alpha),
        "hettest": lambda: _reg.hettest(
            result, variables=variables, version=version, alpha=alpha
        ),
        "white": lambda: _reg.white(
            result,
            variables=variables if isinstance(variables, str) else None,
            alpha=alpha,
        ),
        "imtest": lambda: _reg.imtest(result, alpha=alpha),
        "reset": lambda: _reg.reset(result, powers=powers, rhs=rhs, alpha=alpha),
        "bgodfrey": lambda: _reg.bgodfrey(
            result,
            lags=lags,
            fill=fill,
            version="fstat" if version == "fstat" else "iid",
            alpha=alpha,
        ),
        "beta": lambda: _reg.beta(result),
        "durbinalt": lambda: _reg.durbinalt(
            result,
            lags=lags,
            version="fstat" if version == "fstat" else "iid",
            alpha=alpha,
        ),
        "archlm": lambda: _reg.archlm(
            result,
            lags=lags,
            version="fstat" if version == "fstat" else "iid",
            alpha=alpha,
        ),
        "dwatson": lambda: _estat_dwatson(result, alpha=alpha),
        "vif": lambda: _reg.vif(result, alpha=alpha),
        "ic": lambda: _reg.information_criteria(result),
        "classification": lambda: _reg.classification(result, threshold=threshold),
        "linktest": lambda: _estat_linktest(result, alpha=alpha),
        "normality": lambda: _estat_normality(result, alpha=alpha),
        "leverage": lambda: _estat_leverage(result, alpha=alpha),
        "endogenous": lambda: _estat_endogenous(result, alpha=alpha),
        "overid": lambda: _estat_overid(result, alpha=alpha),
        "firststage": lambda: _estat_firststage(result, alpha=alpha),
        "ptrends": lambda: _estat_did_test(result, "ptrends", alpha=alpha),
        "granger": lambda: _estat_did_test(result, "granger", alpha=alpha),
    }

    # A VAR has its own post-estimation suite (Stata: varlmar, varwle,
    # varstable, vargranger); each returns a table.
    _var_tests = {
        "varlmar": lambda: result.lm_test(lags=2 if var_lags is None else var_lags),
        "varwle": lambda: result.lag_exclusion(),
        "varstable": lambda: result.stability(),
        "vargranger": lambda: result.granger_table(),
        "veclmar": lambda: result.lm_test(lags=2 if var_lags is None else var_lags),
        "vecstable": lambda: result.stability(),
    }
    if test in _var_tests:
        needs = "companion" if test.startswith("vec") else "lag_exclusion"
        if not hasattr(result, needs):
            from ..exceptions import MethodIncompatibility

            raise MethodIncompatibility(
                f"sp.estat: {test!r} applies to a model fitted by "
                + ("sp.vec." if test.startswith("vec") else "sp.var."),
                recovery_hint="Pass the result of sp.var(...) / sp.vec(...).",
            )
        table = _var_tests[test]()
        out_var: Dict[str, Any] = {"test": test, "table": table}
        if test in ("varstable", "vecstable"):
            out_var["stable"] = bool(table.attrs.get("stable"))
        if print_results:
            print(table.to_string(index=False))
        return out_var

    if test == "all" and is_dynamic_panel_result(result):
        # 'all' on an OLS/IV fit means the regression diagnostics; on a
        # dynamic-panel fit those are undefined (there are no fitted values
        # in levels), so it means the three dynamic-panel tests instead.
        outputs = [
            estat_abond(result, alpha=alpha),
            estat_sargan(result, alpha=alpha),
            estat_difference_in_hansen(result, alpha=alpha),
        ]
        if print_results:
            for item in outputs:
                _print_result(item)
        return outputs

    if test == "all":
        return _estat_all(
            result, print_results=print_results, lags=lags, powers=powers, alpha=alpha
        )

    if test not in _dispatch:
        available = ", ".join(sorted(_dispatch.keys()) + ["all"])
        raise ValueError(f"Unknown estat test '{test}'. Available: {available}")

    out = _dispatch[test]()

    if print_results:
        _print_result(out)

    return out


# ======================================================================
#  Helpers: extract arrays from result
# ======================================================================


def _get_residuals(result: Any) -> np.ndarray:
    r = result.data_info.get("residuals")
    if r is None:
        raise ValueError(
            "Residuals not stored in result.data_info['residuals']. "
            "Re-estimate with store_residuals=True or pass residuals manually."
        )
    return np.asarray(r, dtype=float)


def _get_fitted(result: Any) -> np.ndarray:
    yhat = result.data_info.get("fitted_values")
    if yhat is None:
        raise ValueError(
            "Fitted values not stored in result.data_info['fitted_values']."
        )
    return np.asarray(yhat, dtype=float)


def _get_X(result: Any) -> np.ndarray:
    X = result.data_info.get("X")
    if X is None:
        raise ValueError("Design matrix not stored in result.data_info['X'].")
    return np.asarray(X, dtype=float)


def _get_y(result: Any) -> np.ndarray:
    y = result.data_info.get("y")
    if y is None:
        raise ValueError("Response vector not stored in result.data_info['y'].")
    return np.asarray(y, dtype=float)


def _get_nobs(result: Any) -> int:
    n = result.data_info.get("nobs")
    if n is None:
        n = len(_get_residuals(result))
    return int(n)


def _ols_fit(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit OLS via normal equations.  Returns (beta, residuals, yhat)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    resid = y - yhat
    return beta, resid, yhat


def _r_squared(y: np.ndarray, resid: np.ndarray) -> float:
    tss = np.sum((y - y.mean()) ** 2)
    rss = np.sum(resid**2)
    return 1.0 - rss / tss if tss > 0 else 0.0


# ======================================================================
#  Individual test implementations
# ======================================================================

# ------------------------------------------------------------------
#  Durbin-Watson
# ------------------------------------------------------------------


def _estat_dwatson(result: Any, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Durbin-Watson statistic for first-order autocorrelation."""
    resid = _get_residuals(result)

    diff = np.diff(resid)
    dw = float(np.sum(diff**2) / np.sum(resid**2))

    # Heuristic interpretation
    if dw < 1.5:
        interp = (
            f"d = {dw:.4f} < 1.5: possible positive autocorrelation. "
            "Consider Newey-West SEs or AR models."
        )
    elif dw > 2.5:
        interp = (
            f"d = {dw:.4f} > 2.5: possible negative autocorrelation. "
            "Investigate further."
        )
    else:
        interp = (
            f"d = {dw:.4f} is near 2: no strong evidence of "
            "first-order autocorrelation."
        )

    return {
        "test": "Durbin-Watson test",
        "H0": "No first-order autocorrelation",
        "H1": "First-order autocorrelation present",
        "statistic": dw,
        "statistic_label": "d",
        "range": "[0, 4]; d = 2 means no autocorrelation",
        "interpretation": interp,
    }


# ------------------------------------------------------------------
#  Link test (specification)
# ------------------------------------------------------------------


def _estat_linktest(result: Any, *, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Specification link test.

    Re-estimate y on yhat and yhat^2.  If yhat^2 is significant the
    model is misspecified (wrong functional form or omitted variables).
    """
    y = _get_y(result)
    yhat = _get_fitted(result)
    n = len(y)

    X_link = np.column_stack([np.ones(n), yhat, yhat**2])
    beta_link, resid_link, _ = _ols_fit(X_link, y)

    # t-test on yhat^2 coefficient (index 2)
    rss = np.sum(resid_link**2)
    mse = rss / (n - 3)
    XtX_inv = np.linalg.inv(X_link.T @ X_link)
    se_hatsq = np.sqrt(mse * XtX_inv[2, 2])
    t_hatsq = beta_link[2] / se_hatsq if se_hatsq > 0 else 0.0
    pval = float(2.0 * sp_stats.t.sf(abs(t_hatsq), n - 3))

    reject = pval < alpha
    interp = (
        f"REJECT H0 at {alpha:.0%}: yhat^2 is significant (t = {t_hatsq:.4f}). "
        "Model may be misspecified."
        if reject
        else f"Cannot reject H0 at {alpha:.0%}: yhat^2 is not significant. "
        "No evidence of link misspecification."
    )

    return {
        "test": "Specification link test",
        "H0": "Model is correctly specified (yhat^2 coefficient = 0)",
        "H1": "Model is misspecified",
        "statistic": float(t_hatsq),
        "statistic_label": f"t({n - 3})",
        "coef_hatsq": float(beta_link[2]),
        "se_hatsq": float(se_hatsq),
        "pvalue": pval,
        "interpretation": interp,
    }


# ------------------------------------------------------------------
#  Normality of residuals
# ------------------------------------------------------------------


def _estat_normality(result: Any, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Jarque-Bera and Shapiro-Wilk tests on residuals."""
    resid = _get_residuals(result)
    n = len(resid)

    # Jarque-Bera
    mu = resid.mean()
    sigma = resid.std(ddof=0)
    if sigma > 0:
        z = (resid - mu) / sigma
        skew = float(np.mean(z**3))
        kurt_excess = float(np.mean(z**4) - 3.0)
    else:
        skew = 0.0
        kurt_excess = 0.0

    jb = (n / 6.0) * (skew**2 + (kurt_excess**2) / 4.0)
    jb_pval = float(sp_stats.chi2.sf(jb, 2))

    # Shapiro-Wilk (scipy limit: n <= 5000)
    if n <= 5000:
        sw_stat, sw_pval = sp_stats.shapiro(resid)
        sw_stat = float(sw_stat)
        sw_pval = float(sw_pval)
    else:
        sw_stat = None
        sw_pval = None

    reject_jb = jb_pval < alpha
    interp_parts = []
    if reject_jb:
        interp_parts.append(
            f"Jarque-Bera REJECTS normality at {alpha:.0%} "
            f"(skewness = {skew:.4f}, excess kurtosis = {kurt_excess:.4f})."
        )
    else:
        interp_parts.append(f"Jarque-Bera cannot reject normality at {alpha:.0%}.")

    if sw_pval is not None:
        if sw_pval < alpha:
            interp_parts.append(f"Shapiro-Wilk REJECTS normality at {alpha:.0%}.")
        else:
            interp_parts.append(f"Shapiro-Wilk cannot reject normality at {alpha:.0%}.")
    else:
        interp_parts.append("Shapiro-Wilk skipped (n > 5000).")

    out: Dict[str, Any] = {
        "test": "Normality of residuals",
        "H0": "Residuals are normally distributed",
        "H1": "Residuals are not normally distributed",
        "jarque_bera": float(jb),
        "jb_pvalue": jb_pval,
        "skewness": skew,
        "excess_kurtosis": kurt_excess,
        "interpretation": " ".join(interp_parts),
    }

    if sw_stat is not None:
        out["shapiro_wilk"] = sw_stat
        out["sw_pvalue"] = sw_pval

    return out


# ------------------------------------------------------------------
#  Leverage / influence diagnostics
# ------------------------------------------------------------------


def _estat_leverage(result: Any, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Cook's distance, DFBETAS, DFFITS, leverage and leave-one-out fits."""
    resid = _get_residuals(result)
    X = _get_X(result)
    n, k = X.shape

    # Hat matrix diagonal, without forming the n x n hat matrix
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)

    mse = np.sum(resid**2) / (n - k)

    # Cook's distance
    cooks_d = (resid**2 / (k * mse)) * (h / (1.0 - h) ** 2)
    threshold = 4.0 / n

    influential_idx = np.where(cooks_d > threshold)[0]

    # Leave-one-out error variance:
    #   s_(i)^2 = ((n-k) s^2 - e_i^2 / (1-h_i)) / (n-k-1)
    mse_loo = np.maximum(
        ((n - k) * mse - resid**2 / (1.0 - h)) / (n - k - 1),
        1e-16,
    )
    # Internally and externally studentized residuals (Stata: rstandard,
    # rstudent). The second is the t statistic of a dummy for observation i.
    rstandard = resid / np.sqrt(mse * (1.0 - h))
    rstudent = resid / np.sqrt(mse_loo * (1.0 - h))

    # DFBETAS (Belsley, Kuh & Welsch): the change in coefficient j when
    # observation i is dropped, in units of its leave-one-out standard error,
    #   DFBETAS_ij = [(X'X)^{-1} x_i]_j e_i / ((1-h_i) s_(i) sqrt((X'X)^{-1}_jj)).
    # correctness fix (2026-10): the sqrt((X'X)^{-1}_jj) divisor was missing,
    # so the values were on the scale of the regressor, not of the standard
    # error, and the 2/sqrt(n) rule was applied to the wrong quantity.
    dfbeta = (X @ XtX_inv) * (resid / (1.0 - h))[:, None]
    dfbetas = dfbeta / (np.sqrt(mse_loo)[:, None] * np.sqrt(np.diag(XtX_inv))[None, :])

    # DFFITS: the change in the fitted value of observation i when it is
    # dropped, in units of its leave-one-out standard error.
    dffits = rstudent * np.sqrt(h / (1.0 - h))

    # Leave-one-out prediction errors without refitting: y_i - yhat_(i) =
    # e_i / (1 - h_i). Their sum of squares is PRESS. The exact normal
    # prediction interval for y_i from the fit without it is
    #   yhat_(i) +/- t_{n-k-1} s_(i) / sqrt(1 - h_i),
    # since 1 + x_i'(X_(i)'X_(i))^{-1} x_i = 1 / (1 - h_i).
    loo_resid = resid / (1.0 - h)
    loo_half = sp_stats.t.ppf(1.0 - alpha / 2.0, n - k - 1) * np.sqrt(
        mse_loo / (1.0 - h)
    )

    # Threshold: |DFBETAS| > 2/sqrt(n)
    dfbetas_thresh = 2.0 / np.sqrt(n)
    dfbetas_flags = np.any(np.abs(dfbetas) > dfbetas_thresh, axis=1)
    dfbetas_flagged_idx = np.where(dfbetas_flags)[0]

    n_influential = len(influential_idx)
    interp_parts = []
    if n_influential > 0:
        pct = 100.0 * n_influential / n
        interp_parts.append(
            f"{n_influential} observation(s) ({pct:.1f}%) have Cook's D > "
            f"{threshold:.4f} (= 4/n)."
        )
    else:
        interp_parts.append("No observations exceed the Cook's D threshold (4/n).")
    if len(dfbetas_flagged_idx) > 0:
        interp_parts.append(
            f"{len(dfbetas_flagged_idx)} observation(s) have |DFBETAS| > "
            f"{dfbetas_thresh:.4f} (= 2/sqrt(n))."
        )

    return {
        "test": "Leverage and influence diagnostics",
        "cooks_d": cooks_d,
        "cooks_d_threshold": threshold,
        "influential_obs": influential_idx.tolist(),
        "n_influential": n_influential,
        "leverage": h,
        "rstandard": rstandard,
        "rstudent": rstudent,
        "dffits": dffits,
        "loo_residuals": loo_resid,
        "loo_interval_halfwidth": loo_half,
        "press": float(np.sum(loo_resid**2)),
        "dfbetas": dfbetas,
        "dfbetas_threshold": dfbetas_thresh,
        "dfbetas_flagged_obs": dfbetas_flagged_idx.tolist(),
        "interpretation": " ".join(interp_parts),
    }


# ------------------------------------------------------------------
#  IV-specific: endogeneity (Durbin-Wu-Hausman)
# ------------------------------------------------------------------


def _iv_stat(result: Any, *keys: str) -> Any:
    """First non-missing value among ``keys`` in model_info, then diagnostics."""
    for store in (
        getattr(result, "model_info", None),
        getattr(result, "diagnostics", None),
    ):
        if not isinstance(store, dict):
            continue
        for key in keys:
            value = store.get(key)
            if value is not None:
                return value
    return None


def _estat_endogenous(result: Any, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Durbin-Wu-Hausman endogeneity test (for IV results)."""
    # sp.ivreg / sp.iv record the Wu-Hausman F test in ``diagnostics``;
    # ``model_info`` keys are the older spelling.
    dwh = _iv_stat(result, "wu_hausman", "dwh_statistic", "Hausman F-stat")
    dwh_pval = _iv_stat(result, "wu_hausman_pvalue", "dwh_pvalue", "Hausman p-value")

    if dwh is None:
        return {
            "test": "Durbin-Wu-Hausman endogeneity test",
            "error": (
                "Durbin-Wu-Hausman statistic not found on this result. "
                "This test requires an IV/2SLS estimation result."
            ),
            "interpretation": "Not applicable: model was not estimated via IV.",
        }

    reject = bool(dwh_pval < alpha) if dwh_pval is not None else None
    if reject is True:
        interp = (
            f"REJECT H0 at {alpha:.0%}: regressors are endogenous. "
            "IV estimation is warranted."
        )
    elif reject is False:
        interp = (
            f"Cannot reject H0 at {alpha:.0%}: no evidence of endogeneity. "
            "OLS may be consistent and more efficient."
        )
    else:
        interp = "p-value not available."

    # After a robust or clustered fit the Wu-Hausman F is not valid; the
    # regression-based robust F is the test, as in Stata `estat endogenous`.
    robust_f = _iv_stat(result, "Robust regression F (endogeneity)")
    if robust_f is not None:
        robust_p = _iv_stat(result, "Robust regression F p-value")
        df_num = _iv_stat(result, "Robust regression F df (num)")
        df_den = _iv_stat(result, "Robust regression F df (denom)")
        df_pair = None if df_num is None or df_den is None else (df_num, df_den)
        reject = bool(robust_p < alpha) if robust_p is not None else None
        if reject is True:
            interp = (
                f"REJECT H0 at {alpha:.0%}: regressors are endogenous. "
                "IV estimation is warranted."
            )
        elif reject is False:
            interp = (
                f"Cannot reject H0 at {alpha:.0%}: no evidence of endogeneity. "
                "OLS may be consistent and more efficient."
            )
        out: Dict[str, Any] = {
            "test": "Robust regression-based endogeneity test",
            "H0": "Regressors are exogenous",
            "H1": "Regressors are endogenous",
            "statistic": float(robust_f),
            "statistic_label": (
                f"F({int(df_pair[0])}, {int(df_pair[1])})"
                if df_pair is not None
                else "F"
            ),
            "pvalue": float(robust_p) if robust_p is not None else None,
            "interpretation": interp,
            "wu_hausman_F": float(dwh),
            "wu_hausman_pvalue": float(dwh_pval) if dwh_pval is not None else None,
        }
        score = _iv_stat(result, "Robust score chi2 (endogeneity)")
        if score is not None:
            out["robust_score_chi2"] = float(score)
            score_p = _iv_stat(result, "Robust score chi2 p-value")
            out["robust_score_pvalue"] = float(score_p) if score_p is not None else None
        return out

    out = {
        "test": "Durbin-Wu-Hausman endogeneity test",
        "H0": "Regressors are exogenous",
        "H1": "Regressors are endogenous",
        "statistic": float(dwh),
        "statistic_label": "DWH",
        "pvalue": float(dwh_pval) if dwh_pval is not None else None,
        "interpretation": interp,
    }
    # Durbin's score form of the same test: with q endogenous regressors and
    # the Wu-Hausman F on (q, m) degrees of freedom, D = N q F / (m + q F),
    # chi-squared with q degrees of freedom.
    info = getattr(result, "data_info", None) or {}
    q = _iv_stat(result, "N endogenous")
    n, k = info.get("nobs"), len(getattr(result, "params", []))
    if q and n and k:
        q, m = int(q), int(n) - k - int(q)
        if m > 0:
            durbin = int(n) * q * float(dwh) / (m + q * float(dwh))
            out.update(
                df1=q,
                df2=m,
                statistic_label=f"F({q}, {m})",
                durbin=durbin,
                durbin_pvalue=float(sp_stats.chi2.sf(durbin, q)),
            )
    return out


# ------------------------------------------------------------------
#  IV-specific: over-identification (Sargan / Hansen J)
# ------------------------------------------------------------------


def _estat_did_test(result: Any, which: str, *, alpha: float = 0.05) -> Dict[str, Any]:
    """``estat ptrends`` / ``estat granger`` after ``sp.didregress``.

    The tests are fitted with the model (they need the data), so this
    reads them from the result.
    """
    info = getattr(result, "model_info", None) or {}
    stored = info.get(which)
    if not isinstance(stored, dict):
        raise MethodIncompatibility(
            f"estat {which} is a post-estimation test of sp.didregress; this "
            "result does not carry it.",
            recovery_hint="Fit the model with sp.didregress(...).",
        )
    if "unavailable" in stored:
        raise MethodIncompatibility(
            f"estat {which}: {stored['unavailable']}.",
            recovery_hint=(
                "With staggered adoption test pre-trends on an event study: "
                "sp.callaway_santanna(...) then sp.pretrends_test(...)."
            ),
            diagnostics={"treatment_times": info.get("treatment_times")},
        )
    label = (
        "Parallel-trends test (pretreatment time period)"
        if which == "ptrends"
        else "Granger causality test"
    )
    pvalue = float(stored["pvalue"])
    return {
        "test": label,
        "H0": stored["H0"],
        "statistic": float(stored["F"]),
        "statistic_label": f"F({int(stored['df'])}, {int(stored['df_denom'])})",
        "df": (int(stored["df"]), int(stored["df_denom"])),
        "pvalue": pvalue,
        "interpretation": (
            f"REJECT H0 at {alpha:.0%}."
            if pvalue < alpha
            else f"Do not reject H0 at {alpha:.0%}."
        ),
    }


def _estat_overid(result: Any, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Sargan/Hansen J test for over-identifying restrictions."""
    # sp.ivreg reports Sargan under i.i.d. errors and Hansen's J once the
    # vcov is robust or clustered, both in ``diagnostics``.
    sargan = _iv_stat(
        result, "sargan_stat", "hansen_j", "Sargan statistic", "Hansen J statistic"
    )
    sargan_pval = _iv_stat(
        result, "sargan_pvalue", "hansen_j_pvalue", "Sargan p-value", "Hansen J p-value"
    )
    sargan_df = _iv_stat(result, "sargan_df", "overid_df", "Sargan df", "Hansen J df")

    if sargan is None:
        return {
            "test": "Sargan/Hansen J over-identification test",
            "error": (
                "Over-identification statistic not found on this result. "
                "This test requires an IV estimation with more instruments "
                "than endogenous regressors."
            ),
            "interpretation": "Not applicable or model is exactly identified.",
        }

    reject = bool(sargan_pval < alpha) if sargan_pval is not None else None
    if reject is True:
        interp = (
            f"REJECT H0 at {alpha:.0%}: instruments may not all be valid. "
            "Re-examine instrument exogeneity."
        )
    elif reject is False:
        interp = (
            f"Cannot reject H0 at {alpha:.0%}: over-identifying restrictions "
            "are satisfied. Instruments appear valid."
        )
    else:
        interp = "p-value not available."

    label = f"chi2({sargan_df})" if sargan_df else "J"

    return {
        "test": "Sargan/Hansen J over-identification test",
        "H0": "All instruments are valid (exclusion restrictions hold)",
        "H1": "At least one instrument is invalid",
        "statistic": float(sargan),
        "statistic_label": label,
        "df": int(sargan_df) if sargan_df is not None else None,
        "pvalue": float(sargan_pval) if sargan_pval is not None else None,
        "interpretation": interp,
    }


# ------------------------------------------------------------------
#  IV-specific: first-stage F
# ------------------------------------------------------------------


def _estat_firststage(result: Any, *, alpha: float = 0.05) -> Dict[str, Any]:
    """First-stage F-statistic for weak instrument detection."""
    mi = result.model_info

    f_stat = mi.get("first_stage_f") or mi.get("first_stage_F")
    f_pval = mi.get("first_stage_f_pvalue")

    if f_stat is None:
        return {
            "test": "First-stage F-statistic",
            "error": (
                "First-stage F not found in model_info. "
                "This test requires an IV estimation."
            ),
            "interpretation": "Not applicable: model was not estimated via IV.",
        }

    # Stock-Yogo critical values (common thresholds)
    weak = float(f_stat) < 10.0
    if weak:
        interp = (
            f"F = {float(f_stat):.2f} < 10: instruments are weak "
            "(Stock-Yogo rule of thumb). Consider LIML, Fuller, or "
            "Anderson-Rubin confidence sets."
        )
    else:
        interp = (
            f"F = {float(f_stat):.2f} >= 10: instruments are not weak "
            "(Stock-Yogo rule of thumb)."
        )

    out: Dict[str, Any] = {
        "test": "First-stage F-statistic (weak instruments)",
        "H0": "Instruments are weak (excluded instruments have no explanatory power)",
        "statistic": float(f_stat),
        "statistic_label": "F",
        "stock_yogo_threshold": 10.0,
        "interpretation": interp,
    }
    if f_pval is not None:
        out["pvalue"] = float(f_pval)

    return out


# ======================================================================
#  "all" -- run every applicable test
# ======================================================================


def _estat_all(
    result: Any,
    *,
    print_results: bool = True,
    lags: int = 1,
    powers: int = 3,
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Run all applicable tests and return results as a list."""
    outputs: List[Dict[str, Any]] = []

    # Tests that require residuals + X (standard OLS diagnostics)
    has_resid = result.data_info.get("residuals") is not None
    has_X = result.data_info.get("X") is not None
    has_y = result.data_info.get("y") is not None
    has_fitted = result.data_info.get("fitted_values") is not None

    if has_resid and has_X:
        outputs.append(_reg.hettest(result, alpha=alpha))
        outputs.append(_reg.white(result, alpha=alpha))
        outputs.append(_reg.bgodfrey(result, lags=lags, alpha=alpha))
        outputs.append(_estat_dwatson(result, alpha=alpha))
        outputs.append(_reg.vif(result, alpha=alpha))
        outputs.append(_reg.information_criteria(result))
        outputs.append(_estat_normality(result, alpha=alpha))

    if has_y and has_X and has_resid and has_fitted:
        outputs.append(_reg.reset(result, powers=powers, alpha=alpha))
        outputs.append(_estat_linktest(result, alpha=alpha))
        outputs.append(_estat_leverage(result, alpha=alpha))

    # IV-specific tests. sp.ivreg labels its fits "IV-2SLS", "IV-LIML", ...,
    # which an exact match against ("iv", "2sls", ...) never recognised.
    model_type = str(result.model_info.get("model_type", "")).lower()
    tokens = set(re.split(r"[^a-z0-9]+", model_type))
    is_iv = bool(tokens & {"iv", "2sls", "gmm", "liml", "fuller", "jive"}) or (
        _iv_stat(result, "Hausman F-stat", "wu_hausman", "dwh_statistic") is not None
    )
    if is_iv:
        outputs.append(_estat_endogenous(result, alpha=alpha))
        outputs.append(_estat_overid(result, alpha=alpha))
        outputs.append(_estat_firststage(result, alpha=alpha))

    if not outputs:
        outputs.append(
            {
                "test": "estat: no applicable tests",
                "interpretation": (
                    "Could not run any tests. Ensure the result object stores "
                    "residuals, fitted values, design matrix (X), and response (y) "
                    "in data_info."
                ),
            }
        )

    if print_results:
        _print_all(outputs)

    return outputs


# ======================================================================
#  Pretty-printing
# ======================================================================


def _fmt_line(width: int = _LINE_WIDTH) -> str:
    return _HEAVY * width


def _print_dynpanel_rows(out: Dict[str, Any]) -> None:
    """Render the row tables produced by the dynamic-panel estat handlers."""
    test = out["test"]
    rows = out["rows"]
    if test == "abond":
        print(f"  {'order':>6}  {'z':>10}  {'Pr > z':>10}")
        for row in rows:
            print(f"  {row['order']:>6}  {row['z']:>10.4f}  {row['pvalue']:>10.4f}")
    elif test == "sargan":
        print(f"  {'test':<10} {'chi2':>12} {'df':>5} {'Prob > chi2':>12}")
        for row in rows:
            print(
                f"  {row['name']:<10} {row['statistic']:>12.4f} "
                f"{row['df']:>5} {row['pvalue']:>12.4f}"
            )
        print(f"\n  instruments: {out['n_instruments']}   units: {out['n_units']}")
    else:
        print(
            f"  {'instrument subset':<34} {'excl. J':>10} {'df':>4} "
            f"{'C':>10} {'df':>4} {'Prob > chi2':>12}"
        )
        for row in rows:
            print(
                f"  {row['subset'][:34]:<34} {row['hansen_excluding']:>10.4f} "
                f"{row['df_excluding']:>4} {row['statistic']:>10.4f} "
                f"{row['df']:>4} {row['pvalue']:>12.4f}"
            )
    if out.get("interpretation"):
        print()
        for chunk in textwrap.wrap(out["interpretation"], _LINE_WIDTH - 6):
            print(f"  {chunk}")


def _print_result(out: Dict[str, Any]) -> None:
    """Print a single test result in Stata-style formatted output."""
    w = _LINE_WIDTH
    line = _fmt_line(w)

    print(line)
    print(f"  {out.get('label') or out.get('test', 'Test')}")
    print(line)

    # Dynamic-panel GMM tables carry a list of rows rather than a single
    # statistic; render them before the generic scalar path so they are
    # readable instead of an empty banner.
    if out.get("test") in ("abond", "sargan", "difhansen") and "rows" in out:
        _print_dynpanel_rows(out)
        print(line)
        return

    # Hypotheses
    if "H0" in out:
        print(f"  H0: {out['H0']}")
    if "H1" in out:
        print(f"  H1: {out['H1']}")
    if "H0" in out or "H1" in out:
        print()

    # Error message (for IV tests when not applicable)
    if "error" in out:
        print(f"  {out['error']}")
        print(line)
        return

    # VIF table
    if "beta_table" in out:
        table = out["beta_table"]
        print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        print()

    elif "vif_table" in out:
        vif_df = out["vif_table"]
        print(vif_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n  Mean VIF = {out.get('mean_vif', 0):.2f}")
        print()

    elif "table" in out and "sensitivity" in out:
        print(out["table"].to_string())
        print()
        for key in ("sensitivity", "specificity", "ppv", "npv"):
            print(f"  {key:<22} = {100 * out[key]:>8.2f}%")
        print(
            f"  {'correctly classified':<22} = "
            f"{100 * out['correctly_classified']:>8.2f}%"
        )
        print()

    # Information criteria
    elif "AIC" in out:
        if "ll" in out:
            print(f"  {'ll':<10} = {out['ll']:>12.4f}")
        print(f"  {'AIC':<10} = {out['AIC']:>12.4f}")
        print(f"  {'BIC':<10} = {out['BIC']:>12.4f}")
        print(f"  {'HQIC':<10} = {out['HQIC']:>12.4f}")
        print(f"  {'n':<10} = {out['n']:>12d}")
        print(f"  {'k':<10} = {out['k']:>12d}")
        print()

    # Normality (two tests)
    elif "jarque_bera" in out:
        print(f"  {'Jarque-Bera':<16} = {out['jarque_bera']:>12.4f}")
        print(f"  {'  p-value':<16} = {out['jb_pvalue']:>12.4f}")
        print(f"  {'  skewness':<16} = {out['skewness']:>12.4f}")
        print(f"  {'  excess kurt.':<16} = {out['excess_kurtosis']:>12.4f}")
        if "shapiro_wilk" in out:
            print()
            print(f"  {'Shapiro-Wilk':<16} = {out['shapiro_wilk']:>12.4f}")
            print(f"  {'  p-value':<16} = {out['sw_pvalue']:>12.4f}")
        print()

    # Leverage / influence
    elif "cooks_d" in out:
        n_inf = out["n_influential"]
        thresh = out["cooks_d_threshold"]
        print(f"  Cook's D threshold (4/n) = {thresh:.4f}")
        print(f"  Observations exceeding threshold: {n_inf}")
        if n_inf > 0 and n_inf <= 20:
            print(f"  Influential obs indices: {out['influential_obs']}")
        elif n_inf > 20:
            print(f"  (Showing first 20): {out['influential_obs'][:20]}")
        print(f"\n  DFBETAS threshold (2/sqrt(n)) = {out['dfbetas_threshold']:.4f}")
        print(f"  Observations with large DFBETAS: {len(out['dfbetas_flagged_obs'])}")
        print()

    # Standard single-statistic test
    elif "statistic" in out:
        label = out.get("statistic_label", "stat")
        print(f"  {label:<10} = {out['statistic']:>12.4f}")
        if "pvalue" in out and out["pvalue"] is not None:
            print(f"  {'p-value':<10} = {out['pvalue']:>12.4f}")
        if "range" in out:
            print(f"  Range: {out['range']}")
        print()

    # Interpretation arrow
    interp = out.get("interpretation", "")
    if interp:
        # Wrap long lines
        _print_wrapped(f"  -> {interp}", width=w)

    print(line)


def _print_all(outputs: List[Dict[str, Any]]) -> None:
    """Print the comprehensive estat report."""
    w = _LINE_WIDTH
    line = _fmt_line(w)

    print()
    print(line)
    print("  COMPREHENSIVE POST-ESTIMATION DIAGNOSTICS")
    print(line)
    print()

    for out in outputs:
        _print_result(out)
        print()

    print(line)
    print("  End of diagnostics")
    print(line)


def _print_wrapped(text: str, width: int = _LINE_WIDTH) -> None:
    """Simple word-wrap for interpretation strings."""
    words = text.split()
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > width:
            print(current_line)
            current_line = "    " + word  # indent continuation
        else:
            current_line = current_line + " " + word if current_line else word
    if current_line:
        print(current_line)
