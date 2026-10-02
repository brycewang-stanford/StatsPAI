"""
Hausman (1978) specification test.

Tests whether the fixed effects (FE) or random effects (RE) estimator
is appropriate for panel data. Under H0 (RE is consistent), both FE
and RE are consistent but RE is more efficient. Under H1, only FE
is consistent.

References
----------
Hausman, J.A. (1978).
"Specification Tests in Econometrics."
*Econometrica*, 46(6), 1251-1271. [@hausman1978specification]
"""

import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import AssumptionWarning


def hausman_test(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id: str,
    time: str,
    alpha: float = 0.05,
    sigmamore: bool = False,
) -> Dict[str, Any]:
    """
    Hausman test for FE vs RE in panel data.

    Equivalent to Stata's ``hausman fe re``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Dependent variable.
    x : list of str
        Independent variables (time-varying).
    id : str
        Unit identifier.
    time : str
        Time period identifier.
    alpha : float, default 0.05
    sigmamore : bool, default False
        Stata's ``hausman fe re, sigmamore``: both covariance matrices use the
        RE disturbance variance. Use it when the classical statistic is
        negative (``recommendation == "inconclusive"``).

    Returns
    -------
    dict
        ``'statistic'``: Hausman chi² statistic
        ``'df'``: degrees of freedom
        ``'pvalue'``: p-value (``nan`` when the statistic is negative)
        ``'recommendation'``: 'FE', 'RE', or 'inconclusive' when
        ``V_FE - V_RE`` is not positive semi-definite and the statistic is
        negative (an ``AssumptionWarning`` is raised)
        ``'psd_violation'``: whether ``V_FE - V_RE`` fails to be positive
        definite
        ``'beta_fe'``, ``'beta_re'``: coefficient vectors

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_panel(n_units=40, n_periods=8, seed=0)
    >>> result = sp.hausman_test(df, y='y', x=['x'], id='unit', time='time')
    >>> bool(result['recommendation'] in ('FE', 'RE'))
    True
    >>> bool(0.0 <= result['pvalue'] <= 1.0)
    True

    Notes
    -----
    The test statistic is:

    .. math::
        H = (\\hat{\\beta}_{FE} - \\hat{\\beta}_{RE})'
        [V(\\hat{\\beta}_{FE}) - V(\\hat{\\beta}_{RE})]^{-1}
        (\\hat{\\beta}_{FE} - \\hat{\\beta}_{RE})

    Under H0, H ~ χ²(k).

    - Reject H0 (p < 0.05) → Use Fixed Effects
    - Fail to reject → Use Random Effects (more efficient)

    See Hausman (1978, *Econometrica*).
    """
    # One implementation, pinned to Stata (xtreg fe / xtreg re / hausman) in
    # tests/reference_parity/test_hausman_stata_parity.py. The within / GLS
    # estimators this function used to carry mis-estimated the RE variance
    # (statistic 223 where Stata reports 4.4 on the same panel).
    from ..panel.panel_diagnostics import _hausman_from_data

    return _hausman_from_data(data, y, x, id, time, alpha, sigmamore=sigmamore)


def _coef_cov(result: Any) -> "tuple[pd.Series, pd.DataFrame]":
    """Coefficients and their covariance, with the constant named ``_cons``."""
    params = getattr(result, "params", None)
    if params is None:
        from ..exceptions import MethodIncompatibility

        raise MethodIncompatibility(
            "sp.hausman needs two fitted results with coefficients.",
            recovery_hint="Pass the results of sp.regress / sp.iv / sp.panel.",
        )
    names = [
        "_cons" if str(n) in ("Intercept", "const", "_cons") else str(n)
        for n in params.index
    ]
    cov = None
    info = getattr(result, "data_info", None) or {}
    for key in ("var_cov", "vcov"):
        if cov is None and info.get(key) is not None:
            cov = np.asarray(info[key], dtype=float)
    if cov is None or cov.shape != (len(names), len(names)):
        getter = getattr(result, "cov_params", None) or getattr(result, "vcov", None)
        got = getter() if callable(getter) else getter
        cov = None if got is None else np.asarray(got, dtype=float)
    if cov is None or cov.shape != (len(names), len(names)):
        from ..exceptions import MethodIncompatibility

        raise MethodIncompatibility(
            "sp.hausman: a result does not carry the covariance matrix of its "
            "coefficients.",
            recovery_hint="Use results of sp.regress, sp.iv or sp.panel.",
        )
    coef = pd.Series(np.asarray(params, dtype=float), index=names)
    covariance = pd.DataFrame(cov, index=names, columns=names)
    # A fixed-effects fit absorbs the constant. Stata reports it (the mean
    # unit effect, ybar - xbar'b); when the fit carries it, it takes part in
    # the comparison under `constant=True`.
    xt = (getattr(result, "model_info", None) or {}).get("xt") or {}
    if "_cons" not in names and "cons_se" in xt and "cons_cov" in xt:
        cross = pd.Series(xt["cons_cov"]).reindex(names)
        if not cross.isna().any():
            coef.loc["_cons"] = float(xt["cons"])
            covariance.loc["_cons", names] = cross.to_numpy()
            covariance.loc[names, "_cons"] = cross.to_numpy()
            covariance.loc["_cons", "_cons"] = float(xt["cons_se"]) ** 2
    return coef, covariance


def _disturbance_variance(result: Any) -> float:
    """The disturbance variance the result's own standard errors use."""
    model = getattr(result, "model_info", None) or {}
    info = getattr(result, "data_info", None) or {}
    if model.get("rmse") is not None:
        return float(model["rmse"]) ** 2
    resid = info.get("residuals")
    if resid is None:
        return float("nan")
    resid = np.asarray(resid, dtype=float)
    rss = float(info["rss"]) if info.get("rss") is not None else float(resid @ resid)
    # instrumental variables without a small-sample correction (Stata
    # `ivregress` without `small`) divides by N
    kind = str(model.get("model_type") or "")
    if info.get("inference") == "z" and kind.upper().startswith("IV"):
        return rss / resid.size
    df = info.get("df_resid")
    return rss / float(df if df else resid.size)


def hausman(
    consistent: Any,
    efficient: Any,
    *,
    constant: bool = False,
    sigmamore: bool = False,
    sigmaless: bool = False,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Hausman specification test between two fitted models.

    ``consistent`` is consistent whether or not the hypothesis holds (IV,
    fixed effects); ``efficient`` is efficient under the hypothesis and
    inconsistent otherwise (OLS, random effects). A large statistic says the
    two sets of coefficients differ by more than sampling error allows, so
    the hypothesis behind the efficient estimator is rejected.

    Parameters
    ----------
    consistent, efficient : fitted results
        Results with ``params`` and a coefficient covariance, e.g. from
        :func:`statspai.iv` and :func:`statspai.regress`. The coefficients
        they share (by name) are compared.
    constant : bool, default False
        Include the constant in the comparison (Stata's ``constant``).
    sigmamore : bool, default False
        Base both covariance matrices on the disturbance variance of the
        efficient estimator. This is the form to use when comparing IV with
        OLS: the difference of the two covariances is then positive
        semi-definite by construction.
    sigmaless : bool, default False
        Base both on the variance of the consistent estimator.
    alpha : float, default 0.05
        Level used in the interpretation.

    Returns
    -------
    dict
        ``statistic`` (chi-squared), ``df`` (the rank of ``V_b - V_B``),
        ``pvalue``, ``table`` (a DataFrame with the two coefficient vectors,
        their difference and its standard error), ``psd_violation`` and
        ``interpretation``.

    Notes
    -----
    ``H = (b - B)' (V_b - V_B)^- (b - B)`` with a generalised inverse, and
    as many degrees of freedom as ``V_b - V_B`` has rank. Without
    ``sigmamore`` the difference need not be positive semi-definite in a
    finite sample and the statistic can be negative; it is then reported
    with ``pvalue = nan``.

    The test assumes the efficient estimator is fully efficient, which
    fails under heteroskedasticity or clustering. With robust standard
    errors use the regression-based test instead
    (``sp.estat(iv_result, 'endogenous')``, or the Mundlak form for panels).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> z, u = rng.normal(size=n), rng.normal(size=n)
    >>> d = z + 0.8 * u + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1 + 0.5 * d + u, "d": d, "z": z})
    >>> iv = sp.iv("y ~ (d ~ z)", data=df, small=False)
    >>> ols = sp.regress("y ~ d", data=df)
    >>> out = sp.hausman(iv, ols, sigmamore=True)
    >>> int(out["df"]), bool(out["pvalue"] < 0.05)
    (1, True)

    References
    ----------
    hausman1978specification
    """
    from ..exceptions import MethodIncompatibility

    if sigmamore and sigmaless:
        raise MethodIncompatibility(
            "sp.hausman: sigmamore and sigmaless cannot both be set.",
            recovery_hint="Choose one; sigmamore is the usual choice.",
        )
    b, Vb = _coef_cov(consistent)
    B, VB = _coef_cov(efficient)
    names = [n for n in b.index if n in B.index and (constant or n != "_cons")]
    if not names:
        raise MethodIncompatibility(
            "sp.hausman: the two results share no coefficient.",
            recovery_hint="Fit both models on the same regressors.",
        )
    if sigmamore or sigmaless:
        s_b, s_B = _disturbance_variance(consistent), _disturbance_variance(efficient)
        if not (np.isfinite(s_b) and np.isfinite(s_B) and s_b > 0 and s_B > 0):
            raise MethodIncompatibility(
                "sp.hausman: sigmamore / sigmaless need the disturbance "
                "variance of both models, which a result does not store.",
                recovery_hint="Use results of sp.regress, sp.iv or sp.panel.",
            )
        if sigmamore:
            Vb = Vb * (s_B / s_b)
        else:
            VB = VB * (s_b / s_B)
    diff = (b[names] - B[names]).to_numpy()
    V = Vb.loc[names, names].to_numpy() - VB.loc[names, names].to_numpy()
    V = (V + V.T) / 2.0
    eig = np.linalg.eigvalsh(V)
    scale = float(np.abs(eig).max()) if eig.size else 0.0
    rank = int(np.sum(np.abs(eig) > 1e-8 * scale)) if scale > 0 else 0
    psd_violation = bool(scale > 0 and eig.min() < -1e-8 * scale)
    stat = float(diff @ np.linalg.pinv(V, rcond=1e-8, hermitian=True) @ diff)
    pvalue = float(stats.chi2.sf(stat, rank)) if stat >= 0 and rank > 0 else np.nan
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.diag(V))
    table = pd.DataFrame(
        {
            "b": b[names].to_numpy(),
            "B": B[names].to_numpy(),
            "difference": diff,
            "se": se,
        },
        index=names,
    )
    if stat < 0:
        warnings.warn(
            f"Hausman statistic is negative (chi2({rank}) = {stat:.4f}): "
            "V_b - V_B is not positive semi-definite on these data. Rerun "
            "with sigmamore=True.",
            AssumptionWarning,
            stacklevel=2,
        )
        interp = "The statistic is negative: the test is inconclusive."
    elif pvalue < alpha:
        interp = (
            f"chi2({rank}) = {stat:.4f}, p = {pvalue:.4f}. Reject H0: the "
            "coefficients differ systematically; rely on the consistent "
            "estimator."
        )
    else:
        interp = (
            f"chi2({rank}) = {stat:.4f}, p = {pvalue:.4f}. Cannot reject H0: "
            "no systematic difference; the efficient estimator is admissible."
        )
    return {
        "test": "Hausman specification test",
        "statistic": stat,
        "df": rank,
        "pvalue": pvalue,
        "table": table,
        "psd_violation": psd_violation,
        "n_compared": len(names),
        "interpretation": interp,
    }


def _hausman_decision(
    b_diff: np.ndarray,
    V_diff: np.ndarray,
    k: int,
    alpha: float,
) -> Dict[str, Any]:
    """Statistic, p-value and FE / RE recommendation from the FE-RE contrast.

    The chi2(k) reference needs ``V_FE - V_RE`` positive definite. When it is
    not, the quadratic form can be negative; the old code clamped it to 0,
    which reported p = 1 and recommended RE -- an unwarranted conclusion from a
    test whose assumptions had failed. A negative statistic is now reported as
    is, with ``pvalue = nan`` and ``recommendation = "inconclusive"``; a
    non-negative one from a non-positive-definite difference keeps its p-value
    but warns.
    """
    try:
        H = float(b_diff @ np.linalg.inv(V_diff) @ b_diff)
    except np.linalg.LinAlgError:
        H = float(b_diff @ np.linalg.pinv(V_diff) @ b_diff)
    min_eig = float(np.linalg.eigvalsh((V_diff + V_diff.T) / 2).min())

    if H < 0:
        warnings.warn(
            f"Hausman statistic is negative (chi2({k}) = {H:.4f}): "
            "V_FE - V_RE is not positive semi-definite on these data, so the "
            "chi2 reference distribution does not apply and the test cannot "
            "choose between FE and RE. Rerun with sigmamore=True (Stata's "
            "`hausman, sigmamore`), or use the cluster-robust Mundlak test: "
            "sp.panel(..., method='mundlak', cluster='entity') and "
            "sp.test(result, '_mean_x1 _mean_x2 ...').",
            AssumptionWarning,
            stacklevel=3,
        )
        return {
            "statistic": H,
            "df": k,
            "pvalue": float("nan"),
            "recommendation": "inconclusive",
            "psd_violation": True,
            "interpretation": (
                f"chi2({k}) = {H:.4f} < 0: the FE-RE variance difference is "
                "not positive semi-definite, so the test is inconclusive."
            ),
        }
    if min_eig <= 0:
        warnings.warn(
            "V_FE - V_RE is not positive definite (smallest eigenvalue "
            f"{min_eig:.3g}); the chi2({k}) p-value of the Hausman test is "
            "unreliable.",
            AssumptionWarning,
            stacklevel=3,
        )
    pvalue = float(stats.chi2.sf(H, k))
    reject = pvalue <= alpha
    detail = (
        "Reject H0: use Fixed Effects."
        if reject
        else "Cannot reject H0: Random Effects is more efficient."
    )
    return {
        "statistic": H,
        "df": k,
        "pvalue": pvalue,
        "recommendation": "FE" if reject else "RE",
        "psd_violation": min_eig <= 0,
        "interpretation": f"chi2({k}) = {H:.4f}, p = {pvalue:.4f}. {detail}",
    }


CausalResult._CITATIONS["hausman"] = (
    "@article{hausman1978specification,\n"
    "  title={Specification Tests in Econometrics},\n"
    "  author={Hausman, Jerry A.},\n"
    "  journal={Econometrica},\n"
    "  volume={46},\n"
    "  number={6},\n"
    "  pages={1251--1271},\n"
    "  year={1978},\n"
    "  publisher={Wiley}\n"
    "}"
)
