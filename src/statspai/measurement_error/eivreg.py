"""Errors-in-variables regression with assumed reliability ratios.

The model is ``y = X* b + e`` with the regressors observed as
``X = X* + U``, ``U`` independent of ``e`` and of ``X*`` with a diagonal
covariance. The reliability of regressor ``j`` is
``r_j = var(x*_j) / var(x_j)``, so the error variance is ``(1 - r_j)``
times the observed variance. Removing it from the cross-product matrix
undoes the attenuation of least squares::

    A = X'WX - S,   S = diag((1 - r_j) (N - 1) s_j^2),   b = A^{-1} X'Wy

with ``s_j^2`` the (weighted) sample variance of ``x_j`` and ``N`` the
(weighted) number of observations. The covariance is the M-estimation
sandwich ``A^{-1} H'WH A^{-1}`` whose ``i``-th row of ``H`` is
``e_i x_i + (x_i - xbar)^2 (1 - r) b`` (elementwise), with ``W`` squared in
the middle under analytic weights. This is Stata's ``eivreg``, and the
numbers agree with it to rounding (``tests/reference_parity``).

References
----------
[@fuller1987measurement]
[@stefanski2002calculus]
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, IdentificationFailure, MethodIncompatibility

__all__ = ["eivreg"]


def _design(
    formula: str, data: pd.DataFrame, function: str = "sp.eivreg"
) -> Dict[str, Any]:
    """Response, design matrix and term names of ``formula`` on ``data``.

    ``data`` must have a default RangeIndex, so the sample index the
    regression reports is positional.
    """
    import warnings

    from ..regression.ols import regress

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = regress(formula, data=data)
    info = ols.data_info
    omitted = (ols.model_info or {}).get("omitted") or []
    if omitted:
        dropped = [o.get("variable") if isinstance(o, dict) else o for o in omitted]
        raise MethodIncompatibility(
            f"{function}: the regressors are collinear: {dropped} would be " "omitted.",
            recovery_hint="Drop the redundant regressors from the formula.",
            diagnostics={"omitted": omitted},
        )
    return {
        "X": np.asarray(info["X"], dtype=float),
        "y": np.asarray(info["y"], dtype=float),
        "names": [str(v) for v in info["var_names"]],
        "rows": np.asarray(list(info["sample_index"]), dtype=int),
        "depvar": info.get("dependent_var"),
    }


def eivreg(
    formula: str,
    data: pd.DataFrame,
    reliability: Optional[Mapping[str, float]] = None,
    weights: Optional[str] = None,
    *,
    weights_as: Literal["analytic", "frequency"] = "analytic",
    alpha: float = 0.05,
) -> EconometricResults:
    """Linear regression corrected for measurement error in the regressors.

    Each regressor named in ``reliability`` is assumed to be observed with
    classical (additive, independent) error whose share of the observed
    variance is ``1 - reliability``. The estimator removes that variance
    from the cross-product matrix before solving the normal equations,
    which undoes the attenuation of least squares. It is Stata's
    ``eivreg``.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2 + z"``. The model must have an intercept. Rows with a
        missing value in a model variable are dropped.
    data : pandas.DataFrame
        The data.
    reliability : dict, optional
        ``{"x1": 0.8, "x2": 0.75}``: the reliability ratio
        ``var(true x) / var(observed x)`` of each error-prone regressor, in
        ``(0, 1]``. Keys are term names as they appear in the result.
        Regressors not listed have reliability 1 (no error). Without it the
        result is ordinary least squares with the variance below.
    weights : str, optional
        Column of non-negative weights. Rows with weight zero are dropped.
    weights_as : {'analytic', 'frequency'}, default 'analytic'
        How to read ``weights``. ``'analytic'`` is Stata's ``[aw=]``: the
        weights are rescaled to sum to the number of rows. ``'frequency'``
        is ``[fw=]``: integer counts of identical rows, which also set the
        sample size and the degrees of freedom.
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    EconometricResults
        Corrected coefficients, sandwich standard errors and t tests with
        ``N - k`` degrees of freedom. ``diagnostics`` holds the Wald
        ``F-statistic`` of the slopes, ``R-squared`` and ``Root MSE`` as
        Stata defines them for ``eivreg``; ``model_info['reliability']``
        holds the reliability of every term.

    Raises
    ------
    IdentificationFailure
        If a reliability is so small that the corrected second-moment
        matrix of the regressors and the outcome is not positive definite
        (for a single error-prone regressor: below the R-squared of that
        regressor on the other regressors and ``y``). The data are then
        inconsistent with the assumed error variance.

    Notes
    -----
    The correction needs the reliability ratio from outside the data (a
    validation study, test-retest correlation, Cronbach's alpha of a
    scale). With a guessed value the result is a sensitivity analysis, not
    an estimate; try a range of values.

    The variance treats the reliability ratios as known. It follows the
    M-estimation formulas of [@stefanski2002calculus] used by Stata, with no
    small-sample factor.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> x_true = rng.normal(size=n)
    >>> df = pd.DataFrame({"x": x_true + rng.normal(scale=0.5, size=n)})
    >>> df["y"] = 1 + 2 * x_true + rng.normal(size=n)
    >>> naive = sp.regress("y ~ x", data=df).params["x"]
    >>> fit = sp.eivreg("y ~ x", df, reliability={"x": 0.8})
    >>> bool(abs(fit.params["x"] - 2) < abs(naive - 2))
    True

    References
    ----------
    [@fuller1987measurement]
    [@stefanski2002calculus]
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "sp.eivreg: data must be a pandas DataFrame.",
            recovery_hint="Pass the data as a DataFrame.",
        )
    reliability = dict(reliability or {})
    key = str(weights_as).lower()
    if key not in ("analytic", "frequency"):
        raise MethodIncompatibility(
            f"sp.eivreg: weights_as={weights_as!r} is not 'analytic' or "
            "'frequency'. Stata's eivreg has no sampling-weight variance.",
            recovery_hint="Use weights_as='analytic' ([aw=]) or 'frequency' "
            "([fw=]); for a sampling-weight variance bootstrap the estimator.",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"sp.eivreg: alpha={alpha!r} is not in (0, 1).",
            recovery_hint="Use alpha=0.05 for 95% intervals.",
        )

    df = data.reset_index(drop=True)
    if weights is not None:
        check_columns(df, {"weights": weights}, function="sp.eivreg")
        w_all = pd.to_numeric(df[weights], errors="coerce").to_numpy(dtype=float)
        if np.any(w_all[np.isfinite(w_all)] < 0):
            raise MethodIncompatibility(
                "sp.eivreg: negative weights.",
                recovery_hint="Weights must be non-negative.",
            )
        keep = np.isfinite(w_all) & (w_all > 0)
        df = df.loc[keep].reset_index(drop=True)

    d = _design(formula, df)
    X, y, names = d["X"], d["y"], d["names"]
    n, k = X.shape
    if weights is not None:
        w = pd.to_numeric(df[weights], errors="coerce").to_numpy(dtype=float)[d["rows"]]
        if key == "frequency" and not np.allclose(w, np.round(w)):
            raise MethodIncompatibility(
                "sp.eivreg: frequency weights must be integers.",
                recovery_hint="Use weights_as='analytic' for non-integer weights.",
            )
        if key == "analytic":
            w = w * n / w.sum()
    else:
        w = np.ones(n)
    n_eff = float(w.sum())

    const = [j for j in range(k) if np.ptp(X[:, j]) == 0 and X[0, j] != 0]
    if not const:
        raise MethodIncompatibility(
            "sp.eivreg: the model has no intercept. The correction measures "
            "the error variance around the mean, which needs one.",
            recovery_hint="Drop '- 1' / '0 +' from the formula.",
        )
    if n_eff - k <= 0 or n <= k:
        raise DataInsufficient(
            f"sp.eivreg: {n} observations for {k} coefficients.",
            recovery_hint="Use fewer regressors or more data.",
        )
    if np.linalg.matrix_rank(X) < k:
        raise MethodIncompatibility(
            "sp.eivreg: the regressors are collinear.",
            recovery_hint="Drop the redundant regressors (sp.regress reports "
            "which ones it omits).",
        )

    unknown = sorted(set(reliability) - set(names))
    if unknown:
        raise MethodIncompatibility(
            f"sp.eivreg: reliability names {unknown}, which are not terms of "
            f"the model {names}.",
            recovery_hint="Key reliability by the term names of the formula.",
            diagnostics={"terms": names},
        )
    r = np.ones(k)
    for name, value in reliability.items():
        try:
            val = float(value)
        except (TypeError, ValueError):
            val = float("nan")
        if not 0 < val <= 1:
            raise MethodIncompatibility(
                f"sp.eivreg: reliability of {name!r} is {value!r}; it must be "
                "in (0, 1].",
                recovery_hint="A reliability ratio is var(true x) / var(observed "
                "x); 1 means no measurement error.",
            )
        j = names.index(name)
        if j in const:
            raise MethodIncompatibility(
                "sp.eivreg: the intercept cannot have a reliability.",
                recovery_hint="Remove the intercept from reliability.",
            )
        r[j] = val

    mean = (w @ X) / n_eff
    dev = X - mean
    var = (w @ dev**2) / (n_eff - 1)
    S = np.diag((1.0 - r) * (n_eff - 1.0) * var)
    XtWX = X.T @ (w[:, None] * X)
    A = XtWX - S

    # The corrected second moments of (x, y) must form a covariance
    # matrix; otherwise the assumed error variance exceeds what the data
    # allow (Stata: "reliability r() too small").
    slopes = [j for j in range(k) if j not in const]
    Z = np.column_stack([dev[:, slopes], y - (w @ y) / n_eff])
    M = Z.T @ (w[:, None] * Z) - np.diag(np.append(np.diag(S)[slopes], 0.0))
    scale = np.sqrt(np.clip(np.diag(Z.T @ (w[:, None] * Z)), 1e-300, None))
    min_eig = float(np.linalg.eigvalsh(M / np.outer(scale, scale)).min())
    if min_eig <= 1e-12:
        raise IdentificationFailure(
            "sp.eivreg: the reliability is too small for these data: the "
            "corrected covariance matrix of the regressors and the outcome is "
            "not positive definite.",
            recovery_hint="Use a larger reliability. A reliability must exceed "
            "the R-squared of the error-prone regressor on the other "
            "regressors and the outcome.",
            diagnostics={"min_eigenvalue": min_eig, "reliability": reliability},
        )

    b = np.linalg.solve(A, X.T @ (w * y))
    e = y - X @ b
    H = e[:, None] * X + dev**2 * ((1.0 - r) * b)
    A_inv = np.linalg.inv(A)
    middle = w**2 if key == "analytic" else w
    cov = A_inv @ (H.T @ (middle[:, None] * H)) @ A_inv
    cov = (cov + cov.T) / 2.0
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    df_resid = n_eff - k
    rss = float((w * y) @ y - b @ A @ b)
    ybar = float(w @ y) / n_eff
    tss = float(w @ (y - ybar) ** 2)
    b_s = b[slopes]
    if slopes:
        f_stat = float(b_s @ np.linalg.solve(cov[np.ix_(slopes, slopes)], b_s)) / len(
            slopes
        )
        f_p = float(stats.f.sf(f_stat, len(slopes), df_resid))
    else:
        f_stat, f_p = float("nan"), float("nan")
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    index = pd.Index(names)
    model_info: Dict[str, Any] = {
        "model_type": "Errors-in-variables regression",
        "reliability": pd.Series(r, index=index),
        "weights": weights,
        "weights_as": key if weights is not None else None,
        "robust": "eiv-sandwich",
        "alpha": alpha,
    }
    data_info: Dict[str, Any] = {
        "nobs": int(round(n_eff)),
        "df_model": len(slopes),
        "df_resid": df_resid,
        "dependent_var": d["depvar"],
        "var_names": names,
        "var_cov": cov,
        "X": X,
        "y": y,
        "residuals": e,
        "fitted_values": X @ b,
        "rss": rss,
        "tss": tss,
    }
    errors: List[str] = [nm for nm, rr in zip(names, r) if rr < 1]
    diagnostics: Dict[str, Any] = {
        "F-statistic": f_stat,
        "Prob (F-statistic)": f_p,
        "R-squared": r2,
        "Root MSE": float(np.sqrt(max(rss, 0.0) / df_resid)),
        "error_prone_terms": errors,
    }
    return EconometricResults(
        params=pd.Series(b, index=index),
        std_errors=pd.Series(se, index=index),
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
