"""
Impact threshold for a confounding variable (Frank 2000).

A regression coefficient is significant; how strong would an omitted
variable have to be to change that? Frank [@frank2000impact] measures the
strength of a confounder by its *impact*, the product of its partial
correlation with the outcome and its partial correlation with the regressor
of interest, both given the controls already in the model. The ITCV is the
smallest impact that moves the coefficient exactly to the significance
threshold. The impacts of the controls that *are* in the model give the
scale: an ITCV several times the largest of them says an omitted variable
would have to matter more than anything observed.

The companion statistic of Frank, Maroulis, Duong and Kelcey
[@frank2013what] asks the same question in units of the data: the share of
the estimate that would have to be bias, and the number of observations
that would have to be replaced by null cases (RIR).

``sp.itcv`` reproduces Stata ``pkonfound`` / R ``konfound::pkonfound`` and
the calculation in Larcker and Rusticus [@larcker2010use].
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def _partial_with_first(M: np.ndarray) -> np.ndarray:
    """Partial correlation of column 0 with each other column, given the
    remaining ones (and a constant)."""
    P = np.linalg.pinv(np.cov(M, rowvar=False))
    d = np.sqrt(np.diag(P))
    return np.asarray(-P[0, 1:] / (d[0] * d[1:]), dtype=float)


def itcv(
    result: Any,
    variable: str,
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Impact threshold for a confounding variable, and RIR.

    Parameters
    ----------
    result : EconometricResults
        A fitted linear regression (``sp.regress`` and relatives). The
        t statistic is taken as reported, so robust or clustered standard
        errors carry through; the derivation is for classical OLS standard
        errors and is an approximation otherwise.
    variable : str
        The coefficient whose inference is in question.
    alpha : float, default 0.05
        Two-sided significance level that defines the threshold.

    Returns
    -------
    dict
        ``itcv``: the threshold impact; the omitted variable needs partial
        correlations of ``sqrt(|itcv|)`` with outcome and regressor
        (``r_threshold``). Positive when the estimate is positive and
        significant: a confounder positively related to both (or
        negatively to both) undoes it. For an estimate that is *not*
        significant the value is the impact that would make it so, and
        ``significant`` is False.
        ``r_obs``: partial correlation of outcome and regressor implied by
        the t statistic; ``r_crit``: the same at the critical value.
        ``beta_threshold``: the coefficient at the threshold;
        ``percent_bias``: share of the estimate that must be bias to reach
        it; ``rir``: that share of the observations.
        ``impacts``: a DataFrame with one row per other regressor (for
        results that keep their design matrix, as ``sp.regress`` does),
        its partial correlations with the outcome (``r_yz``) and with the
        regressor (``r_xz``), and ``impact``, sorted by impact in the
        direction that threatens the inference;
        ``benchmark`` is the largest of them.

    Notes
    -----
    With ``t`` the t statistic and ``df`` the residual degrees of freedom,
    ``r_obs = t / sqrt(df + t^2)``, ``r_crit`` likewise from the critical
    value, and ``itcv = (r_obs - r_crit) / (1 - |r_crit|)`` (denominator
    ``1 + |r_crit|`` for a non-significant estimate).

    ``pkonfound est se n ncov`` uses ``df = n - ncov - 2`` where ``ncov``
    counts the covariates other than the focal one; with the model's own
    residual degrees of freedom the two agree.

    The ITCV is a description of how much confounding it would take, not a
    test. A large value does not show there is no confounder.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"z": rng.normal(size=400)})
    >>> df["x"] = 0.5 * df["z"] + rng.normal(size=400)
    >>> df["y"] = 0.4 * df["x"] + 0.3 * df["z"] + rng.normal(size=400)
    >>> fit = sp.regress("y ~ x + z", df)
    >>> out = sp.itcv(fit, "x")
    >>> bool(out["significant"]) and out["itcv"] > 0
    True
    >>> list(out["impacts"].index)
    ['z']

    References
    ----------
    frank2000impact, frank2013what, larcker2010use
    """
    from scipy import stats

    params = getattr(result, "params", None)
    ses = getattr(result, "std_errors", None)
    if params is None or ses is None or variable not in params.index:
        raise MethodIncompatibility(
            f"itcv: {variable!r} is not a coefficient of this result.",
            recovery_hint="Pass one of result.params.index.",
        )
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(f"itcv: alpha must lie in (0, 1), got {alpha!r}.")
    model_info = getattr(result, "model_info", {}) or {}
    family = model_info.get("family")
    if family and str(family).lower() not in ("gaussian", "normal"):
        raise MethodIncompatibility(
            "itcv: the threshold is defined for a linear regression; this "
            f"result is a {model_info.get('model_type', family)} fit.",
            recovery_hint="Fit the linear model with sp.regress; "
            "sp.sensemakr and sp.evalue cover other cases.",
        )
    data_info = getattr(result, "data_info", {}) or {}
    df = data_info.get("df_resid")
    n = data_info.get("nobs")
    if df is None or not np.isfinite(df) or df <= 0:
        raise MethodIncompatibility(
            "itcv: the result has no finite residual degrees of freedom; the "
            "threshold is defined for a linear regression.",
            recovery_hint="Fit the model with sp.regress.",
        )
    est, se = float(params[variable]), float(ses[variable])
    if not np.isfinite(se) or se <= 0:
        raise MethodIncompatibility(
            f"itcv: the standard error of {variable!r} is not positive."
        )
    df = float(df)
    t = est / se
    sign = 1.0 if est >= 0 else -1.0
    t_crit = sign * float(stats.t.ppf(1.0 - alpha / 2.0, df))
    r_obs = t / np.sqrt(df + t * t)
    r_crit = t_crit / np.sqrt(df + t_crit * t_crit)
    significant = abs(r_obs) > abs(r_crit)
    denom = 1.0 - abs(r_crit) if significant else 1.0 + abs(r_crit)
    value = (r_obs - r_crit) / denom
    beta_threshold = t_crit * se
    if significant:
        percent_bias = 1.0 - beta_threshold / est
    else:
        percent_bias = 1.0 - est / beta_threshold
    out: Dict[str, Any] = {
        "variable": variable,
        "alpha": alpha,
        "estimate": est,
        "se": se,
        "t": t,
        "df": df,
        "significant": bool(significant),
        "r_obs": float(r_obs),
        "r_crit": float(r_crit),
        "itcv": float(value),
        "r_threshold": float(np.sqrt(abs(value))),
        "beta_threshold": float(beta_threshold),
        "percent_bias": float(percent_bias),
        "rir": None if n is None else int(np.ceil(percent_bias * float(n))),
    }

    X_stored = data_info.get("X")
    y_stored = data_info.get("y")
    var_names = data_info.get("var_names")
    if X_stored is not None and y_stored is not None and var_names is not None:
        X_df = pd.DataFrame(
            np.asarray(X_stored, dtype=float), columns=[str(v) for v in var_names]
        )
        y_df = np.asarray(y_stored, dtype=float)
        controls = [
            c
            for c in X_df.columns
            if c != variable and c != "Intercept" and X_df[c].std() > 0
        ]
    else:
        controls = []
    if controls:
        Z = X_df[controls].to_numpy(dtype=float)
        y = np.asarray(y_df, dtype=float).reshape(len(X_df), -1)[:, 0]
        x = X_df[variable].to_numpy(dtype=float)
        r_yz = _partial_with_first(np.column_stack([y, Z]))
        r_xz = _partial_with_first(np.column_stack([x, Z]))
        impacts = pd.DataFrame(
            {"r_yz": r_yz, "r_xz": r_xz, "impact": r_yz * r_xz},
            index=pd.Index([str(c) for c in controls], name="control"),
        )
        impacts = impacts.sort_values("impact", ascending=sign < 0)
        out["impacts"] = impacts
        benchmark = float(impacts["impact"].iloc[0])
        out["benchmark"] = benchmark
        out["benchmark_control"] = str(impacts.index[0])
    return out
