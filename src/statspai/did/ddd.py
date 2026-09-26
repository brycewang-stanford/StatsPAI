"""
Triple Differences (DDD) estimator.

Extends the standard 2×2 DID by adding a third difference — a within-unit
subgroup that should *not* be affected by treatment — to net out additional
confounders beyond what parallel trends can handle.

Model
-----
Y_{itsg} = α + β₁·D_s + β₂·T_t + β₃·G_g
         + β₄·D×T + β₅·D×G + β₆·T×G
         + δ·D×T×G + X'γ + ε

where:
    D_s = treatment group indicator
    T_t = post-treatment period indicator
    G_g = affected subgroup indicator
    δ   = DDD estimate (the triple interaction)

References
----------
Gruber, J. (1994). "The Incidence of Mandated Maternity Benefits."
*American Economic Review*, 84(3), 622-641.

Olden, A. and Møen, J. (2022).
"The Triple Difference Estimator."
*The Econometrics Journal*, 25(3), 531-553. [@olden2022triple]
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import AssumptionWarning, MethodIncompatibility
from ._core import require_bool

_METHODS = ("3wfe", "dr", "reg", "ipw")


def _delegate_to_heterogeneous(
    data: pd.DataFrame,
    *,
    y: str,
    treat: str,
    time: str,
    subgroup: str,
    covariates: Optional[List[str]],
    id: Optional[str],
    method: str,
    alpha: float,
    weights: Optional[str],
    cluster: Optional[str],
) -> CausalResult:
    """Run the 2x2x2 design through the heterogeneity-robust DDD estimator.

    ``sp.ddd_heterogeneous`` is written for staggered panels, but a two-period
    design with a single treated cohort is the special case it collapses to,
    and it is the path that carries the conditional-parallel-trends estimators
    (and their R ``triplediff`` parity).  The translation is one column: the
    cohort is the post period for the treated group and ``never_value`` for
    everyone else.
    """
    from .ddd_heterogeneous import ddd_heterogeneous

    if id is None:
        raise MethodIncompatibility(
            f"method={method!r} pairs each unit's two periods, so it needs "
            "id=<unit column>. Use method='3wfe' for the pooled "
            "triple-interaction regression, which does not.",
            diagnostics={"method": method},
        )
    if weights is not None:
        raise MethodIncompatibility(
            f"method={method!r} does not take weights= yet; use "
            "method='3wfe', or sp.ddd_heterogeneous directly.",
            diagnostics={"method": method, "weights": weights},
        )
    if cluster is not None:
        raise MethodIncompatibility(
            f"method={method!r} clusters on id= by construction -- its "
            "influence functions are summed within unit -- so a separate "
            f"cluster={cluster!r} would be ignored. Drop it, or use "
            "method='3wfe', which honours it.",
            diagnostics={"method": method, "cluster": cluster},
        )
    periods = sorted(pd.Series(data[time]).dropna().unique())
    if len(periods) != 2:
        raise MethodIncompatibility(
            f"method={method!r} expects the two-period design sp.ddd is for; "
            f"`{time}` takes {len(periods)} values. For staggered adoption "
            "call sp.ddd_heterogeneous, which is built for it.",
            diagnostics={"n_periods": len(periods)},
        )
    treat_vals = sorted(pd.Series(data[treat]).dropna().unique())
    if len(treat_vals) != 2:
        raise MethodIncompatibility(
            f"`{treat}` must be a two-valued group indicator; got "
            f"{len(treat_vals)} values.",
            diagnostics={"n_treat_values": len(treat_vals)},
        )
    post = periods[1]
    try:
        post_num = float(post)
        period_num = pd.to_numeric(data[time], errors="raise")
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"method={method!r} needs a numeric `{time}`, because the cohort "
            "it builds is the post period itself; recode the two periods as "
            "numbers (e.g. 0/1).",
            diagnostics={"time": time},
        ) from exc
    # A sentinel that cannot collide with either period, whatever they are.
    never = float(min(float(v) for v in periods)) - 1.0
    # Two distinct values of `treat` already guarantee a comparison group.
    is_treated = data[treat] == treat_vals[1]
    frame = data.copy()
    frame[time] = period_num
    frame["__cohort__"] = np.where(is_treated, post_num, never)
    res = ddd_heterogeneous(
        frame,
        y=y,
        unit=id,
        time=time,
        cohort="__cohort__",
        subgroup=subgroup,
        never_value=never,
        x=list(covariates) if covariates else None,
        est_method=method,
        alpha=alpha,
    )
    res.model_info["ddd_method"] = method
    res.model_info["ddd_delegated_from"] = "sp.ddd"
    return res


@accepts_aliases(_strict=True, controls="covariates")
def ddd(
    data: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    subgroup: str,
    covariates: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    robust: bool = True,
    alpha: float = 0.05,
    weights: Optional[str] = None,
    *,
    id: Optional[str] = None,
    method: str = "3wfe",
) -> CausalResult:
    """
    Triple Differences (DDD) estimator.

    Uses a within-treatment-group subgroup that is unaffected by treatment
    to eliminate confounders beyond what standard DID parallel trends
    can handle.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    treat : str
        Binary treatment group indicator (0/1).
    time : str
        Binary time period indicator (0 = pre, 1 = post).
    subgroup : str
        Binary affected-subgroup indicator (1 = affected by treatment,
        0 = unaffected within-unit comparison group).
        E.g. low-wage workers (affected) vs high-wage workers (unaffected)
        in a minimum wage study.
    covariates : list of str, optional
        Additional control variables, entered additively in the
        triple-interaction regression. This identifies the DDD ATT only
        when the covariates' effect on the outcome is common to all eight
        (treat, time, subgroup) cells and their distribution does not
        shift across cells; otherwise the triple-interaction coefficient
        is not the covariate-adjusted DDD ATT (Ortiz-Villavicencio and
        Sant'Anna 2025, [@ortiz2025better]). A ``UserWarning`` says so and
        the caveat is recorded in ``model_info['diagnostics']``. For
        conditional-parallel-trends DDD use
        ``sp.ddd_heterogeneous(..., x=covariates, est_method='dr')``,
        the doubly robust estimator of that paper.
    cluster : str, optional
        Cluster variable for cluster-robust standard errors.
    robust : bool, default True
        Use HC1 heteroskedasticity-robust standard errors.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    weights : str, optional
        Column name for analytical weights (e.g. population weights).
        Equivalent to Stata's ``[aweight=...]``.
    id : str, optional
        Unit identifier. Required by every ``method`` other than
        ``'3wfe'``, which needs to pair each unit's two periods.

        .. versionadded:: 1.31.0
    method : {'3wfe', 'dr', 'reg', 'ipw'}, default '3wfe'
        ``'3wfe'`` is the triple-interaction regression described above, and
        is what this function has always computed. The other three hand the
        design to :func:`sp.ddd_heterogeneous` -- the doubly robust,
        outcome-regression and inverse-probability-weighting estimators of
        Ortiz-Villavicencio and Sant'Anna (2025), pinned against R
        ``triplediff`` -- which is the covariate adjustment that identifies
        the DDD ATT under conditional parallel trends. They need ``id=``.

        Without covariates the point estimate is the same either way (the
        cell means are the same object); the standard errors are not,
        because the panel route differences within unit while ``'3wfe'``
        treats the two periods as independent cross-sections.

        .. versionadded:: 1.31.0

    Returns
    -------
    CausalResult
        Results with DDD estimate (triple interaction coefficient),
        standard errors, and full coefficient table.

    Examples
    --------
    Minimum-wage style example — treatment: NJ vs control states;
    time: pre vs post; subgroup: low-wage (affected) vs high-wage
    (unaffected). The triple interaction recovers the planted effect.

    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> nj = rng.integers(0, 2, n)
    >>> post = rng.integers(0, 2, n)
    >>> low_wage = rng.integers(0, 2, n)
    >>> employment = (50 + 3 * nj - 1 * post + 2 * low_wage
    ...               - 2 * (nj * post * low_wage) + rng.normal(0, 3, n))
    >>> state = np.where(nj == 1, rng.choice(['NJ', 'NY', 'CT'], n),
    ...                  rng.choice(['PA', 'OH', 'MD'], n))
    >>> df = pd.DataFrame({'employment': employment, 'nj': nj,
    ...                    'post': post, 'low_wage': low_wage,
    ...                    'state': state})
    >>> result = sp.ddd(df, y='employment', treat='nj', time='post',
    ...                 subgroup='low_wage')
    >>> bool(np.isfinite(result.estimate))
    True

    With cluster-robust standard errors by state:

    >>> result = sp.ddd(df, y='employment', treat='nj', time='post',
    ...                 subgroup='low_wage', cluster='state')
    >>> bool(np.isfinite(result.se))
    True
    """
    robust = require_bool(robust, argument="robust")
    if method not in _METHODS:
        raise MethodIncompatibility(
            f"method must be one of {_METHODS}; got {method!r}.",
            diagnostics={"method": method},
        )
    if method != "3wfe":
        return _delegate_to_heterogeneous(
            data,
            y=y,
            treat=treat,
            time=time,
            subgroup=subgroup,
            covariates=covariates,
            id=id,
            method=method,
            alpha=alpha,
            weights=weights,
            cluster=cluster,
        )

    df = data.copy()
    diagnostics = []
    if covariates:
        msg = (
            "sp.ddd: covariates enter the triple-interaction regression "
            "additively, which identifies the covariate-adjusted DDD ATT only "
            "if their outcome effect is common across all eight cells and "
            "their distribution does not differ across cells "
            "(Ortiz-Villavicencio & Sant'Anna 2025). For conditional parallel "
            "trends pass method='dr' with id=<unit column>, which hands the "
            "design to sp.ddd_heterogeneous."
        )
        warnings.warn(msg, AssumptionWarning, stacklevel=2)
        diagnostics.append(
            {"check": "ddd_additive_covariates", "status": "warn", "message": msg}
        )

    # Validate binary variables
    for col, label in [(treat, "Treatment"), (time, "Time"), (subgroup, "Subgroup")]:
        vals = sorted(df[col].dropna().unique())
        if len(vals) != 2:
            raise ValueError(
                f"{label} variable '{col}' must have exactly 2 values, "
                f"got {len(vals)}: {vals}"
            )

    # Ensure 0/1 coding
    treat_vals = sorted(df[treat].dropna().unique())
    time_vals = sorted(df[time].dropna().unique())
    sub_vals = sorted(df[subgroup].dropna().unique())

    d = (df[treat] == treat_vals[1]).astype(float).values
    t = (df[time] == time_vals[1]).astype(float).values
    g = (df[subgroup] == sub_vals[1]).astype(float).values
    y_arr = df[y].values.astype(float)

    # Two-way interactions
    dt = d * t
    dg = d * g
    tg = t * g

    # Triple interaction = DDD coefficient
    dtg = d * t * g

    # Drop rows with NaN (and invalid weights if provided)
    valid = np.isfinite(y_arr)
    if covariates:
        for cov in covariates:
            valid &= np.isfinite(df[cov].values.astype(float))
    if weights is not None:
        valid &= np.isfinite(df[weights].values.astype(float))
        valid &= df[weights].values.astype(float) > 0

    d, t, g = d[valid], t[valid], g[valid]
    dt, dg, tg, dtg = dt[valid], dg[valid], tg[valid], dtg[valid]
    y_arr = y_arr[valid]

    # Build design matrix
    X_parts = [np.ones(len(y_arr)), d, t, g, dt, dg, tg, dtg]
    X_names = [
        "const",
        treat,
        time,
        subgroup,
        f"{treat}x{time}",
        f"{treat}x{subgroup}",
        f"{time}x{subgroup}",
        f"{treat}x{time}x{subgroup}",
    ]

    if covariates:
        for cov in covariates:
            X_parts.append(
                df.loc[valid, cov].values.astype(float)
                if isinstance(valid, np.ndarray)
                else df[cov].values.astype(float)
            )
            X_names.append(cov)

    X = np.column_stack(X_parts)
    n, k = X.shape

    # --- Analytical weights (WLS) ---
    if weights is not None:
        w_raw = (
            df.loc[valid, weights].values.astype(float)
            if isinstance(valid, np.ndarray)
            else df[weights].values.astype(float)
        )
        if np.any(w_raw < 0):
            raise ValueError(f"Weights column '{weights}' contains negative values.")
        w = w_raw * (n / w_raw.sum())
        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, np.newaxis]
        yw = y_arr * sqrt_w
    else:
        w = None
        Xw = X
        yw = y_arr

    # OLS on (possibly weighted) data
    try:
        XtX_inv = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)

    beta = XtX_inv @ Xw.T @ yw
    resid = y_arr - X @ beta

    # Variance-covariance
    if cluster is not None:
        cl = (
            df.loc[valid, cluster].values
            if isinstance(valid, np.ndarray)
            else df[cluster].values
        )
        unique_cl = np.unique(cl)
        n_cl = len(unique_cl)
        meat = np.zeros((k, k))
        for c_val in unique_cl:
            idx = cl == c_val
            if w is not None:
                score = (Xw[idx] * (sqrt_w[idx] * resid[idx])[:, np.newaxis]).sum(
                    axis=0
                )
            else:
                score = (X[idx] * resid[idx, np.newaxis]).sum(axis=0)
            meat += np.outer(score, score)
        correction = (n_cl / (n_cl - 1)) * ((n - 1) / (n - k))
        vcov = correction * XtX_inv @ meat @ XtX_inv
    elif robust:
        if w is not None:
            # ⚠️ correctness fix (2026-07): the WLS score is w_i x_i e_i, so
            # the HC1 meat is Σ w_i² e_i² x_i x_i' (Stata aweight-robust /
            # R sandwich convention); see the same fix in did_2x2.py.
            hc1_weights = (n / (n - k)) * (w**2 * resid**2)
        else:
            hc1_weights = (n / (n - k)) * resid**2
        meat = (X * (hc1_weights)[:, None]).T @ X
        vcov = XtX_inv @ meat @ XtX_inv
    else:
        if w is not None:
            sigma2 = np.sum(w * resid**2) / (n - k)
        else:
            sigma2 = np.sum(resid**2) / (n - k)
        vcov = sigma2 * XtX_inv

    se = np.sqrt(np.diag(vcov))

    # DDD coefficient is the triple interaction
    ddd_idx = X_names.index(f"{treat}x{time}x{subgroup}")
    estimate = float(beta[ddd_idx])
    est_se = float(se[ddd_idx])
    t_stat = estimate / est_se if est_se > 0 else np.nan
    df_resid = n - k
    pvalue = float(2 * stats.t.sf(abs(t_stat), df_resid))
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    ci = (estimate - t_crit * est_se, estimate + t_crit * est_se)

    # Also extract the DID coefficient (for comparison)
    did_idx = X_names.index(f"{treat}x{time}")
    did_estimate = float(beta[did_idx])

    # Full coefficient table
    t_stats_all = beta / se
    pvals_all = 2 * stats.t.sf(np.abs(t_stats_all), df_resid)
    detail = pd.DataFrame(
        {
            "variable": X_names,
            "coefficient": beta,
            "se": se,
            "tstat": t_stats_all,
            "pvalue": pvals_all,
        }
    )

    # R-squared (weighted if applicable)
    if w is not None:
        y_wmean = np.sum(w * y_arr) / np.sum(w)
        tss = np.sum(w * (y_arr - y_wmean) ** 2)
        rss = np.sum(w * resid**2)
    else:
        tss = np.sum((y_arr - np.mean(y_arr)) ** 2)
        rss = np.sum(resid**2)
    r_squared = 1 - rss / tss if tss > 0 else 0.0

    model_info = {
        "r_squared": round(r_squared, 6),
        "n_obs": n,
        "n_treated": int(d.sum()),
        "n_control": int((1 - d).sum()),
        "n_subgroup": int(g.sum()),
        "n_comparison": int((1 - g).sum()),
        "did_estimate": round(did_estimate, 6),
        "robust_se": robust,
        "cluster": cluster,
        "weights": weights,
        "diagnostics": diagnostics,
    }

    _result = CausalResult(
        method="Triple Differences (DDD)",
        estimand="ATT",
        estimate=estimate,
        se=est_se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key="ddd",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.ddd",
            params={
                "y": y,
                "treat": treat,
                "time": time,
                "subgroup": subgroup,
                "covariates": list(covariates) if covariates else None,
                "cluster": cluster,
                "robust": robust,
                "alpha": alpha,
                "weights": weights,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
