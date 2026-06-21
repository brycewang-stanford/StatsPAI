"""
Lee Bounds and Manski Bounds for partial identification.

Lee Bounds (Lee 2009):
  When treatment causes differential sample selection (attrition),
  standard ATE is not point-identified. Lee bounds trim the
  "excess" observations in the group with higher retention to
  produce sharp bounds on the ATE.

  If proportion observed is p1 in treated and p0 in control:
  - Always-observed proportion q = p0/p1 (if p1 > p0)
  - Lower bound: trim top q fraction of treated outcomes
  - Upper bound: trim bottom q fraction of treated outcomes

Manski Bounds (Manski 1990):
  Under no assumptions beyond bounded outcomes, the ATE lies in:
    [E[Y|D=1] - Y_max * P(D=0) - E[Y|D=0] * P(D=0),
     E[Y|D=1] - Y_min * P(D=0) - E[Y|D=0] * P(D=0)]
  Width = Y_max - Y_min (uninformative without further assumptions).

  With monotone treatment response (MTR) or monotone treatment
  selection (MTS), tighter bounds are obtained.

References
----------
Lee, D. S. (2009). "Training, Wages, and Sample Selection:
Estimating Sharp Bounds on Treatment Effects."
Review of Economic Studies, 76(3), 1071-1102. [@lee2009training]

Manski, C. F. (1990). "Nonparametric Bounds on Treatment Effects."
American Economic Review P&P, 80(2), 319-323. [@manski1990nonparametric]

Imbens, G. W. & Manski, C. F. (2004). "Confidence Intervals for Partially
Identified Parameters." Econometrica, 72(6), 1845-1857.
doi:10.1111/j.1468-0262.2004.00555.x.
(Confidence interval for the partially identified parameter used by
``lee_bounds``; refs verified via Crossref and RePEc/IDEAS.)
"""

from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import brentq

from ..core.results import CausalResult


def _imbens_manski_cn(delta: float, sigma_max: float, alpha: float = 0.05) -> float:
    """Imbens & Manski (2004) critical value for a partially identified scalar.

    Solves ``Phi(C_n + delta / sigma_max) - Phi(-C_n) = 1 - alpha`` for ``C_n``,
    where ``delta`` is the width of the estimated identified set and
    ``sigma_max`` the larger of the two endpoint standard errors. The root lies
    in ``[z_{1-alpha}, z_{1-alpha/2}]`` and interpolates between the one-sided
    critical value (wide bounds) and the two-sided one (point identification).

    References
    ----------
    Imbens, G. W. & Manski, C. F. (2004). "Confidence Intervals for Partially
    Identified Parameters." Econometrica, 72(6), 1845-1857.
    doi:10.1111/j.1468-0262.2004.00555.x.
    """
    z_one = float(sp_stats.norm.ppf(1 - alpha))
    z_two = float(sp_stats.norm.ppf(1 - alpha / 2))
    if not np.isfinite(sigma_max) or sigma_max <= 0:
        return z_two if delta <= 0 else z_one
    ratio = float(delta) / float(sigma_max)

    def _eq(c: float) -> float:
        return float(sp_stats.norm.cdf(c + ratio) - sp_stats.norm.cdf(-c) - (1 - alpha))

    if _eq(z_two) <= 0:  # point-identified limit (delta -> 0)
        return z_two
    if _eq(z_one) >= 0:  # wide-bounds limit (delta / sigma_max -> inf)
        return z_one
    return float(brentq(_eq, z_one, z_two, xtol=1e-10))


# ======================================================================
# Lee Bounds
# ======================================================================


def lee_bounds(
    data: pd.DataFrame,
    y: str,
    treat: str,
    selection: str,
    covariates: Optional[List[str]] = None,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> CausalResult:
    """
    Compute Lee (2009) bounds for ATE under sample selection.

    Parameters
    ----------
    data : pd.DataFrame
        Input data (including units with missing outcomes).
    y : str
        Outcome variable (may have NaN for selected-out units).
    treat : str
        Binary treatment variable (0/1).
    selection : str
        Binary selection/retention indicator (1 = observed, 0 = missing).
    covariates : list of str, optional
        Not used in basic Lee bounds, reserved for conditional bounds.
    n_bootstrap : int, default 500
        Bootstrap iterations for inference.
    alpha : float, default 0.05
        Significance level.
    random_state : int, default 42

    Returns
    -------
    CausalResult
        estimate = midpoint of bounds.
        ci = Imbens-Manski confidence interval for the identified set.
        model_info contains 'lower_bound' and 'upper_bound'.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 600
    >>> training = rng.integers(0, 2, size=n)
    >>> # employment (selection) is higher among the treated -> differential attrition
    >>> employed = (rng.uniform(size=n)
    ...             < np.where(training == 1, 0.8, 0.6)).astype(int)
    >>> wage = 10.0 + 2.0 * training + rng.normal(size=n)
    >>> wage = np.where(employed == 1, wage, np.nan)   # wage missing if not employed
    >>> df = pd.DataFrame({'wage': wage, 'training': training, 'employed': employed})
    >>> result = sp.lee_bounds(df, y='wage', treat='training',
    ...                        selection='employed', n_bootstrap=100)
    >>> bool(result.model_info['lower_bound'] <= result.model_info['upper_bound'])
    True
    """
    cols = [treat, selection]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    df = data.copy()
    D = df[treat].values.astype(float)
    S = df[selection].values.astype(float)

    # Retention rates by treatment status
    p1 = np.mean(S[D == 1])  # P(selected | treated)
    p0 = np.mean(S[D == 0])  # P(selected | control)

    if p1 == 0 or p0 == 0:
        raise ValueError("One treatment group has zero retention.")

    # Observed outcomes
    observed = S == 1
    if y not in df.columns:
        raise ValueError(f"Column '{y}' not found in data")

    Y_obs = df.loc[observed, y].values.astype(float)
    D_obs = D[observed]

    Y1 = Y_obs[D_obs == 1]
    Y0 = Y_obs[D_obs == 0]

    lb, ub = _compute_lee_bounds(Y1, Y0, p1, p0)

    # Bootstrap CI
    rng = np.random.RandomState(random_state)
    n = len(D)
    boot_lb = np.zeros(n_bootstrap)
    boot_ub = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        D_b = D[idx]
        S_b = S[idx]

        p1_b = np.mean(S_b[D_b == 1]) if np.sum(D_b == 1) > 0 else p1
        p0_b = np.mean(S_b[D_b == 0]) if np.sum(D_b == 0) > 0 else p0

        obs_b = S_b == 1
        Y_b_col = df[y].values[idx]
        Y_obs_b = Y_b_col[obs_b].astype(float)
        D_obs_b = D_b[obs_b]

        Y1_b = Y_obs_b[D_obs_b == 1]
        Y0_b = Y_obs_b[D_obs_b == 0]

        if len(Y1_b) > 0 and len(Y0_b) > 0 and p1_b > 0 and p0_b > 0:
            boot_lb[b], boot_ub[b] = _compute_lee_bounds(Y1_b, Y0_b, p1_b, p0_b)
        else:
            boot_lb[b], boot_ub[b] = lb, ub

    # Imbens & Manski (2004) confidence interval for the *parameter* (not the
    # whole identified set). The critical value C_n solves
    #     Phi(C_n + Delta / sigma_max) - Phi(-C_n) = 1 - alpha,
    # where Delta = ub - lb is the estimated width of the identified set and
    # sigma_max = max(se_lb, se_ub). C_n interpolates between the one-sided
    # z_{1-alpha} (wide bounds, width >> SE) and the two-sided z_{1-alpha/2}
    # (point-identified, width -> 0). Applying the two-sided z to *both*
    # endpoints instead -- the previous code -- yields the Horowitz-Manski CI
    # that covers the identified SET and therefore over-covers the parameter.
    se_lb = float(np.std(boot_lb, ddof=1))
    se_ub = float(np.std(boot_ub, ddof=1))
    c_n = _imbens_manski_cn(float(ub - lb), max(se_lb, se_ub), alpha)
    ci_lower = float(lb - c_n * se_lb)
    ci_upper = float(ub + c_n * se_ub)

    midpoint = float((lb + ub) / 2)
    se_mid = float((se_lb + se_ub) / 2)

    if se_mid > 0:
        z_stat = midpoint / se_mid
        pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(z_stat))))
    else:
        pvalue = 0.0

    model_info = {
        "lower_bound": float(lb),
        "upper_bound": float(ub),
        "bound_width": float(ub - lb),
        "retention_treated": float(p1),
        "retention_control": float(p0),
        "trimming_fraction": float(abs(p1 - p0) / max(p1, p0)),
        "n_treated_observed": len(Y1),
        "n_control_observed": len(Y0),
    }

    _result = CausalResult(
        method="Lee Bounds (Lee 2009)",
        estimand="ATE (partially identified)",
        estimate=midpoint,
        se=se_mid,
        pvalue=pvalue,
        ci=(ci_lower, ci_upper),
        alpha=alpha,
        n_obs=int(np.sum(observed)),
        detail=None,
        model_info=model_info,
        _citation_key="lee_bounds",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.bounds.lee_bounds",
            params={
                "y": y,
                "treat": treat,
                "selection": selection,
                "covariates": list(covariates) if covariates else None,
                "n_bootstrap": n_bootstrap,
                "alpha": alpha,
                "random_state": random_state,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _compute_lee_bounds(
    Y1: np.ndarray,
    Y0: np.ndarray,
    p1: float,
    p0: float,
) -> tuple[float, float]:
    """Compute Lee bounds given observed outcomes and retention rates."""
    mean_y0 = np.mean(Y0)

    if p1 > p0:
        # Treated group has higher retention => trim treated
        q = p0 / p1  # fraction to keep
        n1 = len(Y1)
        k = int(np.floor(q * n1))
        Y1_sorted = np.sort(Y1)

        # Lower bound: trim from top (keep lowest q fraction)
        lb = np.mean(Y1_sorted[:k]) - mean_y0 if k > 0 else -np.inf

        # Upper bound: trim from bottom (keep highest q fraction)
        ub = np.mean(Y1_sorted[n1 - k :]) - mean_y0 if k > 0 else np.inf
    elif p0 > p1:
        # Control group has higher retention => trim control
        q = p1 / p0
        n0 = len(Y0)
        k = int(np.floor(q * n0))
        Y0_sorted = np.sort(Y0)

        mean_y1 = np.mean(Y1)
        lb = mean_y1 - np.mean(Y0_sorted[n0 - k :]) if k > 0 else -np.inf
        ub = mean_y1 - np.mean(Y0_sorted[:k]) if k > 0 else np.inf
    else:
        # Equal retention: point identified
        lb = np.mean(Y1) - mean_y0
        ub = lb

    return float(lb), float(ub)


# ======================================================================
# Manski Bounds
# ======================================================================


def manski_bounds(
    data: pd.DataFrame,
    y: str,
    treat: str,
    y_lower: Optional[float] = None,
    y_upper: Optional[float] = None,
    assumption: str = "none",
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> CausalResult:
    """
    Compute Manski (1990) worst-case bounds on ATE.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment variable (0/1).
    y_lower : float, optional
        Known lower bound of the outcome. If None, uses observed min.
    y_upper : float, optional
        Known upper bound of the outcome. If None, uses observed max.
    assumption : str, default 'none'
        Additional assumption:
        - 'none': no assumptions (widest bounds)
        - 'mtr': Monotone Treatment Response (Y(1) >= Y(0) for all)
        - 'mts': Monotone Treatment Selection (selection on levels)
    alpha : float, default 0.05
    n_bootstrap : int, default 500
    random_state : int, default 42

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(2)
    >>> n = 600
    >>> training = rng.integers(0, 2, size=n)
    >>> employed = (rng.uniform(size=n)
    ...            < np.where(training == 1, 0.7, 0.5)).astype(int)
    >>> df = pd.DataFrame({'employed': employed, 'training': training})
    >>> result = sp.manski_bounds(df, y='employed', treat='training',
    ...                           y_lower=0, y_upper=1, n_bootstrap=100)
    >>> bool(result.model_info['lower_bound'] <= result.model_info['upper_bound'])
    True
    """
    cols = [y, treat]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    clean = data[cols].dropna()
    Y = clean[y].values.astype(np.float64)
    D = clean[treat].values.astype(np.float64)
    n = len(Y)

    if y_lower is None:
        y_lower = float(Y.min())
    if y_upper is None:
        y_upper = float(Y.max())

    Y1 = Y[D == 1]
    Y0 = Y[D == 0]
    p = np.mean(D)  # P(D=1)

    lb, ub = _compute_manski_bounds(Y1, Y0, p, y_lower, y_upper, assumption)

    # Bootstrap
    rng = np.random.RandomState(random_state)
    boot_lb = np.zeros(n_bootstrap)
    boot_ub = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        Y_b, D_b = Y[idx], D[idx]
        Y1_b = Y_b[D_b == 1]
        Y0_b = Y_b[D_b == 0]
        p_b = np.mean(D_b)

        if len(Y1_b) > 0 and len(Y0_b) > 0:
            boot_lb[b], boot_ub[b] = _compute_manski_bounds(
                Y1_b, Y0_b, p_b, y_lower, y_upper, assumption
            )
        else:
            boot_lb[b], boot_ub[b] = lb, ub

    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    se_lb = np.std(boot_lb, ddof=1)
    se_ub = np.std(boot_ub, ddof=1)

    ci_lower = float(lb - z_crit * se_lb)
    ci_upper = float(ub + z_crit * se_ub)

    midpoint = float((lb + ub) / 2)
    se_mid = float((se_lb + se_ub) / 2)

    if se_mid > 0:
        pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(midpoint / se_mid))))
    else:
        pvalue = 0.0

    model_info = {
        "lower_bound": float(lb),
        "upper_bound": float(ub),
        "bound_width": float(ub - lb),
        "y_lower": y_lower,
        "y_upper": y_upper,
        "assumption": assumption,
        "p_treated": float(p),
        "mean_y_treated": float(np.mean(Y1)),
        "mean_y_control": float(np.mean(Y0)),
    }

    _result = CausalResult(
        method=f"Manski Bounds (assumption={assumption})",
        estimand="ATE (partially identified)",
        estimate=midpoint,
        se=se_mid,
        pvalue=pvalue,
        ci=(ci_lower, ci_upper),
        alpha=alpha,
        n_obs=n,
        detail=None,
        model_info=model_info,
        _citation_key="manski_bounds",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.bounds.manski_bounds",
            params={
                "y": y,
                "treat": treat,
                "y_lower": y_lower,
                "y_upper": y_upper,
                "assumption": assumption,
                "alpha": alpha,
                "n_bootstrap": n_bootstrap,
                "random_state": random_state,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _compute_manski_bounds(
    Y1: np.ndarray,
    Y0: np.ndarray,
    p: float,
    y_lo: float,
    y_hi: float,
    assumption: str,
) -> tuple[float, float]:
    """Compute Manski bounds under given assumption."""
    e1 = np.mean(Y1)  # E[Y|D=1]
    e0 = np.mean(Y0)  # E[Y|D=0]

    if assumption == "none":
        # No-assumption bounds
        lb = (e1 - y_hi) * p + (y_lo - e0) * (1 - p) + (e1 - e0)
        ub = (e1 - y_lo) * p + (y_hi - e0) * (1 - p) + (e1 - e0)
        # Simplified: lb = e1 - e0 - (y_hi - y_lo)*(1 - p) ... etc
        # Actually the standard Manski bounds for ATE:
        lb = p * e1 + (1 - p) * y_lo - (p * y_hi + (1 - p) * e0)
        ub = p * e1 + (1 - p) * y_hi - (p * y_lo + (1 - p) * e0)
    elif assumption == "mtr":
        # Monotone Treatment Response: Y(1) >= Y(0)
        # Tighter: ATE >= 0
        lb_raw = p * e1 + (1 - p) * y_lo - (p * y_hi + (1 - p) * e0)
        ub_raw = p * e1 + (1 - p) * y_hi - (p * y_lo + (1 - p) * e0)
        lb = max(lb_raw, 0)
        ub = ub_raw
    elif assumption == "mts":
        # Monotone Treatment Selection: E[Y(d)|D=1] >= E[Y(d)|D=0]
        # Implies E[Y|D=1] >= E[Y|D=0] under Y(0)
        lb = 0
        ub = e1 - e0
        if ub < 0:
            lb, ub = e1 - e0, 0
    else:
        raise ValueError(f"Unknown assumption: {assumption}")

    return float(lb), float(ub)


# ======================================================================
# Citations
# ======================================================================

CausalResult._CITATIONS["lee_bounds"] = (
    "@article{lee2009training,\n"
    "  title={Training, Wages, and Sample Selection: Estimating Sharp "
    "Bounds on Treatment Effects},\n"
    "  author={Lee, David S},\n"
    "  journal={The Review of Economic Studies},\n"
    "  volume={76},\n"
    "  number={3},\n"
    "  pages={1071--1102},\n"
    "  year={2009},\n"
    "  publisher={Oxford University Press}\n"
    "}"
)

CausalResult._CITATIONS["manski_bounds"] = (
    "@article{manski1990nonparametric,\n"
    "  title={Nonparametric Bounds on Treatment Effects},\n"
    "  author={Manski, Charles F},\n"
    "  journal={The American Economic Review},\n"
    "  volume={80},\n"
    "  number={2},\n"
    "  pages={319--323},\n"
    "  year={1990}\n"
    "}"
)
