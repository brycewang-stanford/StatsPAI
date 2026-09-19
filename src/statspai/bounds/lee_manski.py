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

import warnings
from typing import List, Optional

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
    se_method: str = "bootstrap",
    trimming: str = "quantile",
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
        Reserved for covariate-tightened bounds, which are not
        implemented; a non-empty list is ignored with a ``UserWarning``
        (it used to be ignored silently).
    n_bootstrap : int, default 500
        Bootstrap iterations for inference.
    alpha : float, default 0.05
        Significance level.
    random_state : int, default 42
    se_method : {'bootstrap', 'analytic'}, default 'bootstrap'
        Standard errors of the two bounds. ``'analytic'`` is the
        asymptotic variance of Lee (2009, Proposition 3) exactly as
        Stata's ``leebounds`` (default ``vce(analytic)``) computes it;
        ``'bootstrap'`` resamples units ``n_bootstrap`` times.

    trimming : {'quantile', 'exact'}, default 'quantile'
        How the ``p = |p1 - p0| / max(p1, p0)`` share is trimmed from the
        arm with higher retention. ``'quantile'`` is Lee's (2009) sample
        estimator: keep every retained outcome at or beyond the sample
        ``p``-quantile (Stata ``_pctile`` definition), which is what
        Stata's ``leebounds`` returns for continuous outcomes.
        ``'exact'`` keeps exactly ``(1 - p) n`` units of mass by giving
        the observations tied at the quantile a fractional weight -- the
        branch ``leebounds`` is written to take at a tie (see Notes).

    Notes
    -----
    Before 1.29 StatsPAI kept ``floor((1 - p) n)`` whole observations,
    one fewer than Lee's quantile rule whenever ``p n`` is not an
    integer, so each bound was off by one observation's worth of mass.

    ``leebounds`` (Tauchmann, v1.5) tests ``y == threshold`` against the
    quantile after storing it in a local macro, which does not
    round-trip a double exactly. For continuous outcomes the tie branch
    therefore never fires and ``leebounds`` computes the quantile rule;
    holding the threshold in a scalar instead makes it compute
    ``trimming='exact'``. Both are reproduced exactly by the reference
    tests.
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
    if covariates:
        warnings.warn(
            "lee_bounds: covariate-tightened Lee bounds are not implemented; "
            "`covariates` is ignored and the unconditional bounds are "
            "returned (it used to be ignored silently).",
            UserWarning,
            stacklevel=2,
        )
    if se_method not in ("bootstrap", "analytic"):
        raise ValueError(
            f"se_method must be 'bootstrap' or 'analytic', got {se_method!r}"
        )
    if trimming not in ("quantile", "exact"):
        raise ValueError(f"trimming must be 'quantile' or 'exact', got {trimming!r}")

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

    lb, ub = _compute_lee_bounds(Y1, Y0, p1, p0, trimming)

    if se_method == "analytic":
        se_lb, se_ub = _lee_analytic_se(Y_obs, D_obs, D, S, trimming)
    else:
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
                boot_lb[b], boot_ub[b] = _compute_lee_bounds(
                    Y1_b, Y0_b, p1_b, p0_b, trimming
                )
            else:
                boot_lb[b], boot_ub[b] = lb, ub

        # Imbens & Manski (2004) confidence interval for the *parameter* (not
        # the whole identified set). The critical value C_n solves
        #     Phi(C_n + Delta / sigma_max) - Phi(-C_n) = 1 - alpha,
        # where Delta = ub - lb is the estimated width of the identified set
        # and sigma_max = max(se_lb, se_ub). C_n interpolates between the
        # one-sided z_{1-alpha} (wide bounds, width >> SE) and the two-sided
        # z_{1-alpha/2} (point-identified, width -> 0). Applying the
        # two-sided z to *both* endpoints instead -- the previous code --
        # yields the Horowitz-Manski CI that covers the identified SET and
        # therefore over-covers the parameter.
        se_lb = float(np.std(boot_lb, ddof=1))
        se_ub = float(np.std(boot_ub, ddof=1))
    c_n = _imbens_manski_cn(float(ub - lb), max(se_lb, se_ub), alpha)
    ci_lower = float(lb - c_n * se_lb)
    ci_upper = float(ub + c_n * se_ub)

    midpoint = float((lb + ub) / 2)
    se_mid = float((se_lb + se_ub) / 2)

    if se_mid > 0:
        z_stat = midpoint / se_mid
        pvalue = float(2 * sp_stats.norm.sf(abs(z_stat)))
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
        "se_lower": float(se_lb),
        "se_upper": float(se_ub),
        "imbens_manski_cn": float(c_n),
        "se_method": se_method,
        "trimming": trimming,
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
                "se_method": se_method,
                "trimming": trimming,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _sample_quantile(y_sorted: np.ndarray, prop: float) -> float:
    """Sample ``prop``-quantile with Stata's ``_pctile`` definition.

    With ``P = prop * n``: ``x_(floor(P) + 1)`` when ``P`` is not an
    integer, ``(x_(P) + x_(P + 1)) / 2`` when it is (1-based order
    statistics).
    """
    n = len(y_sorted)
    P = prop * n
    k = int(np.floor(P + 1e-9))
    if abs(P - round(P)) <= 1e-9 * max(1.0, P):
        k = int(round(P))
        if k <= 0:
            return float(y_sorted[0])
        if k >= n:
            return float(y_sorted[-1])
        return float(0.5 * (y_sorted[k - 1] + y_sorted[k]))
    return float(y_sorted[min(k, n - 1)])


def _lee_trimmed(y: np.ndarray, p_trim: float, top: bool, trimming: str) -> tuple:
    """Trimmed mean of ``y`` after removing a ``p_trim`` share.

    ``top=True`` keeps the upper tail (trims the bottom ``p_trim``),
    ``top=False`` the lower tail.

    ``trimming='quantile'`` is Lee's (2009) sample estimator: keep every
    observation at or beyond the sample ``p_trim`` (resp. ``1 - p_trim``)
    quantile. ``trimming='exact'`` keeps exactly ``(1 - p_trim) n`` units
    of mass, giving the observations tied at the quantile the fractional
    weight that makes the kept mass exact.

    Returns ``(mean, threshold, kept_values, kept_weights)``.
    """
    y = np.sort(np.asarray(y, dtype=float))
    n = len(y)
    thr = _sample_quantile(y, p_trim if top else 1.0 - p_trim)
    beyond = y > thr if top else y < thr
    at = y == thr
    if trimming == "quantile" or not at.any():
        keep = beyond | at
        vals = y[keep]
        w = np.ones(len(vals))
    else:
        n_beyond = int(beyond.sum())
        frac = (n * (1.0 - p_trim) - n_beyond) / int(at.sum())
        vals = np.concatenate([y[beyond], y[at]])
        w = np.concatenate([np.ones(n_beyond), np.full(int(at.sum()), frac)])
    mean = float(np.sum(vals * w) / np.sum(w))
    return mean, thr, vals, w


def _lee_analytic_se(
    Y_obs: np.ndarray,
    D_obs: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    trimming: str = "quantile",
) -> tuple:
    """Lee (2009, Proposition 3) standard errors of (lower, upper).

    Transcribes Stata ``leebounds``' ``leesbound`` routine term by term
    (unit weights): ``V = vb1 + (vb2 + vb3 + vc) / N`` with
    ``vp = (1-q)^2 (odds_t / e_t + odds_c / (1 - e_t))``,
    ``vb1`` the variance of the kept outcomes over the kept count (sample
    variance for ``trimming='quantile'``; the fractionally weighted
    population variance over the kept mass for ``'exact'``, Stata's
    tie branch), ``vb2 = (y_q - mu)^2 q / (P(S=1,T=1)(1-q))``,
    ``vb3 = ((y_q - mu) / (1-q))^2 vp`` and ``vc`` the sample variance of
    the untrimmed arm's outcomes over its cell share. When the control
    arm has the higher retention the roles of the arms swap.
    """
    N = len(D)
    p1 = np.mean(S[D == 1])
    p0 = np.mean(S[D == 0])
    if p1 >= p0:
        t = D
        yt, yc = Y_obs[D_obs == 1], Y_obs[D_obs == 0]
    else:
        t = 1 - D
        yt, yc = Y_obs[D_obs == 0], Y_obs[D_obs == 1]
    est = np.mean((S == 1) & (t == 1))
    esnt = np.mean((S == 1) & (t == 0))
    et = np.mean(t == 1)
    oddsc = np.mean((S == 0) & (t == 0)) / esnt
    oddst = np.mean((S == 0) & (t == 1)) / est
    pt, pc = max(p1, p0), min(p1, p0)
    q = (pt - pc) / pt
    vp = (1 - q) ** 2 * (oddst / et + oddsc / (1 - et))
    vc = np.var(yc, ddof=1) / esnt

    def _var(top: bool) -> float:
        mu, thr, vals, w = _lee_trimmed(yt, q, top=top, trimming=trimming)
        mass = np.sum(w)
        if np.all(w == 1.0):
            vb1 = np.var(vals, ddof=1) / mass
        else:
            vb1 = (np.sum(w * vals**2) / mass - mu**2) / mass
        vb2 = (thr - mu) ** 2 * q / (est * (1 - q))
        vb3 = ((thr - mu) / (1 - q)) ** 2 * vp
        return float(vb1 + (vb2 + vb3 + vc) / N)

    # Top-kept mean -> upper bound of the treated-trimmed contrast.
    v_top, v_bottom = _var(True), _var(False)
    if p1 >= p0:
        return float(np.sqrt(v_bottom)), float(np.sqrt(v_top))
    # Control trimmed: lower bound uses the control's top-kept mean.
    return float(np.sqrt(v_top)), float(np.sqrt(v_bottom))


def _compute_lee_bounds(
    Y1: np.ndarray,
    Y0: np.ndarray,
    p1: float,
    p0: float,
    trimming: str = "quantile",
) -> tuple[float, float]:
    """Compute Lee bounds given observed outcomes and retention rates."""
    mean_y0 = np.mean(Y0)

    if p1 > p0:
        # Treated group has higher retention => trim a (p1 - p0) / p1 share
        # of its retained outcomes.
        q = (p1 - p0) / p1
        lb = _lee_trimmed(Y1, q, top=False, trimming=trimming)[0] - mean_y0
        ub = _lee_trimmed(Y1, q, top=True, trimming=trimming)[0] - mean_y0
    elif p0 > p1:
        # Control group has higher retention => trim control
        q = (p0 - p1) / p0
        mean_y1 = np.mean(Y1)
        lb = mean_y1 - _lee_trimmed(Y0, q, top=True, trimming=trimming)[0]
        ub = mean_y1 - _lee_trimmed(Y0, q, top=False, trimming=trimming)[0]
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
        - 'mtr': Monotone Treatment Response (Y(1) >= Y(0) for all):
          ``[0, worst-case upper]``.
        - 'mts': Monotone Treatment Selection with positive selection,
          ``E[Y(d) | D=1] >= E[Y(d) | D=0]`` (Manski & Pepper 2000):
          ``[worst-case lower, E[Y|D=1] - E[Y|D=0]]``.
        - 'mts_mtr': MTS and MTR jointly: ``[0, E[Y|D=1] - E[Y|D=0]]``.
          Raises ``ValueError`` when ``E[Y|D=1] < E[Y|D=0]``: the two
          assumptions are then jointly refuted and the identified set is
          empty.

        Before 1.29, ``'mts'`` returned the ``'mts_mtr'`` interval
        (i.e. silently imposed MTR as well) and, when the naive
        difference was negative, swapped its endpoints instead of
        reporting the empty set. These match the "worst case", "MTS
        positive selection" and "MTS and MTR positive selection" rows of
        Stata's ``tebounds`` at zero misclassification.
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
            try:
                boot_lb[b], boot_ub[b] = _compute_manski_bounds(
                    Y1_b, Y0_b, p_b, y_lower, y_upper, assumption
                )
            except ValueError:
                # 'mts_mtr' refuted on this resample: no bound to record.
                boot_lb[b], boot_ub[b] = np.nan, np.nan
        else:
            boot_lb[b], boot_ub[b] = lb, ub

    n_refuted = int(np.sum(np.isnan(boot_lb)))
    if n_refuted:
        warnings.warn(
            f"manski_bounds: {n_refuted}/{n_bootstrap} bootstrap resamples "
            f"refute assumption={assumption!r} (empty identified set); the "
            "SEs use the remaining resamples.",
            RuntimeWarning,
            stacklevel=2,
        )
        boot_lb = boot_lb[~np.isnan(boot_lb)]
        boot_ub = boot_ub[~np.isnan(boot_ub)]

    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    se_lb = np.std(boot_lb, ddof=1)
    se_ub = np.std(boot_ub, ddof=1)

    ci_lower = float(lb - z_crit * se_lb)
    ci_upper = float(ub + z_crit * se_ub)

    midpoint = float((lb + ub) / 2)
    se_mid = float((se_lb + se_ub) / 2)

    if se_mid > 0:
        pvalue = float(2 * sp_stats.norm.sf(abs(midpoint / se_mid)))
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
        # No-assumption (worst-case) bounds for the ATE.
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
        # Monotone Treatment Selection (positive): E[Y(d)|D=1] >= E[Y(d)|D=0]
        # caps E[Y(1)] at e1 and floors E[Y(0)] at e0, so the upper bound
        # is the naive contrast; the lower bound is the worst case.
        lb = p * e1 + (1 - p) * y_lo - (p * y_hi + (1 - p) * e0)
        ub = e1 - e0
    elif assumption == "mts_mtr":
        # MTS (positive) and MTR (Y(1) >= Y(0)) together: [0, e1 - e0].
        if e1 - e0 < 0:
            raise ValueError(
                "manski_bounds(assumption='mts_mtr'): E[Y|D=1] - E[Y|D=0] = "
                f"{e1 - e0:.6g} < 0, so MTR (ATE >= 0) and positive MTS "
                "(ATE <= naive difference) are jointly refuted by the data; "
                "the identified set is empty."
            )
        lb = 0.0
        ub = e1 - e0
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
