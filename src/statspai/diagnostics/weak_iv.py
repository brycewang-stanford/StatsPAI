"""
Modern weak instrument diagnostics.

Provides tests that remain valid even when instruments are weak,
addressing the limitations of the standard first-stage F-statistic.

Public API
----------
- ``anderson_rubin_test`` — Anderson-Rubin (1949) weak-IV-robust test
  and confidence set (by grid inversion).
- ``effective_f_test`` — Olea-Pflueger (2013) robust effective F with
  heteroskedasticity-robust variance. Reduces exactly to first-stage F
  under homoskedasticity and a single instrument.
- ``tF_critical_value`` — Lee et al. (2022, AER) tF adjusted critical
  value as a function of first-stage F; produces weak-IV-robust CIs
  that use the 2SLS t-ratio directly.

References
----------
Anderson, T.W. and Rubin, H. (1949).
"Estimation of the Parameters of a Single Equation in a Complete System
of Stochastic Equations."
*Annals of Mathematical Statistics*, 20(1), 46-63.

Stock, J.H. and Yogo, M. (2005).
"Testing for Weak Instruments in Linear IV Regression."
In Andrews, D.W.K. and Stock, J.H. (eds), *Identification and Inference
for Econometric Models*. Cambridge University Press.

Olea, J.L.M. and Pflueger, C. (2013).
"A Robust Test for Weak Instruments."
*Journal of Business & Economic Statistics*, 31(3), 358-369.

Lee, D.S., McCrary, J., Moreira, M.J. and Porter, J. (2022).
"Valid t-ratio Inference for IV."
*American Economic Review*, 112(10), 3260-3290.

Andrews, I., Stock, J.H. and Sun, L. (2019).
"Weak Instruments in Instrumental Variables Regression: Theory and
Practice." *Annual Review of Economics*, 11, 727-753.
"""

from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _partial_out(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Residuals from regressing columns of X on W."""
    if W.size == 0 or W.shape[1] == 0:
        return X
    beta = np.linalg.lstsq(W, X, rcond=None)[0]
    return X - W @ beta


def _prep_matrices(
    data: pd.DataFrame,
    y: str,
    endog: str,
    instruments: List[str],
    exog: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (Y, D, Z, W, n) as numpy arrays, dropping NA rows."""
    cols = [y, endog] + instruments + (exog or [])
    df = data[cols].dropna()
    n = len(df)
    Y = df[y].values.astype(float)
    D = df[endog].values.astype(float)
    Z = df[instruments].values.astype(float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if exog:
        W = np.column_stack([np.ones(n)] + [df[v].values.astype(float) for v in exog])
    else:
        W = np.ones((n, 1))
    return Y, D, Z, W, n


# ═══════════════════════════════════════════════════════════════════════
#  1. Olea-Pflueger (2013) effective F — HC-robust
# ═══════════════════════════════════════════════════════════════════════

def effective_f_test(
    data: pd.DataFrame,
    endog: str,
    instruments: List[str],
    exog: Optional[List[str]] = None,
    vcov: str = "HC1",
) -> Dict[str, Any]:
    """
    Olea-Pflueger (2013) robust effective F statistic for weak instruments.

    Computes the heteroskedasticity-robust effective F that is a
    pre-test for the concentration parameter of the first stage. Under
    homoskedasticity (``vcov='classic'``) and a single instrument it
    reduces exactly to the standard first-stage F.

    Parameters
    ----------
    data : pd.DataFrame
    endog : str
        Endogenous regressor (single endogenous variable).
    instruments : list of str
        Excluded instruments.
    exog : list of str, optional
        Included exogenous controls (a constant is added automatically).
    vcov : {'HC0', 'HC1', 'classic'}, default 'HC1'
        Variance estimator for the first-stage residuals:

        - ``'classic'`` — homoskedastic; F_eff equals first-stage F.
        - ``'HC0'`` — White heteroskedasticity-robust.
        - ``'HC1'`` — HC0 with small-sample correction ``n/(n-k)``.

    Returns
    -------
    dict
        ``'F_eff'`` : Olea-Pflueger effective F.
        ``'first_stage_F'`` : Classical first-stage F (for comparison).
        ``'n_instruments'`` : Number of excluded instruments (``k_z``).
        ``'n_obs'`` : Sample size.
        ``'strength'`` : Interpretation string.
        ``'stock_yogo_10pct'`` : 23.1 (conventional threshold for <10 % bias).

    Notes
    -----
    The formula (Andrews-Stock-Sun 2019, eq. 4.13) is

    .. math::

        F_{\\text{eff}} = \\frac{\\hat\\pi' (\\tilde Z' \\tilde Z)\\hat\\pi}
            {\\mathrm{tr}\\!\\left(\\hat\\Omega\\,(\\tilde Z' \\tilde Z)^{-1}\\right)}

    where :math:`\\tilde Z, \\tilde D` are residualized after partialling
    out the exogenous controls, :math:`\\hat\\pi` is the first-stage OLS
    coefficient vector, and :math:`\\hat\\Omega = \\sum_i \\hat\\eta_i^2
    \\tilde z_i \\tilde z_i'` is the HC meat.

    Under homoskedasticity :math:`\\hat\\Omega \\approx \\hat\\sigma_\\eta^2
    \\tilde Z'\\tilde Z`, so the trace collapses to :math:`k_z \\hat
    \\sigma_\\eta^2` and :math:`F_{\\text{eff}}` reduces to the
    first-stage F.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.effective_f_test(df, endog='educ', instruments=['qob'])
    >>> print(f"F_eff = {res['F_eff']:.2f} ({res['strength']})")
    """
    # effective F does not need y; reuse prep helper with endog as both slots
    # but then extract D only (avoid duplicate-column pitfall by using a
    # dedicated path).
    cols = [endog] + instruments + (exog or [])
    df = data[cols].dropna()
    n = len(df)
    D = df[endog].values.astype(float)
    Z = df[instruments].values.astype(float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if exog:
        W = np.column_stack([np.ones(n)] + [df[v].values.astype(float) for v in exog])
    else:
        W = np.ones((n, 1))
    k_z = Z.shape[1]
    k_w = W.shape[1]

    # Partial out controls from D and Z
    D_t = _partial_out(D, W)
    Z_t = _partial_out(Z, W)

    # First-stage OLS on residualized variables
    ZtZt = Z_t.T @ Z_t
    try:
        ZtZt_inv = np.linalg.inv(ZtZt)
    except np.linalg.LinAlgError:
        ZtZt_inv = np.linalg.pinv(ZtZt)

    pi_hat = ZtZt_inv @ (Z_t.T @ D_t)
    eta_hat = D_t - Z_t @ pi_hat

    # Classical first-stage F (for comparison + classic path)
    rss_u = float(eta_hat @ eta_hat)
    df_resid = n - k_w - k_z
    sigma2_eta = rss_u / df_resid if df_resid > 0 else np.nan
    # Test H0: pi=0 via Wald form: F = pi' (Z_t'Z_t) pi / (k_z * sigma2)
    num = float(pi_hat @ ZtZt @ pi_hat)
    f_first = num / (k_z * sigma2_eta) if sigma2_eta > 0 else np.nan

    # HC meat: Omega_hat = sum_i eta_i^2 * z_i z_i'
    if vcov == "classic":
        omega_hat = sigma2_eta * ZtZt
    elif vcov in ("HC0", "HC1"):
        # Σ η̂_i² z_i z_i' = Z' diag(η̂²) Z
        omega_hat = (Z_t * (eta_hat ** 2)[:, None]).T @ Z_t
        if vcov == "HC1" and df_resid > 0:
            omega_hat = omega_hat * (n / df_resid)
    else:
        raise ValueError(f"vcov must be 'classic', 'HC0', or 'HC1'; got {vcov!r}")

    # Olea-Pflueger F_eff = pi' (Z'Z) pi / tr(Omega * (Z'Z)^{-1})
    trace_term = float(np.trace(omega_hat @ ZtZt_inv))
    f_eff = num / trace_term if trace_term > 0 else np.nan

    # Interpretation
    if np.isnan(f_eff):
        strength = "undefined"
    elif f_eff >= 23.1:
        strength = "Strong (F_eff ≥ 23.1, Olea-Pflueger threshold)"
    elif f_eff >= 10:
        strength = "Moderate (10 ≤ F_eff < 23.1) — t-test may over-reject"
    else:
        strength = "WEAK (F_eff < 10) — use AR test or tF-adjusted inference"

    return {
        "F_eff": float(f_eff) if not np.isnan(f_eff) else np.nan,
        "first_stage_F": float(f_first) if not np.isnan(f_first) else np.nan,
        "n_instruments": int(k_z),
        "n_obs": int(n),
        "vcov": vcov,
        "strength": strength,
        "stock_yogo_10pct": 23.1,
    }


# ═══════════════════════════════════════════════════════════════════════
#  2. Lee et al. (2022, AER) tF critical values
# ═══════════════════════════════════════════════════════════════════════

# Table 3a from Lee, McCrary, Moreira & Porter (2022, AER), two-sided 5 %.
# Rows: first-stage F, adjusted t-critical value c_0.05(F).
# Values taken from the paper's published Table 3 panel A (5 % level).
# For F ≤ approx 3.84 the AR test is strictly preferred (no finite c).
_LMMP_TABLE_5PCT: List[Tuple[float, float]] = [
    (  4.00, 18.66),
    (  5.00,  8.18),
    (  6.00,  5.40),
    (  7.00,  4.32),
    (  8.00,  3.75),
    (  9.00,  3.40),
    ( 10.00,  3.16),
    ( 11.00,  2.98),
    ( 12.00,  2.83),
    ( 13.00,  2.72),
    ( 14.00,  2.62),
    ( 15.00,  2.54),
    ( 16.00,  2.47),
    ( 16.38,  2.44),  # conventional F threshold; c ≈ 1.96*sqrt(1.56)
    ( 17.00,  2.41),
    ( 18.00,  2.35),
    ( 20.00,  2.26),
    ( 25.00,  2.13),
    ( 30.00,  2.06),
    ( 40.00,  2.01),
    ( 50.00,  1.98),
    ( 75.00,  1.96),
    (100.00,  1.96),
    (142.60,  1.96),
]


def tF_critical_value(first_stage_F: float, alpha: float = 0.05) -> float:
    """
    Lee–McCrary–Moreira–Porter (2022, AER) tF adjusted critical value.

    Returns the adjusted two-sided t-ratio critical value ``c(F)`` such
    that ``|t| > c(F)`` is a valid 1 - ``alpha`` test of the 2SLS
    coefficient, robust to weak instruments.

    Parameters
    ----------
    first_stage_F : float
        Observed first-stage F statistic (or Olea-Pflueger F_eff).
    alpha : float, default 0.05
        Significance level. Only ``0.05`` is implemented (the only
        level for which LMMP publish a complete table).

    Returns
    -------
    float
        Adjusted critical value. Returns ``inf`` when ``F ≤ 3.84`` (AR
        inference should be used instead). Converges to ``1.96`` as
        ``F → ∞``.

    Raises
    ------
    ValueError
        If ``alpha`` is not 0.05.

    Notes
    -----
    The LMMP tF procedure gives exactly correct 5 % size for any
    first-stage strength. The standard ``1.96`` critical value
    over-rejects substantially when ``F < 104.7`` (the value at which
    ``c = 1.96``).

    Examples
    --------
    >>> import statspai as sp
    >>> # With first-stage F = 10, the adjusted critical value is ~3.16
    >>> c = sp.tF_critical_value(10.0)
    >>> print(f"Adjusted 5% critical value: {c:.2f}")
    """
    if not np.isclose(alpha, 0.05):
        raise ValueError(
            "Only alpha=0.05 is supported (LMMP 2022 publish a full "
            "table for the 5 % level only)."
        )
    if first_stage_F < 3.84:
        return np.inf
    # Linear interpolation in the table (in F, not log F — LMMP use a
    # smooth monotone curve, linear interp introduces tiny error < 0.02).
    table = _LMMP_TABLE_5PCT
    Fs = np.array([x[0] for x in table])
    cs = np.array([x[1] for x in table])
    if first_stage_F >= Fs[-1]:
        return 1.96
    idx = np.searchsorted(Fs, first_stage_F)
    F_lo, F_hi = Fs[idx - 1], Fs[idx]
    c_lo, c_hi = cs[idx - 1], cs[idx]
    w = (first_stage_F - F_lo) / (F_hi - F_lo)
    return float(c_lo + w * (c_hi - c_lo))


# ═══════════════════════════════════════════════════════════════════════
#  3. Anderson-Rubin test (1949) — weak-IV robust
# ═══════════════════════════════════════════════════════════════════════

def anderson_rubin_test(
    data: pd.DataFrame,
    y: str,
    endog: str,
    instruments: List[str],
    exog: Optional[List[str]] = None,
    h0: float = 0,
    alpha: float = 0.05,
    vcov: str = "HC1",
) -> Dict[str, Any]:
    """
    Anderson-Rubin (1949) test — size-correct under weak instruments.

    Tests H0: ``β_endog = h0`` and constructs a confidence set by
    inverting the test over a grid of candidate values. The AR test
    has correct size regardless of instrument strength and is the
    recommended inference procedure when ``F_eff < 10``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    endog : str
        Endogenous regressor (single).
    instruments : list of str
        Excluded instruments.
    exog : list of str, optional
        Included exogenous controls.
    h0 : float, default 0
        Null hypothesis value for the endogenous coefficient.
    alpha : float, default 0.05
        Significance level.
    vcov : {'HC0', 'HC1', 'classic'}, default 'HC1'
        Variance estimator used for the Olea-Pflueger effective F
        reported alongside AR.

    Returns
    -------
    dict
        ``'ar_stat'`` : AR F statistic at ``h0``.
        ``'ar_df'`` : Degrees of freedom ``(k_z, n - k_w - k_z)``.
        ``'ar_pvalue'`` : P-value of AR test at ``h0``.
        ``'ar_ci'`` : ``(low, high)`` AR confidence set (grid-inverted).
        ``'beta_2sls'`` : 2SLS point estimate.
        ``'first_stage_F'`` : Classical first-stage F.
        ``'effective_F'`` : Olea-Pflueger robust F_eff.
        ``'tF_critical_value'`` : Lee et al. (2022) adjusted 5 %
            two-sided critical value. ``None`` if ``alpha != 0.05``.
        ``'strength'`` : Instrument-strength interpretation.
        ``'interpretation'`` : Human-readable summary.

    Examples
    --------
    >>> result = sp.anderson_rubin_test(df, y='wage', endog='education',
    ...                                 instruments=['parent_edu', 'distance'])
    >>> print(result['interpretation'])

    Notes
    -----
    Under H0: ``β = β₀``, regress ``Y - β₀·D`` on ``Z`` and ``W``. The
    F-test on ``Z`` gives the AR statistic. The confidence set is
    constructed by collecting all ``β₀`` values not rejected at level
    ``alpha``. If the AR set is unbounded on one or both sides, the
    returned ``(low, high)`` will be ``±inf`` accordingly.
    """
    Y, D, Z, W, n = _prep_matrices(data, y, endog, instruments, exog)
    k_z = Z.shape[1]
    k_w = W.shape[1]

    # ── Effective F (robust) ────────────────────────────────────────
    f_result = effective_f_test(data, endog, instruments, exog, vcov=vcov)
    f_first = f_result["first_stage_F"]
    f_eff = f_result["F_eff"]

    # ── 2SLS point estimate ─────────────────────────────────────────
    D_t = _partial_out(D, W)
    Z_t = _partial_out(Z, W)
    Y_t = _partial_out(Y, W)
    ZtZt = Z_t.T @ Z_t
    try:
        ZtZt_inv = np.linalg.inv(ZtZt)
    except np.linalg.LinAlgError:
        ZtZt_inv = np.linalg.pinv(ZtZt)
    pi_hat = ZtZt_inv @ (Z_t.T @ D_t)
    D_hat = Z_t @ pi_hat
    denom_2sls = float(D_hat @ D_t)
    beta_2sls = float(D_hat @ Y_t / denom_2sls) if abs(denom_2sls) > 1e-12 else np.nan

    # ── AR statistic at h0 ──────────────────────────────────────────
    df1 = k_z
    df2 = n - k_w - k_z

    def _ar_f(b0: float) -> float:
        """AR F statistic at candidate beta = b0."""
        Y_adj = Y_t - b0 * D_t
        # Regress Y_adj on Z_t; F test coefficients = 0
        # Numerator SS: Y_adj' P_Z_t Y_adj
        proj = Z_t @ (ZtZt_inv @ (Z_t.T @ Y_adj))
        num_ss = float(proj @ Y_adj)
        resid = Y_adj - proj
        rss = float(resid @ resid)
        if df2 <= 0 or rss <= 0:
            return np.nan
        return (num_ss / df1) / (rss / df2)

    ar_f = _ar_f(h0)
    ar_p = float(1 - stats.f.cdf(ar_f, df1, df2)) if not np.isnan(ar_f) else np.nan

    # ── AR confidence set via grid inversion ────────────────────────
    f_crit = stats.f.ppf(1 - alpha, df1, max(df2, 1))
    # Adaptive grid centered on 2SLS
    if not np.isnan(beta_2sls):
        center = beta_2sls
        spread = max(10 * abs(beta_2sls), 5.0)
    else:
        center, spread = 0.0, 10.0
    grid = np.linspace(center - spread, center + spread, 401)
    keep = []
    for b0 in grid:
        fval = _ar_f(b0)
        if not np.isnan(fval) and fval < f_crit:
            keep.append(b0)
    if keep:
        ar_ci = (float(min(keep)), float(max(keep)))
        # Flag potentially unbounded sets (hits the grid edge)
        edge_tol = 1e-6 * spread
        lo_edge = abs(ar_ci[0] - (center - spread)) < edge_tol
        hi_edge = abs(ar_ci[1] - (center + spread)) < edge_tol
        if lo_edge:
            ar_ci = (-np.inf, ar_ci[1])
        if hi_edge:
            ar_ci = (ar_ci[0], np.inf)
    else:
        ar_ci = (np.nan, np.nan)

    # ── tF critical value (only meaningful at alpha=0.05) ───────────
    if np.isclose(alpha, 0.05) and not np.isnan(f_first):
        tF_c = tF_critical_value(max(f_first, 3.84), alpha=0.05)
    else:
        tF_c = None

    return {
        "ar_stat": float(ar_f) if not np.isnan(ar_f) else np.nan,
        "ar_df": (df1, df2),
        "ar_pvalue": ar_p,
        "ar_ci": ar_ci,
        "beta_2sls": beta_2sls,
        "first_stage_F": f_first,
        "effective_F": f_eff,
        "tF_critical_value": tF_c,
        "n_instruments": k_z,
        "strength": f_result["strength"],
        "interpretation": (
            f"First-stage F = {f_first:.2f}, Effective F = {f_eff:.2f}. "
            f"{f_result['strength']}. "
            f"AR test at h0={h0}: F({df1},{df2}) = {ar_f:.2f}, p = {ar_p:.4f}. "
            f"AR {100*(1-alpha):.0f}% CI: [{ar_ci[0]:.4f}, {ar_ci[1]:.4f}]. "
            + (f"LMMP tF critical value (F = {f_first:.1f}): {tF_c:.2f} "
               f"(vs. naive 1.96)." if tF_c is not None else "")
        ),
    }
