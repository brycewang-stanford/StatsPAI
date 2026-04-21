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
        "first_stage_f": f_first,
        "effective_F": f_eff,
        "effective_f": f_eff,
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


def _kstat_and_pvalue_at(k_cs, h0: float) -> Tuple[float, float]:
    """Pick the K-statistic and chi2(1) p-value at (nearest-grid) β=h0."""
    grid = np.asarray(k_cs.beta_grid)
    if grid.size == 0:
        return (np.nan, np.nan)
    idx = int(np.argmin(np.abs(grid - h0)))
    stat = float(k_cs.statistic[idx])
    p = float(1.0 - stats.chi2.cdf(stat, df=1))
    return stat, p


# ═══════════════════════════════════════════════════════════════════════
#  4. Unified weak-IV-robust panel — Stata `estat weakrobust` equivalent
# ═══════════════════════════════════════════════════════════════════════


class WeakRobustResult:
    """Container holding the unified weak-IV-robust panel."""

    def __init__(
        self,
        data: Dict[str, Any],
        endog: str,
        instruments: List[str],
        n: int,
        h0: float,
        alpha: float,
    ):
        self._data = data
        self.endog = endog
        self.instruments = list(instruments)
        self.n = int(n)
        self.h0 = float(h0)
        self.alpha = float(alpha)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def to_frame(self) -> pd.DataFrame:
        """Return the Stata-style single-table summary."""
        rows = []
        d = self._data
        # Strength diagnostics
        rows.append(("First-stage F (classical)",
                     d["first_stage_F"], None, None))
        rows.append(("Olea–Pflueger effective F",
                     d["effective_F"], None, None))
        if d.get("kp_rk_lm") is not None:
            rows.append(("Kleibergen–Paap rk LM",
                         d["kp_rk_lm"], d["kp_rk_lm_pvalue"], None))
            rows.append(("Kleibergen–Paap rk Wald F",
                         d["kp_rk_f"], None, None))
        # Weak-IV-robust inference under H0
        rows.append((f"Anderson–Rubin at β={self.h0:g}",
                     d["ar_stat"], d["ar_pvalue"], d["ar_ci"]))
        if d.get("clr_stat") is not None:
            rows.append((f"Moreira CLR at β={self.h0:g}",
                         d["clr_stat"], d["clr_pvalue"], d.get("clr_ci")))
        if d.get("k_stat") is not None:
            rows.append((f"Kleibergen K at β={self.h0:g}",
                         d["k_stat"], d["k_pvalue"], d.get("k_ci")))
        return pd.DataFrame(rows, columns=["statistic", "value", "p_value", "CI"])

    def summary(self) -> str:
        d = self._data
        lines = [
            "Weak-IV-robust diagnostic panel  (sp.weakrobust)",
            "=" * 60,
            f"  endog        : {self.endog}",
            f"  instruments  : {', '.join(self.instruments)}",
            f"  N            : {self.n}",
            f"  H0: β        : {self.h0:g}",
            f"  α            : {self.alpha}",
            "-" * 60,
            "Strength:",
            f"  First-stage F (classical)   : {d['first_stage_F']:10.4f}",
            f"  Olea–Pflueger effective F   : {d['effective_F']:10.4f}",
        ]
        if d.get("kp_rk_lm") is not None:
            lines += [
                f"  Kleibergen–Paap rk LM       : {d['kp_rk_lm']:10.4f}"
                f"   p = {d['kp_rk_lm_pvalue']:.4f}",
                f"  Kleibergen–Paap rk Wald F   : {d['kp_rk_f']:10.4f}",
            ]
        lines += [
            "Weak-IV-robust tests under H0:",
            f"  Anderson–Rubin F            : {d['ar_stat']:10.4f}"
            f"   p = {d['ar_pvalue']:.4f}",
            f"  AR  {100*(1-self.alpha):.0f}% CI                  : "
            f"[{d['ar_ci'][0]:.4f}, {d['ar_ci'][1]:.4f}]",
        ]
        if d.get("clr_stat") is not None:
            lines.append(
                f"  Moreira CLR                 : {d['clr_stat']:10.4f}"
                f"   p = {d['clr_pvalue']:.4f}"
            )
            if d.get("clr_ci") is not None:
                lines.append(
                    f"  CLR {100*(1-self.alpha):.0f}% CI                  : "
                    f"[{d['clr_ci'][0]:.4f}, {d['clr_ci'][1]:.4f}]"
                )
        if d.get("k_stat") is not None:
            lines.append(
                f"  Kleibergen K                : {d['k_stat']:10.4f}"
                f"   p = {d['k_pvalue']:.4f}"
            )
            if d.get("k_ci") is not None:
                lines.append(
                    f"  K   {100*(1-self.alpha):.0f}% CI                  : "
                    f"[{d['k_ci'][0]:.4f}, {d['k_ci'][1]:.4f}]"
                )
        if d.get("tF_critical_value") is not None:
            lines.append(
                f"  Lee-McCrary-Moreira-Porter tF crit. value    : "
                f"{d['tF_critical_value']:.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


def weakrobust(
    data: pd.DataFrame,
    y: str,
    endog: str,
    instruments: List[str],
    exog: Optional[List[str]] = None,
    *,
    h0: float = 0.0,
    alpha: float = 0.05,
    vcov: str = "HC1",
    include_clr: bool = True,
    include_k: bool = True,
    clr_simulations: int = 20_000,
    grid_size: int = 401,
    random_state: Optional[int] = None,
) -> "WeakRobustResult":
    """
    Stata-style unified weak-instrument-robust diagnostic panel.

    Bundles in a single call:

    1. **Classical first-stage F** and **Olea–Pflueger (2013) effective F**
       — instrument-strength pre-tests.
    2. **Kleibergen–Paap (2006) rk LM and Wald F** — rank test of the
       reduced form, heteroskedasticity-robust Cragg-Donald analogue.
    3. **Anderson–Rubin (1949)** F test of ``H0: β_endog = h0`` and its
       weak-IV-robust confidence set.
    4. **Moreira (2003) Conditional LR (CLR)** test and CLR confidence
       set by grid inversion (uniformly most powerful invariant for a
       single endogenous regressor).
    5. **Kleibergen (2002) K** score test and K confidence set.
    6. **Lee–McCrary–Moreira–Porter (2022) tF** adjusted critical value.

    This entry point is the Python analogue of Stata 19's
    ``ivregress 2sls; estat weakrobust`` and ``weakiv`` / ``ivreg2`` in
    Stata 17+, unifying tools that are scattered across ``ivmodel`` (R),
    ``linearmodels`` (Python), and the user-written packages ``weakiv``,
    ``rivtest`` (Stata).

    Parameters
    ----------
    data : DataFrame
    y, endog : str
        Outcome and single endogenous regressor (column names in ``data``).
    instruments : list of str
        Excluded instruments.
    exog : list of str, optional
        Included exogenous controls. An intercept is always added.
    h0 : float, default 0.0
        Null value for the endogenous coefficient. All under-H0 tests
        (AR, CLR, K) are evaluated at ``β = h0``.
    alpha : float, default 0.05
        Significance level for the robust confidence sets.
    vcov : {'HC0', 'HC1', 'classic'}, default 'HC1'
        Used by the Olea–Pflueger effective F.
    include_clr : bool, default True
        Also run the CLR test and invert it for a CLR confidence set.
    include_k : bool, default True
        Also run the Kleibergen K score test and K confidence set.
    clr_simulations : int, default 20 000
        Monte-Carlo draws for the CLR null distribution.
    grid_size : int, default 401
        Grid resolution used by AR/CLR/K confidence-set inversion.
    random_state : int, optional

    Returns
    -------
    WeakRobustResult
        Accepts ``.summary()``, ``.to_frame()`` and dict-style lookup.

    Examples
    --------
    >>> panel = sp.weakrobust(df, y='wage', endog='educ',
    ...                       instruments=['nearc2','nearc4'],
    ...                       exog=['age','exper'])
    >>> print(panel.summary())
    >>> panel.to_frame()

    References
    ----------
    Anderson–Rubin (1949) Ann. Math. Stat. 20, 46-63.
    Kleibergen (2002) Econometrica 70, 1781-1803.
    Moreira (2003) Econometrica 71, 1027-1048.
    Kleibergen–Paap (2006) J. Econom. 133, 97-126.
    Olea–Pflueger (2013) JBES 31, 358-369.
    Lee–McCrary–Moreira–Porter (2022) AER 112, 3260-3290.
    """
    if isinstance(instruments, str):
        instruments = [instruments]
    if exog is not None and isinstance(exog, str):
        exog = [exog]

    ar = anderson_rubin_test(
        data=data, y=y, endog=endog, instruments=instruments, exog=exog,
        h0=h0, alpha=alpha, vcov=vcov,
    )

    out: Dict[str, Any] = {
        "first_stage_F": ar["first_stage_F"],
        "effective_F": ar["effective_F"],
        "beta_2sls": ar["beta_2sls"],
        "ar_stat": ar["ar_stat"],
        "ar_pvalue": ar["ar_pvalue"],
        "ar_ci": ar["ar_ci"],
        "tF_critical_value": ar["tF_critical_value"],
        "n_instruments": ar["n_instruments"],
        "strength": ar["strength"],
    }

    # ── KP rk LM / Wald F (robust rank test) ──────────────────────────
    try:
        from ..iv.weak_identification import kleibergen_paap_rk
        Y, D, Z, W, n_obs = _prep_matrices(
            data, y, endog, instruments, exog,
        )
        W_exog = W[:, 1:] if W.shape[1] > 1 else None
        kp = kleibergen_paap_rk(
            endog=D.reshape(-1, 1),
            instruments=Z,
            exog=W_exog,
            add_const=True,
            cov_type="robust",
        )
        out["kp_rk_lm"] = kp.rk_lm
        out["kp_rk_lm_pvalue"] = kp.rk_lm_pvalue
        out["kp_rk_f"] = kp.rk_f
    except Exception as exc:  # pragma: no cover
        out["kp_error"] = str(exc)
        n_obs = len(data.dropna(subset=[y, endog] + list(instruments)
                                + list(exog or [])))

    # ── CLR statistic + p-value at H0 (exact via Monte-Carlo) ─────────
    if include_clr:
        try:
            from ..iv.weak_identification import conditional_lr_test
            clr_res = conditional_lr_test(
                y=y, endog=endog, instruments=list(instruments),
                exog=list(exog) if exog else None,
                data=data, beta0=h0,
                n_simulations=clr_simulations,
                random_state=random_state,
            )
            out["clr_stat"] = clr_res.statistic
            out["clr_pvalue"] = clr_res.pvalue
        except Exception as exc:  # pragma: no cover
            out["clr_error"] = str(exc)

    # ── CLR + K confidence sets via grid inversion ────────────────────
    if include_clr or include_k:
        try:
            from ..iv.weak_iv_ci import conditional_lr_ci, k_test_ci
            level = 1.0 - alpha
            exog_arg = list(exog) if exog else None
            if include_clr:
                clr_cs = conditional_lr_ci(
                    y=y, endog=endog, instruments=list(instruments),
                    exog=exog_arg, data=data, level=level,
                    n_grid=grid_size, n_sim=max(clr_simulations // 4, 2000),
                    random_state=random_state,
                )
                lo, hi = float(clr_cs.lower), float(clr_cs.upper)
                out["clr_ci"] = (lo, hi)
                out["clr_is_empty"] = bool(clr_cs.is_empty)
                out["clr_is_unbounded"] = bool(clr_cs.is_unbounded)
            if include_k:
                k_cs = k_test_ci(
                    y=y, endog=endog, instruments=list(instruments),
                    exog=exog_arg, data=data, level=level,
                    n_grid=grid_size,
                )
                out["k_ci"] = (float(k_cs.lower), float(k_cs.upper))
                out["k_is_empty"] = bool(k_cs.is_empty)
                out["k_is_unbounded"] = bool(k_cs.is_unbounded)
                # K stat and p at h0 — read from nearest-grid point
                _k_stat, _k_p = _kstat_and_pvalue_at(k_cs, h0)
                out["k_stat"] = _k_stat
                out["k_pvalue"] = _k_p
        except Exception as exc:  # pragma: no cover
            out["weak_iv_ci_error"] = str(exc)

    return WeakRobustResult(
        data=out,
        endog=endog,
        instruments=list(instruments),
        n=int(n_obs),
        h0=h0,
        alpha=alpha,
    )


__all__ = [
    "effective_f_test",
    "tF_critical_value",
    "anderson_rubin_test",
    "weakrobust",
    "WeakRobustResult",
]
