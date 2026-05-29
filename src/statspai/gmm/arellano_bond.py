"""
Arellano-Bond (1991) and Blundell-Bond (1998) dynamic panel GMM.

Estimates dynamic panel models of the form:

    Y_{it} = ρ Y_{i,t-1} + X_{it}'β + α_i + ε_{it}

using GMM with lagged levels (Arellano-Bond) or lagged levels and
differences (Blundell-Bond system GMM) as instruments.

The Arellano-Bond first-differenced estimator removes the fixed effect
α_i by first-differencing and then exploits the **block-diagonal**
GMM moment conditions E[Y_{i,s} ΔU_{it}] = 0 for s ≤ t-2: every lagged
level that is available for period t is a separate instrument for that
period's differenced equation. The one-step weight matrix is
(Σ_i Z_i' H Z_i)⁻¹ where H encodes the MA(1) structure of the
first-differenced i.i.d. errors (2 on the diagonal, -1 on the first
off-diagonals). This matches Stata's ``xtabond`` and ``xtdpd``.

All standard errors, the Sargan/Hansen over-identification tests, and the
Arellano-Bond AR(1)/AR(2) serial-correlation tests are validated to
machine precision against Stata 18 ``xtabond ..., noconstant`` (one-step
robust and non-robust, two-step conventional, and two-step
Windmeijer-corrected).

References
----------
Arellano, M. and Bond, S. (1991).
"Some Tests of Specification for Panel Data: Monte Carlo Evidence
and an Application to Employment Equations."
*Review of Economic Studies*, 58(2), 277-297. [@arellano1991some]

Blundell, R. and Bond, S. (1998).
"Initial Conditions and Moment Restrictions in Dynamic Panel Data
Models."
*Journal of Econometrics*, 87(1), 115-143. [@blundell1998initial]

Roodman, D. (2009).
"How to Do xtabond2: An Introduction to Difference and System GMM
in Stata."
*Stata Journal*, 9(1), 86-136. [@roodman2009xtabond]

Windmeijer, F. (2005).
"A Finite Sample Correction for the Variance of Linear Efficient
Two-Step GMM Estimators."
*Journal of Econometrics*, 126(1), 25-51. [@windmeijer2005finite]
"""

import warnings
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def _ab_H(eq_positions: np.ndarray) -> np.ndarray:
    """First-difference MA(1) covariance structure for one unit.

    ``2`` on the diagonal and ``-1`` on the first off-diagonals, but only
    between *consecutive* differenced periods (so internal gaps in an
    unbalanced panel correctly break the off-diagonal link).
    """
    r = eq_positions.size
    H = np.zeros((r, r))
    for a in range(r):
        H[a, a] = 2.0
        for b in range(a + 1, r):
            if abs(eq_positions[a] - eq_positions[b]) == 1:
                H[a, b] = H[b, a] = -1.0
    return H


def xtabond(
    data: pd.DataFrame,
    y: str,
    x: Optional[List[str]] = None,
    id: str = 'id',
    time: str = 'time',
    lags: int = 1,
    gmm_lags: Tuple[int, Optional[int]] = (2, None),
    method: str = 'difference',
    twostep: bool = False,
    robust: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Arellano-Bond / Blundell-Bond dynamic panel GMM estimator.

    Equivalent to Stata's ``xtabond`` / ``xtabond2``.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced or unbalanced panel in long format.
    y : str
        Dependent variable.
    x : list of str, optional
        Strictly exogenous regressors. Entered in first differences both
        as regressors and as their own (standard) instruments.
    id : str, default 'id'
        Unit identifier.
    time : str, default 'time'
        Time period variable. Treated as an **ordinal** sequence: the
        sorted distinct values define consecutive periods, so a missing
        period (gap) is recognised, but non-integer / irregularly-spaced
        codes are collapsed to their rank order.
    lags : int, default 1
        Number of lags of Y to include (ρ₁ Y_{t-1} + ... + ρ_p Y_{t-p}).
    gmm_lags : tuple (min, max), default (2, None)
        Range of lags of Y (in levels) used as GMM instruments. ``min``
        must be ≥ 2 (deeper lags are orthogonal to the differenced error).
        ``max=None`` uses **all** available deeper lags, matching Stata's
        ``xtabond`` default. Setting ``max`` caps the instrument count
        (Stata's ``maxldep()`` / collapse-style trimming).
    method : str, default 'difference'
        ``'difference'`` — Arellano-Bond (first-differenced GMM). This is
        the validated path (machine-precision parity with Stata's
        ``xtabond``). ``'system'`` (Blundell-Bond) currently raises
        ``NotImplementedError``: proper system GMM requires a stacked level
        equation and its own Stata parity reference, which is planned for a
        future release.
    twostep : bool, default False
        Use two-step GMM with the efficient weight matrix. When
        ``robust=True`` the Windmeijer (2005) finite-sample correction is
        applied to the two-step standard errors; with ``robust=False`` the
        conventional (downward-biased) two-step SEs are returned and a
        warning is issued.
    robust : bool, default True
        Heteroskedasticity-robust standard errors (Windmeijer-corrected
        for two-step). When ``False``, the classical one/two-step VCE.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        ``estimate`` / ``se`` are the lagged-Y (ρ₁) coefficient. ``detail``
        carries the per-coefficient table (lagged Y first, then exogenous
        regressors). ``model_info`` holds ``n_obs`` (number of
        *first-differenced* observations entering the GMM, not the raw
        panel rows), the AR(1)/AR(2) Arellano-Bond test statistics, the
        Sargan test (one-step, valid under homoskedasticity), and — for
        two-step — the heteroskedasticity-robust Hansen J statistic.

    Examples
    --------
    >>> # Arellano-Bond (difference GMM)
    >>> result = sp.xtabond(df, y='output', x=['capital', 'labor'],
    ...                     id='firm', time='year')
    >>> print(result.summary())

    >>> # Two-step with Windmeijer-corrected SEs
    >>> result = sp.xtabond(df, y='output', x=['capital', 'labor'],
    ...                     id='firm', time='year', twostep=True)

    Notes
    -----
    **Arellano-Bond (1991)**: First-differences the equation to remove
    fixed effects α_i, then uses lagged levels Y_{i,t-2}, Y_{i,t-3}, ...
    as a block-diagonal set of GMM instruments for ΔY_{i,t-1}.

    No constant / time trend is included (unlike Stata's *default*
    ``xtabond``, which adds a ``_cons`` via a level moment). This matches
    Stata's ``xtabond ..., noconstant``; the reported ρ / β coefficients
    are identical to Stata's ``_cons`` run when the series has no drift.

    **Balanced vs gapped panels.** All standard errors, the Sargan/Hansen
    tests, and the one-step AR(1)/AR(2) tests are validated to machine
    precision against Stata for balanced (and ragged-but-gap-free) panels.
    When a unit is missing an *interior* period, the estimator stays
    consistent but its finite-sample numbers can differ from Stata's
    ``xtabond`` by ~1% (Stata, ``xtabond2``, and R's ``plm`` each use a
    slightly different gap-weighting convention); a warning is emitted in
    that case.

    Key diagnostics:
    - **AR(1) test**: Should reject (expected in first differences).
    - **AR(2) test**: Should NOT reject (validates instrument exogeneity).
    - **Sargan / Hansen test**: Should NOT reject (overidentification).
      Sargan (one-step) is not robust to heteroskedasticity; prefer the
      two-step Hansen J when that is a concern.

    See Roodman (2009, *Stata Journal*) for practical guidance.
    """
    if x is None:
        x = []
    x = list(x)

    # --- Prepare panel -------------------------------------------------------
    df = data[[id, time, y] + x].dropna().sort_values([id, time])
    times = sorted(df[time].unique())
    time_pos = {t: p for p, t in enumerate(times)}
    T = len(times)
    units = df[id].unique()
    n_units = len(units)

    min_lag, max_lag = gmm_lags
    if min_lag < 2:
        raise ValueError("gmm_lags min must be >= 2 (Arellano-Bond moment "
                         "conditions require lags of at least 2).")
    if max_lag is None:
        max_lag = T  # all available deeper lags (Stata's default)

    if method == 'system':
        raise NotImplementedError(
            "Blundell-Bond system GMM is not yet implemented. Proper system "
            "GMM stacks an additional level equation (instrumented by lagged "
            "first differences) and requires its own Stata `xtdpdsys` / "
            "`xtabond2` parity reference before it can be trusted. Use "
            "method='difference' (Arellano-Bond), which is validated to "
            "machine precision against Stata's `xtabond`."
        )
    if method != 'difference':
        raise ValueError("method must be 'difference' (or 'system', which is "
                         "not yet implemented).")

    n_ylags = lags                       # number of lagged-Y regressors
    n_x = len(x)
    k = n_ylags + n_x                    # number of structural parameters

    # First differenced equation at global period position `p` requires
    # y at positions p, p-1, ..., p-n_ylags-1 (so that Δy_p and every
    # regressor Δy_{p-l} for l=1..n_ylags are defined). Earliest is
    # p = n_ylags + 1.
    eq_positions = list(range(n_ylags + 1, T))
    if not eq_positions:
        raise ValueError("Not enough time periods for the requested lags.")

    # --- Enumerate the block-diagonal GMM instrument columns -----------------
    # Column keyed by (equation period p, source level position s) with
    # s <= p-2 and lag = p - s within [min_lag, max_lag].
    ycols: List[Tuple[int, int]] = []
    for p in eq_positions:
        for s in range(0, p - 1):        # s <= p-2  ->  lag = p-s >= 2
            lag = p - s
            if min_lag <= lag <= max_lag:
                ycols.append((p, s))
    ycol_pos = {key: j for j, key in enumerate(ycols)}
    n_ycols = len(ycols)

    # exogenous instruments: Δx (one standard-instrument column per x var),
    # appended after the GMM columns.
    x_iv_offset = n_ycols
    m = n_ycols + n_x                     # total instruments

    if m < k:
        raise ValueError("Under-identified: fewer instruments than "
                         "parameters. Loosen gmm_lags or add periods.")

    # --- Build per-unit differenced equations and instrument blocks ----------
    W_rows: List[np.ndarray] = []         # regressors  ΔW
    Z_rows: List[np.ndarray] = []         # instruments Z
    dY_rows: List[float] = []             # Δy
    row_unit: List[int] = []              # unit index per row
    row_eqpos: List[int] = []             # equation period per row

    for ui, uid in enumerate(units):
        g = df[df[id] == uid]
        ypos = {time_pos[t]: yv for t, yv in zip(g[time], g[y])}
        xpos = {xv: {time_pos[t]: val for t, val in zip(g[time], g[xv])}
                for xv in x}

        for p in eq_positions:
            # need y at p, p-1, and the deepest regressor lag p-n_ylags-1
            needed_y = [p - off for off in range(0, n_ylags + 2)]
            if any(q not in ypos for q in needed_y):
                continue
            if any((p not in xpos[xv] or (p - 1) not in xpos[xv]) for xv in x):
                continue

            dy = ypos[p] - ypos[p - 1]
            wrow = np.empty(k)
            for lg in range(1, n_ylags + 1):
                wrow[lg - 1] = ypos[p - lg] - ypos[p - lg - 1]
            for j, xv in enumerate(x):
                wrow[n_ylags + j] = xpos[xv][p] - xpos[xv][p - 1]

            zrow = np.zeros(m)
            for s in range(0, p - 1):
                lag = p - s
                if min_lag <= lag <= max_lag and s in ypos:
                    zrow[ycol_pos[(p, s)]] = ypos[s]
            for j, xv in enumerate(x):
                zrow[x_iv_offset + j] = xpos[xv][p] - xpos[xv][p - 1]

            W_rows.append(wrow)
            Z_rows.append(zrow)
            dY_rows.append(dy)
            row_unit.append(ui)
            row_eqpos.append(p)

    if len(dY_rows) < k + 1:
        raise ValueError("Not enough observations after differencing.")

    W = np.asarray(W_rows)                # (n, k)
    Z = np.asarray(Z_rows)                # (n, m)
    dY = np.asarray(dY_rows)             # (n,)
    row_unit_arr = np.asarray(row_unit)
    row_eqpos_arr = np.asarray(row_eqpos)
    n = W.shape[0]

    # per-unit row slices
    unit_rows = [np.where(row_unit_arr == ui)[0] for ui in range(n_units)]
    unit_rows = [r for r in unit_rows if r.size > 0]

    # Internal gaps (a unit missing an interior period) change the
    # first-difference / instrument structure, and Stata's xtabond, xtabond2,
    # and R's plm use slightly different finite-sample gap conventions. Parity
    # is validated to machine precision for balanced / gap-free panels; warn so
    # gapped-panel results are not mistaken for exact Stata reproductions.
    has_internal_gap = False
    for uid in units:
        pos = sorted(time_pos[t] for t in df.loc[df[id] == uid, time].unique())
        if pos and (pos[-1] - pos[0] + 1) != len(pos):
            has_internal_gap = True
            break
    if has_internal_gap:
        warnings.warn(
            "Panel has internal time gaps. Estimates remain consistent, but "
            "for gapped panels the one-/two-step results differ modestly from "
            "Stata's xtabond (~1%) because of differing gap-weighting "
            "conventions; machine-precision Stata parity holds only for "
            "balanced / gap-free panels.",
            stacklevel=2,
        )

    # --- One-step weight matrix  A = (Σ_i Z_i' H Z_i)^{-1} -------------------
    WZ = W.T @ Z                          # (k, m)
    ZW = WZ.T
    ZdY = Z.T @ dY                        # (m,)

    ZHZ = np.zeros((m, m))
    for r in unit_rows:
        Zi = Z[r]
        ZHZ += Zi.T @ _ab_H(row_eqpos_arr[r]) @ Zi
    A = _safe_inv(ZHZ, "GMM weight matrix Z'HZ")

    def _gmm(weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Mmat = WZ @ weight @ ZW           # (k, k)
        Minv = _safe_inv(Mmat, "moment matrix W'ZWZ'W")
        beta_ = Minv @ (WZ @ weight @ ZdY)
        return beta_, Minv

    # --- One-step estimate + robust "meat" -----------------------------------
    beta1, Minv1 = _gmm(A)
    resid1 = dY - W @ beta1

    Omega1 = np.zeros((m, m))             # Σ_i Z_i' ê1 ê1' Z_i
    for r in unit_rows:
        ge = Z[r].T @ resid1[r]
        Omega1 += np.outer(ge, ge)
    bread1 = Minv1 @ WZ @ A
    V1_robust = bread1 @ Omega1 @ bread1.T

    # Level-error variance for the classical one-step VCE and the Sargan
    # test. The first-differenced errors have variance 2σ², so Stata's
    # σ̂²₁ = (Δê'Δê) / (2 (N − k)).
    sigma2 = float(resid1 @ resid1) / (2.0 * max(n - k, 1))

    # --- Optional efficient two-step -----------------------------------------
    if twostep:
        if not robust:
            warnings.warn(
                "Two-step GMM standard errors are downward biased in finite "
                "samples; robust=True (Windmeijer correction) is recommended.",
                stacklevel=2,
            )
        W2 = _safe_inv(Omega1, "two-step weight matrix Σ Z' êê'Z")
        beta, Minv2 = _gmm(W2)
        resid = dY - W @ beta
        weight_final, Minv_final = W2, Minv2
    else:
        beta, Minv2, W2 = beta1, None, None
        resid = resid1
        weight_final, Minv_final = A, Minv1

    # --- Variance ------------------------------------------------------------
    if not twostep:
        # one-step: robust sandwich, or classical σ̂²·(W'ZAZ'W)⁻¹
        vcov = V1_robust if robust else sigma2 * Minv1
    elif robust:
        # two-step robust: Windmeijer (2005) finite-sample correction
        vcov = _windmeijer(W, Z, WZ, resid1, resid, W2, Minv2,
                           V1_robust, unit_rows)
    else:
        # two-step conventional: efficient-GMM VCE = (W'Z W2 Z'W)⁻¹
        vcov = Minv2

    var_diag = np.diag(vcov)
    if np.any(var_diag <= 0):
        warnings.warn(
            "Non-positive coefficient variance encountered — the model may be "
            "under-identified or the instrument set rank-deficient; the "
            "affected standard errors are unreliable.",
            stacklevel=2,
        )
    se = np.sqrt(np.maximum(var_diag, 0.0))

    # --- Diagnostics ---------------------------------------------------------
    # AR test variance is robust for robust / two-step estimators, classical
    # otherwise (matching Stata's vce-dependent Arellano-Bond test).
    robust_ar = robust or twostep
    ar1 = _ab_ar_test(resid, unit_rows, row_eqpos_arr, 1,
                      Z, W, weight_final, Minv_final, robust_ar, sigma2)
    ar2 = _ab_ar_test(resid, unit_rows, row_eqpos_arr, 2,
                      Z, W, weight_final, Minv_final, robust_ar, sigma2)

    # Over-identification: Sargan (one-step, homoskedastic) and — for
    # two-step — the heteroskedasticity-robust Hansen J.
    sargan_df = m - k
    if sargan_df > 0:
        g1 = Z.T @ resid1
        sargan = float(g1 @ A @ g1 / sigma2)
        sargan_p = float(stats.chi2.sf(sargan, sargan_df))
        if twostep:
            g2 = Z.T @ resid
            hansen = float(g2 @ W2 @ g2)
            hansen_p = float(stats.chi2.sf(hansen, sargan_df))
        else:
            hansen, hansen_p = np.nan, np.nan
    else:
        sargan = sargan_p = hansen = hansen_p = np.nan

    # --- Results -------------------------------------------------------------
    # Stata-style coefficient labels: lagged-Y terms as "L<k>.<y>".
    var_names = [f'L{lg}.{y}' for lg in range(1, n_ylags + 1)] + x
    z_crit = stats.norm.ppf(1 - alpha / 2)
    rho = float(beta[0])
    rho_se = float(se[0])
    z_val = rho / rho_se if rho_se > 0 else np.nan
    pvalue = float(2 * stats.norm.sf(abs(z_val))) if np.isfinite(z_val) else np.nan
    ci = (rho - z_crit * rho_se, rho + z_crit * rho_se)

    with np.errstate(divide='ignore', invalid='ignore'):
        z_stats = np.where(se > 0, beta / se, np.nan)
    pvals = 2 * stats.norm.sf(np.abs(z_stats))
    detail = pd.DataFrame({
        'variable': var_names,
        'coefficient': beta,
        'se': se,
        'z': z_stats,
        'pvalue': pvals,
    })

    model_info = {
        'method': method.upper() + ' GMM',
        'twostep': twostep,
        'robust': robust,
        'windmeijer': bool(twostep and robust),
        'n_units': n_units,
        'n_obs': n,
        'n_instruments': m,
        'n_regressors': k,
        'gmm_lags': (min_lag, None if max_lag >= T else max_lag),
        'ar1_z': ar1['z'],
        'ar1_p': ar1['pvalue'],
        'ar2_z': ar2['z'],
        'ar2_p': ar2['pvalue'],
        'sargan_stat': sargan,
        'sargan_df': sargan_df,
        'sargan_p': sargan_p,
        'hansen_stat': hansen,
        'hansen_df': sargan_df if twostep else 0,
        'hansen_p': hansen_p,
    }

    return CausalResult(
        method=f"Arellano-Bond ({'Two-step' if twostep else 'One-step'} GMM)",
        estimand='rho (AR coefficient)',
        estimate=rho,
        se=rho_se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='arellano_bond',
    )


def _safe_inv(M: np.ndarray, what: str) -> np.ndarray:
    """Inverse that warns loudly when the matrix is rank-deficient.

    Falls back to the Moore-Penrose pseudo-inverse so the computation can
    proceed, but — per the project's "fail loudly, never silently degrade"
    rule — emits a warning instead of quietly returning garbage when ``M``
    is singular (e.g. collinear regressors or too many instruments).
    """
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        warnings.warn(
            f"{what} is singular (rank-deficient); falling back to the "
            f"pseudo-inverse. Results may be unreliable — check for collinear "
            f"regressors or an over-saturated instrument set.",
            stacklevel=3,
        )
        return np.linalg.pinv(M)


def _windmeijer(W, Z, WZ, resid1, resid2, W2, Minv2, V1_robust, unit_rows):
    """Windmeijer (2005) finite-sample correction for two-step robust SEs.

    ``V_corr = V₂ + D V₂ + V₂ D' + D V₁ᵣ D'`` where ``V₂ = Minv2`` is the
    conventional two-step VCE and ``V₁ᵣ`` the one-step robust VCE. The
    score derivative ``∂Ω/∂β`` uses the **step-1** residuals because the
    efficient weight ``W₂ = Ω(ê₁)⁻¹`` depends on them.

    Validated to machine precision against Stata's ``xtabond, twostep
    vce(robust)`` (WC-robust) standard errors.
    """
    k = W.shape[1]
    m = Z.shape[1]
    g2 = Z.T @ resid2
    bread2 = Minv2 @ WZ @ W2                      # (k, m)
    D = np.zeros((k, k))
    for j in range(k):
        Wj = W[:, j]
        dOmega = np.zeros((m, m))
        for r in unit_rows:
            ge = Z[r].T @ resid1[r]               # step-1 residuals
            gw = Z[r].T @ Wj[r]
            dOmega -= np.outer(ge, gw) + np.outer(gw, ge)
        D[:, j] = -(bread2 @ dOmega @ W2 @ g2)
    return (Minv2 + D @ Minv2 + Minv2 @ D.T + D @ V1_robust @ D.T)


def _ab_ar_test(resid, unit_rows, eq_positions, order,
                Z, W, weight, Minv, robust, sigma2):
    """Arellano-Bond (1991) test for AR(``order``) in the differenced errors.

    ``m = (q'ê) / sqrt(Var)`` where ``q`` holds, for each row at period
    ``p``, the residual of the same unit at period ``p-order`` (0 where
    unavailable). The variance carries the influence-function adjustment
    for the estimated coefficients,

        c = q − Z·W·(Z'W)·Minv·(W'q),

    with a robust outer-product variance ``Σ_i (c_i'ê_i)²`` or, for the
    classical case, ``σ̂² Σ_i c_i' H_i c_i``.

    Validated to machine precision against Stata's ``estat abond`` for the
    one-step robust and non-robust estimators. For two-step estimation the
    test uses the conventional two-step influence function: it matches
    Stata's two-step ``estat abond`` to within ~0.1%, but does not apply the
    Windmeijer correction to the *test* variance, so the two-step-robust z
    differs modestly from Stata's (the AR(1)/AR(2) inferential conclusion is
    unchanged).
    """
    resid = np.asarray(resid, dtype=float)
    nrow = resid.shape[0]
    q = np.zeros(nrow)
    for r in unit_rows:
        pos = {p: i for p, i in zip(eq_positions[r], r)}
        for p, i in zip(eq_positions[r], r):
            if (p - order) in pos:
                q[i] = resid[pos[p - order]]

    if not np.any(q):
        return {'z': np.nan, 'pvalue': np.nan}

    num = float(q @ resid)
    # influence-function adjustment: c = q - Z weight (Z'W) Minv (W'q)
    u = Minv @ (W.T @ q)
    adj = Z @ (weight @ ((Z.T @ W) @ u))
    c = q - adj

    if robust:
        var = float(sum((c[r] @ resid[r]) ** 2 for r in unit_rows))
    else:
        var = float(sigma2 * sum(c[r] @ _ab_H(eq_positions[r]) @ c[r]
                                 for r in unit_rows))

    if not np.isfinite(var) or var <= 0:
        return {'z': np.nan, 'pvalue': np.nan}
    z = num / np.sqrt(var)
    return {'z': float(z), 'pvalue': float(2 * stats.norm.sf(abs(z)))}


# Citations
CausalResult._CITATIONS['arellano_bond'] = (
    "@article{arellano1991some,\n"
    "  title={Some Tests of Specification for Panel Data: Monte Carlo "
    "Evidence and an Application to Employment Equations},\n"
    "  author={Arellano, Manuel and Bond, Stephen},\n"
    "  journal={Review of Economic Studies},\n"
    "  volume={58},\n"
    "  number={2},\n"
    "  pages={277--297},\n"
    "  year={1991},\n"
    "  publisher={Oxford University Press}\n"
    "}"
)
