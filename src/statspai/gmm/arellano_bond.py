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

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


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
        Time period variable.
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
        applied to the two-step standard errors.
    robust : bool, default True
        Heteroskedasticity-robust standard errors (Windmeijer-corrected
        for two-step). When ``False``, the classical one/two-step VCE.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        With AR(1), AR(2) test statistics and the Sargan/Hansen
        over-identification test in ``model_info``. ``detail`` carries the
        per-coefficient table (lagged Y first, then exogenous regressors).

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

    Key diagnostics:
    - **AR(1) test**: Should reject (expected in first differences).
    - **AR(2) test**: Should NOT reject (validates instrument exogeneity).
    - **Sargan/Hansen test**: Should NOT reject (overidentification).

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

    # --- One-step weight matrix  A = (Σ_i Z_i' H_i Z_i)^{-1} -----------------
    # H_i: 2 on the diagonal, -1 on first off-diagonals for *consecutive*
    # differenced periods within the unit (the MA(1) structure of Δε).
    def _H(eqpos: np.ndarray) -> np.ndarray:
        r = eqpos.size
        H = np.zeros((r, r))
        for a in range(r):
            H[a, a] = 2.0
            for b in range(a + 1, r):
                if abs(eqpos[a] - eqpos[b]) == 1:
                    H[a, b] = H[b, a] = -1.0
        return H

    ZHZ = np.zeros((m, m))
    for r in unit_rows:
        Zi = Z[r]
        ZHZ += Zi.T @ _H(row_eqpos_arr[r]) @ Zi
    A = np.linalg.pinv(ZHZ)

    def _gmm(weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        WZ = W.T @ Z                      # (k, m)
        ZW = WZ.T
        ZY = Z.T @ dY                     # (m,)
        Mmat = WZ @ weight @ ZW           # (k, k)
        Minv = np.linalg.pinv(Mmat)
        beta = Minv @ (WZ @ weight @ ZY)
        return beta, Minv

    beta, Minv1 = _gmm(A)
    resid = dY - W @ beta

    # --- Optional two-step ---------------------------------------------------
    if twostep:
        Omega = np.zeros((m, m))
        for r in unit_rows:
            Zi, ei = Z[r], resid[r]
            ge = Zi.T @ ei
            Omega += np.outer(ge, ge)
        W2 = np.linalg.pinv(Omega)
        beta, Minv2 = _gmm(W2)
        resid = dY - W @ beta
        weight_final = W2
        Minv_final = Minv2
    else:
        weight_final = A
        Minv_final = Minv1

    # --- Variance ------------------------------------------------------------
    WZ = W.T @ Z
    if robust:
        Omega = np.zeros((m, m))
        for r in unit_rows:
            Zi, ei = Z[r], resid[r]
            ge = Zi.T @ ei
            Omega += np.outer(ge, ge)
        bread = Minv_final @ WZ @ weight_final
        vcov = bread @ Omega @ bread.T
        if twostep:
            vcov = _windmeijer(W, Z, dY, resid, A, weight_final,
                               Minv_final, unit_rows, vcov)
    else:
        # classical: V = σ²_ε (W'Z A Z'W)^{-1}, σ²_ε the level-error
        # variance estimated from the whitened differenced residuals.
        whit = 0.0
        for r in unit_rows:
            Hi = _H(row_eqpos_arr[r])
            ei = resid[r]
            whit += ei @ np.linalg.pinv(Hi) @ ei
        sigma2 = whit / max(n - k, 1)
        vcov = sigma2 * Minv_final

    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    se = np.maximum(se, 1e-12)

    # --- Diagnostics ---------------------------------------------------------
    ar1 = _ab_ar_test(resid, row_unit_arr, row_eqpos_arr, 1)
    ar2 = _ab_ar_test(resid, row_unit_arr, row_eqpos_arr, 2)

    if m > k:
        g = Z.T @ resid
        if not robust and not twostep:
            sargan = float(g @ A @ g / sigma2)
        else:
            # Hansen J with the (efficient) weight matrix in force
            sargan = float(g @ weight_final @ g)
        hansen_df = m - k
        hansen_p = float(stats.chi2.sf(sargan, hansen_df))
    else:
        sargan = np.nan
        hansen_df = 0
        hansen_p = np.nan

    # --- Results -------------------------------------------------------------
    var_names = [f'_y_lag{lg}' for lg in range(1, n_ylags + 1)] + x
    z_crit = stats.norm.ppf(1 - alpha / 2)
    rho = float(beta[0])
    rho_se = float(se[0])
    z_val = rho / rho_se
    pvalue = float(2 * stats.norm.sf(abs(z_val)))
    ci = (rho - z_crit * rho_se, rho + z_crit * rho_se)

    z_stats = beta / se
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
        'hansen_stat': sargan,
        'hansen_df': hansen_df,
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


def _windmeijer(W, Z, dY, resid2, A1, W2, Minv2, unit_rows, vcov_uncorr):
    """Windmeijer (2005) finite-sample correction for two-step robust SEs.

    Adds the correction term that accounts for the dependence of the
    efficient weight matrix W2 on the first-step residuals.
    """
    WZ = W.T @ Z
    k = W.shape[1]
    m = Z.shape[1]
    bread = Minv2 @ WZ @ W2                       # (k, m)

    # ∂(W2^{-1})/∂β_j evaluated at the two-step estimate, then mapped
    # through the estimator. D_j = -bread @ (∂Ω/∂β_j) @ W2 @ (Z'resid_2).
    g2 = Z.T @ resid2
    D = np.zeros((k, k))
    for j in range(k):
        dOmega = np.zeros((m, m))
        Wj = W[:, j]
        for r in unit_rows:
            Zi = Z[r]
            ei = resid2[r]
            wj = Wj[r]
            ge = Zi.T @ ei
            gw = Zi.T @ wj
            dOmega += -(np.outer(ge, gw) + np.outer(gw, ge))
        D[:, j] = -(bread @ dOmega @ W2 @ g2)
    corrected = (vcov_uncorr + D @ vcov_uncorr + vcov_uncorr @ D.T
                 + D @ vcov_uncorr @ D.T)
    return corrected


def _ab_ar_test(resid, unit_ids, eq_positions, order):
    """Arellano-Bond serial-correlation test of a given ``order``.

    Tests for ``order``-th order autocorrelation in the differenced
    residuals by matching each residual with the residual ``order``
    periods earlier within the same unit.
    """
    resid = np.asarray(resid, dtype=float)
    e_lag = np.full_like(resid, np.nan)
    for uid in np.unique(unit_ids):
        idx = np.where(unit_ids == uid)[0]
        pos = eq_positions[idx]
        pos_to_row = {p: i for p, i in zip(pos, idx)}
        for p, i in zip(pos, idx):
            if (p - order) in pos_to_row:
                e_lag[i] = resid[pos_to_row[p - order]]

    valid = np.isfinite(e_lag)
    if valid.sum() < 5:
        return {'z': 0.0, 'pvalue': 1.0}

    e_v = resid[valid]
    e_lag_v = e_lag[valid]
    num = float(np.sum(e_v * e_lag_v))
    denom = float(np.sqrt(np.sum(e_lag_v ** 2) * np.mean(e_v ** 2)))
    z = num / denom if denom > 0 else 0.0
    pvalue = float(2 * stats.norm.sf(abs(z)))
    return {'z': float(z), 'pvalue': pvalue}


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
