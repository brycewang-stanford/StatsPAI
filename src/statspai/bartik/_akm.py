"""Shared shift-share inference kernels (private).

``_akm_fit`` is a line-by-line port of the reference implementation of
Adao, Kolesar and Morales (2019) inference, R ``ShiftShareSE::reg_ss.fit`` /
``ivreg_ss.fit`` (version 1.1.0), which the Stata ``reg_ss`` / ``ivreg_ss``
commands reproduce. Conventions (all the reference's own):

* ``X`` is the shift-share variable itself, ``W`` the ``n x K`` share
  matrix, ``Z`` the controls *including* the intercept.
* ``hX`` are the coefficients of the control-residualised ``X`` on ``W``
  (the estimated, control-adjusted shocks); ``cR_k = hX_k * W_k' e``.
  ``SE_AKM = sqrt(sum_k cR_k^2) / RX`` with ``RX = ddX' ddX`` (OLS) or
  ``ddY2' ddX`` (IV).
* AKM0 inverts the null-imposed test into a CI; the reported "SE" is the
  CI half-width over ``z_{1-alpha/2}``; its p-value uses the null-imposed
  SE.
* OLS: homoskedastic ``RSS/(n-p)``, EHW with ``n/(n-p)``, region cluster
  with ``G/(G-1) (n-1)/(n-p)``. IV: homoskedastic ``RSS/n``, EHW and region
  cluster without any small-sample factor.
* All p-values and CIs are normal.

``_bhj_aggregate`` reproduces Borusyak, Hull and Jaravel's ``ssaggregate``
(Stata SSC 1.2.2 / R kylebutts/ssaggregate) and the shock-level IV they
recommend (``ivreg2 y (x = g) [aw = s_n], robust``: intercept, HC0).
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

_SE_ROWS = ("Homoscedastic", "EHW", "Reg. cluster", "AKM", "AKM0")


def _resid(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    return v - M @ np.linalg.lstsq(M, v, rcond=None)[0]


def _dqrdc2_rank_columns(x: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Columns R's ``qr()`` keeps: a port of LINPACK ``dqrdc2`` (R's variant).

    Householder QR with *limited* pivoting: a column whose norm, downdated
    as the reduction proceeds, falls below ``tol`` times its original norm
    is moved to the end; ``pivot[1:rank]`` are the survivors in their
    original order. The downdating (with R's 1e-6 recompute threshold) is
    what decides borderline columns, so an exact Gram-Schmidt test does not
    pick the same set on nearly collinear share matrices.
    """
    x = np.array(x, dtype=float, order="F", copy=True)
    n, p = x.shape
    jpvt = np.arange(p)
    qraux = np.linalg.norm(x, axis=0)
    work1 = qraux.copy()
    work2 = qraux.copy()
    work2[work2 == 0] = 1.0
    lup = min(n, p)
    k = p + 1  # dqrdc2's (1-based) rank marker
    for col_l in range(lup):
        while col_l + 1 < k and qraux[col_l] < work2[col_l] * tol:
            # move the negligible column col_l to the end
            for arr in (jpvt, qraux, work1, work2):
                v = arr[col_l]
                arr[col_l : p - 1] = arr[col_l + 1 : p]
                arr[p - 1] = v
            col = x[:, col_l].copy()
            x[:, col_l : p - 1] = x[:, col_l + 1 : p]
            x[:, p - 1] = col
            k -= 1
        if col_l == n - 1:
            break
        nrmxl = np.linalg.norm(x[col_l:, col_l])
        if nrmxl == 0.0:
            continue
        if x[col_l, col_l] != 0.0:
            nrmxl = np.copysign(nrmxl, x[col_l, col_l])
        x[col_l:, col_l] /= nrmxl
        x[col_l, col_l] += 1.0
        if col_l + 1 < p:
            t = -(x[col_l:, col_l] @ x[col_l:, col_l + 1 :]) / x[col_l, col_l]
            x[col_l:, col_l + 1 :] += np.outer(x[col_l:, col_l], t)
            for j in range(col_l + 1, p):
                if qraux[j] == 0.0:
                    continue
                tt = max(1.0 - (abs(x[col_l, j]) / qraux[j]) ** 2, 0.0)
                if abs(tt) < 1e-6:
                    qraux[j] = np.linalg.norm(x[col_l + 1 :, j])
                    work1[j] = qraux[j]
                else:
                    qraux[j] = qraux[j] * np.sqrt(tt)
        qraux[col_l] = x[col_l, col_l]
        x[col_l, col_l] = -nrmxl
    rank = min(k - 1, n)
    return np.asarray(jpvt[:rank], dtype=int)


def _drop_collinear_shares(W: np.ndarray) -> np.ndarray:
    """Share columns kept for AKM inference, as ``ShiftShareSE:::drop_collinear``.

    R: ``keep <- qr(W)$pivot[seq_len(qr(W)$rank)]`` -- LINPACK ``dqrdc2``
    with tolerance 1e-7 (:func:`_dqrdc2_rank_columns`). Until 1.32 this was a
    one-pass Gram-Schmidt test, which on nearly collinear share matrices
    kept a different set (ADH: 781 columns against R's 776) and so a
    different AKM standard error.

    Even R's selection can leave a share matrix so ill-conditioned that the
    control-adjusted shocks, and with them the AKM SE, are numerically
    meaningless (ADH's raw 794-industry shares give an AKM SE of 1.5e4 in R
    and here). That is flagged with a warning; the remedy is to drop
    near-collinear industries before estimation, as BHJ's cleaned share
    file does.
    """
    keep = _dqrdc2_rank_columns(W)
    if len(keep) < W.shape[1]:
        warnings.warn(
            "Share matrix is collinear; dropping "
            f"{W.shape[1] - len(keep)} collinear share column(s) for AKM "
            "inference (as ShiftShareSE does).",
            UserWarning,
            stacklevel=3,
        )
    Wk = W[:, keep]
    if Wk.shape[1]:
        sv = np.linalg.svd(Wk, compute_uv=False)
        if sv[-1] <= 1e-8 * sv[0]:
            warnings.warn(
                "The share matrix is numerically near-singular even after "
                f"dropping exactly collinear columns (condition number "
                f"{sv[0] / max(sv[-1], np.finfo(float).tiny):.1e}); the AKM / "
                "AKM0 standard errors are not reliable. Drop near-collinear "
                "share columns (industries) before estimation.",
                UserWarning,
                stacklevel=3,
            )
    return np.sort(keep)


def _akm_fit(
    y: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    y2: Optional[np.ndarray] = None,
    region_cvar: Optional[np.ndarray] = None,
    beta0: float = 0.0,
    alpha: float = 0.05,
    w: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """AKM / AKM0 inference for OLS (``y2 is None``) or just-identified IV.

    ``w`` are regression weights, as ``ShiftShareSE``'s ``w`` argument: every
    projection becomes weighted least squares (``lm.wfit``) and every score
    carries its weight.

    ``y`` outcome, ``X`` shift-share variable, ``W`` shares, ``Z`` controls
    incl. intercept, ``y2`` the endogenous regressor instrumented by ``X``.
    Returns ``beta`` and dicts ``se`` / ``p`` / ``ci_l`` / ``ci_r`` keyed by
    ``Homoscedastic``, ``EHW``, ``Reg. cluster``, ``AKM``, ``AKM0``.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    W = np.asarray(W, dtype=float)
    keep = _drop_collinear_shares(W)
    W = W[:, keep]
    n = len(y)
    mm = np.column_stack([X, Z])
    p = int(np.linalg.matrix_rank(mm))

    wv = np.ones(n) if w is None else np.asarray(w, dtype=float)
    sw = np.sqrt(wv)

    def _wcoef(M: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(M * sw[:, None], v * sw, rcond=None)[0]

    def _wres(v: np.ndarray, M: np.ndarray) -> np.ndarray:
        return v - M @ _wcoef(M, v)

    ddX = _wres(X, Z)
    hX = _wcoef(W, ddX)
    coef1 = _wcoef(mm, y)
    res1 = y - mm @ coef1

    se = dict.fromkeys(_SE_ROWS, np.nan)
    if y2 is None:
        ddY = _wres(y, Z)
        beta = float(coef1[0])
        resid = res1
        RX = float(wv @ ddX**2)
        se["Homoscedastic"] = np.sqrt((wv @ resid**2) / (n - p) / RX)
        u = wv * resid * ddX
        se["EHW"] = np.sqrt((n / (n - p)) * (u @ u)) / RX
        if region_cvar is not None:
            uc = pd.Series(u).groupby(np.asarray(region_cvar)).sum().to_numpy()
            nc = len(uc)
            se["Reg. cluster"] = (
                np.sqrt((nc / (nc - 1)) * (n - 1) / (n - p) * (uc @ uc)) / RX
            )
        null_resid = ddY - ddX * beta0
        endog_dd = ddX
    else:
        y2 = np.asarray(y2, dtype=float)
        ddY1 = _wres(y, Z)
        ddY2 = _wres(y2, Z)
        coef2 = _wcoef(mm, y2)
        res2 = y2 - mm @ coef2
        beta = float(coef1[0] / coef2[0])
        resid = res1 - res2 * beta
        RX = float(wv @ (ddY2 * ddX))
        se["Homoscedastic"] = np.sqrt((wv @ resid**2) / n) / (
            np.sqrt(wv @ ddX**2) * abs(coef2[0])
        )
        u = wv * resid * ddX
        se["EHW"] = np.sqrt((u @ u) / RX**2)
        if region_cvar is not None:
            uc = pd.Series(u).groupby(np.asarray(region_cvar)).sum().to_numpy()
            se["Reg. cluster"] = np.sqrt((uc @ uc) / RX**2)
        null_resid = ddY1 - ddY2 * beta0
        endog_dd = ddY2

    cR = hX * (W.T @ (wv * resid))
    cR0 = hX * (W.T @ (wv * null_resid))
    cW = hX * (W.T @ (wv * endog_dd))
    se["AKM"] = np.sqrt(np.sum(cR**2)) / abs(RX)
    se0_akm0 = np.sqrt(np.sum(cR0**2)) / abs(RX)

    cv = stats.norm.ppf(1 - alpha / 2)
    Q = RX**2 / cv**2 - np.sum(cW**2)
    Q2 = np.sum(cR * cW) / Q
    mid = beta - Q2
    dis = Q2**2 + np.sum(cR**2) / Q
    if Q > 0:
        ci_akm0 = (mid - np.sqrt(dis), mid + np.sqrt(dis))
        se["AKM0"] = np.sqrt(dis) / cv
        akm0_ci_type = "bounded"
    elif dis > 0:
        # Union of two half-lines (-inf, lo] U [hi, inf), reported as
        # (lo, hi) with lo > hi, exactly as ShiftShareSE does.
        ci_akm0 = (mid + np.sqrt(dis), mid - np.sqrt(dis))
        se["AKM0"] = np.inf
        akm0_ci_type = "union"
    else:
        ci_akm0 = (-np.inf, np.inf)
        se["AKM0"] = np.inf
        akm0_ci_type = "real_line"

    pv, ci_l, ci_r = {}, {}, {}
    for row in _SE_ROWS:
        s = se0_akm0 if row == "AKM0" else se[row]
        pv[row] = 2 * stats.norm.sf(abs(beta - beta0) / s)
        if row == "AKM0":
            ci_l[row], ci_r[row] = ci_akm0
        else:
            ci_l[row] = beta - cv * se[row]
            ci_r[row] = beta + cv * se[row]
    return {
        "beta": beta,
        "se": {k: float(v) for k, v in se.items()},
        "p": pv,
        "ci_l": ci_l,
        "ci_r": ci_r,
        "akm0_se_null": float(se0_akm0),
        "akm0_ci_type": akm0_ci_type,
        "n_shares_used": int(W.shape[1]),
    }


def _bhj_aggregate(
    y: np.ndarray,
    x: np.ndarray,
    S: np.ndarray,
    g: np.ndarray,
    Z: np.ndarray,
    shock_ids,
    y_name: str,
    x_name: str,
    w: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """BHJ shock-level aggregation plus the shock-level IV (HC0).

    ``w`` are location weights (``ssaggregate``'s ``l_weights``): the
    controls are partialled out by weighted least squares and a location's
    exposure enters the shock-level means and ``s_n`` with its weight, so the
    shock-level IV reproduces the *weighted* location-level shift-share IV.

    ``Z`` are the location-level controls including the intercept. Returns
    the aggregated frame (``shock``, ``s_n``, ``y``, ``x``, ``g``) and the
    shock-level IV coefficient / HC0 SE of ``y_n`` on ``x_n`` instrumented
    by ``g_n`` with an intercept and weights ``s_n``.
    """
    wv = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    sw = np.sqrt(wv)

    def _wres(v: np.ndarray, M: np.ndarray) -> np.ndarray:
        return v - M @ np.linalg.lstsq(M * sw[:, None], v * sw, rcond=None)[0]

    y_perp = _wres(np.asarray(y, dtype=float), Z)
    x_perp = _wres(np.asarray(x, dtype=float), Z)

    tot = S.sum(axis=1)
    if np.std(tot, ddof=1) > 1e-5:
        # Incomplete shares: the shock-level IV equals the location-level
        # shift-share IV only if the sum of shares is spanned by the controls.
        resid_tot = _wres(tot, Z)
        ss_res = np.sum(wv * resid_tot**2)
        ss_tot = np.sum(wv * (tot - np.average(tot, weights=wv)) ** 2)
        if 1 - ss_res / ss_tot < 0.9999:
            warnings.warn(
                "Incomplete shares (the sum of exposure shares varies across "
                "locations) and the controls do not span the sum of shares: "
                "the BHJ shock-level IV coefficient does not equal the "
                "location-level shift-share IV. Add the sum of shares as a "
                "control.",
                UserWarning,
                stacklevel=3,
            )

    S = S * wv[:, None]
    S_n = S.sum(axis=0)
    ok = S_n > 0
    if not np.all(ok):
        warnings.warn(
            f"{int((~ok).sum())} shock(s) have zero total exposure and are "
            "dropped from the shock-level aggregation.",
            UserWarning,
            stacklevel=3,
        )
    Sk = S[:, ok]
    Sn = S_n[ok]
    s_n = Sn / Sn.sum()
    ybar = (Sk.T @ y_perp) / Sn
    xbar = (Sk.T @ x_perp) / Sn
    gk = np.asarray(g, dtype=float)[ok]

    Zm = np.column_stack([np.ones_like(gk), gk])
    Xm = np.column_stack([np.ones_like(xbar), xbar])
    A = Zm.T @ (s_n[:, None] * Xm)
    coef = np.linalg.solve(A, Zm.T @ (s_n * ybar))
    e = ybar - Xm @ coef
    meat = Zm.T @ ((s_n * e)[:, None] ** 2 * Zm)
    Ainv = np.linalg.inv(A)
    V = Ainv @ meat @ Ainv.T
    frame = pd.DataFrame(
        {
            "shock": np.asarray(shock_ids)[ok],
            "s_n": s_n,
            y_name: ybar,
            x_name: xbar,
            "g": gk,
        }
    )
    return {
        "shock_data": frame,
        "beta": float(coef[1]),
        "se_hc0": float(np.sqrt(V[1, 1])),
    }
