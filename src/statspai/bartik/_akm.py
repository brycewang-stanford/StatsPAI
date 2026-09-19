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


def _drop_collinear_shares(W: np.ndarray) -> np.ndarray:
    """Keep share columns in order, dropping any in the span of earlier ones.

    Mirrors ``ShiftShareSE:::drop_collinear`` (R ``qr()``: LINPACK dqrdc2,
    tolerance 1e-7 relative to the column's own norm, limited pivoting that
    keeps the original order of the retained columns).
    """
    keep = []
    Q = np.zeros((W.shape[0], 0))
    for j in range(W.shape[1]):
        col = W[:, j]
        norm0 = np.linalg.norm(col)
        r = col - Q @ (Q.T @ col) if Q.shape[1] else col
        if norm0 > 0 and np.linalg.norm(r) > 1e-7 * norm0:
            keep.append(j)
            Q = np.column_stack([Q, r / np.linalg.norm(r)])
    if len(keep) < W.shape[1]:
        warnings.warn(
            "Share matrix is collinear; dropping "
            f"{W.shape[1] - len(keep)} collinear share column(s) for AKM "
            "inference (as ShiftShareSE does).",
            UserWarning,
            stacklevel=3,
        )
    return np.asarray(keep, dtype=int)


def _akm_fit(
    y: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    y2: Optional[np.ndarray] = None,
    region_cvar: Optional[np.ndarray] = None,
    beta0: float = 0.0,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """AKM / AKM0 inference for OLS (``y2 is None``) or just-identified IV.

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

    ddX = _resid(X, Z)
    hX = np.linalg.lstsq(W, ddX, rcond=None)[0]
    coef1 = np.linalg.lstsq(mm, y, rcond=None)[0]
    res1 = y - mm @ coef1

    se = dict.fromkeys(_SE_ROWS, np.nan)
    if y2 is None:
        ddY = _resid(y, Z)
        beta = float(coef1[0])
        resid = res1
        RX = float(ddX @ ddX)
        se["Homoscedastic"] = np.sqrt((resid @ resid) / (n - p) / RX)
        u = resid * ddX
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
        ddY1 = _resid(y, Z)
        ddY2 = _resid(y2, Z)
        coef2 = np.linalg.lstsq(mm, y2, rcond=None)[0]
        res2 = y2 - mm @ coef2
        beta = float(coef1[0] / coef2[0])
        resid = res1 - res2 * beta
        RX = float(ddY2 @ ddX)
        se["Homoscedastic"] = np.sqrt((resid @ resid) / n) / (
            np.sqrt(ddX @ ddX) * abs(coef2[0])
        )
        u = resid * ddX
        se["EHW"] = np.sqrt((u @ u) / RX**2)
        if region_cvar is not None:
            uc = pd.Series(u).groupby(np.asarray(region_cvar)).sum().to_numpy()
            se["Reg. cluster"] = np.sqrt((uc @ uc) / RX**2)
        null_resid = ddY1 - ddY2 * beta0
        endog_dd = ddY2

    cR = hX * (W.T @ resid)
    cR0 = hX * (W.T @ null_resid)
    cW = hX * (W.T @ endog_dd)
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
) -> Dict[str, object]:
    """BHJ shock-level aggregation plus the shock-level IV (HC0).

    ``Z`` are the location-level controls including the intercept. Returns
    the aggregated frame (``shock``, ``s_n``, ``y``, ``x``, ``g``) and the
    shock-level IV coefficient / HC0 SE of ``y_n`` on ``x_n`` instrumented
    by ``g_n`` with an intercept and weights ``s_n``.
    """
    y_perp = _resid(np.asarray(y, dtype=float), Z)
    x_perp = _resid(np.asarray(x, dtype=float), Z)

    tot = S.sum(axis=1)
    if np.std(tot, ddof=1) > 1e-5:
        # Incomplete shares: the shock-level IV equals the location-level
        # shift-share IV only if the sum of shares is spanned by the controls.
        fitted = Z @ np.linalg.lstsq(Z, tot, rcond=None)[0]
        ss_res = np.sum((tot - fitted) ** 2)
        ss_tot = np.sum((tot - tot.mean()) ** 2)
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
