"""Exposure-robust inference for shift-share regressions.

The Adao-Kolesar-Morales (2019) standard errors and null-imposed confidence
sets [@ado2019shift] for a regression whose regressor (OLS) or instrument
(just-identified IV) is a shift-share variable ``X_i = sum_s W_is g_s``, and
the Borusyak-Hull-Jaravel shock-level aggregation used by ``sp.ssaggregate``.

Clean-room implementation (2026-10-20). Written from the paper, the
``ShiftShareSE`` documentation (help pages and the vignette "Standard Errors
in Shift-Share Regressions" [@kolesar2022shiftsharese]) and black-box runs of
that package; the specification is ``docs/dev/cleanroom/akm.md``. No source
code of ``ShiftShareSE`` or of R's QR routine was read. The parity fixtures
(``tests/reference_parity/test_did_synth_shiftshare_parity.py``,
``test_shiftshare_weights_parity.py``, ``test_akm_collinear_shares_R_parity.py``,
``test_akm_ci_shapes_R_parity.py``) were produced by running the reference.

Notation (``w`` are location weights, one when absent):

* ``r(v)`` is the residual of ``v`` from the weighted least-squares
  regression on the controls ``Z``; ``Xr = r(X)``, ``y1r = r(y1)``,
  ``y2r = r(y2)`` (``y2 = X`` for OLS).
* ``beta = <Xr, y1r>_w / <Xr, y2r>_w`` and ``e = y1r - beta * y2r``.
* ``h`` are the coefficients of the weighted regression of ``Xr`` on the
  linearly independent share columns (the control-adjusted shocks), and
  ``R_s(u) = sum_i W_is w_i u_i`` the share-weighted sums of a residual.
* AKM: ``se = sqrt(sum_s (h_s R_s(e))^2) / |<Xr, y2r>_w|`` (vignette,
  feasible formula).
* AKM0: the confidence set ``{b : (<Xr, y1r - b y2r>_w)^2 <= z^2
  sum_s (h_s R_s(y1r - b y2r))^2}``, a quadratic inequality in ``b``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import NumericalInstability

_SE_ROWS = ("Homoscedastic", "EHW", "Reg. cluster", "AKM", "AKM0")

# A share matrix whose kept columns still have a condition number above this
# leaves the control-adjusted shocks numerically undetermined.
_SHARE_CONDITION_LIMIT = 1e8


def _resid(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Residual of ``v`` (vector or matrix) from least squares on ``M``."""
    coef = np.linalg.lstsq(M, v, rcond=None)[0]
    return v - M @ coef


def _weighted_resid(v: np.ndarray, Z: np.ndarray, sw: np.ndarray) -> np.ndarray:
    """Residual of ``v`` from weighted least squares on ``Z`` (weights ``sw**2``)."""
    coef = np.linalg.lstsq(Z * sw[:, None], v * sw, rcond=None)[0]
    return v - Z @ coef


def _limited_pivot_rank_columns(x: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Original indices of the columns a limited-pivoting QR keeps.

    Householder QR that walks the columns left to right. A column is
    accepted when the norm of its part not yet explained by the accepted
    columns (the trailing part after the reflections so far) is at least
    ``tol`` times its own original norm; otherwise it is set aside (moved to
    the right edge) and the next column is tried. This is the "limited
    pivoting" that R's ``?qr`` page describes for its default
    (``LAPACK = FALSE``, ``tol = 1e-7``) decomposition, and the rule the
    ``ShiftShareSE`` vignette (section 3.1) uses to drop collinear sectors.
    Black-box runs of R's ``qr()`` on 700 random designs, including columns
    whose relative residual norm straddles ``tol``, keep the same columns.

    The norms are recomputed from the reflected matrix at every step.
    """
    A = np.array(x, dtype=float, copy=True)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    n, p = A.shape
    original = np.sqrt(np.einsum("ij,ij->j", A, A))
    kept = []
    row = 0
    for j in range(p):
        if row >= n:
            break
        part = A[row:, j]
        norm = float(np.sqrt(part @ part))
        if original[j] == 0.0 or not norm >= tol * original[j]:
            continue
        v = part.copy()
        v[0] += norm if v[0] >= 0.0 else -norm
        vv = float(v @ v)
        if vv > 0.0:
            block = A[row:, j:]
            block -= np.outer(v, (2.0 / vv) * (v @ block))
        kept.append(j)
        row += 1
    return np.asarray(kept, dtype=np.int64)


def _independent_share_columns(W: np.ndarray) -> np.ndarray:
    """Share matrix restricted to its linearly independent columns.

    Columns dropped by :func:`_limited_pivot_rank_columns` are announced in a
    ``UserWarning`` (as the reference announces a collinear share matrix).
    If the kept columns are still numerically near-singular, a second
    warning says that the AKM / AKM0 standard errors are unreliable.
    """
    keep = _limited_pivot_rank_columns(W)
    n_dropped = W.shape[1] - len(keep)
    if n_dropped:
        warnings.warn(
            f"Share matrix is collinear; dropping {n_dropped} collinear share "
            "column(s) for AKM inference (the AKM shocks of the dropped "
            "sectors are absorbed by the kept ones, as in ShiftShareSE).",
            UserWarning,
            stacklevel=3,
        )
    kept = W[:, keep]
    if kept.shape[1]:
        sv = np.linalg.svd(kept, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
        if cond > _SHARE_CONDITION_LIMIT:
            warnings.warn(
                "The share matrix is numerically near-singular even after "
                f"dropping exactly collinear columns (condition number {cond:.3g}); "
                "the control-adjusted shocks, and hence the AKM / AKM0 "
                "standard errors, are not reliably determined. Aggregate "
                "sparse sectors or drop near-empty share columns.",
                UserWarning,
                stacklevel=3,
            )
    return kept


def _two_sided_p(stat: float) -> float:
    if not np.isfinite(stat):
        return 0.0 if np.isinf(stat) else float("nan")
    return float(2.0 * stats.norm.sf(abs(stat)))


def _null_imposed_set(
    a: float, b: float, c: np.ndarray, d: np.ndarray, z: float
) -> Tuple[float, float, str]:
    """Solve ``(a - b t)^2 <= z^2 sum_s (c_s - d_s t)^2`` for ``t``.

    Returns ``(lower, upper, kind)``. ``kind`` is ``"bounded"`` (the set is
    ``[lower, upper]``), ``"union"`` (the set is ``(-inf, upper] U
    [lower, inf)`` with ``upper < lower``) or ``"real_line"``.
    """
    z2 = z * z
    qa = b * b - z2 * float(d @ d)
    qb = -2.0 * (a * b - z2 * float(c @ d))
    qc = a * a - z2 * float(c @ c)
    disc = qb * qb - 4.0 * qa * qc
    if qa > 0.0:
        half = np.sqrt(max(disc, 0.0))
        return (-qb - half) / (2.0 * qa), (-qb + half) / (2.0 * qa), "bounded"
    if qa < 0.0 and disc > 0.0:
        half = np.sqrt(disc)
        # Roots of a downward parabola: the set is outside them.
        hi_root = (-qb - half) / (2.0 * qa)
        lo_root = (-qb + half) / (2.0 * qa)
        return hi_root, lo_root, "union"
    return -np.inf, np.inf, "real_line"


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

    Parameters
    ----------
    y : (n,) outcome.
    X : (n,) shift-share variable (the regressor for OLS, the instrument for
        IV).
    W : (n, S) share matrix.
    Z : (n, q) controls, including the intercept.
    y2 : (n,) endogenous regressor, or ``None`` for OLS.
    region_cvar : (n,) region clusters for the ``"Reg. cluster"`` row.
    beta0 : null value for the p-values.
    alpha : one minus the confidence level.
    w : (n,) non-negative location weights.

    Returns
    -------
    dict with ``beta``; ``se``, ``p``, ``ci_l``, ``ci_r`` keyed by the rows
    ``Homoscedastic``, ``EHW``, ``Reg. cluster`` (NaN without clusters),
    ``AKM`` and ``AKM0``; ``akm0_ci_type`` (``"bounded"``, ``"union"`` or
    ``"real_line"``; for ``"union"`` the set is ``(-inf, ci_r] U [ci_l, inf)``);
    ``akm0_se_null`` (the AKM standard error with the null imposed); and
    ``n_shares_used``.

    Row conventions are the reference's: for IV the homoscedastic variance
    uses ``RSS / n`` and EHW and region-cluster rows have no small-sample
    factor; for OLS the homoscedastic variance uses ``RSS / (n - p)``, EHW is
    multiplied by ``n / (n - p)`` and the cluster row by
    ``G / (G - 1) * (n - 1) / (n - p)``, with ``p = ncol(Z) + 1``. The AKM0
    "standard error" is the length of the confidence set over
    ``2 z_{1 - alpha/2}`` (infinite when the set is unbounded).
    """
    y1 = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float).ravel()
    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    ols = y2 is None
    y2v = X if ols else np.asarray(y2, dtype=float).ravel()
    n = y1.size
    wv = np.ones(n) if w is None else np.asarray(w, dtype=float).ravel()
    sw = np.sqrt(wv)

    Xr = _weighted_resid(X, Z, sw)
    y1r = _weighted_resid(y1, Z, sw)
    y2r = Xr if ols else _weighted_resid(y2v, Z, sw)

    num = float(np.sum(wv * Xr * y1r))
    den = float(np.sum(wv * Xr * y2r))
    scale = float(np.sqrt(np.sum(wv * Xr * Xr) * np.sum(wv * y2r * y2r)))
    if not abs(den) > 1e-14 * max(scale, 1e-300):
        raise NumericalInstability(
            "The shift-share variable is orthogonal to the regressor after "
            "partialling out the controls; the coefficient is not identified.",
            recovery_hint=(
                "Check that the shift-share instrument varies once the "
                "controls are partialled out (weak or absent first stage)."
            ),
        )
    beta = num / den
    e = y1r - beta * y2r

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    se: Dict[str, float] = {}

    # -- conventional rows ------------------------------------------------
    p_ctrl = Z.shape[1] + 1
    rss = float(np.sum(wv * e * e))
    score = wv * Xr * e
    if ols:
        sigma2 = rss / (n - p_ctrl)
        ehw_factor = n / (n - p_ctrl)
    else:
        sigma2 = rss / n
        ehw_factor = 1.0
    se["Homoscedastic"] = float(np.sqrt(sigma2 * np.sum(wv * Xr * Xr)) / abs(den))
    se["EHW"] = float(np.sqrt(ehw_factor * np.sum(score * score)) / abs(den))
    if region_cvar is not None:
        codes = pd.factorize(pd.Series(np.asarray(region_cvar)), sort=False)[0]
        n_groups = int(codes.max()) + 1
        sums = np.bincount(codes, weights=score, minlength=n_groups)
        factor = 1.0
        if ols:
            factor = n_groups / (n_groups - 1) * (n - 1) / (n - p_ctrl)
        se["Reg. cluster"] = float(np.sqrt(factor * np.sum(sums * sums)) / abs(den))
    else:
        se["Reg. cluster"] = float("nan")

    # -- AKM ----------------------------------------------------------------
    shares = _independent_share_columns(W)
    shock_adj = np.linalg.lstsq(shares * sw[:, None], Xr * sw, rcond=None)[0]
    Wt = shares.T * wv  # (S, n): column i scaled by w_i

    def _share_sums(u: np.ndarray) -> np.ndarray:
        return shock_adj * (Wt @ u)

    akm_terms = _share_sums(e)
    se["AKM"] = float(np.sqrt(akm_terms @ akm_terms) / abs(den))

    # -- AKM0 (null imposed) ------------------------------------------------
    c = _share_sums(y1r)
    d = _share_sums(y2r)
    lo, hi, kind = _null_imposed_set(num, den, c, d, z)
    null_terms = c - beta0 * d
    se_null = float(np.sqrt(null_terms @ null_terms) / abs(den))
    se["AKM0"] = float((hi - lo) / (2.0 * z)) if kind == "bounded" else float("inf")

    p: Dict[str, float] = {}
    ci_l: Dict[str, float] = {}
    ci_r: Dict[str, float] = {}
    for row in ("Homoscedastic", "EHW", "Reg. cluster", "AKM"):
        s = se[row]
        if np.isnan(s):
            p[row] = ci_l[row] = ci_r[row] = float("nan")
            continue
        p[row] = _two_sided_p((beta - beta0) / s) if s > 0 else float("nan")
        ci_l[row] = beta - z * s
        ci_r[row] = beta + z * s
    p["AKM0"] = _two_sided_p((beta - beta0) / se_null) if se_null > 0 else float("nan")
    ci_l["AKM0"] = float(lo)
    ci_r["AKM0"] = float(hi)

    return {
        "beta": float(beta),
        "se": se,
        "p": p,
        "ci_l": ci_l,
        "ci_r": ci_r,
        "akm0_se_null": se_null,
        "akm0_ci_type": kind,
        "n_shares_used": int(shares.shape[1]),
    }


def _bhj_aggregate(
    y: np.ndarray,
    x: np.ndarray,
    S: np.ndarray,
    g: np.ndarray,
    Z: np.ndarray,
    shock_ids: Any,
    y_name: str,
    x_name: str,
    w: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Shock-level aggregation of Borusyak, Hull and Jaravel, plus its IV.

    ``y`` and ``x`` are residualised on the controls (weighted least squares
    with the location weights ``w``) and collapsed to one row per shock with
    exposure weights ``e_n = sum_l w_l S_ln``: ``s_n = e_n / sum_n e_n`` and
    ``ybar_n = sum_l w_l S_ln y_l / e_n`` (likewise ``x``). The shock-level
    IV regresses ``ybar`` on ``xbar`` and an intercept, instrumenting
    ``xbar`` with the shocks ``g``, weighted by ``s_n``, with the HC0
    sandwich SE. When the controls do not span the sum of shares, the
    shock-level coefficient is not the location-level one, and a warning
    says so.

    Returns ``{"shock_data", "beta", "se_hc0"}``; ``shock_data`` has the
    columns ``shock``, ``s_n``, ``y_name``, ``x_name`` and ``g``.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    S = np.asarray(S, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    g = np.asarray(g, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    n = y.size
    wv = np.ones(n) if w is None else np.asarray(w, dtype=float).ravel()
    sw = np.sqrt(wv)

    share_sum = S.sum(axis=1)
    left = _weighted_resid(share_sum, Z, sw)
    if np.sqrt(np.sum(wv * left * left)) > 1e-8 * max(
        np.sqrt(np.sum(wv * share_sum * share_sum)), 1e-300
    ):
        warnings.warn(
            "Incomplete shares (the sum of exposure shares varies across "
            "locations) and the controls do not span the sum of shares: the "
            "BHJ shock-level IV coefficient does not equal the location-level "
            "shift-share IV. Add the sum of shares as a control.",
            UserWarning,
            stacklevel=3,
        )

    yr = _weighted_resid(y, Z, sw)
    xr = _weighted_resid(x, Z, sw)
    exposure = S.T @ wv
    s_n = exposure / exposure.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        ybar = (S.T @ (wv * yr)) / exposure
        xbar = (S.T @ (wv * xr)) / exposure

    shock_data = pd.DataFrame(
        {
            "shock": np.asarray(shock_ids),
            "s_n": s_n,
            y_name: ybar,
            x_name: xbar,
            "g": g,
        }
    )

    ok = exposure > 0
    regs = np.column_stack([np.ones(ok.sum()), xbar[ok]])
    inst = np.column_stack([np.ones(ok.sum()), g[ok]])
    wk = s_n[ok]
    cross = inst.T @ (regs * wk[:, None])
    coef = np.linalg.solve(cross, inst.T @ (wk * ybar[ok]))
    u = ybar[ok] - regs @ coef
    bread = np.linalg.inv(cross)
    meat_rows = inst * (wk * u)[:, None]
    vcov = bread @ (meat_rows.T @ meat_rows) @ bread.T
    return {
        "shock_data": shock_data,
        "beta": float(coef[1]),
        "se_hc0": float(np.sqrt(vcov[1, 1])),
    }
