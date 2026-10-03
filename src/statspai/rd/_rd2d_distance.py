"""Distance-based boundary discontinuity estimation (port of R ``rd2d.distance``).

Numerical core behind :func:`statspai.rd2d` with ``approach="distance"``
and :func:`statspai.rd2d_bw` with ``approach="distance"``.  Each function
mirrors one routine of R ``rd2d`` 1.0.0 (named in its docstring).  The
input is an ``(n, neval)`` matrix of *signed* distances: column ``j``
holds each unit's distance to boundary point ``j``, non-negative on the
treated side (R's convention ``d = distance >= 0``).

Scope: sharp and fuzzy designs, ``vce`` in {hc0, hc1, hc2, hc3}, cluster-
robust variances, joint and separate fits, every ``bwselect``, known and
unknown kinks.  Covariate adjustment is not ported.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional

import numpy as np

from ..exceptions import DataInsufficient
from ._rd2d_location import (
    _KERNEL_CANON,
    _ROT_CONSTS,
    _bwselect_base,
    _is_cer,
    _is_common,
    cer_factor,
    cluster_sums,
    joint_scale,
    kernel_weight,
    unique_in_order,
    xx_inv,
)

__all__ = ["rd2d_distance_bw", "rd2d_distance_fit"]


def _bw_rate(n: Any, frm: float, to: float) -> Any:
    """R ``rd2d_bw_rate_factor``."""
    return np.asarray(n, dtype=float) ** (frm - to)


def _bwcheck_limits(dist: np.ndarray, bwcheck: int) -> Any:
    """R ``rd2d_distance_bwcheck_limits``."""
    n = len(dist)
    if n == 0:
        return float("nan"), float("nan")
    k = min(int(bwcheck), n)
    return float(np.partition(dist, k - 1)[k - 1]), float(dist.max())


def _unique_counts(dist: np.ndarray) -> Any:
    """R ``rd2d_distance_unique_counts``."""
    u = np.unique(dist)
    m0 = int((u < 0).sum())
    return m0 + (len(u) - m0), m0, len(u) - m0


def masspoint_counts(D: np.ndarray, masspoints: str, p: int, bwcheck: Any) -> Any:
    """Unique-distance counts per column and the mass-point warning."""
    N, neval = D.shape
    n1 = int((D[:, 0] >= 0).sum())
    M = np.full(neval, N, dtype=float)
    M0 = np.full(neval, N - n1, dtype=float)
    M1 = np.full(neval, n1, dtype=float)
    if masspoints in ("check", "adjust"):
        mass = False
        for j in range(neval):
            M[j], M0[j], M1[j] = _unique_counts(D[:, j])
            if 1 - M[j] / N >= 0.2:
                mass = True
        if mass:
            warnings.warn(
                "Mass points detected in the running variables.",
                RuntimeWarning,
                stacklevel=3,
            )
            if masspoints == "check":
                warnings.warn(
                    "Try using option masspoints=adjust.", RuntimeWarning, stacklevel=3
                )
            if bwcheck is None:
                bwcheck = 50 + p + 1
    return M, M0, M1, bwcheck


def _vce_mult(
    vce: Any,
    eN0: Any,
    eN1: Any,
    p: Any,
    sw0: Any,
    sw1: Any,
    iG0: Any,
    iG1: Any,
    joint: Any,
    clustered: Any,
) -> Any:
    if vce == "hc0":
        return 1.0, 1.0
    if vce == "hc1":
        if clustered and joint:
            return 1.0, 1.0
        if joint:
            w = math.sqrt((eN0 + eN1) / (eN0 + eN1 - 2 * (p + 1)))
            return w, w
        return math.sqrt(eN0 / (eN0 - p - 1)), math.sqrt(eN1 / (eN1 - p - 1))
    h0 = np.sum((sw0 @ iG0) * sw0, axis=1)
    h1 = np.sum((sw1 @ iG1) * sw1, axis=1)
    if vce == "hc2":
        return np.sqrt(1 / (1 - h0)), np.sqrt(1 / (1 - h1))
    return 1 / (1 - h0), 1 / (1 - h1)


def _dist_vce(
    wR: Any, resd: Any, eC: Any, h: Any, k_df: Any, cluster_df: Any, clusters: Any
) -> Any:
    """R ``rd2d_distance_vce``."""
    if eC is None:
        s = resd[:, None] * wR
        return (s.T @ s) * h**2
    n = len(eC)
    g = len(np.unique(eC))
    sc = cluster_sums(wR * resd[:, None], eC, clusters)
    scale = ((n - 1) / (n - k_df)) * (g / (g - 1)) if cluster_df else 1.0
    return (sc.T @ sc) * h**2 * scale


def rd2d_distance_fit(
    y: np.ndarray,
    D: np.ndarray,
    h: np.ndarray,
    p: int,
    kernel: str,
    vce: str,
    bwcheck: Optional[int],
    cluster: Optional[np.ndarray],
    fitmethod: str,
) -> Dict[str, np.ndarray]:
    """R ``rd2d_distance_fit`` (no covariates) with the projected halves.

    Returns per-point ``h0, h1, N0, N1, mu0, mu1, se0, se1`` and the
    projected influence matrices ``P0`` / ``P1`` (rows = observations or
    clusters, columns = evaluation points) from which R builds every
    covariance table.
    """
    N, neval = D.shape
    joint = fitmethod == "joint"
    clustered = cluster is not None
    clusters = unique_in_order(cluster) if clustered else None
    out = {
        k: np.empty(neval) for k in ("h0", "h1", "N0", "N1", "mu0", "mu1", "se0", "se1")
    }
    nrow = len(clusters) if clustered else N
    P0 = np.zeros((nrow, neval))
    P1 = np.zeros((nrow, neval))
    for i in range(neval):
        dist = D[:, i]
        d = dist >= 0
        dist = np.abs(dist)
        s0, s1 = ~d, d
        dd0, dd1 = dist[s0], dist[s1]
        y0, y1 = y[s0], y[s1]
        c0 = cluster[s0] if clustered else None
        c1 = cluster[s1] if clustered else None
        h0, h1 = float(h[i, 0]), float(h[i, 1])
        if bwcheck is not None:
            mn0, mx0 = _bwcheck_limits(dd0, bwcheck)
            mn1, mx1 = _bwcheck_limits(dd1, bwcheck)
            h0 = min(max(h0, mn0), mx0)
            h1 = min(max(h1, mn1), mx1)
        w0 = kernel_weight(dd0 / h0, kernel) / h0**2
        w1 = kernel_weight(dd1 / h1, kernel) / h1**2
        i0, i1 = w0 > 0, w1 > 0
        eN0, eN1 = int(i0.sum()), int(i1.sum())
        u0, u1 = dd0[i0], dd1[i1]
        ew0, ew1 = w0[i0], w1[i1]
        R0 = (u0 / h0)[:, None] ** np.arange(p + 1)[None, :]
        R1 = (u1 / h1)[:, None] ** np.arange(p + 1)[None, :]
        sw0 = np.sqrt(ew0)[:, None] * R0
        sw1 = np.sqrt(ew1)[:, None] * R1
        iG0, iG1 = xx_inv(sw0), xx_inv(sw1)
        sh0, sh1 = ew0[:, None] * R0, ew1[:, None] * R1
        b0 = iG0 @ (R0.T @ (y0[i0] * ew0))
        b1 = iG1 @ (R1.T @ (y1[i1] * ew1))
        r0 = y0[i0] - R0 @ b0
        r1 = y1[i1] - R1 @ b1
        wv0, wv1 = _vce_mult(vce, eN0, eN1, p, sw0, sw1, iG0, iG1, joint, clustered)
        r0, r1 = r0 * wv0, r1 * wv1
        eC0 = c0[i0] if clustered else None
        eC1 = c1[i1] if clustered else None
        k = p + 1
        cluster_df = not (joint and clustered)
        if not clustered:
            half0 = (sh0 * r0[:, None]) @ iG0
            half1 = (sh1 * r1[:, None]) @ iG1
        else:
            n0, n1 = len(eC0), len(eC1)
            g0, g1 = len(np.unique(eC0)), len(np.unique(eC1))
            ww0 = ((n0 - 1) / (n0 - k)) * (g0 / (g0 - 1)) if cluster_df else 1.0
            ww1 = ((n1 - 1) / (n1 - k)) * (g1 / (g1 - 1)) if cluster_df else 1.0
            half0 = (
                cluster_sums(sh0 * r0[:, None], eC0, clusters) * math.sqrt(ww0)
            ) @ iG0
            half1 = (
                cluster_sums(sh1 * r1[:, None], eC1, clusters) * math.sqrt(ww1)
            ) @ iG1
            if joint:
                sc = joint_scale(
                    vce, eN0 + eN1, 2 * k, np.concatenate([eC0, eC1]), True
                )
                half0, half1 = half0 * sc, half1 * sc
        sig0 = _dist_vce(sh0, r0, eC0, h0, p + 1, cluster_df, clusters)
        sig1 = _dist_vce(sh1, r1, eC1, h1, p + 1, cluster_df, clusters)
        cc0 = iG0.T @ (sig0 @ iG0)
        cc1 = iG1.T @ (sig1 @ iG1)
        if joint and clustered:
            sc = joint_scale(
                vce, eN0 + eN1, 2 * (p + 1), np.concatenate([eC0, eC1]), True
            )
            cc0, cc1 = cc0 * sc**2, cc1 * sc**2
        out["se0"][i] = math.sqrt(cc0[0, 0] / h0**2)
        out["se1"][i] = math.sqrt(cc1[0, 0] / h1**2)
        out["mu0"][i], out["mu1"][i] = b0[0], b1[0]
        out["h0"][i], out["h1"][i] = h0, h1
        out["N0"][i], out["N1"][i] = eN0, eN1
        if clustered:
            P0[:, i] = half0[:, 0]
            P1[:, i] = half1[:, 0]
        else:
            idx0 = np.flatnonzero(s0)[i0]
            idx1 = np.flatnonzero(s1)[i1]
            P0[idx0, i] = half0[:, 0]
            P1[idx1, i] = half1[:, 0]
    out["P0"], out["P1"] = P0, P1
    return out


def dist_cov(fa: Any, fb: Any, clustered_joint: bool) -> np.ndarray:
    """R ``rd2d_distance_cov_from_projects`` (both sides)."""
    if clustered_joint:
        return (fa["P1"] - fa["P0"]).T @ (fb["P1"] - fb["P0"])
    return fa["P0"].T @ fb["P0"] + fa["P1"].T @ fb["P1"]


def _poly_lm(y: np.ndarray, x: np.ndarray, degree: int) -> Any:
    """R ``rd2d_distance_poly_lm``: OLS of y on 1, x, ..., x^degree."""
    X = x[:, None] ** np.arange(degree + 1)[None, :]
    k = X.shape[1]
    if X.shape[0] <= k:
        raise DataInsufficient(
            "Too few observations below the cqt quantile to fit the pilot "
            f"polynomial of degree {degree}."
        )
    iXX = xx_inv(X)
    coef = iXX @ (X.T @ y)
    res = y - X @ coef
    s2 = float(res @ res) / (X.shape[0] - k)
    se = np.sqrt(np.maximum(0.0, np.diag(iXX) * s2))
    return coef, se


def _rot_distance(dist: np.ndarray, kernel: str) -> float:
    """R ``rdbw2d_distance_rot``."""
    mu2K, l2K = _ROT_CONSTS[_KERNEL_CANON[kernel]]
    var_hat = 0.5 * float(np.mean(dist**2))
    trace = 1.0 / (2 * math.pi * var_hat**3)
    return ((2 * l2K) / (len(dist) * mu2K * trace)) ** (1 / 6)


def _local_intercepts(y: Any, f: Any, dist: Any, h: Any, p: Any, kernel: Any) -> Any:
    """R ``rdbw2d_distance_local_intercepts_multi``."""
    w = kernel_weight(dist / h, kernel) / h**2
    ind = w > 0
    if ind.sum() <= p + 1:
        return float("nan"), float("nan")
    R = (dist[ind] / h)[:, None] ** np.arange(p + 1)[None, :]
    iG = xx_inv(np.sqrt(w[ind])[:, None] * R)
    by = iG @ (R.T @ (w[ind] * y[ind]))
    bf = iG @ (R.T @ (w[ind] * f[ind]))
    return float(by[0]), float(bf[0])


def _distance_bw_consts(
    y: Any,
    D: Any,
    p: Any,
    kernel: Any,
    vce: Any,
    cluster: Any,
    bwcheck: Any,
    cqt: Any,
    fuzzy: Any,
    bwparam: Any,
    fitmethod: Any,
) -> Any:
    """R ``rdbw2d_distance_bw`` (no covariates, ``rot = NULL``)."""
    N, neval = D.shape
    joint = fitmethod == "joint"
    clustered = cluster is not None
    clusters = unique_in_order(cluster) if clustered else None
    cols = [
        "h.0",
        "h.1",
        "b.0",
        "b.1",
        "v.0",
        "v.1",
        "v.01",
        "r.0",
        "r.1",
        "N.Co",
        "N.Tr",
        "bw.min.0",
        "bw.min.1",
        "bw.max.0",
        "bw.max.1",
    ]
    res = np.full((neval, len(cols)), np.nan)
    warned = False
    for i in range(neval):
        dist = D[:, i]
        d = dist >= 0
        dist = np.abs(dist)
        s0, s1 = ~d, d
        dn = _rot_distance(dist, kernel)
        dd0, dd1 = dist[s0], dist[s1]
        y0, y1 = y[s0].copy(), y[s1].copy()
        c0 = cluster[s0] if clustered else None
        c1 = cluster[s1] if clustered else None
        dn0 = dn1 = dn
        mn0 = mn1 = mx0 = mx1 = float("nan")
        if bwcheck is not None:
            mn0, mx0 = _bwcheck_limits(dd0, bwcheck)
            mn1, mx1 = _bwcheck_limits(dd1, bwcheck)
            dn0 = min(max(dn, mn0), mx0)
            dn1 = min(max(dn, mn1), mx1)
        if fuzzy is not None and bwparam == "main":
            f0, f1 = fuzzy[s0], fuzzy[s1]
            gy0, gf0 = _local_intercepts(y0, f0, dd0, dn0, p, kernel)
            gy1, gf1 = _local_intercepts(y1, f1, dd1, dn1, p, kernel)
            t_itt, t_fs = gy1 - gy0, gf1 - gf0
            if np.isfinite(t_fs) and abs(t_fs) > math.sqrt(np.finfo(float).eps):
                gi, gf = 1 / t_fs, -t_itt / t_fs**2
                y0 = gi * y0 + gf * f0
                y1 = gi * y1 + gf * f1
            elif not warned:
                warnings.warn(
                    "Weak or zero first-stage fuzzy RD estimate detected in bandwidth "
                    "selection; using reduced-form outcome bandwidth.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                warned = True
        th0 = float(np.quantile(dd0, cqt))
        th1 = float(np.quantile(dd1, cqt))
        f0i, f1i = dd0 <= th0, dd1 <= th1
        coef0, se0 = _poly_lm(y0[f0i], dd0[f0i], p + 1)
        coef1, se1 = _poly_lm(y1[f1i], dd1[f1i], p + 1)
        w0 = kernel_weight(dd0 / dn0, kernel) / dn0**2
        w1 = kernel_weight(dd1 / dn1, kernel) / dn1**2
        i0, i1 = w0 > 0, w1 > 0
        eN0, eN1 = int(i0.sum()), int(i1.sum())
        eX0, eX1 = dd0[i0], dd1[i1]
        eY0, eY1 = y0[i0], y1[i1]
        ew0, ew1 = w0[i0], w1[i1]
        pw = np.arange(p + 2)
        sd0 = eY0 - (eX0[:, None] ** pw[None, :]) @ coef0
        sd1 = eY1 - (eX1[:, None] ** pw[None, :]) @ coef1
        R0 = (eX0 / dn0)[:, None] ** np.arange(p + 1)[None, :]
        R1 = (eX1 / dn1)[:, None] ** np.arange(p + 1)[None, :]
        sw0 = np.sqrt(ew0)[:, None] * R0
        sw1 = np.sqrt(ew1)[:, None] * R1
        iG0, iG1 = xx_inv(sw0), xx_inv(sw1)
        sh0, sh1 = ew0[:, None] * R0, ew1[:, None] * R1
        wv0, wv1 = _vce_mult(vce, eN0, eN1, p, sw0, sw1, iG0, iG1, joint, clustered)
        sd0, sd1 = sd0 * wv0, sd1 * wv1
        eC0 = c0[i0] if clustered else None
        eC1 = c1[i1] if clustered else None
        cluster_df = not (joint and clustered)
        sig0 = _dist_vce(sh0, sd0, eC0, dn0, p + 1, cluster_df, clusters)
        sig1 = _dist_vce(sh1, sd1, eC1, dn1, p + 1, cluster_df, clusters)
        pm0 = R0.T @ ((eX0 / dn0) ** (p + 1) * ew0)
        pm1 = R1.T @ ((eX1 / dn1) ** (p + 1) * ew1)
        vec = np.zeros(p + 1)
        vec[0] = 1.0
        cf0 = float(vec @ iG0 @ pm0)
        cf1 = float(vec @ iG1 @ pm1)
        B0 = cf0 * coef0[p + 1]
        B1 = cf1 * coef1[p + 1]
        Rg0 = cf0**2 * se0[p + 1] ** 2
        Rg1 = cf1**2 * se1[p + 1] ** 2
        V0 = float(vec @ iG0 @ sig0 @ iG0 @ vec)
        V1 = float(vec @ iG1 @ sig1 @ iG1 @ vec)
        V01 = 0.0
        if joint and clustered:
            sc = joint_scale(
                vce, eN0 + eN1, 2 * (p + 1), np.concatenate([eC0, eC1]), True
            )
            Vh0 = (
                dn0 * (cluster_sums(sh0 * sd0[:, None], eC0, clusters) @ iG0 @ vec) * sc
            )
            Vh1 = (
                dn1 * (cluster_sums(sh1 * sd1[:, None], eC1, clusters) @ iG1 @ vec) * sc
            )
            V0, V1 = V0 * sc**2, V1 * sc**2
            V01 = float(Vh0 @ Vh1)
        Vd = V0 + V1 - 2 * V01
        hn = (2 * Vd / ((2 * p + 2) * ((B0 - B1) ** 2 + Rg0 + Rg1))) ** (
            1 / (2 * p + 4)
        )
        res[i] = (hn, hn, B0, B1, V0, V1, V01, Rg0, Rg1, eN0, eN1, mn0, mn1, mx0, mx1)
    return {c: res[:, j] for j, c in enumerate(cols)}


def _kink_distance(eval_pts: np.ndarray, kink: np.ndarray) -> np.ndarray:
    """R ``rd2d_distance_to_known_kink``."""
    if not kink.any():
        return np.full(len(eval_pts), np.inf)
    k = eval_pts[kink]
    dx = eval_pts[:, 0][:, None] - k[:, 0][None, :]
    dy = eval_pts[:, 1][:, None] - k[:, 1][None, :]
    return np.sqrt(dx**2 + dy**2).min(axis=1)


def rd2d_distance_bw(
    y: np.ndarray,
    D: np.ndarray,
    eval_pts: Optional[np.ndarray],
    p: int,
    kink_unknown: Any,
    kink_position: np.ndarray,
    kernel: str,
    bwselect: str,
    vce: str,
    bwcheck: Any,
    masspoints: str,
    cluster: Any,
    scaleregul: float,
    cqt: float,
    fitmethod: str,
    fuzzy: Optional[Any] = None,
    bwparam: str = "main",
) -> Dict[str, object]:
    """R ``rdbw2d.distance``: returns ``h0``, ``h1`` per evaluation point."""
    base = _bwselect_base(bwselect)
    common = _is_common(bwselect)
    M, M0, M1, bwcheck = masspoint_counts(D, masspoints, p, bwcheck)
    r = _distance_bw_consts(
        y, D, p, kernel, vce, cluster, bwcheck, cqt, fuzzy, bwparam, fitmethod
    )
    sr = scaleregul
    bc = bwcheck is not None
    if base == "mserd":
        Vd = r["v.0"] + r["v.1"] - 2 * r["v.01"]
        hn = (
            2
            * Vd
            / (
                (2 * p + 2)
                * ((r["b.0"] - r["b.1"]) ** 2 + sr * r["r.0"] + sr * r["r.1"])
            )
        ) ** (1 / (2 * p + 4))
        if bc:
            hn = np.minimum(
                np.maximum(np.maximum(hn, r["bw.min.0"]), r["bw.min.1"]),
                np.minimum(r["bw.max.0"], r["bw.max.1"]),
            )
        h0 = h1 = hn
    elif base == "msetwo":
        h0 = (2 * r["v.0"] / ((2 * p + 2) * (r["b.0"] ** 2 + sr * r["r.0"]))) ** (
            1 / (2 * p + 4)
        )
        h1 = (2 * r["v.1"] / ((2 * p + 2) * (r["b.1"] ** 2 + sr * r["r.1"]))) ** (
            1 / (2 * p + 4)
        )
        if bc:
            h0 = np.minimum(np.maximum(h0, r["bw.min.0"]), r["bw.max.0"])
            h1 = np.minimum(np.maximum(h1, r["bw.min.1"]), r["bw.max.1"])
    elif base == "imserd":
        VV = np.mean(r["v.0"] + r["v.1"] - 2 * r["v.01"])
        BB = np.mean((r["b.0"] - r["b.1"]) ** 2 + sr * r["r.0"] + sr * r["r.1"])
        hn = np.full(len(r["v.0"]), (2 * VV / ((2 * p + 2) * BB)) ** (1 / (2 * p + 4)))
        if bc:
            hn = np.minimum(
                np.maximum(np.maximum(hn, r["bw.min.0"]), r["bw.min.1"]),
                np.minimum(r["bw.max.0"], r["bw.max.1"]),
            )
        h0 = h1 = hn
    else:  # imsetwo
        B0 = np.mean(r["b.0"] ** 2 + sr * r["r.0"])
        B1 = np.mean(r["b.1"] ** 2 + sr * r["r.1"])
        h0 = np.full(
            len(r["v.0"]),
            (2 * np.mean(r["v.0"]) / ((2 * p + 2) * B0)) ** (1 / (2 * p + 4)),
        )
        h1 = np.full(
            len(r["v.1"]),
            (2 * np.mean(r["v.1"]) / ((2 * p + 2) * B1)) ** (1 / (2 * p + 4)),
        )
        if bc:
            h0 = np.minimum(np.maximum(h0, r["bw.min.0"]), r["bw.max.0"])
            h1 = np.minimum(np.maximum(h1, r["bw.min.1"]), r["bw.max.1"])
    h0 = np.array(h0, dtype=float)
    h1 = np.array(h1, dtype=float)
    smooth = 1 / (p + 4) if _is_cer(bwselect) else 1 / (2 * p + 4)
    if _is_cer(bwselect):
        if common:
            h0, h1 = h0 * cer_factor(M, p), h1 * cer_factor(M, p)
        else:
            h0, h1 = h0 * cer_factor(M0, p), h1 * cer_factor(M1, p)
    if kink_position.any():
        dk = _kink_distance(eval_pts, kink_position)
        if common:
            bound = np.maximum(h0 * _bw_rate(M, smooth, 0.25), dk)
            h0, h1 = np.minimum(h0, bound), np.minimum(h1, bound)
        else:
            h0n = np.minimum(h0, np.maximum(h0 * _bw_rate(M0, smooth, 0.25), dk))
            h1n = np.minimum(h1, np.maximum(h1 * _bw_rate(M1, smooth, 0.25), dk))
            h0, h1 = h0n, h1n
    if kink_unknown[0]:
        if common:
            h0, h1 = h0 * _bw_rate(M, smooth, 0.25), h1 * _bw_rate(M, smooth, 0.25)
        else:
            h0, h1 = h0 * _bw_rate(M0, smooth, 0.25), h1 * _bw_rate(M1, smooth, 0.25)
    return {
        "h0": h0,
        "h1": h1,
        "mseconsts": r,
        "M": M,
        "M0": M0,
        "M1": M1,
        "bwcheck": bwcheck,
    }


def rd2d_distance_estimate(
    y: np.ndarray,
    D: np.ndarray,
    eval_pts: Optional[np.ndarray],
    h: Any,
    p: int,
    q: Optional[int],
    kink_unknown: Any,
    kink_position: np.ndarray,
    kernel: str,
    bwselect: str,
    vce: str,
    bwcheck: Any,
    masspoints: str,
    cluster: Any,
    fitmethod: str,
    scaleregul: float,
    cqt: float,
    fuzzy: Optional[Any] = None,
    bwparam: str = "main",
) -> Dict[str, object]:
    """R ``rd2d.distance`` (no covariates, no uniform bands)."""
    N, neval = D.shape
    if q is None:
        q = p if kink_unknown[0] else p + 1
    common = _is_common(bwselect)
    bw_info = None
    if h is None:
        bw_info = rd2d_distance_bw(
            y,
            D,
            eval_pts,
            p,
            kink_unknown,
            kink_position,
            kernel,
            bwselect,
            vce,
            bwcheck,
            masspoints,
            cluster,
            scaleregul,
            cqt,
            fitmethod,
            fuzzy,
            bwparam,
        )
        hfull = np.column_stack([bw_info["h0"], bw_info["h1"]])
        bwselect_used = bwselect
    else:
        harr = np.asarray(h, dtype=float)
        if harr.size == 1:
            hfull = np.full((neval, 2), float(harr.reshape(-1)[0]))
        else:
            hfull = harr.reshape(neval, 2)
        bwselect_used = "user provided"
        common = False
    M, M0, M1, bwcheck = masspoint_counts(D, masspoints, p, bwcheck)
    clustered = cluster is not None
    cj = clustered and fitmethod == "joint"
    ys = [y] if fuzzy is None else [y, fuzzy]
    fp = [
        rd2d_distance_fit(v, D, hfull, p, kernel, vce, bwcheck, cluster, fitmethod)
        for v in ys
    ]
    hrbc = hfull.copy()
    if kink_unknown[1]:
        if common:
            hrbc = hfull * _bw_rate(M, 0.25, 1 / 3)[:, None]
        else:
            hrbc[:, 0] = hfull[:, 0] * _bw_rate(M0, 0.25, 1 / 3)
            hrbc[:, 1] = hfull[:, 1] * _bw_rate(M1, 0.25, 1 / 3)
    reuse = q == p and np.array_equal(hrbc, hfull)
    fq = (
        fp
        if reuse
        else [
            rd2d_distance_fit(v, D, hrbc, q, kernel, vce, bwcheck, cluster, fitmethod)
            for v in ys
        ]
    )
    out: Dict[str, object] = {
        "hfull": hfull,
        "hrbc": hrbc,
        "q": q,
        "bw_info": bw_info,
        "bwselect": bwselect_used,
        "M": M,
        "M0": M0,
        "M1": M1,
        "fit_p": fp,
        "fit_q": fq,
        "bwcheck": bwcheck,
    }
    denom_tol = math.sqrt(np.finfo(float).eps)
    for tag, fits in (("p", fp), ("q", fq)):
        f = fits[0]
        if fuzzy is None:
            tau = f["mu1"] - f["mu0"]
            cov = dist_cov(f, f, cj)
            cov = (cov + cov.T) / 2
            se = np.sqrt(np.diag(cov)) if cj else np.sqrt(f["se0"] ** 2 + f["se1"] ** 2)
            out[f"tau_{tag}"], out[f"se_{tag}"], out[f"cov_{tag}"] = tau, se, cov
            out[f"mu0_{tag}"], out[f"mu1_{tag}"] = f["mu0"], f["mu1"]
            out[f"se0_{tag}"], out[f"se1_{tag}"] = f["se0"], f["se1"]
        else:
            g = fits[1]
            t_itt, t_fs = f["mu1"] - f["mu0"], g["mu1"] - g["mu0"]
            valid = np.isfinite(t_fs) & (np.abs(t_fs) > denom_tol)
            est = np.full(neval, np.nan)
            est[valid] = t_itt[valid] / t_fs[valid]
            ci = dist_cov(f, f, cj)
            ci = (ci + ci.T) / 2
            cf = dist_cov(g, g, cj)
            cf = (cf + cf.T) / 2
            cif = dist_cov(f, g, cj)
            a, b = 1 / t_fs, -t_itt / t_fs**2
            cov = (
                np.outer(a, a) * ci
                + np.outer(a, b) * cif
                + np.outer(b, a) * cif.T
                + np.outer(b, b) * cf
            )
            cov[~np.outer(valid, valid)] = np.nan
            cov = (cov + cov.T) / 2
            out[f"tau_{tag}"], out[f"se_{tag}"], out[f"cov_{tag}"] = (
                est,
                np.sqrt(np.diag(cov)),
                cov,
            )
            out[f"itt_{tag}"], out[f"fs_{tag}"] = t_itt, t_fs
            out[f"se_itt_{tag}"] = np.sqrt(f["se0"] ** 2 + f["se1"] ** 2)
            out[f"se_fs_{tag}"] = np.sqrt(g["se0"] ** 2 + g["se1"] ** 2)
    if fuzzy is not None and not (
        np.all(np.isfinite(out["tau_p"])) and np.all(np.isfinite(out["tau_q"]))
    ):
        warnings.warn(
            "Weak or zero first-stage fuzzy RD estimates detected; returning NaN for "
            "affected fuzzy estimates.",
            RuntimeWarning,
            stacklevel=3,
        )
    return out
