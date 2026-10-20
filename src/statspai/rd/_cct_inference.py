"""Robust bias-corrected local polynomial RD estimate for given ``(h, b)``.

Clean-room rewrite (2026-10-20) from Calonico, Cattaneo and Titiunik (2014,
bib key ``calonico2014robust``, eq. (3.1)-(3.4) and Section 5) and the
covariate-adjusted, fuzzy and clustered extensions in Calonico, Cattaneo,
Farrell and Titiunik (2019, Review of Economics and Statistics 101(3),
Supplemental Appendix §12.4-§12.5 and Part III). R ``rdrobust`` 4.0.0
(GPL-3) was used only as a black box; the conventions it fixed are listed
in ``docs/dev/cleanroom/rd_cct.md``.

On each side, with ``P_p`` the order-``p`` design, ``W_h`` the kernel weights
at ``h`` and ``G_p = P_p' W_h P_p``, the conventional estimator of the
``nu``-th derivative is ``nu! e_nu' G_p^-1 P_p' W_h C``. Its leading bias is
``nu! e_nu' G_p^-1 (P_p' W_h x^(p+1)) beta_(p+1)``; replacing
``beta_(p+1)`` by the order-``q`` fit at ``b`` gives the bias-corrected
estimator, which is again linear in the data:

    a_bc,i = a_cl,i - [nu! e_nu' G_p^-1 P_p' W_h x^(p+1)] * [e_(p+1)' G_q^-1 P_q,i W_b,i].

Both variances are sandwiches of these weights (the robust one uses
``a_bc``, so it carries the variance of the bias estimate).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import NumericalInstability
from ._cct_bandwidth import _drop_incomplete
from ._cct_side import (
    Side,
    cluster_adjusted_residuals,
    cluster_kind,
    kernel_name,
    variance_of,
)

__all__ = ["cct_bias_corrected"]


def _side_operators(
    side: Side,
    h: float,
    b: float,
    p: int,
    q: int,
    nu: int,
    kernel: str,
    vce: str,
    nnmatch: int,
) -> Dict[str, np.ndarray]:
    """Weights, residuals and the window shared by both fits on one side."""
    fit_p = side.fit(h, p, kernel)
    fit_q = side.fit(b, q, kernel)
    if fit_p.singular or fit_q.singular:
        raise NumericalInstability(
            "rdrobust: the local polynomial design is singular inside the "
            f"bandwidth (h={h:.4g}, b={b:.4g}): too few distinct values of the "
            "running variable for the polynomial order.",
            recovery_hint=(
                "Use a larger bandwidth or a lower polynomial order, or a method "
                "built for a discrete running variable (sp.rd_discrete)."
            ),
            diagnostics={"h": h, "b": b, "p": p, "q": q},
            alternative_functions=["sp.rd_discrete", "sp.rdrandinf"],
        )
    union = fit_p.mask | fit_q.mask
    n_union = int(union.sum())

    def spread(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.zeros(n_union)
        out[mask[union]] = values
        return out

    ell = np.zeros(p + 1)
    ell[nu] = math.factorial(nu)
    a_cl = spread(fit_p.weights_for(ell), fit_p.mask)
    pick = np.zeros(q + 1)
    pick[p + 1] = 1.0
    a_bc = a_cl - fit_p.bias_factor(ell) * spread(fit_q.weights_for(pick), fit_q.mask)

    # Residuals live on the union of the two windows: nearest neighbours are
    # searched among all its observations; plug-in residuals come from the
    # order-p fit at h (conventional) and the order-q fit at b (robust), with
    # the hc1 / cluster degrees of freedom counted on the union (black box).
    cols = side.cols[union]
    x_u = side.x[union]
    if vce == "nn" and side.cluster is None:
        ties = np.zeros(n_union, dtype=int)
        from ._nn_variance import nn_residuals

        rho = nn_residuals(x_u, cols, ties, ties, nnmatch)
        rho_cl = rho_bc = rho
    else:
        rho_cl = _plug_in_rows(side, fit_p, union, x_u, cols, vce, n_union)
        rho_bc = _plug_in_rows(side, fit_q, union, x_u, cols, vce, n_union)
    return {
        "union": union,
        "cols": cols,
        "a_cl": a_cl,
        "a_bc": a_bc,
        "rho_cl": rho_cl,
        "rho_bc": rho_bc,
        "coef_p": fit_p.coef,
        "fit_p": fit_p,
    }


def _plug_in_rows(side, fit, union, x_u, cols, vce, n_union) -> np.ndarray:
    """Plug-in residual rows of ``fit`` for every observation of the union."""
    design = np.vander(x_u, fit.order + 1, increasing=True)
    resid = cols - design @ fit.coef
    if side.cluster is not None:
        kind = cluster_kind(vce)
        if kind == "cr1":
            return resid
        # CR2 / CR3 from this fit's hat matrix over the whole union window
        # (rows outside the fit's own window have zero weight).
        w_u = np.zeros(n_union)
        w_u[fit.mask[union]] = fit.w
        return cluster_adjusted_residuals(
            design, w_u, fit.ginv, side.cluster[union], resid, kind
        )
    kind = vce
    if kind == "hc0":
        return resid
    if kind == "hc1":
        k = fit.order + 1
        return resid * math.sqrt(n_union / (n_union - k))
    lev = np.zeros(n_union)
    lev[fit.mask[union]] = fit.leverage()
    if kind == "hc2":
        return resid / np.sqrt(1.0 - lev)[:, None]
    return resid / (1.0 - lev)[:, None]


def _joint_gamma(sides: List[Side], fits: list, n_core: int) -> np.ndarray:
    """Covariate slopes common to both sides (CCFT 2019 eq. (3)).

    The joint WLS of each core column on side-specific polynomials and the
    covariates; by Frisch-Waugh-Lovell the slope pools the within-side
    residual cross-products.
    """
    d = sides[0].cols.shape[1] - n_core
    zz = np.zeros((d, d))
    zy = np.zeros((d, n_core))
    for side, fit in zip(sides, fits):
        resid = side.cols[fit.mask] - fit.design @ fit.coef
        z = resid[:, n_core:]
        zw = z * fit.w[:, None]
        zz += zw.T @ z
        zy += zw.T @ resid[:, :n_core]
    return np.linalg.solve(zz, zy)


def cct_bias_corrected(
    y: np.ndarray,
    x: np.ndarray,
    c: float,
    h_l: float,
    h_r: float,
    b_l: float,
    b_r: float,
    p: int,
    q: int,
    deriv: int,
    kernel: str,
    nnmatch: int = 3,
    covs: Optional[np.ndarray] = None,
    vce: str = "nn",
    cluster: Optional[np.ndarray] = None,
    components: Optional[Dict[str, float]] = None,
    fuzzy: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """Conventional and robust bias-corrected RD estimate with their SEs.

    Parameters
    ----------
    y, x : array
        Outcome and running variable.
    c : float
        Cutoff.
    h_l, h_r, b_l, b_r : float
        Main and bias bandwidths on each side.
    p, q, deriv : int
        Polynomial orders and derivative of interest.
    kernel : {'triangular', 'uniform', 'epanechnikov'}
    nnmatch : int
        Neighbours for ``vce='nn'``.
    covs, cluster, fuzzy, weights : array, optional
        Covariates (common slopes from the joint order-``p`` fit at ``h``),
        cluster ids (CR1 small-sample factor), treatment indicator (the
        effect is the ratio of the outcome and treatment jumps, its bias
        correction and variance by the delta method) and observation weights.
    vce : {'nn', 'hc0', 'hc1', 'hc2', 'hc3', 'cr1', 'cr2', 'cr3'}
        With ``cluster``: ``nn`` / ``hc0`` / ``hc1`` / ``cr1`` are CR1,
        ``hc2`` / ``cr2`` CR2 and ``hc3`` / ``cr3`` CR3 (R rdrobust 4.0.0).
        Without it ``cr1`` / ``cr2`` / ``cr3`` are ``hc1`` / ``hc2`` / ``hc3``.
    components : dict, optional
        Filled with each side's conventional and robust variance,
        ``V_cl_left``, ``V_cl_right``, ``V_rb_left``, ``V_rb_right``.

    Returns
    -------
    tuple
        ``(tau_conventional, tau_bias_corrected, se_conventional, se_robust)``.

    References
    ----------
    calonico2014robust
    """
    kernel = kernel_name(kernel)
    vce = str(vce).lower()
    if cluster is None and vce in ("cr1", "cr2", "cr3"):
        vce = "hc" + vce[2]
    y, x, covs, fuzzy, weights, cluster = _drop_incomplete(
        y, x, covs, fuzzy, weights, cluster
    )
    xc = x - c
    cols: List[np.ndarray] = [np.asarray(y, dtype=float)]
    if fuzzy is not None:
        cols.append(np.asarray(fuzzy, dtype=float))
    n_core = len(cols)
    if covs is not None:
        z = np.asarray(covs, dtype=float)
        z = z[:, None] if z.ndim == 1 else z
        cols.extend(z[:, j] for j in range(z.shape[1]))
    mat = np.column_stack(cols)
    w = None if weights is None else np.asarray(weights, dtype=float)
    cl = None if cluster is None else np.asarray(cluster)
    sides = []
    for m in (xc < 0, xc >= 0):
        sides.append(
            Side(
                xc[m],
                mat[m],
                None if w is None else w[m],
                None if cl is None else cl[m],
            )
        )
    ops = [
        _side_operators(sides[0], h_l, b_l, p, q, deriv, kernel, vce, nnmatch),
        _side_operators(sides[1], h_r, b_r, p, q, deriv, kernel, vce, nnmatch),
    ]

    gamma = None
    if mat.shape[1] > n_core:
        gamma = _joint_gamma(sides, [o["fit_p"] for o in ops], n_core)

    # Jumps of every column (right minus left), conventional and corrected.
    jump_cl = ops[1]["a_cl"] @ ops[1]["cols"] - ops[0]["a_cl"] @ ops[0]["cols"]
    jump_bc = ops[1]["a_bc"] @ ops[1]["cols"] - ops[0]["a_bc"] @ ops[0]["cols"]

    if fuzzy is None:
        s = np.r_[1.0, -gamma[:, 0]] if gamma is not None else np.array([1.0])
        tau_cl = float(jump_cl @ s)
        tau_bc = float(jump_bc @ s)
    else:
        adj = jump_cl[:n_core].copy()
        if gamma is not None:
            adj = adj - jump_cl[n_core:] @ gamma
        tau_y, tau_t = float(adj[0]), float(adj[1])
        s = np.r_[1.0 / tau_t, -tau_y / tau_t**2]
        if gamma is not None:
            s = np.r_[s, -(gamma[:, 0] / tau_t - tau_y * gamma[:, 1] / tau_t**2)]
        tau_cl = tau_y / tau_t
        # Linearised correction: the ratio of corrected jumps is not used.
        tau_bc = tau_cl + float((jump_bc - jump_cl) @ s)

    variances = []
    for side, o in zip(sides, ops):
        variances.append(
            (
                variance_of(side, o["union"], o["a_cl"], o["rho_cl"], s, p + 1, vce),
                variance_of(side, o["union"], o["a_bc"], o["rho_bc"], s, q + 1, vce),
            )
        )
    (v_cl_l, v_rb_l), (v_cl_r, v_rb_r) = variances
    if components is not None:
        components.update(
            V_cl_left=v_cl_l, V_cl_right=v_cl_r, V_rb_left=v_rb_l, V_rb_right=v_rb_r
        )
    return (
        tau_cl,
        tau_bc,
        math.sqrt(v_cl_l + v_cl_r),
        math.sqrt(v_rb_l + v_rb_r),
    )
