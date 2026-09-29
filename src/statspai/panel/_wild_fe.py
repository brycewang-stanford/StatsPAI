"""Wild cluster restricted (WCR) bootstrap after absorbing fixed effects.

``boottest`` (Roodman, MacKinnon, Nielsen and Webb 2019) semantics for
``sp.hdfe_ols(..., wild=True)``: the null is imposed, every bootstrap outcome
is built from the restricted fit plus the restricted residuals flipped by a
cluster-level weight, and the bootstrap regression **re-absorbs the fixed
effects**. The last step is what the previous implementation skipped: it ran
the bootstrap on already-demeaned data, which is only right when every
absorbed effect is nested in the cluster. With a year effect and clusters of
counties the flipped residuals are no longer orthogonal to the year dummies,
and the bootstrap t distribution came out too narrow (WoP Table 5 col. 1:
p = 0.045 against ``boottest``'s 0.218).

Algebra. Let ``M`` be the within (FE-sweeping) projection, ``Xt = M X`` the
demeaned regressors, ``Q = Xt'Xt`` and ``q`` row ``k`` of ``Q^{-1}``. Under
``H0: beta_k = b0`` the restricted residual is ``e(b0) = a - b0 c`` with
``a = M_Z M y`` and ``c = M_Z Xt_k`` (``Z`` the other regressors). A draw
``y* = fit(b0) + e(b0) * w[cluster]`` gives

    beta*_k - b0 = sum_h w_h q.s_h,         s_h = Xt_h' e_h
    score_g      = sum_h w_h (q.A[g, h] - q.D_g Q^{-1} s_h)
                   A[g, h] = Xt_g' (M (e 1_h))_g,  D_g = Xt_g' Xt_g

so each draw costs two ``G x G`` products; ``M`` is applied once per cluster.
Everything is linear in ``b0``, which makes test inversion for the confidence
set cheap. The small-sample factor multiplies ``t`` and ``t*`` alike and
cancels, so the p-value is the symmetric ``P(|t*| >= |t|)`` of ``boottest``.
With Rademacher weights and ``2^G <= n_boot`` every sign vector is enumerated
(``boottest``'s rule), which makes the result exact. Draws with
``|t*| = |t|`` count as ties, not exceedances, as in ``boottest``.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def _draw_weights(
    G: int, n_boot: int, weight_type: str, seed: Optional[int]
) -> Tuple[np.ndarray, bool]:
    wt = weight_type.lower()
    if wt == "rademacher" and 2**G <= n_boot:
        return np.array(list(product((-1.0, 1.0), repeat=G))), True
    rng = np.random.default_rng(seed)
    if wt == "rademacher":
        return rng.choice([-1.0, 1.0], size=(n_boot, G)), False
    if wt == "webb":
        vals = np.array(
            [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)]
        )
        return rng.choice(vals, size=(n_boot, G)), False
    if wt == "mammen":
        s5 = np.sqrt(5.0)
        lo, hi = -(s5 - 1) / 2, (s5 + 1) / 2
        p_lo = (s5 + 1) / (2 * s5)
        return np.where(rng.random((n_boot, G)) < p_lo, lo, hi), False
    raise ValueError(
        f"wild_weight_type={weight_type!r}; use 'rademacher', 'webb' or 'mammen'."
    )


def wild_cluster_fe(
    demean: Callable[[np.ndarray], np.ndarray],
    y_tilde: np.ndarray,
    X_tilde: np.ndarray,
    cluster: np.ndarray,
    k: int,
    *,
    n_boot: int = 999,
    weight_type: str = "webb",
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """WCR bootstrap p-value and test-inversion confidence set for ``beta_k``.

    ``demean`` applies the within projection ``M`` to an ``(n,)`` or
    ``(n, p)`` array on the estimation sample; ``y_tilde`` / ``X_tilde`` are
    already demeaned.
    """
    n, p = X_tilde.shape
    codes, cl_inv = np.unique(cluster, return_inverse=True)
    G = len(codes)
    Q = X_tilde.T @ X_tilde
    Qinv = np.linalg.inv(Q)
    q = Qinv[k]
    b_hat = Qinv @ (X_tilde.T @ y_tilde)

    # restricted residuals e(b0) = a - b0 c
    others = [j for j in range(p) if j != k]
    if others:
        Z = X_tilde[:, others]
        ZtZi = np.linalg.inv(Z.T @ Z)
        a = y_tilde - Z @ (ZtZi @ (Z.T @ y_tilde))
        xk = X_tilde[:, k]
        c = xk - Z @ (ZtZi @ (Z.T @ xk))
    else:
        a, c = y_tilde.copy(), X_tilde[:, k].copy()

    # cluster-level building blocks for e = a and e = c
    def blocks(e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = np.zeros((p, G))  # s_h = Xt_h' e_h
        np.add.at(s.T, cl_inv, X_tilde * e[:, None])
        qs = q @ s  # (G,)
        # A[g, h] projected on q: q . Xt_g' (M(e 1_h))_g
        B = np.empty((G, G))
        xq = X_tilde @ q  # (n,)
        step = 64
        for h0 in range(0, G, step):
            hs = np.arange(h0, min(G, h0 + step))
            E = np.zeros((n, len(hs)))
            for j, h in enumerate(hs):
                m = cl_inv == h
                E[m, j] = e[m]
            ME = demean(E)
            contrib = np.zeros((G, len(hs)))
            np.add.at(contrib, cl_inv, xq[:, None] * ME)
            B[:, hs] = contrib
        # C[g, h] = q . D_g Q^{-1} s_h
        Qs = Qinv @ s  # (p, G)
        C = np.zeros((G, G))
        for g in range(G):
            m = cl_inv == g
            C[g] = (q @ (X_tilde[m].T @ X_tilde[m])) @ Qs
        return qs, B - C

    qs_a, E_a = blocks(a)
    qs_c, E_c = blocks(c)
    W, enumerated = _draw_weights(G, n_boot, weight_type, seed)

    # observed CR0-type score variance for beta_k (factor cancels)
    u_hat = y_tilde - X_tilde @ b_hat
    S_hat = np.zeros((p, G))
    np.add.at(S_hat.T, cl_inv, X_tilde * u_hat[:, None])
    se_hat = float(np.sqrt(np.sum((q @ S_hat) ** 2)))

    def pvalue(b0: float) -> float:
        num = W @ (qs_a - b0 * qs_c)
        sc = W @ (E_a - b0 * E_c).T
        se = np.sqrt(np.sum(sc**2, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            tstar = np.where(se > 0, num / se, 0.0)
        t0 = (b_hat[k] - b0) / se_hat
        # boottest counts strictly larger |t*|; draws that reproduce |t|
        # (e.g. the all-ones and all-minus-ones Rademacher vectors) are ties.
        return float(np.mean(np.abs(tstar) > abs(t0) * (1 + 1e-10)))

    p0 = pvalue(0.0)

    def bound(direction: float) -> float:
        # step out from b_hat until the test rejects, then bisect
        step = 2.0 * se_hat if se_hat > 0 else 1.0
        inside = float(b_hat[k])
        outside = inside + direction * step
        for _ in range(60):
            if pvalue(outside) <= alpha:
                break
            inside, outside = outside, outside + direction * step
            step *= 2
        else:
            return float(direction * np.inf)
        for _ in range(100):
            mid = 0.5 * (inside + outside)
            if pvalue(mid) > alpha:
                inside = mid
            else:
                outside = mid
            if abs(outside - inside) < 1e-9 * max(1.0, abs(inside)):
                break
        return 0.5 * (inside + outside)

    ci = (bound(-1.0), bound(1.0))
    return {
        "p_boot": p0,
        "ci_boot": ci,
        "n_boot": int(W.shape[0]),
        "enumerated": enumerated,
        "n_clusters": G,
        "weight_type": weight_type,
    }
