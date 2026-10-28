"""Nearest-location estimator of spatial treatment effects (Pollmann 2026).

Under Assumption 4 of the paper only the nearest realised treatment
location matters to a unit. Theorem 2 identifies the average effect on
the treated at distance ``d`` and Online Appendix 6 gives the estimator::

    tau_nearest(d) =
        sum_s 1{s in S} sum_i N_i(s) / Pr(N_i(s) = 1 | s in S) w_i(s, d) Y_i
        / (same without Y_i)
      - sum_s 1{s not in S} pi_s / (1 - pi_s)
            sum_i N_i(0) / Pr(N_i(0) = 1 | s not in S) w_i(s, d) Y_i
        / (same without Y_i)

where ``N_i(s)`` marks ``s`` as the nearest realised location of ``i``
and ``N_i(0)`` marks a unit with no realised location within ``d0``.

The paper states that the variance "can be derived analogously" and does
not write it out. Here every summand is a Horvitz-Thompson term of an
event that requires a set of candidates to be realised (``R``) and
another set to be unrealised (``U``); under independent assignment every
joint inclusion probability has a closed form. The standard error is the
Aronow-Samii conservative estimator of the variance of the linearised
ratio: pairs of terms that can occur together contribute their unbiased
covariance estimate, pairs that cannot (their joint probability is zero,
so the product of their outcomes is never observed) are bounded by
``(a_k^2 + a_l^2) / 2``. This is the same device as the dropped terms of
Theorem 1(iv) for the ring estimator.

References
----------
[@pollmann2026causal]
[@aronow2017estimating]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

_CHUNK = 2048


def _closer_logprob(D: np.ndarray, log1mp: np.ndarray) -> np.ndarray:
    """``sum_{s': D[i, s'] < D[i, s]} log(1 - p_s')`` for every (i, s)."""
    n, S = D.shape
    out = np.empty_like(D)
    order = np.argsort(D, axis=1, kind="stable")
    for i in range(n):
        o = order[i]
        ds = D[i, o]
        cum = np.concatenate([[0.0], np.cumsum(log1mp[o])])
        first = np.searchsorted(ds, ds, side="left")  # ties are not closer
        out[i, o] = cum[first]
    return out


def _nearest_realised(D: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Distance from each unit to its nearest realised candidate."""
    return np.asarray(np.where(W[None, :] == 1, D, np.inf).min(axis=1), dtype=float)


def _nearest_variance(
    wk: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
    p: np.ndarray,
    Y: np.ndarray,
    logPt: np.ndarray,
    logPc: np.ndarray,
    Nt: np.ndarray,
    N0: np.ndarray,
    W: np.ndarray,
    mu_t: float,
    mu_c: float,
) -> Dict[str, float]:
    """Conservative design-based variance of the nearest-location estimator.

    Returns ``var`` and its two parts: ``V_cov`` (pairs of terms that can
    be observed together, including each term with itself) and
    ``V_bound`` (the Young-inequality bound for incompatible pairs).
    """
    logp, log1mp = np.log(p), np.log1p(-p)
    support = wk != 0
    Nbar = float((wk * p[None, :]).sum())
    m_i = support.sum(axis=1)

    # ---- population counts for the incompatible-pair bound -------------
    # nt[s]: treated terms whose realised candidate is s.
    nt = support.sum(axis=0).astype(float)
    # cntU[s]: terms (either kind) that need s unrealised.
    cntU = (E * m_i[:, None] + (~E & support)).sum(axis=0).astype(float)
    Mt = np.zeros((p.size, p.size))  # Mt[s', s]: treated terms at s' needing s off
    for i in np.flatnonzero(m_i):
        ss = np.flatnonzero(support[i])
        dsorted = np.sort(D[i, ss])
        cntU += m_i[i] - np.searchsorted(dsorted, D[i], side="right")
        Mt[ss] += D[i][None, :] < D[i, ss][:, None]

    # ---- observed terms ------------------------------------------------
    ti, ts = np.nonzero(Nt & support)
    ci, cs = np.nonzero((W[None, :] == 0) & N0[:, None] & support)
    kt, kc = ti.size, ci.size
    logP = np.concatenate([logp[ts] + logPt[ti, ts], log1mp[cs] + logPc[ci, cs]])
    a = (
        np.concatenate(
            [
                p[ts] * wk[ti, ts] * (Y[ti] - mu_t),
                -p[cs] * wk[ci, cs] * (Y[ci] - mu_c),
            ]
        )
        / Nbar
    )

    # R (must be realised) and U (must be unrealised) as sparse rows.
    rows_u: List[np.ndarray] = []
    cols_u: List[np.ndarray] = []
    for k in range(kt):
        i, s = ti[k], ts[k]
        closer = np.flatnonzero(D[i] < D[i, s])
        rows_u.append(np.full(closer.size, k))
        cols_u.append(closer)
    Ei = [np.flatnonzero(E[i]) for i in range(E.shape[0])]
    for k in range(kc):
        i, s = ci[k], cs[k]
        u = np.union1d(Ei[i], [s])
        rows_u.append(np.full(u.size, kt + k))
        cols_u.append(u)
    K = kt + kc
    S = p.size
    ru = np.concatenate(rows_u) if rows_u else np.zeros(0, int)
    cu = np.concatenate(cols_u) if cols_u else np.zeros(0, int)
    U = sparse.csr_matrix((np.ones(ru.size), (ru, cu)), shape=(K, S))
    R = sparse.csr_matrix((np.ones(kt), (np.arange(kt), ts)), shape=(K, S))

    # Pairs that can be observed together: unbiased covariance terms.
    RL = R.multiply(logp[None, :]).tocsr()
    UL = U.multiply(log1mp[None, :]).tocsr()
    Pk = np.exp(logP)
    v_cov = 0.0
    for start in range(0, K, _CHUNK):
        stop = min(start + _CHUNK, K)
        X = (RL[start:stop] @ R.T + UL[start:stop] @ U.T).tocoo()
        rr, cc = X.row + start, X.col
        v_cov += float(np.sum(a[rr] * a[cc] * -np.expm1(X.data) / (Pk[rr] * Pk[cc])))

    # Incompatible pairs: Young bound, each observed term once per partner.
    n0 = np.asarray(U @ nt).ravel()
    if kt:
        overlap = np.asarray(U[:kt].multiply(Mt[:, ts].T).sum(axis=1)).ravel()
        n0[:kt] += cntU[ts] - overlap
    v_bound = float(np.sum(a**2 * n0 / Pk))
    return {"var": v_cov + v_bound, "V_cov": v_cov, "V_bound": v_bound}


def nearest_rings(
    D: np.ndarray,
    W: np.ndarray,
    p: np.ndarray,
    Y: np.ndarray,
    ring_weights: List[np.ndarray],
    d0: float,
    draws: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    """Per-ring nearest-location estimates; one dict per ring."""
    E = D <= d0
    log1mp = np.log1p(-p)
    logPt = _closer_logprob(D, log1mp)
    # Pr(N_i(0) = 1 | s not realised): every other candidate within d0 off.
    logPc = (E * log1mp[None, :]).sum(axis=1)[:, None] - E * log1mp[None, :]
    dmin = _nearest_realised(D, W)
    Nt = (W[None, :] == 1) & (D <= dmin[:, None])
    N0 = dmin > d0
    dmin_draws = None
    if draws is not None:
        dmin_draws = np.array([_nearest_realised(D, Wb) for Wb in draws])
    odds = p / (1.0 - p)

    out: List[Dict[str, Any]] = []
    for wk in ring_weights:
        pi, ps = np.nonzero(wk)
        pw, pd = wk[pi, ps], D[pi, ps]
        with np.errstate(over="ignore"):
            gt = pw * np.exp(-logPt[pi, ps])  # treated weight when N_i(s) = 1
            gc = pw * odds[ps] * np.exp(-logPc[pi, ps])  # control weight
        yp = Y[pi]

        def _stat(Wv: np.ndarray, dm: np.ndarray) -> Tuple[float, float, int, int]:
            on_t = (Wv[ps] == 1) & (pd <= dm[pi])
            on_c = (Wv[ps] == 0) & (dm[pi] > d0)
            st, sc = gt[on_t].sum(), gc[on_c].sum()
            if st <= 0 or sc <= 0:
                return np.nan, np.nan, int(on_t.sum()), int(on_c.sum())
            mt = float(gt[on_t] @ yp[on_t] / st)
            mc = float(gc[on_c] @ yp[on_c] / sc)
            return mt, mc, int(on_t.sum()), int(on_c.sum())

        mu_t, mu_c, n_t, n_c = _stat(W, dmin)
        row: Dict[str, Any] = {"n_pairs_treated": n_t, "n_pairs_control": n_c}
        if n_t == 0 or n_c == 0:
            out.append(row)
            continue
        comp = _nearest_variance(wk, D, E, p, Y, logPt, logPc, Nt, N0, W, mu_t, mu_c)
        ri = None
        if dmin_draws is not None:
            ri = np.array(
                [
                    np.subtract(*_stat(Wb, dm)[:2])
                    for Wb, dm in zip(draws, dmin_draws)  # type: ignore[arg-type]
                ]
            )
        row.update(mu_t=mu_t, mu_c=mu_c, comp=comp, ri_draws=ri)
        out.append(row)
    return out
