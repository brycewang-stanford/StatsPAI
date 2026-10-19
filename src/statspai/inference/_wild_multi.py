"""Restricted wild bootstrap with multi-way clustering and joint hypotheses.

The engine of :func:`statspai.wild_cluster_boot` for the two cases the
one-coefficient, one-way path does not cover, following Stata ``boottest``
(Roodman, MacKinnon, Nielsen and Webb):

* ``cluster=[a, b, ...]``: the variance of the test statistic is the
  multi-way inclusion-exclusion sum ``sum_s (-1)^(|s|+1) c_s V_s`` over the
  non-empty subsets ``s`` of the cluster variables, each term with its own
  ``c_s = G_s / (G_s - 1) * (n - 1) / (n - k)``. The bootstrap weights are
  drawn per ``bootcluster`` (default: the intersection of all the cluster
  variables, boottest's default). A draw whose variance comes out
  non-positive has no statistic; it is left out of the p-value, as boottest
  does (so the denominator is not a power of two under enumeration).
* several restrictions (``variable=[x1, x2]``, ``h0=[r1, r2]``): the Wald
  statistic ``F = (R b - r)' (R V R')^-1 (R b - r) / q``, every restriction
  imposed on the bootstrap data, ``p = #{F* > F} / B``.

Every draw is linear in its weights (the algebra of ``_wcr_t_stats``): with
``A`` the tested rows of ``(X'X)^-1`` and ``e`` the restricted residuals,
``b*_b = A'Y*`` moves by ``v' w_b`` and the scores of an error cluster are
``K_s w_b``, so a draw costs ``O(q (G_boot + sum_s G_s))``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def _codes(columns: Sequence[np.ndarray]) -> np.ndarray:
    if len(columns) == 1:
        return pd.factorize(np.asarray(columns[0]))[0].astype(np.int64)
    frame = pd.DataFrame({i: np.asarray(c) for i, c in enumerate(columns)})
    return frame.groupby(list(frame.columns), sort=False).ngroup().to_numpy(np.int64)


def wild_wald(
    X: np.ndarray,
    Y: np.ndarray,
    var_names: List[str],
    clusters: List[np.ndarray],
    boot_cluster: np.ndarray,
    variables: List[str],
    h0: Sequence[float],
    *,
    n_boot: int,
    weight_type: str,
    rng: np.random.Generator,
    fe_codes: Any = None,
) -> Dict[str, Any]:
    """Restricted wild bootstrap test of ``beta[variables] = h0``.

    ``fe_codes``: effects absorbed into the design; each draw is
    residualised on them again (see ``_wcr_t_stats``).
    """
    from .wild_bootstrap import _nested, _wild_weight_matrix, absorb_columns

    missing = [v for v in variables if v not in var_names]
    if missing:
        raise MethodIncompatibility(
            f"wild_cluster_boot: {missing} are not coefficients of the fit.",
            recovery_hint=f"Use names from {var_names}.",
            diagnostics={"available": var_names},
        )
    idx = [var_names.index(v) for v in variables]
    q = len(idx)
    r = np.asarray(h0, dtype=float).reshape(-1)
    if r.size == 1 and q > 1:
        r = np.repeat(r, q)
    if r.size != q:
        raise MethodIncompatibility(
            f"wild_cluster_boot: {q} restrictions but {r.size} null values.",
            recovery_hint="Pass h0 as a list with one value per variable.",
        )
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)
    resid = Y - X @ beta

    # error-cluster terms of the inclusion-exclusion sum
    terms = []
    M = len(clusters)
    for size in range(1, M + 1):
        for combo in combinations(range(M), size):
            codes = _codes([clusters[i] for i in combo])
            G = int(codes.max()) + 1
            if G < 2:
                raise MethodIncompatibility(
                    "wild_cluster_boot: a cluster variable has one cluster.",
                    recovery_hint="Cluster on variables with several groups.",
                )
            c = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
            terms.append((codes, G, c * (1.0 if size % 2 else -1.0)))

    A = XtX_inv[idx]  # (q, k)
    XA = X @ A.T  # (n, q): b[idx] = XA' Y

    def var_of(scores_by_term: List[np.ndarray]) -> np.ndarray:
        V = np.zeros((q, q))
        for (codes, G, sc), S in zip(terms, scores_by_term):
            V += sc * (S.T @ S)
        return V

    # observed statistic
    obs_scores = []
    for codes, G, _ in terms:
        S = np.zeros((G, q))
        np.add.at(S, codes, XA * resid[:, None])
        obs_scores.append(S)
    V_obs = var_of(obs_scores)
    diff = beta[idx] - r
    stat_obs = _wald(diff, V_obs) / q

    # restricted fit: beta[idx] = r, the rest by least squares
    other = [j for j in range(k) if j not in idx]
    y_r = Y - X[:, idx] @ r
    b_o = np.linalg.lstsq(X[:, other], y_r, rcond=None)[0] if other else np.zeros(0)
    beta_r = np.zeros(k)
    beta_r[idx] = r
    beta_r[other] = b_o
    fitted_r = X @ beta_r
    e = Y - fitted_r

    boot_codes = pd.factorize(np.asarray(boot_cluster))[0].astype(np.int64)
    G_boot = int(boot_codes.max()) + 1
    W, enumerated = _wild_weight_matrix(G_boot, n_boot, weight_type, rng)

    xe = XA * e[:, None]  # (n, q)
    v = np.zeros((G_boot, q))
    np.add.at(v, boot_codes, xe)
    c0 = XA.T @ fitted_r  # (q,)
    P = X @ XtX_inv  # (n, k)
    Ab = np.zeros((G_boot, k))
    np.add.at(Ab, boot_codes, P * e[:, None])
    K_terms = []
    for codes, G, _ in terms:
        Mt = np.zeros((G, G_boot, q))
        if fe_codes and not all(_nested(f, codes) for f in fe_codes):
            # sum_{i in g} (M_D (XA_j 1_c))_i e_i, cluster by cluster
            for c in range(G):
                rows = codes == c
                cols = np.zeros((n, q))
                cols[rows] = XA[rows]
                A_c = absorb_columns(cols, list(fe_codes)) * e[:, None]
                np.add.at(Mt[c], boot_codes, A_c)
        else:
            np.add.at(Mt, (codes, boot_codes), xe)
        Q = np.zeros((G, q, k))
        np.add.at(Q, codes, XA[:, :, None] * X[:, None, :])
        # scores of cluster c for draw w: sum_g (M[c,g,:] - Q[c] @ Ab[g]) w_g
        K_terms.append(Mt - np.einsum("cqk,gk->cgq", Q, Ab))

    stats_b = np.empty(W.shape[0])
    valid = np.ones(W.shape[0], dtype=bool)
    for lo in range(0, W.shape[0], 1024):
        Wc = W[lo : lo + 1024]
        num = c0[None, :] + Wc @ v - r[None, :]  # (B, q)
        Vb = np.zeros((Wc.shape[0], q, q))
        for (codes, G, sc), K in zip(terms, K_terms):
            S = np.einsum("bg,cgq->bcq", Wc, K)
            Vb += sc * np.einsum("bcq,bcp->bqp", S, S)
        if q == 1:
            var = Vb[:, 0, 0]
            ok = var > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.where(ok, num[:, 0] / np.sqrt(np.where(ok, var, 1.0)), np.nan)
        else:
            # boottest computes the Wald statistic of every draw, also when
            # the multi-way variance is indefinite (a negative statistic
            # then never exceeds the observed one); only a singular one has
            # none
            s = np.full(Wc.shape[0], np.nan)
            ok = np.zeros(Wc.shape[0], dtype=bool)
            for j in range(Wc.shape[0]):
                try:
                    s[j] = _wald(num[j], Vb[j]) / q
                    ok[j] = np.isfinite(s[j])
                except np.linalg.LinAlgError:
                    pass
        stats_b[lo : lo + Wc.shape[0]] = s
        valid[lo : lo + Wc.shape[0]] = ok
    # the identity draw and its negation reproduce the observed statistic;
    # boottest's strict inequality leaves them out of the count
    tie = np.all(W == 1.0, axis=1) | np.all(W == -1.0, axis=1)
    if q == 1:
        t_obs = float(diff[0] / np.sqrt(V_obs[0, 0])) if V_obs[0, 0] > 0 else np.nan
        exceed = np.abs(stats_b) > abs(t_obs)
    else:
        t_obs = np.nan
        exceed = stats_b > stat_obs
    exceed &= valid & ~tie
    p = float(exceed.sum() / valid.sum()) if valid.any() else float("nan")
    out: Dict[str, Any] = {
        "beta_hat": (
            float(beta[idx[0]]) if q == 1 else dict(zip(variables, beta[idx]))
        ),
        "p_boot": p,
        "n_boot": int(W.shape[0]),
        "n_boot_valid": int(valid.sum()),
        "n_boot_requested": int(n_boot),
        "enumerated": bool(enumerated),
        "weight_type": weight_type,
        "n_obs": int(n),
        "n_clusters": [int(c.max()) + 1 for c in (_codes([x]) for x in clusters)],
        "n_boot_clusters": G_boot,
        "h0": float(r[0]) if q == 1 else dict(zip(variables, r)),
    }
    if q == 1:
        out.update(
            method="WCR wild cluster restricted bootstrap (multi-way clustering)",
            se_cluster=float(np.sqrt(V_obs[0, 0])) if V_obs[0, 0] > 0 else np.nan,
            t_stat=t_obs,
        )
    else:
        out.update(
            method="WCR wild cluster restricted bootstrap, joint Wald test",
            statistic=float(stat_obs),
            df=q,
        )
    return out


def _wald(diff: np.ndarray, V: np.ndarray) -> float:
    return float(diff @ np.linalg.solve(V, diff))
