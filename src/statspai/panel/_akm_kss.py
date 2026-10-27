"""Two-way fixed effects and leave-out variance components (``sp.akm``).

Model on the estimation sample: ``y_i = alpha_{w(i)} + psi_{f(i)} + e_i``.

Fixed effects
-------------
Sweeping out the worker effects leaves the firm system ``L psi = F' M_D y``
with ``L = F' M_D F`` (``M_D`` removes worker means). ``L`` is a weighted
graph Laplacian on firms (stayers add nothing to it), singular with one
null direction per connected component. Grounding one firm per component
makes the rest positive definite; the grounded inverse ``G`` (zero rows and
columns at the grounded firms) satisfies ``a' G b = a' L^+ b`` for vectors
summing to zero on each component, which is every vector below.

Leverages and the KSS correction
--------------------------------
A unit change in ``y_i`` moves the firm effects by ``G g_i`` with
``g_i = e_{f(i)} - fbar_{w(i)}`` (``fbar_w``: worker ``w``'s share of rows at
each firm) and the worker effects by ``u_i - H G g_i`` (``u_i``: ``1/T_w`` on
``w``'s rows; ``H`` stacks ``fbar_{w(k)}``). Hence

* ``P_ii = 1/T_w + g_i' G g_i``;
* for the person-year variance ``var(psi) = psi' F' M_1 F psi / n`` the
  quadratic-form weight is ``B_ii = g_i' G Q G g_i`` with
  ``Q = F' M_1 F / n``;
* ``var(alpha)`` and ``cov(alpha, psi)`` add ``1/T_w - 1/n`` and the bilinear
  term ``(fbar_w - n_f/n)' G g_i`` to ``g_i' G (F' P_D F - n_f n_f'/n) G g_i``.

Every term involves only the firms the worker was seen at, so with the
dense ``J x J`` matrices (``exact``) each row costs ``O(s^2)`` for ``s``
firms of its worker; stayers have ``g_i = 0`` and closed forms. Without
dense matrices (``jla``) each quadratic form ``a' K b`` is written as
``(Y a)'(Y b)`` and ``Y`` is replaced by a random Rademacher sketch
``R Y`` of ``k`` rows: ``k`` sparse solves per matrix (Kline, Saggio and
Solvsten 2020, Section 5, random projection).

The KSS estimator is ``theta_PI - sum_i B_ii sigma_i^2`` with the
leave-one-out ``sigma_i^2 = (y_i - ybar) e_i / (1 - P_ii)`` (the outcome
centred at its mean, as the authors' code and pytwoway do, so that the
estimate does not change when a constant is added to ``y``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, cast

import numpy as np

from ..exceptions import NumericalInstability


@dataclass
class TwoWayFit:
    """Fixed effects of the two-way model on one estimation sample."""

    wc: np.ndarray
    fc: np.ndarray
    n_w: int
    n_f: int
    y: np.ndarray
    alpha: np.ndarray  # per worker code
    psi: np.ndarray  # per firm code
    resid: np.ndarray
    T: np.ndarray  # rows per worker
    n_f_rows: np.ndarray  # rows per firm
    comp_firm: np.ndarray  # component of each firm
    grounded: np.ndarray  # bool per firm
    solve: Callable[
        [np.ndarray], np.ndarray
    ]  # grounded solver: rhs (J,) or (J, k) -> solution


def _firm_laplacian(
    wc: np.ndarray, fc: np.ndarray, n_w: int, n_f: int, T: np.ndarray
) -> Tuple[Any, Any, Any]:
    """``L = F' F - F' D (D' D)^{-1} D' F`` as a sparse matrix."""
    from scipy import sparse

    n = len(wc)
    F = sparse.csr_matrix((np.ones(n), (np.arange(n), fc)), shape=(n, n_f))
    D = sparse.csr_matrix((np.ones(n), (np.arange(n), wc)), shape=(n, n_w))
    DtF = (D.T @ F).tocsr()  # worker x firm counts
    L = (
        sparse.diags(np.asarray(F.sum(axis=0)).ravel())
        - DtF.T @ sparse.diags(1.0 / T) @ DtF
    )
    return L.tocsc(), F, DtF


def fit_twoway(
    wc: np.ndarray, fc: np.ndarray, y: np.ndarray, n_w: int, n_f: int
) -> TwoWayFit:
    """Exact two-way fixed effects by a grounded sparse solve.

    Firm effects average zero over the rows of each component; worker
    effects carry the level.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import splu

    T = np.bincount(wc, minlength=n_w).astype(float)
    n_rows_f = np.bincount(fc, minlength=n_f).astype(float)
    L, F, DtF = _firm_laplacian(wc, fc, n_w, n_f, T)
    # components of the firm graph (firms linked through movers)
    off = L.copy()
    off.setdiag(0)
    off.eliminate_zeros()
    _, comp_f = connected_components(abs(off), directed=False)
    grounded = np.zeros(n_f, dtype=bool)
    _, first = np.unique(comp_f, return_index=True)
    grounded[first] = True
    free = np.flatnonzero(~grounded)
    lu = splu(L[free][:, free].tocsc()) if free.size else None

    def solve(rhs: np.ndarray) -> np.ndarray:
        out = np.zeros(rhs.shape, dtype=float)
        if lu is not None:
            out[free] = lu.solve(np.ascontiguousarray(rhs[free]))
        return out

    ybar_w = np.bincount(wc, weights=y, minlength=n_w) / T
    rhs = np.bincount(fc, weights=y - ybar_w[wc], minlength=n_f)
    psi = solve(rhs)
    # each component: firm effects average zero over its rows
    shift = np.bincount(comp_f, weights=psi * n_rows_f) / np.bincount(
        comp_f, weights=n_rows_f
    )
    psi = psi - shift[comp_f]
    alpha = ybar_w - np.bincount(wc, weights=psi[fc], minlength=n_w) / T
    resid = y - alpha[wc] - psi[fc]
    return TwoWayFit(
        wc=wc,
        fc=fc,
        n_w=n_w,
        n_f=n_f,
        y=y,
        alpha=alpha,
        psi=psi,
        resid=resid,
        T=T,
        n_f_rows=n_rows_f,
        comp_firm=comp_f,
        grounded=grounded,
        solve=solve,
    )


def plugin_components(fit: TwoWayFit) -> Dict[str, float]:
    """Person-year (``1/n``) variances of the estimated effects."""
    a = fit.alpha[fit.wc]
    p = fit.psi[fit.fc]
    ac, pc = a - a.mean(), p - p.mean()
    n = len(a)
    return {
        "var_alpha": float(ac @ ac / n),
        "var_psi": float(pc @ pc / n),
        "cov": float(ac @ pc / n),
        "var_resid": float(fit.resid @ fit.resid / n),
    }


# ---------------------------------------------------------------------------
# Per-row g vectors, grouped by the number of firms of the worker
# ---------------------------------------------------------------------------


def _row_supports(
    fit: TwoWayFit,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """For movers' rows, grouped by ``s`` (firms of the worker): row index,
    the worker's firms ``(m, s)``, ``g_i`` and ``fbar_w`` on them."""
    wc, fc, n_f = fit.wc, fit.fc, fit.n_f
    pair = wc.astype(np.int64) * n_f + fc
    upair, cnt = np.unique(pair, return_counts=True)
    pw, pf = upair // n_f, upair % n_f
    s_w = np.bincount(pw, minlength=fit.n_w)
    groups: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for s in np.unique(s_w[wc]):
        if s < 2:
            continue
        rows = np.flatnonzero(s_w[wc] == s)
        # the worker's firms (sorted) and row counts there, s per worker
        sel = s_w[pw] == s
        wk = pw[sel].reshape(-1, s)[:, 0]
        firms = pf[sel].reshape(-1, s)
        counts = cnt[sel].reshape(-1, s).astype(float)
        pos = np.searchsorted(wk, wc[rows])
        F_idx = firms[pos]
        fbar = counts[pos] / fit.T[wc[rows]][:, None]
        g = (F_idx == fc[rows][:, None]).astype(float) - fbar
        groups[int(s)] = (rows, F_idx, g, fbar)
    return groups


def _quad(M_sub: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a_m' M_m b_m`` for stacked (m, s) vectors and (m, s, s) blocks."""
    return cast(np.ndarray, np.einsum("ma,mab,mb->m", a, M_sub, b))


@dataclass
class Leverage:
    P: np.ndarray
    B_psi: np.ndarray
    B_alpha: np.ndarray
    B_cov: np.ndarray
    method: str
    n_draws: Optional[int] = None


def leverages_exact(fit: TwoWayFit) -> Leverage:
    """``P_ii`` and the three ``B_ii`` from dense ``J x J`` matrices."""
    n, J = len(fit.y), fit.n_f
    nf = fit.n_f_rows
    G = fit.solve(np.eye(J))
    G = 0.5 * (G + G.T)
    Q = (np.diag(nf) - np.outer(nf, nf) / n) / n
    K_psi = G @ Q @ G
    T = fit.T
    _, _, DtF = _firm_laplacian(fit.wc, fit.fc, fit.n_w, J, T)
    FPF = (DtF.T @ (DtF.multiply(1.0 / T[:, None]))).toarray()
    K_D = G @ (FPF - np.outer(nf, nf) / n) @ G
    c = G @ (nf / n)
    mats = {"G": G, "psi": K_psi, "D": K_D}

    def dense(Fi: np.ndarray, a: np.ndarray, b: np.ndarray, which: str) -> np.ndarray:
        M = mats[which]
        return _quad(M[Fi[:, :, None], Fi[:, None, :]], a, b)

    return _assemble(fit, dense, c, "exact", None)


def leverages_jla(fit: TwoWayFit, n_draws: int, seed: Optional[int]) -> Leverage:
    """Random-projection (Johnson-Lindenstrauss) version of
    :func:`leverages_exact`: ``3 * n_draws`` sparse solves, no dense ``J x J``
    matrix."""
    rng = np.random.default_rng(seed)
    n, J, k = len(fit.y), fit.n_f, int(n_draws)
    wc, fc, T = fit.wc, fit.fc, fit.T

    def rademacher() -> np.ndarray:
        return cast(
            np.ndarray, rng.choice(np.array([-1.0, 1.0]), size=(n, k)) / np.sqrt(k)
        )

    def firm_sum(V: np.ndarray) -> np.ndarray:  # F' V
        out = np.zeros((J, V.shape[1]))
        np.add.at(out, fc, V)
        return out

    def worker_mean(V: np.ndarray) -> np.ndarray:  # P_D V
        s = np.zeros((fit.n_w, V.shape[1]))
        np.add.at(s, wc, V)
        return cast(np.ndarray, (s / T[:, None])[wc])

    R = rademacher()
    Z0 = fit.solve(firm_sum(R - worker_mean(R)))  # G F' M_D R'
    R = rademacher()
    Zp = fit.solve(firm_sum(R - R.mean(axis=0))) / np.sqrt(n)  # G F' M_1 R' / sqrt(n)
    R = rademacher()
    Zd = fit.solve(firm_sum(worker_mean(R - R.mean(axis=0))))  # G F' P_D M_1 R'
    Zs = {"G": Z0, "psi": Zp, "D": Zd}
    nf = fit.n_f_rows
    c = fit.solve(nf / n)

    def sketch(Fi: np.ndarray, a: np.ndarray, b: np.ndarray, which: str) -> np.ndarray:
        Z = Zs[which][Fi]  # (m, s, k)
        za = np.einsum("ma,mak->mk", a, Z)
        zb = np.einsum("mb,mbk->mk", b, Z)
        return cast(np.ndarray, (za * zb).sum(axis=1))

    return _assemble(fit, sketch, c, "jla", k)


def _assemble(
    fit: TwoWayFit,
    form: Callable[[np.ndarray, np.ndarray, np.ndarray, str], np.ndarray],
    c: np.ndarray,
    method: str,
    k: Optional[int],
) -> Leverage:
    n = len(fit.y)
    T_row = fit.T[fit.wc]
    P = 1.0 / T_row
    B_psi = np.zeros(n)
    B_alpha = (1.0 / T_row - 1.0 / n) / n
    B_cov = np.zeros(n)
    for s, (rows, Fi, g, fbar) in _row_supports(fit).items():
        q0 = form(Fi, g, g, "G")
        qp = form(Fi, g, g, "psi")
        qd = form(Fi, g, g, "D")
        lin = form(Fi, fbar, g, "G") - np.einsum("ma,ma->m", c[Fi], g)
        P[rows] += q0
        B_psi[rows] = qp
        B_alpha[rows] += (-2.0 * lin + qd) / n
        B_cov[rows] = (lin - qd) / n
    return Leverage(
        P=P, B_psi=B_psi, B_alpha=B_alpha, B_cov=B_cov, method=method, n_draws=k
    )


def corrected_components(
    fit: TwoWayFit, lev: Leverage, n_params_extra: int = 0
) -> Dict[str, Dict[str, float]]:
    """Plug-in, homoskedastic (Andrews et al. 2008) and KSS components."""
    n = len(fit.y)
    pi = plugin_components(fit)
    if np.any(lev.P >= 1.0 - 1e-10):
        bad = int(np.sum(lev.P >= 1.0 - 1e-10))
        raise NumericalInstability(
            f"akm: {bad} observation(s) have leverage P_ii >= 1, so their "
            "leave-one-out residual is undefined"
            + (" (random-projection estimate)" if lev.method == "jla" else "")
            + ".",
            recovery_hint="Use connected='leave_one_out' (the default with "
            "variance='kss')"
            + (
                ", or raise n_draws / use leverage='exact'"
                if lev.method == "jla"
                else ""
            )
            + ".",
            diagnostics={"n_bad": bad, "max_P": float(np.max(lev.P))},
        )
    sigma = (fit.y - fit.y.mean()) * fit.resid / (1.0 - lev.P)
    rank = fit.n_w + fit.n_f - len(np.unique(fit.comp_firm)) + n_params_extra
    dof = n - rank
    sigma_ho = float(fit.resid @ fit.resid / dof) if dof > 0 else float("nan")
    tr = {
        "var_alpha": float(lev.B_alpha.sum()),
        "var_psi": float(lev.B_psi.sum()),
        "cov": float(lev.B_cov.sum()),
    }
    kss_bias = {
        "var_alpha": float(lev.B_alpha @ sigma),
        "var_psi": float(lev.B_psi @ sigma),
        "cov": float(lev.B_cov @ sigma),
    }
    ho = {k: pi[k] - sigma_ho * tr[k] for k in tr}
    ho["var_resid"] = sigma_ho
    kss = {k: pi[k] - kss_bias[k] for k in kss_bias}
    kss["var_resid"] = float(sigma.mean())
    return {
        "plugin": pi,
        "homoskedastic": ho,
        "kss": kss,
        "_sigma_i": sigma,  # type: ignore[dict-item]
        "_trace": tr,
        "_bias": kss_bias,
    }


__all__ = [
    "Leverage",
    "TwoWayFit",
    "corrected_components",
    "fit_twoway",
    "leverages_exact",
    "leverages_jla",
    "plugin_components",
]
