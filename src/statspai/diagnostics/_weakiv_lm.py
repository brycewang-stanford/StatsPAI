"""Weak-IV-robust LM tests in the form Stata ``weakiv`` reports them.

Shared engine behind ``sp.weakrobust`` when the model has several
endogenous regressors or the tests are heteroskedasticity- / cluster-robust.
All inputs are already partialled of the included exogenous regressors
(constant included), as ``weakiv`` does before forming the LM statistics.

Notation (the minimum-distance construction of
``weakiv.ado::compute_tests``): ``r = pi_y - Pi b0`` is the reduced-form
moment at the null, ``psi`` its variance, ``D`` the first-stage matrix made
orthogonal to ``r`` and ``shat`` the variance of ``vec(D)``. Then

* ``AR = r' psi^-1 r`` (chi2(k));
* ``K = r' psi^-1 D (D' psi^-1 D)^-1 D' psi^-1 r`` (chi2(p));
* ``J = AR - K`` (chi2(k - p));
* ``rk`` = Kleibergen-Paap rank statistic of ``D`` under ``shat``;
* ``CLR = (AR - rk + sqrt((AR + rk)^2 - 4 J rk)) / 2``.

Under i.i.d. errors ``AR``, ``K`` and ``J`` are rescaled from the
minimum-distance to the LM form (``s2 / s2lm``), and CLR keeps the
minimum-distance values, exactly as ``weakiv`` does. The scale is ``1/N``
throughout (``weakiv`` without ``small``).

The CLR p-value is the exact conditional one. Given ``rk``, ``K ~ chi2(p)``
and ``J ~ chi2(k - p)`` are independent under the null and CLR is
increasing in ``K``, with ``CLR = c`` at ``K = c - J c / (rk + c)``; so
``P(CLR > c | rk) = E_J[ P(chi2(p) > c - J c / (rk + c)) ]``, a
one-dimensional integral. With ``p = 1`` it equals Moreira's conditional
p-value (``weak_iv_ci._clr_conditional_pvalue``), which ``weakiv``
approximates by a 20-node Simpson rule (its ``new_try``).
For ``p >= 2`` ``weakiv`` simulates, drawing ``AR`` and ``K`` as
*independent* chi-squares (``compute_pvals``), although ``AR = K + J``;
StatsPAI does not copy that.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate, stats


def _rk_kp(D: np.ndarray, G: np.ndarray, W: np.ndarray, q: int) -> float:
    """``weakiv.ado::rk_kp`` with ``F = I``: rank statistic of ``G D``."""
    L, K = D.shape
    that = G @ D
    T = np.kron(np.eye(K), G)
    vhat = T @ W @ T.T
    vhat = 0.5 * (vhat + vhat.T)
    vec_that = that.flatten(order="F")
    if q == 0:
        return float(vec_that @ np.linalg.solve(vhat, vec_that))
    U, _, Vt = np.linalg.svd(that)
    V = Vt.T
    u12, u22 = U[:q, q:], U[q:, q:]
    v12, v22 = V[:q, q:], V[q:, q:]

    def _sqrt_psd(M: np.ndarray) -> np.ndarray:
        w, Q = np.linalg.eigh(0.5 * (M + M.T))
        return (Q * np.sqrt(np.clip(w, 0.0, None))) @ Q.T

    aq = np.vstack([u12, u22]) @ np.linalg.inv(u22) @ _sqrt_psd(u22 @ u22.T)
    bq = _sqrt_psd(v22 @ v22.T) @ np.linalg.inv(v22.T) @ np.vstack([v12, v22]).T
    A = np.kron(bq, aq.T)
    lab = A @ vec_that
    vlab = A @ vhat @ A.T
    vlab = 0.5 * (vlab + vlab.T)
    return float(lab @ np.linalg.solve(vlab, lab))


def clr_pvalue(clr: float, rk: float, k: int, p: int) -> float:
    """Exact ``P(CLR > clr | rk)`` with ``k`` instruments, ``p`` regressors."""
    if not np.isfinite(clr) or not np.isfinite(rk):
        return float("nan")
    if clr <= 0:
        return 1.0
    if k == p:  # just identified: CLR = AR = K ~ chi2(p)
        return float(stats.chi2.sf(clr, p))
    if p == 1:
        from ..iv.weak_iv_ci import _clr_conditional_pvalue

        return _clr_conditional_pvalue(float(clr), float(rk), int(k))
    df_j = k - p
    shrink = clr / (rk + clr)

    def _integrand(j: float) -> float:
        return float(
            stats.chi2.sf(max(clr - j * shrink, 0.0), p) * stats.chi2.pdf(j, df_j)
        )

    # K* = clr - j*shrink reaches 0 at j = rk + clr; beyond it the tail is 1.
    edge = rk + clr
    inner = integrate.quad(
        _integrand, 0.0, edge, epsabs=1e-13, epsrel=1e-11, limit=200
    )[0]
    outer = float(stats.chi2.sf(edge, df_j))
    return float(min(max(inner + outer, 0.0), 1.0))


def _cluster_sums(A: np.ndarray, codes: np.ndarray) -> np.ndarray:
    S = np.zeros((int(codes.max()) + 1, A.shape[1]))
    np.add.at(S, codes, A)
    return S


class WeakIVLM:
    """AR, K, J, rk and CLR at ``beta = b0`` (``weakiv`` LM form).

    Parameters
    ----------
    y : (n,) outcome, X : (n, p) endogenous regressors, Z : (n, k)
        excluded instruments -- all partialled of the included exogenous
        regressors.
    vcov : {'iid', 'robust', 'cluster'}
    cluster_codes : (n,) integer cluster codes, required for 'cluster'.

    Everything that does not depend on ``b0`` is computed once, so that
    ``at(b0)`` is cheap enough for confidence-set inversion on a fine grid.
    """

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        *,
        vcov: str = "iid",
        cluster_codes: Optional[np.ndarray] = None,
    ) -> None:
        y = np.asarray(y, dtype=float).reshape(-1)
        n = len(y)
        X = np.asarray(X, dtype=float).reshape(n, -1)
        Z = np.asarray(Z, dtype=float).reshape(n, -1)
        self.n, self.p = X.shape
        self.k = Z.shape[1]
        self.vcov = vcov
        k, p = self.k, self.p
        ZZinv = np.linalg.inv(Z.T @ Z)
        self.ZZinv = ZZinv
        self.Pi = ZZinv @ (Z.T @ X)
        self.pi_y = ZZinv @ (Z.T @ y)
        if vcov == "iid":
            # second moments of [y, X] (raw) and of their first-stage residuals
            R = np.column_stack([y, X])
            Rres = R - Z @ (ZZinv @ (Z.T @ R))
            self.M_raw = R.T @ R / n
            self.M_res = Rres.T @ Rres / n
        elif vcov in ("robust", "cluster"):
            R = np.column_stack([y, X])  # avar (depvar endo) (exexog)
            H = np.einsum("ij,ik->ijk", R, Z).reshape(n, (p + 1) * k)
            if vcov == "cluster":
                if cluster_codes is None:
                    raise ValueError("vcov='cluster' needs cluster_codes")
                H = _cluster_sums(H, np.asarray(cluster_codes))
            S = H.T @ H / n
            Ik = np.kron(np.eye(p), ZZinv)
            self.var_del = n * ZZinv @ S[:k, :k] @ ZZinv
            self.var_pi = n * Ik @ S[k:, k:] @ Ik
            self.var_pidel = n * Ik @ S[k:, :k] @ ZZinv
        else:  # pragma: no cover - validated by callers
            raise ValueError(f"unknown vcov {vcov!r}")

    def at(
        self, b0: Any, clr_alpha: Optional[float] = None, clr_p: bool = True
    ) -> Dict[str, Any]:
        """All statistics and p-values at ``beta = b0``.

        With ``clr_alpha`` the CLR p-value is only computed exactly when it
        matters for a level-``clr_alpha`` decision: ``K <= CLR <= AR``
        pointwise, so ``chi2(p)`` and ``chi2(k)`` tails bound it, and a
        bound on the right side of ``clr_alpha`` is returned instead
        (used when inverting the test on a grid). ``clr_p=False`` skips
        the CLR p-value altogether (``NaN``).
        """
        k, p = self.k, self.p
        b0 = np.asarray(b0, dtype=float).reshape(-1)
        Pi = self.Pi
        r = self.pi_y - Pi @ b0
        a = np.concatenate([[1.0], -b0])
        if self.vcov == "iid":
            s2 = float(a @ self.M_res @ a)
            s_ve = self.M_res[1:, :] @ a
            Svv = self.M_res[1:, 1:]
            psi = s2 * self.ZZinv
            D = -(Pi - np.outer(r, s_ve) / s2)
            shat = np.kron(Svv - np.outer(s_ve, s_ve) / s2, self.ZZinv)
            s2lm = float(a @ self.M_raw @ a)
        else:
            kron = np.kron(b0.reshape(-1, 1), np.eye(k))
            cross = kron.T @ self.var_pidel
            psi = self.var_del - cross - cross.T + kron.T @ self.var_pi @ kron
            psi = 0.5 * (psi + psi.T)
            bracket = self.var_pidel - self.var_pi @ kron
            vec_d = -Pi.flatten(order="F") + bracket @ np.linalg.solve(psi, r)
            D = vec_d.reshape((k, p), order="F")
            shat = self.var_pi - bracket @ np.linalg.solve(psi, bracket.T)
            s2 = s2lm = 1.0
        shat = 0.5 * (shat + shat.T)

        psi_r = np.linalg.solve(psi, r)
        psi_d = np.linalg.solve(psi, D)
        ar = float(r @ psi_r)
        kstat = float(psi_r @ D @ np.linalg.solve(D.T @ psi_d, D.T @ psi_r))
        jstat = ar - kstat
        psi_inv = np.linalg.inv(psi)
        G = np.linalg.cholesky(0.5 * (psi_inv + psi_inv.T))
        rk = _rk_kp(D, G, shat, p - 1)
        clr = 0.5 * (ar - rk + np.sqrt(max((ar + rk) ** 2 - 4.0 * jstat * rk, 0.0)))
        if not rk > 0:
            clr = rk = float("nan")
        if self.vcov == "iid":
            # LM rescaling of AR / K / J; CLR keeps the minimum-distance values
            ar *= s2 / s2lm
            kstat *= s2 / s2lm
            jstat = ar - kstat
        out: Dict[str, Any] = {
            "ar": ar,
            "ar_df": k,
            "ar_p": float(stats.chi2.sf(ar, k)),
            "k": kstat,
            "k_df": p,
            "k_p": float(stats.chi2.sf(kstat, p)),
            "rk": float(rk),
            "rk_df": k - p + 1,
            "rk_p": float(stats.chi2.sf(rk, k - p + 1)) if np.isfinite(rk) else np.nan,
            "clr": float(clr),
        }
        if k > p:
            out["j"] = jstat
            out["j_df"] = k - p
            out["j_p"] = float(stats.chi2.sf(jstat, k - p))
            lower = float(stats.chi2.sf(clr, p)) if np.isfinite(clr) else np.nan
            upper = float(stats.chi2.sf(clr, k)) if np.isfinite(clr) else np.nan
            if not clr_p:
                out["clr_p"] = float("nan")
            elif clr_alpha is not None and lower > clr_alpha:
                out["clr_p"] = lower
            elif clr_alpha is not None and upper <= clr_alpha:
                out["clr_p"] = upper
            else:
                out["clr_p"] = clr_pvalue(float(clr), float(rk), k, p)
        else:  # just identified: K = AR, CLR = AR
            out["j"] = out["j_df"] = out["j_p"] = None
            out["clr_p"] = out["ar_p"]
        return out


def weakiv_tests(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    b0: Any,
    *,
    vcov: str = "iid",
    cluster_codes: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """One-shot ``WeakIVLM(...).at(b0)``."""
    return WeakIVLM(y, X, Z, vcov=vcov, cluster_codes=cluster_codes).at(b0)


# ---------------------------------------------------------------------------
#  Subset and projection AR for one coefficient (i.i.d.)
# ---------------------------------------------------------------------------


def liml_lambda(Y0: np.ndarray, Z: np.ndarray) -> float:
    """Smallest root of ``|Y0'Y0 - lambda Y0'M_Z Y0| = 0`` (``m_liml_lambda``)."""
    proj = Z @ np.linalg.lstsq(Z, Y0, rcond=None)[0]
    A = Y0.T @ Y0
    B = Y0.T @ (Y0 - proj)
    B = 0.5 * (B + B.T)
    w, Q = np.linalg.eigh(B)
    Bm = (Q / np.sqrt(w)) @ Q.T
    return float(np.linalg.eigvalsh(Bm @ A @ Bm).min())


class SubsetAR:
    """AR tests of one endogenous coefficient with the others unrestricted.

    With ``Y0 = [y - x1 b1, X2]`` and ``lambda`` the LIML root above,
    ``min_b2 e'P_Z e / e'M_Z e = lambda - 1`` for ``e = y - x1 b1 - X2 b2``.

    * subset AR (``weakiv, subset()``): ``(N - L)(lambda - 1)`` against
      ``chi2(k - p2)``, ``L`` the number of
      instruments, excluded plus included (LM form);
    * projection AR: ``min_b2`` of the joint LM-form AR ``N e'P e / e'e``,
      ``N (lambda - 1) / lambda``, against ``chi2(k)`` -- the closed form of
      what ``weakiv, project()`` approximates on a grid.
    """

    def __init__(
        self, y: np.ndarray, x1: np.ndarray, X2: np.ndarray, Z: np.ndarray, L: int
    ):
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.x1 = np.asarray(x1, dtype=float).reshape(-1)
        self.X2 = np.asarray(X2, dtype=float).reshape(len(self.y), -1)
        self.Z = np.asarray(Z, dtype=float).reshape(len(self.y), -1)
        self.n = len(self.y)
        self.k = self.Z.shape[1]
        self.p2 = self.X2.shape[1]
        self.L = int(L)

    def lam(self, b1: float) -> float:
        Y0 = np.column_stack([self.y - self.x1 * b1, self.X2])
        return liml_lambda(Y0, self.Z)

    def subset(self, b1: float) -> Dict[str, float]:
        stat = (self.n - self.L) * (self.lam(b1) - 1.0)
        df = self.k - self.p2
        return {"stat": stat, "df": df, "p": float(stats.chi2.sf(stat, df))}

    def projection(self, b1: float) -> Dict[str, float]:
        lam = self.lam(b1)
        stat = self.n * (lam - 1.0) / lam
        return {"stat": stat, "df": self.k, "p": float(stats.chi2.sf(stat, self.k))}


# ---------------------------------------------------------------------------
#  Inverting a test over a scalar coefficient
# ---------------------------------------------------------------------------


def _scan_grid(center: float, scale: float, n_dense: int, n_tail: int) -> np.ndarray:
    scale = float(scale) if np.isfinite(scale) and scale > 0 else 1.0
    dense = center + scale * np.linspace(-50.0, 50.0, n_dense)
    tail = scale * np.geomspace(50.0, 1e6, n_tail)[1:]
    return np.concatenate([center - tail[::-1], dense, center + tail])


def _intervals_from_grid(
    grid: np.ndarray, acc: np.ndarray, pvalue: Callable[[float], float], alpha: float
) -> List[Tuple[float, float]]:
    if not acc.any():
        return []

    def _bisect(out_pt: float, in_pt: float) -> float:
        lo, hi = out_pt, in_pt
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid in (lo, hi):
                break
            if pvalue(mid) > alpha:
                hi = mid
            else:
                lo = mid
        return float(hi)

    intervals: List[Tuple[float, float]] = []
    idx = np.flatnonzero(acc)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    last = len(grid) - 1
    for run in runs:
        i0, i1 = int(run[0]), int(run[-1])
        lo = -np.inf if i0 == 0 else _bisect(float(grid[i0 - 1]), float(grid[i0]))
        hi = np.inf if i1 == last else _bisect(float(grid[i1 + 1]), float(grid[i1]))
        intervals.append((lo, hi))
    return intervals


def invert_pvalues(
    pvalues: Callable[[float], Dict[str, float]],
    keys: Sequence[str],
    alpha: float,
    center: float,
    scale: float,
    n_dense: int = 2001,
    n_tail: int = 200,
) -> Dict[str, List[Tuple[float, float]]]:
    """``{b : pvalues(b)[key] > alpha}`` for each key, as sorted intervals.

    A dense grid over ``center +/- 50 scale`` plus geometric tails out to
    ``+/- 1e6 scale`` decides the shape of each set (one evaluation of
    ``pvalues`` per grid point serves all keys); every boundary between a
    rejected and an accepted grid point is then bisected to machine
    precision. An accepted outermost tail point is reported as an infinite
    endpoint (the statistics converge as ``|b| -> inf``).
    """
    grid = _scan_grid(center, scale, n_dense, n_tail)
    vals = [pvalues(float(b)) for b in grid]
    out: Dict[str, List[Tuple[float, float]]] = {}
    for key in keys:
        acc = np.array([v[key] > alpha for v in vals])
        out[key] = _intervals_from_grid(
            grid, acc, lambda b, kk=key: pvalues(b)[kk], alpha
        )
    return out


def invert_pvalue(
    pvalue: Callable[[float], float],
    alpha: float,
    center: float,
    scale: float,
    n_dense: int = 2001,
    n_tail: int = 200,
) -> List[Tuple[float, float]]:
    """Single-test version of :func:`invert_pvalues`."""
    return invert_pvalues(
        lambda b: {"p": pvalue(b)}, ["p"], alpha, center, scale, n_dense, n_tail
    )["p"]
