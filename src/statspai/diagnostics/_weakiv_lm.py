"""Weak-instrument-robust AR / K / J / CLR tests in the LM form of Stata weakiv.

Clean-room rewrite (2026-10-20) from the papers [@anderson1949estimation;
@kleibergen2002pivotal; @kleibergen2005testing; @moreira2003conditional;
@kleibergen2006generalized] and the documentation of Stata ``weakiv``
(its help file). Stata ``weakiv`` 2.4.07 was used only as a black box: its
printed and stored results on committed fixtures fixed the conventions
below. The specification, with the black-box evidence, is
``docs/dev/cleanroom/weakiv_lm.md``.

This module is the engine behind ``sp.weakrobust`` when there are several
endogenous regressors or the tests use a robust / cluster-robust variance.
Every input is already residualised on the included exogenous regressors
(constant included).

Model and statistics
--------------------
Write the reduced form of ``Y = [y, X]`` on the excluded instruments ``Z``
as ``Y = Z B + U``. Under ``H0: beta = b0`` the moment ``r = B a`` with
``a = (1, -b0')'`` is zero; ``r = (Z'Z)^{-1} Z'e0`` with ``e0 = y - X b0``.
Let ``V`` be the estimated variance of ``vec(B)``, ``psi`` the block for
``r``, ``C`` the covariance of ``vec(Pi)`` (``Pi = B[:, 1:]``) with ``r``.
Kleibergen's first-stage estimator orthogonal to ``r`` is
``vec(D) = vec(Pi) - C psi^{-1} r`` with variance
``S_D = Var(vec Pi) - C psi^{-1} C'``. Then

* ``AR = r' psi^{-1} r``, chi2(k);
* ``K = r' psi^{-1} D (D' psi^{-1} D)^{-1} D' psi^{-1} r``, chi2(p);
* ``J = AR - K``, chi2(k - p);
* ``rk``: the Kleibergen-Paap statistic for ``rank = p - 1`` of ``L D``,
  where ``L`` is the lower-triangular Cholesky factor of ``psi^{-1}``
  (``L L' = psi^{-1}``) and ``Var(vec(L D)) = (I_p x L) S_D (I_p x L)'``;
  chi2(k - p + 1) under under-identification;
* ``CLR = (AR - rk + sqrt((AR + rk)^2 - 4 J rk)) / 2``.

Variance conventions (all fixed by the black-box runs):

* ``'iid'``: ``V = Sigma x (Z'Z)^{-1}``, ``Sigma = U'U / N`` from the
  unrestricted reduced-form residuals (minimum-distance form). AR, K and
  J are then multiplied by ``e0'M_Z e0 / e0'e0`` (the LM form, e.g.
  ``AR = N e0'P_Z e0 / e0'e0``); ``rk`` and ``CLR`` stay in the
  minimum-distance form.
* ``'robust'`` / ``'cluster'``: the sandwich
  ``(I x (Z'Z)^{-1}) M (I x (Z'Z)^{-1})`` with ``M`` the (cluster-summed)
  outer product of the uncentred scores ``[e0_i, x_i'] (x) z_i``, i.e. the
  restricted residual ``e0`` and the regressors themselves. No small-sample
  factor.

The CLR p-value is the exact conditional one. Given ``rk``, ``K`` and ``J``
are independent chi2(p) and chi2(k - p) under the null and CLR is
increasing in ``K``; ``CLR > c`` iff ``K + J c / (rk + c) > c``. Hence
``P(CLR > c | rk) = 1 - P(K + s J <= c)`` with ``s = c / (rk + c)``, a
one-dimensional integral (:func:`clr_pvalue`). For ``p = 1`` this is
Moreira's conditional p-value. (Stata weakiv's documented p-values differ:
for one regressor it evaluates the same integral by a fixed-node rule, for
several it simulates; see the specification.)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate, linalg, optimize, special

from ..exceptions import (
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
    NumericalInstability,
)

_VCOV_TYPES = ("iid", "robust", "cluster")


# ---------------------------------------------------------------------------
#  Small numerical helpers
# ---------------------------------------------------------------------------


def _as_matrix(a: np.ndarray, name: str) -> np.ndarray:
    m = np.asarray(a, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    if m.ndim != 2:
        raise MethodIncompatibility(
            f"{name} must be a vector or a matrix; got {m.ndim} dimensions.",
            recovery_hint=f"Pass {name} as an (n,) or (n, m) array.",
        )
    return m


def _chi2_sf(x: float, df: float) -> float:
    """Upper tail of chi2(df) at ``x`` (``scipy.special.chdtrc``)."""
    return float(special.chdtrc(df, x))


def _cluster_sums(A: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Sum the rows of ``A`` within clusters.

    ``codes`` are integer cluster labels ``0 .. G-1``; row ``g`` of the
    result is the sum of the rows of ``A`` with code ``g``. A 1-d ``A``
    gives a length-``G`` vector.
    """
    codes = np.asarray(codes).astype(np.int64).ravel()
    arr = np.asarray(A, dtype=float)
    if arr.shape[0] != codes.shape[0]:
        raise MethodIncompatibility(
            f"cluster codes have {codes.shape[0]} entries for {arr.shape[0]} rows.",
            recovery_hint="Pass one cluster code per observation.",
        )
    n_g = int(codes.max()) + 1 if codes.size else 0
    out = np.zeros((n_g,) + arr.shape[1:])
    np.add.at(out, codes, arr)
    return out


def _sym_inv(A: np.ndarray, what: str) -> np.ndarray:
    """Inverse of a symmetric positive definite matrix, loudly."""
    try:
        c = linalg.cho_factor(A, lower=True, check_finite=True)
    except (linalg.LinAlgError, ValueError) as exc:
        raise NumericalInstability(
            f"the {what} is not positive definite.",
            recovery_hint="Check for collinear or constant instruments, too "
            "few clusters, or a hypothesised value at which the moment "
            "variance degenerates.",
            diagnostics={"matrix": what},
        ) from exc
    inv = linalg.cho_solve(c, np.eye(A.shape[0]))
    _ret: np.ndarray = 0.5 * (inv + inv.T)
    return _ret


def _rk_kp(D: np.ndarray, G: np.ndarray, W: np.ndarray, q: int) -> float:
    """Kleibergen-Paap rank statistic of ``Theta = G D`` (column scaling ``I``).

    Tests ``rank(Theta) = q`` for the ``(k, p)`` matrix ``Theta``, ``k >= p``,
    whose column-stacked vectorisation of ``D`` has variance ``W`` (so
    ``Var(vec Theta) = (I_p x G) W (I_p x G)'``). With the singular value
    decomposition ``Theta = U S V'`` and ``U_q``, ``V_q`` the singular
    vectors beyond the first ``q``, the statistic is
    ``lam' Omega^{-1} lam`` with ``lam = vec(U_q' Theta V_q)`` and
    ``Omega = (V_q' x U_q') Var(vec Theta) (V_q x U_q)``
    [@kleibergen2006generalized]. Degrees of freedom ``(k - q)(p - q)``.
    The rotation in the paper's definition of the null-space bases is
    orthogonal and leaves the quadratic form unchanged.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim == 1:
        D = D[:, None]
    G = np.asarray(G, dtype=float)
    k, p = D.shape
    if p == 1 and q == 0:
        # one column: the statistic is the Wald form D' W^{-1} D for any G
        d = D[:, 0]
        Wm = np.asarray(W, dtype=float)
        Wm = 0.5 * (Wm + Wm.T)
        try:
            return float(d @ np.linalg.solve(Wm, d))
        except np.linalg.LinAlgError:
            return float(d @ np.linalg.pinv(Wm) @ d)
    theta = G @ D
    T = np.kron(np.eye(p), G)
    w_theta = T @ np.asarray(W, dtype=float) @ T.T
    u, _, vt = np.linalg.svd(theta, full_matrices=True)
    u_q = u[:, q:]
    v_q = vt.T[:, q:]
    lam = (u_q.T @ theta @ v_q).reshape(-1, order="F")
    proj = np.kron(v_q.T, u_q.T)
    omega = proj @ w_theta @ proj.T
    omega = 0.5 * (omega + omega.T)
    try:
        return float(lam @ np.linalg.solve(omega, lam))
    except np.linalg.LinAlgError:
        return float(lam @ np.linalg.pinv(omega) @ lam)


# ---------------------------------------------------------------------------
#  Conditional p-value of the CLR statistic
# ---------------------------------------------------------------------------


def clr_pvalue(clr: float, rk: float, k: int, p: int) -> float:
    """Exact ``P(CLR > clr | rk)`` with ``k`` instruments, ``p`` regressors.

    Under the null and given ``rk``, ``K ~ chi2(p)`` and ``J ~ chi2(k - p)``
    are independent, and ``CLR > c`` exactly when ``K + s J > c`` with
    ``s = c / (rk + c)`` (solve ``CLR(K, J) = c`` for ``K``; CLR is
    increasing in ``K``). The p-value is one minus

    ``P(K + s J <= c) = int_0^c f_p(x) F_{k-p}((c - x) / s) dx``,

    with ``f_p`` the chi2(p) density and ``F_{k-p}`` the chi2(k - p)
    distribution function. Written as
    ``x^{p/2 - 1} (c - x)^{(k-p)/2} g(x)`` the integrand has a smooth
    ``g``, so the algebraic-weight quadrature converges to near machine
    precision (checked against 40-digit arithmetic in the tests). With
    ``p = 1`` this is the conditional p-value of [@moreira2003conditional].

    Limits: ``clr <= 0`` gives 1; ``k = p`` (no ``J``) gives
    ``P(chi2(p) > clr)``; ``rk = 0`` gives ``P(chi2(k) > clr)``;
    ``rk = inf`` gives ``P(chi2(p) > clr)``; ``clr = inf`` gives 0;
    a NaN input gives NaN.
    """
    c = float(clr)
    r = float(rk)
    k = int(k)
    p = int(p)
    if k < p or p < 1:
        raise MethodIncompatibility(
            f"the CLR p-value needs k >= p >= 1; got k={k}, p={p}.",
            recovery_hint="Pass the number of excluded instruments as k and "
            "the number of tested coefficients as p.",
        )
    if np.isnan(c) or np.isnan(r):
        return float("nan")
    if c <= 0.0:
        return 1.0
    if np.isinf(c):
        return 0.0
    if k == p or np.isinf(r):
        return _chi2_sf(c, p)
    r = max(r, 0.0)
    a = 0.5 * (k - p)
    s = c / (r + c)
    log_norm = -(0.5 * p) * np.log(2.0) - special.gammaln(0.5 * p)
    log_scale = a * np.log(0.5 / s)
    inv_gamma = 1.0 / special.gamma(a + 1.0)

    def smooth_part(x: float) -> float:
        # F_{k-p}((c - x) / s) / (c - x)^a times the chi2(p) kernel e^{-x/2}
        z = (c - x) / (2.0 * s)
        if z < 1e-10:
            ratio = inv_gamma * (1.0 - a * z / (a + 1.0))
        else:
            ratio = float(special.gammainc(a, z)) / z**a
        _ret: float = float(np.exp(log_norm + log_scale - 0.5 * x)) * ratio
        return _ret

    mass, _ = integrate.quad(
        smooth_part,
        0.0,
        c,
        weight="alg",
        wvar=(0.5 * p - 1.0, a),
        epsabs=1e-15,
        epsrel=1e-13,
        limit=200,
    )
    return float(min(max(1.0 - mass, 0.0), 1.0))


# ---------------------------------------------------------------------------
#  The joint tests at beta = b0
# ---------------------------------------------------------------------------


class WeakIVLM:
    """AR, K, J, rk and CLR at ``beta = b0`` in the LM form of Stata weakiv.

    Parameters
    ----------
    y : (n,) outcome, X : (n, p) endogenous regressors, Z : (n, k)
        excluded instruments -- all residualised on the included exogenous
        regressors.
    vcov : {'iid', 'robust', 'cluster'}
    cluster_codes : (n,) integer cluster codes ``0 .. G-1``, required for
        'cluster'.

    Notes
    -----
    Everything that does not depend on ``b0`` (the reduced-form
    coefficients and the variance of the stacked scores) is computed once;
    each ``at(b0)`` only forms ``(p + 1) k``-dimensional quadratic forms, so
    the tests can be inverted on a fine grid. The conventions are in the
    module docstring.
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
        if vcov not in _VCOV_TYPES:
            raise MethodIncompatibility(
                f"unknown vcov {vcov!r}; expected one of {list(_VCOV_TYPES)}.",
                recovery_hint="Use vcov='iid', 'robust' or 'cluster'.",
            )
        yv = np.asarray(y, dtype=float).reshape(-1)
        Xm = _as_matrix(X, "X")
        Zm = _as_matrix(Z, "Z")
        n = yv.shape[0]
        if Xm.shape[0] != n or Zm.shape[0] != n:
            raise MethodIncompatibility(
                f"y, X and Z have {n}, {Xm.shape[0]} and {Zm.shape[0]} rows.",
                recovery_hint="Pass arrays with one row per observation.",
            )
        k = Zm.shape[1]
        p = Xm.shape[1]
        if k < p:
            raise IdentificationFailure(
                f"Under-identified: {k} excluded instrument(s) for {p} "
                "endogenous regressors.",
                recovery_hint="Add instruments or drop endogenous regressors.",
                diagnostics={"n_instruments": k, "n_endog": p},
            )
        self.vcov = vcov
        self.n, self.k, self.p = n, k, p
        Ymat = np.column_stack([yv, Xm])
        zz_inv = _sym_inv(Zm.T @ Zm, "instrument cross-product Z'Z")
        self._zz_inv = zz_inv
        self._B = zz_inv @ (Zm.T @ Ymat)  # (k, p+1) reduced-form coefficients
        self._YY = Ymat.T @ Ymat
        m = p + 1
        if vcov == "iid":
            U = Ymat - Zm @ self._B
            self._Sigma = U.T @ U / n
            self._V4 = None
        else:
            # score of observation i: Y_i (x) z_i, column-stacked (c * k + b)
            scores = (Ymat[:, :, None] * Zm[:, None, :]).reshape(n, m * k)
            if vcov == "cluster":
                if cluster_codes is None:
                    raise MethodIncompatibility(
                        "vcov='cluster' needs cluster_codes.",
                        recovery_hint="Pass integer cluster codes, e.g. "
                        "pandas.factorize(column)[0].",
                    )
                codes = np.asarray(cluster_codes).reshape(-1)
                if codes.shape[0] != n:
                    raise MethodIncompatibility(
                        f"cluster_codes has {codes.shape[0]} entries for {n} rows.",
                        recovery_hint="Pass one cluster code per observation.",
                    )
                scores = _cluster_sums(scores, codes)
                n_g = scores.shape[0]
                if n_g < k:
                    raise DataInsufficient(
                        f"{n_g} clusters for {k} instruments: the cluster-robust "
                        "moment variance is singular.",
                        recovery_hint="Use more clusters than instruments, or "
                        "test_vcov='robust'.",
                        diagnostics={"n_clusters": n_g, "n_instruments": k},
                    )
            meat = scores.T @ scores
            bread = np.kron(np.eye(m), zz_inv)
            V = bread @ meat @ bread
            self._V4 = (0.5 * (V + V.T)).reshape(m, k, m, k)
            self._Sigma = None

    # -- internal -----------------------------------------------------------

    def _null_vector(self, b0: Any) -> np.ndarray:
        b = np.atleast_1d(np.asarray(b0, dtype=float)).reshape(-1)
        if b.size != self.p:
            raise MethodIncompatibility(
                f"b0 has {b.size} value(s) for {self.p} endogenous regressor(s).",
                recovery_hint="Pass one hypothesised value per endogenous regressor.",
            )
        if not np.all(np.isfinite(b)):
            raise MethodIncompatibility(
                f"b0 must be finite; got {b.tolist()}.",
                recovery_hint="Pass finite hypothesised coefficients.",
            )
        return b

    def _blocks(self, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``psi`` (k, k), ``C`` (p, k, k) and ``Var(vec Pi)`` (p, k, p, k)."""
        if self._V4 is None:
            S = self._Sigma
            Q = self._zz_inv
            s_ee = float(a @ S @ a)
            s_ve = S[1:, :] @ a
            psi = s_ee * Q
            C = s_ve[:, None, None] * Q[None, :, :]
            Vpp = S[1:, 1:][:, None, :, None] * Q[None, :, None, :]
            return psi, C, Vpp
        V4 = self._V4
        psi = np.einsum("c,cidj,d->ij", a, V4, a)
        C = np.einsum("cidj,d->cij", V4[1:], a)
        Vpp = V4[1:, :, 1:, :]
        return psi, C, Vpp

    # -- public -------------------------------------------------------------

    def at(
        self, b0: Any, clr_alpha: Optional[float] = None, clr_p: bool = True
    ) -> Dict[str, Any]:
        """All statistics and p-values at ``beta = b0``.

        Parameters
        ----------
        b0 : scalar or (p,) hypothesised coefficients.
        clr_alpha : float, optional
            When given, the CLR p-value is computed exactly only where the
            decision at level ``clr_alpha`` is not already settled by the
            bounds ``P(chi2(p) > CLR) <= p_CLR <= P(chi2(k) > CLR)`` (they
            hold because ``K <= CLR <= AR``); otherwise the deciding bound
            is returned. Used when inverting the test on a grid.
        clr_p : bool, default True
            ``False`` skips the CLR p-value (returned as NaN).

        Returns
        -------
        dict with ``ar, ar_df, ar_p, k, k_df, k_p, rk, rk_df, rk_p, clr,
        j, j_df, j_p, clr_p``. With ``k = p`` the ``j`` entries are
        ``None`` and the CLR test is the AR test (``clr_p = ar_p``).
        """
        b = self._null_vector(b0)
        k, p = self.k, self.p
        a = np.concatenate([[1.0], -b])
        r = self._B @ a
        Pi = self._B[:, 1:]
        psi, C, Vpp = self._blocks(a)
        psi = 0.5 * (psi + psi.T)
        psi_inv = _sym_inv(psi, "moment variance at b0")
        # D_j = Pi_j - C_j psi^{-1} r ; S_D = Var(vec Pi) - C psi^{-1} C'
        h = psi_inv @ r
        D = Pi - np.einsum("jab,b->aj", C, h)
        Cmat = C.reshape(p * k, k)
        S_D = Vpp.reshape(p * k, p * k) - Cmat @ psi_inv @ Cmat.T
        S_D = 0.5 * (S_D + S_D.T)

        ar = float(r @ h)
        g = D.T @ h
        DtPD = D.T @ psi_inv @ D
        try:
            kstat = float(g @ np.linalg.solve(DtPD, g))
        except np.linalg.LinAlgError as exc:
            raise NumericalInstability(
                "D' psi^-1 D is singular at b0.",
                recovery_hint="The first stage is rank deficient at this "
                "hypothesis; check for collinear endogenous regressors.",
            ) from exc
        if p == k:
            kstat = ar
        jstat = ar - kstat

        # G = lower Cholesky factor of psi^{-1}; irrelevant when p = 1
        L = np.linalg.cholesky(psi_inv) if p > 1 else np.eye(k)
        rk = _rk_kp(D, L, S_D, p - 1)
        clr = 0.5 * (ar - rk + np.sqrt(max((ar + rk) ** 2 - 4.0 * jstat * rk, 0.0)))

        if self._V4 is None:
            # LM form: rescale by e0'M_Z e0 / e0'e0
            ee = float(a @ self._YY @ a)
            s_md = float(a @ self._Sigma @ a) * self.n
            factor = s_md / ee
            ar_r, k_r, j_r = ar * factor, kstat * factor, jstat * factor
        else:
            ar_r, k_r, j_r = ar, kstat, jstat

        ar_p = _chi2_sf(ar_r, k)
        rk_df = k - p + 1
        out: Dict[str, Any] = {
            "ar": float(ar_r),
            "ar_df": int(k),
            "ar_p": ar_p,
            "k": float(k_r),
            "k_df": int(p),
            "k_p": _chi2_sf(k_r, p),
            "rk": float(rk),
            "rk_df": int(rk_df),
            "rk_p": _chi2_sf(rk, rk_df),
            "clr": float(clr),
        }
        if k > p:
            out["j"] = float(j_r)
            out["j_df"] = int(k - p)
            out["j_p"] = _chi2_sf(j_r, k - p)
        else:
            out["j"] = out["j_df"] = out["j_p"] = None

        if not clr_p:
            out["clr_p"] = float("nan")
        elif k == p:
            out["clr_p"] = ar_p
        elif clr_alpha is not None:
            lower = _chi2_sf(clr, p)
            upper = _chi2_sf(clr, k)
            if lower > clr_alpha:
                out["clr_p"] = lower
            elif upper <= clr_alpha:
                out["clr_p"] = upper
            else:
                out["clr_p"] = clr_pvalue(clr, rk, k, p)
        else:
            out["clr_p"] = clr_pvalue(clr, rk, k, p)
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
    """One-shot ``WeakIVLM(y, X, Z, ...).at(b0)``; see :class:`WeakIVLM`."""
    return WeakIVLM(y, X, Z, vcov=vcov, cluster_codes=cluster_codes).at(b0)


# ---------------------------------------------------------------------------
#  Subset and projection AR for one coefficient (i.i.d.)
# ---------------------------------------------------------------------------


def _liml_excess(Y0: np.ndarray, Z: np.ndarray) -> float:
    """``lambda - 1``: smallest root of ``|Y0'P_Z Y0 - mu Y0'M_Z Y0| = 0``."""
    Ym = _as_matrix(Y0, "Y0")
    Zm = _as_matrix(Z, "Z")
    coef = np.linalg.lstsq(Zm, Ym, rcond=None)[0]
    fit = Zm @ coef
    res = Ym - fit
    A = fit.T @ Ym
    Bm = res.T @ res
    A = 0.5 * (A + A.T)
    Bm = 0.5 * (Bm + Bm.T)
    return float(linalg.eigh(A, Bm, eigvals_only=True, subset_by_index=[0, 0])[0])


def liml_lambda(Y0: np.ndarray, Z: np.ndarray) -> float:
    """Smallest root ``lambda`` of ``|Y0'Y0 - lambda Y0'M_Z Y0| = 0``.

    The LIML eigenvalue of the columns ``Y0`` on instruments ``Z``;
    ``lambda - 1`` is the smallest root of ``|Y0'P_Z Y0 - mu Y0'M_Z Y0| = 0``,
    and for ``e = Y0 (1, -g')'`` it equals ``min_g e'P_Z e / e'M_Z e``.
    """
    return 1.0 + _liml_excess(Y0, Z)


class SubsetAR:
    """AR tests of one endogenous coefficient with the others unrestricted.

    For ``H0: beta_1 = b1`` with the coefficients ``beta_2`` on ``X2`` left
    free, put ``Y0 = [y - x1 b1, X2]`` and let ``lambda`` be its LIML root
    (:func:`liml_lambda`), so that
    ``min_{b2} e'P_Z e / e'M_Z e = lambda - 1`` for
    ``e = y - x1 b1 - X2 b2``.

    * ``subset(b1)``: the subset AR statistic of weakiv's ``subset()``
      option, ``(N - L)(lambda - 1)`` against ``chi2(k - p2)``, with ``L``
      the number of instruments, excluded plus included (black-box
      convention);
    * ``projection(b1)``: the minimum over ``b2`` of the joint LM-form AR
      statistic ``N e'P_Z e / e'e``, which is ``N (lambda - 1) / lambda``,
      against ``chi2(k)``.
    """

    def __init__(
        self, y: np.ndarray, x1: np.ndarray, X2: np.ndarray, Z: np.ndarray, L: int
    ):
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.x1 = np.asarray(x1, dtype=float).reshape(-1)
        self.X2 = _as_matrix(X2, "X2")
        self.Z = _as_matrix(Z, "Z")
        self.n = self.y.shape[0]
        self.k = self.Z.shape[1]
        self.p2 = self.X2.shape[1]
        self.L = int(L)
        if self.k < self.p2 + 1:
            raise IdentificationFailure(
                f"Under-identified: {self.k} excluded instrument(s) for "
                f"{self.p2 + 1} endogenous regressors.",
                recovery_hint="Add instruments or drop endogenous regressors.",
                diagnostics={"n_instruments": self.k, "n_endog": self.p2 + 1},
            )
        # second moments of A = [y, x1, X2]: fitted (A'P_Z A) and residual
        A = np.column_stack([self.y, self.x1, self.X2])
        coef = np.linalg.lstsq(self.Z, A, rcond=None)[0]
        fit = self.Z @ coef
        pp = fit.T @ A
        res = A - fit
        mm = res.T @ res
        self._PP = 0.5 * (pp + pp.T)
        self._MM = 0.5 * (mm + mm.T)

    def _excess(self, b1: float) -> float:
        m = 2 + self.p2
        T = np.zeros((m, 1 + self.p2))
        T[0, 0] = 1.0
        T[1, 0] = -float(b1)
        T[2:, 1:] = np.eye(self.p2)
        A = T.T @ self._PP @ T
        Bm = T.T @ self._MM @ T
        return float(
            linalg.eigh(
                0.5 * (A + A.T),
                0.5 * (Bm + Bm.T),
                eigvals_only=True,
                subset_by_index=[0, 0],
            )[0]
        )

    def lam(self, b1: float) -> float:
        """LIML root ``lambda`` of ``[y - x1 b1, X2]``."""
        return 1.0 + self._excess(b1)

    def subset(self, b1: float) -> Dict[str, float]:
        """Subset AR statistic, df and p-value at ``beta_1 = b1``."""
        stat = (self.n - self.L) * self._excess(b1)
        df = self.k - self.p2
        return {"stat": float(stat), "df": int(df), "p": _chi2_sf(stat, df)}

    def projection(self, b1: float) -> Dict[str, float]:
        """Projection AR statistic, df and p-value at ``beta_1 = b1``."""
        mu = self._excess(b1)
        stat = self.n * mu / (1.0 + mu)
        df = self.k
        return {"stat": float(stat), "df": int(df), "p": _chi2_sf(stat, df)}


# ---------------------------------------------------------------------------
#  Inverting a test over a scalar coefficient
# ---------------------------------------------------------------------------

_DENSE_HALF_WIDTH = 50.0  # dense grid: center +/- 50 scale
_TAIL_REACH = 1e6  # geometric tails out to center +/- 1e6 scale


def _scan_grid(center: float, scale: float, n_dense: int, n_tail: int) -> np.ndarray:
    """Search grid: ``n_dense`` equally spaced points on
    ``center +/- 50 scale`` and, on each side, ``n_tail - 1`` further points
    spaced geometrically out to ``center +/- 1e6 scale``."""
    half = _DENSE_HALF_WIDTH * scale
    dense = np.linspace(center - half, center + half, int(n_dense))
    reach = np.geomspace(_DENSE_HALF_WIDTH, _TAIL_REACH, int(n_tail))[1:] * scale
    return np.concatenate([center - reach[::-1], dense, center + reach])


def _crossing(
    pvalue: Callable[[float], float], alpha: float, lo: float, hi: float
) -> float:
    """The point in ``(lo, hi)`` where ``pvalue - alpha`` changes sign."""
    f_lo = pvalue(lo) - alpha
    f_hi = pvalue(hi) - alpha
    if f_lo == 0.0:
        return float(lo)
    if f_hi == 0.0:
        return float(hi)
    if np.sign(f_lo) == np.sign(f_hi) or not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        # no bracket after re-evaluation: fall back to the midpoint
        return float(0.5 * (lo + hi))
    xtol = 1e-15 * max(1.0, abs(lo), abs(hi))
    return float(
        optimize.brentq(
            lambda b: pvalue(b) - alpha, lo, hi, xtol=xtol, rtol=1e-15, maxiter=500
        )
    )


def _intervals_from_grid(
    grid: np.ndarray, acc: np.ndarray, pvalue: Callable[[float], float], alpha: float
) -> List[Tuple[float, float]]:
    """Maximal runs of accepted grid points, with endpoints refined to the
    crossing ``pvalue = alpha`` between neighbouring grid points; a run that
    reaches the end of the grid is unbounded on that side."""
    acc = np.asarray(acc, dtype=bool)
    m = acc.shape[0]
    out: List[Tuple[float, float]] = []
    i = 0
    while i < m:
        if not acc[i]:
            i += 1
            continue
        j = i
        while j + 1 < m and acc[j + 1]:
            j += 1
        lo = -np.inf if i == 0 else _crossing(pvalue, alpha, grid[i - 1], grid[i])
        hi = np.inf if j == m - 1 else _crossing(pvalue, alpha, grid[j], grid[j + 1])
        out.append((float(lo), float(hi)))
        i = j + 1
    return out


def invert_pvalues(
    pvalues: Callable[[float], Dict[str, float]],
    keys: Sequence[str],
    alpha: float,
    center: float,
    scale: float,
    n_dense: int = 2001,
    n_tail: int = 200,
) -> Dict[str, List[Tuple[float, float]]]:
    """Confidence sets ``{b : pvalues(b)[key] > alpha}`` for several tests.

    ``pvalues(b)`` returns a dict of p-values; one evaluation per grid point
    serves every key. Each set is a sorted list of ``(lo, hi)`` intervals
    (``-inf`` / ``inf`` when unbounded; empty list when nothing is
    accepted). The grid is :func:`_scan_grid`; endpoints are solved to
    machine precision by Brent's method between the grid points where the
    decision changes.
    """
    grid = _scan_grid(center, scale, n_dense, n_tail)
    vals = [pvalues(float(b)) for b in grid]
    out: Dict[str, List[Tuple[float, float]]] = {}
    for key in keys:
        pv = np.array([float(v[key]) for v in vals])
        acc = np.where(np.isfinite(pv), pv > alpha, False)
        out[key] = _intervals_from_grid(
            grid, acc, lambda b, kk=key: float(pvalues(float(b))[kk]), alpha
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
    """Single-test version of :func:`invert_pvalues`: ``{b : pvalue(b) > alpha}``."""
    return invert_pvalues(
        lambda b: {"p": pvalue(b)}, ["p"], alpha, center, scale, n_dense, n_tail
    )["p"]
