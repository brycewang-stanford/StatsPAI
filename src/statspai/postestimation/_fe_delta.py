"""Delta-method covariance of margins after a Poisson fit with absorbed
fixed effects, the effects counted as parameters.

After ``sp.fepois`` (or ``sp.feglm(family='poisson')``) an average marginal
effect or predictive margin depends on the estimated effects. Stata's
``margins`` after ``poisson y x i.g`` (and after ``ppmlhdfe ..., d``)
differentiates through them: the gradient has a block for the effects and
the covariance is that of ``(b, alpha)``. This module computes that number
without forming the ``(k + L)``-square covariance.

With ``Z = [X, D]`` (regressors and effect indicators on the estimation
sample), ``Omega = diag(w mu)``, the Poisson information ``A = Z' Omega Z``
and a gradient ``g = (g_b, g_a)``, the influence of the margin on
observation ``i`` is ``w_i (y_i - mu_i) h_i`` with ``h = Z A^-1 g``.
Partitioning (Frisch-Waugh-Lovell with weights ``Omega``):

    t   = (D' Omega D)^-1 g_a               (any solution; D t is unique)
    v_b = S^-1 (g_b - X' Omega D t),         S = Xt' Omega Xt
    h   = Xt v_b + D t,                      Xt = X - D (D'Omega D)^-1 D'Omega X

so the model-based variance is ``h' Omega h``, the robust one
``sum_i (w_i e_i h_i)^2`` and the clustered one the same over cluster sums.
The fit's own small-sample factor ``c`` is read off its coefficient
covariance: ``V_fit = c * V_raw`` for the ``b`` block computed the same way
(``h = Xt S^-1 e_j``). A covariance that is not a scalar multiple of the
raw one (two-way clustering, Conley) is refused, as is a sample that does
not reproduce the fit (``mu`` and the score equations are checked).

The effects' gradient comes from the margin's derivative with respect to
each evaluation row's linear index: ``g_a = D_eval' u`` with ``u_j`` that
derivative (times the averaging weight), the only way the effects enter.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import cg

from ..exceptions import MethodIncompatibility

__all__ = ["FEDelta", "build_fe_delta"]


def _refuse(fn: str, why: str) -> MethodIncompatibility:
    return MethodIncompatibility(
        f"{fn}: the fit absorbed fixed effects and this quantity depends on "
        f"their estimates; their uncertainty cannot be added here ({why}).",
        recovery_hint="Pass vcov=False for the estimates without standard "
        "errors, or fit the effects as C() terms (e.g. "
        "sp.poisson('y ~ x + C(g)')), whose margins match Stata's.",
        diagnostics={"reason": why},
    )


class FEDelta:
    """Covariance of margins of a fixed-effects Poisson fit (see module)."""

    def __init__(
        self,
        X: np.ndarray,
        codes: List[np.ndarray],
        levels: List[np.ndarray],
        names: List[str],
        omega: np.ndarray,
        score: np.ndarray,
        clusters: Optional[np.ndarray],
        kind: str,
        V_fit: np.ndarray,
        fn: str,
    ) -> None:
        self.names = names
        self.levels = levels
        self.kind = kind
        self.clusters = clusters
        self.omega = omega
        self.score = score  # w_i (y_i - mu_i)
        offsets = np.cumsum([0] + [len(lv) for lv in levels])
        self.offsets = offsets
        n = X.shape[0]
        cols = np.concatenate([c + o for c, o in zip(codes, offsets[:-1])])
        rows = np.tile(np.arange(n), len(codes))
        self.D = sparse.csr_matrix(
            (np.ones(rows.size), (rows, cols)), shape=(n, int(offsets[-1]))
        )
        self.M = (self.D.T @ sparse.diags(omega) @ self.D).tocsr()
        self._diag = np.asarray(self.M.diagonal(), dtype=float)
        self.one_way = len(codes) == 1
        DtOX = self.D.T @ (omega[:, None] * X)
        self.Xt = X - self.D @ self._solve(DtOX, fn)
        S = self.Xt.T @ (omega[:, None] * self.Xt)
        try:
            self.S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - fit would fail
            raise _refuse(fn, "the regressors are collinear with the effects") from exc
        self.XtOD = DtOX.T  # X' Omega D  (k x L)
        k = X.shape[1]
        raw_b = self._raw_cov(self.Xt @ self.S_inv)  # h for unit gradients
        scale = np.sqrt(np.outer(np.diag(raw_b), np.diag(raw_b)))
        ok = np.diag(raw_b) > 0
        if not ok.all() or V_fit.shape != (k, k):
            raise _refuse(fn, "the coefficient covariance cannot be reproduced")
        c = float(np.median(np.diag(V_fit) / np.diag(raw_b)))
        dev = np.max(np.abs(V_fit - c * raw_b) / scale)
        if not np.isfinite(c) or c <= 0 or dev > 1e-6:
            raise _refuse(
                fn,
                "the fit's coefficient covariance is not a scalar multiple of "
                "the model-based, robust or one-way clustered sandwich "
                f"(relative deviation {dev:.2g})",
            )
        self.c = c

    # -- linear algebra ----------------------------------------------------
    def _solve(self, B: Any, fn: str) -> np.ndarray:
        """Solve ``(D' Omega D) T = B`` (columns of ``B``), any solution."""
        B = np.asarray(B.toarray() if sparse.issparse(B) else B, dtype=float)
        squeeze = B.ndim == 1
        B2 = B[:, None] if squeeze else B
        if self.one_way:
            out = B2 / self._diag[:, None]
        else:
            pre = sparse.diags(1.0 / self._diag)
            out = np.empty_like(B2)
            for j in range(B2.shape[1]):
                b = B2[:, j]
                nb = float(np.linalg.norm(b))
                if nb == 0.0:
                    out[:, j] = 0.0
                    continue
                x, info = cg(
                    self.M, b, rtol=1e-14, atol=0.0, maxiter=50 * b.size, M=pre
                )
                if info != 0 or np.linalg.norm(self.M @ x - b) > 1e-9 * nb:
                    raise _refuse(fn, "the fixed-effect system did not converge")
                out[:, j] = x
        res: np.ndarray = out[:, 0] if squeeze else out
        return res

    def _raw_cov(self, H: np.ndarray) -> np.ndarray:
        """Raw covariance ``Cov(h_r, h_s)`` of the columns of ``H`` (n x m)."""
        C: np.ndarray
        if self.kind == "iid":
            C = H.T @ (self.omega[:, None] * H)
            return C
        Phi = self.score[:, None] * H
        if self.kind == "hetero":
            C = Phi.T @ Phi
            return C
        assert self.clusters is not None  # kind == "cluster" carries clusters
        _, inv = np.unique(self.clusters, return_inverse=True)
        Sums = np.zeros((int(inv.max()) + 1, H.shape[1]))
        np.add.at(Sums, inv, Phi)
        return Sums.T @ Sums

    # -- public ------------------------------------------------------------
    def eval_codes(self, frame: pd.DataFrame, fn: str) -> sparse.csr_matrix:
        """Indicator matrix of the effects' levels for the rows of ``frame``."""
        n = len(frame)
        rows, cols = [], []
        for name, lv, off in zip(self.names, self.levels, self.offsets[:-1]):
            if name not in frame.columns:
                raise _refuse(fn, f"the evaluation data lacks {name!r}")
            pos = pd.Index(pd.Series(lv, dtype=object)).get_indexer(
                pd.Series(frame[name].to_numpy(), dtype=object)
            )
            if (pos < 0).any():
                alt = pd.Index(pd.Series(lv).astype(frame[name].dtype))
                pos = alt.get_indexer(frame[name])
            if (pos < 0).any():
                raise _refuse(fn, f"a level of {name!r} the fit never saw")
            rows.append(np.arange(n))
            cols.append(pos + off)
        r = np.concatenate(rows)
        return sparse.csr_matrix(
            (np.ones(r.size), (r, np.concatenate(cols))),
            shape=(n, int(self.offsets[-1])),
        )

    def cov(
        self, G_b: np.ndarray, U: np.ndarray, D_eval: sparse.csr_matrix
    ) -> np.ndarray:
        """Covariance of margins with ``b``-gradients ``G_b`` (m x k) and
        per-evaluation-row index derivatives ``U`` (m x n_eval)."""
        G_b = np.atleast_2d(np.asarray(G_b, dtype=float))
        U = np.atleast_2d(np.asarray(U, dtype=float))
        G_a = np.asarray((D_eval.T @ U.T))  # L x m
        T = self._solve(G_a, "margins")  # L x m
        V_b = self.S_inv @ (G_b.T - self.XtOD @ T)  # k x m
        H = self.Xt @ V_b + self.D @ T  # n x m
        return self.c * self._raw_cov(H)


def _poisson_fit(result: Any) -> bool:
    info = getattr(result, "model_info", None) or {}
    model_type = str(info.get("model_type", ""))
    family = str(info.get("family", "")).lower()
    return model_type.startswith("Poisson") or family == "poisson"


def build_fe_delta(
    result: Any, data: Optional[pd.DataFrame], design: Any, fn: str
) -> FEDelta:
    """The :class:`FEDelta` of a native fixed-effects Poisson fit, or raise
    the refusal naming why it cannot be built."""
    from ..fixest._fixef import fe_state
    from .margins import _offset_without_fe

    if not _poisson_fit(result):
        raise _refuse(
            fn,
            "only Poisson fits are supported (Stata's margins after logit or "
            "probit with i.g is the incidental-parameter estimator, not a "
            "reference for an absorbed fit)",
        )
    state = fe_state(result)
    if state is None or state.sample_index is None:
        raise _refuse(fn, "the fit was answered by pyfixest (e.g. ssc= given)")
    if data is None or not pd.Index(state.sample_index).isin(data.index).all():
        raise _refuse(
            fn, "the estimation data (with its original row labels) is needed"
        )
    info = getattr(result, "model_info", None) or {}
    dinfo = getattr(result, "data_info", None) or {}
    est = data.loc[state.sample_index]
    mu = np.asarray(dinfo.get("fitted_values"), dtype=float)
    resid = np.asarray(dinfo.get("residuals"), dtype=float)
    if mu.shape != (len(est),) or resid.shape != (len(est),):
        raise _refuse(fn, "the fitted values do not match the estimation sample")
    X = np.asarray(design.build(est), dtype=float)
    beta = result.params.to_numpy(dtype=float)
    eta = X @ beta + state.fe_sum + _offset_without_fe(result, est)
    if not np.allclose(np.exp(eta), mu, rtol=1e-6, atol=0.0):
        raise _refuse(fn, "the data does not reproduce the fitted values")
    wname = info.get("weights_name")
    w = (
        est[wname].to_numpy(dtype=float)
        if isinstance(wname, str) and wname in est.columns
        else np.ones(len(est))
    )
    if isinstance(wname, str) and wname not in est.columns:
        raise _refuse(fn, f"the weight column {wname!r} is not in the data")
    vt = str(info.get("vcov_type", "iid")).lower()
    clusters = None
    if vt in ("iid",):
        kind = "iid"
    elif vt in ("hetero", "hc1", "hc0", "robust"):
        kind = "hetero"
    elif vt == "crv" and str(info.get("vcov_type_detail", "")).upper() == "CRV1":
        cl = info.get("clustervar")
        cl = [cl] if isinstance(cl, str) else list(cl or [])
        if len(cl) != 1 or cl[0] not in est.columns:
            raise _refuse(fn, "only one-way clustering is supported")
        clusters = pd.factorize(est[cl[0]])[0]
        kind = "cluster"
    else:
        raise _refuse(fn, f"vcov type {vt!r} is not supported")
    score = w * resid
    # the score equations hold at the fit: a mismatched sample shows here
    foc = np.abs(X.T @ score) / (np.abs(X).T @ np.abs(score) + 1e-300)
    if foc.size and float(foc.max()) > 1e-5:
        raise _refuse(fn, "the data does not satisfy the fit's score equations")
    V_fit = np.atleast_2d(np.asarray(dinfo.get("var_cov"), dtype=float))
    return FEDelta(
        X,
        list(state.codes),
        [np.asarray(lv, dtype=object) for lv in state.levels],
        list(state.names),
        w * mu,
        score,
        clusters,
        kind,
        V_fit,
        fn,
    )


def factor(cov: np.ndarray) -> np.ndarray:
    """``L`` with ``L L' = cov`` (symmetric PSD), for the ``hypotheses`` attrs."""
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2.0)
    L: np.ndarray = vecs * np.sqrt(np.clip(vals, 0.0, None))
    return L


def eval_weights(w: Optional[np.ndarray], n: int) -> np.ndarray:
    """Normalised averaging weights over ``n`` evaluation rows."""
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=float)
    wn: np.ndarray = w / w.sum()
    return wn
