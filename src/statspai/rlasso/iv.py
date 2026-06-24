"""Instrumental-variables estimation with rigorous-Lasso selection.

A faithful port of ``hdm::rlassoIV`` (and its building blocks
``rlassoIVselectZ`` / ``rlassoIVselectX`` / ``tsls``) on top of the
:func:`statspai.rlasso._core.rlasso` engine.

Four regimes, matching hdm exactly:

==================  ==================  ====================================
``select_Z``        ``select_X``        method
==================  ==================  ====================================
``True``  (default) ``False``           ``rlassoIVselectZ`` — Lasso-select
                                        instruments, robust 2SLS/GMM.  This
                                        is the canonical Belloni-Chen-
                                        Chernozhukov-Hansen (2012) eminent-
                                        domain estimator.
``False``           ``True``            ``rlassoIVselectX`` — partial out
                                        high-dim controls from y, d and
                                        every instrument, then 2SLS.
``True``            ``True``  (default) double selection on both Z and X.
``False``           ``False``           plain ``tsls`` (robust 2SLS).
==================  ==================  ====================================

References
----------
Belloni, A., Chen, D., Chernozhukov, V. and Hansen, C. (2012). "Sparse
    Models and Methods for Optimal Instruments With an Application to
    Eminent Domain." *Econometrica*, 80(6), 2369-2429.
    [@belloni2012sparse]

Chernozhukov, V., Hansen, C. and Spindler, M. (2016). "hdm:
    High-Dimensional Metrics." *The R Journal*, 8(2), 185-199.
    [@chernozhukov2016hdm]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ._core import rlasso

# MASS::ginv (used throughout hdm's IV routines) defaults to a singular-value
# cutoff of ``tol = sqrt(.Machine$double.eps)`` relative to the largest
# singular value.  numpy's ``pinv`` defaults to a far tighter ``rcond=1e-15``,
# which keeps spurious near-zero singular values and silently produces a
# different pseudo-inverse on the (collinear) high-dimensional control blocks.
# Matching the MASS tolerance is essential for hdm parity.
_GINV_TOL = float(np.sqrt(np.finfo(float).eps))


def _ginv(A: np.ndarray) -> np.ndarray:
    """Moore-Penrose pseudo-inverse with MASS::ginv's tolerance."""
    out: np.ndarray = np.linalg.pinv(np.asarray(A, dtype=float), rcond=_GINV_TOL)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  tsls  (hdm::tsls.default)
# ═══════════════════════════════════════════════════════════════════════


def _tsls(
    y: np.ndarray,
    d: np.ndarray,
    x: Optional[np.ndarray],
    z: np.ndarray,
    intercept: bool = True,
    homoscedastic: bool = True,
) -> Dict[str, Any]:
    """Two-stage least squares, faithful to ``hdm::tsls.default``.

    Regressors are ``[d, x]`` instrumented by ``[z, x]``.  Returns the
    coefficient vector, sandwich variance (homoskedastic or robust) and
    residuals.
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n = y.shape[0]

    if x is None:
        if intercept:
            x = np.ones((n, 1))
    else:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if intercept:
            x = np.column_stack([np.ones(n), x])

    a1 = d.shape[1]
    a2 = x.shape[1] if x is not None else 0
    k = a1 + a2

    X = np.column_stack([d, x]) if x is not None else d
    Z = np.column_stack([z, x]) if x is not None else z

    Mxz = X.T @ Z
    Mzz = np.linalg.inv(Z.T @ Z)
    M = np.linalg.inv(Mxz @ Mzz @ Mxz.T)
    b = M @ Mxz @ Mzz @ (Z.T @ y)

    e = y - X @ b
    if homoscedastic:
        VC1 = float((e**2).sum() / (n - k)) * M
    else:
        S = (Z * (e**2)).T @ Z / n  # (1/n) Σ e_i² z_i z_i'
        VC1 = n * M @ (Mxz @ Mzz @ S @ Mzz @ Mxz.T) @ M

    se = np.sqrt(np.diag(VC1))
    return {
        "coefficients": b.ravel(),
        "vcov": VC1,
        "se": se,
        "residuals": e.ravel(),
        "k": k,
        "a1": a1,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Result object
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class RLassoIVResult(ResultProtocolMixin):
    """Return of :func:`rlasso_iv`."""

    #: Verified paper.bib keys (CLAUDE.md §10).
    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "belloni2012sparse",
        "chernozhukov2016hdm",
    )

    coef: np.ndarray
    se: np.ndarray
    vcov: np.ndarray
    method: str
    n_obs: int
    treat_names: List[str]
    selection: Dict[str, Any] = field(default_factory=dict)
    residuals: Optional[np.ndarray] = None

    @property
    def tstat(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            out: np.ndarray = self.coef / self.se
        return out

    @property
    def pvalue(self) -> np.ndarray:
        out: np.ndarray = 2.0 * stats.norm.cdf(-np.abs(self.tstat))
        return out

    def conf_int(self, level: float = 0.95) -> np.ndarray:
        zc = float(stats.norm.ppf(0.5 + level / 2.0))
        out: np.ndarray = np.column_stack(
            [self.coef - zc * self.se, self.coef + zc * self.se]
        )
        return out

    def summary(self) -> str:
        ci = self.conf_int()
        lines = [
            f"Rigorous-Lasso IV  ({self.method})",
            "-" * 64,
            f"  Observations : {self.n_obs}",
        ]
        nsel = self.selection.get("n_selected_Z")
        if nsel is not None:
            lines.append(f"  Instruments selected : {nsel}")
        nselx = self.selection.get("n_selected_X")
        if nselx is not None:
            lines.append(f"  Controls selected    : {nselx}")
        lines.append("")
        lines.append("                  coef     std.err      z      P>|z|     95% CI")
        for i, nm in enumerate(self.treat_names):
            lines.append(
                f"  {nm:<12}{self.coef[i]:>10.4f}  {self.se[i]:>9.4f}"
                f"  {self.tstat[i]:>7.3f}  {self.pvalue[i]:>8.4f}"
                f"  [{ci[i, 0]:.3f}, {ci[i, 1]:.3f}]"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  rlassoIVselectZ  (instrument selection only)
# ═══════════════════════════════════════════════════════════════════════


def _select_z(
    x: Optional[np.ndarray],
    d: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    post: bool,
    intercept: bool,
    penalty: Optional[Dict[str, Any]],
    control: Optional[Dict[str, Any]],
    treat_names: List[str],
) -> RLassoIVResult:
    n = len(y)
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    ke = d.shape[1]
    if x is not None:
        Z = np.column_stack([z, x])
    else:
        Z = z

    Dhat_list = []
    select_mat = []
    flag_const = 0
    for i in range(ke):
        di = d[:, i]
        fit = rlasso(
            Z, di, post=post, intercept=intercept, penalty=penalty, control=control
        )
        if fit.index.sum() == 0:
            dihat = np.full(n, di.mean())
            flag_const += 1
            select_mat.append(np.zeros(Z.shape[1], dtype=bool))
        else:
            dihat = fit.predict(Z)
            select_mat.append(fit.index.copy())
        Dhat_list.append(dihat)

    Dhat = np.column_stack(Dhat_list)
    if x is not None:
        Dhat = np.column_stack([Dhat, x])
        d_aug = np.column_stack([d, x])
    else:
        d_aug = d

    alpha = _ginv(Dhat.T @ d_aug) @ (Dhat.T @ y)
    residuals = y - d_aug @ alpha
    Omega = Dhat.T @ (Dhat * (residuals**2)[:, None])
    Qinv = _ginv(d_aug.T @ Dhat)
    vcov_full = Qinv @ Omega @ Qinv.T

    coef = np.atleast_1d(alpha[:ke])
    se = np.sqrt(np.diag(vcov_full))[:ke]
    vcov = vcov_full[:ke, :ke]
    n_sel = int(np.sum([s.sum() for s in select_mat]))
    return RLassoIVResult(
        coef=coef,
        se=se,
        vcov=vcov,
        method="select Z (rlassoIVselectZ)",
        n_obs=n,
        treat_names=treat_names,
        selection={
            "n_selected_Z": n_sel,
            "selection_matrix_Z": np.column_stack(select_mat),
        },
        residuals=residuals,
    )


# ═══════════════════════════════════════════════════════════════════════
#  rlassoIVselectX  (control selection only)
# ═══════════════════════════════════════════════════════════════════════


def _select_x(
    x: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    post: bool,
    penalty: Optional[Dict[str, Any]],
    control: Optional[Dict[str, Any]],
    treat_names: List[str],
) -> RLassoIVResult:
    n = len(y)
    d = np.asarray(d, dtype=float).ravel()
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    num_iv = z.shape[1]

    lasso_d_x = rlasso(x, d, post=post, penalty=penalty, control=control)
    Dr = d - lasso_d_x.predict(x)
    lasso_y_x = rlasso(x, y, post=post, penalty=penalty, control=control)
    Yr = y - lasso_y_x.predict(x)

    Zr = np.empty((n, num_iv))
    nsel_x = int(lasso_y_x.index.sum() + lasso_d_x.index.sum())
    for i in range(num_iv):
        lasso_z_x = rlasso(x, z[:, i], post=post, penalty=penalty, control=control)
        Zr[:, i] = z[:, i] - lasso_z_x.predict(x)
        nsel_x += int(lasso_z_x.index.sum())

    res = _tsls(y=Yr, d=Dr, x=None, z=Zr, intercept=False, homoscedastic=True)
    return RLassoIVResult(
        coef=np.atleast_1d(res["coefficients"]),
        se=np.atleast_1d(res["se"]),
        vcov=np.atleast_2d(res["vcov"]),
        method="select X (rlassoIVselectX)",
        n_obs=n,
        treat_names=treat_names,
        selection={"n_selected_X": nsel_x},
        residuals=res["residuals"],
    )


# ═══════════════════════════════════════════════════════════════════════
#  rlassoIV double selection (select_Z = select_X = True)
# ═══════════════════════════════════════════════════════════════════════


def _select_both(
    x: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    post: bool,
    penalty: Optional[Dict[str, Any]],
    control: Optional[Dict[str, Any]],
    treat_names: List[str],
) -> RLassoIVResult:
    n = len(y)
    d = np.asarray(d, dtype=float).ravel()
    Zmat = np.column_stack([z, x])

    lasso_d_zx = rlasso(Zmat, d, post=post, penalty=penalty, control=control)
    lasso_y_x = rlasso(x, y, post=post, penalty=penalty, control=control)

    if lasso_d_zx.index.sum() == 0:
        return RLassoIVResult(
            coef=np.array([np.nan]),
            se=np.array([np.nan]),
            vcov=np.array([[np.nan]]),
            method="select Z & X (rlassoIV)",
            n_obs=n,
            treat_names=treat_names,
            selection={"note": "no variables selected in d ~ [z, x]"},
        )

    PZ = lasso_d_zx.predict(Zmat)
    lasso_PZ_x = rlasso(x, PZ, post=post, penalty=penalty, control=control)

    n_PZ_sel = int(lasso_PZ_x.index.sum())
    if n_PZ_sel == 0:
        Dr = d - d.mean()
        Zr = PZ - np.asarray(x, dtype=float).mean()
    else:
        Dr = d - lasso_PZ_x.predict(x)
        Zr = lasso_PZ_x.residuals

    if lasso_y_x.index.sum() == 0:
        Yr = y - y.mean()
    else:
        Yr = lasso_y_x.residuals

    res = _tsls(y=Yr, d=Dr, x=None, z=Zr, intercept=False, homoscedastic=False)
    n_sel_z = int(lasso_d_zx.index.sum())
    return RLassoIVResult(
        coef=np.atleast_1d(res["coefficients"]),
        se=np.atleast_1d(res["se"]),
        vcov=np.atleast_2d(res["vcov"]),
        method="select Z & X (rlassoIV)",
        n_obs=n,
        treat_names=treat_names,
        selection={
            "n_selected_Z": n_sel_z,
            "n_selected_X": int(lasso_PZ_x.index.sum()),
        },
        residuals=res["residuals"],
    )


# ═══════════════════════════════════════════════════════════════════════
#  Public dispatcher
# ═══════════════════════════════════════════════════════════════════════


def _grab(v: Any, data: Optional[pd.DataFrame], cols: bool = False) -> np.ndarray:
    out: np.ndarray
    if isinstance(v, str):
        assert data is not None, "string column reference requires `data`"
        out = np.asarray(data[v].values, dtype=float)
    elif cols and isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
        assert data is not None, "column-name reference requires `data`"
        out = np.asarray(data[list(v)].values, dtype=float)
    else:
        out = np.asarray(v, dtype=float)
    return out


def rlasso_iv(
    y: Union[np.ndarray, pd.Series, str],
    d: Union[np.ndarray, pd.Series, str],
    z: Union[np.ndarray, pd.DataFrame, Sequence[str]],
    x: Optional[Union[np.ndarray, pd.DataFrame, Sequence[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    select_Z: bool = True,
    select_X: bool = True,
    post: bool = True,
    intercept: bool = True,
    penalty: Optional[Dict[str, Any]] = None,
    control: Optional[Dict[str, Any]] = None,
) -> RLassoIVResult:
    """Instrumental-variables estimation with rigorous-Lasso selection.

    A faithful port of ``hdm::rlassoIV``.  Estimates the causal effect of
    an endogenous ``d`` on ``y`` using (potentially many) instruments
    ``z`` and optional high-dimensional controls ``x``.

    Parameters
    ----------
    y, d : outcome and the endogenous regressor (array, Series or column
        name).
    z : candidate instruments — ``p_z`` may exceed ``n``.
    x : exogenous controls (optional).
    data : DataFrame backing any string/column-name inputs.
    select_Z : bool, default True
        Lasso-select among the instruments.
    select_X : bool, default True
        Lasso-select among the controls (partialling-out).
    post : bool, default True
        Post-Lasso (OLS refit) inside every selection step.
    intercept : bool, default True
        Passed to the underlying ``rlasso`` first stages.
    penalty, control : dict, optional
        Forwarded to :func:`statspai.rlasso.rlasso` (penalty level,
        loadings, iteration controls).

    Returns
    -------
    RLassoIVResult

    Notes
    -----
    With ``select_Z=True, select_X=False`` this is the Belloni-Chen-
    Chernozhukov-Hansen (2012) optimal-instrument estimator that the
    eminent-domain application made famous; numbers agree with
    ``hdm::rlassoIV(..., select.X=FALSE, select.Z=TRUE)`` to ~1e-6.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> Z = rng.standard_normal((300, 5))  # candidate instruments
    >>> X = rng.standard_normal((300, 10))  # candidate controls
    >>> d = Z[:, 0] + 0.5 * X[:, 0] + rng.standard_normal(300)
    >>> y = 1.0 * d + X[:, 0] + rng.standard_normal(300)
    >>> res = sp.rlasso_iv(y=y, d=d, z=Z, x=X, select_Z=True, select_X=False)
    >>> float(res.se[0]) > 0  # res.coef[0] ~ 1.0 (BCH optimal-IV)
    True
    """
    Y = _grab(y, data).ravel()
    D = _grab(d, data)
    Zmat = _grab(z, data, cols=True)
    if Zmat.ndim == 1:
        Zmat = Zmat.reshape(-1, 1)
    Xmat = None
    if x is not None:
        Xmat = _grab(x, data, cols=True)
        if Xmat.ndim == 1:
            Xmat = Xmat.reshape(-1, 1)

    treat_names = (
        [d]
        if isinstance(d, str)
        else (
            [getattr(d, "name", None) or "d"]
            if D.ndim == 1
            else [f"d{i + 1}" for i in range(D.shape[1])]
        )
    )

    if not select_Z and not select_X:
        res = _tsls(y=Y, d=D, x=Xmat, z=Zmat, intercept=intercept, homoscedastic=False)
        a1 = res["a1"]
        return RLassoIVResult(
            coef=np.atleast_1d(res["coefficients"])[:a1],
            se=np.atleast_1d(res["se"])[:a1],
            vcov=np.atleast_2d(res["vcov"])[:a1, :a1],
            method="plain 2SLS (tsls)",
            n_obs=len(Y),
            treat_names=treat_names,
            residuals=res["residuals"],
        )
    if select_Z and not select_X:
        return _select_z(
            Xmat, D, Y, Zmat, post, intercept, penalty, control, treat_names
        )
    if select_X and not select_Z:
        if Xmat is None:
            raise ValueError("select_X=True requires controls `x`.")
        return _select_x(Xmat, D, Y, Zmat, post, penalty, control, treat_names)
    # both
    if Xmat is None:
        raise ValueError("select_X=True requires controls `x`.")
    return _select_both(Xmat, D, Y, Zmat, post, penalty, control, treat_names)
