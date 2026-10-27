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

import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg as scipy_linalg
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..core._call_record import records_call
from ..core._vcov_spec import accepts_vce
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._core import _cluster_codes, rlasso

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
    codes: Optional[np.ndarray] = None,
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
    if codes is not None:
        # Cluster-robust: sum the moment z_i e_i within cluster; no
        # small-sample factor (the ivlasso / ivreg2 convention).
        n_clusters = int(codes.max()) + 1
        sums = np.zeros((n_clusters, Z.shape[1]))
        np.add.at(sums, codes, Z * e)
        S = sums.T @ sums
        VC1 = M @ (Mxz @ Mzz @ S @ Mzz @ Mxz.T) @ M
    elif homoscedastic:
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
    codes: Optional[np.ndarray] = None,
    robust: bool = True,
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
            Z,
            di,
            post=post,
            intercept=intercept,
            penalty=penalty,
            control=control,
            cluster=codes,
        )
        if fit.index.sum() == 0:
            dihat = np.full(n, di.mean())
            flag_const += 1
            select_mat.append(np.zeros(Z.shape[1], dtype=bool))
        else:
            dihat = fit.predict(Z)
            select_mat.append(fit.index.copy())
        Dhat_list.append(dihat)

    # An endogenous regressor with no *instrument* selected (nothing, or
    # only controls) is not identified: its first stage is a constant or a
    # function of the controls. hdm solves on anyway and returned SEs of
    # order 1e14 (AI-tocracy replication, Table II B); report NaN instead.
    n_inst = np.asarray(z).reshape(n, -1).shape[1]
    unidentified = [
        treat_names[i] if i < len(treat_names) else f"d{i}"
        for i, sm in enumerate(select_mat)
        if not np.any(sm[:n_inst])
    ]
    if unidentified:
        warnings.warn(
            "rlasso_iv: no instrument was selected for "
            f"{unidentified}; the coefficient is not identified and is "
            "reported as NaN. Lower the penalty or check instrument strength.",
            UserWarning,
            stacklevel=3,
        )
        nan = np.full(ke, np.nan)
        return RLassoIVResult(
            coef=nan,
            se=nan.copy(),
            vcov=np.full((ke, ke), np.nan),
            method="select Z (rlassoIVselectZ)",
            n_obs=n,
            treat_names=treat_names,
            selection={
                "n_selected_Z": int(np.sum([s.sum() for s in select_mat])),
                "selection_matrix_Z": np.column_stack(select_mat),
                "unidentified": unidentified,
            },
        )

    Dhat = np.column_stack(Dhat_list)
    # The second stage is the just-identified IV of y on [d, x, 1] with
    # instruments [dhat, x, 1]. Solve it exactly by partialling out the
    # controls (Frisch-Waugh) rather than with ``ginv`` of the joint
    # system as hdm 0.3.2 does: Dhat'[d, x] squares the conditioning of x,
    # and on EminentDomain (x full rank, cond 7e6) MASS's sqrt(eps) cutoff
    # truncates 34 of 81 singular values, so the "estimate" is a truncated
    # SVD that moves with the tolerance (0.227 at sqrt(eps), 0.0054 at
    # 1e-10) while the IV estimate is 0.0154 (= AER::ivreg).
    #
    # The first stage fits an intercept, so the second stage must carry
    # one too. ``hdm::rlassoIVselectZ`` (0.3.2) omits it: with a nonzero
    # mean in y or d its IV slope is then the ratio of uncentred moments
    # and is inconsistent (truth -2 → -0.64 on a known-truth DGP). With
    # every instrument selected the corrected estimator equals 2SLS.
    W_parts = [] if x is None else [x]
    if intercept:
        W_parts.append(np.ones((n, 1)))
    rank = 0
    if W_parts:
        W = np.column_stack(W_parts)
        Q, R, _ = scipy_linalg.qr(W, mode="economic", pivoting=True)
        rdiag = np.abs(np.diag(R))
        rank = int(np.sum(rdiag > max(W.shape) * np.finfo(float).eps * rdiag[0]))
        Q = Q[:, :rank]

        def _partial(v: np.ndarray) -> np.ndarray:
            out: np.ndarray = v - Q @ (Q.T @ v)
            return out

    else:

        def _partial(v: np.ndarray) -> np.ndarray:
            return v

    Dh_r = _partial(Dhat)
    D_r = _partial(d)
    A = Dh_r.T @ D_r
    Ainv = np.linalg.inv(A)
    coef = np.atleast_1d(Ainv @ (Dh_r.T @ _partial(y)))
    # Residuals of the full second stage equal the partialled ones.
    residuals = _partial(y) - D_r @ coef
    if codes is None and not robust:
        # Classical 2SLS with instruments [dhat, x, 1]: s^2 (X'P_Z X)^{-1},
        # s^2 dividing by n minus every second-stage coefficient.
        s2 = float(residuals @ residuals) / max(n - ke - rank, 1)
        Omega = s2 * (Dh_r.T @ Dh_r)
    elif codes is None:
        Omega = Dh_r.T @ (Dh_r * (residuals**2)[:, None])
    else:
        n_clusters = int(codes.max()) + 1
        sums = np.zeros((n_clusters, ke))
        np.add.at(sums, codes, Dh_r * residuals[:, None])
        Omega = sums.T @ sums
    # coef = A^{-1} Dh_r'y_r, so Var(coef) = A^{-1} Omega A^{-1}'. hdm 0.3.2
    # uses ginv(d_aug'Dhat) = A^+' here, i.e. A^+' Omega A^+, which differs
    # whenever controls make A asymmetric (hdm's own tsls(homoscedastic=
    # FALSE) on [dhat, 1, x] gives the A^{-1} Omega A^{-1}' value).
    vcov = Ainv @ Omega @ Ainv.T
    se = np.sqrt(np.diag(vcov))
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
    codes: Optional[np.ndarray] = None,
    intercept: bool = True,
    robust: bool = False,
) -> RLassoIVResult:
    n = len(y)
    d = np.asarray(d, dtype=float).ravel()
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    num_iv = z.shape[1]

    kw: Dict[str, Any] = dict(
        post=post,
        intercept=intercept,
        penalty=penalty,
        control=control,
        cluster=codes,
    )
    lasso_d_x = rlasso(x, d, **kw)
    Dr = d - lasso_d_x.predict(x)
    lasso_y_x = rlasso(x, y, **kw)
    Yr = y - lasso_y_x.predict(x)

    Zr = np.empty((n, num_iv))
    nsel_x = int(lasso_y_x.index.sum() + lasso_d_x.index.sum())
    for i in range(num_iv):
        lasso_z_x = rlasso(x, z[:, i], **kw)
        Zr[:, i] = z[:, i] - lasso_z_x.predict(x)
        nsel_x += int(lasso_z_x.index.sum())

    res = _tsls(
        y=Yr,
        d=Dr,
        x=None,
        z=Zr,
        intercept=False,
        homoscedastic=not robust,
        codes=codes,
    )
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
    codes: Optional[np.ndarray] = None,
    intercept: bool = True,
    robust: bool = True,
) -> RLassoIVResult:
    n = len(y)
    d = np.asarray(d, dtype=float).ravel()
    Zmat = np.column_stack([z, x])

    kw: Dict[str, Any] = dict(
        post=post,
        intercept=intercept,
        penalty=penalty,
        control=control,
        cluster=codes,
    )
    lasso_d_zx = rlasso(Zmat, d, **kw)
    lasso_y_x = rlasso(x, y, **kw)

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
    lasso_PZ_x = rlasso(x, PZ, **kw)

    # With nothing selected the Lasso fit is the intercept alone, so the
    # residual is the centred variable (or the raw one without an
    # intercept). ``hdm::rlassoIV`` 0.3.2 centres the instrument at
    # ``mean(x)``, the grand mean of the whole control matrix, instead of
    # ``mean(PZ)``. The slope is unchanged (Dr and Yr are centred) but the
    # robust variance is not: it inflated the SE five-fold on a DGP whose
    # first-stage fit has mean 7.
    n_PZ_sel = int(lasso_PZ_x.index.sum())
    if n_PZ_sel == 0:
        Dr = d - d.mean() if intercept else d.copy()
        Zr = PZ - PZ.mean() if intercept else PZ.copy()
    else:
        Dr = d - lasso_PZ_x.predict(x)
        Zr = lasso_PZ_x.residuals

    if lasso_y_x.index.sum() == 0:
        Yr = y - y.mean() if intercept else y.copy()
    else:
        Yr = lasso_y_x.residuals

    res = _tsls(
        y=Yr,
        d=Dr,
        x=None,
        z=Zr,
        intercept=False,
        homoscedastic=not robust,
        codes=codes,
    )
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


@records_call("rlasso_iv")
@accepts_vce(
    default=None,
    native={"robust": {"robust": True}, "nonrobust": {"robust": False}},
)
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
    cluster: Optional[Union[np.ndarray, pd.Series, str]] = None,
    robust: Optional[bool] = None,
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
        Fit an intercept in every ``rlasso`` step (all selection paths)
        and in the second stage of the ``select_Z=True, select_X=False``
        path. Set ``False`` only for data that are already centred, e.g.
        within-transformed panels.
    penalty, control : dict, optional
        Forwarded to :func:`statspai.rlasso.rlasso` (penalty level,
        loadings, iteration controls).
    cluster : array-like or str, optional
        Cluster identifier (a column name when ``data`` is given). All
        selection steps use the cluster-Lasso loadings
        [@belloni2016inference] and the reported variance is
        cluster-robust with no small-sample factor (``ivlasso``'s rule).
        For a fixed-effects panel pass within-transformed variables and
        ``intercept=False``. With ``select_Z=True, select_X=False`` and no
        controls the point estimate is the post-Lasso IV estimate of
        Stata's ``ivlasso, cluster()``.
    robust : bool, optional
        Standard error without ``cluster``. ``None`` keeps hdm's choice for
        each path: homoskedastic for ``select_Z=False, select_X=True``,
        heteroskedasticity-robust (HC0) for the other three. ``True`` /
        ``False`` force the robust / classical 2SLS variance on any path;
        ``vce="robust"`` / ``vce="iid"`` set it.

        .. versionchanged:: 1.41
           ``vce="robust"`` used to be accepted and ignored, returning the
           homoskedastic SE of the ``select_X``-only path.

    Returns
    -------
    RLassoIVResult

    Notes
    -----
    With ``select_Z=True, select_X=False`` this is the Belloni-Chen-
    Chernozhukov-Hansen (2012) optimal-instrument estimator that the
    eminent-domain application made famous. Three paths match
    ``hdm::rlassoIV`` 0.3.2 to ~1e-14. This one departs from
    ``hdm::rlassoIVselectZ`` in three places, all reference bugs:

    * the second stage keeps an intercept when ``intercept=True`` and the
      controls do not already span a constant. hdm drops it, so with a
      nonzero mean in ``y`` or ``d`` its slope is inconsistent (truth -2,
      hdm -0.65 on ``tests/reference_parity/_fixtures/
      rlasso_iv_nonzero_mean.csv``). With every instrument selected the
      estimate is now exactly robust 2SLS;
    * the second stage is solved exactly, by partialling the controls out
      with a pivoted QR. hdm takes ``MASS::ginv`` of ``Dhat'[d, x]``, which
      squares the conditioning of ``x``; on EminentDomain logGDP (``x``
      full rank, cond 7e6) the default cutoff discards 34 of 81 singular
      values and hdm's 0.2274 is a truncated-SVD value that moves with the
      tolerance. The IV estimate is 0.0154 (SE 0.0141), equal to
      ``AER::ivreg`` with HC0 SEs to 1e-12;
    * the sandwich is ``A^{-1} Omega A^{-1}'`` with ``A = Dhat'[d, x]``;
      hdm transposes ``A``, which changes the SE whenever the first stage
      leaves some control out (hdm's own ``tsls(homoscedastic=FALSE)`` on
      ``[dhat, 1, x]`` gives the corrected value).

    With double selection and no control selected for the instrument,
    the instrument is centred at its own mean; hdm centres it at the grand
    mean of the control matrix, which leaves the slope unchanged but
    inflates the SE.

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

    codes: Optional[np.ndarray] = None
    if cluster is not None:
        if isinstance(cluster, str):
            if data is None:
                raise MethodIncompatibility(
                    "cluster given as a column name needs `data`."
                )
            cluster = data[cluster].values
        codes, _ = _cluster_codes(cluster, len(Y))
    if robust is not None and not isinstance(robust, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"robust must be True, False or None, got {robust!r}.",
            recovery_hint="Use vce='robust' or vce='iid'.",
        )
    if codes is not None and robust is False:
        raise MethodIncompatibility(
            "cluster= gives a cluster-robust SE; it cannot be combined with "
            "robust=False (vce='iid').",
            recovery_hint="Drop one of cluster= and robust=False.",
        )
    for label, arr in (("y", Y), ("d", D), ("z", Zmat), ("x", Xmat)):
        if arr is not None and not np.all(np.isfinite(arr)):
            raise DataInsufficient(
                f"rlasso_iv: {label} contains missing or infinite values.",
                recovery_hint="Drop or impute the incomplete rows first.",
            )

    if not select_Z and not select_X:
        res = _tsls(
            y=Y,
            d=D,
            x=Xmat,
            z=Zmat,
            intercept=intercept,
            homoscedastic=robust is False,
            codes=codes,
        )
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
            Xmat,
            D,
            Y,
            Zmat,
            post,
            intercept,
            penalty,
            control,
            treat_names,
            codes,
            robust=robust is not False,
        )
    if select_X and not select_Z:
        if Xmat is None:
            raise ValueError("select_X=True requires controls `x`.")
        return _select_x(
            Xmat,
            D,
            Y,
            Zmat,
            post,
            penalty,
            control,
            treat_names,
            codes,
            intercept,
            robust=robust is True,
        )
    # both
    if Xmat is None:
        raise ValueError("select_X=True requires controls `x`.")
    return _select_both(
        Xmat,
        D,
        Y,
        Zmat,
        post,
        penalty,
        control,
        treat_names,
        codes,
        intercept,
        robust=robust is not False,
    )
