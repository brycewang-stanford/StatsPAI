"""Treatment-effect estimation after rigorous-Lasso selection of controls.

A faithful port of ``hdm::rlassoEffect`` (single target) and
``hdm::rlassoEffects`` (many targets) — the high-dimensional analogue of
a partial regression coefficient, valid after Lasso-selecting among a
large set of controls.

Two methods, matching hdm exactly:

- ``"partialling out"`` (default): residualize ``y`` and ``d`` on the
  controls by ``rlasso``, then OLS the residuals.  SE is the textbook
  OLS slope variance.
- ``"double selection"``: take the union of the controls selected when
  regressing ``y`` on ``x`` and ``d`` on ``x``, refit ``y`` on
  ``[d, union]`` by OLS, and report a heteroskedasticity-robust SE.

Both deliver root-``n`` consistent, asymptotically normal estimates of
the structural coefficient on ``d`` under approximate sparsity (Belloni,
Chernozhukov & Hansen, 2014).

References
----------
Belloni, A., Chernozhukov, V. and Hansen, C. (2014). "Inference on
    Treatment Effects After Selection Among High-Dimensional Controls."
    *Review of Economic Studies*, 81(2), 608-650.
    [@belloni2014inference]

Chernozhukov, V., Hansen, C. and Spindler, M. (2016). "hdm:
    High-Dimensional Metrics." *The R Journal*, 8(2), 185-199.
    [@chernozhukov2016hdm]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin, attach_result_protocol
from ..core._call_record import records_call
from ..core._vcov_spec import accepts_vce
from ..exceptions import MethodIncompatibility
from ._absorb import absorb_codes, absorbed_levels, within
from ._core import _cluster_codes, rlasso


@dataclass
class RLassoEffectResult(ResultProtocolMixin):
    """Return of :func:`rlasso_effect`."""

    #: Verified paper.bib keys (CLAUDE.md §10).
    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "belloni2014inference",
        "chernozhukov2016hdm",
    )

    alpha: float
    se: float
    tstat: float
    pvalue: float
    method: str
    n_obs: int
    selection_index: np.ndarray
    target: str = "d"
    #: Final-regression residual and treatment residual (hdm's
    #: ``residuals$e`` / ``residuals$v``); the score of the estimate is
    #: ``e * v / mean(v**2)``. Used for joint inference across targets.
    resid_e: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    resid_v: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    #: Levels of each factor removed by ``absorb=`` (empty without it).
    absorb_levels: Tuple[int, ...] = ()

    def conf_int(self, level: float = 0.95) -> tuple:
        zc = stats.norm.ppf(0.5 + level / 2.0)
        return (self.alpha - zc * self.se, self.alpha + zc * self.se)

    def summary(self) -> str:
        lo, hi = self.conf_int()
        return "\n".join(
            [
                f"Rigorous-Lasso treatment effect  ({self.method})",
                "-" * 60,
                f"  Observations         : {self.n_obs}",
                *(
                    [
                        "  Absorbed levels      : "
                        + ", ".join(map(str, self.absorb_levels))
                    ]
                    if self.absorb_levels
                    else []
                ),
                f"  Controls selected    : {int(self.selection_index.sum())}",
                "",
                "             coef     std.err      z      P>|z|      95% CI",
                f"  {self.target:<8}{self.alpha:>10.4f}  {self.se:>9.4f}"
                f"  {self.tstat:>7.3f}  {self.pvalue:>8.4f}  [{lo:.3f}, {hi:.3f}]",
            ]
        )


def _ols(y: np.ndarray, X: np.ndarray) -> tuple:
    """OLS with intercept; returns (coef[incl intercept], resid, XtX_inv, dof)."""
    n = len(y)
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    XtX_inv = np.linalg.inv(A.T @ A)
    return beta, resid, XtX_inv, n - A.shape[1]


def _cluster_var(
    e: np.ndarray, v: np.ndarray, codes: np.ndarray, n_clusters: int
) -> float:
    """Cluster-robust variance of a partialled-out coefficient.

    ``sum_g (sum_{i in g} v_i e_i)^2 / (sum_i v_i^2)^2`` for final
    residual ``e`` and treatment residual ``v``; no small-sample factor
    (the ``pdslasso`` / ``ivreg2`` convention).
    """
    sums = np.bincount(codes, weights=v * e, minlength=n_clusters)
    return float(sums @ sums) / float(v @ v) ** 2


def _ols_nocons(y: np.ndarray, X: np.ndarray) -> tuple:
    """OLS without an intercept; returns (coef, resid, XtX_inv)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid, np.linalg.inv(X.T @ X)


def _absorbed_effect(
    X: np.ndarray,
    yv: np.ndarray,
    dv: np.ndarray,
    method: str,
    post: bool,
    I3: Optional[np.ndarray],
    penalty: Optional[Dict[str, Any]],
    control: Dict[str, Any],
    codes: Optional[np.ndarray],
    n_clusters: int,
    robust: Optional[bool],
    target: Any,
) -> tuple:
    """``pdslasso ..., fe``: the effect on within-transformed data.

    No intercept anywhere, and no small-sample factor in any variance (the
    ``ivreg2`` defaults pdslasso uses; the absorbed levels are not
    subtracted either): homoskedastic ``e'e / n``, robust HC0, cluster
    without ``G / (G - 1)``.
    """
    n = len(yv)

    def _sel(target_v: np.ndarray) -> Any:
        return rlasso(
            X,
            target_v,
            post=post,
            intercept=False,
            penalty=penalty,
            control=control,
            cluster=codes,
        )

    if method == "partialling out":
        reg1, reg2 = _sel(yv), _sel(dv)
        e_y, v = reg1.residuals, reg2.residuals
        _warn_if_unidentified(v, dv, target)
        alpha = float(v @ e_y) / float(v @ v)
        resid = e_y - alpha * v
        sel = np.asarray(reg1.index | reg2.index, dtype=bool)
    elif method == "double selection":
        union = _sel(dv).index | _sel(yv).index
        if I3 is not None:
            union = union | np.asarray(I3, bool)
        sel = np.asarray(union, dtype=bool)
        Xsel = np.column_stack([dv, X[:, sel]])
        beta, resid, _ = _ols_nocons(yv, Xsel)
        alpha = float(beta[0])
        v = dv - X[:, sel] @ _ols_nocons(dv, X[:, sel])[0] if sel.any() else dv
        _warn_if_unidentified(v, dv, target)
    else:
        raise ValueError(
            f"method must be 'partialling out' or 'double selection', got {method!r}"
        )
    if codes is not None:
        var = _cluster_var(resid, v, codes, n_clusters)
    elif robust or (robust is None and method == "double selection"):
        var = _cluster_var(resid, v, np.arange(n), n)
    else:
        var = float(resid @ resid) / n / float(v @ v)
    return alpha, float(np.sqrt(var)), sel, resid, v


def _warn_if_unidentified(resid_d: np.ndarray, d: np.ndarray, target: Any) -> None:
    """Warn when the target is (numerically) spanned by the selected controls.

    Then the effect is not identified and the reported coefficient and SE
    are ratios of rounding noise -- on hdm's cps2012 example the target
    ``female:hsd08`` (a single non-zero row in an 800-row subsample) has a
    residual variance 4e-30 of its own, and hdm reports -4.7e13 where
    StatsPAI reports 1.9e-36; both are meaningless. hdm does not warn.
    """
    dc = d - d.mean()
    tot = float(dc @ dc)
    if tot <= 0 or float(resid_d @ resid_d) <= 1e-12 * tot:
        warnings.warn(
            f"rlasso_effect: target {target!r} is (numerically) a linear "
            "combination of the selected controls; its effect is not "
            "identified and the reported estimate / SE are meaningless.",
            RuntimeWarning,
            stacklevel=3,
        )


@records_call("rlasso_effect")
@accepts_vce(
    default=None,
    native={"robust": {"robust": True}, "nonrobust": {"robust": False}},
)
def rlasso_effect(
    x: Union[np.ndarray, pd.DataFrame, Sequence[str]],
    y: Union[np.ndarray, pd.Series, str],
    d: Union[np.ndarray, pd.Series, str],
    method: str = "partialling out",
    post: bool = True,
    I3: Optional[np.ndarray] = None,
    data: Optional[pd.DataFrame] = None,
    penalty: Optional[Dict[str, Any]] = None,
    control: Optional[Dict[str, Any]] = None,
    cluster: Optional[Union[np.ndarray, pd.Series, str]] = None,
    robust: Optional[bool] = None,
    absorb: Optional[Any] = None,
) -> RLassoEffectResult:
    """Effect of ``d`` on ``y`` after Lasso-selecting controls ``x``.

    Faithful port of ``hdm::rlassoEffect``.

    Parameters
    ----------
    x : (n, p) controls (array, DataFrame or column names).
    y, d : outcome and the single target regressor.
    method : {"partialling out", "double selection"}
        See the module docstring.
    post : bool, default True
        Post-Lasso inside the selection steps.
    I3 : bool array, optional
        Amelioration set forced into the control set (double-selection
        only) — hdm's ``I3`` argument.
    data : DataFrame backing string/column-name inputs.
    penalty, control : dict, optional
        Forwarded to :func:`statspai.rlasso.rlasso`.
    cluster : array-like or str, optional
        Cluster identifier (a column name when ``data`` is given). Every
        selection step then uses the cluster-Lasso loadings of
        :func:`statspai.rlasso.rlasso` and the standard error is
        cluster-robust [@belloni2016inference] with no small-sample
        factor, as in Stata's ``pdslasso, cluster()`` (multiply the
        variance by ``G / (G - 1)`` for the ``regress, cluster()``
        convention). For a fixed-effects panel pass within-transformed
        ``x``, ``y`` and ``d``. hdm has no counterpart.
    robust : bool, optional
        Standard error without ``cluster``. ``None`` keeps hdm's choice for
        the method: homoskedastic for ``"partialling out"`` (the OLS slope
        variance of the residual regression), heteroskedasticity-robust for
        ``"double selection"``. ``True`` gives the robust variance
        ``sum_i v_i^2 e_i^2 / (sum_i v_i^2)^2`` for partialling out (no
        small-sample factor, the cluster formula with one row per cluster);
        ``False`` gives the OLS variance of the final regression for
        double selection. ``vce="robust"`` / ``vce="iid"`` set it.

        .. versionchanged:: 1.41
           ``vce="robust"`` used to be accepted and ignored, returning the
           homoskedastic SE of partialling out.
    absorb : str, list of str, or array-like, optional
        Fixed effects to absorb (column names when ``data`` is given, or
        one identifier array per factor). ``y``, ``d`` and every control
        are within-transformed before any Lasso, and the estimator then
        follows Stata's ``pdslasso y d (x...), fe`` [@belloni2016inference]
        rather than hdm, which has no fixed-effects option: the selection
        steps use ``control={"iteration": "lassopack"}`` unless ``control``
        says otherwise, the final regressions have no intercept, and no
        variance carries a small-sample factor. ``robust=None`` then means
        homoskedastic ``e'e / n`` for partialling out and HC0 for double
        selection; ``robust=True`` is HC0 for both, ``robust=False`` the
        homoskedastic form for both. Selection matches ``pdslasso ..., fe
        robust`` or ``fe cluster()`` (Stata's default homoskedastic
        penalty is not offered). With several factors the result equals
        ``pdslasso ..., fe partial(i.<f2> ...)``.

    Returns
    -------
    RLassoEffectResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 12))  # candidate controls
    >>> d = X[:, 0] + rng.standard_normal(200)  # treatment
    >>> y = 1.5 * d + X[:, 1] + rng.standard_normal(200)
    >>> res = sp.rlasso_effect(X[:, 1:], y, d, method="partialling out")
    >>> float(res.se) > 0
    True
    >>> bool(np.isfinite(res.alpha))  # ~1.5
    True
    """
    if data is not None:
        X = (
            np.asarray(data[list(x)].values, dtype=float)
            if (isinstance(x, (list, tuple)) and all(isinstance(c, str) for c in x))
            else np.asarray(x, dtype=float)
        )
        yv = (
            np.asarray(data[y].values, dtype=float)
            if isinstance(y, str)
            else np.asarray(y, dtype=float)
        )
        dv = (
            np.asarray(data[d].values, dtype=float)
            if isinstance(d, str)
            else np.asarray(d, dtype=float)
        )
        target = d if isinstance(d, str) else "d"
    else:
        X = np.asarray(x, dtype=float)
        yv = np.asarray(y, dtype=float)
        dv = np.asarray(d, dtype=float)
        target = getattr(d, "name", None) or "d"
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    yv = yv.ravel()
    dv = dv.ravel()
    n = len(yv)
    codes: Optional[np.ndarray] = None
    n_clusters = 0
    if cluster is not None:
        if isinstance(cluster, str):
            if data is None:
                raise MethodIncompatibility(
                    "cluster given as a column name needs `data`."
                )
            cluster = data[cluster].values
        codes, n_clusters = _cluster_codes(cluster, n)
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

    if absorb is not None:
        fe_codes = absorb_codes(absorb, n, data)
        X, yv, dv = within(X, fe_codes), within(yv, fe_codes), within(dv, fe_codes)
        ctrl = {"iteration": "lassopack", **dict(control or {})}
        alpha, se, sel, res_e, res_v = _absorbed_effect(
            X,
            yv,
            dv,
            method,
            post,
            I3,
            penalty,
            ctrl,
            codes,
            n_clusters,
            robust,
            target,
        )
        tval = alpha / se if se > 0 else np.nan
        return RLassoEffectResult(
            alpha=alpha,
            se=se,
            tstat=float(tval),
            pvalue=2.0 * float(stats.norm.cdf(-abs(tval))),
            method=method,
            n_obs=n,
            selection_index=sel,
            target=target,
            resid_e=np.asarray(res_e, dtype=float),
            resid_v=np.asarray(res_v, dtype=float),
            absorb_levels=absorbed_levels(fe_codes),
        )

    if method == "partialling out":
        reg1 = rlasso(X, yv, post=post, penalty=penalty, control=control, cluster=codes)
        yr = reg1.residuals
        reg2 = rlasso(X, dv, post=post, penalty=penalty, control=control, cluster=codes)
        dr = reg2.residuals
        _warn_if_unidentified(dr, dv, target)
        # lm(yr ~ dr) with intercept
        beta, resid, XtX_inv, dof = _ols(yr, dr.reshape(-1, 1))
        alpha = float(beta[1])
        sigma2 = float(resid @ resid) / dof
        var = sigma2 * XtX_inv[1, 1]
        if codes is not None:
            var = _cluster_var(resid, dr - dr.mean(), codes, n_clusters)
        elif robust:
            var = _cluster_var(resid, dr - dr.mean(), np.arange(n), n)
        se = float(np.sqrt(var))
        sel = np.asarray(reg1.index | reg2.index, dtype=bool)
        res_e, res_v = resid, dr
    elif method == "double selection":
        I1 = rlasso(
            X, dv, post=post, penalty=penalty, control=control, cluster=codes
        ).index
        I2 = rlasso(
            X, yv, post=post, penalty=penalty, control=control, cluster=codes
        ).index
        if I3 is not None:
            idx_union = (
                np.asarray(I1, bool) | np.asarray(I2, bool) | np.asarray(I3, bool)
            )
        else:
            idx_union = np.asarray(I1, bool) | np.asarray(I2, bool)
        sum_I = int(idx_union.sum())
        if sum_I == 0:
            Xsel = dv.reshape(-1, 1)
            beta, resid, XtX_inv, dof = _ols(yv, Xsel)
            alpha = float(beta[1])
            xi = resid * np.sqrt(n / (n - sum_I - 1))
            v = dv - dv.mean()
        else:
            Xsel = np.column_stack([dv, X[:, idx_union]])
            beta, resid, XtX_inv, dof = _ols(yv, Xsel)
            alpha = float(beta[1])
            xi = resid * np.sqrt(n / (n - sum_I - 1))
            # reg2 <- lm(d ~ selected controls)  (drop d column → X[:, union])
            _, v, _, _ = _ols(dv, X[:, idx_union])
        _warn_if_unidentified(v, dv, target)
        mv2 = float(np.mean(v**2))
        var = (1.0 / n) * (1.0 / mv2) * float(np.mean(v**2 * xi**2)) * (1.0 / mv2)
        if codes is not None:
            var = _cluster_var(resid, v, codes, n_clusters)
        elif robust is False:
            var = float(resid @ resid) / dof * XtX_inv[1, 1]
        se = float(np.sqrt(var))
        sel = idx_union
        res_e, res_v = xi, v
    else:
        raise ValueError(
            f"method must be 'partialling out' or 'double selection', got {method!r}"
        )

    tval = alpha / se if se > 0 else np.nan
    pval = 2.0 * float(stats.norm.cdf(-abs(tval)))
    return RLassoEffectResult(
        alpha=alpha,
        se=se,
        tstat=float(tval),
        pvalue=pval,
        method=method,
        n_obs=n,
        selection_index=sel,
        target=target,
        resid_e=np.asarray(res_e, dtype=float),
        resid_v=np.asarray(res_v, dtype=float),
    )


@attach_result_protocol
class RLassoEffectsResult(Dict[str, RLassoEffectResult]):
    """Effects of several targets, with their joint covariance.

    A ``dict`` from column name to :class:`RLassoEffectResult`, as
    :func:`rlasso_effects` has always returned, plus what joint inference
    needs. The estimates are asymptotically jointly normal with covariance
    ``Omega / n``, ``Omega[j, l] = E[e_j v_j e_l v_l] / (E[v_j^2] E[v_l^2])``,
    which is what ``confint(<rlassoEffects>, joint = TRUE)`` simulates from
    in hdm [@chernozhukov2016hdm].
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = RLassoEffectResult._citation_keys

    def to_dict(self) -> Dict[str, Any]:
        """Each target's result, and the joint covariance of the estimates."""
        vcov = self.vcov()
        return {
            "effects": {name: res.to_dict() for name, res in self.items()},
            "vcov": {
                "names": list(vcov.index),
                "matrix": vcov.to_numpy().tolist(),
            },
        }

    #: Cluster identifier the effects were estimated with, if any.
    _cluster: Any = None

    def vcov(self) -> pd.DataFrame:
        """Joint covariance ``Omega / n`` of the estimates.

        With ``cluster=`` the scores are summed within cluster first.
        """
        names = list(self)
        e = np.column_stack([self[k].resid_e for k in names])
        v = np.column_stack([self[k].resid_v for k in names])
        n = e.shape[0]
        ev = e * v / np.mean(v**2, axis=0)
        if self._cluster is not None:
            codes, n_clusters = _cluster_codes(self._cluster, n)
            sums = np.zeros((n_clusters, ev.shape[1]))
            np.add.at(sums, codes, ev)
            ev = sums
        return pd.DataFrame(ev.T @ ev / n / n, index=names, columns=names)

    def conf_int(
        self,
        level: float = 0.95,
        joint: bool = False,
        n_draws: int = 100_000,
        seed: Optional[int] = 0,
    ) -> pd.DataFrame:
        """Confidence intervals, pointwise or simultaneous.

        Parameters
        ----------
        level : float, default 0.95
        joint : bool, default False
            ``False``: each interval is ``estimate -/+ z * se`` with the
            standard error of the single-target fit. ``True``: a sup-t
            band, ``estimate -/+ c * sqrt(diag(vcov))``, where ``c`` is the
            ``level`` quantile of ``max_j |Z_j|`` for ``Z`` normal with the
            correlation matrix of the estimates; all intervals then cover
            at once with probability ``level``.
        n_draws : int, default 100_000
            Draws for the critical value (hdm uses 500).
        seed : int or None, default 0

        Returns
        -------
        pd.DataFrame
            Columns ``estimate``, ``se``, ``lower``, ``upper``. The
            critical value is in ``.attrs['critical_value']``.
        """
        names = list(self)
        est = np.array([self[k].alpha for k in names])
        if not joint:
            se = np.array([self[k].se for k in names])
            crit = float(stats.norm.ppf(0.5 + level / 2.0))
        else:
            cov = self.vcov().to_numpy()
            se = np.sqrt(np.diag(cov))
            corr = cov / np.outer(se, se)
            rng = np.random.default_rng(seed)
            # Eigen-root so that a singular correlation matrix (two targets
            # with the same score) is handled.
            w, q = np.linalg.eigh(corr)
            root = q * np.sqrt(np.clip(w, 0.0, None))
            z = rng.standard_normal((int(n_draws), len(names))) @ root.T
            crit = float(np.quantile(np.max(np.abs(z), axis=1), level))
        out = pd.DataFrame(
            {
                "estimate": est,
                "se": se,
                "lower": est - crit * se,
                "upper": est + crit * se,
            },
            index=names,
        )
        out.attrs["critical_value"] = crit
        out.attrs["joint"] = bool(joint)
        return out

    def summary(self) -> str:
        tab = self.conf_int()
        lines = ["Rigorous-Lasso treatment effects", "-" * 60]
        lines.append(tab.round(4).to_string())
        return "\n".join(lines)


@records_call("rlasso_effects")
@accepts_aliases(covariates="X")
@accepts_vce(
    default=None,
    native={"robust": {"robust": True}, "nonrobust": {"robust": False}},
)
def rlasso_effects(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    index: Optional[Sequence[int]] = None,
    method: str = "partialling out",
    post: bool = True,
    data: Optional[pd.DataFrame] = None,
    penalty: Optional[Dict[str, Any]] = None,
    control: Optional[Dict[str, Any]] = None,
    cluster: Optional[Union[np.ndarray, pd.Series, str]] = None,
    robust: Optional[bool] = None,
    absorb: Optional[Any] = None,
) -> "RLassoEffectsResult":
    """Estimate the effect of each targeted column of ``X`` on ``y``.

    Faithful port of ``hdm::rlassoEffects``: for every target column
    ``j`` in ``index``, treat column ``j`` as ``d`` and the remaining
    columns as controls. ``robust`` (and ``vce=``) and ``absorb`` are
    passed to :func:`rlasso_effect` for every target; with ``absorb`` each
    target follows ``pdslasso ..., fe`` (see :func:`rlasso_effect`).

    Returns
    -------
    RLassoEffectsResult
        A ``dict`` mapping ``column name -> RLassoEffectResult``, with
        ``.vcov()`` (joint covariance of the estimates) and
        ``.conf_int(joint=True)`` (simultaneous sup-t band, hdm's
        ``confint(..., joint = TRUE)``).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 8))
    >>> y = X[:, 0] - 0.8 * X[:, 1] + rng.standard_normal(200)
    >>> out = sp.rlasso_effects(X, y, index=[0, 1], method="partialling out")
    >>> len(out)  # one result per targeted column
    2
    >>> all(r.se > 0 for r in out.values())
    True

    Intervals that cover both effects at once are wider than the
    pointwise ones:

    >>> band = out.conf_int(joint=True)
    >>> bool(band.attrs["critical_value"] > 1.96)
    True
    """
    if isinstance(X, pd.DataFrame):
        cols = list(X.columns)
        Xv = X.values.astype(float)
    elif data is not None and isinstance(X, (list, tuple)):
        cols = list(X)
        Xv = data[cols].values.astype(float)
    else:
        Xv = np.asarray(X, dtype=float)
        cols = [f"V{j + 1}" for j in range(Xv.shape[1])]
    yv = np.asarray(
        data[y].values if (data is not None and isinstance(y, str)) else y, dtype=float
    ).ravel()

    if index is None:
        index = list(range(Xv.shape[1]))
    cluster_arr = (
        data[cluster].values
        if (isinstance(cluster, str) and data is not None)
        else cluster
    )

    absorb_arr: Any = None
    if absorb is not None:
        absorb_arr = absorb_codes(absorb, len(yv), data)

    out = RLassoEffectsResult()
    out._cluster = cluster_arr
    for j in index:
        d = pd.Series(Xv[:, j], name=cols[j])
        Xt = np.delete(Xv, j, axis=1)
        res = rlasso_effect(
            Xt,
            yv,
            d,
            method=method,
            post=post,
            penalty=penalty,
            control=control,
            cluster=cluster_arr,
            robust=robust,
            absorb=absorb_arr,
        )
        res.target = cols[j]
        out[cols[j]] = res
    return out
