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

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ._core import rlasso


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

    if method == "partialling out":
        reg1 = rlasso(X, yv, post=post, penalty=penalty, control=control)
        yr = reg1.residuals
        reg2 = rlasso(X, dv, post=post, penalty=penalty, control=control)
        dr = reg2.residuals
        # lm(yr ~ dr) with intercept
        beta, resid, XtX_inv, dof = _ols(yr, dr.reshape(-1, 1))
        alpha = float(beta[1])
        sigma2 = float(resid @ resid) / dof
        var = sigma2 * XtX_inv[1, 1]
        se = float(np.sqrt(var))
        sel = np.asarray(reg1.index | reg2.index, dtype=bool)
    elif method == "double selection":
        I1 = rlasso(X, dv, post=post, penalty=penalty, control=control).index
        I2 = rlasso(X, yv, post=post, penalty=penalty, control=control).index
        if I3 is not None:
            idx_union = (
                np.asarray(I1, bool) | np.asarray(I2, bool) | np.asarray(I3, bool)
            )
        else:
            idx_union = np.asarray(I1, bool) | np.asarray(I2, bool)
        sum_I = int(idx_union.sum())
        if sum_I == 0:
            Xsel = dv.reshape(-1, 1)
            beta, resid, _, _ = _ols(yv, Xsel)
            alpha = float(beta[1])
            xi = resid * np.sqrt(n / (n - sum_I - 1))
            v = dv - dv.mean()
        else:
            Xsel = np.column_stack([dv, X[:, idx_union]])
            beta, resid, _, _ = _ols(yv, Xsel)
            alpha = float(beta[1])
            xi = resid * np.sqrt(n / (n - sum_I - 1))
            # reg2 <- lm(d ~ selected controls)  (drop d column → X[:, union])
            _, v, _, _ = _ols(dv, X[:, idx_union])
        mv2 = float(np.mean(v**2))
        var = (1.0 / n) * (1.0 / mv2) * float(np.mean(v**2 * xi**2)) * (1.0 / mv2)
        se = float(np.sqrt(var))
        sel = idx_union
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
) -> Dict[str, RLassoEffectResult]:
    """Estimate the effect of each targeted column of ``X`` on ``y``.

    Faithful port of ``hdm::rlassoEffects``: for every target column
    ``j`` in ``index``, treat column ``j`` as ``d`` and the remaining
    columns as controls.

    Returns
    -------
    dict
        Mapping ``column name -> RLassoEffectResult``.

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

    out: Dict[str, RLassoEffectResult] = {}
    for j in index:
        d = Xv[:, j]
        Xt = np.delete(Xv, j, axis=1)
        res = rlasso_effect(
            Xt, yv, d, method=method, post=post, penalty=penalty, control=control
        )
        res.target = cols[j]
        out[cols[j]] = res
    return out
