"""Logistic high-dimensional treatment effects (``hdm::rlassologitEffect``).

A faithful port of ``hdm::rlassologitEffect`` (single target) and
``hdm::rlassologitEffects`` (many targets): the *logistic* analogue of
:func:`statspai.rlasso.rlasso_effect`, for a **binary outcome** ``y`` with a
treatment ``d`` and high-dimensional controls ``x``.

The estimator is the Belloni-Chernozhukov-Wei (2016) post-double-selection
procedure for generalized linear models:

1. Rigorous *logistic* Lasso of ``y`` on ``[d, x]`` selects controls ``I1``
   and yields the working weights ``σ²(x) = G(1-G)`` (``G`` the fitted
   probability).
2. Rigorous *linear* Lasso of the weighted treatment on the weighted
   controls selects ``I2`` and yields the orthogonalised score ``z``.
3. A low-dimensional logit of ``y`` on ``[d, controls in I1 ∪ I2]`` gives the
   coefficient on ``d``; inference uses the max of two sandwich variances
   (a score-based form and a model-based form), matching ``hdm`` exactly.

References
----------
Belloni, A., Chernozhukov, V. and Wei, Y. (2016). "Post-Selection Inference
    for Generalized Linear Models with Many Controls." *Journal of Business
    & Economic Statistics*, 34(4), 606-619. [@belloni2016post]

Chernozhukov, V., Hansen, C. and Spindler, M. (2016). "hdm:
    High-Dimensional Metrics." *The R Journal*, 8(2), 185-199.
    [@chernozhukov2016hdm]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ._core import rlasso
from ._logit import rlassologit


@dataclass
class RLassoLogitEffectResult(ResultProtocolMixin):
    """Return of :func:`rlassologit_effect`."""

    #: Verified paper.bib keys (CLAUDE.md §10).
    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "belloni2016post",
        "chernozhukov2016hdm",
    )

    alpha: float
    se: float
    tstat: float
    pvalue: float
    n_obs: int
    n_selected: int
    post: bool
    target: str = "d"

    def conf_int(self, level: float = 0.95) -> tuple:
        zc = stats.norm.ppf(0.5 + level / 2.0)
        return (self.alpha - zc * self.se, self.alpha + zc * self.se)

    def summary(self) -> str:
        lo, hi = self.conf_int()
        return "\n".join(
            [
                "Logistic rigorous-Lasso treatment effect (double selection)",
                "-" * 60,
                f"  Observations         : {self.n_obs}",
                f"  Controls selected    : {self.n_selected}",
                "",
                "             coef     std.err      z      P>|z|      95% CI",
                f"  {self.target:<8}{self.alpha:>10.4f}  {self.se:>9.4f}"
                f"  {self.tstat:>7.3f}  {self.pvalue:>8.4f}  [{lo:.3f}, {hi:.3f}]",
            ]
        )


def _logit_fit(y: np.ndarray, X: np.ndarray) -> tuple:
    """Intercept + ``X`` logistic regression via statsmodels.

    Returns ``(params, fitted_prob)`` with ``params[0]`` the intercept.
    """
    import statsmodels.api as sm

    A = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    model = sm.GLM(y, A, family=sm.families.Binomial())
    res = model.fit()
    return np.asarray(res.params, dtype=float), np.asarray(res.predict(A), dtype=float)


def rlassologit_effect(
    x: Union[np.ndarray, pd.DataFrame, Sequence[str]],
    y: Union[np.ndarray, pd.Series, str],
    d: Union[np.ndarray, pd.Series, str],
    post: bool = True,
    I3: Optional[np.ndarray] = None,
    data: Optional[pd.DataFrame] = None,
) -> RLassoLogitEffectResult:
    """Effect of ``d`` on a binary ``y`` after Lasso-selecting controls ``x``.

    Faithful port of ``hdm::rlassologitEffect``.

    Parameters
    ----------
    x : (n, p) controls (array, DataFrame or column names).
    y : binary outcome.
    d : the single treatment / target regressor.
    post : bool, default True
        Post-Lasso inside the two selection steps.
    I3 : bool array, optional
        Amelioration set forced into the control set (hdm's ``I3``).
    data : DataFrame backing string/column-name inputs.

    Returns
    -------
    RLassoLogitEffectResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((300, 10))
    >>> d = (rng.standard_normal(300) + X[:, 0] > 0).astype(float)
    >>> eta = 0.7 * d + X[:, 1] - 0.5 * X[:, 2]
    >>> y = (rng.uniform(size=300) < 1 / (1 + np.exp(-eta))).astype(float)
    >>> res = sp.rlassologit_effect(X, y, d)
    >>> bool(res.se > 0) and bool(np.isfinite(res.alpha))
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
    n, p = X.shape

    # Step 1: logistic rigorous Lasso of y on [d, x] (explicit hdm penalty).
    la1 = (
        1.1
        / 2.0
        * np.sqrt(n)
        * stats.norm.ppf(1.0 - 0.05 / max(n, (p + 1) * np.log(n)))
    )
    dx = np.column_stack([dv, X])
    l1 = rlassologit(dx, yv, post=post, intercept=True, penalty={"lambda": la1})
    link = np.asarray(l1.predict(dx, type="link"), dtype=float)
    sigma2 = np.exp(link) / (1.0 + np.exp(link)) ** 2
    f = np.sqrt(sigma2)
    I1 = np.asarray(l1.index, dtype=bool)[1:]  # drop the d slot

    # Step 2: weighted linear rigorous Lasso of d on x (explicit hdm penalty).
    la2 = 2.2 * np.sqrt(n) * stats.norm.ppf(1.0 - 0.05 / max(n, p * np.log(n)))
    xf = X * f[:, None]
    df = dv * f
    l2 = rlasso(
        xf,
        df,
        post=post,
        intercept=True,
        penalty={
            "homoscedastic": "none",
            "lambda.start": la2,
            "c": 1.1,
            "gamma": 0.1,
        },
    )
    I2 = np.asarray(l2.index, dtype=bool)
    z = np.asarray(l2.residuals, dtype=float) / np.sqrt(sigma2)

    # Union of the two selected control sets (+ optional amelioration set).
    if I3 is not None:
        sel = I1 | I2 | np.asarray(I3, dtype=bool)
    else:
        sel = I1 | I2
    n_selected = int(sel.sum())

    # Step 3: low-dimensional logit of y on [d, selected controls].
    Xsel = np.column_stack([dv, X[:, sel]]) if n_selected else dv.reshape(-1, 1)
    params, G3 = _logit_fit(yv, Xsel)
    alpha = float(params[1])  # params[0] = intercept, params[1] = d
    w3 = G3 * (1.0 - G3)

    # max-of-two sandwich variance (hdm: l3 has no $index, so S22 uses d only).
    S21 = float(np.mean((yv - G3) ** 2 * z**2)) / float(np.mean(w3 * dv * z)) ** 2
    S22 = n / float(np.sum(w3 * dv**2))
    S2 = max(S21, S22)
    se = float(np.sqrt(S2 / n))

    tval = alpha / se if se > 0 else np.nan
    pval = 2.0 * float(stats.norm.cdf(-abs(tval)))
    return RLassoLogitEffectResult(
        alpha=alpha,
        se=se,
        tstat=float(tval),
        pvalue=pval,
        n_obs=n,
        n_selected=n_selected,
        post=post,
        target=target,
    )


def rlassologit_effects(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    index: Optional[Sequence[int]] = None,
    post: bool = True,
    I3: Optional[np.ndarray] = None,
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, RLassoLogitEffectResult]:
    """Logistic high-dimensional effect of each targeted column of ``X``.

    Faithful port of ``hdm::rlassologitEffects``: for every target column
    ``j`` in ``index``, treat column ``j`` as ``d`` and the remaining
    columns as controls, with a binary outcome ``y``.

    Returns
    -------
    dict
        Mapping ``column name -> RLassoLogitEffectResult``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> X = rng.standard_normal((300, 8))
    >>> eta = 0.8 * X[:, 0] - 0.6 * X[:, 1]
    >>> y = (rng.uniform(size=300) < 1 / (1 + np.exp(-eta))).astype(float)
    >>> out = sp.rlassologit_effects(X, y, index=[0, 1])
    >>> len(out) == 2 and all(r.se > 0 for r in out.values())
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

    out: Dict[str, RLassoLogitEffectResult] = {}
    for j in index:
        d = Xv[:, j]
        Xt = np.delete(Xv, j, axis=1)
        res = rlassologit_effect(Xt, yv, d, post=post, I3=I3)
        res.target = cols[j]
        out[cols[j]] = res
    return out
