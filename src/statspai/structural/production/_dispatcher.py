"""
Unified ``sp.prod_fn(method=...)`` dispatcher.

Parallels ``sp.synth(method=...)`` and ``sp.decompose(method=...)`` —
one entry point selects between Olley-Pakes, Levinsohn-Petrin,
Ackerberg-Caves-Frazer, and Wooldridge.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from ._result import ProductionResult
from .op_lp_acf import (
    ackerberg_caves_frazer,
    levinsohn_petrin,
    olley_pakes,
)
from .wooldridge import wooldridge_prod


_METHOD_DISPATCH = {
    "op": olley_pakes,
    "olley_pakes": olley_pakes,
    "lp": levinsohn_petrin,
    "levinsohn_petrin": levinsohn_petrin,
    "acf": ackerberg_caves_frazer,
    "ackerberg_caves_frazer": ackerberg_caves_frazer,
    "wrdg": wooldridge_prod,
    "wooldridge": wooldridge_prod,
}


def prod_fn(
    data: pd.DataFrame,
    output: str = "y",
    free: Sequence[str] | str | None = None,
    state: Sequence[str] | str | None = None,
    proxy: str | None = None,
    panel_id: str = "id",
    time: str = "year",
    method: str = "acf",
    polynomial_degree: int = 3,
    productivity_degree: int = 1,
    functional_form: str = "cobb-douglas",
    boot_reps: int = 0,
    seed: Optional[int] = None,
    **kwargs,
) -> ProductionResult:
    """Production function estimation — unified interface.

    Parameters
    ----------
    data : DataFrame
        Long panel with one row per (firm, year).
    output : str
        Log output column.
    free : str or list, default ``["l"]``
        Free inputs (e.g. labor).
    state : str or list, default ``["k"]``
        State / predetermined inputs (capital).
    proxy : str, optional
        Productivity proxy. Defaults: ``"i"`` for ``method="op"``,
        ``"m"`` for all others.
    panel_id, time : str
        Panel identifier columns.
    method : {'op', 'lp', 'acf', 'wrdg'}, default ``'acf'``
        Estimator. ACF is the modern default (corrects OP/LP
        identification problem).
    polynomial_degree : int, default 3
        Stage-1 control function polynomial degree.
    productivity_degree : int, default 1
        Productivity AR polynomial degree.  Default ``1`` (linear AR(1))
        is the most numerically robust choice — higher degrees can
        overfit ``omega_t`` given ``omega_{t-1}`` in finite samples and
        flatten the GMM objective surface, which makes the structural
        parameters numerically un-identified even when they are
        identified in population.
    functional_form : {'cobb-douglas', 'translog'}, default 'cobb-douglas'
        Functional form. Translog adds 0.5 * x_j**2 own-quadratic terms
        and x_j*x_k cross terms — output elasticities then vary by
        firm-time and ``ProductionResult.model_info["elasticities"]``
        carries a per-row DataFrame.

        Translog identification caveat: stage-2 instruments are formed
        as polynomial transforms of the same raw set used for
        Cobb-Douglas (``(k, l_lag)`` for ACF, ``(k, l)`` for OP/LP).
        This is standard in the literature but the resulting moment
        system can be near-singular when state and lagged-free inputs
        are highly correlated, so finite-sample variance on the
        higher-order coefficients (``ll``, ``kk``, ``lk``) is
        substantially larger than on the linear ``l`` and ``k`` terms.
        Use bootstrap SEs (``boot_reps>=200``) to gauge this.

        Wooldridge does not yet support translog (raises
        ``NotImplementedError``); use ``method="acf"`` or
        ``method="lp"`` for translog work.
    boot_reps : int, default 0
        Firm-cluster bootstrap replications. ``0`` ⇒ NaN standard errors.
    seed : int, optional

    Returns
    -------
    ProductionResult
        ``.coef`` for elasticities, ``.tfp`` for log-productivity series,
        ``.summary()`` for a Stata-style table.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.prod_fn(df, output="y", free="l", state="k", proxy="m",
    ...                   panel_id="id", time="year",
    ...                   method="acf", boot_reps=200, seed=0)
    >>> res.coef
    {"l": 0.62, "k": 0.32}
    >>> mu = sp.markup(res, revenue="log_rev", input_cost="log_mat",
    ...                 flexible_input="m")

    See Also
    --------
    olley_pakes, levinsohn_petrin, ackerberg_caves_frazer, wooldridge_prod
    markup : De Loecker-Warzynski (2012) firm-time markup.

    References
    ----------
    Olley & Pakes (1996); Levinsohn & Petrin (2003);
    Ackerberg, Caves & Frazer (2015); Wooldridge (2009).
    """
    key = method.lower()
    if key not in _METHOD_DISPATCH:
        raise ValueError(
            f"Unknown method {method!r}. Choose from "
            f"{sorted(set(_METHOD_DISPATCH))}."
        )
    if proxy is None:
        proxy = "i" if key in ("op", "olley_pakes") else "m"
    fn = _METHOD_DISPATCH[key]
    return fn(
        data=data,
        output=output,
        free=free,
        state=state,
        proxy=proxy,
        panel_id=panel_id,
        time=time,
        polynomial_degree=polynomial_degree,
        productivity_degree=productivity_degree,
        functional_form=functional_form,
        boot_reps=boot_reps,
        seed=seed,
        **kwargs,
    )
