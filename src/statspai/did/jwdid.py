"""``sp.jwdid``: Stata ``jwdid`` option names on top of :func:`sp.etwfe`."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import pandas as pd

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility

__all__ = ["jwdid"]

_METHODS = {
    None: None,
    "regress": None,
    "reghdfe": None,
    "ols": None,
    "poisson": "poisson",
    "ppmlhdfe": "poisson",
    "logit": "logit",
}
_PREDICT = {None: "response", "mu": "response", "n": "response", "xb": "link"}


def _as_list(v: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if v is None:
        return None
    if isinstance(v, str):
        return v.split() if " " in v.strip() else [v]
    return list(v)


@accepts_aliases(id="ivar")
def jwdid(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    x: Optional[Union[str, Sequence[str]]] = None,
    method: Optional[str] = None,
    never: bool = False,
    hettype: Optional[str] = None,
    exovar: Optional[Union[str, Sequence[str]]] = None,
    cluster: Optional[str] = None,
    predict: Optional[str] = None,
    alpha: float = 0.05,
    separated: str = "keep",
) -> CausalResult:
    """Wooldridge ETWFE with Stata ``jwdid``'s option names.

    ``sp.jwdid(df, "y", ivar="id", tvar="year", gvar="gvar",
    method="ppmlhdfe", never=True)`` is ``jwdid y, ivar(id) tvar(year)
    gvar(gvar) method(ppmlhdfe) never`` followed by ``estat simple``.  It
    calls :func:`sp.etwfe` with unit fixed effects (``fe='unit'``), the
    design ``jwdid`` fits when ``ivar()`` is given, and returns its result:
    ``sp.etwfe_emfx(result, type='event' | 'group' | 'calendar')`` are the
    other ``estat`` aggregations and ``result.pretrend_test(window=...)``
    is ``estat event, pretrend``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel in long format.
    y : str
        Outcome (``depvar``).
    ivar, tvar, gvar : str
        Unit, time and first-treatment-period columns (``gvar`` = 0 or
        missing for never-treated units).  ``id=`` (StatsPAI's name for the
        panel identifier) is accepted for ``ivar``.
    x : str or list of str, optional
        ``indepvars`` of ``jwdid y x``: covariates that moderate each
        treatment effect.  ``"i.region"`` marks a categorical covariate --
        the per-level ATTs of ``estat simple, over(region)`` are
        ``sp.etwfe_emfx(result, type='simple', by_xvar=True)``.
    method : {None, 'regress', 'ppmlhdfe', 'poisson', 'logit'}
        ``method()``.  ``None`` is ``jwdid``'s default linear model
        (``reghdfe``); ``'ppmlhdfe'`` / ``'poisson'`` the Poisson model.
    never : bool, default False
        ``never``: compare with never-treated units only, which also
        estimates the pre-treatment (lead) effects.
    hettype : str, optional
        ``hettype()``: ``'timecohort'`` (default), ``'time'``, ``'cohort'``,
        ``'event'`` or ``'twfe'``.
    exovar : str or list of str, optional
        ``exovar()``: covariates that enter without interactions.  Stata
        factor terms work, e.g. ``"i.year#i.nodecity"``.
    cluster : str, optional
        ``cluster()``; defaults to ``ivar``.
    predict : {None, 'mu', 'xb'}
        The ``estat ..., predict()`` scale of the reported ATT for a
        nonlinear ``method``: ``None`` / ``'mu'`` the count (response) scale,
        Stata's default; ``'xb'`` the linear index (log points for Poisson).
    alpha : float, default 0.05
        Significance level.
    separated : {'keep', 'drop'}, default 'keep'
        Treatment of units whose outcome is always zero under Poisson; see
        :func:`sp.etwfe`.

    Returns
    -------
    CausalResult
        The :func:`sp.etwfe` result; ``model_info['stata_equivalent']``
        spells out the Stata command it reproduces.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=120, n_periods=8, staggered=True, seed=42)
    >>> r = sp.jwdid(df, "y", ivar="unit", tvar="time", gvar="first_treat")
    >>> r.model_info["stata_equivalent"]
    'jwdid y, ivar(unit) tvar(time) gvar(first_treat)'

    References
    ----------
    Wooldridge (2021), "Two-Way Fixed Effects, the Two-Way Mundlak
    Regression, and Difference-in-Differences Estimators" [@wooldridge2021two].
    """
    m_key = None if method is None else str(method).strip().lower()
    if m_key not in _METHODS:
        raise MethodIncompatibility(
            f"jwdid(method={method!r}) is not supported; use one of "
            "None, 'regress', 'ppmlhdfe', 'poisson', 'logit'.",
            recovery_hint="Pass method='ppmlhdfe' for count outcomes.",
            diagnostics={"method": method},
        )
    family = _METHODS[m_key]
    p_key = None if predict is None else str(predict).strip().lower()
    if p_key not in _PREDICT:
        raise MethodIncompatibility(
            f"jwdid(predict={predict!r}) is not supported; use None, 'mu' or 'xb'.",
            recovery_hint="predict='xb' reports the ATT in log points.",
            diagnostics={"predict": predict},
        )
    if family is None and p_key not in (None, "xb"):
        raise MethodIncompatibility(
            f"jwdid(predict={predict!r}) needs a nonlinear method(); the "
            "linear model has a single scale.",
            recovery_hint="Drop predict=, or pass method='ppmlhdfe'.",
            diagnostics={"predict": predict, "method": method},
        )
    from .wooldridge_did import etwfe

    x_list = _as_list(x)
    exo_list = _as_list(exovar)
    kwargs: dict[str, Any] = dict(
        data=data,
        y=y,
        group=ivar,
        time=tvar,
        first_treat=gvar,
        controls=exo_list,
        cluster=cluster,
        alpha=alpha,
        xvar=x_list,
        cgroup="nevertreated" if never else "notyet",
        family=family,
        fe="unit",
        hettype=hettype,
    )
    if family is not None:
        kwargs.update(scale=_PREDICT[p_key], separated=separated)
    result = etwfe(**kwargs)

    opts = [f"ivar({ivar})", f"tvar({tvar})", f"gvar({gvar})"]
    if m_key not in (None, "regress", "reghdfe", "ols"):
        opts.append(f"method({m_key})")
    if never:
        opts.append("never")
    if hettype is not None:
        opts.append(f"hettype({hettype})")
    if exo_list:
        opts.append(f"exovar({' '.join(exo_list)})")
    if cluster is not None:
        opts.append(f"cluster({cluster})")
    lhs = " ".join([y, *(x_list or [])])
    cmd = f"jwdid {lhs}, {' '.join(opts)}"
    result.model_info["stata_equivalent"] = cmd
    if family is not None:
        result.model_info["stata_estat"] = (
            "estat simple, predict(xb)" if p_key == "xb" else "estat simple"
        )
    return result
