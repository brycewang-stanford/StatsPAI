"""Top-level aliases matching the public-facing article API.

The StatsPAI README and blog posts advertise a short, Stata-like surface
(`sp.rdd`, `sp.frontdoor`, `sp.xlearner`, ...).  Several of these names
are *thin wrappers* over richer implementations that already live in the
submodules — for example ``sp.rdd`` is shorthand for
``sp.rdrobust`` with the running variable named ``x``.

Keeping the aliases in one place (instead of sprinkling ``xxx = yyy``
across ``__init__.py``) makes it easy to:

* verify the article's documented surface with a single audit pass
* change a wrapper's defaults without editing the package root
* write targeted tests that pin the alias → implementation mapping

Every wrapper here delegates to an *existing* implementation and adds
no numerical code of its own. If you change behaviour, change the
underlying module — not this file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .core.results import CausalResult
from .exceptions import MethodIncompatibility

__all__ = [
    "rdd",
    "frontdoor",
    "xlearner",
    "conformal_ite",
    "psm",
    "partial_identification",
    "anderson_rubin_ci",
    "conditional_lr_ci",
    "tF_adjustment",
    # Round-2 additions: namespace-collision fixes + kwarg alignment
    "matrix_completion",
    "causal_discovery",
    "mediation",
    "evalue_rr",
    "policy_tree",
    "dml",
]


def _require_string_option(value: Any, name: str, function: str) -> str:
    if not isinstance(value, str):
        raise MethodIncompatibility(
            f"{function}: `{name}` must be a string option.",
            diagnostics={"function": function, name: repr(value)},
        )
    out = value.lower().strip()
    if not out:
        raise MethodIncompatibility(
            f"{function}: `{name}` must be a non-empty string option.",
            diagnostics={"function": function, name: repr(value)},
        )
    return out


def _require_column_name(value: Any, name: str, function: str) -> str:
    if not isinstance(value, str) or not value:
        raise MethodIncompatibility(
            f"{function}: `{name}` must be a non-empty column name.",
            diagnostics={"function": function, name: repr(value)},
        )
    return value


def _coerce_column_list(
    value: Any,
    name: str,
    function: str,
    *,
    allow_none: bool = False,
    allow_empty: bool = False,
) -> Optional[List[str]]:
    if value is None:
        if allow_none:
            return None
        raise MethodIncompatibility(
            f"{function}: `{name}` must be a column name or list of column names.",
            diagnostics={"function": function, name: None},
        )
    if isinstance(value, str):
        out = [value]
    else:
        try:
            out = list(value)
        except TypeError as exc:
            raise MethodIncompatibility(
                f"{function}: `{name}` must be a column name or list of column names.",
                diagnostics={"function": function, name: repr(value)},
            ) from exc
    if not allow_empty and not out:
        raise MethodIncompatibility(
            f"{function}: `{name}` must contain at least one column name.",
            diagnostics={"function": function, name: out},
        )
    bad = [col for col in out if not isinstance(col, str) or not col]
    if bad:
        raise MethodIncompatibility(
            f"{function}: `{name}` must contain only non-empty string column names.",
            diagnostics={"function": function, name: out, "invalid_columns": bad},
        )
    return out


# ---------------------------------------------------------------------------
# Regression discontinuity
# ---------------------------------------------------------------------------


def rdd(
    data: pd.DataFrame,
    y: str,
    running: str,
    cutoff: float = 0.0,
    *,
    fuzzy: Optional[str] = None,
    **kwargs: Any,
) -> CausalResult:
    """Sharp / fuzzy RD — article-friendly alias for :func:`rdrobust`.

    Parameters match the blog post signature ``sp.rdd(df, y, running, cutoff)``
    and are forwarded to :func:`statspai.rd.rdrobust` using its
    ``(x=<running>, c=<cutoff>)`` convention.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> x = rng.uniform(-1, 1, size=n)
    >>> y = 0.5 * x + 0.8 * (x >= 0) + rng.normal(scale=0.3, size=n)
    >>> df = pd.DataFrame({'y': y, 'x': x})
    >>> res = sp.rdd(df, y='y', running='x', cutoff=0.0)
    >>> round(res.estimate, 2)  # robust RD estimate (true jump 0.8)
    0.81
    """
    from .rd.rdrobust import rdrobust

    return rdrobust(
        data=data,
        y=y,
        x=running,
        c=cutoff,
        fuzzy=fuzzy,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Pearl's front-door criterion
# ---------------------------------------------------------------------------


def frontdoor(
    data: pd.DataFrame,
    y: str,
    d: str,
    m: str,
    X: Optional[List[str]] = None,
    **kwargs: Any,
) -> CausalResult:
    """Front-door adjustment — article-friendly alias for
    :func:`statspai.inference.front_door`.

    ``X`` is mapped to the underlying ``covariates`` argument.

    References
    ----------
    [@pearl1995causal]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> m = 0.7 * d + rng.normal(size=n)   # mediator driven by treatment
    >>> y = 1.0 * m + rng.normal(size=n)   # outcome driven by mediator
    >>> df = pd.DataFrame({'y': y, 'd': d, 'm': m})
    >>> res = sp.frontdoor(df, y='y', d='d', m='m', n_boot=100, seed=0)
    >>> round(res.estimate, 2)  # front-door effect of d on y (true ~0.7)
    0.66
    """
    from .inference.front_door import front_door as _front_door

    return _front_door(
        data=data,
        y=y,
        treat=d,
        mediator=m,
        covariates=X,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Meta-learner shortcuts
# ---------------------------------------------------------------------------


def xlearner(
    data: pd.DataFrame,
    y: str,
    d: str,
    X: List[str],
    **kwargs: Any,
) -> CausalResult:
    """X-Learner CATE — article alias for :func:`metalearner(learner='x')`.

    Kept separate from the generic :func:`metalearner` entry point because
    the blog post advertises ``sp.xlearner(df, y, d, X)`` directly.

    Passing ``learner=...`` is rejected — callers who want a different
    meta-learner should use :func:`sp.metalearner` instead of silently
    getting an X-Learner under a misleading name.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 800
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> y = 1.0 * d + 0.5 * d * x1 + x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> result = sp.xlearner(df, y='y', d='d', X=['x1', 'x2'])
    >>> round(result.estimate, 2)  # ATE (true value 1.0)
    0.84
    >>> result.model_info['cate'].shape  # individual CATE predictions
    (800,)
    """
    if "learner" in kwargs:
        raise MethodIncompatibility(
            "sp.xlearner is fixed to learner='x'. Use sp.metalearner(..., "
            f"learner={kwargs['learner']!r}) for a different meta-learner.",
            recovery_hint=(
                "Call sp.metalearner directly when selecting a non-X learner."
            ),
            diagnostics={"function": "xlearner", "learner": kwargs["learner"]},
            alternative_functions=["sp.metalearner"],
        )
    covariates = _coerce_column_list(X, "X", "xlearner")
    assert covariates is not None

    from .metalearners.metalearners import metalearner

    return metalearner(
        data=data,
        y=y,
        treat=d,
        covariates=covariates,
        learner="x",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Conformal ITE intervals
# ---------------------------------------------------------------------------


def conformal_ite(
    data: pd.DataFrame,
    y: str,
    d: str,
    X: List[str],
    **kwargs: Any,
) -> CausalResult:
    """Conformal ITE — article alias for :func:`conformal_cate`.

    Covers the ``sp.conformal_ite(df, y, d, X)`` shape advertised in the
    2026-04-20 blog post.  Delegates to
    :func:`statspai.conformal_causal.conformal_cate`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> y = 1.0 * d + 0.5 * x1 + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> res = sp.conformal_ite(df, y='y', d='d', X=['x1', 'x2'])
    >>> res.model_info['cate'].shape  # one interval per individual
    (400,)
    >>> res.model_info['cate_lower'].shape == res.model_info['cate_upper'].shape
    True
    >>> res.model_info['coverage_level']  # default 95% conformal coverage
    0.95
    """
    from .conformal_causal.conformal_ite import conformal_cate

    return conformal_cate(
        data=data,
        y=y,
        treat=d,
        covariates=X,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Propensity-score matching
# ---------------------------------------------------------------------------


def psm(
    data: pd.DataFrame,
    y: str,
    d: str,
    X: List[str],
    *,
    method: str = "nn",
    **kwargs: Any,
) -> CausalResult:
    """Propensity-score matching — article alias for :func:`match`
    with ``distance='propensity'``.

    ``method='nn'`` (the common Stata/R shorthand) is translated into the
    richer ``method='nearest'`` API of :func:`statspai.matching.match`.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> result = sp.psm(df, y='log_wage', d='union',
    ...                 X=['education', 'experience', 'tenure'])
    >>> result.summary()
    >>> result.estimate

    >>> # Propensity-score stratification instead of nearest-neighbour
    >>> result = sp.psm(df, y='log_wage', d='union',
    ...                 X=['education', 'experience', 'tenure'],
    ...                 method='stratify')
    """
    from .matching.match import match as _match

    # Map the common PSM aliases to the underlying match() API.
    alias_map = {
        "nn": "nearest",
        "nearest": "nearest",
        "psm": "nearest",
        "stratify": "stratify",
        "cem": "cem",
    }
    internal_method = alias_map.get(method, method)

    return _match(
        data=data,
        y=y,
        treat=d,
        covariates=X,
        method=internal_method,
        distance=kwargs.pop("distance", "propensity"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Partial identification / bounds
# ---------------------------------------------------------------------------


def partial_identification(
    data: pd.DataFrame,
    y: str,
    d: str,
    X: Optional[List[str]] = None,
    *,
    method: str = "manski",
    selection: Optional[str] = None,
    instrument: Optional[str] = None,
    assumptions: Optional[List[str]] = None,  # noqa: ARG001 — reserved
    **kwargs: Any,
) -> Any:
    """Partial identification of ATE — article alias for the ``bounds`` module.

    ``method='manski'``          → :func:`manski_bounds`   (worst-case bounds)
    ``method='lee'``              → :func:`lee_bounds`     (monotone-selection
                                                            bounds; requires
                                                            ``selection=``)
    ``method='horowitz_manski'``  → :func:`horowitz_manski` (requires
                                                            covariates via ``X``)
    ``method='iv'``               → :func:`iv_bounds`      (requires
                                                            ``instrument=``)

    The underlying bounds functions use slightly different parameter names
    (``treat`` vs ``treatment``, ``covariates`` vs ``controls``).  This
    wrapper normalises the public-facing ``(y, d, X)`` surface and routes to
    each backend with its native kwargs.

    The ``assumptions`` keyword is accepted for forward compatibility but
    ignored by all current back-ends; see each underlying function for its
    native assumption interface.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> y = 1.0 + 0.5 * d + rng.normal(0, 1.0, size=n)
    >>> y = np.clip(y, 0.0, 1.0)  # Manski bounds need a bounded outcome
    >>> df = pd.DataFrame({"y": y, "d": d})

    Worst-case (no-assumption) Manski bounds on the ATE:

    >>> res = sp.partial_identification(df, "y", "d", method="manski")
    >>> round(float(res.estimate), 2)
    0.06
    >>> round(float(res.model_info["lower_bound"]), 2)
    -0.44
    >>> round(float(res.model_info["upper_bound"]), 2)
    0.56
    """
    from . import bounds as _bounds

    method = _require_string_option(method, "method", "partial_identification")
    X_final = _coerce_column_list(
        X,
        "X",
        "partial_identification",
        allow_none=True,
        allow_empty=True,
    )

    if method == "manski":
        # manski_bounds uses `treat`; no covariates supported — warn if given.
        if X_final:
            raise MethodIncompatibility(
                "partial_identification(method='manski') does not use "
                "covariates (pure worst-case bounds). Drop X or use "
                "method='horowitz_manski' for a covariate-aware variant.",
                recovery_hint=(
                    "Drop X for Manski bounds or use method='horowitz_manski'."
                ),
                diagnostics={
                    "function": "partial_identification",
                    "method": method,
                    "covariates": X_final,
                },
                alternative_functions=["sp.partial_identification"],
            )
        return _bounds.manski_bounds(data=data, y=y, treat=d, **kwargs)

    if method == "lee":
        # lee_bounds uses `treat` and REQUIRES `selection`.
        if selection is None:
            raise MethodIncompatibility(
                "partial_identification(method='lee') requires "
                "`selection=<column name>` - Lee (2009) bounds are for "
                "sample-selection problems where a binary observability "
                "indicator is needed.",
                recovery_hint=(
                    "Pass the binary observability indicator as selection=..."
                ),
                diagnostics={
                    "function": "partial_identification",
                    "method": method,
                },
            )
        selection_final = _require_column_name(
            selection,
            "selection",
            "partial_identification",
        )
        return _bounds.lee_bounds(
            data=data,
            y=y,
            treat=d,
            selection=selection_final,
            covariates=X_final,
            **kwargs,
        )

    if method in {"horowitz_manski", "horowitz-manski", "hm"}:
        # horowitz_manski uses `treatment` (not `treat`) and REQUIRES
        # `covariates` (cannot be None).
        if not X_final:
            raise MethodIncompatibility(
                "partial_identification(method='horowitz_manski') requires "
                "a non-empty list of covariates via `X=[...]` - the "
                "Horowitz-Manski bounds condition on X.",
                recovery_hint=(
                    "Pass covariates through X=[...] for Horowitz-Manski bounds."
                ),
                diagnostics={
                    "function": "partial_identification",
                    "method": method,
                },
            )
        return _bounds.horowitz_manski(
            data=data,
            y=y,
            treatment=d,
            covariates=X_final,
            **kwargs,
        )

    if method == "iv":
        # iv_bounds uses `treatment`, `instrument`, and `controls` (not
        # `covariates`).  `X` maps to `controls` here.
        if instrument is None:
            raise MethodIncompatibility(
                "partial_identification(method='iv') requires "
                "`instrument=<column name>` for the IV bounds (Manski-Pepper).",
                recovery_hint="Pass the instrument column as instrument=...",
                diagnostics={"function": "partial_identification", "method": method},
            )
        instrument_final = _require_column_name(
            instrument,
            "instrument",
            "partial_identification",
        )
        return _bounds.iv_bounds(
            data=data,
            y=y,
            treatment=d,
            instrument=instrument_final,
            controls=X_final,
            **kwargs,
        )

    raise MethodIncompatibility(
        f"Unknown partial_identification method '{method}'. "
        "Expected one of: 'manski', 'lee', 'horowitz_manski', 'iv'.",
        recovery_hint="Choose one of: manski, lee, horowitz_manski, iv.",
        diagnostics={
            "function": "partial_identification",
            "method": method,
            "valid_methods": ["manski", "lee", "horowitz_manski", "iv"],
        },
    )


# ---------------------------------------------------------------------------
# Weak-IV robust confidence sets (top-level re-exports)
# ---------------------------------------------------------------------------


def anderson_rubin_ci(*args: Any, **kwargs: Any) -> Any:
    """Anderson-Rubin confidence set — re-export of
    :func:`statspai.iv.weak_iv_ci.anderson_rubin_ci`.

    The AR test remains exact under any level of weak identification, so
    the corresponding confidence set is the canonical weak-IV-robust CI.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> z = rng.normal(size=n)
    >>> u = rng.normal(size=n)
    >>> d = 0.8 * z + u + rng.normal(size=n)
    >>> y = 1.5 * d + u + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'z': z})
    >>> ci = sp.anderson_rubin_ci('y', 'd', ['z'], data=df)
    >>> round(ci.lower, 2), round(ci.upper, 2)
    (1.26, 1.57)
    >>> len(ci.as_intervals())  # connected: a single interval
    1
    """
    from .iv.weak_iv_ci import anderson_rubin_ci as _impl

    return _impl(*args, **kwargs)


def conditional_lr_ci(*args: Any, **kwargs: Any) -> Any:
    """Moreira (2003) CLR confidence set — re-export of
    :func:`statspai.iv.weak_iv_ci.conditional_lr_ci`.

    The conditional likelihood-ratio test is similar (correctly sized) under
    weak identification, so its inverted confidence set is a weak-IV-robust
    CI that, unlike Anderson-Rubin, regains efficiency under strong instruments.

    References
    ----------
    [@moreira2003conditional]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> z = rng.normal(size=n)
    >>> u = rng.normal(size=n)
    >>> d = 0.8 * z + u + rng.normal(size=n)
    >>> y = 1.5 * d + u + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'z': z})
    >>> ci = sp.conditional_lr_ci('y', 'd', ['z'], data=df, random_state=0)
    >>> round(ci.lower, 2), round(ci.upper, 2)
    (1.26, 1.57)
    >>> len(ci.as_intervals())  # connected: a single interval
    1
    """
    from .iv.weak_iv_ci import conditional_lr_ci as _impl

    return _impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# Lee-McCrary-Moreira-Porter (2022) tF adjustment
# ---------------------------------------------------------------------------


def tF_adjustment(first_stage_F: float, alpha: float = 0.05) -> float:
    """tF adjusted critical value (Lee, McCrary, Moreira & Porter 2022, AER).

    Alias for :func:`statspai.diagnostics.weak_iv.tF_critical_value`.
    Named after the ``tF`` terminology used in the paper and blog post so
    that ``sp.tF_adjustment(F)`` works as advertised.

    References
    ----------
    [@lee2022valid]

    Examples
    --------
    >>> import statspai as sp
    >>> sp.tF_adjustment(15.0)  # weakish first stage -> inflated crit. value
    2.54
    >>> sp.tF_adjustment(100.0)  # strong first stage -> back to ~1.96
    1.96
    """
    from .diagnostics.weak_iv import tF_critical_value

    return tF_critical_value(first_stage_F, alpha=alpha)


# ---------------------------------------------------------------------------
# Namespace-collision fixes: article advertises sp.matrix_completion /
# sp.causal_discovery / sp.mediation as functions, but those names are
# already bound to the submodules of the same name by the earlier
# ``from .mediation import mediate`` style imports.  These wrappers must
# be re-exported at the end of __init__.py so that the function binding
# wins over the submodule binding (same pattern the package already uses
# for ``sp.iv``).
# ---------------------------------------------------------------------------


def matrix_completion(
    data: pd.DataFrame,
    y: str,
    d: str,
    unit: str,
    time: str,
    **kwargs: Any,
) -> CausalResult:
    """Matrix-completion causal panel estimator (Athey et al., 2021).

    Article-facing alias for :func:`statspai.matrix_completion.mc_panel`,
    renaming ``d → treat`` to match the blog-post convention.

    References
    ----------
    [@athey2021matrix]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for u in range(15):
    ...     ueff = rng.normal()
    ...     for t in range(10):
    ...         treated = 1 if (u >= 12 and t >= 6) else 0
    ...         y = ueff + 0.2 * t + 2.0 * treated + rng.normal(scale=0.3)
    ...         rows.append({'unit': u, 'time': t, 'y': y, 'd': treated})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.matrix_completion(df, y='y', d='d', unit='unit',
    ...                            time='time', n_bootstrap=50)
    >>> round(res.estimate, 2)  # average treatment effect on the treated
    2.86
    """
    # Use importlib rather than `from .matrix_completion import mc_panel`
    # because this function itself is late-bound as `sp.matrix_completion`,
    # which shadows the submodule attribute on the package.
    import importlib

    _mc = importlib.import_module("statspai.matrix_completion")
    out: CausalResult = _mc.mc_panel(
        data=data,
        y=y,
        unit=unit,
        time=time,
        treat=d,
        **kwargs,
    )
    return out


def causal_discovery(
    data: pd.DataFrame,
    method: str = "notears",
    variables: Optional[List[str]] = None,
    **kwargs: Any,
) -> Any:
    """Causal-discovery dispatcher — article-facing alias.

    ``method='notears'`` → :func:`statspai.causal_discovery.notears`
    ``method='pc'``       → :func:`statspai.causal_discovery.pc_algorithm`
    ``method='ges'``      → :func:`statspai.causal_discovery.ges`
    ``method='lingam'``   → :func:`statspai.causal_discovery.lingam`

    The four backends have slightly different signatures — notably,
    ``ges`` and ``lingam`` do not accept a ``variables`` kwarg — so this
    dispatcher subsets the DataFrame up front rather than forwarding
    ``variables=`` to every backend.

    References
    ----------
    [@zheng2018dags]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x1 = rng.normal(size=n)
    >>> x2 = 1.5 * x1 + rng.normal(scale=0.5, size=n)   # x1 -> x2
    >>> x3 = 0.8 * x2 + rng.normal(scale=0.5, size=n)   # x2 -> x3
    >>> df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
    >>> res = sp.causal_discovery(df, method='notears')
    >>> res['variables']
    ['x1', 'x2', 'x3']
    >>> res['n_edges']  # recovers the x1 -> x2 -> x3 chain
    2
    """
    # Same late-bind shadowing trick — use importlib to reach the
    # subpackage explicitly rather than the now-shadowed attribute.
    import importlib

    _cd = importlib.import_module("statspai.causal_discovery")

    method = _require_string_option(method, "method", "causal_discovery")
    valid = {"notears", "pc", "ges", "lingam"}
    if method not in valid:
        raise MethodIncompatibility(
            f"Unknown causal_discovery method {method!r}. "
            f"Expected one of: {sorted(valid)}.",
            recovery_hint="Choose one of: notears, pc, ges, lingam.",
            diagnostics={
                "function": "causal_discovery",
                "method": method,
                "valid_methods": sorted(valid),
            },
        )

    # Normalise the data at the dispatcher level so the per-backend
    # kwargs stay clean.  Only notears / pc support a `variables=`
    # kwarg natively; ges / lingam just take the whole frame.
    if variables is not None:
        data = data[_coerce_column_list(variables, "variables", "causal_discovery")]

    if method == "notears":
        return _cd.notears(data=data, **kwargs)
    if method == "pc":
        return _cd.pc_algorithm(data=data, **kwargs)
    if method == "ges":
        return _cd.ges(data=data, **kwargs)
    # method == "lingam"
    return _cd.lingam(data=data, **kwargs)


def mediation(
    data: pd.DataFrame,
    y: str,
    d: str,
    m: str,
    X: Optional[List[str]] = None,
    **kwargs: Any,
) -> CausalResult:
    """Causal-mediation analysis — article-facing alias for
    :func:`statspai.mediation.mediate`.

    Translates the blog-post ``(y, d, m, X)`` surface to the underlying
    ``(y, treat, mediator, covariates)`` kwargs.

    References
    ----------
    [@imai2010general]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> m = 0.6 * d + rng.normal(size=n)             # treatment -> mediator
    >>> y = 0.8 * m + 0.4 * d + rng.normal(size=n)   # mediator + direct path
    >>> df = pd.DataFrame({'y': y, 'd': d, 'm': m})
    >>> res = sp.mediation(df, y='y', d='d', m='m', n_boot=100, seed=0)
    >>> round(res.estimate, 2)  # ACME (indirect effect via m)
    0.45
    >>> round(res.model_info['total_effect'], 2)
    0.8
    """
    import importlib

    _med = importlib.import_module("statspai.mediation")
    out: CausalResult = _med.mediate(
        data=data,
        y=y,
        treat=d,
        mediator=m,
        covariates=X,
        **kwargs,
    )
    return out


# ---------------------------------------------------------------------------
# kwarg alignment wrappers
# ---------------------------------------------------------------------------


def evalue_rr(
    rr: float,
    rr_lower: Optional[float] = None,
    rr_upper: Optional[float] = None,
    rare_outcome: bool = False,
) -> Dict[str, Any]:
    """E-value computed directly from a risk ratio and its CI bounds.

    The blog post advertises ``sp.evalue(rr, rr_lower)`` which doesn't
    match :func:`statspai.diagnostics.evalue` (that one takes
    ``estimate, se, ci, measure='RR'``).  This is the small convenience
    shim for the risk-ratio case the article actually documents.

    Parameters
    ----------
    rr
        Point-estimate risk ratio.
    rr_lower, rr_upper
        Optional confidence-interval bounds on the risk ratio scale.
    rare_outcome
        Passed through to :func:`evalue` for rare-outcome OR→RR correction.

    Returns
    -------
    dict
        Same dict shape as :func:`statspai.diagnostics.evalue`.

    References
    ----------
    [@vanderweele2017sensitivity]

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.evalue_rr(2.0, rr_lower=1.5, rr_upper=2.7)
    >>> round(res['evalue_estimate'], 2)  # 2 + sqrt(2)
    3.41
    >>> round(res['evalue_ci'], 2)  # E-value for the CI bound nearest the null
    2.37

    Note that ``rr_lower`` and ``rr_upper`` must be passed together (both
    or neither); supplying only one raises ``ValueError``.
    """
    from .diagnostics import evalue as _evalue

    ci: Optional[Tuple[float, float]] = None
    if rr_lower is not None and rr_upper is not None:
        ci = (float(rr_lower), float(rr_upper))
    elif rr_lower is not None or rr_upper is not None:
        raise MethodIncompatibility(
            "evalue_rr: provide BOTH rr_lower and rr_upper, or neither.",
            recovery_hint="Pass both confidence bounds or omit both.",
            diagnostics={
                "function": "evalue_rr",
                "rr_lower": rr_lower,
                "rr_upper": rr_upper,
            },
        )

    return _evalue(
        estimate=float(rr),
        ci=ci,
        measure="RR",
        rare_outcome=rare_outcome,
    )


def policy_tree(
    data: pd.DataFrame,
    y: str,
    d: Optional[str] = None,
    X: Optional[List[str]] = None,
    *,
    treat: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    depth: Optional[int] = None,
    max_depth: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Doubly-robust policy-tree — article-facing alias.

    Accepts **both** naming conventions so existing call sites keep
    working:

    * blog-post form — ``sp.policy_tree(df, y, d, X, depth=3)``
    * library form   — ``sp.policy_tree(df, y, treat=..., covariates=...,
                                         max_depth=3)``

    Passing conflicting names raises ``MethodIncompatibility``.  Delegates to
    :func:`statspai.policy_learning.policy_tree`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> x3 = rng.normal(size=n)
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> tau = 2.0 * (x1 > 0)  # only x1 > 0 benefits from treatment
    >>> y = 1.0 + tau * d + x2 + rng.normal(0, 0.5, size=n)
    >>> df = pd.DataFrame(
    ...     {"x1": x1, "x2": x2, "x3": x3, "d": d, "y": y}
    ... )

    Article form — positional treatment ``d`` and covariates ``X``:

    >>> res = sp.policy_tree(df, "y", "d", ["x1", "x2", "x3"], depth=2)
    >>> res["n_obs"]
    400
    >>> round(float(res["fraction_treated"]), 2)
    0.66

    ``res["rules"]`` holds the human-readable tree; the leaf splitting on
    ``x1`` near 0 recovers the true heterogeneity boundary.
    """
    # Resolve treat / d — refuse silent loss when both given with
    # different values (reviewer flagged the old "treat wins, d ignored"
    # behaviour as a silent-wrong-pick foot-gun).
    if d is not None and treat is not None and d != treat:
        raise MethodIncompatibility(
            f"policy_tree: conflicting treatment columns "
            f"d={d!r} vs treat={treat!r}. Pass only one.",
            recovery_hint="Pass only one of d or treat.",
            diagnostics={"function": "policy_tree", "d": d, "treat": treat},
        )
    treat_final = treat if treat is not None else d
    if treat_final is None:
        raise MethodIncompatibility(
            "policy_tree() missing required argument: pass either 'd' "
            "(article form) or 'treat=' (library form).",
            recovery_hint="Provide the treatment column as d or treat.",
            diagnostics={"function": "policy_tree"},
        )
    treat_final = _require_column_name(treat_final, "treat", "policy_tree")

    # Resolve X / covariates
    X_final = _coerce_column_list(
        X,
        "X",
        "policy_tree",
        allow_none=True,
    )
    covariates_final = _coerce_column_list(
        covariates,
        "covariates",
        "policy_tree",
        allow_none=True,
    )
    if (
        X_final is not None
        and covariates_final is not None
        and X_final != covariates_final
    ):
        raise MethodIncompatibility(
            "policy_tree: conflicting covariate lists — `X` and "
            "`covariates` must agree if both are given.",
            recovery_hint=("Pass covariates through either X or covariates, not both."),
            diagnostics={
                "function": "policy_tree",
                "X": X_final,
                "covariates": covariates_final,
            },
        )
    cov_final = covariates_final if covariates_final is not None else X_final
    if cov_final is None:
        raise MethodIncompatibility(
            "policy_tree() missing required argument: pass either 'X' "
            "(article form) or 'covariates=' (library form).",
            recovery_hint="Provide covariates as X=[...] or covariates=[...].",
            diagnostics={"function": "policy_tree"},
        )

    # Resolve depth / max_depth
    if depth is not None and max_depth is not None and depth != max_depth:
        raise MethodIncompatibility(
            "policy_tree: pass either `depth` or `max_depth`, not both.",
            recovery_hint=(
                "Use depth for the article API or max_depth for the library API."
            ),
            diagnostics={
                "function": "policy_tree",
                "depth": depth,
                "max_depth": max_depth,
            },
        )
    md = depth if depth is not None else max_depth
    if md is None:
        md = 2  # matches underlying default

    from .policy_learning import policy_tree as _pt

    return _pt(
        data=data,
        y=y,
        treat=treat_final,
        covariates=cov_final,
        max_depth=md,
        **kwargs,
    )


def dml(
    data: pd.DataFrame,
    y: str,
    d: Optional[str] = None,
    X: Optional[List[str]] = None,
    *,
    treat: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    model_y: Any = None,
    model_d: Any = None,
    model: str = "plr",
    **kwargs: Any,
) -> CausalResult:
    """Double/Debiased Machine Learning — article-facing alias.

    Accepts **both** naming conventions used across the StatsPAI surface:

    * the blog-post / article form — ``dml(df, 'y', 'd', ['x1', 'x2'])``
      with positional ``d`` (treatment) and ``X`` (covariates), plus
      ``model_y=`` / ``model_d=`` nuisance learners;
    * the underlying library form — ``dml(df, y='y', treat='d',
      covariates=['x1', 'x2'])`` keyword-only, plus ``ml_g`` / ``ml_m``.

    Both routes resolve to :func:`statspai.dml.dml`. ``model_y``
    forwards to ``ml_g`` (outcome nuisance), ``model_d`` to ``ml_m``
    (treatment / propensity nuisance). ``model=`` controls the DML
    variant: ``'plr'``, ``'irm'``, ``'pliv'``, ``'iivm'``.

    Notes
    -----
    All four models are numerically pinned against ``doubleml-for-py``
    (``plr`` / ``pliv`` to machine precision under shared learners and
    folds). Declared scope boundaries — single scalar instrument for
    ``pliv`` / ``iivm`` (use ``sp.scalar_iv_projection`` for multiple
    instruments), one treatment per call, DML2 procedure only,
    ``fold_indices`` for ``'plr'`` only — are detailed in the
    :func:`statspai.dml.dml` docstring and the guide *"sp.dml and the
    DoubleML reference implementation"*.

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
    Newey, W. and Robins, J. (2018). Double/debiased machine learning for
    treatment and structural parameters. *The Econometrics Journal*.
    [@chernozhukov2018double]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> X = rng.normal(size=(n, 3))
    >>> d = 0.5 * X[:, 0] + rng.normal(size=n)
    >>> y = 2.0 * d + X[:, 1] + rng.normal(size=n)
    >>> df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    >>> df['d'], df['y'] = d, y

    Article form — positional treatment and covariates:

    >>> result = sp.dml(df, 'y', 'd', ['x1', 'x2', 'x3'])
    >>> round(result.estimate, 2)  # true effect is 2.0
    1.93

    Library form — keyword ``treat=`` / ``covariates=``, explicit
    ``model=`` variant and nuisance learners:

    >>> result = sp.dml(df, y='y', treat='d',
    ...                 covariates=['x1', 'x2', 'x3'],
    ...                 model='plr', ml_g='lasso', ml_m='lasso')
    """
    from .dml import dml as _dml

    # Resolve treatment / covariates from either naming convention.
    # Refuse silent loss when both are given with conflicting values —
    # matches the same safety rule added in `policy_tree`.
    if d is not None and treat is not None and d != treat:
        raise MethodIncompatibility(
            f"dml: conflicting treatment columns d={d!r} vs "
            f"treat={treat!r}. Pass only one.",
            recovery_hint="Pass only one of d or treat.",
            diagnostics={"function": "dml", "d": d, "treat": treat},
        )
    treat_final = treat if treat is not None else d
    if treat_final is not None:
        treat_final = _require_column_name(treat_final, "treat", "dml")
    X_final = _coerce_column_list(X, "X", "dml", allow_none=True)
    covariates_final = _coerce_column_list(
        covariates,
        "covariates",
        "dml",
        allow_none=True,
    )
    if (
        X_final is not None
        and covariates_final is not None
        and X_final != covariates_final
    ):
        raise MethodIncompatibility(
            "dml: conflicting covariate lists — `X` and `covariates` "
            "must agree if both are given.",
            recovery_hint=("Pass covariates through either X or covariates, not both."),
            diagnostics={
                "function": "dml",
                "X": X_final,
                "covariates": covariates_final,
            },
        )
    cov_final = covariates_final if covariates_final is not None else X_final
    if treat_final is None:
        raise MethodIncompatibility(
            "dml() missing required argument: pass either 'd' (positional "
            "article form) or 'treat=' (library form).",
            recovery_hint="Provide the treatment column as d or treat.",
            diagnostics={"function": "dml"},
        )
    if cov_final is None:
        raise MethodIncompatibility(
            "dml() missing required argument: pass either 'X' (positional "
            "article form) or 'covariates=' (library form).",
            recovery_hint="Provide covariates as X=[...] or covariates=[...].",
            diagnostics={"function": "dml"},
        )

    # Only forward model_y/model_d if set — otherwise let the underlying
    # function pick its defaults.
    if model_y is not None:
        kwargs.setdefault("ml_g", model_y)
    if model_d is not None:
        kwargs.setdefault("ml_m", model_d)

    return _dml(
        data=data,
        y=y,
        treat=treat_final,
        covariates=cov_final,
        model=model,
        **kwargs,
    )
