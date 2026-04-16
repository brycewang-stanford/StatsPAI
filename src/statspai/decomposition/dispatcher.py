"""
Unified dispatcher: ``sp.decompose(method=...)``

Single entry point for all decomposition methods in StatsPAI.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import pandas as pd


# Lazy import registry: name -> (module_path, function_name)
_REGISTRY: Dict[str, tuple] = {
    "oaxaca": ("statspai.decomposition.oaxaca", "oaxaca"),
    "blinder_oaxaca": ("statspai.decomposition.oaxaca", "oaxaca"),
    "gelbach": ("statspai.decomposition.oaxaca", "gelbach"),

    "rif": ("statspai.decomposition.rif", "rif_decomposition"),
    "rif_decomposition": ("statspai.decomposition.rif", "rif_decomposition"),

    "dfl": ("statspai.decomposition.dfl", "dfl_decompose"),
    "dinardo_fortin_lemieux": ("statspai.decomposition.dfl", "dfl_decompose"),

    "ffl": ("statspai.decomposition.ffl", "ffl_decompose"),
    "firpo_fortin_lemieux": ("statspai.decomposition.ffl", "ffl_decompose"),

    "machado_mata": ("statspai.decomposition.machado_mata", "machado_mata"),
    "mm": ("statspai.decomposition.machado_mata", "machado_mata"),

    "melly": ("statspai.decomposition.melly", "melly_decompose"),

    "cfm": ("statspai.decomposition.cfm", "cfm_decompose"),
    "chernozhukov_fernandez_val_melly": ("statspai.decomposition.cfm", "cfm_decompose"),

    "fairlie": ("statspai.decomposition.nonlinear", "fairlie"),
    "bauer_sinning": ("statspai.decomposition.nonlinear", "bauer_sinning"),
    "yun_nonlinear": ("statspai.decomposition.nonlinear", "bauer_sinning"),

    "inequality": ("statspai.decomposition.inequality", "subgroup_decompose"),
    "subgroup": ("statspai.decomposition.inequality", "subgroup_decompose"),
    "shapley_inequality": ("statspai.decomposition.inequality",
                           "shapley_inequality"),
    "gini_source": ("statspai.decomposition.inequality", "source_decompose"),

    "kitagawa": ("statspai.decomposition.kitagawa", "kitagawa_decompose"),
    "das_gupta": ("statspai.decomposition.kitagawa", "das_gupta"),

    "gap_closing": ("statspai.decomposition.causal", "gap_closing"),
    "lundberg": ("statspai.decomposition.causal", "gap_closing"),
    "mediation": ("statspai.decomposition.causal", "mediation_decompose"),
    "natural_effects": ("statspai.decomposition.causal", "mediation_decompose"),
    "causal_jvw": ("statspai.decomposition.causal", "disparity_decompose"),
    "jackson_vanderweele": ("statspai.decomposition.causal",
                            "disparity_decompose"),
    "disparity": ("statspai.decomposition.causal", "disparity_decompose"),
}


def available_methods() -> list[str]:
    """Return list of all registered decomposition method names."""
    return sorted(_REGISTRY.keys())


def decompose(method: str, /, **kwargs) -> Any:
    """
    Unified entry point for all decomposition methods.

    Parameters
    ----------
    method : str
        One of the methods listed in ``available_methods()``. Aliases
        are supported (e.g. 'mm' → 'machado_mata').
    **kwargs : method-specific keyword arguments (see individual
        function signatures for details).

    Returns
    -------
    method-specific result class with ``.summary()``, ``.plot()``,
    ``.to_latex()``, ``._repr_html_()``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.decomposition.datasets.cps_wage()
    >>> r = sp.decompose('oaxaca', data=df, y='log_wage', group='female',
    ...                  x=['education', 'experience', 'tenure'])
    >>> r.summary()

    >>> r = sp.decompose('ffl', data=df, y='log_wage', group='female',
    ...                  x=['education', 'experience', 'tenure'],
    ...                  stat='quantile', tau=0.5)
    >>> r.summary()

    >>> # NOTE: ``method='aipw'`` below is passed through to
    >>> # ``gap_closing``'s own ``method`` parameter; the dispatcher's
    >>> # own method arg is positional-only so there is no collision.
    >>> r = sp.decompose('gap_closing', data=df, y='log_wage',
    ...                  group='female',
    ...                  x=['education', 'experience', 'tenure'],
    ...                  method='aipw')

    Convention warning for ``dfl`` vs ``machado_mata`` / ``melly`` /
    ``cfm``: ``reference=0`` has different semantics across method
    families (reweighting vs coefficient-swap). See the per-method
    docstrings before comparing composition/structure estimates across
    methods.
    """
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown method {method!r}. Available: "
            + ", ".join(available_methods())
        )
    module_path, fn_name = _REGISTRY[method]
    import importlib
    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name)
    return fn(**kwargs)
