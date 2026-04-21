"""
Unified dispatcher: ``sp.conformal(kind=...)``

Single entry point for the conformal causal inference family.  Mirrors
the style of ``sp.synth`` / ``sp.decompose`` / ``sp.dml``.

Examples
--------
>>> import statspai as sp
>>> # Default CATE intervals
>>> r = sp.conformal("cate", data=df, y="y", treat="d",
...                  covariates=["x1", "x2"])
>>> # ITE nested bound
>>> r = sp.conformal("ite", data=df, y="y", treat="d",
...                  covariates=["x1", "x2"], alpha=0.1)
>>> # Dose-response band
>>> r = sp.conformal("continuous", data=train, y="y", treatment="dose",
...                  covariates=["x"], test_data=test, dose_grid=grid)
>>> # Cluster-exchangeable under interference
>>> r = sp.conformal("interference", data=df, y="y", treatment="d",
...                  cluster="village", covariates=["x"],
...                  test_clusters=["v1", "v2"])
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


_REGISTRY: Dict[str, Tuple[str, str]] = {
    # -- Core Lei-Candès 2021 ------------------------------------- #
    "cate": ("statspai.conformal_causal.conformal_ite", "conformal_cate"),
    "conformal_cate": ("statspai.conformal_causal.conformal_ite",
                       "conformal_cate"),

    "counterfactual": ("statspai.conformal_causal.counterfactual",
                       "conformal_counterfactual"),
    "conformal_counterfactual": ("statspai.conformal_causal.counterfactual",
                                 "conformal_counterfactual"),

    "ite": ("statspai.conformal_causal.counterfactual",
            "conformal_ite_interval"),
    "ite_interval": ("statspai.conformal_causal.counterfactual",
                     "conformal_ite_interval"),
    "conformal_ite_interval": ("statspai.conformal_causal.counterfactual",
                               "conformal_ite_interval"),

    "weighted": ("statspai.conformal_causal.counterfactual",
                 "weighted_conformal_prediction"),
    "wcp": ("statspai.conformal_causal.counterfactual",
            "weighted_conformal_prediction"),
    "weighted_conformal_prediction": ("statspai.conformal_causal.counterfactual",
                                      "weighted_conformal_prediction"),

    # -- 2025-2026 frontier --------------------------------------- #
    "density": ("statspai.conformal_causal.conformal_density",
                "conformal_density_ite"),
    "density_ite": ("statspai.conformal_causal.conformal_density",
                    "conformal_density_ite"),
    "conformal_density_ite": ("statspai.conformal_causal.conformal_density",
                              "conformal_density_ite"),

    "multidp": ("statspai.conformal_causal.conformal_multidp",
                "conformal_ite_multidp"),
    "multi_stage": ("statspai.conformal_causal.conformal_multidp",
                    "conformal_ite_multidp"),
    "conformal_ite_multidp": ("statspai.conformal_causal.conformal_multidp",
                              "conformal_ite_multidp"),

    "debiased": ("statspai.conformal_causal.conformal_debiased",
                 "conformal_debiased_ml"),
    "debiased_ml": ("statspai.conformal_causal.conformal_debiased",
                    "conformal_debiased_ml"),
    "conformal_debiased_ml": ("statspai.conformal_causal.conformal_debiased",
                              "conformal_debiased_ml"),

    "fair": ("statspai.conformal_causal.conformal_fair",
             "conformal_fair_ite"),
    "fair_ite": ("statspai.conformal_causal.conformal_fair",
                 "conformal_fair_ite"),
    "conformal_fair_ite": ("statspai.conformal_causal.conformal_fair",
                           "conformal_fair_ite"),

    # -- v1.0 extended frontier ----------------------------------- #
    "continuous": ("statspai.conformal_causal.extended",
                   "conformal_continuous"),
    "dose": ("statspai.conformal_causal.extended", "conformal_continuous"),
    "dose_response": ("statspai.conformal_causal.extended",
                      "conformal_continuous"),
    "conformal_continuous": ("statspai.conformal_causal.extended",
                             "conformal_continuous"),

    "interference": ("statspai.conformal_causal.extended",
                     "conformal_interference"),
    "cluster": ("statspai.conformal_causal.extended",
                "conformal_interference"),
    "conformal_interference": ("statspai.conformal_causal.extended",
                               "conformal_interference"),
}


def available_kinds() -> list[str]:
    """Return the full list of registered conformal ``kind`` names."""
    return sorted(_REGISTRY.keys())


def conformal(kind: str = "cate", /, **kwargs: Any) -> Any:
    """Unified entry point for the conformal causal inference family.

    Parameters
    ----------
    kind : str, default ``"cate"``
        The conformal estimator to run.  Supported values are listed by
        ``sp.conformal_available_kinds()``.
    **kwargs
        Passed through unchanged to the target function.

    Returns
    -------
    The underlying estimator's return object — e.g. ``CausalResult`` for
    :func:`conformal_cate`, ``ConformalITEResult`` for ``kind="ite"``,
    ``ContinuousConformalResult`` for ``kind="continuous"``.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.conformal("cate", data=df, y="y", treat="d",
    ...                   covariates=["x1", "x2"])

    See Also
    --------
    docs/guides/conformal_family.md : the full family guide.
    """
    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown conformal kind {kind!r}. Available: "
            + ", ".join(available_kinds())
        )

    module_path, fn_name = _REGISTRY[kind]
    import importlib
    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name)
    return fn(**kwargs)


__all__ = ["conformal", "available_kinds"]
