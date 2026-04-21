"""
Unified dispatcher: ``sp.interference(design=...)``

Single entry point for the interference / spillover family.  Mirrors
the style of ``sp.synth`` / ``sp.decompose`` / ``sp.dml``.

Examples
--------
>>> import statspai as sp
>>> # Partial interference: clusters don't interact across boundaries
>>> r = sp.interference("partial", data=df, y="y", treat="d",
...                     cluster="household")
>>> # Aronow-Samii Horvitz-Thompson on a known network
>>> r = sp.interference("network_exposure", Y=y, Z=z, adjacency=A,
...                     mapping="as4", p_treat=0.3)
>>> # Orthogonal ML direct + spillover effects
>>> r = sp.interference("network_hte", data=df, y="y", treatment="d",
...                     neighbor_exposure="e", covariates=["x1"])
>>> # Staggered-rollout cluster RCT
>>> r = sp.interference("cluster_staggered", data=df, y="y",
...                     cluster="village", time="month",
...                     first_treat="first_treatment_month")
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


_REGISTRY: Dict[str, Tuple[str, str]] = {
    # -- Partial interference (Hudgens-Halloran / Aronow-Samii 2017) #
    "partial": ("statspai.interference.spillover", "spillover"),
    "spillover": ("statspai.interference.spillover", "spillover"),
    "hudgens_halloran": ("statspai.interference.spillover", "spillover"),

    # -- Network interference ------------------------------------- #
    "network_exposure": ("statspai.interference.network_exposure",
                         "network_exposure"),
    "aronow_samii": ("statspai.interference.network_exposure",
                     "network_exposure"),
    "as": ("statspai.interference.network_exposure", "network_exposure"),
    "horvitz_thompson": ("statspai.interference.network_exposure",
                         "network_exposure"),

    "peer_effects": ("statspai.interference.peer_effects", "peer_effects"),
    "linear_in_means": ("statspai.interference.peer_effects", "peer_effects"),
    "manski": ("statspai.interference.peer_effects", "peer_effects"),
    "bramoulle": ("statspai.interference.peer_effects", "peer_effects"),

    "network_hte": ("statspai.interference.orthogonal", "network_hte"),
    "orthogonal": ("statspai.interference.orthogonal", "network_hte"),
    "parmigiani": ("statspai.interference.orthogonal", "network_hte"),

    "inward_outward": ("statspai.interference.orthogonal",
                       "inward_outward_spillover"),
    "inward_outward_spillover": ("statspai.interference.orthogonal",
                                 "inward_outward_spillover"),
    "directed": ("statspai.interference.orthogonal",
                 "inward_outward_spillover"),

    # -- Cluster RCTs with interference --------------------------- #
    "cluster_matched_pair": ("statspai.interference.cluster_matched_pair",
                             "cluster_matched_pair"),
    "matched_pair": ("statspai.interference.cluster_matched_pair",
                     "cluster_matched_pair"),
    "bai": ("statspai.interference.cluster_matched_pair",
            "cluster_matched_pair"),

    "cluster_cross": ("statspai.interference.cluster_cross",
                      "cluster_cross_interference"),
    "cluster_cross_interference": ("statspai.interference.cluster_cross",
                                   "cluster_cross_interference"),
    "cross_interference": ("statspai.interference.cluster_cross",
                           "cluster_cross_interference"),

    "cluster_staggered": ("statspai.interference.cluster_staggered",
                          "cluster_staggered_rollout"),
    "cluster_staggered_rollout": ("statspai.interference.cluster_staggered",
                                  "cluster_staggered_rollout"),
    "staggered_cluster": ("statspai.interference.cluster_staggered",
                          "cluster_staggered_rollout"),

    "dnc_gnn": ("statspai.interference.dnc_gnn_did", "dnc_gnn_did"),
    "dnc_gnn_did": ("statspai.interference.dnc_gnn_did", "dnc_gnn_did"),
    "dnc": ("statspai.interference.dnc_gnn_did", "dnc_gnn_did"),
}


def available_designs() -> list[str]:
    """Return the full list of registered interference ``design`` names."""
    return sorted(_REGISTRY.keys())


def interference(design: str = "partial", /, **kwargs: Any) -> Any:
    """Unified entry point for the interference / spillover family.

    Parameters
    ----------
    design : str, default ``"partial"``
        The interference design to estimate under.  Supported values are
        listed by ``sp.interference_available_designs()``.
    **kwargs
        Passed through unchanged to the target function.

    Returns
    -------
    The underlying estimator's return object.  See each underlying
    function's docstring for details.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.interference("partial", data=df, y="y",
    ...                      treat="d", cluster="household")

    See Also
    --------
    docs/guides/interference_family.md : the full family guide.
    """
    if design not in _REGISTRY:
        raise ValueError(
            f"Unknown interference design {design!r}. Available: "
            + ", ".join(available_designs())
        )

    module_path, fn_name = _REGISTRY[design]
    import importlib
    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name)
    return fn(**kwargs)


__all__ = ["interference", "available_designs"]
