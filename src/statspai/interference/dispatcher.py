"""
Unified dispatcher: ``sp.interference(design=...)``

Single entry point for the interference / spillover family.  Mirrors
the style of ``sp.synth`` / ``sp.decompose`` / ``sp.dml``.

Examples
--------
>>> import statspai as sp
>>> # Within-cluster spillover (exposure split of cluster mates' treatment)
>>> r = sp.interference("spillover", data=df, y="y", treat="d",
...                     cluster="household")  # doctest: +SKIP
>>> # Hudgens-Halloran two-stage randomized (saturation) design
>>> r = sp.interference("two_stage", data=df, y="y", treat="d",
...                     group="village", saturation="share")  # doctest: +SKIP
>>> # Aronow-Samii Horvitz-Thompson on a known network
>>> r = sp.interference("network_exposure", Y=y, Z=z, adjacency=A, mapping="as4", p_treat=0.3)  # doctest: +SKIP
>>> # Orthogonal ML direct + spillover effects
>>> r = sp.interference("network_hte", data=df, y="y", treatment="d", neighbor_exposure="e", covariates=["x1"])  # doctest: +SKIP
>>> # Staggered-rollout cluster RCT
>>> r = sp.interference("cluster_staggered", data=df, y="y", cluster="village", time="month", first_treat="first_treatment_month")  # doctest: +SKIP

A fully runnable example appears in :func:`statspai.interference` below.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

_REGISTRY: Dict[str, Tuple[str, str]] = {
    # -- Partial interference ------------------------------------- #
    # 'partial' / 'hudgens_halloran' are deprecated aliases of 'spillover'
    # (see _RETARGETED_ALIASES): sp.spillover is not the Hudgens-Halloran
    # estimator, and the next minor release points them to two_stage.
    "partial": ("statspai.interference.spillover", "spillover"),
    "spillover": ("statspai.interference.spillover", "spillover"),
    "hudgens_halloran": ("statspai.interference.spillover", "spillover"),
    "two_stage": ("statspai.interference.two_stage", "two_stage_interference"),
    "saturation": ("statspai.interference.two_stage", "two_stage_interference"),
    # -- Network interference ------------------------------------- #
    "network_exposure": ("statspai.interference.network_exposure", "network_exposure"),
    "aronow_samii": ("statspai.interference.network_exposure", "network_exposure"),
    "as": ("statspai.interference.network_exposure", "network_exposure"),
    "horvitz_thompson": ("statspai.interference.network_exposure", "network_exposure"),
    "peer_effects": ("statspai.interference.peer_effects", "peer_effects"),
    "linear_in_means": ("statspai.interference.peer_effects", "peer_effects"),
    "manski": ("statspai.interference.peer_effects", "peer_effects"),
    "bramoulle": ("statspai.interference.peer_effects", "peer_effects"),
    "network_hte": ("statspai.interference.orthogonal", "network_hte"),
    "orthogonal": ("statspai.interference.orthogonal", "network_hte"),
    "parmigiani": ("statspai.interference.orthogonal", "network_hte"),
    "inward_outward": ("statspai.interference.orthogonal", "inward_outward_spillover"),
    "inward_outward_spillover": (
        "statspai.interference.orthogonal",
        "inward_outward_spillover",
    ),
    "directed": ("statspai.interference.orthogonal", "inward_outward_spillover"),
    # -- Cluster RCTs with interference --------------------------- #
    "cluster_matched_pair": (
        "statspai.interference.cluster_matched_pair",
        "cluster_matched_pair",
    ),
    "matched_pair": (
        "statspai.interference.cluster_matched_pair",
        "cluster_matched_pair",
    ),
    "bai": ("statspai.interference.cluster_matched_pair", "cluster_matched_pair"),
    "cluster_cross": (
        "statspai.interference.cluster_cross",
        "cluster_cross_interference",
    ),
    "cluster_cross_interference": (
        "statspai.interference.cluster_cross",
        "cluster_cross_interference",
    ),
    "cross_interference": (
        "statspai.interference.cluster_cross",
        "cluster_cross_interference",
    ),
    "cluster_staggered": (
        "statspai.interference.cluster_staggered",
        "cluster_staggered_rollout",
    ),
    "cluster_staggered_rollout": (
        "statspai.interference.cluster_staggered",
        "cluster_staggered_rollout",
    ),
    "staggered_cluster": (
        "statspai.interference.cluster_staggered",
        "cluster_staggered_rollout",
    ),
    "dnc_gnn": ("statspai.interference.dnc_gnn_did", "dnc_gnn_did"),
    "dnc_gnn_did": ("statspai.interference.dnc_gnn_did", "dnc_gnn_did"),
    "dnc": ("statspai.interference.dnc_gnn_did", "dnc_gnn_did"),
}

# Aliases whose target changes in the next minor release (MIGRATION.md).
_RETARGETED_ALIASES = frozenset({"partial", "hudgens_halloran"})


def available_designs() -> list[str]:
    """Return the full list of registered interference ``design`` names.

    These are the strings accepted as the first positional argument of
    :func:`statspai.interference`.

    Examples
    --------
    >>> import statspai as sp
    >>> designs = sp.interference_available_designs()
    >>> bool("spillover" in designs and "network_exposure" in designs)
    True
    >>> designs == sorted(designs)  # returned sorted alphabetically
    True
    """
    return sorted(_REGISTRY.keys())


def interference(design: str = "spillover", /, **kwargs: Any) -> Any:
    """Unified entry point for the interference / spillover family.

    Parameters
    ----------
    design : str, default ``"spillover"``
        The interference design to estimate under.  Supported values are
        listed by ``sp.interference_available_designs()``.

        ``"partial"`` and ``"hudgens_halloran"`` currently route to
        :func:`sp.spillover`, which is *not* the Hudgens-Halloran (2008)
        estimator, and emit a ``FutureWarning``: in the next minor release
        they will route to :func:`sp.two_stage_interference` (the
        Hudgens-Halloran two-stage estimator, which needs a ``group`` and
        a ``saturation`` column). Use ``"spillover"`` to keep the current
        estimator or ``"two_stage"`` for Hudgens-Halloran.
    **kwargs
        Passed through unchanged to the target function.

    Returns
    -------
    The underlying estimator's return object.  See each underlying
    function's docstring for details.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> cluster = np.repeat(np.arange(40), 8)
    >>> treat = rng.integers(0, 2, size=cluster.size)
    >>> frac = pd.Series(treat).groupby(cluster).transform("mean").to_numpy()
    >>> y = 1.0 + 0.8 * treat + 0.5 * frac + rng.normal(size=cluster.size)
    >>> df = pd.DataFrame({"y": y, "d": treat, "household": cluster})
    >>> r = sp.interference("spillover", data=df, y="y", treat="d",
    ...                     cluster="household", n_bootstrap=100)
    >>> type(r).__name__
    'CausalResult'

    See Also
    --------
    docs/guides/interference_family.md : the full family guide.
    """
    if design not in _REGISTRY:
        raise ValueError(
            f"Unknown interference design {design!r}. Available: "
            + ", ".join(available_designs())
        )

    if design in _RETARGETED_ALIASES:
        warnings.warn(
            f"sp.interference({design!r}) routes to sp.spillover, which is not "
            "the Hudgens-Halloran estimator. In the next minor release "
            f"{design!r} will route to sp.two_stage_interference (Hudgens-"
            "Halloran two-stage designs; needs group= and saturation=). Pass "
            "'spillover' to keep the current estimator, or 'two_stage' for "
            "Hudgens-Halloran.",
            FutureWarning,
            stacklevel=2,
        )
    module_path, fn_name = _REGISTRY[design]
    import importlib

    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name)
    return fn(**kwargs)


__all__ = ["interference", "available_designs"]
