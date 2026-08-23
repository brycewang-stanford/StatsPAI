"""Shared degrees-of-freedom bookkeeping for absorbed fixed effects.

One place for the rule every HDFE estimator and every HDFE diagnostic has
to agree on: how many degrees of freedom an absorbed fixed-effect block
actually costs once the vcov is clustered. Getting this wrong is invisible
in the point estimate and shows up only as standard errors that disagree
with ``reghdfe`` / ``ivreghdfe`` / ``fixest`` by a few percent.

The rule itself is a property of the reference implementations
(``reghdfe``'s ``dofadjustments(clusters)``, ``fixest``'s
``fixef.K="nested"``), pinned by the parity tests in
``tests/reference_parity/test_iv_hdfe_stata_parity.py``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["fe_nested_in_cluster", "absorbed_dof_charge"]


def fe_nested_in_cluster(fe: Iterable, cluster_frame: pd.DataFrame) -> bool:
    """True when every level of ``fe`` sits inside a single cluster.

    A fixed effect nested within a clustering dimension consumes no
    residual degrees of freedom for cluster-robust inference, because the
    cluster sums already annihilate it. ``reghdfe``/``ivreghdfe`` (the
    ``clusters`` entry of ``dofadjustments()``) and ``fixest``
    (``fixef.K="nested"``) both drop it; charging it anyway inflates the
    standard errors — for the canonical unit-FE / cluster-by-unit panel by
    roughly ``sqrt((N-k)/(N-k-G_unit))``.
    """
    fe_codes = pd.factorize(np.asarray(fe), sort=False)[0]
    for col in cluster_frame.columns:
        c_codes = pd.factorize(np.asarray(cluster_frame[col]), sort=False)[0]
        pairs = pd.DataFrame({"fe": fe_codes, "cl": c_codes}).drop_duplicates()
        if not pairs["fe"].duplicated().any():
            return True
    return False


def absorbed_dof_charge(
    fe_frame: Optional[pd.DataFrame],
    fe_names: Sequence[str],
    fe_cardinality: Sequence[int],
    cluster_frame: Optional[pd.DataFrame],
) -> Tuple[int, List[str]]:
    """Degrees of freedom an absorbed FE block costs, plus the nested names.

    Reproduces ``ivreg2``'s ``e(sdofminus)``: ``sum(G_k - 1)`` over fixed
    effects that are *not* nested within a clustering dimension, plus one
    for the constant the absorbed block spans — unless a nested (hence
    fully redundant) fixed effect has already swallowed it — and never
    less than one.

    Returns
    -------
    (charge, nested_names)
    """
    nested: List[str] = []
    charge = 0
    for name, card in zip(fe_names, fe_cardinality):
        if (
            cluster_frame is not None
            and fe_frame is not None
            and fe_nested_in_cluster(fe_frame[name], cluster_frame)
        ):
            nested.append(name)
        else:
            charge += int(card) - 1
    if not nested:
        charge += 1
    return max(charge, 1), nested
