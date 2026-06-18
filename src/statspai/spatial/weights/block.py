"""Block (regime) spatial weights."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Hashable, List, Sequence

import numpy as np

from .core import W


def block_weights(regimes: Sequence[Any]) -> W:
    """Block (regime) spatial weights — full connectivity within each group.

    Builds a weights object in which every unit is a neighbour of every other
    unit sharing the same regime label, and of no unit outside it. Useful for
    encoding membership in a region, cluster, or treatment block as a spatial
    structure.

    Parameters
    ----------
    regimes : array_like
        Length-``n`` array of group labels (any hashable dtype). Units with
        equal labels become mutual neighbours.

    Returns
    -------
    W
        Spatial weights object with block-diagonal connectivity.

    Examples
    --------
    >>> import statspai as sp
    >>> w = sp.block_weights([0, 0, 1, 1, 1])
    >>> sorted(w.neighbors[2])
    [3, 4]
    """
    regime_array = np.asarray(regimes)
    buckets: DefaultDict[Hashable, List[int]] = defaultdict(list)
    for i, r in enumerate(regime_array):
        buckets[r].append(i)
    neighbors: Dict[int, List[int]] = {i: [] for i in range(len(regime_array))}
    for ids in buckets.values():
        for i in ids:
            neighbors[i] = [j for j in ids if j != i]
    return W(neighbors)
