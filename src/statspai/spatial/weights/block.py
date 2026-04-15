"""Block (regime) spatial weights."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .core import W


def block_weights(regimes) -> W:
    regimes = np.asarray(regimes)
    buckets = defaultdict(list)
    for i, r in enumerate(regimes):
        buckets[r].append(i)
    neighbors = {i: [] for i in range(len(regimes))}
    for ids in buckets.values():
        for i in ids:
            neighbors[i] = [j for j in ids if j != i]
    return W(neighbors)
