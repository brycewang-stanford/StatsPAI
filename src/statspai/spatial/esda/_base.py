"""Shared ESDA result plumbing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SpatialStatistic:
    name: str
    value: float
    expectation: float
    variance: float
    z_score: float
    p_norm: float
    p_sim: Optional[float] = None
    simulations: Optional[np.ndarray] = None
    extras: dict = field(default_factory=dict)

    @property
    def I(self) -> float:           # noqa: E741 (Moran convention)
        return self.value

    @property
    def C(self) -> float:
        return self.value

    @property
    def G(self) -> float:
        return self.value

    def summary(self) -> str:
        lines = [
            f"{self.name}",
            "-" * 40,
            f"Statistic      : {self.value: .6f}",
            f"E[statistic]   : {self.expectation: .6f}",
            f"Var[statistic] : {self.variance: .6f}",
            f"z              : {self.z_score: .4f}",
            f"p (normal)     : {self.p_norm: .4f}",
        ]
        if self.p_sim is not None:
            lines.append(f"p (permutation): {self.p_sim: .4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def permutation_pvalue(observed: float, sims: np.ndarray) -> float:
    """Two-sided empirical p-value from permutation replicates (centered on sims mean)."""
    sims = np.asarray(sims)
    centre = sims.mean()
    larger = np.sum(np.abs(sims - centre) >= abs(observed - centre))
    return (larger + 1) / (len(sims) + 1)
