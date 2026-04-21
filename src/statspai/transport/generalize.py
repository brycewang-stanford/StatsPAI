"""
Generalize an RCT estimate to a named target population.

Thin convenience wrapper around ``transport_weights`` for the
"RCT -> target" workflow emphasised by Hernan-Robins and the
Bareinboim transportability program.
"""

from __future__ import annotations
from typing import Sequence
import pandas as pd

from .weighting import transport_weights, TransportWeightResult


def generalize(
    rct: pd.DataFrame,
    target_population: pd.DataFrame,
    features: Sequence[str],
    treatment: str = "treat",
    outcome: str = "y",
) -> TransportWeightResult:
    """Transport an RCT effect to ``target_population``.

    Expects ``rct`` to contain the treatment indicator, outcome, and
    effect modifiers ``features``. ``target_population`` should
    contain the same ``features`` so the density ratio is estimable.
    """
    return transport_weights(
        source=rct,
        target=target_population,
        features=features,
        treatment=treatment,
        outcome=outcome,
    )
