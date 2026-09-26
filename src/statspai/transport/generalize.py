"""
Generalize an RCT estimate to a named target population.

Thin convenience wrapper around ``transport_weights`` for the
"RCT -> target" workflow emphasised by Hernan-Robins and the
Bareinboim transportability program.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .._aliases import accepts_aliases
from .weighting import TransportWeightResult, transport_weights


@accepts_aliases(treat="treatment", y="outcome")
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

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> ns, nt = 400, 400
    >>> rct = pd.DataFrame({
    ...     "x": rng.normal(0, 1, ns),
    ...     "treat": rng.integers(0, 2, ns),
    ... })
    >>> rct["y"] = rct["treat"] + 0.5 * rct["x"] + rng.normal(0, 1, ns)
    >>> target = pd.DataFrame({"x": rng.normal(0.5, 1, nt)})
    >>> res = sp.transport_generalize(
    ...     rct, target, features=["x"], treatment="treat", outcome="y")
    >>> type(res).__name__
    'TransportWeightResult'
    >>> bool(res.ess <= ns)
    True
    >>> bool(np.isfinite(res.effect_transported))
    True
    """
    return transport_weights(
        source=rct,
        target=target_population,
        features=features,
        treatment=treatment,
        outcome=outcome,
    )
