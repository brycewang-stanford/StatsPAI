"""
Transportability (``sp.transport``): generalize causal effects across
populations. Combines Pearl-Bareinboim identification (selection
diagrams) with Dahabreh-Stuart-style density-ratio weighting.

Quick start
-----------
>>> import numpy as np
>>> import pandas as pd
>>> import statspai as sp
>>> g = sp.dag("X -> Y; W -> Y; W -> X; S -> W")
>>> ident = sp.transport.identify_transport(g, treatment="X", outcome="Y",
...                                         selection_nodes={"S"})
>>> ident.transportable, sorted(ident.admissible_set)
(True, ['W'])
>>> rng = np.random.default_rng(0)
>>> rct = pd.DataFrame({"age": rng.normal(40, 10, 400),
...                     "sex": rng.integers(0, 2, 400),
...                     "treat": rng.integers(0, 2, 400)})
>>> rct["y"] = (rct["treat"] * (1 + 0.02 * (rct["age"] - 40))
...             + rng.normal(size=400))
>>> target_df = pd.DataFrame({"age": rng.normal(50, 10, 400),
...                           "sex": rng.integers(0, 2, 400)})
>>> tw = sp.transport.weights(source=rct, target=target_df,
...                           features=["age", "sex"],
...                           treatment="treat", outcome="y")
>>> type(tw).__name__, bool(tw.ess <= len(rct))
('TransportWeightResult', True)
"""

from .evidence_synthesis import (
    ConcordanceResult,
    EvidenceSynthesisResult,
    HeterogeneityResult,
    heterogeneity_of_effect,
    rwd_rct_concordance,
    synthesise_evidence,
)
from .generalize import generalize
from .identify import TransportIdentificationResult, identify_transport
from .weighting import TransportWeightResult
from .weighting import transport_weights as weights

__all__ = [
    "weights",
    "TransportWeightResult",
    "generalize",
    "identify_transport",
    "TransportIdentificationResult",
    "synthesise_evidence",
    "heterogeneity_of_effect",
    "rwd_rct_concordance",
    "EvidenceSynthesisResult",
    "HeterogeneityResult",
    "ConcordanceResult",
]
