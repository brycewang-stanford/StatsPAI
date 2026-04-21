"""
Transportability (``sp.transport``): generalize causal effects across
populations. Combines Pearl-Bareinboim identification (selection
diagrams) with Dahabreh-Stuart-style density-ratio weighting.

Quick start
-----------
>>> import statspai as sp
>>> g = sp.dag("S -> X; X -> Y; S -> Y")
>>> sp.transport.identify_transport(g, treatment="X", outcome="Y",
...                                 selection_nodes={"S"})
>>> sp.transport.weights(source=rct, target=target_df,
...                      features=["age", "sex"],
...                      treatment="treat", outcome="y")
"""

from .weighting import transport_weights as weights, TransportWeightResult
from .generalize import generalize
from .identify import identify_transport, TransportIdentificationResult

__all__ = [
    "weights",
    "TransportWeightResult",
    "generalize",
    "identify_transport",
    "TransportIdentificationResult",
]
