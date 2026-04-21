"""
Longitudinal causal inference (``sp.longitudinal``).

Unified entry for What If Layer-4 methods (time-varying treatments with
time-varying confounders).  Wraps MSM / g-formula ICE / IPW under a
single dispatcher with a dynamic-regime DSL.

>>> import statspai as sp
>>> r = sp.longitudinal.analyze(
...     data=panel,
...     id="pid",
...     time="visit",
...     treatment="drug",
...     outcome="cd4",
...     time_varying=["cd4_lag", "viral_load_lag"],
...     baseline=["age", "sex"],
...     regime="if cd4_lag < 200 then 1 else 0",
... )
>>> r.summary()

>>> diff = sp.longitudinal.contrast(
...     data=panel, id="pid", time="visit",
...     treatment="drug", outcome="cd4",
...     regime_a="always_treat",
...     regime_b="never_treat",
...     time_varying=["cd4_lag"],
... )
"""

from .regime import Regime, regime, always_treat, never_treat
from .analyze import LongitudinalResult, analyze, contrast

__all__ = [
    "Regime", "regime", "always_treat", "never_treat",
    "LongitudinalResult", "analyze", "contrast",
]
