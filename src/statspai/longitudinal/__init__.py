"""
Longitudinal causal inference (``sp.longitudinal``).

Unified entry for What If Layer-4 methods (time-varying treatments with
time-varying confounders).  Wraps MSM / g-formula ICE / IPW under a
single dispatcher with a dynamic-regime DSL.

>>> import numpy as np
>>> import pandas as pd
>>> import statspai as sp
>>> rng = np.random.default_rng(0)
>>> rows = []
>>> for i in range(150):
...     age, sex = rng.normal(40, 10), int(rng.random() < 0.5)
...     cd4_lag, vl = rng.normal(350, 100), rng.normal(4, 1)
...     for t in range(3):
...         drug = int(rng.random() < (0.7 if cd4_lag < 300 else 0.3))
...         cd4 = cd4_lag + 40 * drug - 5 * vl + rng.normal(0, 30)
...         rows.append({"pid": i, "visit": t, "drug": drug, "cd4": cd4,
...                      "cd4_lag": cd4_lag, "viral_load_lag": vl,
...                      "age": age, "sex": sex})
...         cd4_lag, vl = cd4, vl - 0.2 * drug + rng.normal(0, 0.3)
>>> panel = pd.DataFrame(rows)
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
>>> r.method, r.n, r.n_periods
('msm', 150, 3)

>>> diff = sp.longitudinal.contrast(
...     data=panel, id="pid", time="visit",
...     treatment="drug", outcome="cd4",
...     regime_a="always_treat",
...     regime_b="never_treat",
...     time_varying=["cd4_lag"],
... )
>>> sorted(diff)
['a_result', 'b_result', 'ci', 'contrast', 'regime_a', 'regime_b', 'se']
"""

from .analyze import LongitudinalResult, analyze, contrast
from .regime import Regime, always_treat, never_treat, regime

__all__ = [
    "Regime",
    "regime",
    "always_treat",
    "never_treat",
    "LongitudinalResult",
    "analyze",
    "contrast",
]
