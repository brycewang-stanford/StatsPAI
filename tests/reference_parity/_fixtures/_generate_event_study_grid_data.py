"""Panel for the sp.event_study reference grid (staggered adoption).

150 units over 12 periods; cohorts adopt in periods 5, 7 and 9 and 45
units never do. A time-varying covariate ``x`` and a unit-level weight
``w``. Effects grow with time since adoption so the event-time
coefficients are not all alike.

    python tests/reference_parity/_fixtures/_generate_event_study_grid_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261005)
N, T = 150, 12
g = rng.choice([5, 7, 9, 0], size=N, p=[0.25, 0.25, 0.2, 0.3])
unit = np.repeat(np.arange(1, N + 1), T)
time = np.tile(np.arange(1, T + 1), N)
gg = np.repeat(g, T)
rel = time - gg
x = rng.normal(size=N * T) + 0.3 * np.repeat(rng.normal(size=N), T)
effect = np.where((gg > 0) & (rel >= 0), 0.5 + 0.25 * rel, 0.0)
y = (
    np.repeat(rng.normal(size=N), T)
    + 0.1 * time
    + 0.4 * x
    + effect
    + rng.normal(size=N * T)
)
w = np.repeat(np.exp(rng.normal(scale=0.5, size=N)), T)
out = pd.DataFrame(
    {
        "unit": unit,
        "time": time,
        "g": gg,
        "x": np.round(x, 10),
        "w": np.round(w, 10),
        "y": np.round(y, 10),
    }
)
out.to_csv(Path(__file__).with_name("event_study_grid.csv"), index=False)
print(out.shape, sorted(pd.Series(g).value_counts().to_dict().items()))
