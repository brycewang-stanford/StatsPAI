"""Panel for the sp.event_study option fixture (staggered adoption).

160 units over 12 periods in 16 states of 10 units and 4 regions; cohorts
adopt in periods 4, 6, 8 and 10 and 40 units never do. Beyond the outcome
``y`` it carries what the options read:

* ``x``      a time-varying covariate;
* ``w``      a unit-level weight, ``wt`` a weight that varies over time;
* ``dose``   a unit-level treatment intensity, zero for about a third of
             the units, and ``y_int``, an outcome whose effect from a
             common adoption date (period 7) scales with it;
* ``keep``   1 on about 85% of the rows: the unbalanced subsample.

Effects grow with time since adoption so the event-time coefficients are
not all alike.

    python tests/reference_parity/_fixtures/_generate_event_study_options_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261015)
N, T = 160, 12
g = rng.choice([4, 6, 8, 10, 0], size=N, p=[0.2, 0.2, 0.2, 0.15, 0.25])
unit = np.repeat(np.arange(1, N + 1), T)
time = np.tile(np.arange(1, T + 1), N)
gg = np.repeat(g, T)
state = (unit - 1) // 10 + 1
region = (unit - 1) % 4 + 1
rel = time - gg
alpha = np.repeat(rng.normal(size=N), T)
x = rng.normal(size=N * T) + 0.3 * alpha
effect = np.where((gg > 0) & (rel >= 0), 0.5 + 0.2 * rel, 0.0)
y = alpha + 0.1 * time + 0.4 * x + effect + rng.normal(size=N * T)
w = np.repeat(np.exp(rng.normal(scale=0.5, size=N)), T)
wt = w * np.exp(rng.normal(scale=0.3, size=N * T))
dose_u = np.where(rng.random(N) < 0.35, 0.0, rng.uniform(0.5, 2.0, size=N))
dose = np.repeat(dose_u, T)
rel_c = time - 7
y_int = (
    alpha
    + 0.1 * time
    + np.where(rel_c >= 0, dose * (0.3 + 0.1 * rel_c), 0.0)
    + rng.normal(size=N * T)
)
keep = (rng.random(N * T) > 0.15).astype(int)
out = pd.DataFrame(
    {
        "unit": unit,
        "time": time,
        "g": gg,
        "state": state,
        "region": region,
        "x": np.round(x, 10),
        "w": np.round(w, 10),
        "wt": np.round(wt, 10),
        "dose": np.round(dose, 10),
        "y": np.round(y, 10),
        "y_int": np.round(y_int, 10),
        "keep": keep,
    }
)
out.to_csv(Path(__file__).with_name("event_study_options.csv"), index=False)
print(out.shape, sorted(pd.Series(g).value_counts().to_dict().items()))
