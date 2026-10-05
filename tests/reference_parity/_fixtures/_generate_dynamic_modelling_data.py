"""Synthetic series for tests/reference_parity/test_dynamic_modelling_parity.py.

Writes dynamic_modelling.csv next to this file. The R script
_generate_dynamic_modelling_R.R reads the same bytes.

    python tests/reference_parity/_fixtures/_generate_dynamic_modelling_data.py

Columns
-------
gdp     integrated series in the tens of thousands with drift and AR(1)
        growth; innovation variance about 1.4e5, the scale at which a
        diffuse prior of variance 1e6 stops being diffuse
x       stationary AR(1)
y       1 + 0.8 x + u, with the slope 0.5 higher from row 100 on
c1 c2 c3  c1 and c2 are cointegrated random walks, c3 is a third walk
ret     ARCH(1) series: uncorrelated but not independent
mkt stk a market return and a stock return whose beta drifts upward
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261006)
n = 160
e = rng.normal(size=(n, 9))

growth = np.zeros(n)
growth[0] = 250.0
for t in range(1, n):
    growth[t] = 250.0 + 0.3 * (growth[t - 1] - 250.0) + 380.0 * e[t, 0]
gdp = 12000.0 + np.cumsum(growth)

x = np.zeros(n)
for t in range(1, n):
    x[t] = 0.6 * x[t - 1] + e[t, 1]
y = 1.0 + 0.8 * x + 0.5 * x * (np.arange(n) >= 100) + 0.7 * e[:, 2]

c2 = np.cumsum(0.05 + 0.4 * e[:, 3])
gap = np.zeros(n)
for t in range(1, n):
    gap[t] = 0.5 * gap[t - 1] + 0.3 * e[t, 4]
c1 = 0.5 + 0.9 * c2 + gap
c3 = np.cumsum(0.4 * e[:, 5])

ret = np.zeros(n)
for t in range(1, n):
    ret[t] = e[t, 6] * np.sqrt(0.2 + 0.7 * ret[t - 1] ** 2)

mkt = 0.01 + 0.04 * e[:, 7]
beta = np.linspace(0.7, 1.5, n)
stk = 0.002 + beta * mkt + 0.03 * e[:, 8]

frame = pd.DataFrame(
    {
        "t": np.arange(1, n + 1),
        "gdp": gdp,
        "x": x,
        "y": y,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "ret": ret,
        "mkt": mkt,
        "stk": stk,
    }
).round(8)
frame.to_csv(Path(__file__).with_name("dynamic_modelling.csv"), index=False)
