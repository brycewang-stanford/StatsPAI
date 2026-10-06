"""Synthetic series for test_tsfilter_parity.py (seeded; no real data).

``y``: 160 observations of 100 * log of a trending series: a random walk
with drift plus a stationary AR(2) cycle.

    python tests/reference_parity/_fixtures/_generate_tsfilter_data.py
"""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261007)
n = 160
trend = 1000.0 + np.cumsum(0.5 + 0.6 * rng.normal(size=n))
e = rng.normal(size=n + 50)
c = np.zeros(n + 50)
for t in range(2, n + 50):
    c[t] = 1.3 * c[t - 1] - 0.5 * c[t - 2] + 0.7 * e[t]
y = trend + c[50:]
lines = ["t,y"] + [f"{t + 1},{float(y[t])!r}" for t in range(n)]
out = Path(__file__).parent / "tsfilter.csv"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
