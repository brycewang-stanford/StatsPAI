"""Synthetic series for test_spectral_parity.py (seeded; no real data).

``x``: 150 observations of an AR(2) with a cycle of about 12 periods plus
a linear trend. ``z``: the same length, white noise. The R and Stata
generators also use the first 127 rows of ``x`` (a prime length, so that
R's ``fast = TRUE`` pads).

    python tests/reference_parity/_fixtures/_generate_spectral_data.py
"""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261006)
n = 150
e = rng.normal(size=n + 100)
x = np.zeros(n + 100)
for t in range(2, n + 100):
    x[t] = 1.5 * x[t - 1] - 0.75 * x[t - 2] + e[t]
x = x[100:] + 0.02 * np.arange(n) + 3.0
z = rng.normal(size=n)
lines = ["t,x,z"] + [f"{t + 1},{float(x[t])!r},{float(z[t])!r}" for t in range(n)]
out = Path(__file__).parent / "spectral.csv"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
