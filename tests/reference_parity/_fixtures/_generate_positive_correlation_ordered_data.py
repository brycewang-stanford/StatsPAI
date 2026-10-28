"""Data for the ordered-outcome sp.positive_correlation_test vs Stata fixture.

Run from the repository root (one command):
    python tests/reference_parity/_fixtures/
        _generate_positive_correlation_ordered_data.py
Writes positive_correlation_ordered_data.csv (floats as %.17g, so Stata's
``import delimited`` and pandas read the same doubles). ``cover`` has three
ordered coverage tiers, ``nclaim`` is a claim count top-coded at 2 and
``claim`` its 0/1 version; the latent errors are correlated (0.3).
"""

import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261029)
n = 800
x1 = rng.normal(size=n)
x2 = (rng.uniform(size=n) < 0.4).astype(int)
x3 = rng.normal(size=n)
u = rng.normal(size=n)
a = np.sqrt(0.3)
e1 = a * u + np.sqrt(1 - a**2) * rng.normal(size=n)
e2 = a * u + np.sqrt(1 - a**2) * rng.normal(size=n)
cover = np.digitize(0.4 * x1 - 0.3 * x2 + e1, [-0.4, 0.6])
nclaim = np.digitize(0.3 * x1 + 0.4 * x3 + e2, [0.5, 1.4])
df = pd.DataFrame(
    {
        "cover": cover,
        "nclaim": nclaim,
        "claim": (nclaim > 0).astype(int),
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "g": rng.integers(1, 51, size=n),
    }
)
out = pathlib.Path(__file__).with_name("positive_correlation_ordered_data.csv")
df.to_csv(out, index=False, float_format="%.17g")
