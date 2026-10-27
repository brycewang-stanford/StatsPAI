"""Data for the sp.positive_correlation_test vs Stata biprobit fixture.

Run from the repository root:
    python tests/reference_parity/_fixtures/_generate_positive_correlation_test_data.py
Writes positive_correlation_test_data.csv (floats as %.17g, so Stata's
``import delimited`` and pandas read the same doubles).
"""

import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261028)
n = 600
x1 = rng.normal(size=n)
x2 = (rng.uniform(size=n) < 0.4).astype(int)
x3 = rng.normal(size=n)
u = rng.normal(size=n)
e1 = 0.55 * u + np.sqrt(1 - 0.55**2) * rng.normal(size=n)
e2 = 0.55 * u + np.sqrt(1 - 0.55**2) * rng.normal(size=n)
df = pd.DataFrame(
    {
        "cover": (0.2 + 0.5 * x1 - 0.4 * x2 + e1 > 0).astype(int),
        "claim": (-0.6 + 0.3 * x1 + 0.5 * x3 + e2 > 0).astype(int),
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "g": rng.integers(1, 41, size=n),
    }
)
out = pathlib.Path(__file__).with_name("positive_correlation_test_data.csv")
df.to_csv(out, index=False, float_format="%.17g")
