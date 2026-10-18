"""Data for tests/reference_parity/test_feglm_native_fixest_parity.py.

A binary outcome on a firm x year panel with 1% singleton firms and one
firm in eighty forced to all-zero outcomes, so fixest's perfect-fit rule
drops observations. Run from this directory:
    python _generate_feglm_native_data.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(5)
n = 2000
df = pd.DataFrame(
    dict(
        firm=rng.integers(0, 60, n),
        year=rng.integers(0, 8, n),
        g=rng.integers(0, 20, n),
        x1=rng.normal(size=n),
        x2=rng.normal(size=n),
        w=rng.uniform(0.5, 2.0, n),
    )
)
df["firm"] = np.where(rng.random(n) < 0.01, 1000 + np.arange(n), df.firm)
alpha = rng.normal(size=2000)
eta = 0.5 * df.x1 - 0.4 * df.x2 + 0.7 * alpha[df.firm % 2000]
df["d"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
df.loc[df.firm % 60 == 7, "d"] = 0
df.to_csv("feglm_native_data.csv", index=False, float_format="%.17g")
