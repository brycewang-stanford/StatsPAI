"""Data for tests/reference_parity/test_feols_iv_native_fixest_parity.py.

A firm x year panel with 1% singleton firms, one endogenous regressor x1
(two instruments z, z2) and a second endogenous regressor x3. Run from
this directory:
    python _generate_feols_iv_native_data.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
n = 4000
df = pd.DataFrame(
    dict(
        firm=rng.integers(0, 60, n),
        year=rng.integers(0, 8, n),
        g=rng.integers(0, 25, n),
        z=rng.normal(size=n),
        z2=rng.normal(size=n),
        x2=rng.normal(size=n),
        w=rng.uniform(0.5, 2.0, n),
    )
)
df["firm"] = np.where(rng.random(n) < 0.01, 1000 + np.arange(n), df.firm)
v = rng.normal(size=n)
df["x1"] = df.z + 0.5 * df.z2 + 0.3 * df.x2 + v + 0.1 * (df.firm % 5)
df["x3"] = df.z2 - df.z + rng.normal(size=n)
df["y"] = 2 * df.x1 - 1.5 * df.x2 + 0.5 * df.x3 + v + rng.normal(size=n)
df.to_csv("feols_iv_native_data.csv", index=False, float_format="%.17g")
