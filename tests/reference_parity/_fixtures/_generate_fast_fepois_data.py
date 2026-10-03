"""Count outcome on the ``fast_feols_weights.csv`` panel, for ``sp.fast.fepois``.

Run once after ``_generate_fast_feols_weights_data.py``; commit
``fast_fepois.csv`` so the R generator (``_generate_fast_fepois_R.R``) and
the Python test read the same bytes. The count has a firm and a year
component, so both absorbed dimensions matter, and is over-dispersed, so
the model-based and robust variances differ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261004)
df = pd.read_csv("fast_feols_weights.csv")
firm_effect = rng.normal(scale=0.4, size=df["firm"].max() + 1)
mu = np.exp(
    0.3 * df["x1"] - 0.2 * df["x2"] + firm_effect[df["firm"]] + 0.08 * df["year"]
)
df["cnt"] = rng.negative_binomial(n=3, p=3 / (3 + mu))
df["c2"] = (df["firm"] * 7 + df["year"]) % 20
df.to_csv("fast_fepois.csv", index=False, float_format="%.17g")
print(f"wrote fast_fepois.csv: {len(df)} rows, mean count {df['cnt'].mean():.2f}")
