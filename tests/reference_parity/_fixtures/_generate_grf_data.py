"""Fixed HTE DGP for sp.causal_forest ↔ R grf parity.

CATE varies with X[:, 0]: tau(x) = 1 + 2 * x[:, 0]. Average ATE = 1.
n=1000, p=5 covariates.  Seed=42.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n, p = 1000, 5
X = rng.standard_normal((n, p))
# Treatment is randomized but with mild confounding via X[:, 1]
ps = 1.0 / (1.0 + np.exp(-0.3 * X[:, 1]))
W = (rng.uniform(size=n) < ps).astype(int)

# Heterogeneous treatment effect
tau = 1.0 + 2.0 * X[:, 0]
y0 = X @ np.array([0.5, 0.4, -0.3, 0.2, -0.1])
y = y0 + tau * W + 0.5 * rng.standard_normal(n)

cols = {"y": y, "W": W}
for j in range(p):
    cols[f"X{j+1}"] = X[:, j]
df = pd.DataFrame(cols)

import pathlib
out = pathlib.Path(__file__).parent / "grf_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {out} (n={n}, mean tau={tau.mean():.3f})")
