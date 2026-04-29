"""Fixed observational DGP for sp.bcf ↔ R bcf parity.

Adapted from Hahn-Murray-Carvalho (2020, JBA) DGP-1: linear μ-prognosis
+ heterogeneous treatment effect + propensity confounding.

n=400, p=5 covariates, true τ̄ = 1.0 (homogeneous + small HTE).
Seed=42.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n, p = 400, 5
X = rng.standard_normal((n, p))

# Propensity: mild confounding
ps = 1.0 / (1.0 + np.exp(-(0.5 * X[:, 0] - 0.3 * X[:, 1])))
W = (rng.uniform(size=n) < ps).astype(int)

# Mu: prognostic effect
mu = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]

# Tau: small heterogeneity
tau = 1.0 + 0.3 * X[:, 0]

# Outcome
y = mu + tau * W + 0.5 * rng.standard_normal(n)

cols = {"y": y, "W": W}
for j in range(p):
    cols[f"X{j+1}"] = X[:, j]
df = pd.DataFrame(cols)

import pathlib
out = pathlib.Path(__file__).parent / "bcf_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {out} (n={n}, true ATE ≈ 1.0)")
