"""Generate the fixed DGP for the dml parity test.

Run once; commit the output ``dml_data.csv`` so both the R fixture
generator and the Python test use the exact same dataset.

DGP — semi-linear with confounding:
    e ~ X[:, :3] β_e + ε_e        # propensity component
    d = X[:, :5] β_d + ε_d         # continuous treatment
    d_bin = (e + ε_d > 0)          # binary treatment for IRM
    y = θ * d + X[:, :5] β_y + ε_y, θ = 0.5

n = 1000, p = 10 covariates, all i.i.d. standard normal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n, p = 1000, 10
X = rng.standard_normal((n, p))

beta_d = np.array([0.5, -0.4, 0.3, -0.2, 0.1] + [0.0] * (p - 5))
beta_y = np.array([0.4, 0.3, -0.2, 0.1, -0.1] + [0.0] * (p - 5))
beta_e = np.array([0.6, -0.5, 0.4] + [0.0] * (p - 3))

eps_d = rng.standard_normal(n)
eps_y = rng.standard_normal(n)

d = X @ beta_d + eps_d
e = X @ beta_e
prob = 1.0 / (1.0 + np.exp(-e))
d_bin = (rng.uniform(size=n) < prob).astype(int)

THETA = 0.5
y = THETA * d + X @ beta_y + eps_y

cols = {"y": y, "d": d, "d_bin": d_bin}
for j in range(p):
    cols[f"x{j+1}"] = X[:, j]

df = pd.DataFrame(cols)

import pathlib
out = pathlib.Path(__file__).parent / "dml_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {out}  (n={n}, true theta={THETA})")
