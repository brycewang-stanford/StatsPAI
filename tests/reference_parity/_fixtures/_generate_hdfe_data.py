"""Fixed panel DGP for hdfe ↔ R fixest parity.

Two-way fixed-effects with two regressors and clustered errors.
n_id=80, T=10, n=800.  Seed=42.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n_id, T = 80, 10
ids = np.repeat(np.arange(n_id), T)
years = np.tile(np.arange(T), n_id)
n = len(ids)

alpha_i = rng.standard_normal(n_id)[ids]
mu_t = rng.standard_normal(T)[years]

x1 = rng.standard_normal(n) + 0.3 * alpha_i
x2 = rng.standard_normal(n) + 0.2 * mu_t
y = 0.5 * x1 - 0.3 * x2 + alpha_i + mu_t + 0.5 * rng.standard_normal(n)

df = pd.DataFrame({"id": ids, "year": years, "y": y, "x1": x1, "x2": x2})
import pathlib
out = pathlib.Path(__file__).parent / "hdfe_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {out}  (n={n}, beta_x1=0.5, beta_x2=-0.3)")
