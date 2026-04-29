"""Fixed staggered-DiD DGP for CS / did_imputation / gardner / wooldridge parity.

Three treatment cohorts:
  • Group g=2: treated from t=2 onwards
  • Group g=3: treated from t=3 onwards
  • Group g=0: never treated (control)

Heterogeneous, dynamic effects:
  τ(g, t) = max(0, t-g+1) * 1.0   (linear ramp post-treatment)

n_units=120 (40 per group), T=6 periods.  Seed=42.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
T = 6
groups_per_cohort = 40
groups = (
    [2] * groups_per_cohort   # treated at t=2
    + [3] * groups_per_cohort  # treated at t=3
    + [0] * groups_per_cohort  # never treated
)
n_units = len(groups)

rows = []
for i in range(n_units):
    g = groups[i]
    fe_i = rng.standard_normal()
    for t in range(1, T + 1):
        post = (g != 0) and (t >= g)
        tau = max(0, t - g + 1) if post else 0
        y = (
            fe_i
            + 0.3 * t  # time trend
            + tau  # heterogeneous treatment effect
            + 0.5 * rng.standard_normal()
        )
        rows.append({
            "id": i,
            "year": t,
            "g": g,
            "first_treat": g if g > 0 else 0,
            "post": int(post),
            "tau": tau,
            "y": y,
        })

df = pd.DataFrame(rows)

import pathlib
out = pathlib.Path(__file__).parent / "cs_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {out}  (n={len(df)}, units={n_units}, T={T})")
print(f"  cohort sizes: g=0:{(df.g==0).sum()//T}, g=2:{(df.g==2).sum()//T}, g=3:{(df.g==3).sum()//T}")
