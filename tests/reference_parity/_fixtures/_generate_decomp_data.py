"""Write the simulated inputs of tests/reference_parity/test_decomp_R_parity.py.

Run before ``_generate_decomp_R.R``. Deterministic (fixed seeds); the CSVs
are committed so the R and Python sides read identical bytes.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent

# gap_closing: covariates differ by group, outcome nonlinear in x1 so the
# regression and reweighting counterfactuals genuinely differ.
rng = np.random.default_rng(7)
n = 800
g = rng.integers(0, 2, n)
x1 = rng.normal(0.5 * g, 1)
x2 = rng.binomial(1, 0.3 + 0.3 * g)
y = 1 + 0.7 * x1 + 0.5 * x2 + 0.3 * x1**2 + 0.4 * g + rng.normal(0, 1, n)
pd.DataFrame({"y": y, "group": g, "x1": x1, "x2": x2}).to_csv(
    OUT / "decomp_gap.csv", index=False, float_format="%.17g"
)

# yu_elwert: treatment effect heterogeneous in x1 and by group, so the
# selection component is non-zero.
rng = np.random.default_rng(3)
n = 2000
r = rng.integers(0, 2, n)
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
p = 1 / (1 + np.exp(-(0.5 * r + 0.3 * x1 - 0.2 * x2)))
t = (rng.uniform(size=n) < p).astype(int)
y = (
    1
    + 0.5 * x1
    + 0.3 * x2
    + (0.8 + 0.4 * r + 0.3 * x1) * t
    + 0.2 * r
    + rng.normal(size=n)
)
pd.DataFrame({"y": y, "t": t, "r": r, "x1": x1, "x2": x2}).to_csv(
    OUT / "decomp_ye.csv", index=False, float_format="%.17g"
)
print("wrote decomp_gap.csv, decomp_ye.csv")
