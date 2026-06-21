"""Generate the fixed DGP for the IPW parity test.

Run once; commit the output ``ipw_data.csv`` so both the base-R fixture
generator (``_generate_ipw.R``) and the Python test
(``test_ipw_parity.py``) consume the exact same dataset.

DGP — selection on two observables with homogeneous effect:
    x1 ~ N(0, 1)            (continuous confounder)
    x2 ~ Bernoulli(0.5)     (binary confounder)
    P(t=1 | x) = expit(-0.2 + 0.7*x1 - 0.5*x2)   # well-overlapped
    y = 1 + 2.0*t + 0.8*x1 + 0.5*x2 + N(0, 1)    # true effect = 2.0

n = 800.  Propensities lie comfortably inside (0.02, 0.98), so no
trimming is involved and the unpenalized-logit Hajek estimate is a
smooth functional of the data — ideal for tight cross-language parity.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(20260612)
n = 800

x1 = rng.standard_normal(n)
x2 = rng.binomial(1, 0.5, n).astype(float)

lin = -0.2 + 0.7 * x1 - 0.5 * x2
p = 1.0 / (1.0 + np.exp(-lin))
t = (rng.uniform(size=n) < p).astype(int)

TAU = 2.0
y = 1.0 + TAU * t + 0.8 * x1 + 0.5 * x2 + rng.standard_normal(n)

df = pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})

out = pathlib.Path(__file__).parent / "ipw_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {out}  (n={n}, true tau={TAU}, n_treated={t.sum()})")
