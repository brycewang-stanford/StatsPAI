"""Generate the fixed DGP for the TMLE parity test.

Run once; commit the output ``tmle_data.csv`` so both the R fixture
generator (``_generate_tmle.R``) and the Python test use the exact
same dataset.

DGP — binary outcome, binary treatment, two confounders:
    w1 ~ N(0, 1),  w2 ~ Bernoulli(0.5)
    P(A=1 | W)    = expit(-0.3 + 0.8*w1 + 0.6*w2)
    P(Y=1 | A, W) = expit(-0.4 + 0.8*A + 0.9*w1 + 0.5*w2)

The estimand is the sample average risk difference (SATE):
    truth = mean_i[ P(Y=1|A=1, W_i) - P(Y=1|A=0, W_i) ]
which is exactly computable from the (w1, w2) draws — the Python
test recomputes it from the CSV columns with plain numpy arithmetic
(no statspai involvement).

Confounding is strong by design: w1 raises both treatment take-up and
the outcome, so the naive difference in means is biased upward by
> 4 sigma on this frozen draw (z ≈ 6.0) while a correctly specified
logistic Q-model keeps TMLE within its 4-sigma recovery band.

n = 2000, seed = 7321.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from scipy.special import expit

rng = np.random.default_rng(7321)
n = 2000

w1 = rng.normal(size=n)
w2 = rng.binomial(1, 0.5, n)

p_a = expit(-0.3 + 0.8 * w1 + 0.6 * w2)
a = rng.binomial(1, p_a)

p_y = expit(-0.4 + 0.8 * a + 0.9 * w1 + 0.5 * w2)
y = rng.binomial(1, p_y)

truth = float(
    np.mean(expit(-0.4 + 0.8 + 0.9 * w1 + 0.5 * w2) - expit(-0.4 + 0.9 * w1 + 0.5 * w2))
)

df = pd.DataFrame(
    {
        "y": y.astype(float),
        "a": a.astype(float),
        "w1": w1,
        "w2": w2.astype(float),
    }
)

out = pathlib.Path(__file__).parent / "tmle_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {out}  (n={n}, SATE truth={truth:.6f})")
