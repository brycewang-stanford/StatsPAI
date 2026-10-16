"""Write ml_sandwich_data.csv for test_ml_sandwich_r_parity.py (seeded)."""

import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261017)
n = 400
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
w = rng.uniform(0.5, 2.0, size=n)
expo = rng.uniform(0.5, 3.0, size=n)
eta = -0.3 + 0.8 * x1 - 0.5 * x2
yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
# Overdispersed counts so the sandwich differs visibly from the model SE.
lam = np.exp(0.2 + 0.4 * x1 + 0.3 * x2) * expo * rng.gamma(2.0, 0.5, size=n)
c = rng.poisson(lam)
pd.DataFrame(
    {"yb": yb, "c": c, "x1": x1.round(10), "x2": x2.round(10),
     "w": w.round(10), "expo": expo.round(10)}
).to_csv(pathlib.Path(__file__).with_name("ml_sandwich_data.csv"), index=False)
