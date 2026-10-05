"""Synthetic clustered data for test_linear_model_extensions_parity.py.

60 clusters of unequal size (4 to 12 rows), six regressors, a sampling
weight, and four outcomes: a positive continuous one, a binary one, a
count, and a survival time rounded to whole days (so event times are
heavily tied) with censoring. Nothing here comes from a real dataset.

    python tests/reference_parity/_fixtures/_generate_linear_model_extensions_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261005)
sizes = rng.integers(4, 13, size=60)
ids = np.repeat(np.arange(1, 61), sizes)
n = len(ids)
t = np.concatenate([np.arange(1, m + 1) for m in sizes])
u = rng.normal(scale=0.7, size=60)[ids - 1]
x = rng.normal(size=(n, 6))
x[:, 1] = 0.6 * x[:, 0] + 0.8 * x[:, 1]
x[:, 3] = (x[:, 3] > 0).astype(float)
treat = (rng.uniform(size=60) < 0.5).astype(float)[ids - 1]
lin = 0.5 * x[:, 0] - 0.4 * x[:, 1] + 0.3 * x[:, 3] + 0.25 * treat
e = np.zeros(n)
for g in np.unique(ids):
    k = np.flatnonzero(ids == g)
    eps = rng.normal(size=len(k))
    for j in range(1, len(k)):
        eps[j] = 0.5 * eps[j - 1] + np.sqrt(0.75) * eps[j]
    e[k] = eps
y = np.exp(1.0 + 0.4 * lin + 0.3 * u + 0.35 * e * (1 + 0.5 * np.abs(x[:, 0])))
d = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-0.3 + lin + u)))).astype(int)
c = rng.poisson(np.exp(0.2 + 0.5 * lin + 0.6 * u))
raw = rng.exponential(scale=20 * np.exp(-0.6 * lin - 0.5 * u))
cens = rng.exponential(scale=60, size=n)
time = np.ceil(np.minimum(raw, cens)).clip(1, 90)
event = ((raw <= cens) & (raw <= 90)).astype(int)
w = np.round(rng.uniform(0.5, 4.0, size=n), 3)
df = pd.DataFrame(
    {
        "id": ids,
        "t": t,
        "treat": treat.astype(int),
        "x1": x[:, 0],
        "x2": x[:, 1],
        "x3": x[:, 2],
        "x4": x[:, 3].astype(int),
        "x5": x[:, 4],
        "x6": x[:, 5],
        "w": w,
        "y": y,
        "d": d,
        "c": c,
        "time": time.astype(int),
        "event": event,
        "site": (ids % 3) + 1,
    }
)
# scramble the row order: nothing downstream may rely on sorted clusters
df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
out = Path(__file__).with_name("linear_model_extensions.csv")
df.to_csv(out, index=False, float_format="%.10g")
print(
    out, df.shape, int(df.event.sum()), "events,", df.time.nunique(), "distinct times"
)
