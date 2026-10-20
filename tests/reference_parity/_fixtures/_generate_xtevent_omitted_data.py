"""Panel for the sp.xtevent omitted-endpoint case (Stata drops the left
endpoint, constant within unit once leads past the panel end drop out).

    python tests/reference_parity/_fixtures/_generate_xtevent_omitted_data.py
"""

import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261020)
N, T = 60, 10
g = rng.choice([4, 6, 8, 0], N)
i = np.repeat(np.arange(1, N + 1), T)
t = np.tile(np.arange(1, T + 1), N)
G = g[i - 1]
z = ((G > 0) & (t >= G)).astype(int)
y = 2.0 * z + 0.1 * t + rng.normal(size=N)[i - 1] + rng.normal(size=N * T)
out = pathlib.Path(__file__).with_name("xtevent_omitted.csv")
pd.DataFrame({"id": i, "t": t, "z": z, "y": np.round(y, 10)}).to_csv(out, index=False)
