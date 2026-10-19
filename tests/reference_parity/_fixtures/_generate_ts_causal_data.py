"""Synthetic inputs for tests/reference_parity/test_ts_causal_parity.py.

Writes ts_causal_series.csv (one series: outcome y, endogenous g driven by
the external instrument z, a third variable w, a state variable and its
indicator), ts_causal_panel.csv (30 units x 25 periods) and its unbalanced
version, and ts_causal_var.csv (three variables, 300 periods, for the
historical decomposition).
    python tests/reference_parity/_fixtures/_generate_ts_causal_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(20261019)
T = 240
z = rng.normal(size=T)
eps_g = rng.normal(size=T)
g = np.zeros(T)
y = np.zeros(T)
w = np.zeros(T)
for t in range(1, T):
    g[t] = 0.5 * g[t - 1] + 0.8 * z[t] + 0.6 * eps_g[t]
    y[t] = 0.6 * y[t - 1] + 0.4 * g[t] + 0.2 * w[t - 1] + rng.normal(0, 0.7)
    y[t] += 0.5 * eps_g[t]
    w[t] = 0.3 * w[t - 1] + 0.2 * y[t - 1] + rng.normal(0, 1.0)
state = np.sin(np.arange(T) / 12.0) + rng.normal(0, 0.3, size=T)
pd.DataFrame(
    {
        "t": np.arange(T),
        "y": y,
        "g": g,
        "w": w,
        "z": z,
        "state": state,
        "d": (state > 0).astype(float),
    }
).to_csv(HERE / "ts_causal_series.csv", index=False)

rows = []
for i in range(30):
    a = rng.normal()
    yy = xx = 0.0
    for t in range(25):
        zz = rng.normal()
        xx = 0.4 * xx + zz + rng.normal(0, 0.5)
        c = rng.normal()
        yy = a + 0.5 * yy + 0.7 * xx + 0.3 * c + rng.normal()
        rows.append((i + 1, t + 1, yy, xx, zz, c))
panel = pd.DataFrame(rows, columns=["id", "time", "y", "x", "z", "c"])
panel.to_csv(HERE / "ts_causal_panel.csv", index=False)
unbal = panel[~(panel["id"].isin([3, 7]) & (panel["time"] > 21))]
unbal.to_csv(HERE / "ts_causal_panel_unbal.csv", index=False)

H = np.array([[1.0, 0.3, 0.0], [0.5, 1.0, 0.2], [-0.4, 0.6, 1.0]])
A1 = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
n = 300
e = rng.normal(size=(n, 3))
v = np.zeros((n, 3))
for t in range(1, n):
    v[t] = 0.2 + A1 @ v[t - 1] + H @ e[t]
pd.DataFrame(v, columns=["i", "x", "p"]).to_csv(HERE / "ts_causal_var.csv", index=False)
