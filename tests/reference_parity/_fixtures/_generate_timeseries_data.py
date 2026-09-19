"""Write the simulated inputs of tests/reference_parity/test_timeseries_R_parity.py.

Run before ``_generate_timeseries_R.R`` and ``_generate_timeseries_stata.do``.
Deterministic (fixed seeds); the CSVs are committed so the Python, R and
Stata sides read identical bytes (``%.17g`` round-trips every double).
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
FMT = "%.17g"

# ---- ts_coint.csv: three I(1) series driven by one drifting stochastic
# trend (cointegration rank 2), plus an independent random walk w for the
# Engle-Granger null case. Drift 0.1 makes the deterministic-term cases differ.
rng = np.random.default_rng(20260918)
T = 250
trend = np.cumsum(0.1 + rng.normal(0, 1, T))
u = np.zeros((T, 3))
for t in range(1, T):
    u[t] = 0.5 * u[t - 1] + rng.normal(0, 1, 3)
y1 = trend + u[:, 0]
y2 = 0.8 * trend + 1.0 + u[:, 1]
y3 = -0.5 * trend + u[:, 2]
w = np.cumsum(rng.normal(0, 1, T))
pd.DataFrame({"y1": y1, "y2": y2, "y3": y3, "w": w}).to_csv(
    OUT / "ts_coint.csv", index=False, float_format=FMT
)

# ---- ts_var.csv: stationary trivariate VAR(2) (for irf / Granger).
rng = np.random.default_rng(11)
T = 220
A1 = np.array([[0.5, 0.2, 0.0], [0.1, 0.4, 0.1], [0.0, 0.3, 0.3]])
A2 = np.array([[-0.2, 0.0, 0.1], [0.0, 0.1, 0.0], [0.1, 0.0, -0.1]])
L = np.array([[1.0, 0.0, 0.0], [0.4, 0.8, 0.0], [0.2, 0.3, 0.6]])
Y = np.zeros((T, 3))
for t in range(2, T):
    Y[t] = (
        np.array([0.3, -0.1, 0.2])
        + A1 @ Y[t - 1]
        + A2 @ Y[t - 2]
        + L @ rng.normal(0, 1, 3)
    )
pd.DataFrame(Y, columns=["gdp", "infl", "rate"]).to_csv(
    OUT / "ts_var.csv", index=False, float_format=FMT
)

# ---- ts_garch.csv: GARCH(1,1) with a non-zero mean.
rng = np.random.default_rng(5)
T = 1500
omega, a1, b1, mu = 0.05, 0.08, 0.88, 0.04
s2 = omega / (1 - a1 - b1)
r = np.empty(T)
eps_prev = 0.0
for t in range(T):
    s2 = omega + a1 * eps_prev**2 + b1 * s2
    eps_prev = np.sqrt(s2) * rng.normal()
    r[t] = mu + eps_prev
pd.DataFrame({"r": r}).to_csv(OUT / "ts_garch.csv", index=False, float_format=FMT)

# ---- ts_break.csv: regression y = a + b x + e with a break in both the
# intercept and the slope at t = 130 (of 250), a stable series ys, and a
# two-break mean-shift series ym.
rng = np.random.default_rng(3)
T = 250
x = rng.normal(0, 1, T)
e = rng.normal(0, 1, T)
post = np.arange(T) >= 130
y = 1.0 + 0.5 * x + post * (0.8 + 0.4 * x) + e
ys = 1.0 + 0.5 * x + rng.normal(0, 1, T)
# ym: mean shifts at t = 80 and t = 170 (two breaks) for the global
# Bai-Perron search.
tt = np.arange(T)
ym = 0.0 + 1.5 * (tt >= 80) - 2.5 * (tt >= 170) + rng.normal(0, 1, T)
pd.DataFrame({"t": np.arange(1, T + 1), "y": y, "ys": ys, "x": x, "ym": ym}).to_csv(
    OUT / "ts_break.csv", index=False, float_format=FMT
)

# ---- ts_its.csv: 72 monthly observations, intervention at row 40,
# level +2 and slope +0.08, AR(1) errors (so HAC matters).
rng = np.random.default_rng(8)
T = 72
e = np.zeros(T)
for t in range(1, T):
    e[t] = 0.5 * e[t - 1] + rng.normal(0, 0.6)
tt = np.arange(T)
D = (tt >= 40).astype(float)
y = 10 + 0.05 * tt + 2.0 * D + 0.08 * (tt - 40) * D + e
pd.DataFrame({"month": tt + 1, "y": y}).to_csv(
    OUT / "ts_its.csv", index=False, float_format=FMT
)

# ---- ts_panel.csv: balanced panel N = 20, T = 40 of AR(1) series with
# unit fixed effects, rho = 0.7 (stationary), used for the panel unit-root
# tests; plus yrw = a random-walk panel.
rng = np.random.default_rng(21)
N, T = 20, 40
rows = []
for i in range(N):
    a = rng.normal(0, 1)
    z = np.zeros(T)
    rw = np.zeros(T)
    z[0] = a / 0.3 + rng.normal()
    for t in range(1, T):
        z[t] = a + 0.7 * z[t - 1] + rng.normal()
        rw[t] = rw[t - 1] + rng.normal()
    for t in range(T):
        rows.append({"id": i + 1, "time": t + 1, "y": z[t], "yrw": rw[t]})
pd.DataFrame(rows).to_csv(OUT / "ts_panel.csv", index=False, float_format=FMT)
