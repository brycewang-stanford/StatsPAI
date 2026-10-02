"""Data for tests/reference_parity/test_textbook_methods_stata_parity.py.

Four small synthetic datasets, written once and committed, that the Stata
reference (_generate_textbook_methods_stata.do) and the test both read:

* textbook_ts.csv      120 periods: a regression with AR(1) errors, a
                       bivariate VAR(2), three cointegrated I(1) series
* textbook_panel.csv   40 units x up to 8 years, unbalanced, a regressor
                       correlated with the unit effect
* textbook_cs.csv      400 cross-section rows: heteroskedastic errors, an
                       endogenous regressor with two instruments, a binary
                       outcome, 25 clusters
* textbook_rcm.csv     12 units x 40 periods from a one-factor model; unit 1
                       is treated from period 31

Values are rounded to 8 decimals so both sides read the same doubles.

Run:  python _generate_textbook_methods_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
rng = np.random.default_rng(20261003)


def ar1(n, rho, scale=1.0, start=0.0):
    out = np.zeros(n)
    prev = start
    for t in range(n):
        prev = rho * prev + scale * rng.normal()
        out[t] = prev
    return out


# ------------------------------------------------------------------ time series
T = 120
x1 = ar1(T, 0.5)
x2 = ar1(T, 0.3) + 0.4 * x1
d = (np.arange(T) % 4 == 0).astype(float)
u = ar1(T, 0.6, 0.8) * (1 + 0.4 * np.abs(x1))
y = 1.0 + 0.8 * x1 - 0.5 * x2 + 0.6 * d + u
z = np.zeros((T, 2))
for t in range(2, T):
    z[t, 0] = 0.2 + 0.5 * z[t - 1, 0] - 0.2 * z[t - 2, 0] + 0.3 * z[t - 1, 1] + rng.normal()
    z[t, 1] = -0.1 + 0.2 * z[t - 1, 0] + 0.4 * z[t - 1, 1] + 0.1 * z[t - 2, 1] + rng.normal()
trend = np.cumsum(rng.normal(size=T)) + 0.05 * np.arange(T)
c1 = trend + ar1(T, 0.4, 0.5)
c2 = 0.6 * trend + ar1(T, 0.3, 0.5) + 1.0
c3 = np.cumsum(rng.normal(size=T)) - 0.5 * trend + ar1(T, 0.2, 0.4)
ts = pd.DataFrame(
    {"t": np.arange(1, T + 1), "y": y, "x1": x1, "x2": x2, "d": d,
     "z1": z[:, 0], "z2": z[:, 1], "c1": c1, "c2": c2, "c3": c3}
)
ts.round(8).to_csv(HERE / "textbook_ts.csv", index=False)

# ------------------------------------------------------------------------ panel
N, P = 40, 8
rows = []
for i in range(1, N + 1):
    a = rng.normal()
    e = ar1(P, 0.5, 0.7)
    for k in range(P):
        if rng.random() < 0.08 and k not in (0, 1):
            continue  # unbalanced: a few unit-years are missing
        px1 = 0.6 * a + rng.normal()
        px2 = rng.normal() + 0.1 * k
        pw = rng.normal()
        py = 0.5 + 1.0 * px1 - 0.7 * px2 + 0.3 * pw + a + e[k]
        rows.append((i, 2001 + k, py, px1, px2, pw))
panel = pd.DataFrame(rows, columns=["id", "year", "y", "x1", "x2", "w"])
panel.round(8).to_csv(HERE / "textbook_panel.csv", index=False)

# ---------------------------------------------------------------- cross-section
n = 400
g = rng.integers(1, 26, size=n)
ge = rng.normal(size=26)[g]
x1 = rng.normal(size=n)
x2 = rng.normal(size=n) + 0.3 * x1
dd = (rng.random(n) < 0.4).astype(float)
z1, z2 = rng.normal(size=n), rng.normal(size=n)
v = rng.normal(size=n)
endog = 0.6 * z1 + 0.4 * z2 + 0.3 * x1 + v
err = (0.7 * v + rng.normal(size=n)) * np.exp(0.3 * x1) + 0.5 * ge
yc = 2.0 + 0.5 * endog + 0.8 * x1 - 0.4 * x2 + 0.5 * dd + err
b = (0.3 + 0.8 * x1 - 0.5 * x2 + 0.6 * dd + rng.logistic(size=n) > 0).astype(float)
cs = pd.DataFrame(
    {"y": yc, "endog": endog, "x1": x1, "x2": x2, "d": dd, "z1": z1, "z2": z2,
     "b": b, "g": g}
)
cs.round(8).to_csv(HERE / "textbook_cs.csv", index=False)

# -------------------------------------------------------- regression control
J, TT = 12, 40
factor = np.cumsum(rng.normal(size=TT)) * 0.5
rows = []
for j in range(1, J + 1):
    loading = 0.4 + 0.15 * j
    series = 1.0 + loading * factor + ar1(TT, 0.3, 0.4)
    if j == 1:
        series[30:] += 1.5
    rows += [(j, t + 1, series[t]) for t in range(TT)]
rcm = pd.DataFrame(rows, columns=["unit", "t", "y"])
rcm.round(8).to_csv(HERE / "textbook_rcm.csv", index=False)
print("written:", [p.name for p in sorted(HERE.glob("textbook_*.csv"))])
