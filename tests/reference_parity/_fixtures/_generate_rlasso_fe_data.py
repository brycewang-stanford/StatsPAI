"""Simulated panel for ``test_rlasso_fe_stata_parity.py``.

Writes ``rlasso_fe_panel.csv``: 70 units of unequal length (4 to 8
periods, so the panel is unbalanced), 30 candidate controls that load on the
unit effect, a treatment ``d`` and an outcome ``y`` that both contain unit
and period effects. Values are rounded to six decimals so Stata and Python
read the same numbers.

Run from the repository root:
    python tests/reference_parity/_fixtures/_generate_rlasso_fe_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261028)
n_units, p = 70, 30
rows = []
for i in range(n_units):
    T = int(rng.integers(4, 9))
    a_i = rng.normal(0, 1.5)
    for t in range(T):
        rows.append((i + 1, t + 1, a_i))
df = pd.DataFrame(rows, columns=["id", "year", "a"])
n = len(df)
year_fx = rng.normal(0, 1, 8)
X = rng.standard_normal((n, p)) + 0.8 * df["a"].to_numpy()[:, None]
# cluster-correlated noise in the controls
X += rng.standard_normal((n_units, p))[df["id"].to_numpy() - 1] * 0.5
yfe = year_fx[df["year"].to_numpy() - 1]
d = (
    0.9 * X[:, 0]
    - 0.7 * X[:, 1]
    + 0.5 * X[:, 2]
    + 0.25 * X[:, 7]
    + 2.0 * df["a"].to_numpy()
    + 0.6 * yfe
    + rng.standard_normal(n) * (0.8 + 0.4 * np.abs(X[:, 3]))
)
y = (
    0.5 * d
    + 1.0 * X[:, 0]
    + 0.6 * X[:, 4]
    - 0.4 * X[:, 5]
    + 0.3 * X[:, 7]
    - 1.5 * df["a"].to_numpy()
    + yfe
    + rng.standard_normal(n) * (1.0 + 0.5 * np.abs(X[:, 6]))
)
out = pd.DataFrame({"id": df["id"], "year": df["year"], "y": y, "d": d})
for j in range(p):
    out[f"x{j + 1}"] = X[:, j]
out = out.round(6)
path = Path(__file__).parent / "rlasso_fe_panel.csv"
out.to_csv(path, index=False, float_format="%.6f")
print(path, out.shape)
