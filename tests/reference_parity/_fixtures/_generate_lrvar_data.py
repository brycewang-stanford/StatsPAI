"""Simulated series for tests/reference_parity/test_lrvar_parity.py.

``x`` is an ARMA(1,1) with a non-zero mean; ``y1``, ``y2`` are a bivariate
VAR(1) with correlated innovations. Writes ``lrvar.csv`` next to this file.

    python tests/reference_parity/_fixtures/_generate_lrvar_data.py
"""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261006)
n, burn = 300, 200
e = rng.normal(size=n + burn)
x = np.zeros(n + burn)
for t in range(1, n + burn):
    x[t] = 0.6 * x[t - 1] + e[t] + 0.3 * e[t - 1]
x = 2.0 + x[burn:]

a = np.array([[0.5, 0.2], [-0.1, 0.4]])
chol = np.array([[1.0, 0.0], [0.5, 0.8]])
u = rng.normal(size=(n + burn, 2)) @ chol.T
y = np.zeros((n + burn, 2))
for t in range(1, n + burn):
    y[t] = a @ y[t - 1] + u[t]
y = y[burn:] + np.array([1.0, -0.5])

rows = ["x,y1,y2"] + [
    f"{float(x[t])!r},{float(y[t, 0])!r},{float(y[t, 1])!r}" for t in range(n)
]
out = Path(__file__).parent / "lrvar.csv"
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
