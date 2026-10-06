"""Simulated series for the ARMA part of test_beveridge_nelson_parity.py.

An ARIMA(1,1,1) with drift, 240 observations:
``dy[t] - 0.4 = 0.6 (dy[t-1] - 0.4) + e[t] - 0.35 e[t-1]``, ``e ~ N(0, 0.8^2)``.
Writes ``beveridge_nelson_arma.csv`` with repr precision.

    python tests/reference_parity/_fixtures/_generate_beveridge_nelson_arma_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
rng = np.random.default_rng(2026100607)
n = 240
e = 0.8 * rng.normal(size=n + 100)
x = np.zeros(n + 100)
for t in range(1, n + 100):
    x[t] = 0.6 * x[t - 1] + e[t] - 0.35 * e[t - 1]
dy = 0.4 + x[101:]
y = np.concatenate([[100.0], 100.0 + np.cumsum(dy)])
lines = ["y"] + [repr(float(v)) for v in y]
(HERE / "beveridge_nelson_arma.csv").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
