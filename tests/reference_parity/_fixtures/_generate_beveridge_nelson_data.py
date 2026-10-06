"""Simulated series for tests/reference_parity/test_beveridge_nelson_parity.py.

``y`` is an ARIMA(2,1,0) with drift: its first difference is an AR(2) with
mean 0.4. Writes ``beveridge_nelson.csv`` next to this file.

    python tests/reference_parity/_fixtures/_generate_beveridge_nelson_data.py
"""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261008)
n, burn = 250, 100
e = rng.normal(size=n + burn)
dy = np.zeros(n + burn)
for t in range(2, n + burn):
    dy[t] = 0.4 * (1 - 0.5 + 0.2) + 0.5 * dy[t - 1] - 0.2 * dy[t - 2] + e[t]
y = 100.0 + np.cumsum(dy[burn:])

rows = ["y"] + [f"{float(v)!r}" for v in y]
out = Path(__file__).parent / "beveridge_nelson.csv"
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
