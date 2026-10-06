"""Simulated series for tests/reference_parity/test_xcorr_parity.py.

``y`` is an AR(2); ``x`` is an AR(1) plus ``y`` two periods earlier, so
``y`` leads ``x``. Writes ``xcorr.csv`` next to this file.

    python tests/reference_parity/_fixtures/_generate_xcorr_data.py
"""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261007)
n, burn = 200, 100
e = rng.normal(size=(n + burn, 2))
y = np.zeros(n + burn)
w = np.zeros(n + burn)
for t in range(2, n + burn):
    y[t] = 0.9 * y[t - 1] - 0.3 * y[t - 2] + e[t, 0]
    w[t] = 0.5 * w[t - 1] + e[t, 1]
x = 1.0 + w
x[2:] += 0.6 * y[:-2]
x, y = x[burn:], 3.0 + y[burn:]

rows = ["t,x,y"] + [f"{t + 1},{float(x[t])!r},{float(y[t])!r}" for t in range(n)]
out = Path(__file__).parent / "xcorr.csv"
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
