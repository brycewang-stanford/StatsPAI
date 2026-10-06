"""Simulated three-variable VAR(2) for the impulse-response band tests.

Writes ``irf_bands.csv`` (columns t, y1, y2, y3), the input of
``_generate_irf_bands_Stata.do``. Seed 20261006.
"""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261006)
T, burn = 160, 200
A1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.3, -0.1], [0.0, 0.25, 0.4]])
A2 = np.array([[-0.1, 0.0, 0.05], [0.0, 0.15, 0.0], [0.1, 0.0, -0.2]])
P = np.array([[1.0, 0.0, 0.0], [0.4, 0.8, 0.0], [-0.3, 0.2, 0.6]])
c = np.array([0.2, -0.1, 0.3])
y = np.zeros((T + burn, 3))
for t in range(2, T + burn):
    y[t] = c + A1 @ y[t - 1] + A2 @ y[t - 2] + P @ rng.normal(size=3)
rows = ["t,y1,y2,y3"] + [
    f"{i + 1}," + ",".join(repr(float(v)) for v in row)
    for i, row in enumerate(y[burn:])
]
Path(__file__).with_name("irf_bands.csv").write_text(
    "\n".join(rows) + "\n", encoding="utf-8"
)
