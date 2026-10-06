"""Simulated four-variable system with two cointegrating relations.

Writes ``johansen_lrtest.csv``, the input of
``_generate_johansen_lrtest_R.R``. Two random-walk trends drive
``(c, i, y, rr)``: ``c - y`` and ``i - y`` are stationary. Seed 20261006.
"""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261006)
T = 240
trend = np.cumsum(0.4 + rng.normal(size=T))
rate = np.cumsum(0.5 * rng.normal(size=T))


def ar1(scale: float, phi: float) -> np.ndarray:
    e = rng.normal(scale=scale, size=T)
    out = np.zeros(T)
    for t in range(1, T):
        out[t] = phi * out[t - 1] + e[t]
    return out


y = trend + ar1(0.6, 0.3)
c = -0.5 + trend + 0.2 * rate + ar1(0.5, 0.5)
i = -1.5 + trend - 0.4 * rate + ar1(1.0, 0.6)
rows = ["c,i,y,rr"] + [
    ",".join(repr(float(v)) for v in row) for row in zip(c, i, y, rate)
]
Path(__file__).with_name("johansen_lrtest.csv").write_text(
    "\n".join(rows) + "\n", encoding="utf-8"
)
