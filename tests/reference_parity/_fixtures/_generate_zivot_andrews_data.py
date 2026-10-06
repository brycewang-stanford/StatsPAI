"""Two simulated series for the Zivot-Andrews tests.

Writes ``zivot_andrews.csv``: ``rw`` is a random walk with drift (the
null), ``brk`` is trend stationary with a level shift and a change of
slope after observation 90. Seed 20261006.
"""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261006)
T = 180
rw = np.cumsum(0.2 + rng.normal(size=T))
t = np.arange(1, T + 1)
e = np.zeros(T)
for i in range(1, T):
    e[i] = 0.5 * e[i - 1] + rng.normal(scale=0.7)
brk = 1.0 + 0.10 * t + 4.0 * (t > 90) - 0.05 * np.where(t > 90, t - 90, 0) + e
rows = ["rw,brk"] + [f"{repr(float(a))},{repr(float(b))}" for a, b in zip(rw, brk)]
Path(__file__).with_name("zivot_andrews.csv").write_text(
    "\n".join(rows) + "\n", encoding="utf-8"
)
