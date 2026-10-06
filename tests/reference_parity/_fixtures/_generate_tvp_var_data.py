"""Synthetic data for tests/reference_parity/test_tvp_var_parity.py.

A three-variable VAR(1) whose first own-lag coefficient drifts from 0.2 to
0.8 and whose error variances shift half way, 140 dates. Deterministic:

    python tests/reference_parity/_fixtures/_generate_tvp_var_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261006)
T, K = 140, 3
chol = np.array([[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [-0.2, 0.3, 0.8]])
y = np.zeros((T + 1, K))
for t in range(1, T + 1):
    A = np.array([[0.2, 0.1, 0.0], [0.0, 0.5, 0.2], [0.15, 0.0, 0.3]])
    A[0, 0] = 0.2 + 0.6 * t / T
    scale = 1.0 if t <= T // 2 else 1.6
    y[t] = np.array([0.3, 0.0, -0.2]) + A @ y[t - 1] + scale * chol @ rng.normal(size=K)

lines = ["y1,y2,y3"] + [",".join(repr(float(v)) for v in row) for row in y[1:]]
out = Path(__file__).parent / "tvp_var.csv"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
