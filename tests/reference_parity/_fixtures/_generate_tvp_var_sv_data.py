"""Simulated data for tests/reference_parity/test_tvp_var_sv_parity.py.

A two-variable VAR(1): the own-lag coefficient of ``x`` drifts from 0.2
to 0.8 over the estimation sample and the standard deviation of its
shock doubles halfway. The first 40 rows are the training sample. The
truth is written next to the data.

    python tests/reference_parity/_fixtures/_generate_tvp_var_sv_data.py
"""

from pathlib import Path

import numpy as np

N, TAU = 240, 40


def simulate(seed: int, n: int = N, tau: int = TAU) -> np.ndarray:
    """Columns: x, z, true own-lag coefficient of x, true shock sd of x."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 4))
    for t in range(n):
        u = min(max((t - tau) / (n - tau), 0.0), 1.0)
        a11 = 0.2 + 0.6 * u
        sd = 1.0 if t < tau + (n - tau) // 2 else 2.0
        e = rng.standard_normal(2)
        xl, zl = (out[t - 1, 0], out[t - 1, 1]) if t else (0.0, 0.0)
        shock = sd * e[0]
        out[t, 0] = a11 * xl + 0.1 * zl + shock
        out[t, 1] = 0.3 * zl + 0.5 * shock + 0.7 * e[1]
        out[t, 2], out[t, 3] = a11, sd
    return out


if __name__ == "__main__":
    data = simulate(20261006)
    lines = ["x,z,a11,sd_x"] + [",".join(repr(float(v)) for v in row) for row in data]
    path = Path(__file__).with_name("tvp_var_sv.csv")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
