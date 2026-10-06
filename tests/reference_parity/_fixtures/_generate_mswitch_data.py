"""Simulated series for tests/reference_parity/test_mswitch_parity.py.

``mswitch.csv``: 300 dates driven by a two-state Markov chain ``s2``
(``P(1->1) = 0.95``, ``P(2->2) = 0.90``) and a three-state chain ``s3``.

* ``y_mean``: mean -1 / 2, unit variance
* ``y_var``: mean 0 / 1.5, standard deviation 0.5 / 2
* ``y_x``: mean -1 / 2 plus ``0.8 * x``
* ``y_z``: mean 0 / 1 plus ``z`` times -1 / 1.5
* ``y_ar``: mean 0 / 3 plus AR(2) deviations (0.5, -0.2), sd 0.8
* ``y3``: mean -3 / 0 / 3 on the three-state chain, sd 0.8
* ``y3_ar``: the same means plus AR(1) deviations (0.4), sd 0.7

    python tests/reference_parity/_fixtures/_generate_mswitch_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261006)
T = 300


def chain(P: np.ndarray, n: int) -> np.ndarray:
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = rng.choice(len(P), p=P[s[t - 1]])
    return s


s2 = chain(np.array([[0.95, 0.05], [0.10, 0.90]]), T)
s3 = chain(np.array([[0.90, 0.07, 0.03], [0.05, 0.90, 0.05], [0.04, 0.06, 0.90]]), T)
x = np.zeros(T)
for t in range(1, T):
    x[t] = 0.6 * x[t - 1] + rng.normal()
z = rng.normal(size=T)
u = np.zeros(T)
e = rng.normal(scale=0.8, size=T)
for t in range(2, T):
    u[t] = 0.5 * u[t - 1] - 0.2 * u[t - 2] + e[t]

out = pd.DataFrame(
    {
        "t": np.arange(1, T + 1),
        "s2": s2 + 1,
        "s3": s3 + 1,
        "x": x,
        "z": z,
        "y_mean": np.array([-1.0, 2.0])[s2] + rng.normal(size=T),
        "y_var": np.array([0.0, 1.5])[s2]
        + np.array([0.5, 2.0])[s2] * rng.normal(size=T),
        "y_x": np.array([-1.0, 2.0])[s2] + 0.8 * x + rng.normal(size=T),
        "y_z": np.array([0.0, 1.0])[s2]
        + np.array([-1.0, 1.5])[s2] * z
        + rng.normal(size=T),
        "y_ar": np.array([0.0, 3.0])[s2] + u,
        "y3": np.array([-3.0, 0.0, 3.0])[s3] + rng.normal(scale=0.8, size=T),
    }
)
u3 = np.zeros(T)
e3 = rng.normal(scale=0.7, size=T)
for t in range(1, T):
    u3[t] = 0.4 * u3[t - 1] + e3[t]
out["y3_ar"] = np.array([-3.0, 0.0, 3.0])[s3] + u3
out.to_csv(Path(__file__).parent / "mswitch.csv", index=False, float_format="%.17g")
