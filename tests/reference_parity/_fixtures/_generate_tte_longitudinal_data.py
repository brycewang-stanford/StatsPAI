"""Data for ``test_tte_longitudinal_r_parity.py``.

Run once from the repo root; the CSVs are committed and read by both
``_generate_tte_longitudinal_R.R`` and the test.

* ``tte_point.csv``: one row per person for ``sp.target_trial_emulate``.
  Confounded baseline treatment ``A`` (logit in ``x`` and ``ldl``),
  continuous outcome, eligibility ``age >= 45 and ldl > 110``.
* ``long_ipw.csv``: 400 people x 3 periods with a treatment constant within
  person and a baseline confounder ``B``; the IPW path of
  ``sp.longitudinal_analyze`` (no time-varying confounders).
* ``long_tv.csv``: 500 people x 3 periods with time-varying treatment and
  a time-varying confounder ``L`` affected by past treatment; the ICE
  g-formula path. The outcome is repeated on every row of a person.
* ``ccw_panel.csv``: 400 people x 5 periods for ``sp.clone_censor_weight``;
  treatment depends on the current ``L`` and on past treatment, so clones
  of both strategies deviate at every time, including time zero.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent
rng = np.random.default_rng(20261106)


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---- target_trial_emulate ---------------------------------------------------
n = 600
age = rng.integers(35, 76, n)
ldl = np.round(rng.normal(140, 25, n), 1)
x = rng.normal(size=n)
A = (rng.uniform(size=n) < expit(-0.3 + 0.8 * x + 0.02 * (ldl - 140))).astype(int)
Y = 1 + 1.5 * A + 0.7 * x + 0.02 * (ldl - 140) + rng.normal(size=n)
pd.DataFrame(
    {"id": np.arange(n), "age": age, "ldl": ldl, "x": x, "A": A, "Y": Y}
).to_csv(OUT / "tte_point.csv", index=False)

# ---- longitudinal, IPW path -------------------------------------------------
n, K = 400, 3
B = rng.normal(size=n)
Ai = (rng.uniform(size=n) < expit(0.8 * B)).astype(int)
Yi = 1 + 2 * Ai + B + rng.normal(size=n)
rows = [
    {"id": i, "time": t, "A": Ai[i], "B": B[i], "Y": Yi[i]}
    for i in range(n)
    for t in range(K)
]
pd.DataFrame(rows).to_csv(OUT / "long_ipw.csv", index=False)

# ---- longitudinal, g-formula path ------------------------------------------
n = 500
B = rng.normal(size=n)
L = np.zeros((n, K))
Atv = np.zeros((n, K), dtype=int)
L[:, 0] = 0.3 * B + rng.normal(size=n)
Atv[:, 0] = rng.uniform(size=n) < expit(0.5 * L[:, 0])
for t in (1, 2):
    L[:, t] = 0.5 * L[:, t - 1] + 0.6 * Atv[:, t - 1] + rng.normal(size=n)
    Atv[:, t] = rng.uniform(size=n) < expit(0.5 * L[:, t] + 0.3 * Atv[:, t - 1])
Ytv = (
    1
    + 0.5 * Atv[:, 0]
    + 0.7 * Atv[:, 1]
    + 0.9 * Atv[:, 2]
    + 0.4 * L[:, 0]
    + 0.5 * L[:, 1]
    + 0.6 * L[:, 2]
    + 0.3 * B
    + rng.normal(scale=1.5, size=n)
)
rows = [
    {"id": i, "time": t, "A": Atv[i, t], "L": L[i, t], "B": B[i], "Y": Ytv[i]}
    for i in range(n)
    for t in range(K)
]
pd.DataFrame(rows).to_csv(OUT / "long_tv.csv", index=False)

# ---- clone-censor-weight ----------------------------------------------------
n, K = 400, 5
age = np.round(rng.normal(55, 8, n), 1)
rows = []
for i in range(n):
    prev, lt = 0, rng.normal()
    for t in range(K):
        lt = 0.6 * lt + 0.4 * prev + rng.normal()
        a = int(
            rng.uniform() < expit(-0.4 + 0.8 * lt + 2.0 * prev + 0.03 * (age[i] - 55))
        )
        rows.append({"id": i, "time": t, "treat": a, "L": lt, "age": age[i]})
        prev = a
pd.DataFrame(rows).to_csv(OUT / "ccw_panel.csv", index=False)
print("wrote tte_point.csv, long_ipw.csv, long_tv.csv, ccw_panel.csv")
