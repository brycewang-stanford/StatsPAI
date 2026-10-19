"""Data for the sp.westfall_young / Stata wyoung fixture.

Two small randomized designs whose re-randomizations can be enumerated:
``wyoung_strata.csv`` (two strata of 8 units, 4 treated in each: 4,900
assignments) and ``wyoung_cluster.csv`` (12 clusters of 5, 6 treated
clusters: 924 assignments). Three outcomes each, one covariate ``x``.

    python tests/reference_parity/_fixtures/_generate_wyoung_data.py
"""

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
rng = np.random.default_rng(20261019)

rows = []
for s in range(2):
    z = rng.permutation([1] * 4 + [0] * 4)
    for u in range(8):
        x = rng.normal()
        rows.append(
            dict(
                stratum=s,
                treat=z[u],
                x=round(x, 6),
                y1=round(0.9 * z[u] + x + rng.normal(), 6),
                y2=round(0.4 * z[u] + 0.5 * x + rng.normal(), 6),
                y3=round(rng.normal(), 6),
            )
        )
pd.DataFrame(rows).to_csv(HERE / "wyoung_strata.csv", index=False)

rows = []
cl_t = rng.permutation([1] * 6 + [0] * 6)
for g in range(12):
    ce = rng.normal()
    for u in range(5):
        x = rng.normal()
        rows.append(
            dict(
                cl=g,
                treat=cl_t[g],
                x=round(x, 6),
                y1=round(0.8 * cl_t[g] + x + ce + rng.normal(), 6),
                y2=round(0.3 * cl_t[g] + ce + rng.normal(), 6),
                y3=round(ce + rng.normal(), 6),
            )
        )
pd.DataFrame(rows).to_csv(HERE / "wyoung_cluster.csv", index=False)
