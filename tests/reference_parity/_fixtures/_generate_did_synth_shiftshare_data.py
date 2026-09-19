"""Write the simulated inputs of tests/reference_parity/test_did_synth_shiftshare_parity.py.

Run before ``_generate_did_synth_shiftshare_R.R`` and
``_generate_did_synth_shiftshare_stata.do``. Deterministic (fixed seed); the
CSVs are committed so the Python, R and Stata sides read identical bytes.

Design (a location x industry shift-share DGP):

* ``N = 240`` locations in 12 states, ``K = 20`` industries.
* Shares ``sh1..sh20`` are Dirichlet(0.5) draws scaled by a location-specific
  total in [0.55, 0.95], so the shares do NOT sum to one (the
  incomplete-shares case of Borusyak-Hull-Jaravel). ``ssum`` is that row sum,
  used as the sum-of-shares control.
* Industry shocks ``g`` ~ N(0, 1); the Bartik instrument ``B = sh @ g`` is
  written out as a column so every side uses the same bytes.
* Errors carry an industry component (``sh @ nu``) so observations sharing
  exposure are correlated -- the case AKM / BHJ inference is built for -- and
  the first-stage error is correlated with the structural error (``x`` is
  endogenous).
* A second shock-level column ``gcl`` assigns industries to 6 sector
  clusters (unused by the current tests but kept for the AKM
  ``sector_cvar`` option).
* ``bw`` is a positive location weight (Rotemberg weights with weights).

Column names avoid the substring ``sh`` outside the share columns because R
``ssaggregate`` selects wide-format share columns with ``grepl("sh", ...)``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent

rng = np.random.default_rng(20260918)
N, K, S = 240, 20, 12

state = np.repeat(np.arange(1, S + 1), N // S)
tot = rng.uniform(0.55, 0.95, N)
shares = rng.dirichlet(np.full(K, 0.5), size=N) * tot[:, None]
ssum = shares.sum(axis=1)
g = rng.normal(0.0, 1.0, K)
B = shares @ g

c1 = rng.normal(0.0, 1.0, N)
eta = rng.normal(0.0, 1.0, K)  # industry component of first-stage error
nu = rng.normal(0.0, 1.0, K)  # industry component of structural error
st_eff = rng.normal(0.0, 0.5, S)[state - 1]  # state component
v = shares @ eta + rng.normal(0.0, 0.5, N)
u = 0.8 * (shares @ nu) + 0.6 * v + st_eff + rng.normal(0.0, 1.0, N)
x = 0.2 + 1.0 * B + 0.5 * c1 + 0.8 * ssum + v
y = 1.0 + 1.5 * x + 0.3 * c1 - 0.4 * ssum + u
bw = rng.uniform(0.5, 2.0, N)

loc = pd.DataFrame(
    {
        "loc": np.arange(1, N + 1),
        "state": state,
        "y": y,
        "x": x,
        "B": B,
        "c1": c1,
        "ssum": ssum,
        "bw": bw,
    }
)
for k in range(K):
    loc[f"sh{k + 1}"] = shares[:, k]
loc.to_csv(OUT / "did_synth_shiftshare_loc.csv", index=False, float_format="%.17g")

pd.DataFrame(
    {
        "n": np.arange(1, K + 1),
        "g": g,
        "gcl": (np.arange(K) % 6) + 1,
    }
).to_csv(OUT / "did_synth_shiftshare_shocks.csv", index=False, float_format="%.17g")
