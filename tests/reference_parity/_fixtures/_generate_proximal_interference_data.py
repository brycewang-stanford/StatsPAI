"""Data for the proximal and interference R / Stata parity tests.

Run once from the repository root; the CSVs are committed so the R
generator (``_generate_proximal_interference_R.R``), the Stata generator
(``_generate_proximal_interference_Stata.do``) and the Python tests
(``test_proximal_r_parity.py``, ``test_interference_r_parity.py``) read the
same bytes. Values are rounded to 10 decimals, so every parser returns the
correctly rounded double.

Files
-----
``proximal_ref_data.csv``
    n = 600. Unmeasured U; binary D depends on U and x1; two treatment
    proxies z1, z2 and two outcome proxies w1, w2 of U; a negative-control
    outcome ``nco`` and exposure ``nce``; heteroskedastic errors so the
    HC1 columns differ from the classical ones.
``interference_unit_data.csv``
    n = 500 units: own treatment, inward and outward exposure shares, one
    covariate (``sp.inward_outward_spillover``).
``interference_cluster_data.csv``
    70 clusters of 3 to 14 members, 35 matched pairs; cluster-level
    treatment (one treated cluster per pair) and a neighbour treated share
    (``sp.cluster_cross_interference``, ``sp.cluster_matched_pair``).
``peer_network_data.csv`` / ``peer_network_edges.csv``
    160 nodes on a sparse random graph with two isolates (rows of W that
    sum to zero stay zero); outcome from the linear-in-means model
    (``sp.peer_effects``). Edges are undirected, listed once, 1-based.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent
rng = np.random.default_rng(20261106)


def _write(df: pd.DataFrame, name: str) -> None:
    df.round(10).to_csv(OUT / name, index=False, float_format="%.10f")


# ---------------------------------------------------------------- proximal
n = 600
u = rng.standard_normal(n)
x1 = rng.standard_normal(n)
x2 = rng.binomial(1, 0.4, n).astype(float)
d = (0.8 * u + 0.4 * x1 + rng.standard_normal(n) > 0).astype(float)
z1 = 0.9 * u + 0.3 * x1 + rng.standard_normal(n)
z2 = 0.6 * u - 0.2 * x2 + rng.standard_normal(n)
w1 = 0.8 * u + 0.2 * x1 + rng.standard_normal(n)
w2 = 0.5 * u + 0.4 * x2 + rng.standard_normal(n)
het = 0.5 + 0.5 * np.abs(x1)
y = 1.0 + 1.5 * d + 1.2 * u + 0.5 * x1 - 0.3 * x2 + het * rng.standard_normal(n)
nco = 0.7 * u + 0.2 * x1 + het * rng.standard_normal(n)
nce = 0.6 * u - 0.1 * x2 + rng.standard_normal(n)
_write(
    pd.DataFrame(
        {
            "y": y,
            "d": d,
            "z1": z1,
            "z2": z2,
            "w1": w1,
            "w2": w2,
            "x1": x1,
            "x2": x2,
            "nco": nco,
            "nce": nce,
        }
    ),
    "proximal_ref_data.csv",
)

# ------------------------------------------------- unit-level interference
n = 500
dd = rng.binomial(1, 0.5, n).astype(float)
e_in = rng.beta(2, 3, n)
e_out = np.clip(0.5 * e_in + 0.5 * rng.beta(2, 2, n), 0, 1)
xc = rng.standard_normal(n)
yy = 0.5 + 1.0 * dd + 0.8 * e_in + 0.3 * e_out + 0.4 * xc + rng.standard_normal(n)
_write(
    pd.DataFrame({"y": yy, "d": dd, "e_in": e_in, "e_out": e_out, "x": xc}),
    "interference_unit_data.csv",
)

# ---------------------------------------------- cluster-level interference
rows = []
for p in range(35):
    base = rng.normal(0, 1)
    treated_arm = int(rng.integers(0, 2))
    for arm in (0, 1):
        cid = 2 * p + arm + 1
        treat = int(arm == treated_arm)
        share = float(rng.uniform())
        size = int(rng.integers(3, 15))
        for _ in range(size):
            rows.append(
                {
                    "y": base + 0.7 * treat + 0.5 * share + rng.normal(0, 1),
                    "cluster": cid,
                    "pair": p + 1,
                    "treat": treat,
                    "nshare": share,
                }
            )
_write(pd.DataFrame(rows), "interference_cluster_data.csv")

# ------------------------------------------------------------ peer effects
n = 160
A = np.zeros((n, n))
for i in range(2, n):  # nodes 0 and 1 stay isolated
    for j in range(i + 1, n):
        if rng.uniform() < 0.035:
            A[i, j] = A[j, i] = 1.0
deg = A.sum(1)
Wn = A / np.where(deg == 0, 1.0, deg)[:, None]
xa = rng.standard_normal(n)
xb = rng.binomial(1, 0.5, n).astype(float)
X = np.column_stack([xa, xb])
eps = (0.6 + 0.4 * xb) * rng.standard_normal(n)
lin = 0.5 + X @ np.array([1.0, -0.5]) + Wn @ X @ np.array([0.6, 0.3]) + eps
yp = np.linalg.solve(np.eye(n) - 0.35 * Wn, lin)
_write(pd.DataFrame({"y": yp, "xa": xa, "xb": xb}), "peer_network_data.csv")
iu, ju = np.nonzero(np.triu(A))
pd.DataFrame({"from": iu + 1, "to": ju + 1}).to_csv(
    OUT / "peer_network_edges.csv", index=False
)
