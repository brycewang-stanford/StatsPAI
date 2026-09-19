"""Write the simulated inputs of tests/reference_parity/test_inference_sens_*_parity.py.

Run before ``_generate_inference_sens_R.R`` / ``_generate_inference_sens_stata.do``.
Deterministic (fixed seeds); the CSVs are committed so the R, Stata and Python
sides read identical bytes.

Designs are deliberately small where the reference is a permutation or
wild-bootstrap distribution: with G = 12 bootstrap clusters the Rademacher
grid has 2^12 = 4096 points, and every randomization design below has at most
4900 assignments, so both sides enumerate the *same* finite distribution and
the comparison is deterministic rather than Monte-Carlo.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. Clustered regression data (cluster SEs, jackknife, wild bootstrap, Oster).
#    Unequal cluster sizes, so the delete-one-cluster jackknife centred at the
#    replicate mean differs from the one centred at the full-sample estimate,
#    and the cluster-share centring of the wild bootstrap is not trivial.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20260918)
G = 12
sizes = np.array([8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 42])  # n = 240
s12 = np.repeat(np.arange(1, G + 1), sizes)
n = s12.size
g6 = (s12 + 1) // 2  # six coarse clusters, each the union of two s12 clusters
yr = rng.integers(1, 9, n)  # a second, crossed cluster dimension (8 levels)
d = (s12 <= 5).astype(int)  # treatment assigned at the s12 level (5 treated)
x1 = rng.normal(size=n) + 0.4 * rng.normal(size=G)[s12 - 1]
x2 = rng.normal(size=n)
# A continuous "treatment" correlated with the controls, for Oster (2019).
t = 0.6 * x1 - 0.3 * x2 + rng.normal(size=n)
u_c = rng.normal(0, 0.7, G)[s12 - 1]
u_y = rng.normal(0, 0.5, 8)[yr - 1]
y = 1.0 + 0.35 * d + 0.5 * x1 - 0.25 * x2 + 0.4 * t + u_c + u_y + rng.normal(size=n)
pd.DataFrame(
    {"y": y, "d": d, "t": t, "x1": x1, "x2": x2, "s12": s12, "g6": g6, "yr": yr}
).to_csv(OUT / "inference_sens_reg.csv", index=False, float_format="%.17g")

# ---------------------------------------------------------------------------
# 2. Randomization-inference designs (exact enumeration on both sides).
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
# (a) complete randomization, n = 12, n1 = 6: C(12, 6) = 924 assignments.
n_a = 12
d_a = np.zeros(n_a, dtype=int)
d_a[rng.choice(n_a, 6, replace=False)] = 1
y_a = 1.0 + 0.8 * d_a + rng.normal(size=n_a)
pd.DataFrame({"y": y_a, "d": d_a}).to_csv(
    OUT / "inference_sens_ri_simple.csv", index=False, float_format="%.17g"
)
# (b) cluster randomization, 8 clusters of 3, 4 treated: C(8, 4) = 70.
cl_b = np.repeat(np.arange(1, 9), 3)
treated_cl = rng.choice(np.arange(1, 9), 4, replace=False)
d_b = np.isin(cl_b, treated_cl).astype(int)
y_b = 0.5 + 0.9 * d_b + rng.normal(0, 0.6, 8)[cl_b - 1] + rng.normal(size=cl_b.size)
pd.DataFrame({"y": y_b, "d": d_b, "cl": cl_b}).to_csv(
    OUT / "inference_sens_ri_cluster.csv", index=False, float_format="%.17g"
)
# (c) stratified (block) randomization, 2 strata of 8, 4 treated in each:
#     C(8, 4)^2 = 4900 assignments.
st_c = np.repeat([1, 2], 8)
d_c = np.zeros(16, dtype=int)
for s in (1, 2):
    idx = np.where(st_c == s)[0]
    d_c[rng.choice(idx, 4, replace=False)] = 1
y_c = 0.3 * st_c + 0.7 * d_c + rng.normal(size=16)
pd.DataFrame({"y": y_c, "d": d_c, "st": st_c}).to_csv(
    OUT / "inference_sens_ri_strat.csv", index=False, float_format="%.17g"
)

# ---------------------------------------------------------------------------
# 3. Matched pairs for Rosenbaum bounds. Outcomes are integer scores, so the
#    paired differences are exact integers containing ties AND zeros -- the
#    two places where Wilcoxon signed-rank conventions diverge -- with no
#    floating-point noise that could split a tie on one side only.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(11)
n_pairs = 40
ctrl = rng.integers(20, 41, n_pairs).astype(float)
trt = ctrl + np.round(rng.normal(2.0, 3.0, n_pairs))
trt[[3, 17, 29]] = ctrl[[3, 17, 29]]  # at least three exact zeros
pd.DataFrame(
    {"pair": np.arange(1, n_pairs + 1), "y_t": trt, "y_c": ctrl, "diff": trt - ctrl}
).to_csv(OUT / "inference_sens_pairs.csv", index=False, float_format="%.17g")


# ---------------------------------------------------------------------------
# 4. The full assignment set of each randomization design, one column per
#    assignment, for Stata ``ritest ..., samplingsourcefile()``. Built here
#    with itertools (independently of StatsPAI's own enumeration); R ``ri2``
#    enumerates the same sets itself through randomizr.
# ---------------------------------------------------------------------------
from itertools import combinations, product  # noqa: E402


def _write_perms(name, assignments):
    mat = np.asarray(assignments, dtype=int).T  # units x assignments
    cols = {"id": np.arange(1, mat.shape[0] + 1)}
    cols.update({f"d{j + 1}": mat[:, j] for j in range(mat.shape[1])})
    pd.DataFrame(cols).to_csv(OUT / f"inference_sens_ri_{name}_perms.csv", index=False)


def _complete(n, m):
    for c in combinations(range(n), m):
        a = np.zeros(n, dtype=int)
        a[list(c)] = 1
        yield a


_write_perms("simple", list(_complete(n_a, 6)))
_write_perms("cluster", [a[cl_b - 1] for a in _complete(8, 4)])
_strat_sets = [list(_complete(8, 4)), list(_complete(8, 4))]
_write_perms("strat", [np.concatenate([a1, a2]) for a1, a2 in product(*_strat_sets)])
