"""Generate the fixed IV-DML DGP for the PLIV / IIVM parity tests.

Run once; commit the output ``dml_iv_data.csv`` so the parity test and any
future R-side check consume the exact same dataset. Complements
``dml_data.csv`` (PLR / IRM), which has no instrument.

Two instrumented designs share the same covariate block ``X`` (n=2000,
p=10, i.i.d. standard normal) but differ in the treatment / instrument /
outcome triple:

PLIV — partially linear IV (continuous treatment, continuous instrument):
    z_c ~ N(0, 1)                         # exogenous instrument
    xi  ~ N(0, 1)                         # common confounder -> endogeneity
    d_c = 1.2 z_c + X β_d + xi + 0.3 ε_d  # strong first stage
    y_pliv = θ d_c + X β_y + xi + 0.3 ε_y, θ = 0.5
  OLS of y on d is biased through xi; z_c is excluded from the y equation
  and shifts d, so the PLIV moment identifies θ.

IIVM — interactive IV / LATE (binary treatment, binary instrument):
    z_b ~ Bernoulli(0.5)                  # randomised instrument
    idx = -0.2 + 1.6 z_b + 0.5 X β_d + 0.8 xi + 0.5 ν
    d_b = 1{idx > 0}                       # imperfect, X/xi-dependent compliance
    y_iivm = θ d_b + X β_y + xi + 0.3 ε_y, θ = 0.5 (homogeneous -> LATE≈θ)
  The instrument strongly shifts compliance (P(D=1|Z=1) ≫ P(D=1|Z=0)) while
  xi confounds D and Y, so the LATE is identified only through z_b.

The seeds below are the ones under which ``sp.dml`` and ``doubleml-for-py``
were verified to agree (PLIV to machine precision, IIVM to ~1.5e-3 on the
coefficient); see ``tests/external_parity/test_dml_python_parity.py``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

n, p = 2000, 10
beta_d = np.array([0.5, -0.4, 0.3, -0.2, 0.1] + [0.0] * (p - 5))
beta_y = np.array([0.4, 0.3, -0.2, 0.1, -0.1] + [0.0] * (p - 5))
THETA = 0.5

# --- PLIV design (seed 7) ---
rng = np.random.default_rng(7)
X = rng.standard_normal((n, p))
z_c = rng.standard_normal(n)
xi = rng.standard_normal(n)
d_c = 1.2 * z_c + X @ beta_d + 1.0 * xi + 0.3 * rng.standard_normal(n)
y_pliv = THETA * d_c + X @ beta_y + 1.0 * xi + 0.3 * rng.standard_normal(n)

# --- IIVM design (seed 11), shares the same X column block ---
rng2 = np.random.default_rng(11)
z_b = rng2.binomial(1, 0.5, n)
xi2 = rng2.standard_normal(n)
idx = -0.2 + 1.6 * z_b + 0.5 * (X @ beta_d) + 0.8 * xi2 + 0.5 * rng2.standard_normal(n)
d_b = (idx > 0).astype(int)
y_iivm = THETA * d_b + X @ beta_y + 1.0 * xi2 + 0.3 * rng2.standard_normal(n)

cols = {f"x{j + 1}": X[:, j] for j in range(p)}
cols.update(
    z_c=z_c,
    d_c=d_c,
    y_pliv=y_pliv,  # PLIV block
    z_b=z_b,
    d_b=d_b,
    y_iivm=y_iivm,  # IIVM block
)
df = pd.DataFrame(cols)

out = pathlib.Path(__file__).parent / "dml_iv_data.csv"
df.to_csv(out, index=False)
print(
    f"Wrote {out}  (n={n}, p={p}, true theta={THETA}; "
    f"PLIV first stage corr(z_c,d_c)={np.corrcoef(z_c, d_c)[0, 1]:.3f}, "
    f"IIVM compliance gap={d_b[z_b == 1].mean() - d_b[z_b == 0].mean():.3f})"
)
