"""Zero-inflated count DGP for ``sp.zip_model`` / ``sp.zinb`` <-> R pscl.

The count component is an *overdispersed* gamma-Poisson (true theta=2) so
the ZINB negative-binomial dispersion is finite and well-identified (a
pure-Poisson count component would push theta -> infinity and make the
ZINB fit numerically unstable).  Structural zeros are added through a
logit inflation model on a separate regressor ``z``.

  count : log mu = 0.4 + 0.6 x1 - 0.3 x2,  NB shape theta = 2
  zero  : logit pi = -0.5 + 0.8 z

n = 800, seed = 20240608.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(20240608)
n = 800
x1 = rng.normal(size=n)
x2 = rng.binomial(1, 0.5, size=n).astype(float)
z = rng.normal(size=n)

# Overdispersed count component (gamma-Poisson, theta=2).
mu = np.exp(0.4 + 0.6 * x1 - 0.3 * x2)
theta = 2.0
lam = rng.gamma(shape=theta, scale=mu / theta)
counts = rng.poisson(lam)

# Structural-zero (inflation) component.
pi = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * z)))
structural_zero = (rng.uniform(size=n) < pi).astype(int)
y = np.where(structural_zero == 1, 0, counts)

df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "z": z})
out = pathlib.Path(__file__).parent / "zeroinfl_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {out}  (n={n}, frac_zero={(y == 0).mean():.3f}, NB theta=2)")
