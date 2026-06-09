"""Fixed DGPs for ``sp`` <-> R parity on count / quantile / Tobit models.

One shared design matrix (two regressors, n=701 — odd, to keep the median
LP solution unique so quantreg's simplex and sp's interior-point land on
the same vertex) feeds three outcomes:

  * ``yc``  -- overdispersed counts (gamma-Poisson, true theta=2) for
               ``sp.poisson`` / ``sp.nbreg``;
  * ``yl``  -- continuous, for ``sp.qreg`` (tau = 0.25 / 0.5 / 0.75);
  * ``yt``  -- ``yl`` left-censored at 0, for ``sp.tobit``.

Seed = 20240608.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(20240608)
n = 701
x1 = rng.normal(size=n)
x2 = rng.binomial(1, 0.5, size=n).astype(float)

# Count outcome: gamma-Poisson mixture with mean mu and shape theta=2
# (var = mu + mu^2/theta) so the negative-binomial dispersion is finite
# and glm.nb / sp.nbreg are well-identified away from the Poisson limit.
mu = np.exp(0.3 + 0.5 * x1 - 0.4 * x2)
theta = 2.0
lam = rng.gamma(shape=theta, scale=mu / theta)
yc = rng.poisson(lam)

# Continuous outcome for quantile regression / Tobit.
yl = 1.0 + 0.8 * x1 - 0.5 * x2 + rng.normal(scale=1.0, size=n)
yt = np.maximum(yl, 0.0)  # left-censored at 0

df = pd.DataFrame({"yc": yc, "yl": yl, "yt": yt, "x1": x1, "x2": x2})
out = pathlib.Path(__file__).parent / "count_quantile_data.csv"
df.to_csv(out, index=False)
print(
    f"Wrote {out}  (n={n}, true theta=2, "
    f"poisson/nb beta=(0.3,0.5,-0.4), linear beta=(1,0.8,-0.5))"
)
