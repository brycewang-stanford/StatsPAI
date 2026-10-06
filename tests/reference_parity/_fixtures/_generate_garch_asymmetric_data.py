"""Simulated returns with a leverage effect, for the asymmetric GARCH tests.

Writes ``garch_asymmetric.csv`` (columns t, r), the input of
``_generate_garch_asymmetric_Stata.do``: a threshold GARCH(1,1) in which a
negative shock raises the next variance more than a positive one, with
Student t innovations. Seed 20261007.
"""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261007)
T, burn = 2000, 500
mu, omega, alpha, gamma, beta, nu = 0.04, 0.04, 0.03, 0.14, 0.86, 8.0
n = T + burn
z = rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)
s2 = np.full(n, omega / (1.0 - alpha - gamma / 2.0 - beta))
eps = np.zeros(n)
for t in range(1, n):
    lev = gamma * eps[t - 1] ** 2 * (eps[t - 1] < 0)
    s2[t] = omega + alpha * eps[t - 1] ** 2 + lev + beta * s2[t - 1]
    eps[t] = np.sqrt(s2[t]) * z[t]
r = mu + eps[burn:]
lines = ["t,r"] + [f"{i},{repr(float(v))}" for i, v in enumerate(r)]
Path(__file__).with_name("garch_asymmetric.csv").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
