"""Simulated AR(1)-GARCH(2,1) series with Student t innovations.

Writes ``garch_extensions.csv`` (columns t, r), the input of
``_generate_garch_extensions_Stata.do``. Deterministic: seed 20261006.
"""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261006)
T, burn = 1500, 500
mu, rho, omega, alpha, beta, nu = 0.05, 0.15, 0.05, 0.12, (0.35, 0.45), 7.0
n = T + burn
z = rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)
s2 = np.full(n, omega / (1.0 - alpha - sum(beta)))
eps = np.zeros(n)
u = np.zeros(n)
for t in range(2, n):
    s2[t] = omega + alpha * eps[t - 1] ** 2 + beta[0] * s2[t - 1] + beta[1] * s2[t - 2]
    eps[t] = np.sqrt(s2[t]) * z[t]
    u[t] = rho * u[t - 1] + eps[t]
r = mu + u[burn:]
lines = ["t,r"] + [f"{i},{repr(float(v))}" for i, v in enumerate(r)]
Path(__file__).with_name("garch_extensions.csv").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
