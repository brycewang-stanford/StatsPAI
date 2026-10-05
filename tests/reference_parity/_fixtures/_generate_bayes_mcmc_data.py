"""Synthetic inputs for tests/reference_parity/test_bayes_mcmc_parity.py.

Writes three files next to this script:

* ``bayes_mcmc_chains.csv``     -- one chain of 4,000 draws of six series
  with different dependence (independent, AR(1) mild and strong, a drifting
  start, a skewed marginal, AR(2)).
* ``bayes_mcmc_multichain.csv`` -- four chains of 1,500 draws of three
  series; the last chain has a shifted mean in ``a`` and the spread of
  ``c`` grows with the chain.
* ``bayes_bma.csv``             -- 300 rows, nine candidate regressors and
  a Gaussian, a binary, a count and a positive outcome.

    python tests/reference_parity/_fixtures/_generate_bayes_mcmc_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
rng = np.random.default_rng(20261006)


def ar1(n, rho, mu=0.0, sd=1.0):
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return mu + sd * x


n = 4000
d = pd.DataFrame(
    {
        "iid": 2 + rng.normal(size=n),
        "ar_mild": ar1(n, 0.5, mu=1.0),
        "ar_strong": ar1(n, 0.95, mu=-3.0, sd=0.3),
        "drift": ar1(n, 0.3) + np.linspace(2, 0, n) ** 3 + 4,
        "skew": rng.gamma(2.0, 1.5, size=n),
    }
)
e = rng.normal(size=n)
x = np.zeros(n)
for t in range(2, n):
    x[t] = 1.1 * x[t - 1] - 0.3 * x[t - 2] + e[t]
d["ar2"] = 10 + x
d.to_csv(HERE / "bayes_mcmc_chains.csv", index=False, float_format="%.10g")

parts = []
for i in range(4):
    c = pd.DataFrame(
        {
            "chain": i + 1,
            "a": ar1(1500, 0.6, mu=0.2 * (i == 3)),
            "b": ar1(1500, 0.2, mu=5.0),
            "c": rng.normal(size=1500) * (1 + 0.3 * i),
        }
    )
    parts.append(c)
pd.concat(parts).to_csv(
    HERE / "bayes_mcmc_multichain.csv", index=False, float_format="%.10g"
)

n, K = 300, 9
X = rng.normal(size=(n, K))
X[:, 3] = 0.6 * X[:, 0] + 0.8 * rng.normal(size=n)
X[:, 7] = rng.uniform(size=n) < 0.4
b = np.array([1.0, 0, 0, 0.25, 0.5, 0, 0, -0.35, 0.12])
bm = pd.DataFrame(X, columns=[f"x{j + 1}" for j in range(K)])
eta = 0.4 + X @ b
bm["y"] = eta + rng.normal(size=n)
bm["yb"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-(eta - 0.5)))).astype(int)
bm["yc"] = rng.poisson(np.exp(0.3 * eta))
bm["yg"] = rng.gamma(shape=3.0, scale=np.exp(0.3 * eta) / 3.0)
bm["grp"] = rng.choice(["a", "b", "c"], size=n)
bm.to_csv(HERE / "bayes_bma.csv", index=False, float_format="%.10g")
