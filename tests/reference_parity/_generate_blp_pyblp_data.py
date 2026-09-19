"""Write the BLP parity dataset ``_fixtures/blp_pyblp_data.csv`` (fixed seed).

80 markets with 4-10 products each. Utility of consumer i for product j in
market t:
    u_ijt = 1.0 + 1.0 * x1_jt - 2.0 * p_jt + xi_jt
            + 1.0 * nu_i1 * x1_jt + 0.3 * nu_i2 * p_jt + eps_ijt
with price endogenous (correlated with xi). Excluded instruments carried in
the file: cost shifters w1, w2 and the Gandhi-Houde differentiation
instruments gh_x1 = sum_{k != j} (x1_k - x1_j)^2 and gh_w1 (same on w1).
Shares are integrated with 20 000 pseudo-random consumers per market, so
the data are a draw from a random-coefficients logit; the parity tests do
NOT use these consumers (both sides are handed the same Halton nodes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "_fixtures" / "blp_pyblp_data.csv"


def main() -> None:
    rng = np.random.default_rng(20260922)
    rows = []
    for t in range(80):
        J = int(rng.integers(4, 11))
        x1 = rng.normal(0.0, 1.0, J)
        w1 = rng.uniform(0.0, 1.0, J)
        w2 = rng.uniform(0.0, 1.0, J)
        xi = rng.normal(0.0, 0.3, J)
        price = 1.5 + 0.8 * w1 + 0.5 * w2 + 0.3 * x1 + 0.5 * xi + rng.uniform(0, 0.3, J)
        nu = rng.normal(size=(20000, 2))
        delta = 1.0 + 1.0 * x1 - 2.0 * price + xi
        mu = 1.0 * np.outer(nu[:, 0], x1) + 0.3 * np.outer(nu[:, 1], price)
        v = delta[None, :] + mu
        ev = np.exp(v)
        s = (ev / (1.0 + ev.sum(axis=1, keepdims=True))).mean(axis=0)
        gh_x1 = ((x1[None, :] - x1[:, None]) ** 2).sum(axis=1)
        gh_w1 = ((w1[None, :] - w1[:, None]) ** 2).sum(axis=1)
        for j in range(J):
            rows.append(
                {
                    "market_id": t,
                    "product_id": j,
                    "share": s[j],
                    "price": price[j],
                    "x1": x1[j],
                    "w1": w1[j],
                    "w2": w2[j],
                    "gh_x1": gh_x1[j],
                    "gh_w1": gh_w1[j],
                }
            )
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
