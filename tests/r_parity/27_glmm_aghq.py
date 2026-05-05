"""StatsPAI GLMM with adaptive Gauss-Hermite quadrature parity.
Module 27.

Same DGP as Module 26 but with nAGQ=8 (8-point AGHQ instead of
Laplace approximation). Tolerance: rel < 1e-3 (AGHQ converges to
the exact integral and should agree across implementations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "27_glmm_aghq"


def make_data(n_groups: int = 60, n_per: int = 25,
              seed: int = PARITY_SEED) -> pd.DataFrame:
    """Same DGP as Module 26."""
    rng = np.random.default_rng(seed)
    gid = np.repeat(np.arange(n_groups), n_per)
    n = n_groups * n_per
    x1 = rng.normal(size=n)
    group_re = rng.normal(scale=0.8, size=n_groups)
    linpred = -0.5 + 0.7 * x1 + group_re[gid]
    prob = 1 / (1 + np.exp(-linpred))
    y = (rng.uniform(size=n) < prob).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "gid": gid})


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    fit = sp.melogit(data=df, y="y", x_fixed=["x1"], group="gid", nAGQ=8)

    rows: list[ParityRecord] = [
        ParityRecord(MODULE, "py", "beta_intercept",
                     estimate=float(fit.params["_cons"]),
                     se=float(fit.bse["_cons"]),
                     n=int(fit.n_obs)),
        ParityRecord(MODULE, "py", "beta_x1",
                     estimate=float(fit.params["x1"]),
                     se=float(fit.bse["x1"]),
                     n=int(fit.n_obs)),
        ParityRecord(MODULE, "py", "logLik",
                     estimate=float(fit.log_likelihood),
                     n=int(fit.n_obs)),
    ]

    write_results(MODULE, "py", rows,
                  extra={"family": "binomial",
                         "link": "logit",
                         "nAGQ": 8,
                         "n_groups": int(fit.n_groups)})


if __name__ == "__main__":
    main()
