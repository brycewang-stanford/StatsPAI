"""StatsPAI cross-section stochastic frontier parity (Python side).
Module 28.

Generates a deterministic Cobb-Douglas frontier DGP with half-normal
inefficiency. The companion 28_frontier.R uses sfaR::sfacross with
the same distribution. Tolerance: rel < 1e-2 on the production
frontier coefficients (frontier optimisation has multiple local
optima, but on a clean DGP both implementations should converge to
the same solution).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "28_frontier"


def make_data(n: int = 500, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lnk = rng.normal(0, 1, n)
    lnl = rng.normal(0, 1, n)
    v = rng.normal(0, 0.3, n)
    u = np.abs(rng.normal(0, 0.5, n))
    lny = 2.0 + 0.6 * lnk + 0.4 * lnl + v - u
    return pd.DataFrame({"lny": lny, "lnk": lnk, "lnl": lnl})


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    fit = sp.frontier(data=df, y="lny", x=["lnk", "lnl"],
                      dist="half-normal", seed=PARITY_SEED)

    rows: list[ParityRecord] = [
        ParityRecord(MODULE, "py", "beta_intercept",
                     estimate=float(fit.params["_cons"]),
                     se=float(fit.std_errors["_cons"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "beta_lnk",
                     estimate=float(fit.params["lnk"]),
                     se=float(fit.std_errors["lnk"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "beta_lnl",
                     estimate=float(fit.params["lnl"]),
                     se=float(fit.std_errors["lnl"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "sigma_u",
                     estimate=float(fit.model_info["sigma_u_mean"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "sigma_v",
                     estimate=float(fit.model_info["sigma_v_mean"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "lambda",
                     estimate=float(fit.model_info["lambda"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "mean_efficiency",
                     estimate=float(fit.model_info["mean_efficiency_jlms"]),
                     n=int(len(df))),
    ]

    write_results(MODULE, "py", rows,
                  extra={"distribution": "half-normal",
                         "cost": False,
                         "method": fit.model_info["method"]})


if __name__ == "__main__":
    main()
