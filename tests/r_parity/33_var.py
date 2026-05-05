"""StatsPAI VAR parity (Python side) -- Module 33.

Generates a deterministic 2-variable VAR(2) and runs sp.var. The
companion 33_var.R uses vars::VAR.

Tolerance: rel < 1e-3 on the coefficient matrix entries (closed-
form OLS-by-equation; both implementations compute the same thing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "33_var"
LAGS = 2


def make_data(T: int = 200, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y1 = np.zeros(T)
    y2 = np.zeros(T)
    for t in range(1, T):
        y1[t] = 0.5 * y1[t - 1] + 0.2 * y2[t - 1] + rng.normal(0, 0.5)
        y2[t] = -0.3 * y1[t - 1] + 0.6 * y2[t - 1] + rng.normal(0, 0.5)
    return pd.DataFrame({"y1": y1, "y2": y2})


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    fit = sp.var(data=df, variables=["y1", "y2"], lags=LAGS)

    rows: list[ParityRecord] = []
    for eq in ["y1", "y2"]:
        coefs = fit.coefs[eq]
        for term in coefs.index:
            beta = float(coefs.loc[term, "coef"])
            se = float(coefs.loc[term, "se"])
            rows.append(ParityRecord(
                module=MODULE, side="py",
                statistic=f"eq_{eq}__{term}",
                estimate=beta, se=se, n=int(fit.n_obs)))

    rows.append(ParityRecord(
        module=MODULE, side="py", statistic="logLik",
        estimate=float(fit.log_likelihood), n=int(fit.n_obs)))

    write_results(MODULE, "py", rows,
                  extra={"lags": LAGS, "n_obs": int(fit.n_obs)})


if __name__ == "__main__":
    main()
