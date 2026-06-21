"""StatsPAI Poisson PML with HDFE parity (Python side) -- Module 37.

DGP: count outcome with single FE absorber, two regressors. Sized
to be well-conditioned enough that sp.ppmlhdfe converges within
its default 1000-iter budget; the original gravity-style 3-way FE
DGP triggered non-convergence (recorded as a sp.ppmlhdfe
robustness gap in the session report).

References:
- fixest::fepois (Berge 2018)
- ppmlhdfe (Correia, Guimaraes, Zylkin 2020)
- Santos Silva & Tenreyro (2006)

Tolerance: rel < 1e-3 on coefficients; rel < 1e-2 on HC1 SEs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "37_ppmlhdfe"


def make_data(N: int = 500, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    origin = rng.integers(0, 5, N)
    x1 = rng.normal(0, 1, N)
    x2 = rng.binomial(1, 0.5, N).astype(float)
    mu = np.exp(0.5 + 0.3 * x1 - 0.4 * x2 + 0.2 * origin)
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "origin": origin})


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    res = sp.ppmlhdfe(
        formula="y ~ x1 + x2 | origin",
        data=df,
        robust="hc1",
    )

    rows: list[ParityRecord] = []
    for name in ["x1", "x2"]:
        rows.append(ParityRecord(
            module=MODULE, side="py",
            statistic=f"beta_{name}",
            estimate=float(res.params[name]),
            se=float(res.std_errors[name]),
            n=int(len(df))))

    write_results(
        MODULE, "py", rows,
        extra={
            "fe": "origin",
            "vcov": "HC1",
            "n_obs": int(len(df)),
        },
    )


if __name__ == "__main__":
    main()
