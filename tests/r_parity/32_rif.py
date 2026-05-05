"""StatsPAI RIF/UQR decomposition parity (Python side) -- Module 32.

Generates a deterministic wage-gap dataset and runs sp.decomposition
.rif_decomposition at the median (tau=0.5). The companion 32_rif.R
uses dineq::rifr.

Tolerance: rel < 1e-2 on the total_diff and on each detailed
contribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "32_rif"


def make_data(n: int = 1000, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    female = rng.binomial(1, 0.5, n)
    educ = rng.normal(15, 3, n)
    exper = rng.normal(10, 5, n) - 1.0 * (female == 1)
    log_wage = (
        2.0 + 0.07 * educ + 0.02 * exper - 0.20 * female
        + rng.normal(0, 0.3, n)
    )
    return pd.DataFrame({
        "log_wage": log_wage, "educ": educ, "exper": exper,
        "female": female,
    })


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    res = sp.decomposition.rif_decomposition(
        "log_wage ~ educ + exper",
        data=df, group="female",
        statistic="quantile", tau=0.5,
    )

    rows: list[ParityRecord] = [
        ParityRecord(MODULE, "py", "total_diff",
                     estimate=float(res.total_diff), n=int(len(df))),
        ParityRecord(MODULE, "py", "explained",
                     estimate=float(res.explained), n=int(len(df))),
        ParityRecord(MODULE, "py", "unexplained",
                     estimate=float(res.unexplained), n=int(len(df))),
    ]

    for _, row in res.detailed.iterrows():
        v = row["variable"]
        rows.append(ParityRecord(
            MODULE, "py", f"explained_{v}",
            estimate=float(row["explained"]), n=int(len(df))))

    write_results(MODULE, "py", rows,
                  extra={"statistic": "quantile", "tau": 0.5})


if __name__ == "__main__":
    main()
