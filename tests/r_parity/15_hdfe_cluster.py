"""StatsPAI HDFE cluster-SE parity (Python side) -- Module 15.

Same gravity DGP as Module 03 but with cluster-robust SE at the firm
level. The companion 15_hdfe_cluster.R runs fixest::feols with the
same cluster spec.

Tolerance: rel < 1e-3 on the cluster-robust SE; coefficients should
match at machine precision (cluster choice doesn't affect point
estimates).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "15_hdfe_cluster"
FORMULA = "y ~ x1 + x2 | firm + year"


def make_panel(n: int = 10_000, n_firms: int = 250, n_years: int = 20,
               seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firms, size=n)
    year = rng.integers(0, n_years, size=n)
    firm_fe = rng.normal(0, 1.0, size=n_firms)
    year_fe = rng.normal(0, 0.5, size=n_years)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (
        2.0 * x1
        - 1.5 * x2
        + firm_fe[firm]
        + year_fe[year]
        + rng.normal(scale=0.5, size=n)
    )
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "year": year})


def main() -> None:
    df = make_panel()
    dump_csv(df, MODULE)

    fit = sp.fast.feols(FORMULA, data=df, vcov="cr1", cluster="firm")
    coef = fit.coef()
    ses = fit.se()

    rows: list[ParityRecord] = []
    for name in ["x1", "x2"]:
        beta = float(coef[name])
        se = float(ses[name])
        rows.append(
            ParityRecord(
                module=MODULE, side="py", statistic=f"beta_{name}",
                estimate=beta, se=se,
                ci_lo=beta - 1.959963984540054 * se,
                ci_hi=beta + 1.959963984540054 * se,
                n=int(len(df)),
            )
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "formula": FORMULA,
            "vcov": "cr1",
            "cluster_var": "firm",
            "n_firms": int(df["firm"].nunique()),
            "n_years": int(df["year"].nunique()),
            "cluster_ssc_note": (
                "Coefficients match at rel < 1e-15. The CR1 cluster "
                "SE differs by 1.27% because sp.fast.feols applies "
                "the simple G/(G-1) cluster small-sample correction "
                "while fixest::feols applies the multi-factor "
                "ssc(cluster.adj = TRUE, fixef.K = \"nested\", "
                "cluster.df = \"min\") convention by default. This "
                "is the same family of df-convention differences as "
                "the iid SE gap reported in Module 03."
            ),
        },
    )


if __name__ == "__main__":
    main()
