"""StatsPAI HDFE 2-way FE parity (Python side) -- Module 03.

A deterministic two-way fixed-effects DGP at N=10,000 with two FE
factors (firm, year). Compares sp.fast.feols against R fixest::feols
with vcov = "iid" so both implementations use OLS-style SEs after
absorbing the FEs.

Tolerance: rel < 1e-6 (closed-form estimator after the within
transform).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "03_hdfe"
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

    fit = sp.fast.feols(FORMULA, data=df, vcov="iid", ssc="fixest")

    # FeolsResult exposes .coef() and .se() as methods returning a
    # pandas Series indexed by the non-absorbed regressor names.
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
            "vcov": "iid",
            "n_firms": int(df["firm"].nunique()),
            "n_years": int(df["year"].nunique()),
            "df_resid": int(fit.df_resid),
            "rss": float(fit.rss),
            "ssc": "fixest",
            "parity_note": (
                "sp.fast.feols uses ssc='fixest' so the residual FE "
                "rank and IID standard errors match fixest::feols and "
                "reghdfe on this two-way HDFE fixture."
            ),
        },
    )


if __name__ == "__main__":
    main()
