"""StatsPAI original-data parity (Python side) -- Module 01.

Runs sp.regress + sp.ivreg on the *original* Wooldridge::card NLSYM
extract (dumped by the R-side companion). Compares against:
  (a) Card (1995) Table 2 columns 2 and 5 published numbers;
  (b) the R-side lm + AER::ivreg run on the same bytes.
"""
from __future__ import annotations

import statspai as sp

from _common import OrigRecord, read_csv, write_results


MODULE = "01_card_original"


def main() -> None:
    df = read_csv(MODULE)
    n = len(df)

    ols = sp.regress(
        "lwage ~ educ + exper + expersq + black + south + smsa",
        data=df, robust="hc1",
    )
    iv = sp.ivreg(
        "lwage ~ exper + expersq + black + south + smsa + (educ ~ nearc4)",
        data=df, robust="hc1",
    )

    rows = [
        OrigRecord(
            module=MODULE, side="py", statistic="ols_beta_educ",
            estimate=float(ols.params["educ"]),
            se=float(ols.std_errors["educ"]),
            n=n, published=0.075,
            citation="Card (1995) Table 2, col 2",
        ),
        OrigRecord(
            module=MODULE, side="py", statistic="iv_beta_educ_nearc4",
            estimate=float(iv.params["educ"]),
            se=float(iv.std_errors["educ"]),
            n=n, published=0.132,
            citation="Card (1995) Table 2, col 5",
        ),
    ]

    write_results(MODULE, "py", rows,
                  extra={"data_source": "wooldridge::card", "n_obs": n})


if __name__ == "__main__":
    main()
