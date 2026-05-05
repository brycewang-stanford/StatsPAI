"""StatsPAI original-data parity (Python side) -- Module 04.

Runs sp.regress (naive + adjusted) and sp.psm on the *original*
MatchIt::lalonde NSW+PSID-1 sample (Dehejia-Wahba 1999).
"""
from __future__ import annotations

import statspai as sp

from _common import OrigRecord, read_csv, write_results


MODULE = "04_lalonde_original"


def main() -> None:
    df = read_csv(MODULE)
    n = len(df)

    naive = sp.regress("re78 ~ treat", data=df, robust="hc1")
    adj = sp.regress(
        "re78 ~ treat + age + educ + black + hispanic + married + nodegree + re74 + re75",
        data=df, robust="hc1",
    )
    psm = sp.psm(
        df, y="re78", d="treat",
        X=["age", "educ", "black", "hispanic", "married",
           "nodegree", "re74", "re75"],
        method="nn",
    )

    rows = [
        OrigRecord(
            module=MODULE, side="py", statistic="naive_ols_att",
            estimate=float(naive.params["treat"]),
            se=float(naive.std_errors["treat"]),
            n=n, published=-8498,
            citation="Dehejia-Wahba (1999) Table 3, naive OLS",
        ),
        OrigRecord(
            module=MODULE, side="py", statistic="adj_ols_att",
            estimate=float(adj.params["treat"]),
            se=float(adj.std_errors["treat"]),
            n=n, published=218,
            citation="Dehejia-Wahba (1999) Table 3, covariate-adjusted OLS",
        ),
        OrigRecord(
            module=MODULE, side="py", statistic="psm_att",
            estimate=float(psm.estimate),
            se=float(psm.se) if psm.se is not None else None,
            n=n, published=1794,
            citation="Dehejia-Wahba (1999) Table 4, PSM 1:1 NN",
        ),
    ]

    write_results(MODULE, "py", rows,
                  extra={"data_source": "MatchIt::lalonde", "n_obs": n})


if __name__ == "__main__":
    main()
