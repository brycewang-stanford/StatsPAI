"""StatsPAI original-data parity (Python side) -- Module 04b.

Runs sp.regress + sp.psm on the *true* Dehejia-Wahba NSW+PSID-1
sample (2675 obs, 185 treated) from causalsens::lalonde.psid.
Replaces the earlier module 04 which used MatchIt::lalonde
(614 obs, not the NSW+PSID-1 sample).
"""
from __future__ import annotations

import statspai as sp

from _common import OrigRecord, read_csv, write_results


MODULE = "04b_nsw_psid_original"


def main() -> None:
    df = read_csv(MODULE)
    n = len(df)

    naive = sp.regress("re78 ~ treat", data=df, robust="hc1")
    adj = sp.regress(
        "re78 ~ treat + age + education + black + hispanic + "
        "married + nodegree + re74 + re75",
        data=df, robust="hc1",
    )
    psm = sp.psm(
        df, y="re78", d="treat",
        X=["age", "education", "black", "hispanic", "married",
           "nodegree", "re74", "re75"],
        method="nn",
    )

    rows = [
        OrigRecord(
            module=MODULE, side="py", statistic="naive_ols_att",
            estimate=float(naive.params["treat"]),
            se=float(naive.std_errors["treat"]),
            n=n, published=-15205,
            citation="Dehejia-Wahba (1999) naive OLS on NSW+PSID-1",
        ),
        OrigRecord(
            module=MODULE, side="py", statistic="adj_ols_att",
            estimate=float(adj.params["treat"]),
            se=float(adj.std_errors["treat"]),
            n=n, published=700,
            citation="Dehejia-Wahba (1999) covariate-adjusted OLS on NSW+PSID-1",
        ),
        OrigRecord(
            module=MODULE, side="py", statistic="psm_att",
            estimate=float(psm.estimate),
            se=float(psm.se) if psm.se is not None else None,
            n=n, published=1690,
            citation="Dehejia-Wahba (1999) PSM 1:1 NN on NSW+PSID-1",
        ),
    ]
    write_results(MODULE, "py", rows,
                  extra={"data_source": "causalsens::lalonde.psid", "n_obs": n})


if __name__ == "__main__":
    main()
