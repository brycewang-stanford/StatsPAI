"""StatsPAI OLS cluster-robust SE parity (Python side) -- Module 14.

Runs sp.regress on the mpdta replica with cluster-robust SEs at the
county level. The companion 14_ols_cluster.R runs lm + sandwich::
vcovCL on identical data with the same cluster.

Tolerance: rel < 1e-3. CR2 vs CR1 conventions can produce small df-
adjustment differences but the leading-order term is identical.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "14_ols_cluster"
FORMULA = "lemp ~ treat + year"


def main() -> None:
    df = sp.datasets.mpdta()
    dump_csv(df, MODULE)

    fit = sp.regress(FORMULA, data=df, cluster="countyreal")

    rows: list[ParityRecord] = []
    for name in fit.params.index:
        beta = float(fit.params[name])
        se = float(fit.std_errors[name])
        canonical = "(Intercept)" if name == "Intercept" else name
        rows.append(
            ParityRecord(
                module=MODULE, side="py",
                statistic=f"beta_{canonical}",
                estimate=beta, se=se,
                ci_lo=beta - 1.959963984540054 * se,
                ci_hi=beta + 1.959963984540054 * se,
                n=int(fit.data_info.get("n_obs", len(df))),
            )
        )

    write_results(MODULE, "py", rows,
                  extra={"formula": FORMULA, "vcov": "cluster",
                         "cluster_var": "countyreal"})


if __name__ == "__main__":
    main()
