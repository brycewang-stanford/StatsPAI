"""StatsPAI 2SLS parity (Python side) -- Module 02.

Re-uses the Card 1995 replica with nearc4 as the instrument for educ.
HC1 robust SE on both sides.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "02_iv"
FORMULA = "lwage ~ exper + expersq + black + south + smsa + (educ ~ nearc4)"


def main() -> None:
    df = sp.datasets.card_1995()
    dump_csv(df, MODULE)

    fit = sp.ivreg(FORMULA, data=df, robust="hc1")

    rows: list[ParityRecord] = []
    for name in fit.params.index:
        beta = float(fit.params[name])
        se = float(fit.std_errors[name])
        canonical = "(Intercept)" if name == "Intercept" else name
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"beta_{canonical}",
                estimate=beta,
                se=se,
                ci_lo=beta - 1.959963984540054 * se,
                ci_hi=beta + 1.959963984540054 * se,
                n=int(fit.data_info.get("n_obs", len(df))),
            )
        )

    write_results(MODULE, "py", rows, extra={"formula": FORMULA, "vcov": "HC1"})


if __name__ == "__main__":
    main()
