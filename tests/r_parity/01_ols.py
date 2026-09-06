"""StatsPAI OLS parity (Python side) -- Module 01.

Dumps sp.datasets.card_1995(simulated=True) to data/01_ols.csv and
runs sp.regress
with HC1 robust SEs. The companion 01_ols.R reads the same CSV and
runs lm() + sandwich::vcovHC(type="HC1"). Tolerance: rel < 1e-6
(closed-form estimator).
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "01_ols"
FORMULA = "lwage ~ educ + exper + expersq + black + south + smsa"


def main() -> None:
    # simulated=True is load-bearing, not decoration. This module *writes*
    # data/01_ols.csv, and the committed fixture -- together with the R and
    # Stata goldens computed on it -- is the calibrated replica. The
    # loader's default later became the real extract, so a bare
    # card_1995() call silently regenerates the fixture from different
    # data and breaks the same-byte premise of the row. Original-data
    # Card parity is a separate ledger (tests/orig_parity/01_card_original).
    df = sp.datasets.card_1995(simulated=True)
    dump_csv(df, MODULE)

    fit = sp.regress(FORMULA, data=df, robust="hc1")

    rows: list[ParityRecord] = []
    for name in fit.params.index:
        beta = float(fit.params[name])
        se = float(fit.std_errors[name])
        # Normalise the Python "Intercept" label to R's "(Intercept)"
        # so the parity comparator can join row-for-row.
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
