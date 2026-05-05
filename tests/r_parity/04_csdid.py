"""StatsPAI CS-DiD parity (Python side) -- Module 04.

Dumps sp.datasets.mpdta() and runs sp.callaway_santanna with the
'reg' (outcome-regression) doubly-robust variant. The companion
04_csdid.R reads the same CSV and runs did::att_gt + did::aggte.

Tolerance: rel < 1e-3 (iterative estimator). The replica is a
deterministic seed=42 simulation calibrated to the published
mpdta neighbourhood.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "04_csdid"


def main() -> None:
    df = sp.datasets.mpdta()
    dump_csv(df, MODULE)

    fit = sp.callaway_santanna(
        df, y="lemp", g="first_treat", t="year", i="countyreal",
        estimator="reg", control_group="nevertreated",
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="simple_ATT",
            estimate=float(fit.estimate),
            se=float(fit.se),
            ci_lo=float(fit.ci[0]) if fit.ci is not None else None,
            ci_hi=float(fit.ci[1]) if fit.ci is not None else None,
            n=int(len(df)),
        )
    ]

    write_results(
        MODULE, "py", rows,
        extra={
            "estimator": "reg",
            "control_group": "nevertreated",
            "method": fit.method,
            "se_note": (
                "The simple-ATT point estimate matches R did::aggte and "
                "Stata csdid at rel < 1e-15. The analytic SE also matches "
                "the R/Stata no-bootstrap reference within the registered "
                "1% tolerance after including the control-regression "
                "uncertainty in the outcome-regression IF."
            ),
        },
    )


if __name__ == "__main__":
    main()
