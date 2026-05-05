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
            "se_known_gap": (
                "The simple-ATT SE under-covers because the current "
                "aggregation treats per-(g,t) influence functions as "
                "independent. did::att_gt + aggte applies the full "
                "influence-function multiplier bootstrap and reports "
                "a larger SE; on this DGP the ratio R/sp is ~2.8x. "
                "The point estimate matches at rel < 1e-15. The "
                "remediation is on the v0.9.6 / 1.14 roadmap; see "
                "tests/coverage_monte_carlo/FINDINGS.md and JSS "
                "draft section 5.6."
            ),
        },
    )


if __name__ == "__main__":
    main()
