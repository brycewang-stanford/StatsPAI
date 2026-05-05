"""StatsPAI Wooldridge ETWFE parity (Python side) -- Module 17.

Runs sp.etwfe on the mpdta replica. The companion 17_etwfe.R uses
etwfe::etwfe, the canonical R port of Wooldridge's extended
two-way fixed-effects estimator.

Tolerance: rel < 1e-3 on the pooled ATT.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "17_etwfe"


def main() -> None:
    df = sp.datasets.mpdta()
    dump_csv(df, MODULE)

    fit = sp.etwfe(df, y="lemp", group="countyreal", time="year",
                   first_treat="first_treat")

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_etwfe",
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
            "method": "Wooldridge ETWFE",
            "n_cohorts": int(fit.model_info["n_cohorts"]),
            "aggregation_note": (
                "Wooldridge ETWFE pools cohort-specific ATTs; sp uses "
                "cohort-share weighting while etwfe::etwfe + emfx() uses "
                "treated-observation weighting. On mpdta the gap is "
                "~8% (sp -0.038 vs R -0.035); same family as the BJS "
                "and Sun-Abraham aggregation gaps in Modules 5 and 16."
            ),
        },
    )


if __name__ == "__main__":
    main()
