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
                   first_treat="first_treat", cluster="countyreal",
                   panel=False)
    simple = sp.etwfe_emfx(fit, type="simple", weighting="treated")

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_etwfe",
            estimate=float(simple.estimate),
            se=float(simple.se),
            ci_lo=float(simple.ci[0]) if simple.ci is not None else None,
            ci_hi=float(simple.ci[1]) if simple.ci is not None else None,
            n=int(len(df)),
        )
    ]

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "Wooldridge ETWFE + emfx simple aggregation",
            "n_cohorts": int(fit.model_info["n_cohorts"]),
            "aggregation_note": (
                "Parity row uses the R etwfe default no-ivar fixed-effect "
                "structure (StatsPAI panel=False) and sp.etwfe_emfx("
                "weighting='treated'), which averages cohort-time marginal "
                "effects over treated post-period observations like "
                "etwfe::emfx(type='simple'). Point estimates and clustered "
                "delta-method SEs match the R reference."
            ),
        },
    )


if __name__ == "__main__":
    main()
