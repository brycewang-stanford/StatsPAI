"""StatsPAI Sun-Abraham event-study parity (Python side) -- Module 05.

Runs sp.sun_abraham(..., aggregation="fixest_att") on the mpdta
replica and emits the R/Stata-compatible weighted average ATT plus
the dynamic event-study coefficients at relative times -3, -2, 0, 1,
2 (post-period anchor reference). The
companion R script runs fixest::feols(... | sunab(g, t)) on the
same CSV.

Tolerance: rel < 1e-3 for the simple weighted-average ATT; the
event-study coefficients are reported as a sanity-check.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "05_sunab"


def main() -> None:
    df = sp.datasets.mpdta()
    dump_csv(df, MODULE)

    fit = sp.sun_abraham(
        df, y="lemp", g="first_treat", t="year", i="countyreal",
        aggregation="fixest_att",
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="weighted_avg_ATT",
            estimate=float(fit.estimate),
            se=float(fit.se),
            ci_lo=float(fit.ci[0]) if fit.ci is not None else None,
            ci_hi=float(fit.ci[1]) if fit.ci is not None else None,
            n=int(len(df)),
        )
    ]
    rows.append(
        ParityRecord(
            module=MODULE, side="py", statistic="event_time_avg_ATT",
            estimate=float(fit.model_info["att_event_time"]),
            se=float(fit.model_info["se_event_time"]),
            n=int(len(df)),
        )
    )

    # Per-relative-time event-study coefficients.
    es = fit.model_info.get("event_study")
    if es is not None:
        for _, row in es.iterrows():
            rt = int(row["relative_time"])
            rows.append(
                ParityRecord(
                    module=MODULE, side="py",
                    statistic=f"att_rel_{rt}",
                    estimate=float(row["att"]),
                    se=float(row["se"]),
                    ci_lo=float(row["ci_lower"]),
                    ci_hi=float(row["ci_upper"]),
                    n=int(len(df)),
                )
            )

    write_results(
        MODULE, "py", rows,
        extra={
            "control_group": "nevertreated",
            "method": fit.method,
            "aggregation": fit.model_info["summary_aggregation"],
            "aggregation_parity_note": (
                "The weighted_avg_ATT row uses "
                "sp.sun_abraham(..., aggregation='fixest_att'), which "
                "weights post-treatment cohort-time cells by treated "
                "cohort size and matches fixest::summary(..., agg='att') "
                "on the mpdta fixture. The historical equal-weighted "
                "post-event-time summary is retained as event_time_avg_ATT."
            ),
        },
    )


if __name__ == "__main__":
    main()
