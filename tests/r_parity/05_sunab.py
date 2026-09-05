"""StatsPAI Sun-Abraham event-study parity (Python side) -- Module 05.

Runs sp.sun_abraham(..., aggregation="fixest_att") on the mpdta
replica and emits the R/Stata-compatible weighted average ATT plus
the dynamic event-study coefficients at relative times -3, -2, 0, 1,
2 (post-period anchor reference). The
companion R script runs fixest::feols(... | sunab(g, t)) on the
same CSV.

Rows: weighted_avg_ATT (aggregation='fixest_att', matches
fixest::summary(agg='att') and its clustered SE at machine level),
att_rel_<e> (default share_variance=True: Sun & Abraham 2021 Prop. 3 /
Stata eventstudyinteract variance, matched to Stata at machine level and
to fixest wherever a single cohort is eligible), and
att_rel_<e>_fixedshare (share_variance=False: fixest::sunab's
fixed-share variance, matched to fixest at machine level everywhere).
Both settings share the fixest/reghdfe "nested" degrees-of-freedom rule.

Tolerance: rel_est 1e-6 on every row; rel_se is registered in
compare.py::TOLERANCES and bounds the documented share-term gap on the
att_rel_<e> rows against R only.
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

    # fixest convention: cohort shares treated as fixed in the variance.
    fit_fs = sp.sun_abraham(
        df, y="lemp", g="first_treat", t="year", i="countyreal",
        aggregation="fixest_att", share_variance=False,
    )
    es_fs = fit_fs.model_info.get("event_study")
    if es_fs is not None:
        for _, row in es_fs.iterrows():
            rt = int(row["relative_time"])
            rows.append(
                ParityRecord(
                    module=MODULE, side="py",
                    statistic=f"att_rel_{rt}_fixedshare",
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
            "share_variance": True,
            "dof_K": int(fit.model_info["dof_K"]),
            "dof_convention": fit.model_info["dof_convention"],
            "variance_parity_note": (
                "att_rel_<e> rows carry the Sun & Abraham (2021, Prop. 3) "
                "cohort-share term (Stata eventstudyinteract convention); "
                "att_rel_<e>_fixedshare rows set share_variance=False and "
                "reproduce fixest::sunab's fixed-share variance. K follows "
                "the fixest/reghdfe nested rule: 12 observed cohort-by-"
                "relative-time cells + 5 year effects = 17 on mpdta."
            ),
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
