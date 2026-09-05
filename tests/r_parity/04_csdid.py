"""StatsPAI CS-DiD parity (Python side) -- Module 04.

Dumps sp.datasets.mpdta() and runs sp.callaway_santanna with the
'reg' (outcome-regression) doubly-robust variant. The companion
04_csdid.R reads the same CSV and runs did::att_gt + did::aggte.

Tolerance: rel < 1e-3 (iterative estimator). The replica is a
deterministic seed=42 simulation calibrated to the published
mpdta neighbourhood.
"""

from __future__ import annotations

import re

import statspai as sp

from _common import ParityRecord, dump_csv, write_results

MODULE = "04_csdid"


def main() -> None:
    df = sp.datasets.mpdta()
    dump_csv(df, MODULE)

    fit = sp.callaway_santanna(
        df,
        y="lemp",
        g="first_treat",
        t="year",
        i="countyreal",
        estimator="reg",
        control_group="nevertreated",
        base_period="universal",
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="simple_ATT",
            estimate=float(fit.estimate),
            se=float(fit.se),
            ci_lo=float(fit.ci[0]) if fit.ci is not None else None,
            ci_hi=float(fit.ci[1]) if fit.ci is not None else None,
            n=int(len(df)),
        )
    ]

    # Aggregation VECTORS, not just the headline scalar. The event study
    # is the object a parallel-trends claim is read off, and until now
    # this module pinned one number and left the whole path unpinned.
    for agg_type, label in (
        ("dynamic", "event"),
        ("group", "group"),
        ("calendar", "calendar"),
    ):
        agg = sp.aggte(fit, type=agg_type, bstrap=False, cband=False)
        tidy = agg.tidy()
        # dynamic exposes its cells as type="event_study" with terms
        # "event_+1"; group and calendar as type="group_time" with terms
        # "att(g=2004.0)" / "att(t=2005.0)".
        cells = tidy[tidy["type"].isin(("event_study", "group_time"))]
        for term, est, se in zip(cells["term"], cells["estimate"], cells["std_error"]):
            key = re.sub(r"^att\((?:g|t)=", "", str(term)).rstrip(")")
            key = key.replace("event_", "")
            if key.endswith(".0"):
                key = key[:-2]
            rows.append(
                ParityRecord(
                    module=MODULE,
                    side="py",
                    statistic=f"{label}_{key}",
                    estimate=float(est),
                    se=float(se),
                )
            )
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"{label}_overall",
                estimate=float(agg.estimate),
                se=float(agg.se),
            )
        )

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "estimator": "reg",
            "control_group": "nevertreated",
            "base_period": "universal",
            "method": fit.method,
            "group_overall_stata_gap": (
                "One row is not a three-way match. group_overall: StatsPAI "
                "and R did::aggte agree to 1.2e-16; Stata csdid's estat group "
                "GAverage standard error differs by 2.7e-3 (0.27%). The point "
                "estimate matches all three. Mechanism (reconstructed, "
                "2026-09-05): csdid aggregates the per-cohort influence "
                "functions with the cohort shares held fixed, whereas did and "
                "StatsPAI add the share-estimation term (did:::wif). Rebuilding "
                "the fixed-share aggregate from StatsPAI's joint cell "
                "influence functions returns csdid's SE to 1e-8; an earlier "
                "independent-cell reconstruction overshot (0.00746) because "
                "it dropped the covariance between cells that share control "
                "units."
            ),
            "base_period_note": (
                "base_period is pinned explicitly on BOTH sides. StatsPAI "
                "defaults to 'universal' and R did defaults to 'varying'. "
                "The simple ATT cannot see the difference -- it averages "
                "post-treatment cells only -- so this module passed for "
                "years without the option being matched. The event-study "
                "path can see it, and would have reported a spurious "
                "disagreement on every pre-treatment cell."
            ),
            "se_note": (
                "The simple-ATT point estimate matches R did::aggte and "
                "Stata csdid at rel < 1e-15. The analytic SE now matches the "
                "R/Stata no-bootstrap reference at rel ~ 4e-16, once the "
                "outcome-regression IF carries the control-regression "
                "uncertainty AND sp.aggte carries the cohort-share "
                "weight-estimation term. The registered rel_se tolerance was "
                "1% while that term was missing; it is now 1e-9."
            ),
        },
    )


if __name__ == "__main__":
    main()
