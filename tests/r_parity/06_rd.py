"""StatsPAI RD CCT bias-corrected parity (Python side) -- Module 06.

Runs sp.rdrobust(..., bwselect="cct") on the Lee 2008 senate replica
so the Track A row exercises the same Calonico-Cattaneo-Titiunik
``rdrobust`` bandwidth selector used by R and Stata.  The legacy
StatsPAI internal ``mserd`` selector is still recorded as a diagnostic
row, but the parity headline is the canonical CCT path.

Tolerance: rel < 1e-6 against R/Stata CCT defaults.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "06_rd"


def main() -> None:
    df = sp.datasets.lee_2008_senate()
    dump_csv(df, MODULE)

    # Canonical R/Stata rdrobust bandwidth selector via the official
    # rdrobust Python port.
    fit = sp.rdrobust(df, y="voteshare_next", x="margin", c=0.0, bwselect="cct")

    rows: list[ParityRecord] = []
    for label in ("conventional", "robust"):
        d = fit.model_info[label]
        rows.append(
            ParityRecord(
                module=MODULE, side="py", statistic=f"default_{label}_est",
                estimate=float(d["estimate"]),
                se=float(d["se"]),
                ci_lo=float(d["ci"][0]),
                ci_hi=float(d["ci"][1]),
                n=int(len(df)),
            )
        )

    rows.append(
        ParityRecord(
            module=MODULE, side="py", statistic="default_bandwidth_h",
            estimate=float(fit.model_info["bandwidth_h"]), n=int(len(df)),
        )
    )
    rows.append(
        ParityRecord(
            module=MODULE, side="py", statistic="default_bandwidth_b",
            estimate=float(fit.model_info["bandwidth_b"]), n=int(len(df)),
        )
    )

    # Legacy internal-selector diagnostic.  This keeps the old default
    # visible without mixing it with the R/Stata default-h parity rows.
    legacy = sp.rdrobust(df, y="voteshare_next", x="margin", c=0.0)
    rows.append(
        ParityRecord(
            module=MODULE, side="py",
            statistic="legacy_internal_mserd_bandwidth_h",
            estimate=float(legacy.model_info["bandwidth_h"]), n=int(len(df)),
        )
    )
    rows.append(
        ParityRecord(
            module=MODULE, side="py",
            statistic="legacy_internal_mserd_robust_est",
            estimate=float(legacy.model_info["robust"]["estimate"]),
            se=float(legacy.model_info["robust"]["se"]),
            ci_lo=float(legacy.model_info["robust"]["ci"][0]),
            ci_hi=float(legacy.model_info["robust"]["ci"][1]),
            n=int(len(df)),
        )
    )

    # Forced-bandwidth replicate at the legacy h = b = 0.042287 so the
    # local-polynomial estimator math remains separately pinned.
    H_FORCED = float(legacy.model_info["bandwidth_h"])
    fit_forced = sp.rdrobust(df, y="voteshare_next", x="margin", c=0.0,
                              h=H_FORCED, b=H_FORCED)
    for label in ("conventional", "robust"):
        d = fit_forced.model_info[label]
        rows.append(
            ParityRecord(
                module=MODULE, side="py",
                statistic=f"forced_h{H_FORCED}_{label}_est",
                estimate=float(d["estimate"]),
                se=float(d["se"]),
                ci_lo=float(d["ci"][0]),
                ci_hi=float(d["ci"][1]),
                n=int(len(df)),
            )
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "kernel": fit.model_info["kernel"],
            "p": fit.model_info["polynomial_p"],
            "q": fit.model_info["polynomial_q"],
            "bwselect": fit.model_info["bwselect"],
            "bandwidth_parity_note": (
                "Track A uses sp.rdrobust(..., bwselect='cct'), which "
                "delegates to the official rdrobust Python port and "
                "matches R/Stata rdrobust default mserd bandwidths on "
                "the Lee-2008 fixture. The legacy StatsPAI internal "
                "mserd selector is retained as legacy_internal_mserd_* "
                "diagnostic rows and should not be used as the "
                "cross-language default-h parity claim."
            ),
        },
    )


if __name__ == "__main__":
    main()
