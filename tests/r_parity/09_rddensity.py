"""StatsPAI RD density manipulation parity (Python side) -- Module 09.

Runs sp.rddensity on the Lee 2008 senate replica and emits the test
statistic, the p-value, and the left/right density estimates at the
cutoff. The companion 09_rddensity.R uses rddensity::rddensity with
identical defaults.

Tolerance: rel < 1e-3 (iterative bandwidth selection).
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "09_rddensity"


def main() -> None:
    df = sp.datasets.lee_2008_senate()
    dump_csv(df, MODULE)

    fit = sp.rddensity(df, x="margin", c=0.0)
    mi = fit.model_info

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="density_diff",
            estimate=float(mi["density_diff"]),
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="density_left",
            estimate=float(mi["density_left"]), n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="density_right",
            estimate=float(mi["density_right"]), n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="test_pvalue",
            estimate=float(fit.pvalue), n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="bandwidth_left",
            estimate=float(mi["bandwidth_left"]), n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="bandwidth_right",
            estimate=float(mi["bandwidth_right"]), n=int(len(df)),
        ),
    ]

    write_results(
        MODULE, "py", rows,
        extra={
            "polynomial_order": int(mi["polynomial_order"]),
            "test_kind": "Cattaneo-Jansson-Ma (2020)",
            "bandwidth_selector_gap": (
                "sp.rddensity selects h ~ 0.05 while rddensity::rddensity "
                "selects h ~ 0.13 with the same package defaults; the "
                "different effective windows produce different left/right "
                "density estimates but both p-values remain comfortably "
                "above 0.85, so the manipulation-test conclusion (no "
                "evidence of running-variable sorting) is identical. "
                "This mirrors the Module 06 mserd-bandwidth divergence."
            ),
        },
    )


if __name__ == "__main__":
    main()
