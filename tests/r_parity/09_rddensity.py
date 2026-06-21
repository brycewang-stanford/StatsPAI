"""StatsPAI RD density manipulation parity (Python side) -- Module 09.

Runs the **native Python** sp.rddensity (backend="native") on the Lee
2008 senate replica and emits the left/right density estimates, default
combination bandwidths, robust p-value, and density difference at the
cutoff. The companion 09_rddensity.R uses rddensity::rddensity with
identical defaults.

Tier: T2 native reference parity.  The native implementation ports the
default rddensity unrestricted triangular-kernel path: rdbwdensity
combination bandwidths, mass-point ECDF handling, and jackknife CJM
local-polynomial density inference.  The optional backend='r' bridge is
still available for users who want to delegate directly to the R package,
but it is not used here as the comparator.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "09_rddensity"


def main() -> None:
    df = sp.datasets.lee_2008_senate()
    dump_csv(df, MODULE)

    fit = sp.rddensity(df, x="margin", c=0.0, backend="native")
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
            module=MODULE, side="py", statistic="bandwidth_left",
            estimate=float(mi["bandwidth_left"]), n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="bandwidth_right",
            estimate=float(mi["bandwidth_right"]), n=int(len(df)),
        ),
    ]
    _pv = getattr(fit, "pvalue", None)
    if _pv is not None:
        rows.append(ParityRecord(
            module=MODULE, side="py", statistic="test_pvalue",
            estimate=float(_pv), n=int(len(df))))

    write_results(
        MODULE, "py", rows,
        extra={
            "polynomial_order": int(mi.get("polynomial_order", 2)),
            "backend": mi.get("backend", "native"),
            "validation_tier": mi.get("validation_tier"),
            "reference_backend": mi.get("reference_backend"),
            "test_kind": "Cattaneo-Jansson-Ma (2020)",
            "tier": "T2",
            "native_note": (
                "Headline row is the NATIVE Python CJM density test "
                "(backend='native'). The native path ports rddensity's "
                "default unrestricted triangular-kernel CJM estimator, "
                "including rdbwdensity combination bandwidths, mass-point "
                "ECDF handling, and jackknife inference, so the Lee row is "
                "graded T2 native reference parity. The optional backend='r' "
                "bridge remains a convenience feature, not the comparator."
            ),
        },
    )


if __name__ == "__main__":
    main()
