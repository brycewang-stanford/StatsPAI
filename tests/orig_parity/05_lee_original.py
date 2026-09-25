"""StatsPAI original-data parity (Python side) -- Module 05.

Runs the native sp.rdrobust default (CCT MSE-optimal bandwidth,
robust bias-corrected inference, implemented in StatsPAI itself) on the
*original* rdrobust::rdrobust_RDsenate extract (a Lee-style sharp RD on
the Senate vote-share margin).

The ``bwselect="cct"`` option delegates to the official rdrobust Python
port; a row computed that way compares the authors' code with itself, so
it must never appear in this ledger (see
tests/test_orig_parity_native_contract.py).
"""

from __future__ import annotations

import statspai as sp

from _common import OrigRecord, read_csv, write_results

MODULE = "05_lee_original"


def _scalar(v):
    """Coerce an sp.rdrobust output cell (DataFrame / Series / scalar) to float."""
    if hasattr(v, "iloc"):
        return float(v.iloc[0])
    return float(v)


def main() -> None:
    df = read_csv(MODULE)
    n = len(df)

    fit = sp.rdrobust(df, y="y", x="x", c=0.0, kernel="triangular")
    conventional = fit.model_info["conventional"]
    robust = fit.model_info["robust"]
    h = fit.model_info["bandwidth_h"]
    h_left, h_right = h if isinstance(h, tuple) else (h, h)

    rows = [
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="rd_jump_conventional",
            estimate=_scalar(conventional["estimate"]),
            se=_scalar(conventional["se"]),
            n=n,
            published=7.99,
            citation="Lee (2008) Table 1; CCT (2014) Table 4 conventional sharp-RD jump",
            extra={"h_left": h_left, "h_right": h_right},
        ),
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="rd_jump_robust",
            estimate=_scalar(robust["estimate"]),
            se=_scalar(robust["se"]),
            n=n,
            published=None,
            citation="rdrobust::rdrobust bias-corrected robust point and SE",
            extra={"h_left": h_left, "h_right": h_right},
        ),
    ]

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "data_source": "rdrobust::rdrobust_RDsenate",
            "n_obs": n,
            "bwselect": str(fit.model_info.get("bwselect", "mserd")),
            "h_left": h_left,
            "h_right": h_right,
        },
    )


if __name__ == "__main__":
    main()
