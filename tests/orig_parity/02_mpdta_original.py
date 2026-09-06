"""StatsPAI original-data parity (Python side) -- Module 02.

Runs sp.callaway_santanna on the *original* did::mpdta extract.
"""

from __future__ import annotations

import statspai as sp

from _common import OrigRecord, read_csv, write_results

MODULE = "02_mpdta_original"


def main() -> None:
    df = read_csv(MODULE)
    n = len(df)

    fit = sp.callaway_santanna(
        df,
        y="lemp",
        g="first_treat",
        t="year",
        i="countyreal",
        estimator="reg",
        control_group="nevertreated",
    )
    dyn = sp.aggte(fit, type="dynamic")

    # The did vignette does not print a simple-aggregation ATT for
    # mpdta, so the simple row carries no published anchor: the claim
    # is same-byte parity against did::aggte(type="simple").  The
    # vignette *does* print the overall dynamic/event-study ATT
    # (-0.0772, xformla = ~1), which anchors the second row.
    rows = [
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="simple_ATT",
            estimate=float(fit.estimate),
            se=float(fit.se),
            n=n,
            published=None,
            citation="did::aggte(type='simple') on the same bytes; no printed vignette anchor",
        ),
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="dynamic_overall_ATT",
            estimate=float(dyn.estimate),
            se=float(dyn.se),
            n=n,
            published=-0.0772,
            citation="Callaway-Sant'Anna 'did' vignette (did-basics), overall dynamic ATT, xformla=~1",
        ),
    ]

    write_results(MODULE, "py", rows, extra={"data_source": "did::mpdta", "n_obs": n})


if __name__ == "__main__":
    main()
