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
        df, y="lemp", g="first_treat", t="year", i="countyreal",
        estimator="reg", control_group="nevertreated",
    )

    rows = [
        OrigRecord(
            module=MODULE, side="py", statistic="simple_ATT",
            estimate=float(fit.estimate), se=float(fit.se),
            n=n, published=-0.0454,
            citation="Callaway-Sant'Anna (2021) R 'did' vignette aggte simple",
        ),
    ]

    write_results(MODULE, "py", rows,
                  extra={"data_source": "did::mpdta", "n_obs": n})


if __name__ == "__main__":
    main()
