"""StatsPAI original-data parity (Python side) -- Module 03.

Runs sp.synth(method='classic') on the *original* Synth::basque
data (Abadie-Gardeazabal 2003).
"""
from __future__ import annotations

import statspai as sp

from _common import OrigRecord, read_csv, write_results


MODULE = "03_basque_original"


def main() -> None:
    df = read_csv(MODULE)
    n = len(df)

    fit = sp.synth(
        df, outcome="gdppc", unit="region", time="year",
        treated_unit="Basque Country (Pais Vasco)",
        treatment_time=1970,
        method="classic",
    )

    rows = [
        OrigRecord(
            module=MODULE, side="py", statistic="avg_post_gap",
            estimate=float(fit.estimate),
            se=float(fit.se) if fit.se is not None else None,
            n=n, published=-0.855,
            citation="Abadie-Gardeazabal (2003) Figure 2 / Synth vignette",
        ),
    ]

    write_results(MODULE, "py", rows,
                  extra={"data_source": "Synth::basque", "n_obs": n})


if __name__ == "__main__":
    main()
