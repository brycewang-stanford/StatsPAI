"""StatsPAI original-data parity (Python side) -- Module 03.

Runs the canonical ADH/Synth specification on the *original*
Synth::basque data (Abadie-Gardeazabal 2003).
"""

from __future__ import annotations

import statspai as sp

from _common import OrigRecord, read_csv, write_results

MODULE = "03_basque_original"


def main() -> None:
    df = read_csv(MODULE)
    n = len(df)
    pre_years = list(range(1955, 1970))
    special_predictors = [("gdppc", year, "mean") for year in pre_years]

    fit = sp.synth(
        df,
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country (Pais Vasco)",
        treatment_time=1970,
        method="classic",
        special_predictors=special_predictors,
        n_random_starts=0,
        placebo=False,
    )

    rows = [
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="avg_post_gap",
            estimate=float(fit.estimate),
            se=float(fit.se) if fit.se is not None else None,
            n=n,
            published=-0.855,
            citation="Abadie-Gardeazabal (2003) Figure 2 / Synth vignette",
        ),
    ]

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "data_source": "Synth::basque",
            "n_obs": n,
            "specification": (
                "ADH/Synth canonical setup: each 1955-1969 pre-treatment "
                "outcome year is a special predictor and V is nested-"
                "optimized against pre-period SSR."
            ),
        },
    )


if __name__ == "__main__":
    main()
