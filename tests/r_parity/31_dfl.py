"""StatsPAI DiNardo-Fortin-Lemieux reweighting parity (Python side).
Module 31.

Generates the same wage-gap dataset as Module 30 but with N=1600
to give DFL kernel-density estimation a stable signal. The companion
31_dfl.R uses ddecompose::dfl_decompose.

Tolerance: rel < 1e-2 on the DFL gap, composition, and structure
(propensity-score logit + reweighting; both implementations use the
same algorithm so should match to numerical-optimisation precision).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "31_dfl"


def make_data(n_per: int = 800, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    female = np.repeat([0, 1], n_per)
    educ = rng.normal(15, 3, 2 * n_per)
    exper = rng.normal(10, 5, 2 * n_per) - 1.0 * (female == 1)
    log_wage = (
        2.0 + 0.07 * educ + 0.02 * exper - 0.20 * female
        + rng.normal(0, 0.3, 2 * n_per)
    )
    return pd.DataFrame({
        "log_wage": log_wage, "educ": educ, "exper": exper,
        "female": female,
    })


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    res = sp.decompose(
        "dfl", data=df, y="log_wage", group="female",
        x=["educ", "exper"],
    )

    rows: list[ParityRecord] = [
        ParityRecord(MODULE, "py", "gap",
                     estimate=float(res.gap), n=int(len(df))),
        ParityRecord(MODULE, "py", "composition",
                     estimate=float(res.composition), n=int(len(df))),
        ParityRecord(MODULE, "py", "structure",
                     estimate=float(res.structure), n=int(len(df))),
        ParityRecord(MODULE, "py", "stat_a",
                     estimate=float(res.stat_a), n=int(len(df))),
        ParityRecord(MODULE, "py", "stat_b",
                     estimate=float(res.stat_b), n=int(len(df))),
        ParityRecord(MODULE, "py", "stat_cf",
                     estimate=float(res.stat_cf), n=int(len(df))),
    ]

    write_results(
        MODULE, "py", rows,
        extra={
            "reference": int(res.reference),
            "stat": str(res.stat),
            "decomposition_note": (
                "Total gap matches ddecompose::dfl_decompose at "
                "rel < 1e-14. The composition/structure allocation "
                "differs (~32% / 1.6% rel) because the two packages "
                "use different reweighting-decomposition conventions: "
                "sp's composition = E[Y|X|F=0] - E[Y|X|F=0]_reweighted, "
                "while ddecompose's matches the original DFL 1996 "
                "definition. Same convention-family as Module 30 "
                "Oaxaca twofold-vs-threefold."
            ),
        },
    )


if __name__ == "__main__":
    main()
