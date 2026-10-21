"""Write the two panels of ``test_cs_parity_switch.py``.

``cs_parity_mpdta.csv`` is the real ``mpdta`` (Callaway and Sant'Anna 2021;
the extract shipped with R ``did`` and used by Stata ``csdid``'s examples),
read from a Stata copy of it at full double precision, plus ``state`` (the
first digits of the county FIPS code, 29 values) as a time-invariant
cluster variable.

``cs_parity_always.csv`` is the same panel with 35 never-treated counties
recoded as always treated: 20 to ``first_treat = 2003`` (treated in the
first period) and 15 to ``first_treat = 2001`` (before the panel starts).
Their outcome is lowered by 0.05 so that any leak of these units into an
estimate would show.

    python tests/reference_parity/_fixtures/_generate_cs_parity_data.py /path/to/mpdta.dta

Run from the repository root.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent


def main(dta: str) -> None:
    d = pd.read_stata(dta)
    d = d[["countyreal", "year", "lpop", "lemp", "first_treat"]].astype(
        {"countyreal": "int64", "year": "int64", "first_treat": "int64"}
    )
    d["state"] = d["countyreal"] // 1000
    d = d.sort_values(["countyreal", "year"]).reset_index(drop=True)
    d.to_csv(HERE / "cs_parity_mpdta.csv", index=False)

    rng = np.random.default_rng(20261021)
    never = np.sort(d.loc[d["first_treat"] == 0, "countyreal"].unique())
    pick = rng.choice(never, 35, replace=False)
    a = d.copy()
    a.loc[a["countyreal"].isin(pick[:20]), "first_treat"] = 2003
    a.loc[a["countyreal"].isin(pick[20:]), "first_treat"] = 2001
    a.loc[a["first_treat"].isin([2001, 2003]), "lemp"] += -0.05
    a.to_csv(HERE / "cs_parity_always.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
