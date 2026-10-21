"""Write the two data sets of ``test_cs_parity_rc_unbal.py``.

Both are built from ``cs_parity_mpdta.csv`` (the real ``mpdta``, written by
``_generate_cs_parity_data.py``), so the outcome, the covariate ``lpop``,
the cohorts and ``state`` are the real ones.

``cs_parity_rcs.csv`` is a repeated cross-section: each county-year is kept
independently with a probability that varies by year (0.55 to 0.85), and
every kept row gets its own id ``rid``. Cohort counts then differ from year
to year, which is what separates Stata ``csdid``'s per-cell weights from R
``did``'s cohort shares.

``cs_parity_unbal.csv`` is an unbalanced panel: 12 percent of county-years
are deleted at random (each county keeps at least one year). About half of
the counties stay complete, so R ``did``'s default (keep complete units)
and csdid's (each cell uses the units seen in both of its periods) give
different numbers.

Both files carry a sampling weight ``w`` (drawn from U(0.2, 3), three
decimals, from its own seed so the rows kept above do not depend on it). On
the repeated cross-section every row gets its own weight; on the panel the
weight is a county's, constant over its years, as R ``did`` and Stata
``csdid`` require of a panel weight. Unweighted cases ignore the column. The
weighted cases on the balanced panel take each county's ``w`` from
``cs_parity_unbal.csv`` (every county keeps a row there) and attach it to
``cs_parity_mpdta.csv``, in R, Stata and the test alike.

    python tests/reference_parity/_fixtures/_generate_cs_parity_rc_unbal_data.py

Run from the repository root.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
_KEEP_RCS = {2003: 0.55, 2004: 0.85, 2005: 0.65, 2006: 0.75, 2007: 0.6}


def _weights(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sampling weights in [0.2, 3], rounded so the CSV holds them exactly."""
    return np.round(rng.uniform(0.2, 3.0, n), 3)


def main() -> None:
    d = pd.read_csv(HERE / "cs_parity_mpdta.csv")

    rng = np.random.default_rng(20261022)
    p = d["year"].map(_KEEP_RCS).to_numpy()
    rcs = d.loc[rng.random(len(d)) < p].copy()
    rcs.insert(0, "rid", np.arange(1, len(rcs) + 1, dtype="int64"))
    rcs = rcs.drop(columns="countyreal").reset_index(drop=True)
    rcs["w"] = _weights(np.random.default_rng(20261024), len(rcs))
    rcs.to_csv(HERE / "cs_parity_rcs.csv", index=False)

    rng = np.random.default_rng(20261023)
    drop = rng.random(len(d)) < 0.12
    u = d.loc[~drop].copy()
    # every county keeps at least one row
    lost = sorted(set(d["countyreal"]) - set(u["countyreal"]))
    if lost:
        u = pd.concat([u, d[d["countyreal"].isin(lost)].groupby("countyreal").head(1)])
    u = u.sort_values(["countyreal", "year"]).reset_index(drop=True)
    counties = np.sort(d["countyreal"].unique())
    w_county = dict(
        zip(counties, _weights(np.random.default_rng(20261025), len(counties)))
    )
    u["w"] = u["countyreal"].map(w_county)
    u.to_csv(HERE / "cs_parity_unbal.csv", index=False)


if __name__ == "__main__":
    main()
