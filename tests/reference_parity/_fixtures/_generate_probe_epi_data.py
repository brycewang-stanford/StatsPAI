"""Data for the 2026-11 epidemiology probe fixtures.

Writes ``probe_epi_survival.csv``: 320 subjects, two arms, three strata,
integer follow-up times with heavy ties, delayed entry, non-integer case
weights and three covariates. The last observation of arm 0 is an event
and that of arm 1 is censored, which is the case in which the default
``tau`` of ``survRM2::rmst2`` is not the smaller of the arms' largest
times.

Writes ``probe_epi_meta.csv``: 14 studies with a heterogeneous effect.

Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_probe_epi_data.py

then ``Rscript tests/reference_parity/_fixtures/_generate_probe_epi_R.R``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def survival() -> pd.DataFrame:
    rng = np.random.default_rng(20261113)
    n = 320
    arm = rng.integers(0, 2, n)
    site = rng.integers(0, 3, n)
    age = np.round(rng.normal(60, 9, n))
    ecog = rng.integers(0, 3, n)
    haz = 0.03 * np.exp(0.02 * (age - 60) + 0.45 * ecog - 0.5 * arm + 0.2 * site)
    t = rng.exponential(1.0 / haz)
    c = rng.uniform(5, 70, n)
    time = np.ceil(np.minimum(t, c))  # integer days: many ties
    event = (t <= c).astype(int)
    # delayed entry strictly before the exit time
    entry = np.floor(rng.uniform(0, 0.5, n) * (time - 1))
    w = np.round(rng.uniform(0.4, 2.5, n), 3)
    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "arm": arm,
            "site": site,
            "age": age,
            "ecog": ecog,
            "entry": entry,
            "time": time,
            "event": event,
            "w": w,
        }
    )
    # arm 0 ends in an event, arm 1 in a censored observation
    i0 = df.index[df.arm == 0][-1]
    df.loc[i0, ["time", "event", "entry"]] = [df.time.max() + 3, 1, 0]
    i1 = df.index[df.arm == 1][-1]
    df.loc[i1, ["time", "event", "entry"]] = [df.time.max() + 4, 0, 0]
    return df


def meta() -> pd.DataFrame:
    rng = np.random.default_rng(1113)
    k = 14
    se = np.round(rng.uniform(0.08, 0.45, k), 4)
    theta = rng.normal(-0.35, 0.3, k)
    yi = np.round(theta + rng.normal(0, se), 4)
    return pd.DataFrame(
        {"study": [f"S{i + 1:02d}" for i in range(k)], "yi": yi, "sei": se}
    )


if __name__ == "__main__":
    survival().to_csv(HERE / "probe_epi_survival.csv", index=False)
    meta().to_csv(HERE / "probe_epi_meta.csv", index=False)
