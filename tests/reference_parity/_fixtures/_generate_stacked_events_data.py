"""Panel with repeated (non-absorbing) events per unit -- e.g. several
minimum-wage increases in one state -- for ``sp.stacked_did(events=)``.

    python tests/reference_parity/_fixtures/_generate_stacked_events_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "stacked_events.csv"


def main() -> None:
    rng = np.random.default_rng(20261002)
    units, periods = 40, 30
    d = pd.DataFrame(
        dict(
            unit=np.repeat(np.arange(1, units + 1), periods),
            t=np.tile(np.arange(1, periods + 1), units),
        )
    )
    ev = np.zeros((units, periods), dtype=int)
    for i in range(units):
        if i < 12:  # never an event
            continue
        n_ev = rng.integers(1, 4)
        for s in rng.choice(np.arange(4, periods - 2), size=n_ev, replace=False):
            ev[i, s] = 1
    d["event"] = ev.ravel()
    d["w"] = np.repeat(rng.uniform(0.5, 3.0, units), periods)
    cum = ev.cumsum(axis=1).ravel()  # effect accumulates with each event
    d["y"] = (
        np.repeat(rng.normal(size=units), periods)
        + np.tile(rng.normal(size=periods), units)
        + 0.5 * cum
        + rng.normal(size=len(d))
    )
    d.to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
