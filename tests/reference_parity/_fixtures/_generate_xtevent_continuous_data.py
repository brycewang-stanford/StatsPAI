"""Panel with a continuous policy that changes several times per unit, for
``sp.xtevent`` (Stata ``xtevent``, Freyaldenhoven et al.).

    python tests/reference_parity/_fixtures/_generate_xtevent_continuous_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "xtevent_continuous.csv"


def main() -> None:
    rng = np.random.default_rng(20261001)
    units, periods = 80, 20
    d = pd.DataFrame(
        dict(
            id=np.repeat(np.arange(1, units + 1), periods),
            t=np.tile(np.arange(1, periods + 1), units),
        )
    )
    z = np.zeros((units, periods))
    for i in range(units):
        level = 0.0
        for s in range(periods):
            if rng.uniform() < 0.12:
                level += rng.normal(0.5, 0.4)
            z[i, s] = level
    d["z"] = z.ravel()
    d["x"] = rng.normal(size=len(d))
    alpha = np.repeat(rng.normal(size=units), periods)
    gamma = np.tile(rng.normal(size=periods), units)
    d["y"] = alpha + gamma + 0.8 * d.z + 0.3 * d.x + rng.normal(size=len(d))
    d["cl"] = (d.id - 1) // 4
    d.to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
