"""Staggered panel with two units treated from the first period (no
untreated history, so their fixed effects cannot be imputed), for
``sp.did_imputation(autosample=True)``.

    python tests/reference_parity/_fixtures/_generate_bjs_autosample_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "bjs_autosample.csv"


def main() -> None:
    rng = np.random.default_rng(11)
    units, periods = 60, 8
    d = pd.DataFrame(
        dict(
            i=np.repeat(np.arange(units), periods),
            t=np.tile(np.arange(1, periods + 1), units),
        )
    )
    g = rng.choice([0, 4, 6], units).astype(float)
    g[5] = 1
    g[9] = 1
    d["g"] = np.repeat(g, periods)
    d.loc[d.g == 0, "g"] = np.nan
    d["w"] = np.repeat(rng.uniform(0.5, 2, units), periods)
    effect = np.where(d.t >= d.g.fillna(99), 1 + 0.2 * (d.t - d.g.fillna(0)), 0)
    d["y"] = (
        np.repeat(rng.normal(size=units), periods)
        + 0.1 * d.t
        + effect
        + rng.normal(size=len(d))
    )
    d.to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
