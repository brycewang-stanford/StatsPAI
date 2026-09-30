"""Data for the ``sp.regress(collinear='omit')`` fixture.

A cross-section with a three-level factor, its hand-made dummies and an
exact continuous dependence (``wsum = x + w1``), and a staggered panel with
no never-treated cohort whose full set of event-time dummies is collinear
with the unit and period effects (two must go).

    python tests/reference_parity/_fixtures/_generate_regress_collinear_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20260930)
    n = 300
    g = rng.integers(0, 3, n)
    x = rng.normal(size=n)
    w1 = x + 0.3 * (g == 1) + rng.normal(size=n)
    cs = pd.DataFrame(
        dict(
            y=0.5 * x + 0.4 * (g == 2) + rng.normal(size=n),
            g=g,
            x=x,
            d0=(g == 0).astype(int),
            d1=(g == 1).astype(int),
            d2=(g == 2).astype(int),
            w1=w1,
            wsum=x + w1,
        )
    )
    cs.to_csv(HERE / "regress_collinear_cs.csv", index=False, float_format="%.17g")

    units, periods = 60, 10
    es = pd.DataFrame(
        dict(
            u=np.repeat(np.arange(units), periods), t=np.tile(np.arange(periods), units)
        )
    )
    es["g"] = np.repeat(rng.choice([3, 5, 7], units), periods)
    rel = es.t - es.g
    for k in sorted(rel.unique()):
        if k != -1:
            es[f"e{'m' if k < 0 else 'p'}{abs(k)}"] = (rel == k).astype(int)
    es["y"] = (rel >= 0) * 1.0 + rng.normal(size=len(es))
    es.to_csv(HERE / "regress_collinear_es.csv", index=False, float_format="%.17g")
    print("wrote regress_collinear_cs.csv, regress_collinear_es.csv")


if __name__ == "__main__":
    main()
