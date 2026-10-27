"""Fixtures for ``sp.cox(tvc=, texp=)`` vs Stata ``stcox, tvc() texp()``.

Writes ``cox_tvc_cont.csv`` (continuous times, no ties) and
``cox_tvc_ties.csv`` (integer times with heavy ties, strata, clusters,
probability weights, delayed entry). The log hazard ratio of ``race``
declines with time in both, so the ``tvc`` coefficient is far from zero.

Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_cox_tvc_data.py

then ``_generate_cox_tvc_Stata.do`` and ``_generate_cox_tvc_R.R``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def _times(rng, x1, race, n, scale):
    # piecewise-exponential draw with log HR of race = 0.8 - 0.25 t
    t = np.zeros(n)
    alive = np.ones(n, bool)
    grid = np.linspace(0.0, 12.0, 241)
    for a, b in zip(grid[:-1], grid[1:]):
        mid = 0.5 * (a + b)
        lam = scale * np.exp(0.4 * x1 + (0.8 - 0.25 * mid) * race)
        u = rng.exponential(1.0, n) / lam
        die = alive & (u < (b - a))
        t[die] = a + u[die]
        alive &= ~die
    t[alive] = 12.0 + rng.exponential(1.0, alive.sum())
    return t


def main() -> None:
    rng = np.random.default_rng(20261028)
    n = 400
    x1 = rng.normal(size=n)
    race = rng.binomial(1, 0.4, n).astype(float)
    t = _times(rng, x1, race, n, 0.15)
    c = rng.uniform(2.0, 14.0, n)
    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "time": np.minimum(t, c),
            "died": (t <= c).astype(int),
            "x1": x1,
            "race": race,
        }
    )
    df.to_csv(HERE / "cox_tvc_cont.csv", index=False, float_format="%.17g")

    n = 600
    x1 = rng.normal(size=n)
    race = rng.binomial(1, 0.45, n).astype(float)
    grp = rng.integers(0, 2, n)
    t = _times(rng, x1, race, n, 0.12 * np.exp(0.3 * grp))
    c = rng.uniform(3.0, 15.0, n)
    time = np.ceil(np.minimum(t, c))  # integer times: many ties
    died = (t <= c).astype(int)
    entry = np.where(rng.random(n) < 0.3, rng.integers(0, 3, n), 0).astype(float)
    entry = np.minimum(entry, time - 1)  # start < stop on every row
    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "time": time,
            "died": died,
            "x1": x1,
            "race": race,
            "grp": grp,
            "cl": rng.integers(1, 41, n),
            "w": np.round(rng.uniform(0.5, 2.5, n), 3),
            "entry": entry,
        }
    )
    df.to_csv(HERE / "cox_tvc_ties.csv", index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
