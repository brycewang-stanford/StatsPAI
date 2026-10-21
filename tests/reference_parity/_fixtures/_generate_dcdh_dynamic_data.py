"""Panel for sp.did_multiplegt's dynamic / placebo / controls conventions.

Checked against Stata ``did_multiplegt_old`` (with and without
``robust_dynamic``) and R ``DIDmultiplegt`` 0.1.4, see
``_generate_dcdh_dynamic_Stata.do`` and ``_generate_dcdh_dynamic_R.R``.

Design: 60 groups, 9 periods, 1-4 individuals per (group, period) cell.
``dc`` is a count treatment (0-3) that moves up and down; groups start at 0,
1 or 2, some never move, a few switch for the first time in the last two
periods. ``db`` is a binary treatment with its own on/off path. ``x`` is a
control correlated with the treatment. ``keep`` (constant within a cell)
marks the unbalanced variant: about 6% of cells, and period 1 of two groups,
are dropped when ``keep == 0`` rows are removed.

Writes dcdh_dynamic_panel.csv next to this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _path(rng: np.random.Generator, start: int, first: int, top: int) -> list:
    """A treatment path that stays at ``start`` until ``first`` (1-based
    period of the first change, 99 = never), then wanders within [0, top]."""
    d = [start]
    for t in range(2, 10):
        prev = d[-1]
        if t < first:
            d.append(prev)
        elif t == first:
            step = 1 if prev == 0 else (-1 if prev == top else int(rng.choice([-1, 1])))
            d.append(prev + step)
        elif rng.random() < 0.35:
            step = 1 if prev == 0 else (-1 if prev == top else int(rng.choice([-1, 1])))
            d.append(prev + step)
        else:
            d.append(prev)
    return d


def build(seed: int = 20261022) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(60):
        start = [0, 0, 1, 2][g % 4]
        if g < 12:
            first = 99  # never switches: a not-yet-switched control throughout
        elif g < 18:
            first = int(rng.integers(8, 10))  # late first switch (8 or 9)
        else:
            first = int(rng.integers(2, 8))
        dc = _path(rng, start, first, 3)
        bstart = int(start > 0)
        bfirst = 99 if g < 10 else int(rng.integers(2, 10))
        db = _path(rng, bstart, bfirst, 1)
        fx = rng.normal(scale=0.5)
        cum = np.cumsum(dc)
        drop_first = g in (25, 41)
        for t in range(1, 10):
            keep = int(not (drop_first and t == 1) and rng.random() > 0.06)
            n = int(rng.integers(1, 5))
            for _ in range(n):
                x = rng.normal() + 0.4 * dc[t - 1]
                e = rng.normal()
                rows.append(
                    {
                        "g": g,
                        "t": t,
                        "dc": dc[t - 1],
                        "db": db[t - 1],
                        "x": x,
                        "keep": keep,
                        "y": fx
                        + 0.15 * t
                        + 0.6 * dc[t - 1]
                        + 0.1 * cum[t - 1]
                        + 0.5 * x
                        + e,
                        "yb": fx + 0.15 * t + 0.8 * db[t - 1] + 0.5 * x + e,
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).with_name("dcdh_dynamic_panel.csv")
    build().to_csv(out, index=False, float_format="%.17g")
    print(out)
