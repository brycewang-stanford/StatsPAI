"""Data for the ``sp.suest`` factor / fixed-effect / two-way fixture.

A survey-like unbalanced panel: 60 respondents ``r`` observed in about
eight of 12 months ``m``; a realized outcome ``yr`` and a forecast ``yf``
of it (``yf`` missing on 12 rows, so the equations use different samples),
a regressor ``x``, a control ``w`` and two categorical controls, ``f``
(4 levels, coded 1-4) and ``g`` (5 levels, coded 10-50). Errors share a
respondent and a month shock, so both clustering dimensions matter.

    python tests/reference_parity/_fixtures/_generate_suest_fe_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "suest_fe_data.csv"


def main() -> None:
    rng = np.random.default_rng(20261028)
    rows = []
    for r in range(60):
        months = np.sort(rng.choice(12, size=rng.integers(6, 11), replace=False))
        rows += [(r, m) for m in months]
    d = pd.DataFrame(rows, columns=["r", "m"])
    n = len(d)
    a_r, a_m = rng.normal(size=60), rng.normal(size=12)
    f = rng.integers(1, 5, size=60)[d.r]
    g = 10 * rng.integers(1, 6, size=n)
    x = rng.normal(size=n) + 0.3 * a_m[d.m]
    w = rng.normal(size=n)
    shock = a_r[d.r] + a_m[d.m]
    d["yr"] = 0.6 * x + 0.2 * w + 0.3 * f + 0.01 * g + shock + rng.normal(size=n)
    d["yf"] = (
        0.4 * x - 0.2 * f + 0.02 * g + 0.8 * shock + 0.5 * a_m[d.m]
        + rng.normal(size=n)
    )  # fmt: skip
    d["x"], d["w"], d["f"], d["g"] = x, w, f, g
    d.loc[rng.choice(n, size=12, replace=False), "yf"] = np.nan
    d.to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT} ({n} rows)")


if __name__ == "__main__":
    main()
