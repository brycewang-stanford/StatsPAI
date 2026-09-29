"""Panel for the weak-IV (AR / effective F) fixture with nested fixed effects.

90 units in 9 regions over 12 years; the year effect is nested in the
region x year effect and the unit effect in the unit clusters -- the
structure of the Web of Power (QJE 2023) Table 4 IV. One instrument of
moderate strength, one exogenous control.

    python tests/reference_parity/_fixtures/_generate_weakiv_nested_fe_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "weakiv_nested_fe.csv"


def main() -> None:
    rng = np.random.default_rng(20260929)
    rows = []
    a_u = rng.normal(size=90)
    a_rt = rng.normal(size=(9, 12))
    for u in range(90):
        r = u // 10
        for t in range(12):
            z = rng.normal() + 0.5 * (t >= 6) * (u % 3 == 0)
            w1 = rng.normal()
            v = rng.normal()
            d = 0.25 * z + 0.4 * w1 + v + a_u[u] * 0.3
            y = 0.5 * d + 0.3 * w1 + a_u[u] + a_rt[r, t] + 0.8 * v + rng.normal()
            rows.append(
                dict(
                    unit=u,
                    region=r,
                    year=2000 + t,
                    rXy=r * 100 + t,
                    y=y,
                    d=d,
                    z=z,
                    w1=w1,
                )
            )
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
