"""Data for the ``sp.spec_curve(fe=)`` fixture: 48 firms in 12 industries
over 6 years plus one firm observed once (a singleton ``reghdfe`` drops).

    python tests/reference_parity/_fixtures/_generate_spec_curve_fe_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "spec_curve_fe.csv"


def main() -> None:
    rng = np.random.default_rng(20260930)
    rows = []
    a_f = rng.normal(size=49)
    a_t = rng.normal(size=6)
    a_i = rng.normal(size=12)
    for f in range(49):
        ind = f // 4 if f < 48 else 0
        for t in range(6 if f < 48 else 1):
            w = rng.normal()
            x = 0.5 * a_f[f] + 0.3 * a_t[t] + 0.3 * w + rng.normal()
            y = 0.4 * x + 0.3 * w + a_f[f] + a_t[t] + a_i[ind] + rng.normal()
            rows.append(dict(firm=f, ind=ind, year=t, y=y, x=x, w=w))
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
