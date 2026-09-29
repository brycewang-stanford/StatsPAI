"""Data for the ``sp.ssc`` preset fixture: 40 firms in 10 industries over
8 years (firm nested in the industry cluster), an instrument, a control.

    python tests/reference_parity/_fixtures/_generate_ssc_presets_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "ssc_presets.csv"


def main() -> None:
    rng = np.random.default_rng(20260929)
    rows = []
    a_f = rng.normal(size=40)
    a_i = rng.normal(size=10)
    for f in range(40):
        ind = f // 4
        for t in range(8):
            z = rng.normal()
            w = rng.normal()
            v = rng.normal()
            x = 0.7 * z + 0.3 * w + 0.5 * a_f[f] + v
            y = 0.5 * x + 0.2 * w + a_f[f] + a_i[ind] + 0.6 * v + rng.normal()
            rows.append(dict(firm=f, ind=ind, year=t, y=y, x=x, z=z, w=w))
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
