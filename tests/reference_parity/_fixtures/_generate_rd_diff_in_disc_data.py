"""Pre/post RD data for the ``sp.rd_diff_in_disc`` fixture.

150 sites observed before and after a policy that switches on at the cutoff
only in the post period; site effects correlate the two periods.

    python tests/reference_parity/_fixtures/_generate_rd_diff_in_disc_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "rd_diff_in_disc.csv"


def main() -> None:
    rng = np.random.default_rng(20260929)
    rows = []
    for s in range(150):
        a = rng.normal(scale=0.5)
        for per in (0, 1):
            for _ in range(10):
                x = rng.uniform(-1, 1)
                above = float(x >= 0)
                y = (
                    1.0 * x
                    + 0.4 * x**2
                    + 0.3 * above
                    + 0.5 * above * per
                    + 0.2 * per
                    + a
                    + rng.normal(scale=0.4)
                )
                rows.append(dict(site=s, post=per, x=x, y=y))
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
