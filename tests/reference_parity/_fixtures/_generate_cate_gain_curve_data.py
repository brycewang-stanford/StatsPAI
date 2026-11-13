"""Data for ``test_cate_gain_curve_fklearn_parity.py``.

3,000 rows from a randomized experiment: a continuous dose ``dose`` and a
binary treatment ``d``, both assigned independently of everything else; an
effect that grows with ``x``; and two rankings, ``score`` (informative:
``x`` plus noise) and ``noise`` (uninformative). The rankings are written
with nine decimals and have no ties; the other columns have six.

    python tests/reference_parity/_fixtures/_generate_cate_gain_curve_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    rng = np.random.default_rng(20261117)
    n = 3000
    x = rng.uniform(0, 1, n)
    dose = rng.uniform(0, 10, n)
    d = (rng.uniform(size=n) < 0.5).astype(int)
    df = pd.DataFrame(
        {
            "y_dose": np.round((1 + 2 * x) * dose + rng.normal(size=n) * 3, 6),
            "y_bin": np.round((0.5 + 2 * x) * d + x + rng.normal(size=n), 6),
            "dose": np.round(dose, 6),
            "d": d,
            "score": np.round(x + rng.normal(size=n) * 0.3, 9),
            "noise": np.round(rng.normal(size=n), 9),
        }
    )
    assert df["score"].is_unique and df["noise"].is_unique
    df.to_csv(HERE / "cate_gain_curve.csv", index=False)
