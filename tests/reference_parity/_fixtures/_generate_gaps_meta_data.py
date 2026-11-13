"""Data for the meta-analysis items of the 2026-11 gap list.

Writes ``gaps_meta.csv``: 22 studies with two moderators and a small-study
effect (the less precise a study, the larger its reported effect, and the
least favourable imprecise studies are missing), so that trim-and-fill has
something to impute and the moderators explain part of the heterogeneity.

* ``yi``, ``sei``: effect and standard error.
* ``dose``: continuous moderator.
* ``blind``: 0 / 1 moderator.

Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_gaps_meta_data.py

then ``Rscript tests/reference_parity/_fixtures/_generate_gaps_meta_R.R``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def meta() -> pd.DataFrame:
    rng = np.random.default_rng(20261113)
    k_all = 40
    se = rng.uniform(0.06, 0.5, k_all)
    dose = np.round(rng.uniform(1, 10, k_all), 1)
    blind = rng.integers(0, 2, k_all)
    theta = 0.15 + 0.04 * dose - 0.2 * blind + rng.normal(0, 0.18, k_all)
    yi = theta + rng.normal(0, se)
    # selection: imprecise studies are reported only when they look favourable
    keep = (se < 0.2) | (yi / se > 0.9)
    df = pd.DataFrame(
        {
            "yi": np.round(yi, 4),
            "sei": np.round(se, 4),
            "dose": dose,
            "blind": blind,
        }
    )[keep].head(22)
    df.insert(0, "study", [f"T{i + 1:02d}" for i in range(len(df))])
    return df.reset_index(drop=True)


if __name__ == "__main__":
    out = meta()
    out.to_csv(HERE / "gaps_meta.csv", index=False)
    print(f"wrote gaps_meta.csv ({len(out)} studies)")
