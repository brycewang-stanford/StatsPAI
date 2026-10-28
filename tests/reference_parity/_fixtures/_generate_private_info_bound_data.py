"""Data for tests/reference_parity/test_private_info_bound_stata_parity.py.

A synthetic HRS-like sample: two rejection classes, households of one or
two people, age 60-84, a pricing covariate, and an elicitation with focal
masses at 0 and 1 and continuous answers in between. Covariates are
continuous, so no two residuals tie (the author's ``makem.do`` breaks ties
by sort position).

Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_private_info_bound_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent


def main() -> None:
    rng = np.random.default_rng(20261028)
    n = 2400
    reject = rng.integers(0, 2, size=n)
    age = rng.integers(60, 85, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.uniform(size=n)
    v = rng.normal(size=n) * np.where(reject == 1, 0.8, 0.3)
    p = norm.cdf(-1.0 + 0.04 * (age - 70) + 0.4 * x1 + 0.3 * x2 + v)
    loss = (rng.uniform(size=n) < p).astype(int)
    noisy = norm.cdf(v + rng.normal(scale=0.5, size=n))
    u = rng.uniform(size=n)
    z = np.where(u < 0.15, 0.0, np.where(u > 0.9, 1.0, np.round(noisy, 2)))
    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "hhid": np.arange(n) // 2 + 1,
            "loss": loss,
            "z": z,
            "age": age,
            "agep": (age - 80) / 15.0,
            "x1": np.round(x1, 6),
            "x2": np.round(x2, 6),
            "reject_list": reject + 1,
            # The paper's age cells, nested in the rejection class.
            "agegrp": (age - 60) // 5 + 26 * reject,
        }
    )
    df.to_csv(HERE / "private_info_bound_data.csv", index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
