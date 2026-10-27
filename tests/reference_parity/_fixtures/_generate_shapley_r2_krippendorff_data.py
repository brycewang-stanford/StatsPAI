"""Fixtures for the Shapley-R2 and Krippendorff-alpha reference tests.

Writes three CSVs next to this script; the Stata (.do) and R (.R) generators
in the same folder read them byte for byte:

* ``shapley_r2_data.csv``: n = 300, six correlated regressors, a positive
  weight column and an always-included control ``z``.
* ``krippendorff_data.csv``: 80 units (rows) x 5 coders (columns), values on
  a 1..6 scale, about 22% missing, one unit with a single coding and one
  with none.
* ``krippendorff_small.csv``: 12 units x 4 coders, the textbook example in
  the help file of Stata ``kalpha`` (there stored coders x units).

Floats are written with ``%.17g`` so every reader recovers the same
doubles (Stata reads them with ``import delimited ..., asdouble``).

Regenerate with:
    python tests/reference_parity/_fixtures/_generate_shapley_r2_krippendorff_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def shapley_data() -> pd.DataFrame:
    rng = np.random.default_rng(20261028)
    n = 300
    cov = 0.4 * np.ones((6, 6)) + 0.6 * np.eye(6)
    cov[0, 1] = cov[1, 0] = 0.8
    X = rng.multivariate_normal(np.zeros(6), cov, size=n)
    z = rng.normal(size=n) + 0.5 * X[:, 0]
    beta = np.array([0.8, 0.3, -0.5, 0.2, 0.0, 0.6])
    y = 1.0 + X @ beta + 0.7 * z + rng.normal(scale=1.5, size=n)
    w = rng.uniform(0.5, 2.0, size=n)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(1, 7)])
    df.insert(0, "y", y)
    df["z"] = z
    df["w"] = w
    return df


def krippendorff_data() -> pd.DataFrame:
    rng = np.random.default_rng(1970)
    n_units, n_coders = 80, 5
    truth = rng.integers(1, 7, size=n_units)
    vals = np.empty((n_units, n_coders))
    for j in range(n_coders):
        noise = rng.choice([-2, -1, 0, 0, 0, 0, 1, 2], size=n_units)
        vals[:, j] = np.clip(truth + noise, 1, 6)
    vals[rng.uniform(size=vals.shape) < 0.22] = np.nan
    # a few units with a single value (not pairable) and one all-missing row
    vals[3, 1:] = np.nan
    vals[17, :] = np.nan
    return pd.DataFrame(vals, columns=[f"c{j}" for j in range(1, n_coders + 1)])


def krippendorff_small() -> pd.DataFrame:
    # kalpha help-file example: 4 observers (rows there) x 12 units.
    obs = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
    ]
    arr = np.array([[np.nan if v is None else v for v in r] for r in obs])
    return pd.DataFrame(arr.T, columns=["c1", "c2", "c3", "c4"])


if __name__ == "__main__":
    shapley_data().to_csv(
        HERE / "shapley_r2_data.csv", index=False, float_format="%.17g"
    )
    krippendorff_data().to_csv(
        HERE / "krippendorff_data.csv", index=False, float_format="%.17g", na_rep=""
    )
    krippendorff_small().to_csv(
        HERE / "krippendorff_small.csv", index=False, float_format="%.17g", na_rep=""
    )
