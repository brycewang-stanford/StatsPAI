"""Data for ``test_metalearner_tools_parity.py``.

``metalearner_train.csv``: 1,200 rows, covariates ``x1``-``x3``, a binary
treatment ``d`` with a logistic propensity, an outcome ``y`` whose effect
``1 + 0.8 x1 - 0.5 x2`` varies with the covariates, and ``fold``: the
labels 0-4 of ``KFold(5, shuffle=True, random_state=42)``, handed to both
sides so the cross-fitting uses one partition. ``metalearner_new.csv``:
200 further rows of covariates for out-of-sample prediction. Six decimals.

    python tests/reference_parity/_fixtures/_generate_metalearner_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    rng = np.random.default_rng(20261118)
    n, m = 1200, 200
    X = np.round(rng.normal(size=(n + m, 3)), 6)
    p = 1 / (1 + np.exp(-(0.2 + 0.6 * X[:, 0] - 0.4 * X[:, 2])))
    d = (rng.uniform(size=n + m) < p).astype(int)
    tau = 1 + 0.8 * X[:, 0] - 0.5 * X[:, 1]
    y = 0.5 + X @ np.array([0.7, -0.3, 0.4]) + tau * d + rng.normal(size=n + m)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    df["d"] = d
    df["y"] = np.round(y, 6)
    train = df.iloc[:n].copy()
    train["fold"] = -1
    splitter = KFold(5, shuffle=True, random_state=42)
    for k, (_, te) in enumerate(splitter.split(train[["x1", "x2", "x3"]])):
        train.iloc[te, train.columns.get_loc("fold")] = k
    train.to_csv(HERE / "metalearner_train.csv", index=False)
    df.iloc[n:][["x1", "x2", "x3"]].to_csv(HERE / "metalearner_new.csv", index=False)
    print(train["d"].mean(), train["fold"].value_counts().to_dict())
