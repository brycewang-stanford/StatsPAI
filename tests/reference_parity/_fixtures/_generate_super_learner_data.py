"""Data for ``test_super_learner_R_parity.py``.

400 rows, four covariates, a continuous outcome ``y`` (nonlinear in ``x1``,
so no single learner of the library is right) and a binary outcome ``yb``.
``fold_reg`` and ``fold_clf`` are the validation folds ``sp.super_learner``
draws with ``n_folds=5, random_state=42`` (``KFold`` for regression,
``StratifiedKFold`` on the outcome for classification); the R script hands
them to ``SuperLearner`` as ``validRows``, so both sides cross-validate on
the same split. Values are written with six decimals.

    python tests/reference_parity/_fixtures/_generate_super_learner_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    rng = np.random.default_rng(20261113)
    n = 400
    X = rng.normal(size=(n, 4))
    y = (
        1.0
        + X[:, 0]
        + 0.8 * X[:, 0] ** 2
        - 0.5 * X[:, 1]
        + 0.3 * X[:, 2] * X[:, 3]
        + rng.normal(size=n)
    )
    eta = -0.3 + 0.9 * X[:, 0] - 0.6 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]
    yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    df = pd.DataFrame(np.round(X, 6), columns=["x1", "x2", "x3", "x4"])
    df["y"] = np.round(y, 6)
    df["yb"] = yb
    Xr = df[["x1", "x2", "x3", "x4"]].to_numpy()
    df["fold_reg"] = 0
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(Xr)):
        df.loc[te, "fold_reg"] = k + 1
    df["fold_clf"] = 0
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    for k, (_, te) in enumerate(splitter.split(Xr, df["yb"].to_numpy())):
        df.loc[te, "fold_clf"] = k + 1
    df.to_csv(HERE / "super_learner_data.csv", index=False)
    print(df.describe().T[["mean", "std"]])
