"""DoubleML reference for ``sp.dml(cluster=[c1, c2])`` (two-way cluster DML).

Run::

    python tests/reference_parity/_fixtures/_generate_dml_twoway_doubleml.py

Writes ``dml_twoway_data.csv`` and ``dml_twoway_doubleml.json``. Each model
sp.dml dispatches to is fitted by Python DoubleML (BSD-3) on
``DoubleMLClusterData`` with two cluster variables, deterministic learners
and committed per-dimension cluster folds (``fold1`` / ``fold2``): the
``K x K`` cells test units in fold ``k`` of the first and fold ``l`` of the
second dimension and train on units in neither. The remaining difference
is the estimator: the cell-weighted score solution and the two-way
Chiang-Kato-Ma-Sasaki (2022) variance.
"""

from __future__ import annotations

import json
import pathlib

import doubleml as dml
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression

HERE = pathlib.Path(__file__).parent
K = 3


def _logit() -> LogisticRegression:
    return LogisticRegression(penalty=None, max_iter=1000)


def make_data() -> pd.DataFrame:
    rng = np.random.default_rng(20261019)
    G1, G2, n = 30, 24, 1500
    c1 = rng.integers(0, G1, n)
    c2 = rng.integers(0, G2, n)
    a1, a2 = rng.normal(size=G1), rng.normal(size=G2)
    x1 = rng.normal(size=n) + 0.5 * a1[c1]
    x2 = rng.normal(size=n) + 0.5 * a2[c2]
    z = (rng.normal(size=n) + 0.3 * x1 > 0).astype(float)
    d = 0.9 * z + 0.6 * x1 - 0.3 * x2 + 0.5 * a1[c1] + rng.normal(size=n)
    db = (0.5 * x1 - 0.4 * x2 + 0.3 * a2[c2] + rng.normal(size=n) > 0).astype(float)
    di = (0.8 * z + 0.3 * x1 + rng.normal(size=n) > 0.5).astype(float)
    u = a1[c1] + a2[c2]
    eps = rng.normal(size=(3, n))
    df = pd.DataFrame(
        {
            "c1": c1,
            "c2": c2,
            "x1": x1,
            "x2": x2,
            "z": z,
            "d": d,
            "db": db,
            "di": di,
            "y": 1 + 1.5 * d + x1 - 0.5 * x2 + u + eps[0],
            "yb": 1 + 1.5 * db + x1 - 0.5 * x2 + u + eps[1],
            "yi": 1 + 1.5 * di + x1 - 0.5 * x2 + u + eps[2],
        }
    )
    for j, (col, G) in enumerate((("c1", G1), ("c2", G2))):
        perm = np.random.default_rng(10 + j).permutation(G)
        f = np.empty(G, dtype=int)
        f[perm] = np.arange(G) % K
        df[f"fold{j + 1}"] = f[df[col].to_numpy()]
    return df


MODELS = {
    "plr": (
        dml.DoubleMLPLR,
        "y",
        "d",
        None,
        lambda: dict(ml_l=LinearRegression(), ml_m=LinearRegression()),
    ),
    "irm": (
        dml.DoubleMLIRM,
        "yb",
        "db",
        None,
        lambda: dict(ml_g=LinearRegression(), ml_m=_logit()),
    ),
    "pliv": (
        dml.DoubleMLPLIV,
        "y",
        "d",
        "z",
        lambda: dict(
            ml_l=LinearRegression(), ml_m=LinearRegression(), ml_r=LinearRegression()
        ),
    ),
    "iivm": (
        dml.DoubleMLIIVM,
        "yi",
        "di",
        "z",
        lambda: dict(ml_g=LinearRegression(), ml_m=_logit(), ml_r=_logit()),
    ),
}


def main() -> None:
    df = make_data()
    df.to_csv(HERE / "dml_twoway_data.csv", index=False, float_format="%.17g")
    df = pd.read_csv(HERE / "dml_twoway_data.csv")  # the bytes the test reads
    f1, f2 = df["fold1"].to_numpy(), df["fold2"].to_numpy()
    cf1 = df.groupby("c1")["fold1"].first()
    cf2 = df.groupby("c2")["fold2"].first()
    smpls, smpls_cluster = [], []
    for k in range(K):
        for m in range(K):
            smpls.append(
                (
                    np.flatnonzero((f1 != k) & (f2 != m)),
                    np.flatnonzero((f1 == k) & (f2 == m)),
                )
            )
            smpls_cluster.append(
                (
                    [cf1.index[cf1 != k].to_numpy(), cf2.index[cf2 != m].to_numpy()],
                    [cf1.index[cf1 == k].to_numpy(), cf2.index[cf2 == m].to_numpy()],
                )
            )
    out = {}
    for name, (cls, y, d, z, learners) in MODELS.items():
        data = dml.DoubleMLClusterData(
            df,
            y_col=y,
            d_cols=d,
            cluster_cols=["c1", "c2"],
            x_cols=["x1", "x2"],
            z_cols=z,
        )
        obj = cls(data, n_folds=K, draw_sample_splitting=False, **learners())
        obj.set_sample_splitting([smpls], all_smpls_cluster=[smpls_cluster])
        obj.fit()
        out[name] = {"theta": float(obj.coef[0]), "se": float(obj.se[0])}
    out["_meta"] = {
        "doubleml": dml.__version__,
        "sklearn": sklearn.__version__,
        "n_folds": K,
        "n_clusters": [int(df["c1"].nunique()), int(df["c2"].nunique())],
    }
    (HERE / "dml_twoway_doubleml.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote dml_twoway_data.csv and dml_twoway_doubleml.json")


if __name__ == "__main__":
    main()
