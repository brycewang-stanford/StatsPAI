"""DoubleML reference for ``sp.dml(cluster=...)`` (one-way cluster DML).

Run::

    python tests/reference_parity/_fixtures/_generate_dml_cluster_doubleml.py

Writes ``dml_cluster_data.csv`` and ``dml_cluster_doubleml.json``. Every
model sp.dml dispatches to (PLR, IRM, PLIV, IIVM) is fitted by DoubleML on
``DoubleMLClusterData`` with one cluster variable, deterministic learners
(OLS / unpenalised logit) and the committed cluster-level folds, so the only
remaining difference is the estimator: the fold-weighted score solution and
the Chiang-Kato-Ma-Sasaki (2022) variance. As in
``_generate_ml_causal_doubleml.py``, the Python package is the reference
because R DoubleML refuses externally set sample splits with clustered data.
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
K = 4


def _logit() -> LogisticRegression:
    return LogisticRegression(penalty=None, max_iter=1000)


def make_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260926)
    n_clusters, size = 60, 12
    n = n_clusters * size
    g = np.repeat(np.arange(n_clusters), size)
    u = rng.normal(size=n_clusters)[g]
    x1 = rng.normal(size=n) + 0.5 * u
    x2 = rng.normal(size=n)
    z = (rng.normal(size=n) + 0.3 * x1 > 0).astype(float)
    d = 0.9 * z + 0.6 * x1 - 0.3 * x2 + 0.5 * u + rng.normal(size=n)
    db = (0.5 * x1 - 0.4 * x2 + 0.3 * u + rng.normal(size=n) > 0).astype(float)
    di = (0.8 * z + 0.3 * x1 + rng.normal(size=n) > 0.5).astype(float)
    eps = rng.normal(size=(3, n))
    df = pd.DataFrame(
        {
            "g": g,
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
    perm = np.random.default_rng(3).permutation(n_clusters)
    cluster_fold = np.empty(n_clusters, dtype=int)
    cluster_fold[perm] = np.arange(n_clusters) % K
    df["fold"] = cluster_fold[g]
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
    df.to_csv(HERE / "dml_cluster_data.csv", index=False, float_format="%.17g")
    df = pd.read_csv(HERE / "dml_cluster_data.csv")  # the bytes the test reads
    fold = df["fold"].to_numpy()
    cluster_fold = df.groupby("g")["fold"].first().to_numpy()
    smpls = [[(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(K)]]
    smpls_cluster = [
        [
            ([np.flatnonzero(cluster_fold != k)], [np.flatnonzero(cluster_fold == k)])
            for k in range(K)
        ]
    ]
    out = {}
    for name, (cls, y, d, z, learners) in MODELS.items():
        data = dml.DoubleMLClusterData(
            df, y_col=y, d_cols=d, cluster_cols="g", x_cols=["x1", "x2"], z_cols=z
        )
        obj = cls(data, n_folds=K, draw_sample_splitting=False, **learners())
        obj.set_sample_splitting(smpls, all_smpls_cluster=smpls_cluster)
        obj.fit()
        out[name] = {"theta": float(obj.coef[0]), "se": float(obj.se[0])}
    out["_meta"] = {
        "doubleml": dml.__version__,
        "sklearn": sklearn.__version__,
        "n_folds": K,
        "n_clusters": int(df["g"].nunique()),
    }
    (HERE / "dml_cluster_doubleml.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote dml_cluster_data.csv and dml_cluster_doubleml.json")


if __name__ == "__main__":
    main()
