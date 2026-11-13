"""DoubleML reference for ``sp.dml_sensitivity(benchmark_covariates=)`` and
for ``sp.dml(model='iivm')`` under one-sided non-compliance.

Run in an environment with ``doubleml`` (0.11.4 here; it is not a StatsPAI
dependency)::

    python tests/reference_parity/_fixtures/_generate_dml_benchmark_doubleml.py

Reads ``dml_benchmark_data.csv`` (see ``_generate_dml_benchmark_data.py``)
and writes ``dml_benchmark_doubleml.json``. Learners are deterministic (OLS
and an unpenalised logit solved by Newton) and the folds are the ``fold``
column, so the only remaining difference between the packages is the
estimator.

* ``benchmarks``: ``DoubleML.sensitivity_benchmark(benchmarking_set=...)``
  for PLR and IRM (ATE) of ``re78`` on ``treat``: ``cf_y``, ``cf_d``,
  ``rho`` and ``delta_theta`` for three benchmark sets.
* ``iivm_onesided``: ``DoubleMLIIVM`` of ``yz`` on ``dz`` with instrument
  ``z`` and ``subgroups={'always_takers': False, 'never_takers': True}``,
  DoubleML's way of saying that ``P(D = 1 | Z = 0, X) = 0``.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import doubleml as dml
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression

HERE = pathlib.Path(__file__).parent
X = ["age", "educ", "black", "hispanic", "married", "nodegree", "re74", "re75"]
SETS = [["re74"], ["black"], ["re74", "re75"]]


def _logit() -> LogisticRegression:
    return LogisticRegression(C=np.inf, solver="newton-cholesky", max_iter=200)


def main() -> None:
    warnings.simplefilter("ignore")
    df = pd.read_csv(HERE / "dml_benchmark_data.csv")
    fold = df["fold"].to_numpy()
    smpls = [[(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(5)]]
    out: dict = {"benchmarks": {}}
    data = dml.DoubleMLData(df, "re78", "treat", X)
    models = {
        "plr": lambda: dml.DoubleMLPLR(
            data, LinearRegression(), LinearRegression(), draw_sample_splitting=False
        ),
        "irm": lambda: dml.DoubleMLIRM(
            data, LinearRegression(), _logit(), draw_sample_splitting=False
        ),
    }
    for name, make in models.items():
        obj = make()
        obj.set_sample_splitting(smpls)
        obj.fit()
        rows = {}
        for group in SETS:
            b = obj.sensitivity_benchmark(benchmarking_set=group).iloc[0]
            rows["+".join(group)] = {k: float(b[k]) for k in b.index}
        out["benchmarks"][name] = {
            "theta": float(obj.coef[0]),
            "se": float(obj.se[0]),
            "sets": rows,
        }
    data_iv = dml.DoubleMLData(df, "yz", "dz", X, z_cols="z")
    iivm = dml.DoubleMLIIVM(
        data_iv,
        LinearRegression(),
        _logit(),
        _logit(),
        subgroups={"always_takers": False, "never_takers": True},
        draw_sample_splitting=False,
    )
    iivm.set_sample_splitting(smpls)
    iivm.fit()
    out["iivm_onesided"] = {"theta": float(iivm.coef[0]), "se": float(iivm.se[0])}
    out["_meta"] = {"doubleml": dml.__version__, "sklearn": sklearn.__version__}
    (HERE / "dml_benchmark_doubleml.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote dml_benchmark_doubleml.json")


if __name__ == "__main__":
    main()
