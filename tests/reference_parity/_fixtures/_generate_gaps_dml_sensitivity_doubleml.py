"""DoubleML reference for ``sp.dml_sensitivity(rho=, level=)``.

Run in an environment with ``doubleml`` (0.11.4 here; it is not a StatsPAI
dependency)::

    python tests/reference_parity/_fixtures/_generate_gaps_dml_sensitivity_doubleml.py

Reads ``dml_benchmark_data.csv`` (see ``_generate_dml_benchmark_data.py``)
and writes ``gaps_dml_sensitivity_doubleml.json``. Learners are deterministic
(OLS and an unpenalised logit solved by Newton) and the folds are the
``fold`` column, so the only remaining difference between the packages is
the estimator.

For PLR and IRM (ATE) of ``re78`` on ``treat`` and each ``(cf_y, cf_d, rho,
level)`` scenario it stores ``sensitivity_params`` after
``sensitivity_analysis(cf_y, cf_d, rho, level, null_hypothesis=0)``: the
two bounds, their standard errors, the interval for the range, and the
robustness values ``rv`` and ``rva``.
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
SCENARIOS = [
    {"cf_y": 0.03, "cf_d": 0.03, "rho": 1.0, "level": 0.95},
    {"cf_y": 0.03, "cf_d": 0.03, "rho": 0.5, "level": 0.95},
    {"cf_y": 0.03, "cf_d": 0.03, "rho": 1.0, "level": 0.90},
    {"cf_y": 0.10, "cf_d": 0.05, "rho": 0.3, "level": 0.99},
    {"cf_y": 0.10, "cf_d": 0.05, "rho": -0.7, "level": 0.80},
]


def _logit() -> LogisticRegression:
    return LogisticRegression(C=np.inf, solver="newton-cholesky", max_iter=200)


def main() -> None:
    warnings.simplefilter("ignore")
    df = pd.read_csv(HERE / "dml_benchmark_data.csv")
    fold = df["fold"].to_numpy()
    smpls = [[(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(5)]]
    data = dml.DoubleMLData(df, "re78", "treat", X)
    models = {
        "plr": lambda: dml.DoubleMLPLR(
            data, LinearRegression(), LinearRegression(), draw_sample_splitting=False
        ),
        "irm": lambda: dml.DoubleMLIRM(
            data, LinearRegression(), _logit(), draw_sample_splitting=False
        ),
    }
    out: dict = {}
    for name, make in models.items():
        obj = make()
        obj.set_sample_splitting(smpls)
        obj.fit()
        rows = []
        for sc in SCENARIOS:
            obj.sensitivity_analysis(null_hypothesis=0.0, **sc)
            p = obj.sensitivity_params
            rows.append(
                {
                    **sc,
                    "theta_lower": float(p["theta"]["lower"][0]),
                    "theta_upper": float(p["theta"]["upper"][0]),
                    "se_lower": float(p["se"]["lower"][0]),
                    "se_upper": float(p["se"]["upper"][0]),
                    "ci_lower": float(p["ci"]["lower"][0]),
                    "ci_upper": float(p["ci"]["upper"][0]),
                    "rv": float(p["rv"][0]),
                    "rva": float(p["rva"][0]),
                }
            )
        out[name] = {
            "theta": float(obj.coef[0]),
            "se": float(obj.se[0]),
            "scenarios": rows,
        }
    out["_meta"] = {"doubleml": dml.__version__, "sklearn": sklearn.__version__}
    (HERE / "gaps_dml_sensitivity_doubleml.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote gaps_dml_sensitivity_doubleml.json")


if __name__ == "__main__":
    main()
