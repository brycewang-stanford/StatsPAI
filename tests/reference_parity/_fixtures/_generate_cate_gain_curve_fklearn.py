"""Reference curves for ``sp.cate_gain_curve`` from ``fklearn``.

``fklearn.causal.validation.curves.effect_curves`` (cumulative effect,
cumulative gain, relative cumulative gain; ``min_rows=30, steps=100`` on
3,000 rows, so the steps are 30, 60, ..., 3000) and ``effect_by_segment``
(ten quantile groups) on ``cate_gain_curve.csv``, for a continuous and a
binary treatment and for an informative and an uninformative ranking.
fklearn is Apache-2.0.

Run with an interpreter that has ``fklearn`` installed (not the repo
venv)::

    python tests/reference_parity/_fixtures/_generate_cate_gain_curve_fklearn.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import fklearn
import numpy as np
import pandas as pd
from fklearn.causal.effects import linear_effect
from fklearn.causal.validation.curves import effect_by_segment, effect_curves

HERE = pathlib.Path(__file__).parent
df = pd.read_csv(HERE / "cate_gain_curve.csv")

cases = {}
for treat, outcome in (("dose", "y_dose"), ("d", "y_bin")):
    for ranking in ("score", "noise"):
        curves = effect_curves(
            df,
            treatment=treat,
            outcome=outcome,
            prediction=ranking,
            min_rows=30,
            steps=100,
        )
        seg = effect_by_segment(
            df, treatment=treat, outcome=outcome, prediction=ranking, segments=10
        )
        cases[f"{treat}_{ranking}"] = {
            "ate": float(linear_effect(df, treat, outcome)),
            "n": [int(v) for v in curves["samples_count"]],
            "cumulative_effect": [float(v) for v in curves["cumulative_effect_curve"]],
            "cumulative_gain": [float(v) for v in curves["cumulative_gain_curve"]],
            "relative_gain": [
                float(v) for v in curves["relative_cumulative_gain_curve"]
            ],
            "effect_by_segment": [float(v) for v in seg],
        }

out = {
    "versions": {
        "python": sys.version.split()[0],
        "fklearn": getattr(fklearn, "__version__", "4.3.0"),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },
    "cases": cases,
}
(HERE / "cate_gain_curve_fklearn.json").write_text(
    json.dumps(out) + "\n", encoding="utf-8"
)
print(out["versions"], {k: v["ate"] for k, v in cases.items()})
