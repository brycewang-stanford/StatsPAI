"""sp.causal(...).sensitivity_panel(): the E-value reads the estimate on its scale.

Until 1.40.2 the workflow passed its estimate (a difference on the outcome's
scale) to sp.evalue as a risk ratio, so the E-value changed with the units
of y. It now goes through sp.evalue_from_result as a mean difference
standardised by sd(y); a result whose scale is unknown is skipped and
recorded as a degradation (CLAUDE.md section 3.7).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.workflow._degradation import WorkflowDegradedWarning


def _data(scale: float = 1.0, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-0.5 * x1))).astype(int)
    y = scale * (1.0 + 0.8 * t + 0.5 * x1 - 0.3 * x2 + rng.normal(size=n))
    return pd.DataFrame({"Y": y, "T": t, "x1": x1, "x2": x2})


def _workflow(df: pd.DataFrame, result):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = sp.causal(
            df, y="Y", treatment="T", covariates=["x1", "x2"], design="observational"
        )
    w.result = result
    return w


def _evalue_row(w) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = w.sensitivity_panel()
    return panel[panel["method"].str.startswith("E-value")].iloc[0]


def test_ols_coefficient_is_read_as_a_standardised_difference():
    df = _data()
    fit = sp.regress("Y ~ T + x1 + x2", data=df, robust="hc1")
    row = _evalue_row(_workflow(df, fit))
    sd = df["Y"].std()
    ref = sp.evalue(
        estimate=fit.params["T"] / sd, se=fit.std_errors["T"] / sd, measure="SMD"
    )["evalue_estimate"]
    assert row["statistic"] == pytest.approx(ref, rel=1e-12)


def test_evalue_does_not_depend_on_the_units_of_y():
    a = _data(scale=1.0)
    b = _data(scale=1000.0)
    ra = _evalue_row(_workflow(a, sp.regress("Y ~ T + x1 + x2", data=a, robust="hc1")))
    rb = _evalue_row(_workflow(b, sp.regress("Y ~ T + x1 + x2", data=b, robust="hc1")))
    assert ra["statistic"] == pytest.approx(rb["statistic"], rel=1e-9)


def test_unknown_scale_is_skipped_and_recorded():
    df = _data()
    df["Yb"] = (df["Y"] > df["Y"].median()).astype(int)
    fit = sp.logit("Yb ~ T + x1 + x2", data=df)
    w = _workflow(df, fit)
    with pytest.warns(WorkflowDegradedWarning):
        panel = w.sensitivity_panel()
    row = panel[panel["method"].str.startswith("E-value")].iloc[0]
    assert np.isnan(row["statistic"])
    assert "not computed" in row["interpretation"]
    assert any(d["section"] == "sensitivity_panel: E-value" for d in w.degradations)
