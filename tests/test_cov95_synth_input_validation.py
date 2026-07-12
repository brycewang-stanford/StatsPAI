"""Coverage tests for sp.synth input-validation contracts.

Targets the ``_require_*`` / ``_coerce_column_list`` guards in
``statspai.synth.scm`` (open-unit ``alpha``, non-negative ``penalization``,
non-empty column names, string options, column lists). Each bad input must
raise ``MethodIncompatibility`` loudly rather than fail silently downstream
(CLAUDE.md section 7).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _panel(n_units=6, n_periods=12, treatment_time=8, effect=3.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        for t in range(n_periods):
            treated = i == 0 and t >= treatment_time
            y = 1.0 * i + 0.2 * t + (effect if treated else 0.0) + rng.normal(0, 0.1)
            rows.append({"unit": f"u{i}", "time": t, "y": y})
    return pd.DataFrame(rows)


COMMON = dict(outcome="y", unit="unit", time="time", treated_unit="u0",
              treatment_time=8)


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.mark.parametrize("alpha", [1.5, 0.0, 1.0, -0.1])
def test_alpha_must_be_open_unit(panel, alpha):
    with pytest.raises(MethodIncompatibility):
        sp.synth(panel, alpha=alpha, **COMMON)


@pytest.mark.parametrize("pen", [-1.0, -0.001])
def test_penalization_must_be_nonnegative(panel, pen):
    with pytest.raises(MethodIncompatibility):
        sp.synth(panel, penalization=pen, **COMMON)


def test_empty_outcome_name_rejected(panel):
    kw = {**COMMON, "outcome": ""}
    with pytest.raises(MethodIncompatibility):
        sp.synth(panel, **kw)


def test_non_string_unit_rejected(panel):
    kw = {**COMMON, "unit": 123}
    with pytest.raises(MethodIncompatibility):
        sp.synth(panel, **kw)


def test_method_must_be_string(panel):
    with pytest.raises(MethodIncompatibility):
        sp.synth(panel, method=42, **COMMON)


def test_empty_covariate_list_rejected(panel):
    # A covariate argument that resolves to an empty list is a user error.
    with pytest.raises(MethodIncompatibility):
        sp.synth(panel, covariates=[""], **COMMON)


def test_non_dataframe_input_rejected():
    with pytest.raises(MethodIncompatibility):
        sp.synth([1, 2, 3], **COMMON)
