"""sp.fepois turns pyfixest's demeaning failure into an actionable error."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp

pf = pytest.importorskip("pyfixest")


def test_demeaning_failure_points_to_ppmlhdfe(monkeypatch):
    def _fail(**kwargs):
        raise ValueError("Demeaning failed after 100_000 iterations.")

    monkeypatch.setattr(pf, "fepois", _fail)
    df = pd.DataFrame(
        {"y": [1.0, 2.0, 0.0, 3.0], "x": np.arange(4.0), "g": [0, 0, 1, 1]}
    )
    with pytest.raises(sp.NumericalInstability) as info:
        sp.fepois("y ~ x | g", data=df)
    assert "sp.ppmlhdfe" in str(info.value.recovery_hint)
    assert isinstance(info.value, ValueError)  # old `except ValueError` still works


def test_other_value_errors_pass_through(monkeypatch):
    def _fail(**kwargs):
        raise ValueError("something else")

    monkeypatch.setattr(pf, "fepois", _fail)
    df = pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0], "g": [0, 1]})
    with pytest.raises(ValueError, match="something else"):
        sp.fepois("y ~ x | g", data=df)
