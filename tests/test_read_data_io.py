"""``sp.read_data`` on .dta files, with and without the optional pyreadstat.

Regression: the pyreadstat import sat outside the ``try`` block, so the
documented pandas fallback was unreachable and a core install raised
``ModuleNotFoundError`` on every .dta file.
"""

import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def labelled_dta(tmp_path):
    path = tmp_path / "lab.dta"
    df = pd.DataFrame(
        {"wage": [10.0, 12.0, np.nan], "female": np.array([0, 1, 1], dtype="int8")}
    )
    df.to_stata(
        path,
        write_index=False,
        variable_labels={"wage": "Monthly wage in CNY"},
        value_labels={"female": {0: "male", 1: "female"}},
    )
    return path


def test_dta_without_pyreadstat_keeps_labels(labelled_dta):
    with mock.patch.dict(sys.modules, {"pyreadstat": None}):
        df = sp.read_data(str(labelled_dta))
    assert list(df.columns) == ["wage", "female"]
    # numeric codes kept (same layout as the pyreadstat path), not categoricals
    assert pd.api.types.is_numeric_dtype(df["female"])
    assert df["female"].tolist() == [0, 1, 1]
    assert np.isnan(df["wage"].iloc[2])
    assert df.attrs["_labels"] == {"wage": "Monthly wage in CNY"}
    assert df.attrs["_value_labels"] == {"female": {0: "male", 1: "female"}}
    assert sp.get_label(df, "wage") == "Monthly wage in CNY"


def test_dta_with_pyreadstat_matches_fallback(labelled_dta):
    pytest.importorskip("pyreadstat")
    with mock.patch.dict(sys.modules, {"pyreadstat": None}):
        fallback = sp.read_data(str(labelled_dta))
    native = sp.read_data(str(labelled_dta))
    np.testing.assert_allclose(
        native.to_numpy(dtype=float), fallback.to_numpy(dtype=float)
    )
    assert native.attrs["_labels"] == fallback.attrs["_labels"]
