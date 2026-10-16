"""A misspelled column raises ``ColumnNotFound`` with a did-you-mean.

Every public entry point below is called with exactly one column name
misspelled.  The error must be a :class:`~statspai.exceptions.ColumnNotFound`
(so also a ``MethodIncompatibility``, ``DataInsufficient`` and
``ValueError``), and ``diagnostics["did_you_mean"]`` must point at the
intended column under the argument the caller used, so an agent can repair
the call without parsing the message.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._input_validation import check_columns, missing_named_columns
from statspai.core.utils import formula_unknown_columns
from statspai.exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    out = pd.DataFrame(
        {
            "id": np.repeat(np.arange(40), 10),
            "year": np.tile(np.arange(2000, 2010), 40),
        }
    )
    out["x1"] = rng.normal(size=n)
    out["x2"] = rng.normal(size=n)
    out["z"] = rng.normal(size=n)
    out["treat"] = (rng.random(n) < 0.5).astype(int)
    out["wage"] = out["x1"] + out["treat"] + rng.normal(size=n)
    out["running"] = rng.uniform(-1, 1, n)
    out["female"] = (rng.random(n) < 0.5).astype(int)
    out["first_treat"] = np.where(
        out["id"] < 15, 0, np.where(out["id"] < 30, 2004, 2006)
    )
    out["post"] = (out["year"] >= 2005).astype(int)
    out["d"] = ((out["first_treat"] > 0) & (out["year"] >= out["first_treat"])).astype(
        int
    )
    out["m"] = out["x1"] + rng.normal(size=n)
    out["yb"] = (out["wage"] > out["wage"].median()).astype(int)
    out["cnt"] = rng.poisson(2, n)
    return out


# (id, call, expected did_you_mean)
CASES = [
    # --- formulas (shared design-matrix layer) ---
    (
        "regress_rhs",
        lambda df: sp.regress("wage ~ x1 + xx2", data=df),
        {"xx2": "x2"},
    ),
    ("regress_lhs", lambda df: sp.regress("wagee ~ x1", data=df), {"wagee": "wage"}),
    (
        "regress_cluster",
        lambda df: sp.regress("wage ~ x1", data=df, cluster="idd"),
        {"cluster": "id"},
    ),
    ("ivreg", lambda df: sp.ivreg("wage ~ x1 + (treat ~ zz)", data=df), {"zz": "z"}),
    ("logit", lambda df: sp.logit("yb ~ x11", data=df), {"x11": "x1"}),
    ("probit", lambda df: sp.probit("yb ~ x11", data=df), {"x11": "x1"}),
    ("poisson", lambda df: sp.poisson("cnt ~ x11", data=df), {"x11": "x1"}),
    ("nbreg", lambda df: sp.nbreg("cnt ~ x11", data=df), {"x11": "x1"}),
    (
        "glm",
        lambda df: sp.glm("cnt ~ x11", data=df, family="poisson"),
        {"x11": "x1"},
    ),
    ("qreg", lambda df: sp.qreg("wage ~ x11", data=df), {"x11": "x1"}),
    ("tobit", lambda df: sp.tobit("wage ~ x11", data=df), {"x11": "x1"}),
    ("mlogit", lambda df: sp.mlogit("cnt ~ x11", data=df), {"x11": "x1"}),
    ("ologit", lambda df: sp.ologit("cnt ~ x11", data=df), {"x11": "x1"}),
    (
        "regress_transformed",
        lambda df: sp.regress("wage ~ np.log(x11 ** 2) + C(idd)", data=df),
        {"x11": "x1", "idd": "id"},
    ),
    (
        "panel",
        lambda df: sp.panel(df, "wage ~ x11", entity="id", time="year"),
        {"x11": "x1"},
    ),
    (
        "panel_entity",
        lambda df: sp.panel(df, "wage ~ x1", entity="idd", time="year"),
        {"entity": "id"},
    ),
    (
        "did",
        lambda df: sp.did(df, y="wagee", treat="treat", time="post"),
        {"y": "wage"},
    ),
]


@pytest.mark.parametrize("name,call,expected", CASES, ids=[c[0] for c in CASES])
def test_misspelled_column_raises_column_not_found(df, name, call, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ColumnNotFound) as excinfo:
            call(df)
    err = excinfo.value
    assert isinstance(err, MethodIncompatibility)
    assert isinstance(err, DataInsufficient)
    assert isinstance(err, ValueError)
    assert err.diagnostics["did_you_mean"] == expected
    # the suggestion reaches the agent through the recovery hint too
    for suggestion in expected.values():
        assert f"did you mean {suggestion!r}" in err.recovery_hint
    assert err.to_dict()["kind"] == "column_not_found"


def test_misspelled_function_in_formula_is_not_a_column_error(df):
    # lg() is a misspelled *function*, not a column: keep the formula error
    with pytest.raises(MethodIncompatibility) as excinfo:
        sp.regress("wage ~ lg(x1)", data=df)
    assert not isinstance(excinfo.value, ColumnNotFound)


def test_formula_scanner_skips_functions_keywords_and_strings():
    frame = pd.DataFrame({"y": [1.0], "x1": [1.0], "g": [1]})
    assert (
        formula_unknown_columns(
            "y ~ np.log(x1) + C(g, Treatment(reference=2)) + I(x1**2)", frame
        )
        == []
    )
    assert formula_unknown_columns('yy ~ Q("x 1") + C(gg, Treatment("a"))', frame) == [
        "x 1",
        "yy",
        "gg",
    ]


def test_check_columns_labels_list_entries_by_position():
    frame = pd.DataFrame({"wage": [1.0], "x1": [0.0], "x2": [0.0]})
    assert missing_named_columns(
        {"y": "wage", "covariates": ["x1", "xx2"], "cluster": None}, frame.columns
    ) == {"covariates[1]": "xx2"}
    with pytest.raises(ColumnNotFound) as excinfo:
        check_columns(
            frame,
            {"y": "wag", "covariates": ["x1", "x22"]},
            function="sp.demo",
            extra_hint="Pass the frame the model was fitted on.",
        )
    err = excinfo.value
    assert str(err).startswith("sp.demo: Column(s) not found in data")
    assert err.diagnostics["did_you_mean"] == {"y": "wage", "covariates[1]": "x2"}
    assert err.recovery_hint.endswith("Pass the frame the model was fitted on.")


def test_check_columns_skips_arrays_and_passes_when_present():
    frame = pd.DataFrame({"wage": [1.0, 2.0]})
    check_columns(frame, {"y": "wage", "weights": np.ones(2), "cluster": None})
