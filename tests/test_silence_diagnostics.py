"""``sp.silence``: a named diagnostic's warning is withheld, its record kept.

Each test states what the caller is entitled to: the warning is gone inside
the block and back after it, the result still carries the diagnostic, other
diagnostics are untouched, and a name that is not a diagnostic is refused.
"""

import threading
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core._spec_counter import MULTIPLICITY_WARNING
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(autouse=True)
def _fresh_counter():
    sp.reset_spec_counter()
    yield
    sp.reset_spec_counter()


def _frame(n=80, holes=6, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["y"] = df.x + rng.normal(size=n)
    df.loc[: holes - 1, "x"] = np.nan
    return df


def _names(record):
    return [getattr(w.message, "diagnostics", {}).get("test") for w in record]


def test_warning_names_its_diagnostic():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sp.regress("y ~ x", data=_frame())
    assert _names(rec) == ["listwise_deletion"]


def _fit_names(formula, df, **kw):
    """The fit, and the diagnostics it warned about."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fit = sp.regress(formula, data=df, **kw)
    return fit, _names(rec)


def test_silenced_inside_the_block_recorded_on_the_result_and_back_after():
    df = _frame()
    with sp.silence("listwise_deletion"):
        fit, names = _fit_names("y ~ x", df)
    assert names == []
    info = fit.model_info["listwise_deletion"]
    assert info == {"n_rows_dropped_missing": 6, "missing_by_column": {"x": 6}}
    assert _fit_names("y ~ x", df)[1] == ["listwise_deletion"]


def test_only_the_named_diagnostic_is_silenced():
    df = _frame()
    with sp.silence("specification_search"):
        assert _fit_names("y ~ x", df)[1] == ["listwise_deletion"]


def test_specification_search_is_silenced_but_still_counted():
    df = _frame(holes=0)
    rhs = ["x", "z", "x + z", "x + I(z**2)", "z + I(x**2)"]
    variances = ["nonrobust", "hc1", "hc2", "hc3"]
    warned = []
    with sp.silence(["specification_search", "listwise_deletion"]):
        for r in rhs:
            for v in variances:
                fit, names = _fit_names(f"y ~ {r}", df, robust=v)
                warned += names
    assert warned == []
    assert fit.model_info["multiplicity"]["specifications_tried"] == 20
    assert MULTIPLICITY_WARNING == 20
    found = [v for v in fit.violations() if v.get("test") == "specification_search"]
    assert found and found[0]["severity"] == "warning"
    assert _fit_names("y ~ x + z + I(z**3)", df)[1] == ["specification_search"]


def test_blocks_nest_and_the_decorator_form_works():
    df = _frame()

    @sp.silence("listwise_deletion")
    def fitted():
        return _fit_names("y ~ x", df)[1]

    assert fitted() == []
    assert fitted() == []  # the decorator is reusable
    with sp.silence("specification_search"):
        with sp.silence("listwise_deletion"):
            assert _fit_names("y ~ x", df)[1] == []
        assert _fit_names("y ~ x", df)[1] == ["listwise_deletion"]  # inner ended
    assert _fit_names("y ~ x", df)[1] == ["listwise_deletion"]


def test_the_setting_does_not_leak_into_another_thread():
    df = _frame()
    seen = []

    def other():
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            sp.regress("y ~ x", data=df)
        seen.extend(_names(rec))

    with sp.silence("listwise_deletion"):
        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert seen == ["listwise_deletion"]


@pytest.mark.parametrize("bad", ["listwise", ["listwise_deletion", "nope"], []])
def test_a_name_that_is_not_a_diagnostic_is_refused_at_the_call(bad):
    with pytest.raises(MethodIncompatibility, match="not a diagnostic"):
        sp.silence(bad)
