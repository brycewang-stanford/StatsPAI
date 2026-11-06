"""One policy for ``inf`` in the data, whichever estimator reads it.

Before 2026-11 the same data gave five behaviours: ``regress`` raised,
``feols`` dropped the row with a pyfixest warning, ``rdrobust`` dropped it
silently, ``callaway_santanna`` / ``synth`` returned a changed estimate with
numpy RuntimeWarnings, ``logit`` / ``glm`` returned NaN coefficients and
``dml`` raised sklearn's ValueError (rOpenSci audit probe P4). Now ``inf`` in
a used column raises ``DataInsufficient`` naming the column and the count,
with one recovery hint; NaN keeps each estimator's missing-value rule.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._input_validation import (
    INFINITE_VALUES_HINT,
    clean_frame,
    infinite_values_by_column,
    require_finite,
)
from statspai.exceptions import DataInsufficient


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["d"] = (rng.uniform(size=n) < 0.5).astype(int)
    df["y"] = 1 + df.x + 0.5 * df.d + rng.normal(size=n)
    df["yb"] = (df.y > 1).astype(float)
    df["r"] = rng.uniform(-1, 1, n)
    df["yr"] = df.r + (df.r > 0) + rng.normal(scale=0.3, size=n)
    return df


def _poke(df: pd.DataFrame, col: str, value: float = np.inf) -> pd.DataFrame:
    out = df.copy()
    out.loc[5, col] = value
    return out


CALLS = {
    "regress": lambda d: sp.regress("y ~ x + d", data=d),
    "feols": lambda d: sp.feols("y ~ x + d", data=d),
    "logit": lambda d: sp.logit("yb ~ x + d", data=d),
    "glm": lambda d: sp.glm("y ~ x + d", data=d),
    "dml": lambda d: sp.dml(d, y="y", treat="d", covariates=["x", "z"], model="plr"),
    "rdrobust": lambda d: sp.rdrobust(d, y="yr", x="r", c=0),
}
COLUMNS = {
    "regress": ["y", "x"],
    "feols": ["y", "x"],
    "logit": ["yb", "x"],
    "glm": ["y", "x"],
    "dml": ["y", "x"],
    "rdrobust": ["yr", "r"],
}


@pytest.mark.parametrize(
    "name,col",
    [(n, c) for n, cols in COLUMNS.items() for c in cols],
)
@pytest.mark.parametrize("sign", [1, -1], ids=["+inf", "-inf"])
def test_inf_is_refused_with_the_shared_message(frame, name, col, sign):
    with pytest.raises(DataInsufficient) as err:
        CALLS[name](_poke(frame, col, sign * np.inf))
    assert err.value.recovery_hint == INFINITE_VALUES_HINT
    assert err.value.diagnostics["infinite_by_column"] == {col: 1}
    assert repr(col) in str(err.value)


@pytest.mark.parametrize("name", ["regress", "feols", "glm", "dml"])
def test_nan_still_follows_listwise_deletion(frame, name):
    clean = frame.drop(index=5).reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = CALLS[name](_poke(frame, "x", np.nan))
        ref = CALLS[name](clean)
    est = getattr(got, "estimate", None)
    if est is None:
        assert got.params["x"] == pytest.approx(ref.params["x"], rel=1e-10)
    else:
        assert est == pytest.approx(ref.estimate, rel=1e-6)


def test_derived_inf_in_a_formula_term_is_refused(frame):
    d = frame.assign(w=np.abs(frame.x))
    d.loc[0, "w"] = 0.0
    with pytest.raises(DataInsufficient, match="np.log"):
        sp.regress("y ~ np.log(w)", data=d)


@pytest.fixture(scope="module")
def mpdta() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.datasets.mpdta()


def _cs(df: pd.DataFrame):
    return sp.callaway_santanna(df, y="lemp", g="first_treat", t="year", i="countyreal")


def test_callaway_santanna_refuses_inf_outcome(mpdta):
    d = mpdta.copy()
    d.loc[3, "lemp"] = np.inf
    with pytest.raises(DataInsufficient) as err:
        _cs(d)
    assert err.value.diagnostics["infinite_by_column"] == {"lemp": 1}


def test_callaway_santanna_inf_cohort_still_means_never_treated(mpdta):
    ref = _cs(mpdta)
    d = mpdta.copy()
    d["first_treat"] = d["first_treat"].astype(float).replace(0, np.inf)
    got = _cs(d)
    assert got.estimate == pytest.approx(ref.estimate, rel=1e-12)
    assert got.se == pytest.approx(ref.se, rel=1e-12)


def test_did_2x2_refuses_inf(frame):
    d = frame.assign(post=(frame.index % 2).astype(int))
    d.loc[7, "y"] = np.inf
    with pytest.raises(DataInsufficient, match="infinite"):
        sp.did_2x2(d, y="y", treat="d", time="post")


# --------------------------------------------------------------------------- #
#  The shared primitive
# --------------------------------------------------------------------------- #


def test_counts_inf_separately_from_nan():
    df = pd.DataFrame(
        {
            "a": [1.0, np.inf, -np.inf, np.nan],
            "b": [np.nan, np.nan, 1.0, 2.0],
            "c": pd.array([1.0, None, np.inf, 0.0], dtype="Float64"),
            "o": pd.Series([1, np.inf, "x", None], dtype=object),
            "i": [1, 2, 3, 4],
        }
    )
    assert infinite_values_by_column(
        df, ["a", "b", "c", "o", "i", "missing", None]
    ) == {
        "a": 2,
        "c": 1,
        "o": 1,
    }


def test_require_finite_reports_rows_and_columns():
    df = pd.DataFrame({"a": [np.inf, 1.0, 2.0], "b": [np.inf, -np.inf, 0.0]})
    with pytest.raises(DataInsufficient) as err:
        require_finite(df, ["a", "b"], function="demo")
    d = err.value.diagnostics
    assert d["infinite_by_column"] == {"a": 1, "b": 2}
    assert d["n_infinite"] == 3
    assert d["n_rows_infinite"] == 2
    assert str(err.value).startswith("demo: 3 infinite value(s)")


def test_require_finite_passes_nan_and_clean_frames():
    df = pd.DataFrame({"a": [np.nan, 1.0], "s": ["u", "v"]})
    require_finite(df, ["a", "s"], function="demo")


def test_clean_frame_refuses_inf_before_dropping_nan():
    df = pd.DataFrame({"y": [1.0, 2.0, np.inf, 4.0], "x": [1.0, np.nan, 3.0, 4.0]})
    with pytest.raises(DataInsufficient, match="infinite"):
        clean_frame(df, ["y", "x"], function="demo", n_params=1)
    out = clean_frame(df.replace(np.inf, 3.0), ["y", "x"], function="demo", n_params=1)
    assert len(out) == 3
