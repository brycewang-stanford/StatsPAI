"""``sp.stata`` reads the panel declaration ``sp.xtset`` stored on the data.

A Stata dataset carries its ``xtset``; ``sp.xtset`` keeps it in
``df.attrs``, and ``sp.stata`` adopts it as if the snippet began with
``xtset id t`` -- including the sort by panel and time that ``xtset``
does. Reference: Stata 18 MP on the unsorted, gapped panel of
``reference_parity/_fixtures/stata_xtset_attrs.csv``
(``_generate_stata_xtset_attrs.do``): ``L.x``, ``F.x``, ``D.x``,
``L2.x``, ``_n`` after ``xtset`` and ``regress y L.x D.x``.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"


@pytest.fixture()
def panel():
    full = pd.read_csv(FX / "stata_xtset_attrs.csv")
    return full[["id", "t", "x", "y"]].copy(), full


def _run(commands, data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(commands, data=data, **kw)


def test_lags_and_row_order_after_sp_xtset_match_stata(panel):
    df, full = panel
    sp.xtset(df, id="id", time="t")
    run = _run(
        "gen double lx = L.x\ngen double fx = F.x\ngen double dx = D.x\n"
        "gen double l2x = L2.x\ngen n = _n",
        df,
        return_data=True,
    )
    got = run.data.merge(full, on=["id", "t"], suffixes=("", "_ref"))
    for col in ("lx", "fx", "dx", "l2x"):
        np.testing.assert_allclose(
            got[col], got[f"stata_{col}"], rtol=0, atol=1e-15, equal_nan=True
        )
    # xtset sorts by panel and time: _n is Stata's row number
    np.testing.assert_array_equal(got["n"], got["stata_n"])
    assert list(run.data[["id", "t"]].itertuples(index=False)) == sorted(
        df[["id", "t"]].itertuples(index=False)
    )
    # the caller's frame is untouched
    assert "lx" not in df.columns


def test_regression_with_ts_operators_after_sp_xtset_matches_stata(panel):
    df, _ = panel
    sp.xtset(df, id="id", time="t")
    fit = _run("regress y L.x D.x", df)
    np.testing.assert_allclose(
        fit.params.to_numpy(),
        [-0.17361526055852408, 0.85196664655777432, 0.9694796762372746],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        fit.std_errors.to_numpy()[1:],
        [0.36220711434893554, 0.23182111618179088],
        rtol=1e-12,
    )
    assert fit.data_info["nobs"] == 26


def test_same_as_writing_xtset_in_the_snippet(panel):
    df, _ = panel
    plain = df.copy()
    sp.xtset(df, id="id", time="t")
    a = _run("regress y L.x D.x", df)
    b = _run("xtset id t\nregress y L.x D.x", plain)
    np.testing.assert_array_equal(a.params.to_numpy(), b.params.to_numpy())


def test_without_a_declaration_ts_operators_are_still_refused(panel):
    df, _ = panel
    with pytest.raises(MethodIncompatibility):
        _run("regress y L.x", df)


def test_delta_and_time_series_declarations_carry_over():
    ts = pd.DataFrame({"t": [1, 3, 5, 7], "x": [1.0, 2.0, 4.0, 8.0]})
    sp.xtset(ts, time="t", delta=2)
    run = _run("gen double lx = L.x", ts, return_data=True)
    # delta(2): L.x at t is x at t - 2
    np.testing.assert_array_equal(run.data["lx"], [np.nan, 1.0, 2.0, 4.0])


def test_use_of_a_declared_frame_restores_its_declaration(panel):
    df, full = panel
    other = df.copy()
    sp.xtset(other, id="id", time="t")
    run = _run(
        "use wave\ngen double lx = L.x",
        pd.DataFrame({"z": [1.0]}),
        files={"wave": other},
        return_data=True,
    )
    got = run.data.merge(full, on=["id", "t"])
    np.testing.assert_allclose(got["lx"], got["stata_lx"], atol=1e-15, equal_nan=True)


def test_a_stale_declaration_is_ignored_with_a_warning(panel):
    df, _ = panel
    sp.xtset(df, id="id", time="t")
    sub = df[["x", "y"]]  # pandas carries attrs to the column subset
    assert sub.attrs.get("statspai_xtset")
    with pytest.warns(UserWarning, match="ignored"):
        sp.stata("summarize x", data=sub)
