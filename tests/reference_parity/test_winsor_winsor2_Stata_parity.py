"""``sp.winsor`` against Stata ``winsor2`` (percentiles, ``if``, ``by()``).

``sp.winsor`` claimed ``winsor2`` equivalence but used numpy's linear
interpolation; a replication found every winsorized control differing
from Stata by 2e-4 to 8e-4 (a 99th percentile of 51,003.45 against
51,015), and no way to write ``winsor2 ... if year >= 2007``.

Reference: Stata 18 ``winsor2`` on ``_fixtures/winsor2_data.csv`` (997
rows, 20 missing), from ``_generate_winsor2_Stata.do``:
``cuts(1 99)``; ``cuts(5 95)`` with an ``if`` qualifier; ``cuts(2.5 97.5)
by(g)``.  The cutoffs are order statistics or averages of two, computed
identically: values agree to the CSV round trip (``rtol=1e-14``), except
that ``winsor2 ..., by()`` stores its output as ``float``, hence
``rtol=1e-7`` there.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "winsor2_data.csv")


@pytest.fixture(scope="module")
def ref():
    return pd.read_csv(_FIX / "winsor2_Stata.csv")


def _same(a, b, rtol=1e-14):
    np.testing.assert_allclose(
        np.asarray(a, float), np.asarray(b, float), rtol=rtol, equal_nan=True
    )


def test_default_cuts_match_winsor2(data, ref):
    out = sp.winsor(data, vars=["x", "z"], cuts=(1, 99))
    _same(out["x_w"], ref["x_a"])
    _same(out["z_w"], ref["z_a"])


def test_subset_matches_winsor2_if(data, ref):
    out = sp.winsor(
        data, vars=["x", "z"], cuts=(5, 95), subset="year >= 2007 & year <= 2020"
    )
    _same(out["x_w"], ref["x_b"])
    _same(out["z_w"], ref["z_b"])
    # replace=True leaves rows outside the subset untouched.
    rep = sp.winsor(
        data,
        vars=["x"],
        cuts=(5, 95),
        replace=True,
        subset=data["year"].between(2007, 2020),
    )
    outside = ~data["year"].between(2007, 2020)
    _same(rep.loc[outside, "x"], data.loc[outside, "x"])


def test_by_matches_winsor2_by(data, ref):
    out = sp.winsor(data, vars=["x", "z"], cuts=(2.5, 97.5), by="g")
    _same(out["x_w"], ref["x_c"], rtol=1e-7)  # winsor2 by() stores float
    _same(out["z_w"], ref["z_c"], rtol=1e-7)


def test_linear_method_keeps_the_old_numbers(data):
    out = sp.winsor(data, vars=["x"], method="linear")
    x = data["x"].dropna()
    lo, hi = np.percentile(x, [1, 99])
    _same(out["x_w"], data["x"].clip(lo, hi))
    assert not np.allclose(
        out["x_w"].dropna(), sp.winsor(data, vars=["x"])["x_w"].dropna()
    )
