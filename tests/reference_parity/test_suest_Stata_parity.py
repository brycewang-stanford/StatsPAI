"""``sp.suest`` against Stata ``suest`` after ``regress``.

There was no cross-equation cluster-robust inference: ``sp.sureg`` is
Zellner FGLS with no clusters and a common sample, and ``sp.test`` works
within one model -- so the "joint test" rows of an experiment's tables
(UCT, QJE 2016: every table's ``suest ..., cluster()`` line) had to be
hand-coded. ``sp.suest`` stacks the equations' influence functions.

Reference: Stata 18 ``suest e1 e2 e3, vce(robust | cluster cl)`` after three
``regress`` fits on ``_fixtures/suest_data.csv`` (the second equation has
three missing outcomes), from ``_generate_suest_Stata.do``: coefficients,
the full joint covariance and both tests agree to 1e-12.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "suest_Stata.json").read_text(encoding="utf-8"))
EQS = ["y1 ~ x + w", "y2 ~ x", "y3 ~ x + w"]


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "suest_data.csv")


@pytest.mark.parametrize("vce, cl", [("robust", None), ("cluster", "cl")])
def test_matches_stata_suest(data, vce, cl):
    ref = R[vce]
    r = sp.suest(data, EQS, cluster=cl)
    np.testing.assert_allclose(r.params.to_numpy(), ref["b"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        r.vcov.to_numpy(), np.reshape(ref["V"], (8, 8)), rtol=1e-11, atol=1e-16
    )
    eq = r.test_equal("x")
    assert eq["chi2"] == pytest.approx(ref["eq_chi2"], rel=1e-11)
    assert eq["df"] == ref["eq_df"]
    assert r.test_zero("x")["chi2"] == pytest.approx(ref["zero_chi2"], rel=1e-11)


def test_equations_keep_their_own_samples(data):
    r = sp.suest(data, EQS, cluster="cl")
    assert r.n_obs == {"y1": 300, "y2": 297, "y3": 300}
    assert list(r.params.index[:3]) == ["y1:x", "y1:w", "y1:_cons"]


def test_tuple_equations_and_errors(data):
    a = sp.suest(data, [("y1", ["x", "w"]), ("y2", ["x"])], cluster="cl")
    b = sp.suest(data, ["y1 ~ x + w", "y2 ~ x"], cluster="cl")
    np.testing.assert_allclose(a.vcov.to_numpy(), b.vcov.to_numpy())
    with pytest.raises(sp.MethodIncompatibility, match="distinct"):
        sp.suest(data, ["y1 ~ x", "y1 ~ w"])
    with pytest.raises(sp.MethodIncompatibility, match="Unknown"):
        a.wald([{"y9:x": 1.0}])
