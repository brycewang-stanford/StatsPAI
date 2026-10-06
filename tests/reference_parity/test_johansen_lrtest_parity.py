"""``sp.johansen_lrtest`` against ``urca`` 1.3-4 (``blrtest``, ``bh5lrtest``,
``alrtest``) in the three deterministic cases ``ca.jo`` supports.

Reference: ``_fixtures/johansen_lrtest_R.json`` from
``_fixtures/_generate_johansen_lrtest_R.R`` on the simulated
``_fixtures/johansen_lrtest.csv``. Everything here is an eigenvalue of the
same product-moment matrices, so the agreement is to rounding; 1e-8 leaves
room for the two LAPACK builds.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
CASES = {"none": "c", "const": "rc", "trend": "rt"}
H = np.array([[1, 0, -1, 0], [0, 1, -1, 0], [0, 0, 0, 1]], dtype=float).T


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "johansen_lrtest.csv")


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "johansen_lrtest_R.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("ecdet", list(CASES))
def test_rank_statistics(data, ref, ecdet):
    fit = sp.johansen(data, lags=2, trend=CASES[ecdet])
    np.testing.assert_allclose(fit.test_stats, ref[ecdet]["trace"], rtol=1e-8)
    assert fit.n_used == len(data) - 3


@pytest.mark.parametrize("ecdet", list(CASES))
def test_beta_restriction(data, ref, ecdet):
    r = ref[ecdet]["beta"]
    test = sp.johansen_lrtest(data, rank=2, lags=2, trend=CASES[ecdet], beta=H)
    assert test.statistic == pytest.approx(r["stat"], rel=1e-8)
    assert test.df == r["df"]
    assert test.pvalue == pytest.approx(r["p"], rel=1e-7)
    # restricted vectors, first element normalised to one on both sides
    np.testing.assert_allclose(test.beta.to_numpy(), np.array(r["V"]), rtol=1e-6)


@pytest.mark.parametrize("ecdet", list(CASES))
def test_known_vectors(data, ref, ecdet):
    one = sp.johansen_lrtest(
        data, rank=2, lags=2, trend=CASES[ecdet], beta_known=[1, 0, -1, 0]
    )
    assert one.statistic == pytest.approx(ref[ecdet]["known1"]["stat"], rel=1e-8)
    assert one.df == ref[ecdet]["known1"]["df"]
    assert one.pvalue == pytest.approx(ref[ecdet]["known1"]["p"], rel=1e-7)
    # the whole space known: urca's blrtest with H = the two vectors
    both = sp.johansen_lrtest(
        data, rank=2, lags=2, trend=CASES[ecdet], beta_known=H[:, :2]
    )
    assert both.statistic == pytest.approx(ref[ecdet]["known2"]["stat"], rel=1e-8)
    assert both.df == ref[ecdet]["known2"]["df"]


@pytest.mark.parametrize("ecdet", list(CASES))
def test_loading_restriction(data, ref, ecdet):
    weak = sp.johansen_lrtest(
        data, rank=2, lags=2, trend=CASES[ecdet], loading=np.eye(4)[:, :3]
    )
    assert weak.statistic == pytest.approx(ref[ecdet]["loading"]["stat"], rel=1e-8)
    assert weak.df == ref[ecdet]["loading"]["df"]
    # the excluded equation does not adjust
    assert np.allclose(weak.loading.loc["rr"], 0.0)
    two = sp.johansen_lrtest(
        data, rank=2, lags=2, trend=CASES[ecdet], loading=np.eye(4)[:, :2]
    )
    assert two.statistic == pytest.approx(ref[ecdet]["loading2"]["stat"], rel=1e-8)
    assert two.df == ref[ecdet]["loading2"]["df"]
