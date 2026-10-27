"""``C(f)`` in ``sp.hdfe_ols`` is the same term as ``i.f``.

``sp.regress`` takes patsy formulas, so users (and the QJE November 2026
article that prompted this) write ``C(arr)``; ``sp.hdfe_ols`` used to reject
it. The identity below is exact: the same design matrix, so the same numbers.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(20261028)
    n = 600
    d = pd.DataFrame(
        {
            "arr": rng.integers(0, 5, n),
            "strata": rng.integers(0, 6, n),
            "high": rng.integers(0, 2, n),
            "trans": rng.integers(0, 2, n),
            "rec": rng.integers(0, 40, n),
            "lis": rng.integers(0, 150, n),
        }
    )
    d["y"] = 0.3 * (d.arr > 0) + d.high - 0.4 * d.high * d.trans + rng.normal(size=n)
    return d


def _same(a, b):
    assert list(a.params.index) == list(b.params.index)
    np.testing.assert_array_equal(a.params.values, b.params.values)
    np.testing.assert_array_equal(a.std_errors.values, b.std_errors.values)


def test_c_equals_i_with_absorb(df):
    _same(
        sp.hdfe_ols("y ~ C(arr) | strata", data=df, vce="robust"),
        sp.hdfe_ols("y ~ i.arr | strata", data=df, vce="robust"),
    )


def test_c_factorial_two_way_cluster(df):
    _same(
        sp.hdfe_ols("y ~ C(high) * C(trans)", data=df, cluster=["rec", "lis"]),
        sp.hdfe_ols("y ~ i.high * i.trans", data=df, cluster=["rec", "lis"]),
    )


def test_c_treatment_reference_is_ib(df):
    for spelled in ("C(arr, Treatment(reference=2))", "C(arr, Treatment(2))"):
        _same(
            sp.hdfe_ols(f"y ~ {spelled} | strata", data=df),
            sp.hdfe_ols("y ~ ib2.arr | strata", data=df),
        )


def test_c_point_estimates_match_regress(df):
    # sp.regress with the strata as dummies is the same projection.
    a = sp.hdfe_ols("y ~ C(arr) | strata", data=df)
    b = sp.regress("y ~ C(arr) + C(strata)", data=df)
    for lvl in range(1, 5):
        assert a.params[f"arr::{lvl}"] == pytest.approx(
            b.params[f"C(arr)[T.{lvl}]"], rel=1e-10
        )


def test_c_with_base_level_on_absorb_side_is_rejected(df):
    with pytest.raises(ValueError, match="base level"):
        sp.hdfe_ols("y ~ high | C(strata, Treatment(1))", data=df)
