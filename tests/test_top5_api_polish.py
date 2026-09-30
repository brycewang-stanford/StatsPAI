"""API friction from the top-5 replication list (1.33).

Each test pins one item the replications tripped over: a module shadowed by
its function, missing result accessors, a key name, ``C()`` in ordered-model
formulas, and a multinomial IIA test that used the wrong restricted model.
"""

from __future__ import annotations

import pathlib
import types
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"


def test_rdrobust_submodule_is_a_module_and_still_callable():
    import statspai.rd.rdrobust as m

    assert isinstance(m, types.ModuleType)
    assert hasattr(m, "rdplot")
    assert callable(m)
    assert isinstance(sp.rdrobust, types.FunctionType)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 1500)
    df = pd.DataFrame(dict(x=x, y=0.4 * (x >= 0) + x + rng.normal(size=1500) * 0.3))
    a = m(df, y="y", x="x", c=0)
    b = sp.rdrobust(df, y="y", x="x", c=0)
    assert float(a.estimate) == float(b.estimate)


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(1)
    m = 400
    df = pd.DataFrame(
        dict(x=rng.normal(size=m), w=rng.normal(size=m), f=rng.integers(0, 15, m))
    )
    df["y"] = df.x + 0.5 * df.w + rng.normal(size=m)
    return df


def test_result_accessors(panel):
    r = sp.regress("y ~ x + w", data=panel)
    h = sp.hdfe_ols("y ~ x + w | f", data=panel)
    f = sp.feols("y ~ x + w | f", data=panel)
    assert r.nobs == h.nobs == f.nobs == len(panel)
    assert r.r2_adj == pytest.approx(r.diagnostics["Adj. R-squared"])
    # feols (pyfixest) and hdfe_ols (native) report the same e(r2_a)
    assert f.r2_adj == pytest.approx(h.r2_adj, rel=1e-10)
    V = h.cov_params()
    assert list(V.index) == list(V.columns) == ["x", "w"]
    np.testing.assert_allclose(V.to_numpy(), h.vcov)
    assert list(r.cov_params().index) == list(r.params.index)


def test_oster_bounds_exposes_delta_star(panel):
    ob = sp.oster_bounds(panel, y="y", treat="x", controls=["w"])
    assert ob["delta_star"] == ob["delta_for_zero"]


def test_ordered_and_multinomial_accept_categorical_formula_terms():
    df = pd.read_csv(_FIX / "oprobit_dummies.csv").query("g < 20")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.oprobit("y ~ x + z + C(g)", data=df)
        dummies = pd.get_dummies(df.g, prefix="g", drop_first=True, dtype=float)
        b = sp.oprobit(
            data=pd.concat([df, dummies], axis=1),
            y="y",
            x=["x", "z"] + list(dummies.columns),
        )
        m = sp.mlogit("y ~ x + C(g)", data=df)
    np.testing.assert_allclose(
        a.params.to_numpy()[-3:], b.params.to_numpy()[-3:], rtol=1e-10
    )
    assert a.params["x"] == pytest.approx(b.params["x"], rel=1e-10)
    # category labels keep the outcome's own values (not patsy's floats)
    assert m.params.index[0].startswith("[1]")


def test_mlogit_iia_matches_stata_hausman():
    """``mlogit y x z if y != j`` vs the full model, ``hausman, alleqs
    constant`` (Stata 18): df = rank of the difference, chi2 to 1e-5 (the
    generalised inverse is not unique on a singular difference)."""
    df = pd.read_csv(_FIX / "oprobit_dummies.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sp.mlogit("y ~ x + z", data=df)
    stata = {1: 5.203862137842, 2: 8.899590546351, 3: 1.656819037984}
    for j, chi2 in stata.items():
        row = m.iia_test[j]
        assert row["df"] == 4
        assert row["chi2"] == pytest.approx(chi2, rel=2e-5)
