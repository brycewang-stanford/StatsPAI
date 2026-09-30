"""``sp.qreg`` exposes its whole coefficient table; ``sp.sqreg(reps=)``
bootstraps all quantiles jointly (Stata ``sqreg``). Top-5 replication list.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.regression.quantile import _qreg_fit


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    m = 800
    df = pd.DataFrame(dict(x1=rng.normal(size=m), x2=rng.normal(size=m)))
    df["y"] = df.x1 * (1 + 0.5 * rng.uniform(size=m)) + 0.5 * df.x2
    df["y"] += rng.standard_t(3, m)
    return df


def test_qreg_accessors_cover_every_regressor(data):
    r = sp.qreg(data, formula="y ~ x1 + x2", quantile=0.5)
    assert isinstance(r, sp.CausalResult)
    assert list(r.params.index) == ["const", "x1", "x2"]
    assert r.params["x1"] == pytest.approx(r.estimate)
    V = r.vcov()
    np.testing.assert_allclose(np.sqrt(np.diag(V)), r.std_errors.to_numpy())
    ci = r.conf_int()
    assert ci.loc["x1", 0] < r.params["x1"] < ci.loc["x1", 1]


def test_dual_lp_equals_primal_vertex(data):
    """The dual LP gives the primal LP's vertex, ties included."""
    from scipy import sparse
    from scipy.optimize import linprog

    X = np.column_stack([np.ones(len(data)), data.x1.round(), data.x2.round()])
    Y = data.y.round().to_numpy()
    for tau in (0.1, 0.5, 0.9):
        n, k = X.shape
        eye = sparse.identity(n, format="csc")
        primal = linprog(
            np.r_[np.zeros(k), tau * np.ones(n), (1 - tau) * np.ones(n)],
            A_eq=sparse.hstack([sparse.csc_matrix(X), eye, -eye], format="csc"),
            b_eq=Y,
            bounds=[(None, None)] * k + [(0, None)] * (2 * n),
            method="highs-ipm",
        ).x[:k]
        np.testing.assert_allclose(_qreg_fit(Y, X, tau), primal, atol=1e-10)


def test_sqreg_joint_bootstrap(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.sqreg(data, y="y", x=["x1"], quantiles=[0.25, 0.75], reps=60, seed=3)
    assert list(r.params.index) == ["q25:_cons", "q25:x1", "q75:_cons", "q75:x1"]
    for q in (0.25, 0.75):
        single = sp.qreg(data, y="y", x=["x1"], quantile=q)
        lab = f"q{int(q * 100)}:x1"
        assert r.params[lab] == pytest.approx(single.params["x1"], rel=1e-10)
    # joint covariance: the cross-quantile block is estimated, not zero
    V = r.vcov()
    assert V.loc["q25:x1", "q75:x1"] != 0
    t = sp.test(r, "q25:x1 = q75:x1")
    assert 0 <= t["pvalue"] <= 1
    table = sp.sqreg(data, y="y", x=["x1"], quantiles=[0.5])
    assert isinstance(table, pd.DataFrame)
