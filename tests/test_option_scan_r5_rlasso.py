"""Option scan, round 5: ``rlasso/``.

Truths are identities: a heteroskedasticity-robust variance is the cluster
variance with every row its own cluster, and a homoskedastic variance is the
OLS one on the same regressors.
"""

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient


@pytest.fixture(scope="module")
def het():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.standard_normal((n, 10))
    d = X[:, 0] + rng.standard_normal(n)
    y = d + X[:, 1] + 2.0 * np.abs(d) * rng.standard_normal(n)
    Z = rng.standard_normal((n, 3))
    dd = Z @ np.array([1.0, 0.8, 0.6]) + X[:, 0] + rng.standard_normal(n)
    yy = dd + X[:, 1] + (1 + np.abs(dd)) * rng.standard_normal(n)
    return X, y, d, Z, dd, yy


# -- vce='robust' must give a robust standard error -------------------------


@pytest.mark.parametrize("method", ["partialling out", "double selection"])
def test_rlasso_effect_robust_equals_singleton_cluster(het, method):
    X, y, d, *_ = het
    n = len(y)
    rob = sp.rlasso_effect(X, y, d, method=method, vce="robust")
    cl = sp.rlasso_effect(X, y, d, method=method, cluster=np.arange(n))
    assert rob.alpha == pytest.approx(cl.alpha, rel=1e-12)
    # HC0: sum v_i^2 e_i^2 / (sum v_i^2)^2. Double selection keeps hdm's
    # n / (n - s - 1) factor on the residuals.
    if method == "partialling out":
        assert rob.se == pytest.approx(cl.se, rel=1e-10)
    else:
        s_sel = int(rob.selection_index.sum())
        factor = np.sqrt(n / (n - s_sel - 1))
        assert rob.se == pytest.approx(cl.se * factor, rel=1e-10)


def test_rlasso_effect_partialling_out_default_is_hdm_iid(het):
    """The default stays hdm's homoskedastic SE (snapshot-pinned elsewhere)."""
    X, y, d, *_ = het
    a = sp.rlasso_effect(X, y, d)
    b = sp.rlasso_effect(X, y, d, robust=False)
    assert a.se == b.se


def test_rlasso_effect_double_selection_nonrobust_is_ols(het):
    X, y, d, *_ = het
    res = sp.rlasso_effect(X, y, d, method="double selection", vce="iid")
    sel = res.selection_index
    import pandas as pd

    frame = pd.DataFrame(X[:, sel], columns=[f"c{j}" for j in range(sel.sum())])
    frame["y"], frame["d"] = y, d
    rhs = " + ".join(["d"] + list(frame.columns[: sel.sum()]))
    ols = sp.regress(f"y ~ {rhs}", data=frame)
    assert res.alpha == pytest.approx(ols.params["d"], rel=1e-10)
    assert res.se == pytest.approx(ols.std_errors["d"], rel=1e-10)


def test_rlasso_iv_select_x_robust_equals_singleton_cluster(het):
    X, _, _, Z, dd, yy = het
    n = len(yy)
    kw = dict(y=yy, d=dd, z=Z, x=X, select_Z=False, select_X=True)
    rob = sp.rlasso_iv(vce="robust", **kw)
    cl = sp.rlasso_iv(cluster=np.arange(n), **kw)
    assert rob.coef[0] == pytest.approx(cl.coef[0], rel=1e-12)
    assert rob.se[0] == pytest.approx(cl.se[0], rel=1e-10)


def test_rlasso_iv_select_z_nonrobust_is_classical_2sls(het):
    _, _, _, Z, dd, yy = het
    import pandas as pd

    res = sp.rlasso_iv(y=yy, d=dd, z=Z, select_Z=True, select_X=False, robust=False)
    assert res.selection["n_selected_Z"] == 3
    frame = pd.DataFrame(
        {"y": yy, "d": dd, "z1": Z[:, 0], "z2": Z[:, 1], "z3": Z[:, 2]}
    )
    ref = sp.iv("y ~ (d ~ z1 + z2 + z3)", data=frame)
    assert res.coef[0] == pytest.approx(ref.params["d"], rel=1e-10)
    assert res.se[0] == pytest.approx(ref.std_errors["d"], rel=1e-10)


# -- missing values ----------------------------------------------------------


def test_rlasso_refuses_missing_values(het):
    X, y, d, *_ = het
    X2 = X.copy()
    X2[5, 2] = np.nan
    with pytest.raises(DataInsufficient):
        sp.rlasso(X2, y)
    with pytest.raises(DataInsufficient):
        sp.rlasso_effect(X2, y, d)
    y2 = y.copy()
    y2[3] = np.nan
    with pytest.raises(DataInsufficient):
        sp.rlasso(X, y2)
