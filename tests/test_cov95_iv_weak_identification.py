"""Coverage tests for statspai.iv.weak_identification internal helpers,
summaries, error paths, and robust/cluster covariance branches.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

_wid = importlib.import_module("statspai.iv.weak_identification")


def _two_endog(n=600, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 3))
    v = rng.normal(size=(n, 2))
    d1 = 0.8 * z[:, 0] + 0.3 * z[:, 1] + v[:, 0]
    d2 = 0.6 * z[:, 1] + 0.4 * z[:, 2] + v[:, 1]
    D = np.column_stack([d1, d2])
    return D, z, rng


def test_as_matrix_1d():
    a = _wid._as_matrix(np.array([1.0, 2.0, 3.0]))
    assert a.shape == (3, 1)


def test_collect_names_series_and_array():
    s = pd.Series([1.0, 2.0], name="foo")
    assert _wid._collect_names(s, "p") == ["foo"]
    s2 = pd.Series([1.0, 2.0])  # no name
    assert _wid._collect_names(s2, "p") == ["p0"]
    arr = np.zeros((5, 2))
    assert _wid._collect_names(arr, "z") == ["z0", "z1"]
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert _wid._collect_names(df, "x") == ["a", "b"]


def test_extract_exog_list_of_str():
    df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]})
    W = _wid._extract_exog(df, ["x1", "x2"], 3, add_const=True)
    assert W.shape == (3, 3)  # const + 2
    W2 = _wid._extract_exog(None, None, 3, add_const=False)
    assert W2.shape == (3, 0)


def test_sqrtm_sym():
    M = np.array([[2.0, 0.5], [0.5, 1.0]])
    S = _wid._sqrtm_sym(M)
    np.testing.assert_allclose(S @ S, M, atol=1e-8)


def test_kleibergen_paap_summary_and_robust():
    D, Z, _ = _two_endog(seed=1)
    r = _wid.kleibergen_paap_rk(D, Z, cov_type="robust")
    assert r.n_endog == 2 and r.n_instruments == 3
    s = r.summary()
    assert "Kleibergen-Paap" in s
    assert "rk LM statistic" in s


def test_kleibergen_paap_nonrobust():
    D, Z, _ = _two_endog(seed=2)
    r = _wid.kleibergen_paap_rk(D, Z, cov_type="nonrobust")
    assert r.cov_type == "nonrobust"
    assert np.isfinite(r.rk_lm)


def test_kleibergen_paap_cluster():
    D, Z, rng = _two_endog(seed=3)
    cl = rng.integers(0, 10, size=D.shape[0])
    r = _wid.kleibergen_paap_rk(D, Z, cov_type="cluster", cluster=cl)
    assert "cluster" in r.cov_type


def test_kleibergen_paap_cluster_missing_raises():
    D, Z, _ = _two_endog(seed=4)
    with pytest.raises(ValueError, match="cluster"):
        _wid.kleibergen_paap_rk(D, Z, cov_type="cluster")


def test_kleibergen_paap_unknown_cov_raises():
    D, Z, _ = _two_endog(seed=5)
    with pytest.raises(ValueError, match="Unknown cov_type"):
        _wid.kleibergen_paap_rk(D, Z, cov_type="bogus")


def test_kleibergen_paap_underidentified_raises():
    D, Z, _ = _two_endog(seed=6)
    # 2 endog, only 1 instrument
    with pytest.raises(ValueError, match="Under-identified"):
        _wid.kleibergen_paap_rk(D, Z[:, :1])


def test_sanderson_windmeijer_two_endog_summary():
    D, Z, _ = _two_endog(seed=7)
    r = _wid.sanderson_windmeijer(D, Z, endog_names=["d1", "d2"])
    frame = r.to_frame()
    assert list(frame.index) == ["d1", "d2"]
    s = r.summary()
    assert "Sanderson-Windmeijer" in s
    assert "df_denom" in s


def test_sanderson_windmeijer_single_endog():
    rng = np.random.default_rng(8)
    n = 400
    z = rng.normal(size=(n, 2))
    d = (0.7 * z[:, 0] + 0.3 * z[:, 1] + rng.normal(size=n)).reshape(-1, 1)
    r = _wid.sanderson_windmeijer(d, z, endog_names=["d"])
    assert np.isfinite(r.sw_f["d"])


def test_sanderson_windmeijer_underidentified_raises():
    D, Z, _ = _two_endog(seed=9)
    with pytest.raises(ValueError, match="Under-identified"):
        _wid.sanderson_windmeijer(D, Z[:, :1])


def test_sanderson_windmeijer_name_mismatch_raises():
    D, Z, _ = _two_endog(seed=10)
    with pytest.raises(ValueError, match="endog_names length"):
        _wid.sanderson_windmeijer(D, Z, endog_names=["only_one"])


def test_extract_exog_ndarray():
    W = _wid._extract_exog(None, np.array([[1.0], [2.0], [3.0]]), 3,
                           add_const=True)
    assert W.shape == (3, 2)


def test_residualize_no_exog_passthrough():
    M = np.arange(6.0).reshape(3, 2)
    out = _wid._residualize(M, None)
    np.testing.assert_array_equal(out, M)
    out2 = _wid._residualize(M, np.empty((3, 0)))
    np.testing.assert_array_equal(out2, M)


def test_sanderson_windmeijer_too_few_obs_raises():
    # df2 = n - n_W - k - (p-1) <= 0 with tiny n and many instruments
    rng = np.random.default_rng(20)
    n = 6
    z = rng.normal(size=(n, 5))
    d = rng.normal(size=(n, 1))
    with pytest.raises(ValueError, match="Not enough observations"):
        _wid.sanderson_windmeijer(d, z, endog_names=["d"], add_const=True)


def test_sanderson_windmeijer_perfect_fit_nan():
    # y_j exactly a linear combination of Z -> rss == 0 -> nan branch (440-442)
    n = 50
    rng = np.random.default_rng(21)
    z = rng.normal(size=(n, 2))
    # endog is exact linear combo of instruments (no noise)
    d = (1.3 * z[:, 0] - 0.7 * z[:, 1]).reshape(-1, 1)
    r = _wid.sanderson_windmeijer(d, z, endog_names=["d"], add_const=False)
    # perfect fit -> rss ~ 0; f is inf or the nan branch fires
    assert "d" in r.sw_f


def test_conditional_lr_test_runs_and_summary():
    rng = np.random.default_rng(11)
    n = 500
    z = rng.normal(size=(n, 2))
    v = rng.normal(size=n)
    d = 0.8 * z[:, 0] + 0.4 * z[:, 1] + v
    y = 1.0 + 1.5 * d + 0.5 * v + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "z1": z[:, 0], "z2": z[:, 1]})
    r = _wid.conditional_lr_test(
        y="y", endog="d", instruments=["z1", "z2"], data=df,
        n_simulations=2000, random_state=0,
    )
    assert 0.0 <= r.pvalue <= 1.0
    s = r.summary()
    assert "Moreira" in s and "CLR statistic" in s


def test_conditional_lr_test_array_instruments():
    rng = np.random.default_rng(12)
    n = 400
    z = rng.normal(size=(n, 2))
    v = rng.normal(size=n)
    d = 0.8 * z[:, 0] + 0.4 * z[:, 1] + v
    y = 1.5 * d + 0.5 * v + rng.normal(size=n)
    r = _wid.conditional_lr_test(
        y=y, endog=d, instruments=z, n_simulations=1500, random_state=1,
    )
    assert np.isfinite(r.statistic)
