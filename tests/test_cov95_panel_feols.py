"""Coverage tests for statspai.panel.feols / hdfe / kernels / hdfe_rust.

Covers the FEOLS summary/repr, the no-FE fallback (_ols_no_fe), pure
absorption (no regressors), se_type override, ndarray weights, list
cluster, the wild-bootstrap branch, plus HDFE primitive edge cases
(NaN-in-FE raise, K==0 raise, invalid solver, weighted demean) and the
Rust-bridge RuntimeError fallback. Real synthetic panels; properties
asserted, not fabricated numbers.
"""
import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.panel.feols import feols, FEOLSResult
from statspai.panel import _hdfe_kernels as kernels
from statspai.panel import hdfe_rust


@pytest.fixture
def fe_df():
    rng = np.random.default_rng(3)
    n_firm, n_year = 20, 8
    rows = []
    for f in range(n_firm):
        fe_f = rng.normal(0, 1)
        for yr in range(n_year):
            fe_y = 0.2 * yr
            x1 = rng.normal()
            x2 = rng.normal()
            y = 1.0 + 0.5 * x1 - 0.3 * x2 + fe_f + fe_y + rng.normal(0, 0.4)
            rows.append({"firm": f, "year": yr, "y": y,
                         "x1": x1, "x2": x2, "w": 1.0 + abs(rng.normal())})
    return pd.DataFrame(rows)


# ── FEOLS basic + summary/repr/coef/se ──────────────────────────────────

def test_feols_twoway_fe_summary(fe_df):
    res = feols("y ~ x1 + x2 | firm + year", data=fe_df, cluster="firm")
    assert isinstance(res, FEOLSResult)
    s = res.summary()
    assert "FEOLS" in s
    assert "R² (within)" in s or "within" in s
    assert repr(res) == s
    # coef / se property aliases
    assert res.coef.equals(res.params)
    assert res.se.equals(res.std_errors)
    # within R2 sane, df_resid positive, SEs positive
    assert 0.0 <= res.r2_within <= 1.0
    assert res.df_resid > 0
    assert (res.std_errors > 0).all()
    assert res.se_type == "cluster"


def test_feols_multiway_cluster_se_type(fe_df):
    res = feols("y ~ x1 | firm + year", data=fe_df, cluster=["firm", "year"])
    assert res.se_type == "multiway_cluster"
    assert res.cluster_info["cluster"] == ["firm", "year"]
    assert len(res.cluster_info["n_clusters"]) == 2


def test_feols_se_type_override(fe_df):
    res = feols("y ~ x1 | firm", data=fe_df, se_type="iid")
    assert res.se_type == "iid"


def test_feols_ndarray_weights(fe_df):
    w = fe_df["w"].to_numpy()
    res = feols("y ~ x1 | firm", data=fe_df, weights=w)
    assert isinstance(res, FEOLSResult)
    assert (res.std_errors > 0).all()


def test_feols_string_weights(fe_df):
    res = feols("y ~ x1 | firm", data=fe_df, weights="w")
    assert isinstance(res, FEOLSResult)


def test_feols_wild_cluster_bootstrap(fe_df):
    res = feols("y ~ x1 | firm", data=fe_df, cluster="firm",
                wild=True, wild_n_boot=199, wild_seed=0)
    assert res.se_type == "wild_cluster"
    assert "wild_p" in res.cluster_info
    assert "wild_ci" in res.cluster_info
    for name in res.params.index:
        assert 0.0 <= res.cluster_info["wild_p"][name] <= 1.0


def test_feols_wild_multiway_not_implemented(fe_df):
    with pytest.raises(NotImplementedError, match="multi-way"):
        feols("y ~ x1 | firm", data=fe_df,
              cluster=["firm", "year"], wild=True)


# ── no-FE fallback (_ols_no_fe) ─────────────────────────────────────────

def test_feols_no_fe_iid(fe_df):
    res = feols("y ~ x1 + x2", data=fe_df)
    assert isinstance(res, FEOLSResult)
    assert res.se_type == "iid"
    assert "_const" in res.params.index
    assert res.dof_fe == 0
    assert res.n_fe == []


def test_feols_no_fe_clustered(fe_df):
    res = feols("y ~ x1 + x2", data=fe_df, cluster="firm")
    assert res.se_type == "cluster"
    assert (res.std_errors >= 0).all()


def test_feols_no_fe_no_regressors_raises(fe_df):
    with pytest.raises(ValueError, match="at least one regressor or one FE"):
        feols("y ~ 1", data=fe_df)


def test_feols_bad_formula_raises(fe_df):
    with pytest.raises(ValueError, match="parse"):
        feols("not a formula", data=fe_df)


def test_feols_non_bare_column_raises(fe_df):
    with pytest.raises(ValueError, match="bare column names"):
        feols("y ~ np.log(x1) | firm", data=fe_df)


def test_feols_empty_after_dropna():
    df = pd.DataFrame({"y": [np.nan, np.nan], "x1": [1.0, 2.0],
                       "firm": [0, 1]})
    with pytest.raises(ValueError, match="No non-missing rows"):
        feols("y ~ x1 | firm", data=df)


# ── HDFE primitives via sp.demean / sp.absorb_ols ───────────────────────

def test_demean_nan_in_fe_raises(fe_df):
    fe = fe_df["firm"].astype(float).to_numpy()
    fe[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sp.demean(fe_df["y"].to_numpy(), fe)


def test_absorb_ols_weighted_and_cluster(fe_df):
    y = fe_df["y"].to_numpy()
    X = fe_df[["x1", "x2"]].to_numpy()
    fe = fe_df[["firm", "year"]].to_numpy()
    w = fe_df["w"].to_numpy()
    out = sp.absorb_ols(y, X, fe, weights=w, cluster=fe_df["firm"].to_numpy())
    assert out["coef"].shape == (2,)
    assert np.all(out["se"] >= 0)


def test_absorb_ols_multiway_cluster(fe_df):
    y = fe_df["y"].to_numpy()
    X = fe_df[["x1"]].to_numpy()
    fe = fe_df[["firm"]].to_numpy()
    cl = [fe_df["firm"].to_numpy(), fe_df["year"].to_numpy()]
    out = sp.absorb_ols(y, X, fe, cluster=cl)
    assert out["coef"].shape == (1,)


def test_absorber_invalid_solver_raises(fe_df):
    with pytest.raises(ValueError, match="solver"):
        sp.Absorber(fe_df[["firm"]].to_numpy(), solver="not_a_solver")


def test_demean_1d_standalone(fe_df):
    # 1-D demean path + standalone sp.demean returning (xw, keep_mask)
    xw, keep = sp.demean(fe_df["y"].to_numpy(), fe_df[["firm"]].to_numpy())
    assert xw.ndim == 1
    assert keep.dtype == bool
    assert xw.shape[0] == int(keep.sum())


def test_absorber_residualize_alias_and_repr(fe_df):
    ab = sp.Absorber(fe_df[["firm", "year"]].to_numpy())
    x = fe_df["x1"].to_numpy()
    r1 = ab.residualize(x)
    r2 = ab.demean(x)
    assert np.allclose(r1, r2)
    assert "Absorber" in repr(ab)


@pytest.mark.parametrize("solver", ["lsmr", "lsqr"])
def test_absorb_ols_krylov_solvers(fe_df, solver):
    y = fe_df["y"].to_numpy()
    X = fe_df[["x1", "x2"]].to_numpy()
    fe = fe_df[["firm", "year"]].to_numpy()
    out = sp.absorb_ols(y, X, fe, solver=solver)
    out_map = sp.absorb_ols(y, X, fe, solver="map")
    # Krylov and MAP within-projection should agree closely
    assert np.allclose(out["coef"], out_map["coef"], atol=1e-4)


def test_absorb_ols_weighted_solve(fe_df):
    y = fe_df["y"].to_numpy()
    X = fe_df[["x1", "x2"]].to_numpy()
    fe = fe_df[["firm"]].to_numpy()
    w = fe_df["w"].to_numpy()
    out = sp.absorb_ols(y, X, fe, weights=w)
    assert out["coef"].shape == (2,)


def test_absorb_ols_weighted_krylov(fe_df):
    # Weighted Krylov path (sqrt-w row scaling of the FE design)
    y = fe_df["y"].to_numpy()
    X = fe_df[["x1", "x2"]].to_numpy()
    fe = fe_df[["firm", "year"]].to_numpy()
    w = fe_df["w"].to_numpy()
    out = sp.absorb_ols(y, X, fe, weights=w, solver="lsmr")
    assert out["coef"].shape == (2,)


def test_absorb_ols_df_exhausted_raises():
    # n_kept tiny, too many regressors/FE groups -> df_resid <= 0
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "x1": [0.1, 0.9, 0.4, 0.7],
        "x2": [0.2, 0.3, 0.8, 0.5],
        "x3": [0.5, 0.1, 0.6, 0.2],
        "fe": [0, 0, 1, 1],
    })
    with pytest.raises(ValueError, match="Degrees of freedom exhausted"):
        sp.absorb_ols(df["y"].to_numpy(),
                      df[["x1", "x2", "x3"]].to_numpy(),
                      df[["fe"]].to_numpy())


def test_absorb_ols_no_fe_columns_raises(fe_df):
    empty_fe = np.empty((len(fe_df), 0))
    with pytest.raises(ValueError, match="at least one fixed-effect column"):
        sp.absorb_ols(fe_df["y"].to_numpy(),
                      fe_df[["x1"]].to_numpy(), empty_fe)


# ── kernel dispatchers (numba present) ──────────────────────────────────

def test_kernels_sweep_dispatch():
    col = np.array([1.0, 2.0, 3.0, 4.0])
    codes = np.array([0, 0, 1, 1], dtype=np.int64)
    counts = np.array([2.0, 2.0])
    kernels.sweep(col, codes, counts)
    # group means subtracted: group0 mean 1.5, group1 mean 3.5
    assert np.allclose(col, [-0.5, 0.5, -0.5, 0.5])


def test_kernels_sweep_weighted_dispatch():
    col = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    codes = np.array([0, 0, 1, 1], dtype=np.int64)
    wsum = np.array([2.0, 2.0])
    kernels.sweep_weighted(col, weights, codes, wsum)
    assert np.allclose(col, [-0.5, 0.5, -0.5, 0.5])


def test_kernels_numpy_fallback_paths():
    # Exercise the pure-NumPy reference kernels directly (the Numba path is
    # JIT-compiled and invisible to coverage; the NumPy ones are the
    # fallback when Numba is absent and must stay correct).
    col = np.array([1.0, 2.0, 3.0, 4.0])
    codes = np.array([0, 0, 1, 1], dtype=np.int64)
    counts = np.array([2.0, 2.0])
    kernels._sweep_numpy(col, codes, counts)
    assert np.allclose(col, [-0.5, 0.5, -0.5, 0.5])

    col2 = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    wsum = np.array([2.0, 2.0])
    kernels._sweep_weighted_numpy(col2, weights, codes, wsum)
    assert np.allclose(col2, [-0.5, 0.5, -0.5, 0.5])


# ── Rust bridge fallback (extension absent on main) ─────────────────────

def test_rust_bridge_raises_when_unavailable():
    if hdfe_rust.HAS_RUST:
        pytest.skip("Rust extension installed; fallback path not reachable.")
    codes = np.array([0, 1], dtype=np.int64)
    y = np.array([1.0, 2.0])
    sums = np.zeros(2)
    counts = np.array([1, 1], dtype=np.int64)
    with pytest.raises(RuntimeError, match="not installed"):
        hdfe_rust.group_demean_rust(codes, y, sums, counts)
