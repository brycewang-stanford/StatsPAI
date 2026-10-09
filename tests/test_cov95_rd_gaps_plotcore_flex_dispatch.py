"""Branch tests for ``rd/_rdplot_core.py``, ``rd/rd_flex.py`` and ``sp.rd``.

The numerical core of ``sp.rdplot`` / ``sp.rdplotdensity`` (kernels, scale,
per-side bandwidths, mass points, rank-deficient designs), the fuzzy and
single-fold paths of ``sp.rd.rd_flex``, and the dispatcher routes for the
multi-cutoff, multi-score, 2D and distributional estimators.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.rd import _rdplot_core as pc

rd_pkg = importlib.import_module("statspai.rd")
flex_mod = importlib.import_module("statspai.rd.rd_flex")


def _xy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 1.0 * (x >= 0) + 0.5 * x + rng.normal(0, 0.3, n)
    return x, y


# ------------------------------------------------------- rdplot numbers


def test_epanechnikov_weights_and_singular_gram():
    x = np.array([-0.5, 0.0, 0.25, 2.0])
    w = pc._kweight(x, 0.0, 0.5, "epa")
    np.testing.assert_allclose(w, [0.0, 1.5, 1.125, 0.0])
    np.testing.assert_allclose(w, pc._kweight(x, 0.0, 0.5, "epanechnikov"))
    # two identical columns: not positive definite
    assert pc._xx_inv(np.ones((5, 2))) is None
    np.testing.assert_allclose(pc._xx_inv(np.eye(2) * 2.0), np.eye(2) / 4.0)


def test_rdplot_numbers_validation():
    x, y = _xy(100)
    with pytest.raises(MethodIncompatibility, match="binselect must be one of"):
        pc.rdplot_numbers(y, x, binselect="quantile")
    with pytest.raises(MethodIncompatibility, match="within the range of x"):
        pc.rdplot_numbers(y, x, c=5.0)


def test_scale_multiplies_the_selected_number_of_bins():
    x, y = _xy()
    base = pc.rdplot_numbers(y, x, p=2)
    twice = pc.rdplot_numbers(y, x, p=2, scale=2)
    mixed = pc.rdplot_numbers(y, x, p=2, scale=(1, 3))
    assert twice["J"] == (2 * base["J"][0], 2 * base["J"][1])
    assert mixed["J"] == (base["J"][0], 3 * base["J"][1])
    assert len(mixed["vars_bins"]["rdplot_mean_y"]) == sum(mixed["J"])


def test_per_side_bandwidth_and_one_dimensional_covariate():
    x, y = _xy()
    rng = np.random.default_rng(9)
    z = rng.normal(size=len(x))
    out = pc.rdplot_numbers(y, x, p=1, h=(0.3, 0.6), kernel="triangular")
    assert out["h"] == (0.3, 0.6)
    assert out["N_h"] == (
        int(((x < 0) & (x >= -0.3)).sum()),
        int((x >= 0).sum() - (x > 0.6).sum()),
    )
    flat = pc.rdplot_numbers(y + z, x, p=1, covs=z)
    col = pc.rdplot_numbers(y + z, x, p=1, covs=z.reshape(-1, 1))
    np.testing.assert_allclose(
        flat["vars_poly"]["rdplot_y"], col["vars_poly"]["rdplot_y"], rtol=1e-12
    )


def test_mass_points_switch_spacings_rules_to_polynomial_ones():
    rng = np.random.default_rng(1)
    x = np.repeat([-2.0, -1.0, 1.0, 2.0], 40)
    y = (x >= 0) + 0.1 * x + rng.normal(0, 0.1, 160)
    out = pc.rdplot_numbers(y, x, p=1, binselect="qs", masspoints="adjust")
    assert out["mass_points_adjusted"] is True and out["binselect"] == "qspr"
    off = pc.rdplot_numbers(y, x, p=1, binselect="espr", masspoints="off")
    assert off["mass_points_adjusted"] is False and off["binselect"] == "espr"


def test_rank_deficient_polynomial_uses_the_pseudo_inverse():
    # Two support points per side cannot identify a quartic; the fit must
    # still interpolate the two cell means (minimum-norm solution).
    rng = np.random.default_rng(2)
    x = np.repeat([-2.0, -1.0, 1.0, 2.0], 40)
    y = (x >= 0) + 0.1 * x + rng.normal(0, 0.1, 160)
    out = pc.rdplot_numbers(y, x, p=4, nbins=2, masspoints="off")
    xs, ys = out["vars_poly"]["rdplot_x"], out["vars_poly"]["rdplot_y"]
    assert np.isfinite(ys).all()
    for v in (-2.0, 2.0):
        fitted = ys[np.argmin(np.abs(xs - v))]
        assert fitted == pytest.approx(y[x == v].mean(), abs=1e-6)


# ---------------------------------------------------------- lpdensity


@pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov"])
def test_lpdensity_without_ties_does_not_depend_on_the_mass_point_switch(kernel):
    rng = np.random.default_rng(3)
    data = rng.normal(size=400)
    grid = np.array([-0.5, 0.0, 0.5])
    a = pc.lpdensity_numbers(data, grid, 0.8, kernel=kernel, mass_points=True)
    b = pc.lpdensity_numbers(data, grid, 0.8, kernel=kernel, mass_points=False)
    for key in ("f_p", "f_q", "se_p", "se_q"):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-9)
    # and it estimates the N(0, 1) density
    np.testing.assert_allclose(b["f_p"], [0.352, 0.399, 0.352], atol=0.08)


@pytest.mark.parametrize("mass_points", [True, False])
def test_lpdensity_is_nan_where_the_window_is_empty(mass_points):
    data = np.linspace(0.0, 1.0, 50)
    out = pc.lpdensity_numbers(
        data, np.array([0.5, 10.0]), 0.3, mass_points=mass_points
    )
    assert out["nh"].tolist()[1] == 0
    assert np.isfinite(out["f_p"][0]) and np.isnan(out["f_p"][1])
    assert np.isnan(out["se_p"][1])


# ------------------------------------------------------------- rd_flex


def _flex_df(n=900, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.1 + 0.8 * (x >= 0)).astype(float)
    y = 2.0 * d + 0.4 * x + 1.5 * w1 - 1.0 * w2 + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {"y": y, "x": x, "d": d, "w1": w1, "w2": w2, "cl": rng.integers(0, 90, n)}
    )


def test_rd_flex_fuzzy_clustered_recovers_the_effect_and_cuts_variance():
    pytest.importorskip("sklearn")
    df = _flex_df()
    res = sp.rd.rd_flex(
        df,
        y="y",
        x="x",
        W=["w1", "w2"],
        learner="ridge",
        fuzzy="d",
        cluster="cl",
        n_folds=3,
        random_state=0,
    )
    assert abs(res.estimate - 2.0) < 0.5
    info = res.model_info
    flex = info.get("flex", info)
    assert flex["se_flex"] < flex["se_plain"]
    assert flex["var_reduction"] > 0.5
    with pytest.raises((KeyError, ValueError, MethodIncompatibility), match="nope"):
        sp.rd.rd_flex(df, y="y", x="x", W=["w1"], fuzzy="nope", learner="ridge")


def test_rd_flex_single_fold_is_the_in_sample_fit():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    df = _flex_df(200)
    W, y = df[["w1", "w2"]].to_numpy(), df["y"].to_numpy()
    resid, r2 = flex_mod._crossfit_residualise(W, y, "ridge", LinearRegression(), 1, 0)
    X = np.column_stack([np.ones(len(y)), W])
    ols = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(resid, ols, atol=1e-9)
    assert r2 == pytest.approx(1 - ols @ ols / np.sum((y - y.mean()) ** 2))
    # named learner instead of a supplied estimator
    resid2, r2b = flex_mod._crossfit_residualise(W, y, "ridge", None, 1, 0)
    assert 0 < r2b <= r2 + 1e-12 and abs(resid2.mean()) < 1e-8


def test_rd_flex_unknown_learner():
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError, match="Unknown learner 'svm'"):
        flex_mod._make_learner("svm", 0)


# ---------------------------------------------------------- dispatcher


def _multi_df(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = (x >= 0) * 1.0 + (x >= 0.5) * 0.5 + 0.3 * x + rng.normal(0, 0.3, n)
    x2 = rng.uniform(-1, 1, n)
    return pd.DataFrame({"x": x, "y": y, "x2": x2, "d": (x >= 0).astype(float)})


def test_rd_rename_moves_the_alias_and_refuses_both():
    kw = {"x": "score", "other": 1}
    rd_pkg._rd_rename(kw, {"x": "running", "c": "cutoff"})
    assert kw == {"running": "score", "other": 1}
    with pytest.raises(TypeError, match="pick one"):
        rd_pkg._rd_rename({"x": "a", "running": "b"}, {"x": "running"})


def test_dispatcher_running_alias():
    df = _multi_df()
    with pytest.raises(TypeError, match="both 'running' and 'x'"):
        sp.rd(df, y="y", x="x", running="x")
    a = sp.rd(df, y="y", running="x", cutoff=0.0, h=0.3, manipulation_test=False)
    b = sp.rdrobust(df, y="y", x="x", c=0.0, h=0.3, manipulation_test=False)
    assert a.estimate == b.estimate and a.se == b.se


def test_dispatcher_multi_cutoff_routes():
    df = _multi_df()
    direct = sp.rdmc(df, y="y", x="x", cutoffs=[0.0, 0.5])
    via_kw = sp.rd(df, y="y", x="x", method="rdmc", cutoffs=[0.0, 0.5])
    via_c = sp.rd(df, y="y", x="x", c=[0.0, 0.5], method="rdmc")
    assert via_kw.pooled_estimate == direct.pooled_estimate
    assert via_c.pooled_estimate == direct.pooled_estimate
    assert direct.n_cutoffs == 2

    ext = sp.rd_multi_extrapolate(df, y="y", x="x", cutoffs=[-0.3, 0.3])
    assert (
        sp.rd(df, y="y", x="x", c=[-0.3, 0.3], method="multi_extrapolate").estimate
        == ext.estimate
    )
    assert (
        sp.rd(
            df, y="y", x="x", method="rd_multi_extrapolate", cutoffs=[-0.3, 0.3]
        ).estimate
        == ext.estimate
    )


def test_dispatcher_multi_score_2d_and_distributional_routes():
    df = _multi_df()
    ms = sp.rd(df, y="y", method="rdms", x1="x", x2="x2")
    assert ms.estimate == sp.rdms(df, y="y", x1="x", x2="x2").estimate

    two = sp.rd(df, y="y", method="rd2d", x1="x", x2="x2", treatment="d", h=0.5)
    ref = sp.rd2d(df, y="y", x1="x", x2="x2", treatment="d", h=0.5)
    assert two.estimate == ref.estimate and abs(two.estimate - 1.0) < 0.4

    ddd = sp.rd(df, y="y", x="x", c=0.0, method="distributional_design")
    ref = sp.rd.rd_distributional_design(df, y="y", running="x", cutoff=0.0)
    np.testing.assert_allclose(
        np.asarray(ddd.rdd_effect, dtype=float),
        np.asarray(ref.rdd_effect, dtype=float),
    )
    assert ddd.bandwidth == ref.bandwidth
