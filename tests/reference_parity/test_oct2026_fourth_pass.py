"""Fourth batch of known-truth anchors from the 2026-10 pass.

Fixes. ``sp.llm_annotator_correct`` (binary attenuation factor; reported
standard error), ``sp.discos_test`` (the sup-norm test had no reference
distribution), ``sp.overlap_weighted_did`` (unit-level bootstrap for
panels).

Recoveries. The Bayesian estimators against a planted parameter and their
frequentist counterparts (skipped without PyMC), ``sp.bayes_dml`` in its
conjugate mode, ``sp.did_calibrated_simulation`` and the proxy-variable
production-function estimators on a design that satisfies their
scalar-unobservable condition.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

pytestmark = pytest.mark.filterwarnings("ignore")


# --------------------------------------------------------------------- #
#  llm_annotator_correct: attenuation of a misclassified regressor
# --------------------------------------------------------------------- #


def _annotated(seed, prevalence, p01, p10, n=4000, n_val=800):
    """Effect 1.0 of a binary label observed with error; first rows audited."""
    rng = np.random.default_rng(seed)
    t = rng.binomial(1, prevalence, n)
    u = rng.uniform(size=n)
    flip = np.where(t == 1, u < p10, u < p01)
    obs = np.where(flip, 1 - t, t)
    y = 1.0 * t + rng.normal(size=n)
    human = np.where(np.arange(n) < n_val, t, np.nan)
    return pd.Series(obs), pd.Series(y), pd.Series(human)


class TestAnnotatorBinaryCorrection:
    @pytest.mark.parametrize(
        "prevalence, p01, p10, old_limit",
        [
            (0.5, 0.10, 0.10, 1.000),
            (0.2, 0.10, 0.10, 0.832),
            (0.2, 0.05, 0.30, 1.084),
            (0.5, 0.05, 0.30, 1.067),
        ],
    )
    def test_unbiased_at_any_prevalence_and_error_pattern(
        self, prevalence, p01, p10, old_limit
    ):
        # Dividing by 1 - p01 - p10 is the correction for a misclassified
        # outcome. For a misclassified regressor the attenuation is
        # (1 - p01 - p10) Var(T) / Var(T_obs), which is the gap in
        # predictive values; `old_limit` is what the former divisor
        # converged to (300 replications matched it to the third digit).
        est, se = [], []
        for seed in range(30):
            obs, y, human = _annotated(seed, prevalence, p01, p10)
            res = sp.llm_annotator_correct(
                annotations_llm=obs, outcome=y, annotations_human=human
            )
            est.append(res.estimate)
            se.append(res.se)
        mean = float(np.mean(est))
        sem = float(np.std(est, ddof=1) / np.sqrt(len(est)))
        assert abs(mean - 1.0) <= 4.0 * sem
        if abs(old_limit - 1.0) > 0.05:
            assert abs(mean - old_limit) > 4.0 * sem

    def test_reported_se_carries_the_validation_noise(self):
        # 300 replications at prevalence 0.2: across-seed SD 0.074,
        # delta-method SE 0.077, first-order SE 0.059.
        est, se, first = [], [], []
        for seed in range(40):
            obs, y, human = _annotated(seed, 0.2, 0.10, 0.10)
            res = sp.llm_annotator_correct(
                annotations_llm=obs, outcome=y, annotations_human=human
            )
            est.append(res.estimate)
            se.append(res.se)
            first.append(res.model_info["first_order_se"])
            assert res.model_info["se_correction"] == "delta_method"
        sd = float(np.std(est, ddof=1))
        assert 0.8 < np.mean(se) / sd < 1.5
        assert np.mean(first) < 0.9 * np.mean(se)

    def test_continuous_and_multiclass_paths_cover(self):
        rng = np.random.default_rng(0)
        hits = {"continuous": [], "multiclass": []}
        for seed in range(30):
            rng = np.random.default_rng(seed)
            n = 3000
            audited = np.arange(n) < 600
            s = rng.normal(size=n)
            y = 1.0 * s + rng.normal(size=n)
            res = sp.llm_annotator_correct(
                annotations_llm=pd.Series(s + rng.normal(0, 0.7, n)),
                outcome=pd.Series(y),
                annotations_human=pd.Series(np.where(audited, s, np.nan)),
            )
            hits["continuous"].append(abs(res.estimate - 1.0) <= 1.96 * res.se)
            t = rng.choice(3, size=n, p=[0.5, 0.3, 0.2])
            noisy = np.where(rng.uniform(size=n) < 0.15, rng.integers(0, 3, n), t)
            y = 1.0 * (t == 1) + 2.0 * (t == 2) + rng.normal(size=n)
            res = sp.llm_annotator_correct(
                annotations_llm=pd.Series(noisy),
                outcome=pd.Series(y),
                annotations_human=pd.Series(np.where(audited, t, np.nan)),
            )
            hits["multiclass"].append(abs(res.estimate - 1.0) <= 1.96 * res.se)
        # 300 replications each: 96.7% and 98.0%.
        assert np.mean(hits["continuous"]) >= 0.83
        assert np.mean(hits["multiclass"]) >= 0.83


# --------------------------------------------------------------------- #
#  discos_test: a test needs a reference distribution
# --------------------------------------------------------------------- #


def _micro_panel(seed, shift, units=30, periods=5, t0=3, m=80):
    """Individual-level outcomes; unit 0 shifted from period `t0` on."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 0.5, units)
    parts = []
    for j in range(units):
        for t in range(periods):
            y = mu[j] + 0.1 * t + rng.normal(0, 1, m)
            if j == 0 and t >= t0:
                y = y + shift
            parts.append(pd.DataFrame({"unit": j, "time": t, "y": y}))
    return pd.concat(parts, ignore_index=True)


def _fit_discos(seed, shift, placebo=True):
    return sp.discos(
        _micro_panel(seed, shift),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=0,
        treatment_time=3,
        placebo=placebo,
        seed=seed,
    )


class TestDiscosTest:
    def test_sup_norm_test_holds_its_size_under_the_null(self):
        # The 'ks' option used to run scipy's two-sample KS test on the
        # points of the two quantile functions as if they were samples.
        # It rejected a true null in 57% of 40 replications. The
        # statistic is now ranked among the placebo units' (3.3% over 60
        # replications, with 29 placebos).
        rejections = 0
        for seed in range(15):
            out = sp.discos_test(_fit_discos(seed, 0.0), test="ks")
            assert out["n_placebos"] == 29
            assert out["pvalue"] >= 1.0 / 30 - 1e-12
            rejections += int(out["reject"])
        assert rejections <= 3

    def test_sup_norm_test_has_power(self):
        # A shift of 0.8 SD: 60% rejection for 'ks', 97% for 'cvm'.
        ks, cvm = [], []
        for seed in range(8):
            res = _fit_discos(seed, 0.8)
            ks.append(sp.discos_test(res, test="ks")["pvalue"])
            cvm.append(sp.discos_test(res, test="cvm")["pvalue"])
        assert np.mean(np.array(cvm) < 0.05) >= 0.6
        assert np.median(ks) < 0.15

    def test_statistic_is_the_largest_quantile_gap(self):
        res = _fit_discos(0, 0.8)
        out = sp.discos_test(res, test="ks")
        gap = np.abs(
            res.model_info["treated_quantiles"]
            - res.model_info["counterfactual_quantiles"]
        )
        assert out["statistic"] == pytest.approx(float(gap.max()), rel=1e-12)

    @pytest.mark.parametrize("test", ["ks", "cvm", "stochastic_dominance"])
    def test_no_placebos_no_p_value(self, test):
        res = _fit_discos(1, 0.8, placebo=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = sp.discos_test(res, test=test)
        assert np.isnan(out["pvalue"])
        assert np.isfinite(out["statistic"])

    def test_missing_placebos_are_announced(self):
        res = _fit_discos(1, 0.8, placebo=False)
        with pytest.warns(RuntimeWarning, match="no placebo fits"):
            sp.discos_test(res, test="ks")


# --------------------------------------------------------------------- #
#  overlap_weighted_did: resample units on a panel
# --------------------------------------------------------------------- #


def _two_period_panel(seed, n=600):
    """ATT 1.5; a unit effect as large as the noise sits in both periods."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    tr = rng.binomial(1, 1 / (1 + np.exp(-x)))
    u = rng.normal(size=n)
    pre = 1 + 0.5 * x + u + rng.normal(size=n)
    post = 1 + 0.5 * x + u + 0.4 + 1.5 * tr + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": np.r_[pre, post],
            "treat": np.tile(tr, 2),
            "time": np.repeat([0, 1], n),
            "x": np.tile(x, 2),
            "uid": np.tile(np.arange(n), 2),
        }
    )


class TestOverlapWeightedDidPanelSE:
    kw = dict(y="y", treat="treat", time="time", covariates=["x"])

    def test_unit_bootstrap_matches_the_sampling_spread(self):
        # 150 replications: across-seed SD 0.110, SE with id 0.119 (96.7%
        # coverage), SE without id 0.172 (100% coverage).
        est, se_units, se_rows = [], [], []
        for seed in range(25):
            df = _two_period_panel(seed)
            with_id = sp.overlap_weighted_did(df, id="uid", **self.kw)
            without = sp.overlap_weighted_did(df, **self.kw)
            assert with_id.estimate == without.estimate
            est.append(with_id.estimate)
            se_units.append(with_id.se)
            se_rows.append(without.se)
        sd = float(np.std(est, ddof=1))
        assert abs(np.mean(est) - 1.5) <= 4.0 * sd / np.sqrt(len(est))
        assert 0.75 < np.mean(se_units) / sd < 1.5
        assert np.mean(se_rows) > 1.25 * np.mean(se_units)

    def test_bootstrap_kind_is_recorded(self):
        df = _two_period_panel(0, n=200)
        assert (
            sp.overlap_weighted_did(df, id="uid", **self.kw).model_info["bootstrap"]
            == "units"
        )
        assert sp.overlap_weighted_did(df, **self.kw).model_info["bootstrap"] == "rows"


# --------------------------------------------------------------------- #
#  Recoveries without PyMC
# --------------------------------------------------------------------- #


def test_bayes_dml_conjugate_posterior_is_centred_on_the_dml_estimate():
    # A N(0, 10^2) prior against a likelihood with SE 0.02 moves nothing:
    # the posterior is the DML estimate and its standard error.
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 3))
    d = 0.5 * X[:, 0] + rng.normal(size=n)
    y = 1.0 * d + X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"]).assign(d=d, y=y)
    res = sp.bayes_dml(df, y="y", treatment="d", covariates=["x1", "x2", "x3"])
    assert res.posterior_mean == pytest.approx(res.dml_point, abs=1e-4)
    assert res.posterior_sd == pytest.approx(res.dml_se, rel=1e-4)
    assert abs(res.posterior_mean - 1.0) <= 4.0 * res.posterior_sd
    # A tight prior at zero pulls it: precision-weighted average.
    tight = sp.bayes_dml(
        df, y="y", treatment="d", covariates=["x1", "x2", "x3"], prior_sd=res.dml_se
    )
    assert tight.posterior_mean == pytest.approx(res.dml_point / 2, rel=1e-3)
    assert tight.posterior_sd == pytest.approx(res.dml_se / np.sqrt(2), rel=1e-3)


def test_did_calibrated_simulation_recovers_a_planted_effect():
    df = sp.dgp_did(n_units=120, n_periods=8, effect=0.5, staggered=True, seed=0)
    df["g"] = df["first_treat"].fillna(0)
    kw = dict(
        y="y",
        id="unit",
        time="time",
        cohort="g",
        estimators=("twfe", "did_imputation"),
        n_sims=30,
        seed=0,
    )
    null = sp.did_calibrated_simulation(df, effect=0.0, **kw)
    planted = sp.did_calibrated_simulation(df, effect=0.4, **kw)
    frames = []
    for res in (null, planted):
        table = next(v for v in vars(res).values() if isinstance(v, pd.DataFrame))
        frames.append(table.set_index("estimator"))
    for name in ("twfe", "did_imputation"):
        a, b = frames[0].loc[name], frames[1].loc[name]
        # Same draws, effect added on top: the estimate moves by 0.4.
        assert b["mean_estimate"] - a["mean_estimate"] == pytest.approx(0.4, abs=1e-8)
        assert abs(a["bias"]) <= 4.0 * a["mc_se_bias"] + 0.02
        assert b["reject_rate"] > 0.9
        assert a["reject_rate"] < 0.35


def _proxy_panel(seed, firms=300, years=8):
    """Cobb-Douglas with beta_l 0.6, beta_k 0.3. Both proxies are exact
    functions of capital and productivity."""
    rng = np.random.default_rng(seed)
    rows = []
    for fid in range(firms):
        k = rng.normal(2, 0.5)
        om = rng.normal(0, 0.3)
        for yr in range(years):
            om = 0.7 * om + rng.normal(0, 0.2)
            k = 0.9 * k + 0.3 * rng.normal(1, 0.3)
            lab = 0.6 * k + om + rng.normal(1, 0.2)
            y = 0.6 * lab + 0.3 * k + om + rng.normal(0, 0.1)
            rows.append(
                dict(
                    id=fid,
                    year=yr,
                    y=y,
                    l=lab,
                    k=k,
                    i=1 + 0.4 * k + om,
                    m=0.5 + 0.5 * k + om,
                )
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "name, proxy, tol_l, tol_k",
    [
        # Eight seeds: OP and LP 0.601 / 0.302 (SD 0.014 / 0.044); ACF
        # 0.582 / 0.324 (SD 0.057 / 0.076).
        ("olley_pakes", "i", 0.06, 0.18),
        ("levinsohn_petrin", "m", 0.06, 0.18),
        ("ackerberg_caves_frazer", "m", 0.23, 0.31),
    ],
)
def test_proxy_variable_estimators_recover_the_elasticities(name, proxy, tol_l, tol_k):
    res = getattr(sp, name)(
        _proxy_panel(0), output="y", free="l", state="k", proxy=proxy
    )
    assert res.coef["l"] == pytest.approx(0.6, abs=tol_l)
    assert res.coef["k"] == pytest.approx(0.3, abs=tol_k)


# --------------------------------------------------------------------- #
#  Bayesian estimators (need the `bayes` extra)
# --------------------------------------------------------------------- #

NUTS = dict(draws=500, tune=500, chains=2, progressbar=False)


@pytest.fixture(scope="module")
def pymc():
    return pytest.importorskip("pymc")


def test_bayes_did_agrees_with_the_two_by_two_contrast(pymc):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        tr = i < 200
        a = rng.normal()
        for t in range(4):
            post = t >= 2
            y = a + 0.3 * t + 1.5 * (tr and post) + rng.normal()
            rows.append(dict(id=i, t=t, treat=int(tr), post=int(post), y=y))
    df = pd.DataFrame(rows)
    res = sp.bayes_did(
        df, y="y", treat="treat", post="post", unit="id", time="t", **NUTS
    )
    cell = df.groupby(["treat", "post"])["y"].mean()
    did = (cell[1, 1] - cell[1, 0]) - (cell[0, 1] - cell[0, 0])
    assert res.posterior_mean == pytest.approx(did, abs=0.06)
    assert abs(res.posterior_mean - 1.5) <= 4.0 * res.posterior_sd
    assert 0.04 < res.posterior_sd < 0.15


def test_bayes_fuzzy_rd_posterior_covers_the_complier_effect(pymc):
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.uniform(-1, 1, n)
    d = rng.binomial(1, 0.2 + 0.6 * (x >= 0))
    y = 0.5 * x + 1.0 * d + rng.normal(0, 0.5, n)
    df = pd.DataFrame(dict(y=y, d=d, x=x))
    res = sp.bayes_fuzzy_rd(df, y="y", treat="d", running="x", **NUTS)
    assert res.hdi_lower < 1.0 < res.hdi_upper
    assert res.posterior_sd < 0.4


def test_bayes_hte_iv_recovers_the_average_and_the_slope(pymc):
    rng = np.random.default_rng(0)
    n = 3000
    z, m, u = rng.normal(size=(3, n))
    d = 0.8 * z + 0.5 * u + rng.normal(size=n)
    y = (1.0 + 0.5 * m) * d + u + rng.normal(size=n)
    df = pd.DataFrame(dict(y=y, d=d, z=z, m=m))
    res = sp.bayes_hte_iv(
        df, y="y", treat="d", instrument="z", effect_modifiers=["m"], **NUTS
    )
    assert abs(res.posterior_mean - 1.0) <= 4.0 * res.posterior_sd
    slope = res.cate_slopes.set_index("term").loc["m"]
    assert abs(slope["estimate"] - 0.5) <= 4.0 * slope["std_error"]
    # OLS on the same data is confounded upward by u.
    assert np.polyfit(d, y, 1)[0] > 1.15


def test_bayes_its_level_shift(pymc):
    rng = np.random.default_rng(0)
    t = np.arange(200)
    post = t >= 100
    y = 1 + 0.02 * t + 2.0 * post + 0.03 * (t - 100) * post + rng.normal(0, 0.5, 200)
    res = sp.bayes_its(
        pd.DataFrame(dict(y=y, t=t)), y="y", time="t", intervention=100, **NUTS
    )
    assert abs(res.posterior_mean - 2.0) <= 4.0 * res.posterior_sd
    assert res.posterior_sd < 0.3


@pytest.fixture(scope="module")
def roy_data():
    # MTE(u) = 2 - 2u, so the ATE is 1 and the treated-minus-untreated
    # contrast at propensity p is g(p) = 2 - p.
    rng = np.random.default_rng(0)
    n = 3000
    z = rng.normal(size=n)
    v = rng.uniform(size=n)
    p = 1 / (1 + np.exp(-1.2 * z))
    d = (v < p).astype(int)
    y0 = rng.normal(0, 0.5, n)
    y = np.where(d == 1, y0 + 2 - 2 * v, y0)
    return pd.DataFrame(dict(y=y, d=d, z=z))


def test_bayes_mte_latent_mode_recovers_the_mte_curve(pymc, roy_data):
    res = sp.bayes_mte(
        roy_data,
        y="y",
        treat="d",
        instrument="z",
        poly_u=1,
        mte_method="hv_latent",
        draws=400,
        tune=600,
        chains=2,
        progressbar=False,
    )
    slope, intercept = np.polyfit(
        res.mte_curve["u"], res.mte_curve["posterior_mean"], 1
    )
    assert intercept == pytest.approx(2.0, abs=0.25)
    assert slope == pytest.approx(-2.0, abs=0.4)
    assert res.posterior_mean == pytest.approx(1.0, abs=0.15)


def test_bayes_mte_default_mode_is_the_effect_at_propensity(pymc, roy_data):
    # The documented caveat, pinned: the default curve is g(p), which has
    # half the slope of the MTE here and averages 1.5, not 1.
    res = sp.bayes_mte(roy_data, y="y", treat="d", instrument="z", poly_u=1, **NUTS)
    slope, intercept = np.polyfit(
        res.mte_curve["u"], res.mte_curve["posterior_mean"], 1
    )
    assert intercept == pytest.approx(2.0, abs=0.3)
    assert slope == pytest.approx(-1.0, abs=0.4)
    assert "treatment-effect-at-propensity" in res.method


# --------------------------------------------------------------------- #
#  Identities for dispatchers and helpers
# --------------------------------------------------------------------- #


def test_rd_dispatcher_is_rdrobust_by_default():
    df = sp.dgp_rd(n=2000, seed=0)
    via = sp.rd(df, "y", "x", 0)
    direct = sp.rdrobust(df, y="y", x="x", c=0)
    assert via.estimate == direct.estimate
    assert via.se == direct.se


def test_balance_tables_report_the_standardised_mean_difference():
    rng = np.random.default_rng(0)
    n = 3000
    x1, x2 = rng.normal(size=(2, n))
    t = rng.binomial(1, 1 / (1 + np.exp(-x1)))
    df = pd.DataFrame(dict(t=t, x1=x1, x2=x2))

    def smd(v):
        pooled = np.sqrt((v[t == 1].var(ddof=1) + v[t == 0].var(ddof=1)) / 2)
        return (v[t == 1].mean() - v[t == 0].mean()) / pooled

    want = [smd(x1), smd(x2)]
    ps = sp.ps_balance(df, "t", ["x1", "x2"]).table
    diag = sp.balance_diagnostics(df, "t", ["x1", "x2"]).table
    np.testing.assert_allclose(ps["smd_raw"], want, atol=1e-10)
    np.testing.assert_allclose(diag["smd_raw"], want, atol=1e-10)
    # x1 drives selection; the propensity weights balance it.
    assert abs(want[0]) > 0.5
    assert abs(ps["smd_weighted"].iloc[0]) < 0.1
    np.testing.assert_allclose(ps["smd_weighted"], diag["smd_weighted"], atol=1e-10)


def test_spatial_helpers_on_plane_geometry():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString, Point, Polygon

    crs = "EPSG:3857"
    pts = gpd.GeoDataFrame(
        dict(id=[0, 1, 2]),
        geometry=[Point(0, 0), Point(3000, 4000), Point(0, 5000)],
        crs=crs,
    )
    line = gpd.GeoDataFrame(
        dict(id=[0]), geometry=[LineString([(-10000, 0), (10000, 0)])], crs=crs
    )
    polys = gpd.GeoDataFrame(
        dict(pid=["a", "b"]),
        geometry=[
            Polygon([(0, -1000), (4000, -1000), (4000, 1000), (0, 1000)]),
            Polygon([(4000, -1000), (20000, -1000), (20000, 1000), (4000, 1000)]),
        ],
        crs=crs,
    )
    # Distances to the x-axis, in km.
    np.testing.assert_allclose(sp.distance_to_feature(pts, line), [0.0, 4.0, 5.0])
    lengths = sp.line_length_in_polygon(line, polys, polygon_id="pid").set_index("pid")
    assert lengths.loc["a", "line_length_km"] == pytest.approx(4.0)
    assert lengths.loc["b", "line_length_km"] == pytest.approx(6.0)
    # Two of the three points lie within 4.5 km of the line.
    assert sp.share_within_buffer(pts, line, buffer_km=4.5) == pytest.approx(2 / 3)


def _event_panel(seed, pre_slope):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(200):
        tr = i < 100
        a = rng.normal()
        for t in range(8):
            y = a + 0.2 * t + (pre_slope * t if tr else 0.0) + rng.normal()
            if tr and t >= 4:
                y += 1.0
            rows.append(dict(id=i, t=t, g=(4 if tr else np.nan), y=y))
    return pd.DataFrame(rows)


def test_pretrends_test_size_and_power():
    # 40 replications: 2.5% rejection under parallel trends, 42.5% with a
    # differential trend of 0.15 per period.
    null, trend = [], []
    for seed in range(20):
        for slope, store in ((0.0, null), (0.3, trend)):
            es = sp.event_study(
                _event_panel(seed, slope),
                y="y",
                treat_time="g",
                time="t",
                unit="id",
                window=(-4, 3),
            )
            store.append(sp.pretrends_test(es)["pvalue"])
    assert np.mean(np.array(null) < 0.05) <= 0.2
    assert np.mean(np.array(trend) < 0.05) >= 0.7
