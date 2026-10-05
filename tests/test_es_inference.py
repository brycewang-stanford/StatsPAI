"""Joint event-study covariance and uniform bands across the DiD family.

``sp.event_study_vcov`` reads the joint covariance each estimator produces;
``sp.uniform_bands`` turns it into a sup-t band; ``sp.honest_did`` now uses
it for the Rambachan-Roth FLCI on every estimator that exposes one.
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

_FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "es_inference"


@pytest.fixture(scope="module")
def mpdta():
    return sp.datasets.mpdta()


def _fits(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "cs": sp.callaway_santanna(
                df, y="lemp", g="first_treat", t="year", i="countyreal"
            ),
            "aggte": sp.aggte(
                sp.callaway_santanna(
                    df, y="lemp", g="first_treat", t="year", i="countyreal"
                ),
                type="dynamic",
                bstrap=False,
            ),
            "twfe": sp.event_study(
                df, y="lemp", treat_time="first_treat", time="year", unit="countyreal"
            ),
            "sa": sp.sun_abraham(
                df, y="lemp", g="first_treat", t="year", i="countyreal"
            ),
            "gardner": sp.gardner_did(
                df,
                y="lemp",
                group="countyreal",
                time="year",
                first_treat="first_treat",
                event_study=True,
                horizon=[-3, -2, -1, 0, 1, 2, 3],
            ),
            "bjs": sp.did_imputation(
                df,
                y="lemp",
                group="countyreal",
                time="year",
                first_treat="first_treat",
                horizon=[-3, -2, -1, 0, 1, 2, 3],
            ),
            "bjs_insample": sp.did_imputation(
                df,
                y="lemp",
                group="countyreal",
                time="year",
                first_treat="first_treat",
                horizon=[-3, -2, -1, 0, 1, 2, 3],
                pretrend_method="in-sample",
            ),
            "stacked": sp.stacked_did(
                df,
                y="lemp",
                group="countyreal",
                time="year",
                first_treat="first_treat",
                window=(-3, 3),
            ),
            "lp": sp.lp_did(
                df,
                y="lemp",
                unit="countyreal",
                time="year",
                treatment="treat",
                horizons=(-3, 3),
            ),
            "dcdh": sp.did_multiplegt_dyn(
                df,
                y="lemp",
                group="countyreal",
                time="year",
                treatment="treat",
                dynamic=3,
                placebo=3,
                se_method="analytic",
            ),
            "etwfe": sp.etwfe(
                df,
                y="lemp",
                group="countyreal",
                time="year",
                first_treat="first_treat",
                cgroup="nevertreated",
            ),
        }


@pytest.fixture(scope="module")
def fits(mpdta):
    return _fits(mpdta)


_JOINT = {
    "cs": True,
    "aggte": True,
    "twfe": True,
    "sa": True,
    "gardner": True,
    "bjs": False,  # leads from a separate regression: block diagonal
    "bjs_insample": True,
    "stacked": True,
    "lp": True,
    "dcdh": True,
    "etwfe": True,
}


@pytest.mark.parametrize("name", sorted(_JOINT))
def test_diagonal_reproduces_reported_ses(fits, name):
    """Correctness: sqrt(diag V) equals each estimator's own SEs."""
    es = sp.event_study_vcov(fits[name])
    assert es.joint is _JOINT[name]
    assert es.vcov.shape == (es.times.size, es.times.size)
    assert np.allclose(es.vcov, es.vcov.T, rtol=0, atol=1e-15)
    # PSD up to rounding.
    assert np.linalg.eigvalsh(es.vcov).min() > -1e-12
    fit = fits[name]
    if name == "etwfe":
        fit = sp.etwfe_emfx(fit, type="event", include_leads=True)
    if name in ("cs",):
        fit = fits["aggte"]
    from statspai.did.es_inference import _es_frame

    table = _es_frame(fit).set_index("relative_time")
    se = table.loc[es.times, "se"].to_numpy(dtype=float)
    np.testing.assert_allclose(np.sqrt(np.diag(es.vcov)), se, rtol=1e-12)
    np.testing.assert_allclose(
        es.beta, table.loc[es.times, "att"].to_numpy(dtype=float), rtol=1e-12
    )


def test_bjs_joint_covariance_matches_stata_did_imputation(mpdta):
    """The horizon block reproduces Stata did_imputation's e(V).

    Tolerance 2e-5: Stata's iterative fixed-effect solver leaves its own
    *point estimates* up to 8e-6 relative from the exact least-squares
    imputation (StatsPAI's, reproduced independently by a dense lstsq in
    this test); e(V) inherits that gap (observed 1.3e-6).
    """
    ref = json.loads(
        (_FIX / "mpdta_did_imputation_stata.json").read_text(encoding="utf-8")
    )
    fit = sp.did_imputation(
        mpdta,
        y="lemp",
        group="countyreal",
        time="year",
        first_treat="first_treat",
        horizon=[0, 1, 2, 3],
    )
    V = fit.model_info["event_study_vcov"].loc[[0, 1, 2, 3], [0, 1, 2, 3]].to_numpy()
    np.testing.assert_allclose(V, np.asarray(ref["V"]), rtol=2e-5)
    np.testing.assert_allclose(
        fit.model_info["event_study"]["att"].to_numpy(), ref["b"], rtol=2e-5
    )

    # The exact imputation, independently: dense OLS of Y(0) on unit and
    # year dummies over untreated rows.
    d = mpdta.copy()
    D = ((d.first_treat > 0) & (d.year >= d.first_treat)).to_numpy()
    X = np.hstack(
        [
            pd.get_dummies(d.countyreal.astype(str)).to_numpy(dtype=float),
            pd.get_dummies(d.year.astype(str), drop_first=True).to_numpy(dtype=float),
        ]
    )
    b = np.linalg.lstsq(X[~D], d.lemp.to_numpy()[~D], rcond=None)[0]
    tau = d.lemp.to_numpy() - X @ b
    rel = (d.year - d.first_treat).to_numpy()
    exact = [tau[D & (rel == k)].mean() for k in range(4)]
    np.testing.assert_allclose(
        fit.model_info["event_study"]["att"].to_numpy(), exact, rtol=1e-9
    )


def test_dcdh_analytic_cross_covariance_agrees_with_bootstrap(mpdta):
    """T3: analytic cross-horizon correlations vs the unit bootstrap."""
    kw = dict(
        y="lemp",
        group="countyreal",
        time="year",
        treatment="treat",
        dynamic=3,
        placebo=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.event_study_vcov(
            sp.did_multiplegt_dyn(mpdta, se_method="analytic", **kw)
        )
        b = sp.event_study_vcov(
            sp.did_multiplegt_dyn(
                mpdta, se_method="bootstrap", n_boot=400, seed=3, **kw
            )
        )
    ca = a.vcov / np.outer(np.sqrt(np.diag(a.vcov)), np.sqrt(np.diag(a.vcov)))
    cb = b.vcov / np.outer(np.sqrt(np.diag(b.vcov)), np.sqrt(np.diag(b.vcov)))
    # 400 draws: the SE of a correlation near 0.5 is ~0.04; allow 4 SEs.
    assert np.max(np.abs(ca - cb)) < 0.16


def test_uniform_band_is_wider_than_pointwise_and_contains_it(fits):
    band = sp.uniform_bands(fits["sa"], which="post")
    assert band.attrs["joint"] is True
    assert band.attrs["crit_uniform"] > band.attrs["crit_pointwise"]
    assert (band["cband_lower"] <= band["ci_lower"]).all()
    assert (band["cband_upper"] >= band["ci_upper"]).all()
    assert (band["relative_time"] >= 0).all()


def test_sidak_critical_value_under_independence():
    """Diagonal covariance: the sup-t value is the Sidak quantile."""
    k, alpha = 5, 0.05
    es = pd.DataFrame(
        {"relative_time": np.arange(k), "att": np.zeros(k), "se": np.ones(k)}
    )
    res = sp.CausalResult(
        method="toy",
        estimand="ATT",
        estimate=0.0,
        se=1.0,
        pvalue=1.0,
        ci=(-1.0, 1.0),
        alpha=alpha,
        n_obs=10,
        model_info={"event_study": es},
    )
    with pytest.warns(UserWarning, match="no joint event-study covariance"):
        band = sp.uniform_bands(res, alpha=alpha, n_draws=400_000, seed=11)
    sidak = stats.norm.ppf((1 + (1 - alpha) ** (1 / k)) / 2)
    assert band.attrs["joint"] is False
    # Monte Carlo SE of a 95% quantile of max|Z| with 4e5 draws is ~0.003.
    assert band.attrs["crit_uniform"] == pytest.approx(sidak, abs=0.01)


def test_perfect_correlation_collapses_to_pointwise():
    k = 4
    V = np.full((k, k), 0.25)
    es = pd.DataFrame(
        {"relative_time": np.arange(k), "att": np.zeros(k), "se": np.full(k, 0.5)}
    )
    res = sp.CausalResult(
        method="toy",
        estimand="ATT",
        estimate=0.0,
        se=1.0,
        pvalue=1.0,
        ci=(-1.0, 1.0),
        alpha=0.05,
        n_obs=10,
        model_info={
            "event_study": es,
            "event_study_vcov": pd.DataFrame(V, index=range(k), columns=range(k)),
        },
    )
    band = sp.uniform_bands(res, n_draws=200_000, seed=5)
    assert band.attrs["crit_uniform"] == pytest.approx(1.959964, abs=0.01)


def test_uniform_band_simultaneous_coverage():
    """Gaussian coverage of the sup-t band under an AR(1) correlation."""
    rng = np.random.default_rng(0)
    k = 6
    R = 0.7 ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    se = np.linspace(0.5, 1.5, k)
    V = R * np.outer(se, se)
    frame = pd.DataFrame({"relative_time": np.arange(k), "att": np.zeros(k), "se": se})
    base = dict(
        method="toy",
        estimand="ATT",
        estimate=0.0,
        se=1.0,
        pvalue=1.0,
        ci=(-1.0, 1.0),
        alpha=0.05,
        n_obs=10,
    )
    res = sp.CausalResult(
        **base,
        model_info={
            "event_study": frame,
            "event_study_vcov": pd.DataFrame(V, index=range(k), columns=range(k)),
        },
    )
    c = sp.uniform_bands(res, n_draws=200_000, seed=1).attrs["crit_uniform"]
    L = np.linalg.cholesky(V)
    draws = rng.standard_normal((20_000, k)) @ L.T
    covered = np.all(np.abs(draws) <= c * se, axis=1).mean()
    # MC SE of the coverage with 2e4 draws is ~0.0015.
    assert covered == pytest.approx(0.95, abs=0.006)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"alpha": 1.2}, "alpha"),
        ({"which": "middle"}, "which"),
        ({"n_draws": 10}, "n_draws"),
        ({"window": (7, 9)}, "No 'all' event times"),
        ({"window": "x"}, "window"),
    ],
)
def test_uniform_bands_rejects_bad_arguments(fits, kwargs, match):
    with pytest.raises(sp.MethodIncompatibility, match=match):
        sp.uniform_bands(fits["sa"], **kwargs)


def test_event_study_vcov_rejects_non_event_study(mpdta):
    fit = sp.gardner_did(
        mpdta, y="lemp", group="countyreal", time="year", first_treat="first_treat"
    )
    with pytest.raises(sp.MethodIncompatibility):
        sp.event_study_vcov(fit)


def test_honest_did_uses_flci_beyond_callaway_santanna(fits):
    """The FLCI now runs on every estimator with a joint covariance."""
    for name in ("sa", "twfe", "gardner", "stacked", "etwfe", "lp", "dcdh"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = sp.honest_did(fits[name], e=0, method="smoothness")
        assert out.attrs["interval"] == "flci", name
    # BJS with auxiliary-regression leads has no pre/post cross covariance,
    # so it still falls back -- loudly.
    with pytest.warns(UserWarning, match="worst-case-bias"):
        out = sp.honest_did(fits["bjs"], e=0, method="smoothness")
    assert out.attrs["interval"] == "worst_case_bias"


def test_etwfe_and_sun_abraham_flci_agree(fits):
    """Both equal CS never-treated without covariates, so the FLCI agrees."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.honest_did(fits["etwfe"], e=0, method="smoothness")
        b = sp.honest_did(fits["sa"], e=0, method="smoothness")
    np.testing.assert_allclose(a["ci_lower"], b["ci_lower"], atol=5e-4)
    np.testing.assert_allclose(a["ci_upper"], b["ci_upper"], atol=5e-4)


def test_etwfe_emfx_event_headline_has_delta_method_se(fits):
    r = sp.etwfe_emfx(fits["etwfe"], type="event", include_leads=True)
    V = r.model_info["vcov"]
    post = [t for t in V.index if t >= 0]
    w = np.full(len(post), 1.0 / len(post))
    assert np.isfinite(r.se) and r.se > 0
    assert r.se == pytest.approx(float(np.sqrt(w @ V.loc[post, post].to_numpy() @ w)))
    det = r.detail.set_index("event_time")
    assert r.estimate == pytest.approx(float(det.loc[post, "estimate"].mean()))


def test_ddd_additive_covariates_warn_and_record():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "d": rng.integers(0, 2, n),
            "t": rng.integers(0, 2, n),
            "g": rng.integers(0, 2, n),
            "x": rng.normal(size=n),
        }
    )
    df["y"] = df.d * df.t * df.g + df.x + rng.normal(size=n)
    with pytest.warns(UserWarning, match="ddd_heterogeneous"):
        r = sp.ddd(df, y="y", treat="d", time="t", subgroup="g", covariates=["x"])
    assert r.model_info["diagnostics"][0]["check"] == "ddd_additive_covariates"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.ddd(df, y="y", treat="d", time="t", subgroup="g")


def test_window_restricts_covered_times_and_shrinks_critical_value(fits):
    full = sp.uniform_bands(fits["sa"], which="post")
    part = sp.uniform_bands(fits["sa"], which="post", window=(0, 1))
    assert part["relative_time"].tolist() == [0, 1]
    assert part.attrs["window"] == (0, 1)
    assert part.attrs["crit_uniform"] < full.attrs["crit_uniform"]


def test_event_study_plot_can_draw_the_simultaneous_band(fits):
    import matplotlib

    matplotlib.use("Agg")
    plain = sp.enhanced_event_study_plot(fits["sa"])[1]
    banded = sp.enhanced_event_study_plot(fits["sa"], uniform_band=True)[1]
    # One extra filled region per window (pre and post).
    assert len(banded.collections) == len(plain.collections) + 2


def test_event_study_plot_warns_when_no_band_is_available():
    import matplotlib

    matplotlib.use("Agg")
    es = pd.DataFrame(
        {
            "relative_time": [-1, 0, 1],
            "att": [0.0, 1.0, 1.2],
            "se": [0.0, 0.0, 0.0],
            "ci_lower": [0.0, 1.0, 1.2],
            "ci_upper": [0.0, 1.0, 1.2],
        }
    )
    res = sp.CausalResult(
        method="toy",
        estimand="ATT",
        estimate=1.0,
        se=0.0,
        pvalue=1.0,
        ci=(1.0, 1.0),
        alpha=0.05,
        n_obs=9,
        model_info={"event_study": es},
    )
    with pytest.warns(UserWarning, match="no simultaneous band"):
        sp.enhanced_event_study_plot(res, uniform_band=True)


# --------------------------------------------------------------------------
# terms=: a hand-written event study, and the sup-t test
# --------------------------------------------------------------------------


def _hand_event_study(seed=0, n=300, pretrend=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        treated = i < n // 3
        a = rng.normal()
        for t in range(1, 9):
            e = t - 5
            y = a + 0.1 * t + rng.normal(scale=0.5)
            if treated:
                y += pretrend * e + (1.0 if e >= 0 else 0.0)
            row = {"i": i, "t": t, "y": y, "tr": int(treated)}
            for k in (-4, -3, -2, 0, 1, 2, 3):
                name = f"lead{-k}" if k < 0 else f"lag{k}"
                row[name] = int(treated and e == k)
            rows.append(row)
    return pd.DataFrame(rows)


LEADS = ["lead4", "lead3", "lead2"]
LAGS = ["lag0", "lag1", "lag2", "lag3"]


def _hand_fit(df):
    rhs = " + ".join(LEADS + LAGS)
    return sp.regress(f"y ~ C(t) + tr + {rhs}", data=df, cluster="i")


def test_uniform_bands_terms_on_a_regression():
    fit = _hand_fit(_hand_event_study())
    band = sp.uniform_bands(fit, terms=LEADS)
    assert list(band["term"]) == LEADS
    np.testing.assert_allclose(band["estimate"], fit.params[LEADS], rtol=0, atol=0)
    np.testing.assert_allclose(band["se"], fit.std_errors[LEADS], rtol=1e-10)
    a = band.attrs
    assert a["crit_uniform"] > a["crit_pointwise"]
    # three coefficients: between the pointwise and the Bonferroni value
    from scipy import stats

    assert a["crit_uniform"] < stats.norm.ppf(1 - 0.05 / 6) + 0.02
    assert a["max_t"] == pytest.approx(
        float(np.max(np.abs(fit.params[LEADS] / fit.std_errors[LEADS])))
    )
    assert 0.0 <= a["supt_pvalue"] <= 1.0
    # the band rejects exactly when the sup-t p-value is below alpha
    outside = (band["cband_lower"] > 0) | (band["cband_upper"] < 0)
    assert bool(outside.any()) == (a["supt_pvalue"] < 0.05)


def test_supt_test_size_and_power():
    """Flat pre-trends: rejects about 5% of the time; a trend: nearly always."""
    rejections = 0
    reps = 120
    for seed in range(reps):
        band = sp.uniform_bands(
            _hand_fit(_hand_event_study(seed=seed, n=150)), terms=LEADS, n_draws=4000
        )
        rejections += band.attrs["supt_pvalue"] < 0.05
    # binomial(120, 0.05): 3 sd above the mean is 13
    assert rejections <= 13
    trending = _hand_fit(_hand_event_study(seed=1, pretrend=0.15))
    assert sp.uniform_bands(trending, terms=LEADS).attrs["supt_pvalue"] < 0.01


def test_supt_pvalue_on_an_event_study_result():
    df = sp.datasets.mpdta()
    fit = sp.sun_abraham(df, y="lemp", g="first_treat", t="year", i="countyreal")
    band = sp.uniform_bands(fit, which="pre")
    t_stats = np.abs(band["att"] / band["se"])
    assert band.attrs["max_t"] == pytest.approx(float(t_stats.max()))
    assert (band.attrs["supt_pvalue"] < 0.05) == bool(
        t_stats.max() > band.attrs["crit_uniform"]
    )


def test_uniform_bands_terms_refusals():
    fit = _hand_fit(_hand_event_study())
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not among"):
        sp.uniform_bands(fit, terms=["lead9"])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="do not apply"):
        sp.uniform_bands(fit, terms=LEADS, which="pre")
