"""Regression tests for the pass over Facure, *Causal Inference in Python*.

Every test here runs on simulated data. The same behaviour on the book's
own data is in ``tests/external_parity/test_facure_causal_inference_in_python.py``
(skipped without the data), and the findings are written up in
``docs/dev/2026-10-05-facure-causal-inference-in-python-review.md``.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

# --------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def obs():
    """Selection on observables with one three-level categorical confounder."""
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    g = rng.integers(0, 3, n)
    shift = np.array([0.0, 1.5, -1.0])[g]
    p = 1 / (1 + np.exp(-(0.5 * x + shift)))
    d = (rng.random(n) < p).astype(int)
    y = 1.0 * d + x + 2 * shift + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "d": d,
            "x": x,
            "g_num": g,
            "g_str": np.array(["a", "b", "c"])[g],
        }
    )


@pytest.fixture(scope="module")
def staggered():
    """Daily staggered panel with date-typed time and cohort columns."""
    rng = np.random.default_rng(1)
    days = pd.date_range("2021-05-01", periods=20, freq="D")
    rows = []
    for u in range(60):
        first = [5, 10, 0][u % 3]
        alpha = rng.normal()
        for k, day in enumerate(days, start=1):
            on = int(first != 0 and k >= first)
            rows.append(
                {
                    "unit": u,
                    "date": day,
                    "t": k,
                    "g": first,
                    "cohort": days[first - 1] if first else pd.Timestamp("2100-01-01"),
                    "y": alpha + 0.1 * k + 2.0 * on + rng.normal(scale=0.5),
                }
            )
    return pd.DataFrame(rows)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# --------------------------------------------------------------------- #
#  DAG recommender
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec, latent",
    [
        ("Z -> T; T -> Y; T <-> Y", None),
        ("U -> T; U -> Y; Z -> T; T -> Y", ["U"]),
    ],
)
def test_recommender_finds_the_instrument(spec, latent):
    rec = sp.dag_recommend_estimator(sp.dag(spec, latent=latent), "T", "Y")
    assert rec.estimator == "iv"
    assert rec.instrument == "Z"


def test_recommender_rejects_an_invalid_instrument():
    # Z affects Y directly, or shares a latent cause with it
    for spec in ("Z -> T; T -> Y; T <-> Y; Z -> Y", "Z -> T; T -> Y; T <-> Y; Z <-> Y"):
        rec = sp.dag_recommend_estimator(sp.dag(spec), "T", "Y")
        assert rec.estimator == "identify"


def test_recommender_conditional_instrument():
    g = sp.dag("W -> Z; W -> Y; Z -> T; T -> Y; T <-> Y")
    rec = sp.dag_recommend_estimator(g, "T", "Y")
    assert rec.estimator == "iv" and rec.adjustment_set == {"W"}
    assert "W + (T ~ Z)" in rec.sp_call


def test_front_door_needs_every_directed_path_intercepted():
    direct = sp.dag("T -> M; M -> Y; T -> Y; U -> T; U -> Y", latent=["U"])
    assert sp.dag_recommend_estimator(direct, "T", "Y").estimator == "identify"
    classic = sp.dag("T -> M; M -> Y; U -> T; U -> Y", latent=["U"])
    rec = sp.dag_recommend_estimator(classic, "T", "Y")
    assert rec.estimator == "front_door" and rec.mediators == ["M"]


def test_recommended_calls_run():
    rng = np.random.default_rng(2)
    n = 400
    Z, U = rng.normal(size=n), rng.normal(size=n)
    T = (Z + U + rng.normal(size=n) > 0).astype(int)
    M = (T + rng.normal(size=n) > 0.5).astype(int)
    Y = M + T + U + rng.normal(size=n)
    df = pd.DataFrame({"T": T, "Y": Y, "Z": Z, "M": M, "U": U})
    graphs = {
        "regress": sp.dag("Z -> T; Z -> Y; T -> Y"),
        "iv": sp.dag("Z -> T; T -> Y; T <-> Y"),
        "front_door": sp.dag("T -> M; M -> Y; T <-> Y"),
        "identify": sp.dag("T -> Y; T <-> Y"),
    }
    for name, g in graphs.items():
        rec = sp.dag_recommend_estimator(g, "T", "Y")
        assert rec.estimator == name
        call = rec.sp_call.split("#")[0].strip()
        if name == "front_door":
            call = call[:-1] + ", n_boot=5)"
        _quiet(eval, call, {"sp": sp, "df": df, "dag": g})
        for alt in rec.alternatives:
            if alt.startswith("sp.ipw(df"):
                _quiet(eval, alt[:-1] + ", n_bootstrap=5)", {"sp": sp, "df": df})


# --------------------------------------------------------------------- #
#  IV formula: the linearmodels block
# --------------------------------------------------------------------- #


def test_iv_accepts_square_brackets():
    rng = np.random.default_rng(3)
    n = 500
    z1, z2, x, u = (rng.normal(size=n) for _ in range(4))
    d = z1 + 0.5 * z2 + u + rng.normal(size=n)
    y = 2 * d + x + u + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2, "x": x})
    a = sp.iv("y ~ x + (d ~ z1 + z2)", data=df)
    b = sp.iv("y ~ 1 + [d ~ z1 + z2] + x", data=df)
    np.testing.assert_allclose(b.params["d"], a.params["d"], rtol=1e-12)
    np.testing.assert_allclose(b.std_errors["d"], a.std_errors["d"], rtol=1e-12)


# --------------------------------------------------------------------- #
#  DiD: repeated rows, 0/1 flags, dates
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def block_design():
    """Group flag x post flag, ten days before and ten after for each unit."""
    rng = np.random.default_rng(4)
    rows = []
    for u in range(40):
        treated = int(u < 12)
        alpha = rng.normal()
        for day in range(20):
            post = int(day >= 10)
            # early post days are below the later ones, so "first row per
            # cell" and "all rows per cell" give different answers
            rows.append(
                {
                    "unit": u,
                    "day": day,
                    "treated": treated,
                    "post": post,
                    "y": alpha
                    + 0.3 * post
                    + 1.0 * treated * post * (day - 9) / 5.5
                    + rng.normal(scale=0.3),
                }
            )
    return pd.DataFrame(rows)


def test_did_block_design_uses_every_row(block_design):
    res = _quiet(sp.did, block_design, y="y", treat="treated", time="post", id="unit")
    cell = block_design.groupby(["treated", "post"])["y"].mean()
    by_hand = (cell[1, 1] - cell[1, 0]) - (cell[0, 1] - cell[0, 0])
    assert res.estimate == pytest.approx(by_hand, rel=1e-10)
    assert "2x2" in res.method
    explicit = _quiet(
        sp.did, block_design, y="y", treat="treated", time="post", cluster="unit"
    )
    assert res.se == pytest.approx(explicit.se, rel=1e-12)


def test_callaway_santanna_refuses_repeated_unit_period_rows(block_design):
    with pytest.raises(MethodIncompatibility, match="one row per unit and period"):
        _quiet(
            sp.callaway_santanna, block_design, y="y", g="treated", t="post", i="unit"
        )


def test_did_warns_when_a_group_flag_is_read_as_a_cohort(block_design):
    with pytest.warns(sp.exceptions.AssumptionWarning, match="read as the cohort"):
        try:
            sp.did(block_design, y="y", treat="treated", time="day", id="unit")
        except sp.exceptions.StatsPAIError:
            pass


@pytest.mark.parametrize(
    "fn, numeric, dated",
    [
        (
            "callaway_santanna",
            dict(g="g", t="t", i="unit"),
            dict(g="cohort", t="date", i="unit"),
        ),
        (
            "sun_abraham",
            dict(g="g", t="t", i="unit"),
            dict(g="cohort", t="date", i="unit"),
        ),
        (
            "did_imputation",
            dict(group="unit", time="t", first_treat="g"),
            dict(group="unit", time="date", first_treat="cohort"),
        ),
        (
            "etwfe",
            dict(group="unit", time="t", first_treat="g"),
            dict(group="unit", time="date", first_treat="cohort"),
        ),
    ],
)
def test_staggered_estimators_take_dates(staggered, fn, numeric, dated):
    f = getattr(sp, fn)
    a = _quiet(f, staggered, y="y", **numeric)
    b = _quiet(f, staggered, y="y", **dated)
    assert b.estimate == pytest.approx(a.estimate, rel=1e-12)
    assert b.se == pytest.approx(a.se, rel=1e-12)
    cal = b.model_info["calendar_time"]
    assert cal["periods"][0] == pd.Timestamp("2021-05-01") and cal["regular"]
    assert "calendar_time" not in a.model_info


def test_index_calendar_time_rules(staggered):
    from statspai.did._core import index_calendar_time

    untouched, info = index_calendar_time(staggered, "t", "g", function="f")
    assert untouched is staggered and info is None

    nat = staggered.assign(cohort=staggered["cohort"].where(staggered["g"] != 0))
    for frame in (staggered, nat):
        out, info = index_calendar_time(frame, "date", "cohort", function="f")
        np.testing.assert_array_equal(out["date"], staggered["t"])
        np.testing.assert_array_equal(out["cohort"], staggered["g"])

    with pytest.raises(MethodIncompatibility, match="holds dates"):
        index_calendar_time(staggered, "date", "g", function="f")
    with pytest.raises(MethodIncompatibility, match="holds dates"):
        index_calendar_time(staggered, "t", "cohort", function="f")

    gaps = staggered[staggered["t"] % 7 != 3]
    with pytest.warns(UserWarning, match="not evenly spaced"):
        _, info = index_calendar_time(gaps, "date", "cohort", function="f")
    assert not info["regular"]


# --------------------------------------------------------------------- #
#  Treatments that are not 0/1
# --------------------------------------------------------------------- #


def test_two_group_tables_refuse_other_codings(obs):
    arms = obs.assign(arm=np.array(["ctl", "a", "b"])[obs["g_num"]])
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.balance_table(arms, treat="arm", covariates=["x"])
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.balance_check(arms, treatment="arm", covariates=["x"])
    with pytest.raises(MethodIncompatibility, match="not found"):
        sp.balance_table(obs, treat="d", covariates=["x", "nope"])
    ok = sp.balance_table(
        obs.assign(d=obs["d"].astype(bool)),
        treat="d",
        covariates=["x"],
        output="dataframe",
    )
    assert len(ok) >= 1


def test_cate_eval_refuses_a_dose():
    rng = np.random.default_rng(5)
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.cate_eval(
            rng.normal(size=50),
            rng.normal(size=50),
            rng.uniform(0, 10, 50),
            X=rng.normal(size=(50, 2)),
        )


# --------------------------------------------------------------------- #
#  Synthetic control
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sc_panel():
    rng = np.random.default_rng(11)
    T0, T1, J = 40, 15, 12
    F = rng.normal(size=(T0 + T1, 2))
    L = rng.uniform(0, 1, (J + 1, 2))
    Y = F @ L.T + rng.normal(0, 0.3, (T0 + T1, J + 1))
    Y[T0:, 0] += 1.0
    long = pd.DataFrame(
        [(t, j, Y[t, j]) for t in range(T0 + T1) for j in range(J + 1)],
        columns=["t", "u", "y"],
    )
    return long, T0


# scinference::scinference(Y[,1], Y[,-1], T1=15, T0=40,
#   inference_method="ttest", K=K, alpha=0.1) on the matrix of `sc_panel`,
# scinference 0.0.0.9000, R 4.x, run 2026-10-05: att, se, lb, ub.
_SCINFERENCE = {
    2: (0.965953352551677, 0.0591155296587198, 0.592712587628118, 1.33919411747524),
    3: (0.976697468778584, 0.166004271363532, 0.491967390119944, 1.46142754743722),
    4: (0.998509556707641, 0.0856750097647046, 0.796885121451096, 1.20013399196419),
}


@pytest.mark.parametrize("k", [2, 3, 4])
def test_synth_ttest_matches_scinference(sc_panel, k):
    long, T0 = sc_panel
    res = sp.synth(
        long,
        outcome="y",
        unit="u",
        time="t",
        treated_unit=0,
        treatment_time=T0,
        inference="ttest",
        n_folds=k,
        alpha=0.1,
    )
    att, se, lb, ub = _SCINFERENCE[k]
    # Same simplex least-squares problem in both; the solvers agree to 1e-9.
    assert res.estimate == pytest.approx(att, rel=1e-8)
    assert res.se == pytest.approx(se, rel=1e-7)
    assert res.ci[0] == pytest.approx(lb, rel=1e-7)
    assert res.ci[1] == pytest.approx(ub, rel=1e-7)
    assert res.model_info["df"] == k - 1
    direct = sp.synth_ttest(
        long,
        outcome="y",
        unit="u",
        time="t",
        treated_unit=0,
        treatment_time=T0,
        n_folds=k,
        alpha=0.1,
    )
    assert direct.estimate == res.estimate


def test_synth_inference_and_treated_unit_are_validated(sc_panel):
    long, T0 = sc_panel
    base = dict(outcome="y", unit="u", time="t", treatment_time=T0)
    with pytest.raises(MethodIncompatibility, match="inference='jackknife'"):
        sp.synth(long, treated_unit=0, inference="jackknife", **base)
    with pytest.raises(MethodIncompatibility, match="inference='nonsense'"):
        sp.synth(long, treated_unit=0, inference="nonsense", **base)
    with pytest.raises(MethodIncompatibility, match="one treated unit"):
        sp.synth(long, treated_unit=[0, 1], **base)
    one = _quiet(sp.synth, long, treated_unit=[0], placebo=False, **base)
    ref = _quiet(sp.synth, long, treated_unit=0, placebo=False, **base)
    assert one.estimate == ref.estimate


def test_design_result_repr_is_short(sc_panel):
    long, T0 = sc_panel
    res = sp.synth_experimental_design(
        long[long["t"] < T0], unit="u", time="t", outcome="y", k=2, random_state=0
    )
    assert len(repr(res)) < 300


# --------------------------------------------------------------------- #
#  Power: sigma
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "design, kwargs",
    [
        ("rct", {}),
        ("did", dict(n_periods=4, n_treated_periods=2)),
        ("rd", {}),
        ("iv", dict(first_stage_f=20)),
        ("cluster_rct", dict(cluster_size=10, icc=0.1)),
        ("ols", {}),
    ],
)
def test_power_depends_on_effect_over_sigma(design, kwargs):
    std = sp.power(design, n=400, effect_size=0.2, **kwargs).power
    raw = sp.power(design, n=400, effect_size=0.5, sigma=2.5, **kwargs).power
    assert raw == pytest.approx(std, rel=1e-12)
    bigger_sd = sp.power(design, n=400, effect_size=0.2, sigma=2.0, **kwargs).power
    assert bigger_sd < std


def test_sample_size_for_a_raw_effect():
    # two-sided 5%, 80% power: n_total = 4 (z_.975 + z_.8)^2 sigma^2 / delta^2
    from scipy.stats import norm

    sigma, delta = 0.2, 0.08
    exact = 4 * (norm.ppf(0.975) + norm.ppf(0.8)) ** 2 * sigma**2 / delta**2
    res = sp.power("rct", effect_size=delta, power_target=0.8, sigma=sigma)
    assert abs(res.n - exact) <= 1.0


# --------------------------------------------------------------------- #
#  Categorical covariates
# --------------------------------------------------------------------- #


def test_expand_categorical_covariates_unit():
    from statspai.core._covariates import expand_categorical_covariates as expand

    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "s": ["b", "a", "c", "a", None, "b"],
            "k": pd.Categorical([3, 1, 2, 1, 2, 3]),
            "flag": [True, False, True, False, True, False],
            "code": [3, 1, 2, 1, 2, 3],
        }
    )
    same, covs, info = expand(df, ["x", "code"], function="f")
    assert same is df and covs == ["x", "code"] and info is None

    out, covs, info = expand(df, ["x", "s", "k", "flag", "C(code)"], function="f")
    assert covs == [
        "x",
        "s[T.b]",
        "s[T.c]",
        "k[T.2]",
        "k[T.3]",
        "flag",
        "code[T.2]",
        "code[T.3]",
    ]
    assert info["levels"]["s"] == ["a", "b", "c"]
    assert out["s[T.b]"].tolist()[:4] == [1.0, 0.0, 0.0, 0.0]
    assert np.isnan(out["s[T.b]"].iloc[4]) and np.isnan(out["s[T.c]"].iloc[4])
    assert out["flag"].dtype == float
    assert "s[T.b]" not in df.columns

    ids = pd.DataFrame({"id": [f"u{i}" for i in range(10)]})
    with pytest.raises(MethodIncompatibility, match="identifier"):
        expand(ids, ["id"], function="f")
    with pytest.raises(MethodIncompatibility, match="not in the data"):
        expand(df, ["C(missing)"], function="f")


@pytest.mark.parametrize(
    "fn, kwargs",
    [
        ("ipw", dict(se_method="sandwich")),
        ("aipw", dict(cross_fit=False)),
        ("ebalance", {}),
        ("g_computation", dict(n_boot=5, seed=0)),
    ],
)
def test_estimators_expand_categorical_covariates(obs, fn, kwargs):
    f = getattr(sp, fn)
    dummies = pd.get_dummies(obs, columns=["g_str"], drop_first=True, dtype=float)
    ref = _quiet(
        f, dummies, y="y", treat="d", covariates=["x", "g_str_b", "g_str_c"], **kwargs
    )
    as_number = _quiet(f, obs, y="y", treat="d", covariates=["x", "g_num"], **kwargs)
    assert abs(as_number.estimate - ref.estimate) > 1e-4
    variants = [
        (obs, ["x", "g_str"]),
        (obs, ["x", "C(g_num)"]),
        (obs.astype({"g_num": "category"}), ["x", "g_num"]),
    ]
    for frame, covs in variants:
        res = _quiet(f, frame, y="y", treat="d", covariates=covs, **kwargs)
        assert res.estimate == pytest.approx(ref.estimate, rel=1e-9)
        assert res.model_info["covariate_expansion"]["levels"]
    assert "covariate_expansion" not in as_number.model_info


def test_predict_cate_with_a_categorical_covariate(obs):
    from sklearn.linear_model import LinearRegression

    res = _quiet(
        sp.metalearner,
        obs,
        y="y",
        treat="d",
        covariates=["x", "g_str"],
        learner="t",
        outcome_model=LinearRegression(),
    )
    new = pd.DataFrame({"x": [0.0, 0.0], "g_str": ["c", "a"]})
    assert sp.predict_cate(res, new).shape == (2,)
    with pytest.raises(MethodIncompatibility, match="not in the fitted data"):
        sp.predict_cate(res, pd.DataFrame({"x": [0.0], "g_str": ["z"]}))


# --------------------------------------------------------------------- #
#  R-learner with a continuous treatment
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def dose():
    rng = np.random.default_rng(123)
    n = 1500
    x_h = rng.integers(1, 4, n).astype(float)
    x_c = rng.uniform(-1, 1, n)
    t = rng.normal(10 + x_c + 3 * x_c**2, 0.5)
    y = rng.normal((1 + x_h) * t - 5 * x_c, 0.5)
    folds = np.arange(n) % 4
    return pd.DataFrame({"x_h": x_h, "x_c": x_c, "t": t, "y": y}), folds


def _dose_models():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor

    def nuisance():
        return GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=0)

    return nuisance(), nuisance(), DecisionTreeRegressor(max_depth=3, random_state=0)


def test_r_learner_continuous_treatment(dose):
    df, folds = dose
    my, mt, mf = _dose_models()
    res = sp.metalearner(
        df,
        y="y",
        treat="t",
        covariates=["x_c", "x_h"],
        learner="r",
        outcome_model=my,
        propensity_model=mt,
        cate_model=mf,
        n_folds=4,
        fold_indices=folds,
    )
    assert res.estimand == "APE"
    assert res.model_info["treatment_type"] == "continuous"
    cate = res.model_info["cate"]
    for level in (1.0, 2.0, 3.0):  # true effect of one more unit: 1 + x_h
        assert cate[df["x_h"] == level].mean() == pytest.approx(1 + level, abs=0.15)
    # The headline is the partially linear coefficient on the same residuals.
    plr = sp.dml(
        df,
        y="y",
        treat="t",
        covariates=["x_c", "x_h"],
        model="plr",
        model_y=my,
        model_d=mt,
        fold_indices=folds,
        n_folds=4,
    )
    assert res.estimate == pytest.approx(plr.estimate, rel=1e-10)
    assert res.se == pytest.approx(plr.se, rel=1e-3)


def test_r_learner_continuous_matches_econml(dose):
    econml_dml = pytest.importorskip("econml.dml")
    df, folds = dose
    my, mt, mf = _dose_models()
    X = df[["x_c", "x_h"]].to_numpy()
    res = sp.metalearner(
        df,
        y="y",
        treat="t",
        covariates=["x_c", "x_h"],
        learner="r",
        outcome_model=my,
        propensity_model=mt,
        cate_model=mf,
        n_folds=4,
        fold_indices=folds,
    )
    splits = [
        (np.flatnonzero(folds != k), np.flatnonzero(folds == k)) for k in range(4)
    ]
    ref = econml_dml.NonParamDML(
        model_y=my, model_t=mt, model_final=mf, cv=splits, random_state=0
    )
    ref.fit(df["y"].to_numpy(), df["t"].to_numpy(), X=X)
    # Same folds, same learners, same weighted final stage: the same numbers.
    np.testing.assert_allclose(res.model_info["cate"], ref.effect(X), rtol=1e-10)


def test_continuous_treatment_is_refused_where_it_has_no_meaning(dose):
    from sklearn.linear_model import LogisticRegression

    df, _ = dose
    for learner in ("s", "t", "x", "dr"):
        with pytest.raises(ValueError, match="learner='r' only"):
            sp.metalearner(df, y="y", treat="t", covariates=["x_c"], learner=learner)
    with pytest.raises(MethodIncompatibility, match="needs a regressor"):
        sp.metalearner(
            df,
            y="y",
            treat="t",
            covariates=["x_c"],
            learner="r",
            propensity_model=LogisticRegression(),
        )


# --------------------------------------------------------------------- #
#  Switchback experiments
# --------------------------------------------------------------------- #


def test_switchback_optimal_points():
    plan = sp.switchback_design(12, m=2, seed=0)
    assert plan.loc[plan["randomize"], "period"].tolist() == [1, 5, 7, 9]
    assert sp.switchback_design(15, m=3)["randomize"].astype(int).tolist() == [
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
    ]
    assert sp.switchback_design(7, m=0)["randomize"].all()
    assert sp.switchback_design(9, m=2, design=3)["randomize"].sum() == 3
    with pytest.raises(MethodIncompatibility, match="multiple of m"):
        sp.switchback_design(13, m=2)
    # the path only changes where the coin is flipped
    d = plan["treat"].to_numpy()
    assert not np.any(np.diff(d)[~plan["randomize"].to_numpy()[1:]])


def test_switchback_exact_moments_under_the_optimal_design():
    """Enumerate all 16 assignment paths of T = 12, m = 2.

    The Horvitz-Thompson estimator is unbiased for the lag-2 effect, its
    variance is equation (10) of Bojinov, Simchi-Levi and Zhao (2023), and
    the estimator of Corollary 1 is above it on average.
    """
    T, m = 12, 2
    rng = np.random.default_rng(3)
    base = rng.normal(5, 1, T)
    delta = np.array([1.5, 1.0, 0.5])

    def outcome(path):
        lag = [
            sum(delta[j] * (path[t - j] if t - j >= 0 else 0) for j in range(3))
            for t in range(T)
        ]
        return base + np.array(lag)

    points = sp.switchback_design(T, m=m)["randomize"].to_numpy()
    estimates, variances = [], []
    for coins in itertools.product([0, 1], repeat=int(points.sum())):
        path = np.array(coins)[np.cumsum(points) - 1]
        frame = pd.DataFrame({"y": outcome(path), "d": path})
        res = sp.switchback(frame, y="y", treat="d", m=m, n_draws=0)
        estimates.append(res.estimate)
        variances.append(res.se**2)
    assert np.mean(estimates) == pytest.approx(delta.sum(), rel=1e-12)

    n = T // m
    Y1 = outcome(np.ones(T, int)).reshape(n, m).sum(1)[1:]
    Y0 = outcome(np.zeros(T, int)).reshape(n, m).sum(1)[1:]
    lemma2 = (
        (Y1[0] + Y0[0]) ** 2
        + sum(
            3 * Y1[k] ** 2 + 3 * Y0[k] ** 2 + 2 * Y1[k] * Y0[k] for k in range(1, n - 2)
        )
        + (Y1[n - 2] + Y0[n - 2]) ** 2
        + sum(2 * (Y1[k] + Y0[k]) * (Y1[k + 1] + Y0[k + 1]) for k in range(n - 2))
    ) / (T - m) ** 2
    assert np.var(estimates) == pytest.approx(lemma2, rel=1e-12)
    assert np.mean(variances) >= lemma2


def test_switchback_inference_and_validation():
    plan = sp.switchback_design(120, m=2, seed=1)
    rng = np.random.default_rng(1)
    d = plan["treat"].to_numpy()
    lagged = d + np.r_[0, d[:-1]] + np.r_[0, 0, d[:-2]]
    plan["y"] = 10 + 1.0 * lagged + rng.normal(size=120)
    res = sp.switchback(plan, y="y", treat="treat", m=2, design="randomize", seed=1)
    assert res.model_info["optimal_design"] and res.se > 0
    assert 0.0 <= res.model_info["randomization_pvalue"] <= 1.0
    assert res.ci[0] < res.estimate < res.ci[1]
    shuffled = plan.sample(frac=1, random_state=0)
    again = sp.switchback(
        shuffled, y="y", treat="treat", m=2, design="randomize", time="period", seed=1
    )
    assert again.estimate == res.estimate

    every = sp.switchback(
        plan, y="y", treat="treat", m=2, design=np.ones(120, bool), seed=1
    )
    assert np.isnan(every.se)
    assert every.pvalue == every.model_info["randomization_pvalue"]

    wrong = plan.assign(treat=rng.integers(0, 2, 120))
    with pytest.raises(MethodIncompatibility, match="not randomization points"):
        sp.switchback(wrong, y="y", treat="treat", m=2, design="randomize")
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.switchback(plan.assign(treat=plan["treat"] * 2), y="y", treat="treat", m=2)


# --------------------------------------------------------------------- #
#  Gain curve
# --------------------------------------------------------------------- #


def test_cate_gain_curve():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 1, n)
    dose = rng.uniform(0, 10, n)
    df = pd.DataFrame(
        {"x": x, "dose": dose, "y": (1 + 2 * x) * dose + rng.normal(size=n)}
    )
    good = sp.cate_gain_curve(df, cate="x", y="y", treat="dose")
    bad = sp.cate_gain_curve(df, cate=-x, y="y", treat="dose")
    noise = sp.cate_gain_curve(df, cate=rng.normal(size=n), y="y", treat="dose")
    assert good.auc > 5 * abs(noise.auc) and bad.auc < -5 * abs(noise.auc)
    assert good.ate == pytest.approx(2.0, abs=0.1)
    assert good.by_quantile["effect"].is_monotonic_increasing
    assert good.curve["cumulative_gain"].iloc[-1] == pytest.approx(0.0, abs=1e-12)
    raw = sp.cate_gain_curve(df, cate="x", y="y", treat="dose", normalize=False)
    assert raw.curve["cumulative_gain"].iloc[-1] == pytest.approx(raw.ate, rel=1e-12)

    binary = df.assign(d=(rng.random(n) < 0.5).astype(int))
    binary["y"] = (1 + 2 * x) * binary["d"] + rng.normal(size=n)
    res = sp.cate_gain_curve(binary, cate="x", y="y", treat="d")
    diff = (
        binary.loc[binary["d"] == 1, "y"].mean()
        - binary.loc[binary["d"] == 0, "y"].mean()
    )
    assert res.ate == pytest.approx(diff, rel=1e-12)
    with pytest.raises(MethodIncompatibility, match="values for"):
        sp.cate_gain_curve(df, cate=np.ones(3), y="y", treat="dose")


# --------------------------------------------------------------------- #
#  margins
# --------------------------------------------------------------------- #


def test_margins_says_to_pass_data_for_factor_terms(obs):
    fit = sp.regress("y ~ x * C(g_num)", data=obs)
    with pytest.raises(MethodIncompatibility) as err:
        sp.margins(fit, variables=["x"])
    assert "pass data=" in str(getattr(err.value, "recovery_hint", "")) + str(err.value)
    assert len(sp.margins(fit, data=obs, variables=["x"])) == 1


# --------------------------------------------------------------------- #
#  Second round: the remaining estimators, bootstrap, population design
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fn, numeric, dated",
    [
        (
            "gardner_did",
            dict(group="unit", time="t", first_treat="g"),
            dict(group="unit", time="date", first_treat="cohort"),
        ),
        (
            "stacked_did",
            dict(group="unit", time="t", first_treat="g", window=(-3, 3)),
            dict(group="unit", time="date", first_treat="cohort", window=(-3, 3)),
        ),
        (
            "wooldridge_did",
            dict(group="unit", time="t", first_treat="g"),
            dict(group="unit", time="date", first_treat="cohort"),
        ),
        (
            "twfe_decomposition",
            dict(group="unit", time="t", first_treat="g"),
            dict(group="unit", time="date", first_treat="cohort"),
        ),
    ],
)
def test_more_staggered_estimators_take_dates(staggered, fn, numeric, dated):
    # gardner_did returned 3.13 for 1.93 on a date-typed panel before this.
    f = getattr(sp, fn)
    a = _quiet(f, staggered, y="y", **numeric)
    b = _quiet(f, staggered, y="y", **dated)
    assert b.estimate == pytest.approx(a.estimate, rel=1e-12)
    assert b.se == pytest.approx(a.se, rel=1e-12)


def test_bacon_decomposition_takes_dates(staggered):
    d = staggered.assign(
        d=((staggered["g"] > 0) & (staggered["t"] >= staggered["g"])).astype(int)
    )
    a = _quiet(sp.bacon_decomposition, d, y="y", treat="d", time="t", id="unit")
    b = _quiet(sp.bacon_decomposition, d, y="y", treat="d", time="date", id="unit")
    assert b["beta_twfe"] == pytest.approx(a["beta_twfe"], rel=1e-12)


@pytest.mark.parametrize(
    "fn, kwargs, key",
    [
        ("gardner_did", dict(group="unit", time="t", first_treat="g"), "controls"),
        ("did_imputation", dict(group="unit", time="t", first_treat="g"), "controls"),
        ("etwfe", dict(group="unit", time="t", first_treat="g"), "controls"),
        ("event_study", dict(treat_time="g", time="t", unit="unit"), "covariates"),
    ],
)
def test_did_controls_expand_text_columns(staggered, fn, kwargs, key):
    d = staggered.assign(
        shift=np.where((staggered["unit"] + staggered["t"]) % 2 == 0, "day", "night")
    )
    d["shift_night"] = (d["shift"] == "night").astype(float)
    f = getattr(sp, fn)
    a = _quiet(f, d, y="y", **kwargs, **{key: ["shift_night"]})
    b = _quiet(f, d, y="y", **kwargs, **{key: ["shift"]})
    assert b.estimate == pytest.approx(a.estimate, rel=1e-10)


def test_matching_family_expands_text_columns(obs):
    dummies = pd.get_dummies(obs, columns=["g_str"], drop_first=True, dtype=float)
    ref = _quiet(
        sp.psmatch2,
        dummies,
        treat="d",
        outcome="y",
        covariates=["x", "g_str_b", "g_str_c"],
    )
    res = _quiet(sp.psmatch2, obs, treat="d", outcome="y", covariates=["x", "g_str"])
    assert res.att == pytest.approx(ref.att, rel=1e-10)
    tree = _quiet(
        sp.policy_tree, obs, y="y", treat="d", covariates=["x", "g_str"], depth=1
    )
    assert tree is not None


def test_cate_gain_curve_bootstrap():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.uniform(0, 1, n)
    dose = rng.uniform(0, 10, n)
    df = pd.DataFrame(
        {
            "x": x,
            "dose": dose,
            "store": np.arange(n) % 60,
            "y": (1 + 2 * x) * dose + rng.normal(size=n),
        }
    )
    plain = sp.cate_gain_curve(df, cate="x", y="y", treat="dose")
    assert plain.auc_se is None and plain.auc_ci is None
    res = sp.cate_gain_curve(df, cate="x", y="y", treat="dose", n_boot=200, seed=1)
    assert res.auc == plain.auc and res.auc_se > 0
    assert res.auc_ci[0] < res.auc < res.auc_ci[1] and res.auc_ci[0] > 0
    again = sp.cate_gain_curve(df, cate="x", y="y", treat="dose", n_boot=200, seed=1)
    assert again.auc_ci == res.auc_ci
    clustered = sp.cate_gain_curve(
        df, cate="x", y="y", treat="dose", n_boot=50, seed=1, cluster="store"
    )
    assert clustered.auc_se > 0
    # A ranking with no information: the interval covers zero in most draws.
    covered = 0
    for s in range(40):
        noise = np.random.default_rng(100 + s).normal(size=n)
        r = sp.cate_gain_curve(df, cate=noise, y="y", treat="dose", n_boot=80, seed=s)
        covered += r.auc_ci[0] <= 0.0 <= r.auc_ci[1]
    assert covered >= 34  # nominal 38 of 40; binomial slack


def test_population_design_recovers_a_representative_set():
    """Three latent markets; the population average loads on all three.

    A treated set that tracks the average needs a unit from each market,
    and the remaining units must still cover all three.
    """
    rng = np.random.default_rng(7)
    T, per = 40, 6
    factors = rng.normal(size=(T, 3)).cumsum(axis=0)
    rows = []
    for j in range(3 * per):
        series = factors[:, j // per] + rng.normal(scale=0.05, size=T)
        rows += [
            {"unit": f"u{j:02d}", "t": t, "y": series[t], "pop": 1.0 + j // per}
            for t in range(T)
        ]
    df = pd.DataFrame(rows)
    res = sp.synth_experimental_design(
        df,
        unit="unit",
        time="t",
        outcome="y",
        k=3,
        criterion="population",
        n_search=400,
        random_state=0,
    )
    markets = {int(u[1:]) // per for u in res.selected}
    assert markets == {0, 1, 2}
    assert res.method == "population_matching_search"
    assert res.expected_variance < 0.05 * res.baseline_variance
    assert res.weights["treated"].sum() == pytest.approx(1.0, abs=1e-6)
    assert res.weights["control"].sum() == pytest.approx(1.0, abs=1e-6)
    assert not np.any((res.weights["treated"] > 0) & (res.weights["control"] > 0))
    assert "population average" in res.summary()

    weighted = sp.synth_experimental_design(
        df,
        unit="unit",
        time="t",
        outcome="y",
        k=3,
        criterion="population",
        population_weights="pop",
        n_search=50,
        random_state=0,
    )
    shares = weighted.ranking.groupby("unit")["population_share"].first()
    assert shares["u17"] == pytest.approx(3 * shares["u00"])

    default = sp.synth_experimental_design(
        df, unit="unit", time="t", outcome="y", k=3, random_state=0
    )
    assert default.method == "loo_sc_fit_ranking"
    with pytest.raises(MethodIncompatibility, match="criterion"):
        sp.synth_experimental_design(
            df, unit="unit", time="t", outcome="y", k=3, criterion="best"
        )
    with pytest.raises(MethodIncompatibility, match="criterion='population' only"):
        sp.synth_experimental_design(
            df, unit="unit", time="t", outcome="y", k=3, population_weights="pop"
        )
