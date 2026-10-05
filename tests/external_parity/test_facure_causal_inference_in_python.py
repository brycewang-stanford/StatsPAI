"""Matheus Facure, *Causal Inference in Python* (O'Reilly, 2023), by chapter.

The book's notebooks (https://github.com/matheusfacure/causal-inference-in-python-code,
MIT licence) write most estimators out by hand with pandas, statsmodels,
scikit-learn and cvxpy, and store their printed results. Here the numbers
are recomputed with the ``sp.*`` function a user would reach for and
compared with the stored outputs, quoted in the tests.

The data are not redistributed. Point ``STATSPAI_FACURE_DIR`` at the
notebooks' ``data`` folder:

    STATSPAI_FACURE_DIR=/path/to/causal-inference-in-python/data \\
        pytest tests/external_parity/test_facure_causal_inference_in_python.py

It is skipped otherwise. What the pass found is in
``docs/dev/2026-10-05-facure-causal-inference-in-python-review.md``.

Tolerances. Closed forms (OLS, 2SLS, Horvitz-Thompson, matching) are held
to 1e-9 relative. Anything through a logit to 1e-5: the book fits it with
scikit-learn's lbfgs or statsmodels at default tolerances. The synthetic
control numbers of the book come from cvxpy at its default solver accuracy
(its weights show entries of -8e-6), so those are compared at 1e-2 and the
tight comparison is against a constrained solver run to 1e-16 in the test.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ROOT = os.environ.get("STATSPAI_FACURE_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "management_training.csv").is_file(),
    reason="STATSPAI_FACURE_DIR does not point at the book's data folder",
)


def _read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(Path(ROOT) / name, **kw)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# --------------------------------------------------------------------- #
#  Chapter 2: A/B test
# --------------------------------------------------------------------- #


def test_ch02_difference_in_conversion():
    data = _read("cross_sell_email.csv")
    two = data[data["cross_sell_email"] != "long"]
    res = sp.ttest(two, "conversion", by="cross_sell_email", unequal=True)
    text = res.summary()
    # book: diff 0.0824468 (short - no_email), t = 2.2379512318715364
    assert "0.0824468" in text
    assert "2.2380" in text


def test_ch02_balance_refuses_three_arms():
    data = _read("cross_sell_email.csv")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="0/1"):
        sp.balance_table(data, treat="cross_sell_email", covariates=["gender", "age"])


def test_ch02_sample_size_with_raw_effect_and_sigma():
    data = _read("cross_sell_email.csv")
    sd = data.loc[data["cross_sell_email"] == "no_email", "conversion"].std()
    res = sp.power("rct", effect_size=0.08, power_target=0.8, sigma=sd)
    # book rule of thumb: 16 sigma^2 / delta^2 = 103 per arm
    assert abs(res.n / 2 - 103) <= 2


# --------------------------------------------------------------------- #
#  Chapter 4: regression
# --------------------------------------------------------------------- #


def test_ch04_saturated_model_average_slope():
    rnd = _read("risk_data_rnd.csv")
    fit = sp.regress("default ~ credit_limit * C(credit_score1_buckets)", data=rnd)
    me = sp.margins(fit, data=rnd, variables=["credit_limit"])
    # book: size-weighted average of the group slopes
    assert me["dy/dx"].iloc[0] == pytest.approx(4.490445628748722e-06, rel=1e-8)


# --------------------------------------------------------------------- #
#  Chapter 5: propensity score
# --------------------------------------------------------------------- #

_CONT = ["tenure", "last_engagement_score", "department_score"]
_CAT = ["n_of_reports", "gender", "role"]


def _training() -> pd.DataFrame:
    return _read("management_training.csv")


def test_ch05_ipw_with_categorical_covariates():
    df = _training()
    covs = _CONT + [f"C({c})" for c in _CAT]
    res = _quiet(
        sp.ipw,
        df,
        y="engagement_score",
        treat="intervention",
        covariates=covs,
        normalize=False,
        n_bootstrap=20,
        seed=0,
    )
    # book: 0.2659787088076121 (Horvitz-Thompson, logit with C() terms)
    assert res.estimate == pytest.approx(0.2659787088076121, rel=1e-5)
    assert set(res.model_info["covariate_expansion"]["levels"]) == set(_CAT)


def test_ch05_category_dtype_is_not_read_as_a_number():
    df = _training()
    as_cat = df.astype({c: "category" for c in _CAT})
    a = _quiet(
        sp.aipw,
        as_cat,
        y="engagement_score",
        treat="intervention",
        covariates=_CONT + _CAT,
        cross_fit=False,
    )
    # book: 0.27115831057931455 (sklearn logit); the same model with exact
    # maximum likelihood gives 0.2711626
    assert a.estimate == pytest.approx(0.27115831057931455, rel=1e-4)
    numeric = _quiet(
        sp.aipw,
        df,
        y="engagement_score",
        treat="intervention",
        covariates=_CONT + _CAT,
        cross_fit=False,
    )
    assert abs(numeric.estimate - a.estimate) > 1e-3


def test_ch05_propensity_score_matching_ate():
    df = pd.get_dummies(_training(), columns=_CAT, drop_first=True, dtype=float)
    covs = _CONT + [c for c in df.columns if any(c.startswith(k + "_") for k in _CAT)]
    res = _quiet(
        sp.match,
        df,
        y="engagement_score",
        treat="intervention",
        covariates=covs,
        method="psm",
        estimand="ATE",
    )
    # book: 0.28777443474045966 (1-NN on the propensity score, both arms)
    assert res.estimate == pytest.approx(0.28777443474045966, rel=1e-6)


# --------------------------------------------------------------------- #
#  Chapter 6: evaluating a CATE ranking with a continuous treatment
# --------------------------------------------------------------------- #


def test_ch06_effect_by_quantile_and_gain():
    import statsmodels.formula.api as smf

    data = _read("daily_restaurant_sales.csv")
    train = data.query("day<'2018-01-01'")
    test = data.query("day>='2018-01-01'")
    f = "sales ~ discounts*(C(month)+C(weekday)+is_holiday+competitors_price)"
    m = smf.ols(f, data=train).fit()
    pred = (m.predict(test.assign(discounts=test["discounts"] + 1)) - m.predict(test))
    res = sp.cate_gain_curve(
        test.assign(cate_pred=pred.to_numpy()),
        cate="cate_pred",
        y="sales",
        treat="discounts",
    )
    assert res.ate == pytest.approx(32.16196368039615, rel=1e-10)
    book = [20.494153, 24.782101, 27.494156, 28.833993, 29.604257]
    book += [32.216500, 35.889459, 36.846889, 39.125449, 44.272549]
    np.testing.assert_allclose(res.by_quantile["effect"], book, rtol=1e-6)
    # book: 181.7457, with one extra unit in every step of its curve
    assert res.auc == pytest.approx(181.7457, rel=2e-3)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="0/1"):
        sp.cate_eval(pred.to_numpy(), test["sales"], test["discounts"])


# --------------------------------------------------------------------- #
#  Chapter 8: difference-in-differences
# --------------------------------------------------------------------- #


def _south() -> pd.DataFrame:
    return _read("short_offline_mkt_south.csv").astype({"date": "datetime64[ns]"})


def test_ch08_block_design_with_a_post_flag():
    mkt = _south()
    res = _quiet(sp.did, mkt, y="downloads", treat="treated", time="post", id="city")
    # book: 0.6917359536407233; before the fix this call returned -0.659
    assert res.estimate == pytest.approx(0.6917359536407233, rel=1e-10)
    assert "2x2" in res.method
    clustered = _quiet(
        sp.did, mkt, y="downloads", treat="treated", time="post", cluster="city"
    )
    assert res.se == pytest.approx(clustered.se, rel=1e-12)


def test_ch08_repeated_unit_period_rows_are_refused():
    mkt = _south()
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="one row per unit"):
        _quiet(sp.callaway_santanna, mkt, y="downloads", g="treated", t="post", i="city")


def test_ch08_staggered_with_dates():
    c = _read("offline_mkt_staggered.csv").astype(
        {"date": "datetime64[ns]", "cohort": "datetime64[ns]"}
    )
    w = c[c["region"] == "W"].copy()
    d0 = w["date"].min()
    w["t"] = (w["date"] - d0).dt.days
    w["g"] = np.where(w["cohort"].dt.year == 2100, 0, (w["cohort"] - d0).dt.days)
    for fn, kw_num, kw_date in [
        (
            sp.callaway_santanna,
            dict(g="g", t="t", i="city"),
            dict(g="cohort", t="date", i="city"),
        ),
        (
            sp.did_imputation,
            dict(group="city", time="t", first_treat="g"),
            dict(group="city", time="date", first_treat="cohort"),
        ),
        (
            sp.sun_abraham,
            dict(g="g", t="t", i="city"),
            dict(g="cohort", t="date", i="city"),
        ),
    ]:
        a = _quiet(fn, w, y="downloads", **kw_num)
        b = _quiet(fn, w, y="downloads", **kw_date)
        assert b.estimate == pytest.approx(a.estimate, rel=1e-12)
        assert b.se == pytest.approx(a.se, rel=1e-12)
        assert b.model_info["calendar_time"]["periods"][0] == d0
    # book: the cohort-by-date saturated TWFE gives 2.259766144685074
    bjs = _quiet(sp.did_imputation, w, y="downloads", group="city", time="date",
                 first_treat="cohort")
    assert bjs.estimate == pytest.approx(2.259766144685074, rel=1e-8)


# --------------------------------------------------------------------- #
#  Chapter 9: synthetic control
# --------------------------------------------------------------------- #


def _online() -> pd.DataFrame:
    df = _read("online_mkt.csv").astype({"date": "datetime64[ns]"})
    df["y"] = 100 * df["app_download"] / df["population"]
    return df


def test_ch09_synthetic_control_of_the_treated_average():
    from scipy.optimize import minimize

    df = _online()
    treated = list(df.loc[df["treated"] == 1, "city"].unique())
    start = pd.Timestamp("2022-05-01")
    res = _quiet(sp.geolift, df, outcome="y", geo="city", time="date",
                 treated_geos=treated, treatment_time=start)
    piv = df.pivot(index="date", columns="city", values="y")
    pre = piv.index < start
    A = piv.drop(columns=treated)[pre].to_numpy()
    b = piv[treated].mean(axis=1)[pre].to_numpy()
    n = A.shape[1]
    sol = minimize(
        lambda w: np.sum((A @ w - b) ** 2),
        np.ones(n) / n,
        jac=lambda w: 2 * A.T @ (A @ w - b),
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        method="SLSQP",
        options={"ftol": 1e-16, "maxiter": 5000},
    )
    post_gap = piv[treated].mean(axis=1)[~pre] - piv.drop(columns=treated)[~pre] @ sol.x
    assert res.estimate == pytest.approx(float(post_gap.mean()), rel=1e-6)
    # book (cvxpy, default accuracy): 0.003327040979396121
    assert res.estimate == pytest.approx(0.003327040979396121, rel=1e-2)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="one treated unit"):
        sp.synth(df, outcome="y", unit="city", time="date",
                 treated_unit=treated, treatment_time=start)


def test_ch09_debiased_t_test():
    df = _online()
    treated = list(df.loc[df["treated"] == 1, "city"].unique())
    res = _quiet(sp.geolift, df, outcome="y", geo="city", time="date",
                 treated_geos=treated, treatment_time=pd.Timestamp("2022-05-01"),
                 inference="ttest", alpha=0.1)
    # book: ATT 0.003324950193632595, SE 0.0006318346108228888,
    # 90% CI [0.0014800022408602205, 0.005169898146404969]
    assert res.estimate == pytest.approx(0.003324950193632595, rel=1e-6)
    assert res.se == pytest.approx(0.0006318346108228888, rel=1e-6)
    assert res.ci[0] == pytest.approx(0.0014800022408602205, rel=1e-5)
    assert res.ci[1] == pytest.approx(0.005169898146404969, rel=1e-5)


# --------------------------------------------------------------------- #
#  Chapter 10: switchback experiments
# --------------------------------------------------------------------- #


def test_ch10_switchback_every_period():
    df = _read("sb_exp_every.csv")
    res = sp.switchback(df, y="delivery_time", treat="d", m=2, design="every", seed=0)
    assert res.estimate == pytest.approx(-7.426440677966101, rel=1e-12)
    assert np.isnan(res.se)


def test_ch10_switchback_optimal_design():
    df = _read("sb_exp_opt.csv")
    res = sp.switchback(df, y="delivery_time", treat="d", m=2,
                        design="rand_points", seed=0)
    assert res.estimate == pytest.approx(-9.921016949152545, rel=1e-12)
    # book: [-18.490627362048095, -1.351406536256997] with 1.96
    half = 1.96 * res.se
    assert res.estimate - half == pytest.approx(-18.490627362048095, rel=1e-10)
    assert res.estimate + half == pytest.approx(-1.351406536256997, rel=1e-10)
    assert res.model_info["optimal_design"]


# --------------------------------------------------------------------- #
#  Chapter 11: instruments and discontinuities
# --------------------------------------------------------------------- #


def test_ch11_two_stage_least_squares():
    df = _read("prime_card.csv")
    wald = sp.iv("pv ~ 1 + [prime_card ~ prime_elegible]", data=df)
    # book: LATE 757.6973795343938, SE 80.52861026141942 (no small-sample factor)
    assert wald.params["prime_card"] == pytest.approx(757.6973795343938, rel=1e-10)
    n, k = len(df), 2
    assert wald.std_errors["prime_card"] * np.sqrt((n - k) / n) == pytest.approx(
        80.52861026141942, rel=1e-9
    )
    both = sp.iv(
        "pv ~ 1 + [prime_card ~ prime_elegible + credit_score] + income + age", data=df
    )
    # book: 693.12072518, SE 12.164694395033125
    assert both.params["prime_card"] == pytest.approx(693.12072518, rel=1e-9)
    assert both.std_errors["prime_card"] * np.sqrt((n - 4) / n) == pytest.approx(
        12.164694395033125, rel=1e-9
    )


def test_ch11_fuzzy_discontinuity_global_linear():
    dd = _read("prime_card_discontinuity.csv")
    res = _quiet(sp.rdrobust, dd, y="pv", x="balance", c=5000, fuzzy="prime_card",
                 h=1e9, kernel="uniform")
    conventional = float(res.model_info["conventional"]["estimate"])
    # The book codes the 319 accounts exactly at the threshold as below it
    # (balance > 0) and reports 732.8534752298891; rdrobust's convention is
    # x >= c treated, which gives 727.3358180959675 on the same two lines.
    assert conventional == pytest.approx(727.3358180959675, rel=1e-8)
