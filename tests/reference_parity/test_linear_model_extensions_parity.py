"""Linear-model extensions against R and Stata on one committed file.

The functions and options here came out of a pass over Peng Ding's *Linear
Model and Extensions* (2024): HC4, influence measures, regression through
the origin, the cauchit link, generalized estimating equations, ridge,
Box-Cox, best subsets, weighted quantile regression and the robust
variance of the Cox model under tied event times.

``_fixtures/linear_model_extensions.csv`` is synthetic (60 clusters of 4 to
12 rows; ``_generate_linear_model_extensions_data.py``). The references are
``linear_model_extensions_R.json`` (R 4.5: sandwich, MASS, leaps, gee,
quantreg, survival; ``_generate_linear_model_extensions_R.R``) and
``linear_model_extensions_Stata.csv`` (Stata 18: xtgee, boxcox, stcox;
``_generate_linear_model_extensions_Stata.do``). Both are stored as the
programs wrote them.

Tolerances. ``EXACT`` (1e-9 relative) where both sides evaluate a closed
form or solve the same convex problem to machine precision. ``ITER`` (1e-6)
where the reference stops an iteration at its own tolerance: R ``glm`` on a
non-canonical link, Stata ``xtgee`` / ``boxcox`` / ``stcox``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
EXACT = 1e-9
ITER = 1e-6
F_ALL = "y ~ x1 + x2 + x3 + x4 + x5 + x6 + treat"
GEE_X = ["_cons", "treat", "x1", "x2", "x4"]
FAMILIES = [("gaussian", "ly"), ("binomial", "d"), ("poisson", "c")]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    data = pd.read_csv(FIX / "linear_model_extensions.csv")
    data["ly"] = np.log(data["y"])
    return data


@pytest.fixture(scope="module")
def R() -> dict:
    text = (FIX / "linear_model_extensions_R.json").read_text(encoding="utf-8")
    return json.loads(text)


@pytest.fixture(scope="module")
def stata() -> pd.Series:
    table = pd.read_csv(FIX / "linear_model_extensions_Stata.csv")
    return table.set_index(["model", "term"])["value"]


def close(ours, ref, rtol):
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float).ravel(),
        np.asarray(ref, dtype=float).ravel(),
        rtol=rtol,
        atol=1e-12,
    )


# ---------------------------------------------------------------- OLS


def test_hc4_matches_sandwich(df, R):
    close(sp.regress(F_ALL, df, robust="hc4").std_errors, R["ols"]["hc4"], EXACT)
    weighted = sp.regress(F_ALL, df, robust="HC4", weights="w")
    close(weighted.std_errors, R["ols"]["hc4_w"], EXACT)


def test_hc4_discounts_leverage_more_than_hc3():
    # one far-out design point with a large residual: HC4's exponent at
    # that point is min(4, n h / k) > 2, so its variance exceeds HC3's
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(size=59), 12.0]
    y = x + np.r_[rng.normal(size=59), 6.0]
    data = pd.DataFrame({"x": x, "y": y})
    se3 = sp.regress("y ~ x", data, robust="hc3").std_errors["x"]
    se4 = sp.regress("y ~ x", data, robust="hc4").std_errors["x"]
    assert se4 > se3


def test_influence_measures_match_r(df, R):
    out = sp.estat(sp.regress(F_ALL, df), "leverage", print_results=False)
    close(out["leverage"], R["ols"]["hat"], EXACT)
    close(out["rstudent"], R["ols"]["rstudent"], EXACT)
    close(out["dffits"], R["ols"]["dffits"], EXACT)
    close(out["cooks_d"], R["ols"]["cook"], EXACT)
    close(out["press"], R["ols"]["press"], EXACT)


def test_leave_one_out_quantities_equal_refits(df):
    small = df.head(40).reset_index(drop=True)
    fit = sp.regress("y ~ x1 + x2 + treat", small)
    out = sp.estat(fit, "leverage", print_results=False, alpha=0.1)
    for i in (0, 17, 39):
        rest = small.drop(index=i)
        refit = sp.regress("y ~ x1 + x2 + treat", rest)
        pred = refit.predict(small.iloc[[i]], what="prediction", alpha=0.1)
        assert small.loc[i, "y"] - pred["yhat"].iloc[0] == pytest.approx(
            out["loo_residuals"][i], rel=1e-9
        )
        half = (pred["upper"].iloc[0] - pred["lower"].iloc[0]) / 2
        assert half == pytest.approx(out["loo_interval_halfwidth"][i], rel=1e-9)


def test_regression_through_the_origin_on_x_and_one_minus_x(df, R):
    # Goodman's ecological regression: correlation -1, yet full rank
    fit = sp.regress("y ~ 0 + x1 + I(1 - x1)", df)
    close(fit.params, R["origin"]["est"], EXACT)
    close(fit.std_errors, R["origin"]["se"], EXACT)


def test_through_origin_still_refuses_proportional_columns(df):
    with pytest.raises(sp.NumericalInstability, match="cosine"):
        sp.regress("y ~ 0 + x1 + I(2 * x1)", df, collinear="raise")
    one = df.assign(one=1.0, uno=1.0)
    fit = sp.regress("y ~ 0 + one + x1", one)
    close(fit.params, sp.regress("y ~ x1", df).params, EXACT)
    with pytest.raises(sp.NumericalInstability, match="constant"):
        sp.regress("y ~ 0 + one + uno + x1", one, collinear="raise")


def test_caret_inside_identity_is_a_power(df, R):
    fit = sp.regress("y ~ x1 + I(x1^2)", df)
    close(fit.params, R["square"]["est"], EXACT)
    close(fit.std_errors, R["square"]["se"], EXACT)
    assert list(fit.params.index) == ["Intercept", "x1", "I(x1 ** 2)"]
    close(
        fit.predict(df.head(4)),
        sp.regress("y ~ x1 + I(x1**2)", df).predict(df.head(4)),
        EXACT,
    )
    for estimator in (sp.logit, sp.poisson):
        a = estimator("d ~ x1 + I(x1^2)", df).params
        b = estimator("d ~ x1 + I(x1**2)", df).params
        close(a, b, EXACT)


def test_caret_outside_identity_keeps_its_formula_meaning():
    from statspai.core.utils import r_power_in_identity

    assert r_power_in_identity("y ~ (a + b)^2 + I((x - 1)^3)") == (
        "y ~ (a + b)^2 + I((x - 1)**3)"
    )
    assert r_power_in_identity("y ~ MI(x^2)") == "y ~ MI(x^2)"


# ---------------------------------------------------------------- GLM


def test_cauchit_link_matches_r_glm(df, R):
    fit = sp.glm(
        "d ~ x1 + x2 + x4 + treat",
        df,
        family="binomial",
        link="cauchit",
        information="expected",
        tol=1e-13,
    )
    close(fit.params, R["cauchit"]["est"], ITER)
    close(fit.std_errors, R["cauchit"]["se"], ITER)
    assert fit.diagnostics["Deviance"] == pytest.approx(
        R["cauchit"]["deviance"], rel=1e-10
    )


# ---------------------------------------------------------------- GEE


@pytest.mark.parametrize("family,dep", FAMILIES)
@pytest.mark.parametrize("corstr", ["independence", "exchangeable", "ar-m"])
def test_gee_matches_r_gee(df, R, family, dep, corstr):
    # the fixture stores gee's "AR-M" (Mv = 1) fits under the key "ar1"
    ref = R["gee"][f"{family}_{corstr.replace('ar-m', 'ar1')}"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.gee(
            f"{dep} ~ treat + x1 + x2 + x4",
            df,
            id="id",
            time="t",
            family=family,
            corstr=corstr,
            tol=1e-12,
        )
    close(fit.params, ref["est"], EXACT)
    close(fit.model_info["se_model"], ref["naive"], EXACT)
    close(fit.std_errors, ref["robust"], EXACT)
    close(fit.model_info["scale"], ref["scale"], EXACT)
    if corstr != "independence":
        close(fit.model_info["corr_alpha"], ref["alpha"], EXACT)


def test_the_two_ar1_moments_agree_on_a_balanced_panel_and_part_on_a_ragged_one(df):
    rng = np.random.default_rng(8)
    G, m = 150, 6
    g = np.repeat(np.arange(G), m)
    e = rng.normal(size=G * m)
    for j in range(1, m):
        e[j::m] = 0.6 * e[j - 1 :: m] + 0.8 * e[j::m]
    data = pd.DataFrame(
        {"g": g, "t": np.tile(np.arange(m), G), "x": rng.normal(size=G * m)}
    )
    data["y"] = 1 + 0.5 * data["x"] + e
    # balanced: the same moment once the pooled one drops its N - p term
    pooled = sp.gee("y ~ x", data, id="g", time="t", corstr="ar1", dof_correction=False)
    by_cluster = sp.gee("y ~ x", data, id="g", time="t", corstr="ar-m")
    close(pooled.model_info["corr_alpha"], by_cluster.model_info["corr_alpha"], 1e-8)
    close(pooled.params, by_cluster.params, 1e-8)
    assert pooled.model_info["corr_alpha"] == pytest.approx(0.6, abs=0.06)
    # ragged (the committed file): the estimators differ
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.gee("ly ~ treat + x1", df, id="id", time="t", corstr="ar1")
        b = sp.gee("ly ~ treat + x1", df, id="id", time="t", corstr="ar-m")
    assert abs(a.model_info["corr_alpha"] - b.model_info["corr_alpha"]) > 5e-3


@pytest.mark.parametrize("family,dep", FAMILIES)
@pytest.mark.parametrize(
    "corstr,stata_name",
    [("independence", "independent"), ("exchangeable", "exchangeable"), ("ar1", "ar1")],
)
@pytest.mark.parametrize("tag,dof", [("n", False), ("nmp", True)])
def test_gee_matches_stata_xtgee(df, stata, family, dep, corstr, stata_name, tag, dof):
    key = f"{family}_{stata_name}_{tag}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.gee(
            f"{dep} ~ treat + x1 + x2 + x4",
            df,
            id="id",
            time="t",
            family=family,
            corstr=corstr,
            dof_correction=dof,
            scale=None if family == "gaussian" else 1.0,
            tol=1e-12,
        )
    close(fit.params, [stata[key, f"b_{v}"] for v in GEE_X], ITER)
    close(
        fit.model_info["se_model"], [stata[key, f"se_model_{v}"] for v in GEE_X], ITER
    )
    # xtgee, vce(robust) is the same sandwich times G / (G - 1)
    G = fit.model_info["n_clusters"]
    close(
        fit.model_info["se_robust"] * np.sqrt(G / (G - 1)),
        [stata[key, f"se_robust_{v}"] for v in GEE_X],
        ITER,
    )
    close(fit.model_info["corr_alpha"], stata[key, "alpha"], ITER)


def test_gee_independence_is_the_pooled_glm_with_clustered_sandwich(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gee = sp.gee("c ~ treat + x1", df, id="id", family="poisson")
        glm = sp.poisson("c ~ treat + x1", df, cluster="id")
    close(gee.params.to_numpy(), glm.params.to_numpy(), 1e-8)
    # sp.poisson's clustered covariance is the same sandwich times G / (G - 1)
    G = df["id"].nunique()
    ratio = glm.std_errors.to_numpy() / gee.std_errors.to_numpy()
    close(ratio, np.full(3, np.sqrt(G / (G - 1))), 1e-8)


def test_gee_row_order_and_cluster_labels_do_not_matter(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.gee("ly ~ treat + x1", df, id="id", time="t", corstr="ar1")
        shuffled = df.sample(frac=1.0, random_state=3).assign(
            id=lambda d_: "c" + d_["id"].astype(str)
        )
        b = sp.gee("ly ~ treat + x1", shuffled, id="id", time="t", corstr="ar1")
    close(a.params, b.params, 1e-10)
    close(a.std_errors, b.std_errors, 1e-10)


def test_gee_recovers_an_exchangeable_correlation():
    rng = np.random.default_rng(11)
    G, m, rho = 400, 6, 0.4
    g = np.repeat(np.arange(G), m)
    x = rng.normal(size=G * m)
    e = np.sqrt(rho) * rng.normal(size=G)[g] + np.sqrt(1 - rho) * rng.normal(size=G * m)
    data = pd.DataFrame({"g": g, "x": x, "y": 1 + 0.5 * x + e})
    fit = sp.gee("y ~ x", data, id="g", corstr="exchangeable")
    assert fit.model_info["corr_alpha"] == pytest.approx(rho, abs=0.05)
    assert fit.params["x"] == pytest.approx(0.5, abs=4 * fit.std_errors["x"])
    # the efficient working correlation beats independence on a
    # within-cluster regressor
    ind = sp.gee("y ~ x", data, id="g")
    assert fit.std_errors["x"] < ind.std_errors["x"]


def test_gee_unstructured_reduces_to_exchangeable_truth():
    rng = np.random.default_rng(5)
    G, m = 600, 4
    g = np.repeat(np.arange(G), m)
    t = np.tile(np.arange(m), G)
    e = np.sqrt(0.5) * rng.normal(size=G)[g] + np.sqrt(0.5) * rng.normal(size=G * m)
    data = pd.DataFrame({"g": g, "t": t, "x": rng.normal(size=G * m)})
    data["y"] = 0.3 * data["x"] + e
    fit = sp.gee("y ~ x", data, id="g", time="t", corstr="unstructured")
    wc = fit.model_info["working_correlation"]
    assert wc.shape == (m, m)
    off = wc[~np.eye(m, dtype=bool)]
    assert np.all(np.abs(off - 0.5) < 0.1)


def test_gee_refuses_bad_input(df):
    with pytest.raises(sp.MethodIncompatibility, match="corstr"):
        sp.gee("ly ~ x1", df, id="id", corstr="toeplitz")
    with pytest.raises(sp.MethodIncompatibility, match="not found"):
        sp.gee("ly ~ x1", df, id="village")
    with pytest.raises(sp.DataInsufficient, match="two clusters"):
        sp.gee("ly ~ x1", df.assign(one=1), id="one")


def test_gee_warns_with_few_clusters(df):
    few = df[df["id"] <= 12]
    with pytest.warns(UserWarning, match="12 clusters"):
        sp.gee("ly ~ x1", few, id="id")


# ---------------------------------------------------------------- ridge


def test_ridge_matches_lm_ridge(df, R):
    ref = R["ridge"]
    fit = sp.ridge(F_ALL, df, lambda_=ref["lambda"])
    names = ["Intercept"] + ref["names"][1:]
    close(fit.path[names].to_numpy(), np.asarray(ref["coef"]), EXACT)
    close(fit.path["gcv"], ref["gcv"], EXACT)
    close(fit.lambda_hkb, ref["kHKB"], EXACT)
    close(fit.lambda_lw, ref["kLW"], EXACT)
    assert fit.lambda_ == ref["lambda"][int(np.argmin(ref["gcv"]))]


def test_ridge_limits(df):
    ols = sp.regress(F_ALL, df).params
    zero = sp.ridge(F_ALL, df, lambda_=0.0)
    close(zero.params[ols.index], ols, 1e-9)
    assert zero.path["df"].iloc[0] == pytest.approx(8.0)
    huge = sp.ridge(F_ALL, df, lambda_=1e12)
    assert np.max(np.abs(huge.params.drop("Intercept"))) < 1e-6
    assert huge.params["Intercept"] == pytest.approx(df["y"].mean(), rel=1e-6)
    close(zero.predict(df.head(5)), sp.regress(F_ALL, df).predict(df.head(5)), 1e-9)


def test_ridge_refuses_bad_input(df):
    with pytest.raises(sp.MethodIncompatibility, match=">= 0"):
        sp.ridge(F_ALL, df, lambda_=[-1.0])
    with pytest.raises(sp.MethodIncompatibility, match="select"):
        sp.ridge(F_ALL, df, select="cv")
    with pytest.raises(sp.MethodIncompatibility, match="do not vary"):
        sp.ridge(y="y", x=["x1", "k"], data=df.assign(k=3.0))


# ---------------------------------------------------------------- Box-Cox


def test_boxcox_profile_matches_mass(df, R):
    ref = R["boxcox"]
    fit = sp.boxcox(F_ALL, df, lambdas=ref["lambda"])
    # MASS drops the constants of the log likelihood: equal up to a shift
    gap = fit.profile["loglik"].to_numpy() - np.asarray(ref["loglik"])
    assert np.ptp(gap) < 1e-9


def test_boxcox_matches_stata(df, stata):
    fit = sp.boxcox(F_ALL, df)
    assert fit.lambda_ == pytest.approx(stata["boxcox", "lambda"], abs=1e-7)
    assert fit.loglik == pytest.approx(stata["boxcox", "ll"], rel=1e-8)
    tests = fit.tests.set_index("lambda")
    for null, tag in [(-1.0, "m1"), (0.0, "0"), (1.0, "1")]:
        assert tests.loc[null, "loglik"] == pytest.approx(
            stata["boxcox", f"ll_{tag}"], rel=1e-8
        )
        assert tests.loc[null, "chi2"] == pytest.approx(
            stata["boxcox", f"chi2_{tag}"], rel=1e-4, abs=1e-6
        )


def test_boxcox_interval_is_the_likelihood_ratio_set(df):
    from scipy import stats

    fit = sp.boxcox(F_ALL, df, alpha=0.1)
    cut = fit.loglik - stats.chi2.ppf(0.9, 1) / 2
    ends = sp.boxcox(F_ALL, df, lambdas=list(fit.ci)).profile["loglik"]
    close(ends, [cut, cut], 1e-8)
    assert fit.ci[0] < fit.lambda_ < fit.ci[1]


def test_boxcox_refuses_nonpositive_outcomes(df):
    with pytest.raises(sp.MethodIncompatibility, match="strictly positive"):
        sp.boxcox("c ~ x1", df)


# ---------------------------------------------------------------- subsets


def test_best_subset_matches_leaps(df, R):
    ref = R["subsets"]
    cand = ["x1", "x2", "x3", "x4", "x5", "x6", "treat"]
    fit = sp.best_subset(df, "y", cand, criterion="bic")
    close(fit.history["rss"], ref["rss"], EXACT)
    close(fit.history["adj_r_squared"], ref["adjr2"], EXACT)
    np.testing.assert_allclose(fit.history["cp"], ref["cp"], atol=1e-8)
    for ours, theirs in zip(fit.history["variables"], ref["which"]):
        theirs = [theirs] if isinstance(theirs, str) else theirs
        assert sorted(ours) == sorted(theirs)
    step = sp.stepwise(df, "y", cand, criterion="bic", verbose=False)
    assert fit.final_model["bic"] <= step.final_model["bic"] + 1e-9


def test_best_subset_is_exact_with_forced_regressors(df):
    import itertools

    cand = ["x1", "x2", "x3", "x5", "x6"]
    fit = sp.best_subset(df, "ly", cand, force=["treat"], criterion="aic")
    assert fit.selected[0] == "treat"

    def rss(cols):
        fit_ = sp.regress("ly ~ treat + " + " + ".join(cols), df)
        return float(np.sum(np.asarray(fit_.residuals()) ** 2))

    for size in range(1, len(cand) + 1):
        best = min(rss(c) for c in itertools.combinations(cand, size))
        assert fit.history["rss"].iloc[size - 1] == pytest.approx(best, rel=1e-10)


def test_best_subset_refuses_bad_input(df):
    with pytest.raises(sp.MethodIncompatibility, match="limited to 30"):
        wide = pd.DataFrame(np.random.default_rng(0).normal(size=(80, 32)))
        wide.columns = [f"v{i}" for i in range(32)]
        sp.best_subset(wide, "v0", list(wide.columns[1:]))
    with pytest.raises(sp.MethodIncompatibility, match="criterion"):
        sp.best_subset(df, "y", ["x1", "x2"], criterion="r2")


# ---------------------------------------------------------------- quantile


@pytest.mark.parametrize("tau", [0.25, 0.5, 0.75])
def test_weighted_quantile_regression_matches_quantreg(df, R, tau):
    ref = R["rq"][f"tau{tau}"]
    f = "ly ~ x1 + x2 + x4 + treat"
    close(sp.qreg(df, f, quantile=tau, vce="ker").std_errors, ref["ker"], EXACT)
    ker = sp.qreg(df, f, quantile=tau, vce="ker", weights="w")
    close(ker.params, ref["coef_w"], 1e-8)
    close(ker.std_errors, ref["ker_w"], 1e-8)
    nid = sp.qreg(df, f, quantile=tau, vce="nid", weights=df["w"].to_numpy())
    close(nid.std_errors, ref["nid_w"], 1e-8)
    close(sp.qreg(df, f, quantile=tau, vce="nid").std_errors, ref["nid"], 1e-8)


def test_integer_weights_equal_replicated_rows(df):
    small = df.head(120).assign(k=lambda d_: 1 + (d_["id"] % 3))
    expanded = small.loc[small.index.repeat(small["k"])].reset_index(drop=True)
    a = sp.qreg(small, "ly ~ x1 + treat", quantile=0.4, weights="k")
    b = sp.qreg(expanded, "ly ~ x1 + treat", quantile=0.4)
    close(a.params, b.params, 1e-8)
    assert a.model_info["vce"] == "robust" and a.model_info["weighted"]
    assert a.model_info["pseudo_r2"] == pytest.approx(
        b.model_info["pseudo_r2"], abs=1e-8
    )


def test_quantile_weights_must_be_positive(df):
    with pytest.raises(sp.MethodIncompatibility, match="strictly positive"):
        sp.qreg(df.assign(z=0.0), "ly ~ x1", weights="z")


# ---------------------------------------------------------------- Cox

COX_X = ["treat", "x1", "x2", "x4"]


def test_cox_robust_variance_under_efron_ties_matches_survival(df, R):
    ref = R["cox"]
    assert ref["n_tied_times"] >= 30  # the fixture is there for its ties
    kw = dict(data=df, duration="time", event="event", x=COX_X)
    efron = sp.cox(robust="hc0", **kw)
    close(efron.params, ref["efron"]["est"], EXACT)
    close(efron.std_errors, ref["efron"]["robust"], EXACT)
    breslow = sp.cox(robust="hc0", ties="breslow", **kw)
    close(breslow.std_errors, ref["breslow"]["robust"], EXACT)
    # the two tie rules give different robust variances on these data;
    # before 1.39 the Efron fit reported the Breslow residuals' variance
    assert np.max(np.abs(efron.std_errors / breslow.std_errors - 1)) > 1e-3
    strata = sp.cox(robust="hc0", strata="site", **kw)
    close(strata.std_errors, ref["efron_strata"]["robust"], EXACT)
    cluster = sp.cox(cluster="id", **kw)
    G = df["id"].nunique()
    close(
        cluster.std_errors * np.sqrt((G - 1) / G), ref["efron_cluster"]["robust"], EXACT
    )


def test_cox_matches_stata_stcox(df, stata):
    kw = dict(data=df, duration="time", event="event", x=COX_X)
    n = len(df)
    robust = sp.cox(robust="hc0", **kw)
    close(robust.params, [stata["cox_efron_robust", f"b_{v}"] for v in COX_X], ITER)
    # stcox, vce(robust) is the same sandwich times N / (N - 1)
    close(
        robust.std_errors * np.sqrt(n / (n - 1)),
        [stata["cox_efron_robust", f"se_{v}"] for v in COX_X],
        ITER,
    )
    cluster = sp.cox(cluster="id", **kw)
    close(
        cluster.std_errors, [stata["cox_efron_cluster", f"se_{v}"] for v in COX_X], ITER
    )


def test_cox_score_residuals_sum_to_zero_at_the_estimate(df):
    from statspai.survival.models import _cox_score_individual

    fit = sp.cox(data=df, duration="time", event="event", x=COX_X)
    X = df[COX_X].to_numpy(float)
    T = df["time"].to_numpy(float)
    E = df["event"].to_numpy(float)
    for breslow in (False, True):
        beta = sp.cox(
            data=df,
            duration="time",
            event="event",
            x=COX_X,
            ties="breslow" if breslow else "efron",
        ).params.to_numpy()
        scores = _cox_score_individual(beta, X, T, E, breslow=breslow)
        assert np.max(np.abs(scores.sum(axis=0))) < 1e-6
    assert fit.params.shape == (4,)


def test_cox_formula_builds_factors_and_interactions(df):
    by_formula = sp.cox("time ~ treat*x1 + C(site) + I(x2^2)", df, event="event")
    wide = df.assign(
        tx=df["treat"] * df["x1"],
        s2=(df["site"] == 2).astype(float),
        s3=(df["site"] == 3).astype(float),
        x2sq=df["x2"] ** 2,
    )
    by_hand = sp.cox(
        data=wide,
        duration="time",
        event="event",
        x=["s2", "s3", "treat", "x1", "tx", "x2sq"],
    )
    assert len(by_formula.params) == 6
    close(
        np.sort(by_formula.params.to_numpy()), np.sort(by_hand.params.to_numpy()), 1e-8
    )


# ---------------------------------------------------------------- Kaplan-Meier


def _km_at(table: pd.DataFrame, t: float) -> pd.Series:
    return table[table["time"] <= t].iloc[-1]


def test_kaplan_meier_log_interval_matches_survfit(df, R):
    ref = R["km"]
    table = sp.kaplan_meier(df, "time", "event", conf_type="log").survival_table
    for i, t in enumerate(ref["time"]):
        row = _km_at(table, t)
        close(
            [row["survival"], row["std_err"], row["ci_lower"], row["ci_upper"]],
            [ref["surv"][i], ref["se"][i], ref["lower"][i], ref["upper"][i]],
            EXACT,
        )


def test_kaplan_meier_loglog_interval_matches_stata_sts(df, stata):
    table = sp.kaplan_meier(df, "time", "event", conf_type="log-log").survival_table
    for t in (5, 15, 30, 60):
        row = _km_at(table, t)
        close(
            [row["survival"], row["ci_lower"], row["ci_upper"]],
            [stata["km", f"s_{t}"], stata["km", f"lb_{t}"], stata["km", f"ub_{t}"]],
            1e-6,  # sts stores its results in single precision
        )


def test_kaplan_meier_default_interval_is_unchanged_and_announced(df):
    with pytest.warns(DeprecationWarning, match="log-log"):
        default = sp.kaplan_meier(df, "time", "event").survival_table
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = sp.kaplan_meier(df, "time", "event", conf_type="plain").survival_table
    pd.testing.assert_frame_equal(default, plain)
    row = _km_at(plain, 30)
    half = 1.959963984540054 * row["std_err"]
    assert row["ci_lower"] == pytest.approx(row["survival"] - half)
    assert row["ci_upper"] == pytest.approx(row["survival"] + half)
    with pytest.raises(ValueError, match="conf_type"):
        sp.kaplan_meier(df, "time", "event", conf_type="arcsine")


# ---------------------------------------------------------------- probabilities


def test_predicted_probabilities_of_categorical_models(df):
    data = df.assign(
        grade=pd.cut(
            df["ly"], [-np.inf, 0.6, 1.3, np.inf], labels=["lo", "mid", "hi"]
        ).astype(str)
    )
    for fit_fn in (sp.mlogit, sp.ologit, sp.oprobit):
        fit = fit_fn("grade ~ x1 + I(x2^2) + C(site)", data)
        inside = fit.predict()
        again = fit.predict(data)
        assert list(inside.columns) == ["hi", "lo", "mid"]
        close(again.to_numpy(), inside.to_numpy(), 1e-10)
        close(again.sum(axis=1), np.ones(len(data)), 1e-12)
        plain = fit_fn("grade ~ x1 + x3", data)
        close(
            plain.predict(data.head(7)).to_numpy(),
            plain.predict().head(7).to_numpy(),
            1e-10,
        )
    with pytest.raises(sp.MethodIncompatibility, match="do not reproduce"):
        fit.predict(data[data["site"] == 2].head())


def test_string_outcome_with_a_built_formula_keeps_every_category(df):
    data = df.assign(
        grade=np.where(df["ly"] > 1.3, "hi", np.where(df["ly"] > 0.6, "mid", "lo"))
    )
    built = sp.mlogit("grade ~ x1 + I(x2^2)", data)
    by_hand = sp.mlogit("grade ~ x1 + x2sq", data.assign(x2sq=data["x2"] ** 2))
    assert built.model_info["n_categories"] == 3
    assert built.model_info["log_likelihood"] == pytest.approx(
        by_hand.model_info["log_likelihood"], rel=1e-10
    )


# ---------------------------------------------------------------- follow-ups


def test_cox_wald_and_score_tests_match_survival(df, R):
    ref = R["cox_tests"]
    kw = dict(data=df, duration="time", event="event", x=COX_X)
    diag = sp.cox(**kw).diagnostics
    assert diag["LR chi2"] == pytest.approx(ref["lr"], rel=EXACT)
    assert diag["Wald chi2"] == pytest.approx(ref["wald"], rel=1e-8)
    assert round(diag["Wald chi2"], 2) == ref["wald_printed"]
    assert diag["Score chi2"] == pytest.approx(ref["score"], rel=EXACT)
    robust = sp.cox(robust="hc0", **kw).diagnostics
    assert robust["Wald chi2"] == pytest.approx(ref["wald_robust"], rel=1e-8)
    assert robust["Score chi2"] == pytest.approx(ref["score"], rel=EXACT)


def test_score_test_of_a_group_indicator_is_the_logrank_test(df):
    # with no shared event times the two are the same statistic; ties add
    # the hypergeometric factor (n - d) / (n - 1) to the log-rank variance
    untied = df.assign(time=df["time"] + np.arange(len(df)) / (10.0 * len(df)))
    fit = sp.cox(data=untied, duration="time", event="event", x=["treat"])
    logrank = sp.logrank_test(untied, "time", "event", "treat")
    assert fit.diagnostics["Score chi2"] == pytest.approx(
        logrank["test_statistic"], rel=1e-9
    )


def test_binary_predictions_carry_a_standard_error(df, R):
    ref = R["logit_predict"]
    fit = sp.logit("d ~ x1 + x2 + x4 + treat", df, tol=1e-13)
    out = fit.predict(df.head(8), what="confidence")
    close(out["yhat"], ref["fit"], ITER)
    close(out["se"], ref["se"], ITER)
    assert (
        (out["lower"] > 0) & (out["upper"] < 1) & (out["lower"] < out["yhat"])
    ).all()
    close(fit.predict(df.head(8)), ref["fit"], ITER)  # the old call is unchanged
    for fn, link in [(sp.probit, "probit"), (sp.cloglog, "cloglog")]:
        ours = fn("d ~ x1 + I(x2^2)", df).predict(df.head(5), what="confidence")
        glm = sp.glm("d ~ x1 + I(x2^2)", df, family="binomial", link=link)
        close(
            ours.to_numpy(), glm.predict(df.head(5), what="confidence").to_numpy(), 1e-6
        )
    with pytest.raises(sp.MethodIncompatibility, match="prediction interval"):
        fit.predict(df.head(3), what="prediction")


def test_zero_inflated_fits_name_their_likelihood_like_the_rest(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = [
            sp.zip_model("c ~ x1 + treat", df),
            sp.zinb("c ~ x1 + treat", df),
            sp.hurdle("c ~ x1 + treat", df),
        ]
    for fit in fits:
        diag = fit.diagnostics
        assert diag["Log-Likelihood"] == diag["ll"]
        assert diag["AIC"] == diag["aic"] and diag["BIC"] == diag["bic"]


# ---------------------------------------------------------------- sp.stata


def test_stata_lines_for_xtgee_and_boxcox_reproduce_stata(df, stata):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.stata(
            "xtset id t\nxtgee c treat x1 x2 x4, family(poisson) corr(exchangeable)",
            data=df,
        )
        ar1 = sp.stata(
            "xtset id t\nxtgee d treat x1 x2 x4, f(binomial) c(ar 1) nmp vce(robust)",
            data=df,
        )
    key = "poisson_exchangeable_n"
    close(fit.params, [stata[key, f"b_{v}"] for v in GEE_X], ITER)
    close(fit.std_errors, [stata[key, f"se_model_{v}"] for v in GEE_X], ITER)
    key = "binomial_ar1_nmp"
    close(ar1.params, [stata[key, f"b_{v}"] for v in GEE_X], ITER)
    G = df["id"].nunique()
    close(
        ar1.std_errors * np.sqrt(G / (G - 1)),
        [stata[key, f"se_robust_{v}"] for v in GEE_X],
        ITER,
    )
    bc = sp.stata("boxcox y x1 x2 x3 x4 x5 x6 treat", data=df)
    assert bc.lambda_ == pytest.approx(stata["boxcox", "lambda"], abs=1e-7)


def test_xtgee_translation_writes_out_stata_conventions():
    out = sp.from_stata("xtgee d x1, family(binomial) i(id)")
    assert out["arguments"] == {
        "formula": "d ~ x1",
        "id": "id",
        "family": "binomial",
        "corstr": "exchangeable",  # xtgee's default
        "vce": "model",  # and its default covariance
        "dof_correction": False,  # N, not N - p
        "scale": 1.0,  # held at one for this family
    }
    assert not sp.from_stata("xtgee y x1, corr(stationary 2) i(id)")["ok"]
    assert not sp.from_stata("xtgee y x1, family(nbinomial) i(id)")["ok"]
    assert not sp.from_stata("boxcox y x1, model(theta)")["ok"]
    assert sp.from_stata("xtgee y x1, i(id) eform")["untranslated_options"] == ["eform"]
