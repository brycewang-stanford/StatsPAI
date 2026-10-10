"""The GRF family (instrumental, multi-arm, linear-model, prediction and
survival forests): what happens after the trees are grown.

A fitted forest defines, for a point ``x``, weights ``alpha_i(x)`` over the
training rows (``forest_weights``).  Every prediction of the family is a
closed-form function of those weights, the sample weights and the stored
(centred) data -- a weighted mean, a local moment solve, a weighted
quantile, a Kaplan-Meier curve -- and every average effect is a
closed-form function of the out-of-bag predictions and the nuisances.
The tests rebuild each from its definition with numpy and compare exactly,
with and without ``weights=`` and ``clusters=``, so an option that is read
and then dropped on one path shows up as a mismatch.

Forests are small (400 rows, 100 trees) and fitted once per module.
``ATOL = 1e-10``: both sides are the same sums in a different order.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.forest._grf_family import (
    ForestOptions,
    importance_from_split_frequencies,
    names_or_default,
    normal_pvalue,
    observation_weights,
    one_hot_newdata,
    score_average,
    user_nuisance,
    with_stream,
)
from statspai.forest.iv_forest import iv_debiasing_weights, iv_dr_scores
from statspai.forest.multi_arm_forest import multi_arm_scores

ATOL = 1e-10
N = 400
KW = dict(n_estimators=100, random_state=21)


def _wmean(a, v):
    return np.sum(a * v) / np.sum(a)


def _local_solve(a, y, R):
    """Slope of the ``a``-weighted regression of ``y`` on ``R`` with an
    intercept: the local moment solve of a causal / lm / multi-arm forest."""
    R = R.reshape(len(y), -1)
    Rc = R - np.array([_wmean(a, R[:, j]) for j in range(R.shape[1])])
    yc = y - _wmean(a, y)
    return np.linalg.solve((Rc * a[:, None]).T @ Rc, (Rc * a[:, None]).T @ yc)


def _mean_se(values, w, clusters=None):
    n = len(values)
    est = np.sum(w * values) / np.sum(w)
    if clusters is None:
        var = np.sum(w**2 * (values - est) ** 2) / np.sum(w) ** 2 * n / (n - 1)
    else:
        labels = np.unique(clusters)
        sums = np.array([np.sum((w * (values - est))[clusters == g]) for g in labels])
        var = np.sum(sums**2) / np.sum(w) ** 2 * len(labels) / (len(labels) - 1)
    return est, np.sqrt(var)


def _cluster_hc1(D, resid, w, clusters):
    """``sandwich::vcovCL(type = "HC1")`` for a weighted fit."""
    n, k = D.shape
    bread = np.linalg.inv(D.T @ (D * w[:, None]))
    ef = D * (w * resid)[:, None]
    labels = np.unique(clusters)
    sums = np.vstack([ef[clusters == g].sum(axis=0) for g in labels])
    G = len(labels)
    meat = sums.T @ sums * G / (G - 1) * (n - 1) / (n - k)
    return bread @ meat @ bread


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(301)
    X = rng.normal(size=(N, 3))
    Z = rng.binomial(1, 0.5, N).astype(float)
    U = rng.normal(size=N)
    W = (0.3 * U + Z + rng.normal(size=N) > 0.6).astype(float)
    Y = X[:, 1] + (1 + X[:, 0]) * W + U + rng.normal(size=N)
    arm = rng.integers(0, 3, N)
    Y3 = X[:, 1] + (1 + X[:, 0]) * (arm == 1) - 0.5 * (arm == 2) + rng.normal(size=N)
    return dict(
        X=X,
        Z=Z,
        W=W,
        Y=Y,
        arm=arm,
        Y3=Y3,
        cl=rng.integers(0, 35, N),
        sw=rng.uniform(0.3, 3.0, N),
        new=rng.normal(size=(6, 3)),
    )


# --------------------------------------------------------------------------- #
#  Instrumental forest
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def iv_plain(data):
    return sp.iv_forest(
        y=data["Y"], treat=data["W"], instrument=data["Z"], covariates=data["X"], **KW
    )


@pytest.fixture(scope="module")
def iv_weighted(data):
    return sp.iv_forest(
        y=data["Y"],
        treat=data["W"],
        instrument=data["Z"],
        covariates=data["X"],
        clusters=data["cl"],
        weights=data["sw"],
        **KW,
    )


@pytest.mark.parametrize("which", ["plain", "weighted"])
def test_iv_prediction_is_the_local_wald_ratio(data, iv_plain, iv_weighted, which):
    fit = iv_plain if which == "plain" else iv_weighted
    w = np.ones(N) if which == "plain" else data["sw"]
    yc = data["Y"] - fit._y_hat
    wc = data["W"] - fit._w_hat
    zc = data["Z"] - fit._z_hat
    hand = []
    for alpha in fit.forest_weights(data["new"]) * w:
        zd = zc - _wmean(alpha, zc)
        hand.append(
            np.sum(alpha * zd * (yc - _wmean(alpha, yc)))
            / np.sum(alpha * zd * (wc - _wmean(alpha, wc)))
        )
    pred = fit.predict(data["new"])
    np.testing.assert_allclose(pred["predictions"], hand, atol=ATOL)
    # A single vector is one row.
    one = fit.predict(data["new"][0])
    assert one.shape == (1, 1)
    assert one["predictions"].iloc[0] == pytest.approx(hand[0], abs=ATOL)
    # Out-of-bag weights of a row give its out-of-bag prediction.
    oob = fit.forest_weights()[:5] * w
    hand_oob = []
    for alpha in oob:
        zd = zc - _wmean(alpha, zc)
        hand_oob.append(
            np.sum(alpha * zd * (yc - _wmean(alpha, yc)))
            / np.sum(alpha * zd * (wc - _wmean(alpha, wc)))
        )
    np.testing.assert_allclose(fit.cate[:5], hand_oob, atol=ATOL)
    np.testing.assert_array_equal(fit.predict()["predictions"], fit.cate)


def test_iv_scores_and_average_follow_the_documented_formulas(data, iv_weighted):
    fit = iv_weighted
    Y, W, Z, sw, cl = (data[k] for k in ("Y", "W", "Z", "sw", "cl"))
    tau = fit.cate
    # Binary instrument: Var(Z | X) = z_hat (1 - z_hat).
    g = (Z - fit._z_hat) / (fit._z_hat * (1 - fit._z_hat) * fit._compliance)
    scores = tau + g * (Y - fit._y_hat - tau * (W - fit._w_hat))
    np.testing.assert_allclose(fit.get_scores(), scores, atol=ATOL)
    np.testing.assert_allclose(sp.get_scores(fit), scores, atol=ATOL)
    np.testing.assert_allclose(
        iv_debiasing_weights(Z, fit._z_hat, fit._z_var, fit._compliance), g, atol=ATOL
    )
    np.testing.assert_allclose(
        iv_dr_scores(Y, W, fit._y_hat, fit._w_hat, tau, g), scores, atol=ATOL
    )
    est, se = _mean_se(scores, sw, cl)
    assert fit.late == pytest.approx(est, abs=ATOL)
    assert fit.se == pytest.approx(se, abs=ATOL)
    z = stats.norm.ppf(0.975)
    assert fit.ci == pytest.approx((est - z * se, est + z * se), abs=ATOL)
    assert fit.pvalue == pytest.approx(2 * stats.norm.sf(abs(est / se)), rel=1e-8)
    assert fit.detail["n_clusters"] == len(np.unique(cl))
    assert fit.detail["instrument"] == "binary"
    ate = fit.average_treatment_effect(alpha=0.2)
    assert ate["estimand"] == "ACLATE" and ate["estimate"] == pytest.approx(est)
    assert ate["ci_high"] == pytest.approx(est + stats.norm.ppf(0.9) * se, abs=ATOL)
    # The weights matter on this design: the unweighted mean is elsewhere.
    assert abs(est - scores.mean()) > 1e-4


def test_iv_alpha_sets_the_reported_interval(data, iv_plain):
    wide = sp.iv_forest(
        y=data["Y"],
        treat=data["W"],
        instrument=data["Z"],
        covariates=data["X"],
        alpha=0.2,
        **KW,
    )
    assert wide.late == iv_plain.late and wide.se == iv_plain.se
    z = stats.norm.ppf(0.9)
    assert wide.ci == pytest.approx(
        (wide.late - z * wide.se, wide.late + z * wide.se), abs=ATOL
    )
    assert "80% CI" in wide.summary() and "ACLATE" in repr(wide)


def test_iv_score_overrides(data, iv_plain):
    fit = iv_plain
    Y, W, Z = data["Y"], data["W"], data["Z"]
    tau = fit.cate
    resid = Y - fit._y_hat - tau * (W - fit._w_hat)
    # A supplied compliance score replaces the estimated one.
    delta = np.full(N, 0.4)
    g = (Z - fit._z_hat) / (fit._z_var * delta)
    np.testing.assert_allclose(
        fit.get_scores(compliance_score=0.4), tau + g * resid, atol=ATOL
    )
    np.testing.assert_allclose(
        sp.get_scores(fit, compliance_score=delta), tau + g * resid, atol=ATOL
    )
    # Supplied debiasing weights replace the whole representer.
    dw = np.linspace(-1, 1, N)
    np.testing.assert_allclose(
        fit.get_scores(debiasing_weights=dw), tau + dw * resid, atol=ATOL
    )
    est, se = _mean_se(tau + dw * resid, np.ones(N))
    out = fit.average_treatment_effect(debiasing_weights=dw)
    assert out["estimate"] == pytest.approx(est, abs=ATOL)
    assert out["se"] == pytest.approx(se, abs=ATOL)
    with pytest.raises(DataInsufficient, match="compliance score is zero"):
        fit.get_scores(compliance_score=0.0)
    with pytest.raises(MethodIncompatibility, match="compliance_score must have"):
        fit.get_scores(compliance_score=np.ones(5))


def test_iv_blp_is_weighted_ols_of_the_scores(data, iv_weighted):
    fit = iv_weighted
    sw, cl, X = data["sw"], data["cl"], data["X"]
    scores = fit.get_scores()
    w = sw / sw.sum()
    D = np.column_stack([np.ones(N), X[:, 0]])
    beta = np.linalg.solve(D.T @ (D * w[:, None]), D.T @ (w * scores))
    V = _cluster_hc1(D, scores - D @ beta, w, cl)
    tab = fit.best_linear_projection(A=X[:, 0], vce="HC1", alpha=0.1)
    np.testing.assert_allclose(tab["coef"], beta, atol=ATOL)
    np.testing.assert_allclose(tab["se"], np.sqrt(np.diag(V)), atol=ATOL)
    np.testing.assert_allclose(
        tab["ci_upper"], beta + stats.norm.ppf(0.95) * tab["se"], atol=ATOL
    )
    via_sp = sp.best_linear_projection(fit, A=X[:, 0], vce="HC1", alpha=0.1)
    pd.testing.assert_frame_equal(via_sp, tab)
    # Without covariates the projection is the average effect.
    assert fit.best_linear_projection().loc["Intercept", "coef"] == pytest.approx(
        fit.late, abs=ATOL
    )


def test_iv_continuous_instrument_estimates_its_conditional_variance(data):
    rng = np.random.default_rng(302)
    Zc = data["Z"] + rng.normal(size=N)
    with warnings.catch_warnings():
        # The noisy instrument is weak in places; that warning is not the
        # subject of this test.
        warnings.simplefilter("ignore", UserWarning)
        fit = sp.iv_forest(
            y=data["Y"], treat=data["W"], instrument=Zc, covariates=data["X"], **KW
        )
    assert fit.detail["instrument"] == "continuous"
    assert "regression forest" in fit.detail["var_z_source"]
    assert np.all(fit._z_var > 0)
    g = (Zc - fit._z_hat) / (fit._z_var * fit._compliance)
    np.testing.assert_allclose(
        fit.get_scores(),
        fit.cate + g * (data["Y"] - fit._y_hat - fit.cate * (data["W"] - fit._w_hat)),
        atol=ATOL,
    )


def test_rows_with_missing_values_are_dropped_not_imputed(data, iv_plain):
    """A fit on data with missing rows is the fit on the complete rows."""
    Y = data["Y"].copy()
    X = data["X"].copy()
    Y[:5] = np.nan
    X[5:9, 1] = np.nan
    keep = np.ones(N, dtype=bool)
    keep[:9] = False
    holes = sp.iv_forest(y=Y, treat=data["W"], instrument=data["Z"], covariates=X, **KW)
    clean = sp.iv_forest(
        y=data["Y"][keep],
        treat=data["W"][keep],
        instrument=data["Z"][keep],
        covariates=data["X"][keep],
        **KW,
    )
    assert holes.n_obs == N - 9 and holes.detail["n_dropped_missing"] == 9
    np.testing.assert_array_equal(holes.cate, clean.cate)
    assert holes.late == clean.late and holes.se == clean.se
    # Nuisances may be given for the input rows or for the kept rows.
    y_hat = np.linspace(0, 1, N)
    full = sp.iv_forest(
        y=Y, treat=data["W"], instrument=data["Z"], covariates=X, Y_hat=y_hat, **KW
    )
    short = sp.iv_forest(
        y=Y,
        treat=data["W"],
        instrument=data["Z"],
        covariates=X,
        Y_hat=y_hat[keep],
        **KW,
    )
    np.testing.assert_array_equal(full._y_hat, y_hat[keep])
    np.testing.assert_array_equal(full.cate, short.cate)
    assert full.detail["nuisance_source"]["Y_hat"] == "user-supplied"
    # A missing weight or cluster id drops the row too.
    sw = data["sw"].copy()
    sw[:3] = np.nan
    assert (
        sp.iv_forest(
            y=data["Y"],
            treat=data["W"],
            instrument=data["Z"],
            covariates=data["X"],
            weights=sw,
            **KW,
        ).n_obs
        == N - 3
    )


def test_iv_named_columns_and_frame_prediction(data):
    frame = pd.DataFrame(data["X"], columns=["a", "b", "c"]).assign(
        y=data["Y"], w=data["W"], z=data["Z"], cl=data["cl"], sw=data["sw"]
    )
    by_name = sp.iv_forest(
        frame,
        y="y",
        treat="w",
        instrument="z",
        covariates=["a", "b", "c"],
        clusters="cl",
        weights="sw",
        **KW,
    )
    by_array = sp.iv_forest(
        y=data["Y"],
        treat=data["W"],
        instrument=data["Z"],
        covariates=data["X"],
        clusters=data["cl"],
        weights=data["sw"],
        **KW,
    )
    np.testing.assert_array_equal(by_name.cate, by_array.cate)
    assert by_name.feature_names == ["a", "b", "c"]
    assert by_array.feature_names == ["covariates1", "covariates2", "covariates3"]
    new = frame[["c", "a", "y", "b"]].head(4)
    np.testing.assert_array_equal(
        by_name.predict(new)["predictions"],
        by_name.predict(frame[["a", "b", "c"]].to_numpy()[:4])["predictions"],
    )
    with pytest.raises(MethodIncompatibility, match=r"lacks columns \['c'\]"):
        by_name.predict(frame[["a", "b"]].head(2))
    assert by_name.split_frequencies(2).shape == (2, 3)
    imp = sp.variable_importance(by_name)
    assert list(imp.index) == ["a", "b", "c"] and imp.sum() == pytest.approx(1.0)
    assert sp.instrumental_forest.__doc__ == sp.iv_forest.__doc__


@pytest.mark.parametrize(
    "override, exc, match",
    [
        (dict(instrument=np.ones(N)), DataInsufficient, "no variation"),
        (dict(weights=np.zeros(N)), MethodIncompatibility, "weights must be"),
        (dict(weights=-np.ones(N)), MethodIncompatibility, "weights must be"),
        (dict(clusters=np.zeros(N)), DataInsufficient, "two clusters"),
        (dict(reduced_form_weight=1.5), MethodIncompatibility, "reduced_form"),
        (dict(alpha=1.5), MethodIncompatibility, "alpha"),
        (dict(alpha="x"), MethodIncompatibility, "alpha"),
        (dict(n_estimators=0), MethodIncompatibility, "n_estimators"),
        (dict(min_samples_leaf=0), MethodIncompatibility, "min_samples_leaf"),
        (dict(max_samples=0.0), MethodIncompatibility, "max_samples"),
        (dict(max_samples=0.8), MethodIncompatibility, "max_samples"),
        (dict(honesty_fraction=1.0), MethodIncompatibility, "honesty_fraction"),
        (dict(split_alpha=0.3), MethodIncompatibility, "split_alpha"),
        (dict(imbalance_penalty=-1), MethodIncompatibility, "imbalance_penalty"),
        (dict(ci_group_size=0), MethodIncompatibility, "ci_group_size"),
        (dict(Y_hat=np.zeros(7)), MethodIncompatibility, "Y_hat must have shape"),
        (dict(Z_hat=np.full(N, np.nan)), MethodIncompatibility, "non-finite"),
        (dict(compliance_score=np.ones(3)), MethodIncompatibility, "compliance"),
        (dict(covariates=None), MethodIncompatibility, "covariates= is required"),
        (dict(covariates=[]), MethodIncompatibility, "must not be empty"),
        (dict(covariates=np.zeros((N, 2, 2))), MethodIncompatibility, "matrix"),
        (dict(y=np.zeros(N - 1)), MethodIncompatibility, "different numbers"),
        (dict(y=np.zeros((N, 2))), MethodIncompatibility, "single column"),
        (dict(data=np.zeros((N, 2))), MethodIncompatibility, "pandas DataFrame"),
    ],
)
def test_iv_forest_refuses_unusable_inputs(data, override, exc, match):
    args = dict(
        y=data["Y"], treat=data["W"], instrument=data["Z"], covariates=data["X"]
    )
    args.update(KW)
    args.update(override)
    with pytest.raises(exc, match=match):
        sp.iv_forest(**args)


def test_iv_forest_remaining_input_rules(data):
    args = dict(
        y=data["Y"], treat=data["W"], instrument=data["Z"], covariates=data["X"]
    )
    with pytest.raises(MethodIncompatibility, match="cannot be combined"):
        sp.iv_forest(
            weights=data["sw"],
            clusters=data["cl"],
            equalize_cluster_weights=True,
            **args,
            **KW,
        )
    with pytest.raises(DataInsufficient, match="fewer than 3 complete rows"):
        sp.iv_forest(
            y=data["Y"][:2],
            treat=data["W"][:2],
            instrument=data["Z"][:2],
            covariates=data["X"][:2],
        )
    frame = pd.DataFrame({"y": data["Y"]})
    with pytest.raises(MethodIncompatibility, match=r"Missing columns: \['nope'\]"):
        sp.iv_forest(frame, y="nope", treat=data["W"], instrument=data["Z"])
    with pytest.raises(MethodIncompatibility, match=r"Missing columns: \['nope'\]"):
        sp.iv_forest(
            frame,
            y="y",
            treat=data["W"],
            instrument=data["Z"],
            covariates=data["X"],
            clusters="nope",
        )
    with pytest.warns(DeprecationWarning, match="n_bootstrap"):
        old = sp.iv_forest(n_bootstrap=50, **args, **KW)
    assert old.detail["nuisance_source"]["Z_hat"] == "regression forest (OOB)"
    with pytest.warns(UserWarning, match="near zero"):
        weak = sp.iv_forest(compliance_score=0.01, **args, **KW)
    assert weak.detail["compliance_score_range"] == (0.01, 0.01)
    assert weak.detail["nuisance_source"]["compliance_score"] == "user-supplied"


def test_prediction_rows_are_validated(data, iv_plain):
    with pytest.raises(MethodIncompatibility, match="3 covariate columns"):
        iv_plain.predict(np.zeros((2, 5)))
    with pytest.raises(MethodIncompatibility, match="non-finite"):
        iv_plain.predict(np.array([[0.0, np.nan, 1.0]]))
    with pytest.raises(MethodIncompatibility, match="one row per training"):
        iv_plain.best_linear_projection(A=data["X"][:10])
    with pytest.raises(MethodIncompatibility, match="unsupported vcov_type"):
        iv_plain.best_linear_projection(vce="HC9")
    with pytest.raises(MethodIncompatibility, match="alpha"):
        iv_plain.average_treatment_effect(alpha=0.0)
    var = iv_plain.predict(data["new"], estimate_variance=True)
    assert list(var.columns) == ["predictions", "variance_estimates"]
    assert np.all(var["variance_estimates"] > 0)
    assert iv_plain.predict(estimate_variance=True).shape == (N, 2)


def test_equalized_cluster_weights_reach_the_family_averages():
    """Clusters of unequal size with ``equalize_cluster_weights``: each row
    counts ``1 / (size of its cluster)`` in the average effect."""
    rng = np.random.default_rng(305)
    sizes = rng.integers(2, 20, 50)
    cl = np.repeat(np.arange(50), sizes)
    n = cl.size
    X = rng.normal(size=(n, 3))
    Z = rng.binomial(1, 0.5, n).astype(float)
    W = (Z + rng.normal(size=n) > 0.6).astype(float)
    # The effect grows with cluster size, so the weighting moves the mean.
    Y = X[:, 1] + (1 + X[:, 0] + 0.1 * sizes[cl]) * W + rng.normal(size=n)
    w = 1.0 / sizes[cl]
    iv = sp.iv_forest(
        y=Y,
        treat=W,
        instrument=Z,
        covariates=X,
        clusters=cl,
        equalize_cluster_weights=True,
        **KW,
    )
    est, se = _mean_se(iv.get_scores(), w, cl)
    assert iv.late == pytest.approx(est, abs=ATOL)
    assert iv.se == pytest.approx(se, abs=ATOL)
    assert abs(est - iv.get_scores().mean()) > 1e-3
    arm = rng.integers(0, 3, n)
    ma = sp.multi_arm_forest(
        y=Y,
        treat=arm,
        covariates=X,
        clusters=cl,
        equalize_cluster_weights=True,
        **KW,
    )
    for j, k in enumerate((1, 2)):
        est, se = _mean_se(ma.get_scores()[:, j], w, cl)
        assert ma.ate[k] == pytest.approx(est, abs=ATOL)
        assert ma.ate_se[k] == pytest.approx(se, abs=ATOL)
    # Without clusters the option has nothing to equalize.
    plain = sp.regression_forest(y=Y, covariates=X, n_estimators=20)
    same = sp.regression_forest(
        y=Y, covariates=X, n_estimators=20, equalize_cluster_weights=True
    )
    np.testing.assert_array_equal(plain.predictions, same.predictions)


# --------------------------------------------------------------------------- #
#  Little bags switched off
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def no_bags(data):
    kw = dict(n_estimators=40, random_state=3, ci_group_size=1)
    return {
        "iv_forest": sp.iv_forest(
            y=data["Y"],
            treat=data["W"],
            instrument=data["Z"],
            covariates=data["X"],
            **kw,
        ),
        "multi_arm_forest": sp.multi_arm_forest(
            y=data["Y3"], treat=data["arm"], covariates=data["X"], **kw
        ),
        "lm_forest": sp.lm_forest(
            y=data["Y"], regressors=data["W"], covariates=data["X"], **kw
        ),
        "regression_forest": sp.regression_forest(
            y=data["Y"], covariates=data["X"], **kw
        ),
        "probability_forest": sp.probability_forest(
            y=data["arm"], covariates=data["X"], **kw
        ),
    }


@pytest.mark.parametrize(
    "name",
    [
        "iv_forest",
        "multi_arm_forest",
        "lm_forest",
        "regression_forest",
        "probability_forest",
    ],
)
def test_training_row_variance_is_refused_without_little_bags(no_bags, name):
    with pytest.raises(MethodIncompatibility, match="ci_group_size >= 2"):
        no_bags[name].predict(estimate_variance=True)


@pytest.mark.parametrize(
    "name",
    [
        "iv_forest",
        "multi_arm_forest",
        "lm_forest",
        "regression_forest",
        "probability_forest",
    ],
)
def test_new_row_variance_is_refused_without_little_bags(no_bags, data, name):
    with pytest.raises(MethodIncompatibility, match="ci_group_size >= 2"):
        no_bags[name].predict(data["new"], estimate_variance=True)


def test_no_bags_fixture_still_estimates_average_effects(no_bags):
    assert np.isfinite(no_bags["iv_forest"].se)
    assert no_bags["iv_forest"].cate_variance is None
    assert no_bags["multi_arm_forest"].cate_variance is None
    assert no_bags["lm_forest"].coefficient_variance is None
    assert no_bags["regression_forest"].variance is None


# --------------------------------------------------------------------------- #
#  Multi-arm forest
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def ma_weighted(data):
    return sp.multi_arm_forest(
        y=data["Y3"],
        treat=data["arm"],
        covariates=data["X"],
        clusters=data["cl"],
        weights=data["sw"],
        **KW,
    )


def _ma_scores(Y, arm, m, e, tau):
    mu0 = m - np.sum(e[:, 1:] * tau, axis=1)
    mu = np.column_stack([mu0, mu0[:, None] + tau])
    resid = Y - mu[np.arange(len(Y)), arm]
    return np.column_stack(
        [
            tau[:, k - 1] + ((arm == k) / e[:, k] - (arm == 0) / e[:, 0]) * resid
            for k in range(1, e.shape[1])
        ]
    )


def test_multi_arm_prediction_is_the_local_regression_on_arm_indicators(
    data, ma_weighted
):
    fit = ma_weighted
    e = fit.propensities.to_numpy()
    onehot = np.eye(3)[data["arm"]]
    hand = np.array(
        [
            _local_solve(a, data["Y3"] - fit._y_hat, onehot[:, 1:] - e[:, 1:])
            for a in fit.forest_weights(data["new"]) * data["sw"]
        ]
    )
    pred = fit.predict(data["new"])
    assert list(pred.columns) == ["1 - 0", "2 - 0"]
    np.testing.assert_allclose(pred.to_numpy(), hand, atol=ATOL)
    with_var = fit.predict(data["new"], estimate_variance=True)
    assert list(with_var.columns)[2:] == ["variance[1 - 0]", "variance[2 - 0]"]
    assert (with_var.iloc[:, 2:] > 0).all().all()
    oob = fit.predict(estimate_variance=True)
    np.testing.assert_array_equal(oob["1 - 0"], fit.cate[1])
    np.testing.assert_array_equal(oob["variance[2 - 0]"], fit.cate_variance[2])


def test_multi_arm_scores_and_averages(data, ma_weighted):
    fit = ma_weighted
    Y, arm, sw, cl = data["Y3"], data["arm"], data["sw"], data["cl"]
    e = fit.propensities.to_numpy()
    np.testing.assert_allclose(e.sum(axis=1), 1.0, atol=1e-12)
    tau = fit.predict().to_numpy()
    scores = _ma_scores(Y, arm, fit._y_hat, e, tau)
    np.testing.assert_allclose(fit.get_scores(), scores, atol=ATOL)
    np.testing.assert_allclose(sp.get_scores(fit), scores, atol=ATOL)
    np.testing.assert_allclose(
        multi_arm_scores(Y, arm, fit._y_hat, e, tau), scores, atol=ATOL
    )
    table = fit.average_treatment_effect(alpha=0.1)
    for j, k in enumerate((1, 2)):
        est, se = _mean_se(scores[:, j], sw, cl)
        assert fit.ate[k] == pytest.approx(est, abs=ATOL)
        assert fit.ate_se[k] == pytest.approx(se, abs=ATOL)
        z = stats.norm.ppf(0.975)
        assert fit.ci[k] == pytest.approx((est - z * se, est + z * se), abs=ATOL)
        assert fit.pvalue[k] == pytest.approx(2 * stats.norm.sf(abs(est / se)))
        row = table.loc[f"{k} - 0"]
        assert row["ci_high"] == pytest.approx(
            est + stats.norm.ppf(0.95) * se, abs=ATOL
        )
    assert fit.detail["arm_counts"] == {str(k): int(np.sum(arm == k)) for k in range(3)}
    assert "1 vs 0" in fit.summary() and repr(fit).startswith("MultiArmForestResult")


def test_multi_arm_blp_stacks_one_projection_per_contrast(data, ma_weighted):
    fit = ma_weighted
    scores = fit.get_scores()
    w = data["sw"] / data["sw"].sum()
    D = np.column_stack([np.ones(N), data["X"][:, 0]])
    tab = fit.best_linear_projection(A=data["X"][:, 0], vce="HC1")
    assert tab.index.names == ["contrast", "term"]
    for j, label in enumerate(("1 - 0", "2 - 0")):
        beta = np.linalg.solve(D.T @ (D * w[:, None]), D.T @ (w * scores[:, j]))
        V = _cluster_hc1(D, scores[:, j] - D @ beta, w, data["cl"])
        np.testing.assert_allclose(tab.loc[label, "coef"], beta, atol=ATOL)
        np.testing.assert_allclose(tab.loc[label, "se"], np.sqrt(np.diag(V)), atol=ATOL)
    pd.testing.assert_frame_equal(
        sp.best_linear_projection(fit, A=data["X"][:, 0], vce="HC1"), tab
    )


def test_two_arms_reduce_to_the_binary_aipw_score(data):
    """With two arms and supplied nuisances the multi-arm score is the
    binary AIPW score of the same nuisances and predictions."""
    T = (data["arm"] > 0).astype(int)
    e = np.full(N, 0.6)
    m = 0.2 * data["X"][:, 1]
    fit = sp.multi_arm_forest(
        y=data["Y3"],
        treat=T,
        covariates=data["X"],
        W_hat=np.column_stack([1 - e, e]),
        Y_hat=m,
        **KW,
    )
    tau = fit.cate[1]
    Y = data["Y3"]
    aipw = tau + (T - e) / (e * (1 - e)) * (Y - m - (T - e) * tau)
    np.testing.assert_allclose(fit.get_scores()[:, 0], aipw, atol=ATOL)
    assert fit.detail["nuisance_source"] == {
        "Y_hat": "user-supplied",
        "W_hat": "user-supplied",
    }
    # The prediction is the local regression of Y - m on T - e.
    hand = [
        _local_solve(a, Y - m, (T - e).astype(float))[0]
        for a in fit.forest_weights(data["new"])
    ]
    np.testing.assert_allclose(fit.predict(data["new"])["1 - 0"], hand, atol=ATOL)


def test_reference_arm_and_labels(data):
    names = np.array(["ctrl", "low", "high"])[data["arm"]]
    fit = sp.multi_arm_forest(
        y=data["Y3"], treat=names, covariates=data["X"], reference="low", **KW
    )
    assert fit.arms == ["low", "ctrl", "high"] and fit.reference == "low"
    assert fit.contrasts == ["ctrl", "high"]
    assert list(fit.predict().columns) == ["ctrl - low", "high - low"]
    assert list(fit.propensities.columns) == ["low", "ctrl", "high"]
    # Arm codes follow the reordering: code 0 is the reference.
    np.testing.assert_array_equal(fit._arm_codes == 0, names == "low")
    with pytest.raises(MethodIncompatibility, match="not a treatment value"):
        sp.multi_arm_forest(
            y=data["Y3"], treat=names, covariates=data["X"], reference="none"
        )
    frame = pd.DataFrame(data["X"], columns=list("abc")).assign(y=data["Y3"], t=names)
    by_name = sp.multi_arm_forest(
        frame, y="y", treat="t", covariates=list("abc"), reference="low", **KW
    )
    np.testing.assert_array_equal(by_name.cate["high"], fit.cate["high"])
    with pytest.raises(MethodIncompatibility, match=r"Missing columns: \['nope'\]"):
        sp.multi_arm_forest(frame, y="y", treat="nope", covariates=list("abc"))


def test_propensity_bounds_clip_the_scores_only(data):
    bounds = (0.3, 0.36)
    fit = sp.multi_arm_forest(
        y=data["Y3"],
        treat=data["arm"],
        covariates=data["X"],
        propensity_bounds=bounds,
        **KW,
    )
    free = sp.multi_arm_forest(
        y=data["Y3"], treat=data["arm"], covariates=data["X"], **KW
    )
    # The forest is grown on the raw propensities, so predictions agree.
    np.testing.assert_array_equal(fit.cate[1], free.cate[1])
    raw = fit.propensities.to_numpy()
    assert raw.min() < bounds[0] and raw.max() > bounds[1]  # the bounds bind
    tau = fit.predict().to_numpy()
    clipped = np.clip(raw, *bounds)
    np.testing.assert_allclose(
        fit.get_scores(),
        _ma_scores(data["Y3"], data["arm"], fit._y_hat, clipped, tau),
        atol=ATOL,
    )
    assert fit.detail["propensity_bounds"] == bounds
    for bad in ((0.9, 0.1), (0.0, 0.5), (0.1, 0.5, 0.9)):
        with pytest.raises(MethodIncompatibility, match="propensity_bounds"):
            sp.multi_arm_forest(
                y=data["Y3"],
                treat=data["arm"],
                covariates=data["X"],
                propensity_bounds=bad,
            )


def test_multi_arm_refuses_unusable_inputs(data):
    args = dict(y=data["Y3"], treat=data["arm"], covariates=data["X"], **KW)
    with pytest.raises(DataInsufficient, match="at least two treatment arms"):
        sp.multi_arm_forest(**{**args, "treat": np.zeros(N)})
    with pytest.raises(MethodIncompatibility, match="y must be one column"):
        sp.multi_arm_forest(**{**args, "y": np.zeros((N, 2))})
    with pytest.raises(MethodIncompatibility, match="W_hat must have shape"):
        sp.multi_arm_forest(W_hat=np.full((N, 2), 0.5), **args)
    zero = np.column_stack([np.zeros(N), np.full((N, 2), 0.5)])
    with pytest.raises(DataInsufficient, match="propensities are 0"):
        sp.multi_arm_forest(W_hat=zero, **args)
    thin = np.column_stack([np.full(N, 0.005), np.full(N, 0.5), np.full(N, 0.495)])
    with pytest.warns(UserWarning, match="smallest estimated arm propensity"):
        sp.multi_arm_forest(W_hat=thin, **args)
    # A missing arm label drops the row.
    arm = data["arm"].astype(float)
    arm[:6] = np.nan
    fit = sp.multi_arm_forest(**{**args, "treat": arm})
    assert fit.n_obs == N - 6 and fit.detail["n_dropped_missing"] == 6
    # An arm whose rows are all incomplete is reported, not silently lost.
    y = data["Y3"].copy()
    y[data["arm"] == 2] = np.nan
    with pytest.raises(DataInsufficient, match="no complete rows"):
        sp.multi_arm_forest(**{**args, "y": y})


# --------------------------------------------------------------------------- #
#  Linear-model forest
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("weighted", [False, True])
def test_lm_forest_prediction_is_the_local_regression(data, weighted):
    R = np.column_stack([data["W"], data["X"][:, 2]])
    Xc = data["X"][:, :2]
    fit = sp.lm_forest(
        y=data["Y"],
        regressors=R,
        covariates=Xc,
        weights=data["sw"] if weighted else None,
        **KW,
    )
    w = data["sw"] if weighted else np.ones(N)
    new = data["new"][:, :2]
    hand = np.array(
        [
            _local_solve(a, (fit._Y - fit._y_hat)[:, 0], fit._W - fit._w_hat)
            for a in fit.forest_weights(new) * w
        ]
    )
    out = fit.predict(new, estimate_variance=True)
    assert out["predictions"].shape == (6, 2, 1)
    np.testing.assert_allclose(out["predictions"][:, :, 0], hand, atol=ATOL)
    assert np.all(out["variance_estimates"] > 0)
    oob = fit.predict()
    assert oob["predictions"] is fit.coefficients
    assert fit.regressor_names == ["regressors1", "regressors2"]
    assert fit.outcome_names == ["y"]
    assert "h[regressors1 -> y]" in fit.summary()
    assert repr(fit) == f"LMForestResult(K=2, q=1, n={N})"
    assert sp.variable_importance(fit).sum() == pytest.approx(1.0)


def test_lm_forest_with_supplied_nuisances_and_two_outcomes(data):
    Y2 = np.column_stack([data["Y"], data["Y3"]])
    fit = sp.lm_forest(
        y=Y2,
        regressors=data["W"],
        covariates=data["X"],
        Y_hat=np.zeros((N, 2)),
        W_hat=0.5,
        **KW,
    )
    assert fit.coefficients.shape == (N, 1, 2)
    assert fit.detail["nuisance_source"] == {
        "Y_hat": "user-supplied",
        "W_hat": "user-supplied",
    }
    # Each outcome's coefficient is its own local regression on W - 0.5.
    A = fit.forest_weights(data["new"])
    for q in range(2):
        hand = [_local_solve(a, Y2[:, q], data["W"] - 0.5)[0] for a in A]
        np.testing.assert_allclose(
            fit.predict(data["new"])["predictions"][:, 0, q], hand, atol=ATOL
        )
    with pytest.raises(MethodIncompatibility, match="Y_hat must have shape"):
        sp.lm_forest(
            y=Y2, regressors=data["W"], covariates=data["X"], Y_hat=np.zeros(N), **KW
        )


# --------------------------------------------------------------------------- #
#  Prediction forests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("weighted", [False, True])
def test_regression_forest_is_the_weighted_mean_under_forest_weights(data, weighted):
    w = data["sw"] if weighted else np.ones(N)
    fit = sp.regression_forest(
        y=data["Y"],
        covariates=data["X"],
        weights=data["sw"] if weighted else None,
        **KW,
    )
    A = fit.forest_weights(data["new"])
    np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        fit.predict(data["new"])["predictions"],
        (A * w) @ data["Y"] / (A * w).sum(axis=1),
        atol=ATOL,
    )
    Ao = fit.forest_weights()
    # Out-of-bag weights put nothing on the row itself.
    assert np.all(np.diag(Ao) == 0)
    np.testing.assert_allclose(
        fit.predictions, (Ao * w) @ data["Y"] / (Ao * w).sum(axis=1), atol=ATOL
    )
    assert fit.forest_type == "regression_forest" and fit.num_trees == 100
    assert "regression_forest (GRF engine)" in fit.summary()
    both = fit.predict(data["new"], estimate_variance=True)
    assert np.all(both["variance_estimates"] > 0)
    assert fit.predict(estimate_variance=True)["variance_estimates"].shape == (N,)


@pytest.mark.parametrize("weighted", [False, True])
def test_multi_regression_and_probability_forests(data, weighted):
    w = data["sw"] if weighted else np.ones(N)
    wts = data["sw"] if weighted else None
    Y2 = np.column_stack([data["Y"], data["Y3"]])
    mr = sp.multi_regression_forest(y=Y2, covariates=data["X"], weights=wts, **KW)
    A = mr.forest_weights(data["new"]) * w
    np.testing.assert_allclose(
        mr.predict(data["new"]).to_numpy(), A @ Y2 / A.sum(axis=1)[:, None], atol=ATOL
    )
    assert mr.outcome_names == ["y1", "y2"] and mr.variance is None

    labels = np.array(["lo", "mid", "hi"])[data["arm"]]
    pf = sp.probability_forest(y=labels, covariates=data["X"], weights=wts, **KW)
    assert pf.classes == ["hi", "lo", "mid"]
    onehot = np.column_stack([(labels == c).astype(float) for c in pf.classes])
    A = pf.forest_weights(data["new"]) * w
    probs = pf.predict(data["new"])
    assert list(probs.columns) == ["hi", "lo", "mid"]
    np.testing.assert_allclose(
        probs.to_numpy(), A @ onehot / A.sum(axis=1)[:, None], atol=ATOL
    )
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    with_var = pf.predict(data["new"], estimate_variance=True)
    assert list(with_var.columns)[3:] == ["variance_hi", "variance_lo", "variance_mid"]


@pytest.mark.parametrize("weighted", [False, True])
def test_quantile_forest_is_the_weighted_quantile_under_forest_weights(data, weighted):
    w = data["sw"] if weighted else np.ones(N)
    qs = (0.9, 0.1, 0.5)  # deliberately unsorted
    fit = sp.quantile_forest(
        y=data["Y"],
        covariates=data["X"],
        quantiles=qs,
        weights=data["sw"] if weighted else None,
        **KW,
    )
    order = np.argsort(data["Y"])
    y_sorted = data["Y"][order]

    def wq(alpha, q):
        cdf = np.cumsum(alpha[order]) / alpha.sum()
        return y_sorted[np.searchsorted(cdf, q)]

    A = fit.forest_weights(data["new"]) * w
    pred = fit.predict(data["new"])
    assert list(pred.columns) == ["q0.9", "q0.1", "q0.5"]
    hand = np.array([[wq(a, q) for q in qs] for a in A])
    np.testing.assert_allclose(pred.to_numpy(), hand, atol=ATOL)
    # Other quantiles can be requested after the fit, for new rows and for
    # the training rows (out-of-bag).
    np.testing.assert_allclose(
        fit.predict(data["new"], quantiles=[0.25])["q0.25"],
        [wq(a, 0.25) for a in A],
        atol=ATOL,
    )
    Ao = fit.forest_weights()[:8] * w
    np.testing.assert_allclose(
        fit.predict(quantiles=[0.5])["q0.5"].to_numpy()[:8],
        [wq(a, 0.5) for a in Ao],
        atol=ATOL,
    )
    np.testing.assert_array_equal(
        fit.predict()["q0.5"], fit.predict(quantiles=[0.5])["q0.5"]
    )
    assert fit.quantiles == list(qs)


def test_prediction_forest_argument_rules(data):
    for bad in ([], [0.0, 0.5], [0.5, 1.0], [1.2]):
        with pytest.raises(MethodIncompatibility, match="strictly between 0 and 1"):
            sp.quantile_forest(y=data["Y"], covariates=data["X"], quantiles=bad)
    with pytest.raises(MethodIncompatibility, match="y must be one column"):
        sp.quantile_forest(y=np.zeros((N, 2)), covariates=data["X"])
    with pytest.raises(MethodIncompatibility, match="y must be one column"):
        sp.regression_forest(y=np.zeros((N, 2)), covariates=data["X"])
    with pytest.raises(DataInsufficient, match="at least two classes"):
        sp.probability_forest(y=np.zeros(N), covariates=data["X"])
    frame = pd.DataFrame({"a": data["X"][:, 0], "y": data["Y"]})
    with pytest.raises(MethodIncompatibility, match=r"Missing columns: \['nope'\]"):
        sp.probability_forest(frame, y="nope", covariates=["a"])
    with pytest.raises(MethodIncompatibility, match=r"Missing columns: \['nope'\]"):
        sp.regression_forest(frame, y="y", covariates=["a"], weights="nope")
    fit = sp.regression_forest(frame, y="y", covariates=["a"], n_estimators=20)
    with pytest.raises(MethodIncompatibility, match="quantiles= is for quantile"):
        fit.predict(quantiles=[0.5])
    assert fit.outcome_names == ["y"] and fit.feature_names == ["a"]
    for call in (sp.get_scores, sp.best_linear_projection):
        with pytest.raises(MethodIncompatibility):
            call(fit)
    with pytest.raises(MethodIncompatibility, match="unsupported object"):
        sp.variable_importance(object())
    split = sp.quantile_forest(
        y=data["Y"],
        covariates=data["X"],
        quantiles=(0.5,),
        regression_splitting=True,
        n_estimators=20,
    )
    assert split.predictions.shape == (N, 1)


def test_categorical_covariates_in_the_family(data):
    rng = np.random.default_rng(303)
    frame = pd.DataFrame(
        {"y": data["Y"], "a": data["X"][:, 0], "g": rng.choice(["u", "v"], N)}
    )
    fit = sp.regression_forest(frame, y="y", covariates=["a", "g"], n_estimators=30)
    assert fit.feature_names == ["a", "g[u]", "g[v]"]
    design = np.column_stack([frame["a"], frame["g"] == "u", frame["g"] == "v"])
    np.testing.assert_array_equal(fit._X, design.astype(float))
    np.testing.assert_array_equal(
        fit.predict(frame.head(5))["predictions"],
        fit.predict(design[:5].astype(float))["predictions"],
    )
    unseen = frame.head(3).copy()
    unseen.loc[unseen.index[0], "g"] = "w"
    with pytest.raises(MethodIncompatibility, match="levels that were not"):
        fit.predict(unseen)
    # one_hot_newdata leaves a frame that already has the columns alone.
    ready = pd.DataFrame(design[:2], columns=fit.feature_names)
    assert one_hot_newdata(ready, fit.feature_names) is ready


def test_family_column_interface_accepts_an_explicit_factor_term(data):
    frame = pd.DataFrame({"y": data["Y"], "a": data["X"][:, 0], "k": data["arm"] + 1})
    fit = sp.regression_forest(frame, y="y", covariates=["a", "C(k)"], n_estimators=30)
    assert fit.feature_names == ["a", "k[1]", "k[2]", "k[3]"]


# --------------------------------------------------------------------------- #
#  Survival forest
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def surv(data):
    rng = np.random.default_rng(304)
    # Rounded times give ties, which the Kaplan-Meier product must handle.
    time = np.round(rng.exponential(size=N) * np.exp(0.3 * data["X"][:, 0]), 1) + 0.1
    event = rng.binomial(1, 0.7, N).astype(float)
    return time, event


def _curves(alpha, time, event, grid):
    S = np.ones((alpha.shape[0], grid.size))
    H = np.zeros_like(S)
    for r, a in enumerate(alpha):
        s, h = 1.0, 0.0
        for j, tj in enumerate(grid):
            at_risk = a[time >= tj].sum()
            died = a[(time == tj) & (event == 1)].sum()
            if at_risk > 0:
                s *= 1.0 - died / at_risk
                h += died / at_risk
            S[r, j], H[r, j] = s, h
    return S, H


@pytest.mark.parametrize("weighted", [False, True])
def test_survival_curves_are_forest_weighted_kaplan_meier(data, surv, weighted):
    time, event = surv
    w = data["sw"] if weighted else np.ones(N)
    fit = sp.survival_forest(
        time=time,
        event=event,
        covariates=data["X"],
        weights=data["sw"] if weighted else None,
        n_estimators=60,
        random_state=4,
    )
    grid = fit.failure_times
    np.testing.assert_array_equal(grid, np.unique(time[event == 1]))
    S, H = _curves(fit.forest_weights(data["new"]) * w, time, event, grid)
    km = fit.predict(data["new"])
    np.testing.assert_allclose(km.to_numpy(), S, atol=ATOL)
    assert list(km.columns) == [f"{t:g}" for t in grid]
    na = fit.predict(data["new"], prediction_type="Nelson-Aalen")
    np.testing.assert_allclose(na.to_numpy(), np.exp(-H), atol=ATOL)
    # Curves are right-continuous steps equal to 1 before the first event.
    at = fit.predict(data["new"], failure_times=[0.0, grid[3] + 1e-9, 1e9])
    np.testing.assert_allclose(
        at.to_numpy(), np.column_stack([np.ones(6), S[:, 3], S[:, -1]]), atol=ATOL
    )
    # Training rows: out-of-bag curves, in either form.
    So, Ho = _curves(fit.forest_weights()[:4] * w, time, event, grid)
    np.testing.assert_allclose(fit.predictions[:4], So, atol=ATOL)
    np.testing.assert_allclose(
        fit.predict(prediction_type="na").to_numpy()[:4], np.exp(-Ho), atol=ATOL
    )
    np.testing.assert_array_equal(
        fit.predict(prediction_type="km").to_numpy(), fit.predictions
    )
    assert fit.detail["n_events"] == int(event.sum())
    assert "Kaplan-Meier" in fit.summary() and f"n={N}" in repr(fit)


def test_survival_forest_on_a_user_grid_and_as_nelson_aalen(data, surv):
    time, event = surv
    grid = np.array([0.5, 1.0, 2.0])
    fit = sp.survival_forest(
        time=time,
        event=event,
        covariates=data["X"],
        failure_times=grid,
        prediction_type="nelson_aalen",
        n_estimators=40,
        random_state=4,
    )
    assert fit.prediction_type == "Nelson-Aalen"
    np.testing.assert_array_equal(fit.failure_times, grid)
    curves = fit.predict(data["new"]).to_numpy()
    assert curves.shape == (6, 3)
    assert np.all(np.diff(curves, axis=1) <= 1e-12) and np.all(curves <= 1.0)
    assert sp.variable_importance(fit).shape == (3,)


@pytest.mark.parametrize(
    "override, exc, match",
    [
        (dict(time=-np.ones(N)), MethodIncompatibility, "non-negative"),
        (dict(event=np.full(N, 2.0)), MethodIncompatibility, "binary 0/1"),
        (dict(event=np.zeros(N)), DataInsufficient, "no observed events"),
        (dict(prediction_type="cox"), MethodIncompatibility, "prediction_type"),
        (dict(failure_times=[2.0, 1.0]), MethodIncompatibility, "increasing"),
        (dict(failure_times=[]), MethodIncompatibility, "non-empty"),
        (dict(failure_times=[1.0, np.inf]), MethodIncompatibility, "finite"),
    ],
)
def test_survival_forest_refuses_unusable_inputs(data, surv, override, exc, match):
    time, event = surv
    args = dict(time=time, event=event, covariates=data["X"], n_estimators=20)
    args.update(override)
    with pytest.raises(exc, match=match):
        sp.survival_forest(**args)


# --------------------------------------------------------------------------- #
#  Shared helpers
# --------------------------------------------------------------------------- #


def test_importance_from_a_hand_made_split_table():
    counts = np.array([[8, 2, 0], [3, 3, 6], [0, 0, 0]])
    # Shares within depth, weights 1, 1/4, 1/9; the empty third depth adds
    # nothing to the numerator but stays in the normaliser (docstring).
    wsum = 1 + 1 / 4 + 1 / 9
    hand = (np.array([0.8, 0.2, 0.0]) + np.array([0.25, 0.25, 0.5]) / 4) / wsum
    np.testing.assert_allclose(importance_from_split_frequencies(counts), hand)
    # decay_exponent = 0 weights the depths equally.
    flat = (np.array([0.8, 0.2, 0.0]) + np.array([0.25, 0.25, 0.5])) / 3
    np.testing.assert_allclose(importance_from_split_frequencies(counts, 0.0), flat)


def test_score_average_and_small_helpers():
    scores = np.array([1.0, 2.0, 4.0, 7.0])
    w = np.array([1.0, 1.0, 2.0, 2.0])
    out = score_average(scores, w, None, alpha=0.1)
    est, se = _mean_se(scores, w)
    assert out["estimate"] == pytest.approx(est) and out["se"] == pytest.approx(se)
    assert out["ci_low"] == pytest.approx(est - stats.norm.ppf(0.95) * se)
    assert out["pvalue"] == pytest.approx(2 * stats.norm.sf(abs(est / se)))
    cl = np.array([0, 0, 1, 2])
    out_cl = score_average(scores, w, cl, alpha=0.05)
    assert out_cl["se"] == pytest.approx(_mean_se(scores, w, cl)[1])
    assert np.isnan(normal_pvalue(1.0, 0.0)) and np.isnan(normal_pvalue(1.0, np.inf))
    assert normal_pvalue(1.96, 1.0) == pytest.approx(0.05, abs=1e-4)
    assert names_or_default(["a", "b"], 2, "x") == ["a", "b"]
    assert names_or_default(["a"], 2, "x") == ["x1", "x2"]
    assert names_or_default(None, 1, "x") == ["x1"]
    np.testing.assert_allclose(
        observation_weights(None, False, np.array([1.0, 3.0]), 2), [0.25, 0.75]
    )
    np.testing.assert_allclose(
        observation_weights(np.array([0, 0, 1]), True, None, 3), [0.25, 0.25, 0.5]
    )
    np.testing.assert_allclose(
        observation_weights(np.array([0, 0, 1]), False, None, 3), np.full(3, 1 / 3)
    )


def test_user_nuisance_shapes():
    np.testing.assert_array_equal(
        user_nuisance(0.5, 3, 2, "W_hat", "ctx"), np.full((3, 2), 0.5)
    )
    assert user_nuisance([1.0, 2.0, 3.0], 3, 1, "Y_hat", "ctx").shape == (3, 1)
    with pytest.raises(MethodIncompatibility, match="must have shape"):
        user_nuisance(np.zeros((3, 2)), 3, 1, "Y_hat", "ctx")
    with pytest.raises(MethodIncompatibility, match="non-finite"):
        user_nuisance([1.0, np.nan, 3.0], 3, 1, "Y_hat", "ctx")


def test_auxiliary_forests_get_distinct_seeds():
    base = {"seed": 42, "other": 1}
    seeds = {
        name: with_stream(base, name)["seed"]
        for name in ("Y_hat", "W_hat", "Z_hat", "compliance", "var_z", "var_w")
    }
    assert len(set(seeds.values())) == len(seeds) and 42 not in seeds.values()
    # Neighbouring user seeds do not collide with each other's streams.
    nxt = {name: with_stream({"seed": 43}, name)["seed"] for name in seeds}
    assert set(seeds.values()).isdisjoint(nxt.values())
    assert with_stream(base, "Y_hat")["other"] == 1 and base["seed"] == 42
    assert ForestOptions(random_state=None).seed == 0
