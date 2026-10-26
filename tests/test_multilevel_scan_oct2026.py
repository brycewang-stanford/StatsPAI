"""Option scan (Oct 2026) of ``statspai.multilevel``.

Each test states the correct behaviour for an option combination that
returned a wrong number, crashed or went unvalidated before the scan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _three_level(seed: int = 3) -> pd.DataFrame:
    """Schools > classes > pupils, class ids unique across schools."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(30):
        u_s = rng.normal(0, 0.7)
        for c in range(4):
            u_c = rng.normal(0, 0.5)
            for _ in range(8):
                x = rng.normal()
                y = 1.0 + 0.5 * x + u_s + u_c + rng.normal(0, 0.5)
                rows.append({"school": s, "klass": s * 10 + c, "x": x, "y": y})
    return pd.DataFrame(rows)


def _two_level(
    seed: int = 0, n_groups: int = 40, n_per: int = 15, sigma_u: float = 1.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), n_per)
    x = rng.normal(size=g.size)
    y = 2.0 + 0.5 * x + rng.normal(0, sigma_u, n_groups)[g] + rng.normal(0, 0.5, g.size)
    return pd.DataFrame({"y": y, "x": x, "g": g})


def _ordinal(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(30), 12)
    x = rng.normal(size=g.size)
    latent = 0.9 * x + rng.normal(0, 0.6, 30)[g] + rng.logistic(size=g.size)
    y = np.ones(g.size, dtype=int)
    y[latent > -1.0] = 2
    y[latent > 1.0] = 3
    return pd.DataFrame({"y": y, "x": x, "gid": g})


def _binary(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(25), 12)
    x = rng.normal(size=g.size)
    p = 1.0 / (1.0 + np.exp(-(-0.3 + 0.8 * x + rng.normal(0, 0.7, 25)[g])))
    return pd.DataFrame(
        {"y": (rng.uniform(size=g.size) < p).astype(float), "x": x, "g": g}
    )


# ---------------------------------------------------------------------------
# M1 / M2: three-level BLUPs and predictions
# ---------------------------------------------------------------------------


def test_three_level_predictions_do_not_depend_on_how_classes_are_numbered():
    uniq = _three_level()
    rep = uniq.assign(klass=uniq["klass"] % 10)  # class 0..3 in every school
    r_u = sp.mixed(uniq, "y", ["x"], group=["school", "klass"])
    r_r = sp.mixed(rep, "y", ["x"], group=["school", "klass"])
    # The likelihood already works on (school, class) pairs.
    np.testing.assert_allclose(
        r_r.fixed_effects.to_numpy(), r_u.fixed_effects.to_numpy(), rtol=1e-8
    )
    np.testing.assert_allclose(
        r_r.predict(rep).to_numpy(), r_u.predict(uniq).to_numpy(), rtol=1e-8
    )
    assert r_r.random_effects.index.is_unique
    assert len(r_r.random_effects) == 30 * 4


def test_three_level_predict_without_data_is_the_conditional_prediction():
    df = _three_level()
    r = sp.mixed(df, "y", ["x"], group=["school", "klass"])
    np.testing.assert_allclose(
        r.predict().to_numpy(), r.predict(df).to_numpy(), rtol=1e-10
    )


# ---------------------------------------------------------------------------
# M3: LR test against the linear model
# ---------------------------------------------------------------------------


# Stata 18 `mixed y x || g:, ml` / `, reml` on _two_level(seed=4, n_per=5,
# sigma_u=0.2): e(chi2_c), e(p_c). The ML statistic agrees to the fits'
# convergence (3e-7); Stata's p is chibar2(01), half the chi2(1) tail.
_STATA_LR = {
    "ml": (15.3955044846924, 0.000043597744994749),
    "reml": (15.901683212677, 0.000033359677152748),
}


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_lr_test_vs_linear_model_matches_stata_chibar(method):
    df = _two_level(seed=4, n_per=5, sigma_u=0.2)
    r = sp.mixed(df, "y", ["x"], "g", method=method)
    chi2, p = _STATA_LR[method]
    assert r._lr_test["chi2"] == pytest.approx(chi2, rel=1e-6)
    assert r._lr_test["p"] == pytest.approx(p, rel=1e-5)
    assert r._lr_test["distribution"] == "chibar2(01)"
    # identity: p is half the chi2(1) tail of the reported statistic
    assert r._lr_test["p"] == pytest.approx(
        0.5 * stats.chi2.sf(r._lr_test["chi2"], 1), rel=1e-12
    )


def test_lr_test_with_two_variance_parameters_is_plain_chi2():
    df = _two_level(seed=4, n_per=5, sigma_u=0.2)
    r = sp.mixed(df, "y", ["x"], "g", x_random=["x"], cov_type="diagonal", method="ml")
    lr = r._lr_test
    assert lr["distribution"] == "chi2(2)"
    # Stata `mixed y x || g: x, ml`: chi2(2) = 16.2865520871234
    assert lr["chi2"] == pytest.approx(16.2865520871234, rel=1e-5)
    assert lr["p"] == pytest.approx(stats.chi2.sf(lr["chi2"], 2), rel=1e-12)


# ---------------------------------------------------------------------------
# M4: predict() index
# ---------------------------------------------------------------------------


def test_predict_without_data_carries_the_estimation_rows_labels():
    df = _two_level()
    df.index = df.index + 1000
    df.loc[1003, "x"] = np.nan  # dropped from the fit
    r = sp.mixed(df, "y", ["x"], "g")
    p = r.predict()
    used = df.dropna(subset=["y", "x", "g"])
    assert p.index.equals(used.index)
    np.testing.assert_allclose(p.to_numpy(), r.predict(used).to_numpy(), rtol=1e-12)


# ---------------------------------------------------------------------------
# M5: repeated index labels
# ---------------------------------------------------------------------------


def test_meologit_with_repeated_index_labels_equals_the_reset_index_fit():
    df = _ordinal()
    dup = df.copy()
    dup.index = np.repeat(np.arange(len(df) // 2), 2)
    a = sp.meologit(df, y="y", x_fixed=["x"], group="gid")
    b = sp.meologit(dup, y="y", x_fixed=["x"], group="gid")
    np.testing.assert_array_equal(
        a.fixed_effects.to_numpy(), b.fixed_effects.to_numpy()
    )
    np.testing.assert_array_equal(a.thresholds.to_numpy(), b.thresholds.to_numpy())


def test_mixed_categorical_group_with_repeated_index_labels():
    df = _two_level()
    df["g"] = pd.Categorical(df["g"])
    dup = df.copy()
    dup.index = np.repeat(np.arange(len(df) // 3), 3)
    a = sp.mixed(df, "y", ["x"], "g")
    b = sp.mixed(dup, "y", ["x"], "g")
    np.testing.assert_array_equal(
        a.fixed_effects.to_numpy(), b.fixed_effects.to_numpy()
    )


# ---------------------------------------------------------------------------
# M6: ordinal outcome coding
# ---------------------------------------------------------------------------


def test_meologit_ordered_categorical_equals_numeric_codes():
    df = _ordinal()
    labels = {1: "low", 2: "medium", 3: "high"}
    cat = df.assign(
        y=pd.Categorical(
            df["y"].map(labels), categories=["low", "medium", "high"], ordered=True
        )
    )
    a = sp.meologit(df, y="y", x_fixed=["x"], group="gid")
    b = sp.meologit(cat, y="y", x_fixed=["x"], group="gid")
    np.testing.assert_allclose(
        b.fixed_effects.to_numpy(), a.fixed_effects.to_numpy(), rtol=1e-10
    )
    np.testing.assert_allclose(
        b.thresholds.to_numpy(), a.thresholds.to_numpy(), rtol=1e-10
    )


def test_meologit_refuses_unordered_string_outcome():
    df = _ordinal()
    df["y"] = df["y"].map({1: "low", 2: "medium", 3: "high"})
    with pytest.raises(MethodIncompatibility):
        sp.meologit(df, y="y", x_fixed=["x"], group="gid")


def test_meologit_ignores_unused_categories():
    df = _ordinal()
    a = sp.meologit(df, y="y", x_fixed=["x"], group="gid")
    cat = df.assign(y=pd.Categorical(df["y"], categories=[1, 2, 3, 4], ordered=True))
    b = sp.meologit(cat, y="y", x_fixed=["x"], group="gid")
    np.testing.assert_allclose(
        b.fixed_effects.to_numpy(), a.fixed_effects.to_numpy(), rtol=1e-8
    )
    assert len(b.thresholds) == 2


# ---------------------------------------------------------------------------
# M7 / M8 / M9 / M10: meglm options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["poisson", "gaussian", "gamma", "nbinomial"])
def test_meglm_trials_is_refused_outside_binomial(family):
    rng = np.random.default_rng(1)
    df = _two_level()
    df["y"] = rng.poisson(2.0, len(df)) + 1.0
    df["n"] = 3.0
    with pytest.raises(MethodIncompatibility):
        sp.meglm(df, "y", ["x"], "g", family=family, trials="n")


def test_meglm_binomial_trials_counts_rows_in_n_obs_and_bic():
    rng = np.random.default_rng(2)
    df = _binary()
    df["n"] = 5.0
    df["y"] = rng.binomial(5, 0.4, len(df)).astype(float)
    r = sp.meglm(df, "y", ["x"], "g", family="binomial", trials="n")
    assert r.n_obs == len(df)
    assert r.bic == pytest.approx(r.n_params * np.log(len(df)) - 2 * r.log_likelihood)


def test_meglm_conf_int_uses_the_fit_alpha():
    r = sp.melogit(_binary(), y="y", x_fixed=["x"], group="g", alpha=0.10)
    ci = r.conf_int()
    half = (ci["upper"] - ci["lower"]) / 2
    np.testing.assert_allclose(
        half.to_numpy(), stats.norm.ppf(0.95) * r.bse.to_numpy(), rtol=1e-12
    )


def test_meglm_binomial_rejects_outcome_outside_zero_one():
    df = _binary()
    df["y"] = df["y"] + 1.0  # 1/2 coding
    with pytest.raises((MethodIncompatibility, ValueError)):
        sp.melogit(df, y="y", x_fixed=["x"], group="g")


def test_mepoisson_rejects_negative_counts():
    df = _two_level()
    df["y"] = np.round(df["y"]) - 3.0  # some negative
    assert (df["y"] < 0).any()
    with pytest.raises((MethodIncompatibility, ValueError)):
        sp.mepoisson(df, y="y", x_fixed=["x"], group="g")


# ---------------------------------------------------------------------------
# M11: lrtest sample / method checks
# ---------------------------------------------------------------------------


def test_lrtest_refuses_mixed_fits_on_different_samples():
    df = _two_level()
    small = df.iloc[:-30]
    r = sp.mixed(small, "y", ["x"], "g", method="ml")
    f = sp.mixed(df, "y", ["x"], "g", x_random=["x"], method="ml")
    with pytest.raises((MethodIncompatibility, ValueError)):
        sp.lrtest(r, f)
