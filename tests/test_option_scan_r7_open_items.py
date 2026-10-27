"""Option scan round 7: items left open by round 6.

Each test states the decided behaviour. The references are R
``InvariantCausalPrediction`` 0.8 (run as a black box), Stata 18
``estat gof`` after ``logit [iw=]``, an analytic first-order condition
(NOTEARS) and invariances (row order, global RNG state).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


# --------------------------------------------- 1. regtable shared labels
def _two_models():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = 1 + 2 * df.x1 - df.x2 + rng.normal(size=n)
    m1 = sp.regress("y ~ x1", data=df)
    m2 = sp.regress("y ~ x2", data=df)
    m3 = sp.regress("y ~ x1 + x2", data=df)
    return m1, m2, m3


def _label_rows(text, label):
    return [ln for ln in text.splitlines() if ln.strip().startswith(label)]


def test_coef_labels_sharing_a_label_give_one_row():
    m1, m2, _ = _two_models()
    tab = sp.regtable(m1, m2, coef_labels={"x1": "Treatment", "x2": "Treatment"})
    text = tab.to_text()
    rows = _label_rows(text, "Treatment")
    assert len(rows) == 1
    # each column carries its own model's estimate
    assert f"{m1.params['x1']:.3f}" in rows[0]
    assert f"{m2.params['x2']:.3f}" in rows[0]


def test_coef_map_sharing_a_label_gives_one_row_in_map_order():
    m1, m2, _ = _two_models()
    tab = sp.regtable(
        m1, m2, coef_map={"x1": "Treatment", "x2": "Treatment", "Intercept": "Const"}
    )
    df = tab.to_dataframe()
    labels = [str(i) for i in df.index if str(i).strip()]
    assert labels.count("Treatment") == 1
    assert labels.index("Treatment") < labels.index("Const")


def test_label_equal_to_another_variable_name_merges_like_esttab_rename():
    m1, m2, _ = _two_models()
    text = str(sp.esttab(m1, m2, labels={"x2": "x1"}))
    rows = _label_rows(text, "x1")
    assert len(rows) == 1
    assert f"{m2.params['x2']:.3f}" in rows[0]


def test_shared_label_within_one_model_is_refused():
    m1, _, m3 = _two_models()
    with pytest.raises(MethodIncompatibility) as exc:
        sp.regtable(m1, m3, coef_labels={"x1": "T", "x2": "T"})
    assert exc.value.recovery_hint


def test_shared_label_merges_multi_se_rows():
    m1, m2, _ = _two_models()
    tab = sp.regtable(
        m1,
        m2,
        coef_labels={"x1": "T", "x2": "T"},
        multi_se={"Boot": [{"x1": 0.111}, {"x2": 0.222}]},
    )
    text = tab.to_text()
    assert "[0.111]" in text and "[0.222]" in text
    assert len(_label_rows(text, "T ")) == 1


# ------------------------------------------------ 2. optimal_design et al.
@pytest.mark.parametrize(
    "kw",
    [
        dict(mde=0.2, n=400),
        dict(design="cluster", mde=0.2, icc=0.05, cluster_size=20, n_clusters=40),
        dict(n=400, n_clusters=40),
        dict(design="cluster", n=400, icc=0.05, cluster_size=20),
        dict(mde=0.2, cost_per_cluster=100.0, cost_per_unit=10.0),
        dict(design="cluster", mde=0.2, icc=0.05, cluster_size=20, cost_per_unit=10.0),
    ],
)
def test_optimal_design_refuses_arguments_that_cannot_matter(kw):
    with pytest.raises(MethodIncompatibility) as exc:
        sp.optimal_design(**kw)
    assert exc.value.recovery_hint


def test_optimal_design_baseline_mean_is_deprecated_and_inert():
    ref = sp.optimal_design(mde=0.2)
    with pytest.warns(DeprecationWarning, match="1.43"):
        got = sp.optimal_design(mde=0.2, baseline_mean=5.0)
    assert (got.n_total, got.n_per_arm) == (ref.n_total, ref.n_per_arm)


def _attrition_frame():
    rng = np.random.default_rng(3)
    n = 400
    t = rng.integers(0, 2, n)
    y = 1.0 * t + rng.normal(size=n)
    o = (rng.random(n) < np.where(t == 1, 0.9, 0.75)).astype(int)
    return pd.DataFrame(
        {
            "y": np.where(o == 1, y, np.nan),
            "t": t,
            "o": o,
            "x": rng.normal(size=n),
        }
    )


def test_attrition_bounds_alpha_is_deprecated_and_inert():
    df = _attrition_frame()
    ref = sp.attrition_bounds(df, "y", "t", "o")
    with pytest.warns(DeprecationWarning, match="1.43"):
        got = sp.attrition_bounds(df, "y", "t", "o", alpha=0.1)
    assert got == ref


def test_balance_check_alpha_is_deprecated_and_inert():
    df = _attrition_frame()
    ref = sp.balance_check(df, "t", ["x"])
    with pytest.warns(DeprecationWarning, match="1.43"):
        got = sp.balance_check(df, "t", ["x"], alpha=0.1)
    pd.testing.assert_frame_equal(got.table, ref.table)


# ------------------------------------------------------- 3. ITS time order
def _its_frame():
    rng = np.random.default_rng(4)
    t = np.arange(72)
    y = 2 + 0.05 * t + (t >= 36) * 1.5 + rng.normal(scale=0.4, size=72)
    return pd.DataFrame({"y": y, "month": t + 100})


@pytest.mark.parametrize(
    "kw", [dict(), dict(seasonality_period=12), dict(hac_small_sample=True)]
)
def test_its_shuffled_frame_is_bit_identical_to_sorted(kw):
    df = _its_frame()
    shuffled = df.sample(frac=1.0, random_state=11)
    a = sp.its(df, y="y", time="month", intervention=36, **kw)
    b = sp.its(shuffled, y="y", time="month", intervention=36, **kw)
    assert (a.level_change, a.slope_change) == (b.level_change, b.slope_change)
    assert (a.se_level, a.se_slope) == (b.se_level, b.se_slope)
    pd.testing.assert_frame_equal(a.coefficients, b.coefficients, check_exact=True)


def test_its_repeated_time_point_is_refused():
    df = _its_frame()
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    with pytest.raises(MethodIncompatibility):
        sp.its(df, y="y", time="month", intervention=36)


def test_bayes_its_repeated_time_point_is_refused():
    df = _its_frame()
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    with pytest.raises(MethodIncompatibility):
        sp.bayes_its(df, y="y", time="month", intervention=36)


def test_bayes_its_shuffled_frame_is_bit_identical_to_sorted():
    pytest.importorskip("pymc")
    df = _its_frame()
    kw = dict(y="y", time="month", intervention=36, draws=200, tune=200, chains=1)
    a = sp.bayes_its(df, **kw)
    b = sp.bayes_its(df.sample(frac=1.0, random_state=11), **kw)
    assert (a.posterior_mean, a.hdi_lower, a.hdi_upper) == (
        b.posterior_mean,
        b.hdi_lower,
        b.hdi_upper,
    )


# ---------------------------------------------------------- 4. ICP alpha
def _icp_data():
    rng = np.random.default_rng(7)
    n = 600
    env = np.repeat([1, 2, 3], 200)
    x1 = rng.normal(size=n) + (env == 2) * 1.0 + (env == 3) * 2.0
    x2 = rng.normal(size=n)
    y = 1.5 * x1 + rng.normal(size=n)
    x3 = y + rng.normal(size=n) * (1 + (env == 3))
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3}), y, env


# R InvariantCausalPrediction 0.8, ICP(X, Y, factor(E), alpha = a)$ConfInt
# for X1, on the same data written with DataFrame.to_csv. Both packages
# accept {X1} and {X1, X2}.
@pytest.mark.parametrize(
    "alpha, lo, hi",
    [
        (0.05, 1.38305556524474, 1.54180679450957),
        (0.2, 1.40408208468622, 1.52075877000986),
    ],
)
def test_icp_interval_follows_alpha_as_r_icp(alpha, lo, hi):
    X, y, env = _icp_data()
    res = sp.icp(X, y, env, alpha=alpha)
    assert res.parents == {"X1"}
    # read.csv vs the in-memory doubles: last-bit differences only
    np.testing.assert_allclose(res.coefficients["X1"], (lo, hi), rtol=1e-9)


# ---------------------------------------------------- 5. NOTEARS diagonal
def test_notears_inner_solution_is_stationary_with_zero_diagonal():
    # One inner solve (max_iter=1: rho=1, alpha=0, lambda1=0). With the
    # diagonal held at zero the off-diagonal gradient of
    # 0.5/n ||X - XW||^2 + 0.5 h(W)^2 vanishes at the returned W. Zeroing a
    # free diagonal after the solve left a gradient of 0.279 here.
    from scipy.linalg import expm

    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(size=n)
    b = 0.8 * a + rng.normal(size=n) * 0.5
    c = 0.7 * b + rng.normal(size=n) * 0.5
    d = 0.6 * c + rng.normal(size=n) * 0.5
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
    W = sp.notears(df, lambda1=0.0, max_iter=1, w_threshold=0.0)["adjacency"].to_numpy()
    assert np.all(np.diag(W) == 0.0)
    A = df.to_numpy()
    A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-8)
    E = expm(W * W)
    h = np.trace(E) - 4
    grad = -A.T @ (A - A @ W) / n + h * E.T * 2 * W
    off = ~np.eye(4, dtype=bool)
    assert np.abs(grad[off]).max() < 1e-4


# ------------------------------------------- 6. logit_gof after [iw=]
def _logit_data():
    # the data of tests/test_diagnostics_scan_oct2026.py::_logit_data
    rng = np.random.default_rng(2)
    n = 300
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "c": rng.normal(size=n),
            "w": rng.uniform(0.3, 3, n),
            "g": rng.integers(0, 30, n),
        }
    )
    df["yb"] = (0.3 + df.x1 + rng.logistic(size=n) > 0).astype(int)
    df["x1r"] = np.round(df.x1 / 0.5) * 0.5
    df["cr"] = np.round(df.c)
    return df


# Stata 18: logit yb x1 c [iw=w]; estat gof  (and the rounded covariates)
@pytest.mark.parametrize(
    "formula, chi2, dof, patterns",
    [
        ("yb ~ x1 + c", 491.8320967759776, 297, 300),
        ("yb ~ x1r + cr", 84.785549874785, 49, 52),
    ],
)
def test_logit_gof_after_importance_weights_matches_stata(formula, chi2, dof, patterns):
    res = sp.logit(formula, data=_logit_data(), weights="w")
    out = sp.logit_gof(res)
    # Newton stopping points differ in the last digits of the coefficients
    assert out["statistic"] == pytest.approx(chi2, rel=1e-9)
    assert (out["df"], out["n_patterns"], out["n"]) == (dof, patterns, 300)


def test_logit_gof_after_sampling_weights_is_refused():
    # Stata: "pweights not allowed"
    res = sp.logit(
        "yb ~ x1 + c", data=_logit_data(), weights="w", weights_as="sampling"
    )
    with pytest.raises(MethodIncompatibility):
        sp.logit_gof(res)


def test_stata_estat_gof_after_pweights_is_refused():
    # sp.stata refuses [iw=]; a weighted logit there came from [pw=].
    df = _logit_data()[["yb", "x1", "c", "w"]]
    with pytest.raises(MethodIncompatibility, match="pweights not allowed"):
        sp.stata("logit yb x1 c [pw=w]\nestat gof", data=df)


def test_logit_hosmer_lemeshow_after_weights_is_refused():
    res = sp.logit("yb ~ x1 + c", data=_logit_data(), weights="w")
    with pytest.raises(MethodIncompatibility):
        sp.logit_gof(res, groups=10)


def test_logit_gof_unit_weights_equal_unweighted():
    df = _logit_data().assign(one=1.0)
    a = sp.logit_gof(sp.logit("yb ~ x1r + cr", data=df, weights="one"))
    b = sp.logit_gof(sp.logit("yb ~ x1r + cr", data=df))
    assert a["statistic"] == pytest.approx(b["statistic"], rel=1e-10)


# ------------------------------------------- 7. bayes_dml(mode='full')
def test_bayes_dml_full_mode_is_deprecated_and_refused():
    df = pd.DataFrame({"y": [0.0, 1.0], "d": [0.0, 1.0], "x": [0.0, 1.0]})
    with pytest.warns(DeprecationWarning, match="1.43"):
        with pytest.raises(MethodIncompatibility):
            sp.bayes_dml(df, y="y", treatment="d", covariates=["x"], mode="full")


def test_bayes_dml_schema_no_longer_offers_full():
    params = sp.function_schema("bayes_dml")["parameters"]["properties"]
    assert params["mode"]["enum"] == ["conjugate"]


# --------------------------------------- 8. neural models and global RNG
def _neural_frame():
    rng = np.random.default_rng(5)
    n = 120
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["z"] = rng.normal(size=n)
    df["d"] = (df.x1 + rng.normal(size=n) > 0).astype(int)
    df["dc"] = df.z + df.x1 + rng.normal(size=n)
    df["y"] = df.d + df.x2 + rng.normal(size=n)
    return df


@pytest.mark.parametrize(
    "call",
    [
        lambda df: sp.tarnet(df, y="y", treat="d", covariates=["x1", "x2"],
                             epochs=2, n_bootstrap=0, random_state=1),
        lambda df: sp.cfrnet(df, y="y", treat="d", covariates=["x1", "x2"],
                             epochs=2, n_bootstrap=0, random_state=1),
        lambda df: sp.dragonnet(df, y="y", treat="d", covariates=["x1", "x2"],
                                epochs=2, n_bootstrap=0, random_state=1),
        lambda df: sp.deepiv(df, y="y", treat="dc", instruments=["z"],
                             covariates=["x1"], first_stage_epochs=2,
                             second_stage_epochs=2, random_state=1),
    ],
)  # fmt: skip
def test_neural_fit_leaves_global_numpy_rng_untouched(call):
    torch = pytest.importorskip("torch")
    df = _neural_frame()
    np.random.seed(2026)
    np.random.random(3)
    before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        call(df)
    after = np.random.get_state()
    assert after[0] == before[0]
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2:] == before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
