"""Correctness and boundary tests for ``sp.fmm`` (reference comparisons with
Stata ``fmm`` and R ``flexmix`` are in ``tests/reference_parity``)."""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special

import statspai as sp
from statspai.exceptions import ColumnNotFound
from statspai.regression import _fmm_core
from statspai.regression.fmm import _check_starts

_FIX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"


@pytest.fixture(scope="module")
def gauss2() -> pd.DataFrame:
    return pd.read_csv(_FIX / "fmm_gauss2.csv")


@pytest.fixture(scope="module")
def fit2(gauss2):
    return sp.fmm("y ~ x1 + x2", gauss2, 2, class_covariates="z", seed=11)


# ---------------------------------------------------------------------------
# k = 1 is the one-class model: closed forms
# ---------------------------------------------------------------------------


def test_one_class_gaussian_is_ols_with_ml_variance(gauss2):
    res = sp.fmm("y ~ x1 + x2", gauss2, 1)
    X = np.column_stack([np.ones(len(gauss2)), gauss2["x1"], gauss2["x2"]])
    y = gauss2["y"].to_numpy()
    b = np.linalg.solve(X.T @ X, X.T @ y)
    s2 = float(np.sum((y - X @ b) ** 2) / len(y))
    names = ["y:1.Class#_cons", "y:1.Class#x1", "y:1.Class#x2"]
    np.testing.assert_allclose(res.params[names], b, rtol=1e-10)
    np.testing.assert_allclose(res.params["/var(e.y)#1.Class"], s2, rtol=1e-10)
    # OIM: s2 (X'X)^-1 for the coefficients, 2 s2^2 / n for the variance
    np.testing.assert_allclose(
        res.std_errors[names], np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))), rtol=1e-8
    )
    np.testing.assert_allclose(
        res.std_errors["/var(e.y)#1.Class"], np.sqrt(2 * s2**2 / len(y)), rtol=1e-8
    )
    ll = -0.5 * len(y) * (np.log(2 * np.pi * s2) + 1)
    assert abs(res.loglik - ll) <= 1e-10 * abs(ll)
    np.testing.assert_allclose(res.class_probs["estimate"], [1.0], rtol=1e-12)


def test_one_class_logit_and_poisson_are_the_glm_mle():
    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(size=n)
    d = pd.DataFrame(
        {
            "x": x,
            "b": (rng.uniform(size=n) < special.expit(0.3 + 0.8 * x)).astype(int),
            "c": rng.poisson(np.exp(0.5 + 0.4 * x)),
        }
    )
    X = np.column_stack([np.ones(n), x])
    for fam, col in (("logit", "b"), ("poisson", "c")):
        res = sp.fmm(f"{col} ~ x", d, 1, family=fam)
        bhat = res.params[[f"{col}:1.Class#_cons", f"{col}:1.Class#x"]].to_numpy()
        eta = X @ bhat
        mu = special.expit(eta) if fam == "logit" else np.exp(eta)
        # score equations of the GLM: X'(y - mu) = 0
        assert np.max(np.abs(X.T @ (d[col].to_numpy() - mu))) < 1e-8
        v = mu * (1 - mu) if fam == "logit" else mu
        se = np.sqrt(np.diag(np.linalg.inv((X.T * v) @ X)))
        np.testing.assert_allclose(res.std_errors.to_numpy(), se, rtol=1e-8)


# ---------------------------------------------------------------------------
# invariances
# ---------------------------------------------------------------------------


def test_frequency_weights_equal_duplicated_rows(gauss2):
    sub = gauss2.iloc[:300].copy()
    w = sp.fmm("y ~ x1 + x2", sub, 2, class_covariates="z", weights="fw", seed=1)
    dup = sub.loc[sub.index.repeat(sub["fw"])].reset_index(drop=True)
    u = sp.fmm("y ~ x1 + x2", dup, 2, class_covariates="z", seed=1)
    np.testing.assert_allclose(w.params, u.params, rtol=1e-8)
    np.testing.assert_allclose(w.std_errors, u.std_errors, rtol=1e-7)
    assert abs(w.loglik - u.loglik) <= 1e-10 * abs(u.loglik)
    assert w.n_ic == len(dup) and abs(w.bic - u.bic) < 1e-7


def test_seed_and_row_order_do_not_change_the_estimate(gauss2, fit2):
    other = sp.fmm("y ~ x1 + x2", gauss2, 2, class_covariates="z", seed=12345)
    shuffled = gauss2.sample(frac=1.0, random_state=3)
    perm = sp.fmm("y ~ x1 + x2", shuffled, 2, class_covariates="z", seed=99)
    for r in (other, perm):
        np.testing.assert_allclose(r.params, fit2.params, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(r.std_errors, fit2.std_errors, rtol=1e-7)
    # posteriors follow the rows, by label
    np.testing.assert_allclose(
        perm.posterior.loc[gauss2.index].to_numpy(),
        fit2.posterior.to_numpy(),
        atol=1e-8,
    )


def test_class_order_rules(gauss2, fit2):
    means = fit2.class_means["estimate"].to_numpy()
    assert np.all(np.diff(means) > 0)
    # y -> -y reverses the class means, not the shares: under the default
    # rule the classes swap (and the base with them), under 'share' not
    neg = gauss2.assign(y=-gauss2["y"])
    by_mean = sp.fmm("y ~ x1 + x2", neg, 2, class_covariates="z", seed=11)
    by_share = sp.fmm(
        "y ~ x1 + x2", neg, 2, class_covariates="z", class_order="share", seed=11
    )
    assert np.all(np.diff(by_share.class_probs["estimate"].to_numpy()) < 0)
    for r in (by_mean, by_share):
        assert abs(r.loglik - fit2.loglik) < 1e-9 * abs(fit2.loglik)
    mem = ["2.Class:z", "2.Class:_cons"]
    np.testing.assert_allclose(
        by_mean.params[mem], -fit2.params[mem].to_numpy(), rtol=1e-7
    )
    np.testing.assert_allclose(
        by_share.params[mem], fit2.params[mem].to_numpy(), rtol=1e-7
    )
    for a, b in ((1, 2), (2, 1)):
        np.testing.assert_allclose(
            by_mean.params[f"y:{a}.Class#x1"],
            -fit2.params[f"y:{b}.Class#x1"],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            by_share.params[f"y:{a}.Class#x1"],
            -fit2.params[f"y:{a}.Class#x1"],
            rtol=1e-7,
        )


def test_posterior_is_bayes_rule_and_shares_average_the_prior(fit2):
    post, prior = fit2.posterior.to_numpy(), fit2.prior.to_numpy()
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        prior.mean(axis=0), fit2.class_probs["estimate"], rtol=1e-12
    )
    # at the MLE the posterior and prior shares agree (score of the constant)
    np.testing.assert_allclose(post.mean(axis=0), prior.mean(axis=0), atol=1e-9)
    np.testing.assert_allclose(
        fit2.fitted.mean(axis=0), fit2.class_means["estimate"], rtol=1e-12
    )


def test_predict_newdata_reproduces_the_sample(gauss2, fit2):
    for kind in ("mu", "classpr", "classposteriorpr", "marginal"):
        a = fit2.predict(type=kind)
        b = fit2.predict(gauss2, type=kind)
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12, atol=1e-14)
    m = fit2.predict(type="marginal")["marginal"].to_numpy()
    np.testing.assert_allclose(
        m, (fit2.prior.to_numpy() * fit2.fitted.to_numpy()).sum(axis=1)
    )
    with pytest.raises(ColumnNotFound):
        fit2.predict(gauss2.drop(columns="y"), type="classposteriorpr")
    pr = fit2.predict(gauss2.drop(columns="y").head(5), type="classpr")
    np.testing.assert_allclose(pr.to_numpy(), fit2.prior.to_numpy()[:5], rtol=1e-12)
    with pytest.raises(sp.MethodIncompatibility):
        fit2.predict(type="density")


def test_missing_rows_dropped_and_duplicate_index_kept(gauss2):
    d = gauss2.copy()
    d.index = np.repeat(np.arange(len(d) // 2), 2)  # duplicated labels
    d.loc[d.index[:3], "z"] = np.nan
    d.iloc[5, d.columns.get_loc("x1")] = np.nan
    res = sp.fmm("y ~ x1 + x2", d, 2, class_covariates="z", seed=1)
    assert res.n_obs == len(d) - 5  # labels 0 and 1 (two rows each), plus one row
    assert len(res.posterior) == res.n_obs


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def test_tidy_summary_and_tables(fit2):
    t = fit2.tidy()
    assert len(t) == fit2.df_model == len(fit2.params)
    assert set(t["equation"]) == {"class", "outcome", "variance"}
    var = t[t["equation"] == "variance"]
    assert var["p_value"].isna().all() and (var["conf_low"] > 0).all()
    np.testing.assert_allclose(var["conf_low"] * var["conf_high"], var["estimate"] ** 2)
    tab = fit2.coef_table(2)
    assert list(tab.index) == ["_cons", "x1", "x2", "var(e.y)"]
    assert "Class membership" in fit2.summary() and "estat" not in repr(fit2)
    assert fit2.cite()  # verified bib keys
    with pytest.raises(KeyError):
        fit2.coef_table(3)


def test_robust_and_cluster_sandwich_identity(gauss2):
    """Cluster sandwich with one row per cluster is the robust sandwich up
    to the small-sample factor (both N/(N-1) here)."""
    d = gauss2.iloc[:400].assign(row=np.arange(400))
    r = sp.fmm("y ~ x1 + x2", d, 2, vce="robust", seed=1)
    c = sp.fmm("y ~ x1 + x2", d, 2, cluster="row", seed=1)
    np.testing.assert_allclose(r.std_errors, c.std_errors, rtol=1e-10)
    assert c.model_info["n_clusters"] == 400


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------


def test_input_validation(gauss2):
    f = "y ~ x1 + x2"
    with pytest.raises(sp.MethodIncompatibility):
        sp.fmm(f, gauss2, 2, family="probit")
    with pytest.raises(sp.MethodIncompatibility):
        sp.fmm(f, gauss2, 0)
    with pytest.raises(sp.MethodIncompatibility):
        sp.fmm(f, gauss2, 2, family="logit")  # y is not 0/1
    with pytest.raises(sp.MethodIncompatibility):
        sp.fmm(f, gauss2.assign(y=-gauss2["fw"]), 2, family="poisson")
    with pytest.raises(sp.MethodIncompatibility):
        sp.fmm(f, gauss2, 2, vce="cluster")
    with pytest.raises(sp.MethodIncompatibility):
        sp.fmm(f, gauss2, 2, weights="fw", weights_as="sampling", vce="oim")
    with pytest.raises(sp.MethodIncompatibility):
        sp.fmm(f, gauss2, 2, weights_as="sampling")  # no weights
    with pytest.raises(ColumnNotFound):
        sp.fmm(f, gauss2, 2, class_covariates=["zz"])
    with pytest.raises(sp.DataInsufficient):
        sp.fmm(f, gauss2.head(8), 2)
    with pytest.raises(sp.DataInsufficient):
        sp.fmm(f, gauss2.iloc[:300].assign(g=1), 2, cluster="g")
    with pytest.raises(sp.IdentificationFailure):
        sp.fmm(f, gauss2.assign(z2=2 * gauss2["z"]), 2, class_covariates=["z", "z2"])


def test_noiseless_classes_make_the_likelihood_unbounded():
    """Two exact lines: every class variance collapses to zero, where the
    Gaussian mixture likelihood is unbounded. sp.fmm must refuse, not
    report a variance of 1e-30."""
    x = np.linspace(-2, 2, 120)
    y = np.where(np.arange(120) % 2 == 0, 1 + 2 * x, 5 - x)
    d = pd.DataFrame({"y": y, "x": x})
    with pytest.raises(sp.NumericalInstability):
        sp.fmm("y ~ x", d, 2, n_starts=4, seed=0)


def test_em_flags_a_collapsing_class():
    rng = np.random.default_rng(1)
    n = 200
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = X @ [1.0, 1.0] + rng.normal(size=n)
    y[:2] = X[:2] @ [9.0, 0.0]  # a class can sit exactly on two points
    model = _fmm_core.MixtureModel(y, X, np.ones((n, 1)), 2, "gaussian", np.ones(n))
    R = np.zeros((n, 2))
    R[:, 0] = 1.0
    R[:2] = [0.0, 1.0]
    with pytest.raises(sp.NumericalInstability):
        _fmm_core.em(model, R, 50, 1e-10)


def test_unreplicated_maximum_warns():
    starts = [{"ll": -100.0}, {"ll": -120.0}, {"ll": -120.0}]
    with pytest.warns(sp.ConvergenceWarning, match="not replicated"):
        _check_starts(starts, -100.0, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_starts(starts + [{"ll": -100.0 + 1e-9}], -100.0, 4)
        _check_starts(starts[:1], -100.0, 1)


def test_score_and_hessian_are_exact(gauss2):
    """The analytic score equals the complex-step derivative of the
    per-observation log-likelihood to rounding, and the complex-step
    Hessian equals a central difference of the total score."""
    from statspai.regression._optim_helpers import complex_step_scores

    d = gauss2.iloc[:200]
    X = np.column_stack([np.ones(200), d["x1"]])
    Z = np.column_stack([d["z"], np.ones(200)])
    for fam, y in (
        ("gaussian", d["y"].to_numpy()),
        ("logit", (d["y"] > 3).astype(float).to_numpy()),
        ("poisson", d["fw"].to_numpy(dtype=float)),
    ):
        m = _fmm_core.MixtureModel(y, X, Z, 2, fam, np.ones(200))
        theta = np.linspace(-0.4, 0.5, m.size)
        np.testing.assert_allclose(
            m.scores(theta),
            complex_step_scores(m.obs_loglik, theta),
            rtol=1e-10,
            atol=1e-12,
        )
        h = 1e-6
        fd = np.column_stack(
            [
                (m.total_score(theta + h * e) - m.total_score(theta - h * e)) / (2 * h)
                for e in np.eye(m.size)
            ]
        )
        np.testing.assert_allclose(m.hessian(theta), fd, rtol=1e-6, atol=1e-6)
