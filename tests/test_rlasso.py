"""Unit tests for the rigorous-Lasso module (``statspai.rlasso``).

These are R-free behavioural / edge-case tests (the numerical-parity
contract against ``hdm`` lives in
``tests/reference_parity/test_rlasso_parity.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.rlasso import (
    RlassoClassifier,
    RlassoRegressor,
    rlasso,
    rlasso_effect,
    rlasso_effects,
    rlasso_iv,
)


@pytest.fixture
def sparse_reg():
    """High-dim sparse regression: n=200, p=50, 4 true signals."""
    rng = np.random.default_rng(11)
    n, p = 200, 50
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:4] = [3.0, -2.0, 1.5, 1.0]
    y = X @ beta + rng.normal(size=n)
    return X, y, beta


@pytest.fixture
def iv_dgp():
    """Many-instrument IV: n=300, 25 instruments (3 strong), 10 controls."""
    rng = np.random.default_rng(22)
    n, px, pz = 300, 10, 25
    X = rng.normal(size=(n, px))
    Z = rng.normal(size=(n, pz))
    piz = np.zeros(pz)
    piz[:3] = [1.2, -1.0, 0.8]
    gx = np.zeros(px)
    gx[:3] = [0.7, -0.5, 0.4]
    u = rng.normal(size=n)
    d = Z @ piz + X @ gx + 0.7 * u + rng.normal(size=n)
    beta = 1.5
    y = beta * d + X @ gx + u + rng.normal(size=n)
    return X, Z, d, y, beta


# ─────────────────────────────── public API ───────────────────────────────


def test_public_symbols_exported():
    for name in ["rlasso", "rlasso_effect", "rlasso_iv"]:
        assert hasattr(sp, name), f"sp.{name} not exported"


def test_rlasso_recovers_sparse_support(sparse_reg):
    X, y, beta = sparse_reg
    fit = rlasso(X, y, post=True)
    # the 4 true signals must be selected
    assert set(range(4)).issubset(set(np.where(fit.index)[0]))
    # post-Lasso coefficients close to truth on the support
    np.testing.assert_allclose(fit.beta[:4], beta[:4], atol=0.3)


def test_rlasso_post_vs_lasso_shrinkage(sparse_reg):
    X, y, _ = sparse_reg
    post = rlasso(X, y, post=True)
    lasso = rlasso(X, y, post=False)
    # post-Lasso un-shrinks: larger |coef| on the shared support
    shared = post.index & lasso.index
    assert np.sum(np.abs(post.beta[shared])) >= np.sum(np.abs(lasso.beta[shared]))


def test_rlasso_predict_residual_identity(sparse_reg):
    X, y, _ = sparse_reg
    fit = rlasso(X, y, post=True)
    # residuals == y - predict(X), the relation rlassoIV/rlassoEffect rely on
    np.testing.assert_allclose(fit.residuals, y - fit.predict(X), atol=1e-10)


def test_rlasso_intercept_recovered():
    rng = np.random.default_rng(5)
    n, p = 150, 20
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:3] = [2.0, -1.0, 1.5]
    y = 7.0 + X @ beta + rng.normal(size=n)
    fit = rlasso(X, y, post=True, intercept=True)
    assert abs(fit.intercept - 7.0) < 0.5


def test_rlasso_no_signal_returns_empty_support():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(120, 30))
    y = rng.normal(size=120)  # pure noise, uncorrelated with X
    fit = rlasso(X, y, post=True)
    # almost surely nothing (or very little) selected; never errors
    assert fit.n_selected <= 3
    assert np.isfinite(fit.sigma)


def test_rlasso_dataframe_colnames_flow_through(sparse_reg):
    X, y, _ = sparse_reg
    cols = [f"feat{j}" for j in range(X.shape[1])]
    fit = rlasso(X, y, colnames=cols)
    assert all(s.startswith("feat") for s in fit.selected)
    assert fit.summary().startswith("Rigorous Lasso")


def test_penalty_c_controls_sparsity(sparse_reg):
    X, y, _ = sparse_reg
    loose = rlasso(X, y, penalty={"c": 0.5})
    tight = rlasso(X, y, penalty={"c": 3.0})
    # a bigger slack constant c ⇒ bigger penalty ⇒ fewer selected
    assert tight.n_selected <= loose.n_selected


# ─────────────────────────────── rlasso_effect ─────────────────────────────


def test_rlasso_effect_recovers_partial_effect():
    rng = np.random.default_rng(33)
    n, p = 250, 40
    X = rng.normal(size=(n, p))
    g = np.zeros(p)
    g[:3] = [1.0, -0.8, 0.6]
    d = X @ g + rng.normal(size=n)
    alpha = 2.0
    y = alpha * d + X @ g + rng.normal(size=n)
    for method in ("partialling out", "double selection"):
        res = rlasso_effect(X, y, d, method=method)
        assert abs(res.alpha - alpha) < 0.25, method
        assert res.se > 0
        lo, hi = res.conf_int()
        assert lo < res.alpha < hi


def test_rlasso_effect_invalid_method_raises(sparse_reg):
    X, y, _ = sparse_reg
    with pytest.raises(ValueError, match="partialling out"):
        rlasso_effect(X, y, X[:, 0], method="bogus")


def test_rlasso_effects_multiple_targets(sparse_reg):
    X, y, _ = sparse_reg
    out = rlasso_effects(X[:, :5], y, index=[0, 1], method="partialling out")
    assert set(out.keys()) == {"V1", "V2"}
    for r in out.values():
        assert np.isfinite(r.alpha) and r.se > 0


# ─────────────────────────────── rlasso_iv ─────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(select_Z=True, select_X=False),
        dict(select_Z=False, select_X=True),
        dict(select_Z=True, select_X=True),
        dict(select_Z=False, select_X=False),
    ],
)
def test_rlasso_iv_recovers_effect(iv_dgp, kwargs):
    X, Z, d, y, beta = iv_dgp
    res = rlasso_iv(y=y, d=d, z=Z, x=X, **kwargs)
    assert abs(res.coef[0] - beta) < 0.3, kwargs
    assert res.se[0] > 0
    ci = res.conf_int()
    assert ci[0, 0] < res.coef[0] < ci[0, 1]


def test_rlasso_iv_dataframe_inputs(iv_dgp):
    X, Z, d, y, beta = iv_dgp
    df = pd.DataFrame(
        np.column_stack([y, d, X, Z]),
        columns=["y", "d"]
        + [f"x{j}" for j in range(X.shape[1])]
        + [f"z{j}" for j in range(Z.shape[1])],
    )
    res = rlasso_iv(
        y="y",
        d="d",
        x=[f"x{j}" for j in range(X.shape[1])],
        z=[f"z{j}" for j in range(Z.shape[1])],
        data=df,
        select_Z=True,
        select_X=False,
    )
    assert res.treat_names == ["d"]
    assert abs(res.coef[0] - beta) < 0.3
    assert "Rigorous-Lasso IV" in res.summary()


def test_rlasso_iv_select_x_requires_controls(iv_dgp):
    X, Z, d, y, _ = iv_dgp
    with pytest.raises(ValueError, match="requires controls"):
        rlasso_iv(y=y, d=d, z=Z, x=None, select_Z=False, select_X=True)


def test_rlasso_iv_pvalue_and_tstat(iv_dgp):
    X, Z, d, y, _ = iv_dgp
    res = rlasso_iv(y=y, d=d, z=Z, x=X, select_Z=True, select_X=False)
    np.testing.assert_allclose(res.tstat[0], res.coef[0] / res.se[0])
    assert 0.0 <= res.pvalue[0] <= 1.0


# ────────────────────────── DML nuisance learner ──────────────────────────


def test_rlasso_regressor_sklearn_contract(sparse_reg):
    from sklearn.base import clone

    X, y, _ = sparse_reg
    est = RlassoRegressor(post=True, c=1.1)
    cloned = clone(est)  # DML cross-fitting clones every fold
    assert cloned.get_params() == est.get_params()
    est.fit(X, y)
    assert est.predict(X).shape == (X.shape[0],)
    assert hasattr(est, "coef_") and hasattr(est, "intercept_")


def test_rlasso_classifier_valid_probabilities(sparse_reg):
    X, y, _ = sparse_reg
    d = (y > np.median(y)).astype(int)
    clf = RlassoClassifier().fit(X, d)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.all((proba > 0) & (proba < 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(clf.predict(X))).issubset({0, 1})


def test_dml_accepts_rlasso_alias():
    from statspai.dml._learners import resolve_learner

    assert isinstance(
        resolve_learner("rlasso", kind="regressor", role="ml_g"), RlassoRegressor
    )
    assert isinstance(
        resolve_learner("rlasso", kind="classifier", role="ml_m"), RlassoClassifier
    )


def test_dml_plr_with_rlasso_learner_recovers_effect():
    rng = np.random.default_rng(44)
    n, p = 400, 20
    X = rng.normal(size=(n, p))
    g = np.zeros(p)
    g[:3] = [1.0, -0.7, 0.5]
    m = np.zeros(p)
    m[:3] = [0.8, 0.6, -0.4]
    d = X @ m + rng.normal(size=n)
    theta = 1.5
    y = theta * d + X @ g + rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(p)])
    df["y"] = y
    df["d"] = d
    res = sp.dml(
        data=df,
        y="y",
        treat="d",
        covariates=[f"x{j}" for j in range(p)],
        model="plr",
        ml_g="rlasso",
        ml_m="rlasso",
        n_folds=5,
    )
    assert abs(float(res.estimate) - theta) < 0.25


# ─────────────────── §3 unified result-object contract ────────────────────


def test_result_objects_expose_protocol(sparse_reg, iv_dgp):
    """RLassoFit / RLassoIVResult / RLassoEffectResult honour the agent-native
    result contract (CLAUDE.md §3): to_dict / to_latex / cite."""
    X, y, _ = sparse_reg
    Xc, Z, d, yv, _ = iv_dgp
    objs = [
        rlasso(X, y),
        rlasso_effect(X, y, X[:, 0]),
        rlasso_iv(y=yv, d=d, z=Z, x=Xc, select_Z=True, select_X=False),
    ]
    for r in objs:
        assert isinstance(r.to_dict(), dict) and r.to_dict()
        assert r.to_latex().startswith("\\begin{table}")
        cite = r.cite()
        assert isinstance(cite, str) and cite
        # json form is structured + points at paper.bib
        assert r.cite(format="json")["source"] == "paper.bib"


def test_result_cite_keys_are_in_paper_bib():
    """Zero-hallucination (CLAUDE.md §10): every cite key the rlasso results
    advertise must exist in paper.bib."""
    import pathlib
    import re

    bib = (pathlib.Path(__file__).resolve().parents[1] / "paper.bib").read_text(
        encoding="utf-8"
    )
    have = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE))
    from statspai.rlasso._core import RLassoFit
    from statspai.rlasso.effect import RLassoEffectResult
    from statspai.rlasso.iv import RLassoIVResult

    for cls in (RLassoFit, RLassoEffectResult, RLassoIVResult):
        for key in cls._citation_keys:
            assert key in have, f"{cls.__name__} cites missing bib key {key!r}"


# ───────────────────── sp.iv(method='rlasso') dispatcher ───────────────────


def test_iv_dispatcher_routes_rlasso(iv_dgp):
    """sp.iv(method='rlasso') routes to rlasso_iv and matches a direct call."""
    X, Z, d, y, _ = iv_dgp
    df = pd.DataFrame(
        np.column_stack([y, d, X, Z]),
        columns=["y", "d"]
        + [f"x{j}" for j in range(X.shape[1])]
        + [f"z{j}" for j in range(Z.shape[1])],
    )
    xcols = [f"x{j}" for j in range(X.shape[1])]
    zcols = [f"z{j}" for j in range(Z.shape[1])]

    # no controls -> instrument selection (ergonomic default)
    routed = sp.iv(method="rlasso", y="y", endog="d", instruments=zcols, data=df)
    direct = rlasso_iv(y="y", d="d", z=zcols, data=df, select_Z=True, select_X=False)
    assert type(routed).__name__ == "RLassoIVResult"
    np.testing.assert_allclose(routed.coef[0], direct.coef[0], atol=1e-12)

    # with controls -> hdm double-selection default
    routed_x = sp.iv(
        method="rlasso", y="y", endog="d", instruments=zcols, exog=xcols, data=df
    )
    direct_x = rlasso_iv(
        y="y", d="d", z=zcols, x=xcols, data=df, select_Z=True, select_X=True
    )
    np.testing.assert_allclose(routed_x.coef[0], direct_x.coef[0], atol=1e-12)

    # alias + formula path both resolve
    assert np.isfinite(
        sp.iv(
            method="rigorous_lasso", y="y", endog="d", instruments=zcols, data=df
        ).coef[0]
    )
    fr = sp.iv("y ~ (d ~ " + "+".join(zcols) + ")", data=df, method="rlasso")
    np.testing.assert_allclose(fr.coef[0], routed.coef[0], atol=1e-12)


# ─────────────────────────────── rlassologit ──────────────────────────────


@pytest.fixture
def logit_dgp():
    """Binary outcome, n=400, p=30, 3 true signals."""
    rng = np.random.default_rng(7)
    n, p = 400, 30
    X = rng.standard_normal((n, p))
    b = np.zeros(p)
    b[:3] = [1.8, -1.4, 1.0]
    prob = 1.0 / (1.0 + np.exp(-(X @ b)))
    y = (rng.uniform(size=n) < prob).astype(float)
    return X, y, b


def test_rlassologit_recovers_support(logit_dgp):
    from statspai.rlasso import rlassologit

    X, y, _ = logit_dgp
    fit = rlassologit(X, y, post=True)
    assert set(range(3)).issubset(set(np.where(fit.index)[0]))
    # predict returns valid probabilities + log-odds
    pr = fit.predict(X, type="response")
    assert pr.min() > 0 and pr.max() < 1
    link = fit.predict(X, type="link")
    np.testing.assert_allclose(pr, 1.0 / (1.0 + np.exp(-link)), atol=1e-12)
    assert fit.summary().startswith("Logistic Rigorous Lasso")


def test_rlassologit_result_contract(logit_dgp):
    from statspai.rlasso import rlassologit

    X, y, _ = logit_dgp
    fit = rlassologit(X, y)
    assert isinstance(fit.to_dict(), dict)
    assert fit.to_latex().startswith("\\begin{table}")
    assert fit.cite() == "chernozhukov2016hdm"


def test_rlassologit_predict_type_validation(logit_dgp):
    from statspai.rlasso import rlassologit

    X, y, _ = logit_dgp
    fit = rlassologit(X, y)
    with pytest.raises(ValueError, match="response.*link|link"):
        fit.predict(X, type="bogus")


def test_rlassologit_rejects_non_binary_y(logit_dgp):
    """§7 fail-loudly: continuous / {1,2}-coded / single-class y must raise,
    not slide silently into the IRLS solver and emit overflow garbage."""
    from statspai.rlasso import RlassologitClassifier, rlassologit

    X, _, _ = logit_dgp
    n = X.shape[0]
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="binary 0/1"):
        rlassologit(X, rng.standard_normal(n))  # continuous
    with pytest.raises(ValueError, match="binary 0/1"):
        rlassologit(X, np.where(rng.uniform(size=n) < 0.5, 1.0, 2.0))  # {1, 2}
    with pytest.raises(ValueError, match="both classes"):
        rlassologit(X, np.ones(n))  # single class
    # the classifier wrapper inherits the guard (it delegates to rlassologit)
    with pytest.raises(ValueError, match="binary 0/1"):
        RlassologitClassifier().fit(X, rng.standard_normal(n))


def test_rlassologit_classifier_is_genuine_logistic(logit_dgp):
    from statspai.rlasso import RlassologitClassifier, rlassologit

    X, y, _ = logit_dgp
    clf = RlassologitClassifier().fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    # proba[:,1] is the logistic response of the underlying rlassologit fit
    np.testing.assert_allclose(
        proba[:, 1], rlassologit(X, y).predict(X, type="response"), atol=1e-10
    )
    # clone-safe params
    params = clf.get_params()
    assert RlassologitClassifier(**params).get_params() == params


def test_dml_irm_with_rlassologit_propensity(logit_dgp):
    rng = np.random.default_rng(8)
    n, p = 600, 15
    X = rng.standard_normal((n, p))
    m = np.zeros(p)
    m[:3] = [1.0, -0.8, 0.6]
    ps = 1.0 / (1.0 + np.exp(-(X @ m)))
    d = (rng.uniform(size=n) < ps).astype(int)
    g = np.zeros(p)
    g[:3] = [1.0, 0.5, -0.7]
    y = 0.8 * d + X @ g + rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(p)])
    df["y"] = y
    df["d"] = d
    res = sp.dml(
        data=df,
        y="y",
        treat="d",
        covariates=[f"x{j}" for j in range(p)],
        model="irm",
        ml_g="rlasso",
        ml_m="rlassologit",
        n_folds=5,
    )
    assert abs(float(res.estimate) - 0.8) < 0.35
