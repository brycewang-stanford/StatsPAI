"""sp.RlassoClassifier as a propensity learner.

``sp.ipw(ps_learner=sp.RlassoClassifier())`` returned -1.23 on the design
below (-2.19 on another draw) whose ATE is 1.2, where logit, boosting and
the logistic rigorous Lasso gave 1.07 to 1.24. The classifier
is not wrong: its fit equals ``hdm::rlasso`` on the 0/1 label (pinned
below to R), its columns are [P(0), P(1)] and ``sp.ipw`` reads column 1.
It is a linear-probability model. For rows with extreme covariates the
linear prediction leaves [0, 1] and is clipped to ``clip = 1e-3``; a
treated row clipped there gets inverse weight 1000 instead of the ~30 its
true propensity implies, and two or three such rows carry the estimate.
``hdm``'s own treatment-effect functions fit propensities with
``rlassologit`` (``sp.RlassologitClassifier``), which recovers the ATE.

Fix: ``predict_proba`` warns whenever the clip binds, and the three
rigorous-Lasso learners refuse ``sample_weight`` instead of ignoring it.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning, MethodIncompatibility


def _det_fixture():
    """Deterministic (RNG-free) fixture, rebuilt identically in R."""
    n, p = 300, 8
    i = np.arange(1, n + 1)
    X = np.column_stack(
        [np.sin(0.7 * i * j) + np.cos(1.3 * i + j) for j in range(1, p + 1)]
    )
    u = np.mod(np.sin(i * 12.9898) * 43758.5453, 1.0)
    lin = 1.2 * X[:, 0] - 0.9 * X[:, 1] + 0.5 * X[:, 2]
    d = (u < 1 / (1 + np.exp(-lin))).astype(float)
    return X, d


def test_rlasso_classifier_equals_hdm_rlasso_on_the_label():
    """hdm 0.3.2 (MIT), R 4.5:
    n <- 300; p <- 8; i <- 1:n
    X <- sapply(1:p, function(j) sin(0.7*i*j) + cos(1.3*i + j))
    u <- (sin(i*12.9898)*43758.5453) %% 1
    d <- as.numeric(u < plogis(1.2*X[,1] - 0.9*X[,2] + 0.5*X[,3]))
    f <- rlasso(X, d, post=TRUE, intercept=TRUE)
    f$intercept; f$beta; sum(predict(f, newdata=X) < 1e-3)   # 3
    """
    X, d = _det_fixture()
    with pytest.warns(AssumptionWarning, match="3 of 300 rows") as rec:
        clf = sp.RlassoClassifier().fit(X, d)
        proba = clf.predict_proba(X)
    want_coef = np.array([0.224245581480628, -0.138404820695657, 0, 0, 0, 0, 0, 0])
    # same algorithm, same data: agreement to rounding of the printed digits
    np.testing.assert_allclose(clf.intercept_, 0.48107777587404, rtol=1e-12)
    np.testing.assert_allclose(clf.coef_, want_coef, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(clf.classes_, [0.0, 1.0])
    raw = clf.intercept_ + X @ clf.coef_
    np.testing.assert_allclose(proba[:, 1], np.clip(raw, 1e-3, 1 - 1e-3), rtol=1e-12)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-15)
    diag = rec[0].message.diagnostics
    assert diag["n_below"] == int(np.sum(raw < 1e-3)) == 3
    assert diag["n_above"] == 0


def test_rlassologit_classifier_equals_hdm_rlassologit():
    """Same fixture; g <- rlassologit(X, d, post=TRUE, intercept=TRUE); g$beta."""
    X, d = _det_fixture()
    clf = sp.RlassologitClassifier().fit(X, d)
    want = np.array([1.35734055908854, -0.991172666601475, 0.643185483323035])
    # glmnet path vs the port's coordinate descent: agreement to ~1e-9
    np.testing.assert_allclose(clf.coef_[:3], want, rtol=1e-8)
    np.testing.assert_array_equal(clf.coef_[3:], 0.0)


@pytest.fixture(scope="module")
def ate_design():
    rng = np.random.default_rng(0)
    n, p = 2000, 10
    X = rng.normal(size=(n, p))
    e = 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2])))
    D = rng.binomial(1, e)
    Y = 1.2 * D + X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(p)])
    df["y"], df["d"] = Y, D
    return df, [f"x{j}" for j in range(p)]


def test_ipw_with_rlassologit_recovers_the_ate(ate_design):
    df, cov = ate_design
    r = sp.ipw(
        df, "y", "d", cov, ps_learner=sp.RlassologitClassifier(), seed=1, n_bootstrap=30
    )
    # true ATE 1.2; the estimate sits within 3 bootstrap SE of it
    assert abs(r.estimate - 1.2) < 3 * r.se, (r.estimate, r.se)


def test_ipw_with_rlasso_lpm_warns_at_the_source(ate_design):
    df, cov = ate_design
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sp.ipw(
            df, "y", "d", cov, ps_learner=sp.RlassoClassifier(), seed=1, n_bootstrap=2
        )
    msgs = [str(w.message) for w in rec if issubclass(w.category, AssumptionWarning)]
    assert any(m.startswith("RlassoClassifier: the linear-probability") for m in msgs)
    # sp.ipw's own overlap diagnostic fires as well
    assert any("effective sample" in m for m in msgs)


def test_no_warning_when_the_clip_does_not_bind():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 2))
    d = (rng.uniform(size=400) < 0.4 + 0.1 * np.tanh(X[:, 0])).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        p1 = sp.RlassoClassifier().fit(X, d).predict_proba(X)[:, 1]
    assert p1.min() > 1e-3 and p1.max() < 1 - 1e-3


@pytest.mark.parametrize(
    "cls", [sp.RlassoRegressor, sp.RlassoClassifier, sp.RlassologitClassifier]
)
def test_rigorous_lasso_learners_refuse_sample_weight(cls):
    X, d = _det_fixture()
    with pytest.raises(TypeError, match="sample_weight"):
        cls().fit(X, d, sample_weight=np.ones(len(d)))
    cls().fit(X, d, sample_weight=None)  # the unweighted call is unchanged


def test_superlearner_refuses_rlasso_with_weights():
    X, d = _det_fixture()
    sl = sp.SuperLearner(
        library=[sp.RlassologitClassifier()], n_folds=2, task="classification"
    )
    with pytest.raises(MethodIncompatibility, match="sample_weight"):
        sl.fit(X, d, sample_weight=np.ones(len(d)))
