"""Nuisance learners of every documented kind in sp.aipw / sp.ipw /
sp.ltmle / sp.lmtp, and two input checks (sp.bart_causal labels,
GroupSequentialResult.decision non-finite z).

``sp.SuperLearner(task='classification').predict_proba`` returns the
``(n,)`` vector of P(Y = 1), not scikit-learn's ``(n, 2)`` array, and the
class had no ``get_params``; ``ps_learner=`` / ``g_learner=`` raised
IndexError and ``ratio_learner=`` raised sklearn's clone TypeError. The
identity used below: a SuperLearner whose library is a single learner puts
weight 1 on it, so it must give that learner's estimate.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

import statspai as sp
from statspai.core._learners import positive_class_proba
from statspai.exceptions import MethodIncompatibility


def _expit(z):
    return 1 / (1 + np.exp(-z))


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(1)
    n = 600
    L0 = rng.normal(size=n)
    A0 = rng.binomial(1, _expit(0.5 * L0))
    L1 = 0.5 * L0 + 0.6 * A0 + rng.normal(size=n)
    A1 = rng.binomial(1, _expit(0.5 * L1 + 0.3 * A0))
    Y = 1 + 0.8 * A0 + 1.2 * A1 + L1 + 0.5 * L0 + rng.normal(size=n)
    return pd.DataFrame(dict(L0=L0, A0=A0, L1=L1, A1=A1, Y=Y, W=rng.normal(size=n)))


def _lr():
    return LogisticRegression(C=1e9, max_iter=2000)


def _sl1():
    return sp.SuperLearner(library=[_lr()], n_folds=2, task="classification")


CALLS = {
    "aipw": lambda d, L: sp.aipw(
        d, "Y", "A1", ["L0", "L1", "A0"], ps_learner=L
    ).estimate,
    "ipw": lambda d, L: sp.ipw(
        d,
        "Y",
        "A1",
        ["L0", "L1", "A0"],
        ps_learner=L,
        se_method="bootstrap",
        n_bootstrap=3,
        seed=7,
    ).estimate,
    "ltmle": lambda d, L: sp.ltmle(
        d, "Y", ["A0", "A1"], [["L0"], ["L1"]], g_learner=L
    ).ate,
    "lmtp": lambda d, L: sp.lmtp(
        d,
        "Y",
        ["A0", "A1"],
        shift=1.0,
        covariates_time=[[], ["L1"]],
        baseline=["W"],
        ratio_learner=L,
    ).estimate,
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_single_learner_superlearner_equals_the_learner(df, name):
    want = CALLS[name](df, _lr())
    got = CALLS[name](df, _sl1())
    np.testing.assert_allclose(got, want, rtol=1e-9)


@pytest.mark.parametrize("name", ["aipw", "ltmle", "lmtp"])
def test_one_dimensional_predict_proba_learner(df, name):
    class OneD:  # no get_params: deep-copied, P(Y = 1) as a vector
        def fit(self, X, y, sample_weight=None):
            self.m = _lr().fit(X, y)
            return self

        def predict_proba(self, X):
            return self.m.predict_proba(X)[:, 1]

    np.testing.assert_allclose(
        CALLS[name](df, OneD()), CALLS[name](df, _lr()), rtol=1e-12
    )


def test_superlearner_is_sklearn_cloneable():
    sl = sp.SuperLearner(
        library=[_lr()], n_folds=3, task="classification", random_state=5
    )
    c = clone(sl)
    assert c is not sl and c.get_params()["n_folds"] == 3
    assert c.get_params()["random_state"] == 5 and c._estimator_type == "classifier"
    with pytest.raises(MethodIncompatibility, match="unknown parameter"):
        sl.set_params(bogus=1)


def test_positive_class_column_follows_classes_():
    class Rev:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            p = np.linspace(0.1, 0.9, len(X))
            return np.column_stack([p, 1 - p])  # column 0 is class 1

    X = np.zeros((5, 1))
    np.testing.assert_array_equal(
        positive_class_proba(Rev(), X, context="t"), np.linspace(0.1, 0.9, 5)
    )

    class Three:
        def predict_proba(self, X):
            return np.full((len(X), 3), 1 / 3)

    with pytest.raises(MethodIncompatibility, match="shape"):
        positive_class_proba(Three(), X, context="t")


def test_bart_causal_label_treatment_is_refused(df):
    d = df.assign(arm=df["A1"].map({0: "ctl", 1: "trt"}))
    with pytest.raises(MethodIncompatibility, match="holds labels"):
        sp.bart_causal(d, "Y", "arm", ["L0"], draws=10, burnin=10, chains=1, seed=1)
    kw = dict(draws=10, burnin=10, chains=1, seed=1)
    a = sp.bart_causal(df.assign(t=df["A1"].astype(bool)), "Y", "t", ["L0"], **kw)
    b = sp.bart_causal(df, "Y", "A1", ["L0"], **kw)
    assert a.estimate == b.estimate


@pytest.mark.parametrize("z", [[np.nan], [np.inf], [5.0, np.nan], ["a"]])
def test_group_sequential_decision_refuses_non_finite_z(z):
    g = sp.group_sequential(3, efficacy="obf", side="upper")
    with pytest.raises(MethodIncompatibility):
        g.decision(z)
    # a finite z below the first efficacy bound continues
    assert g.decision([1.0])["action"].tolist() == ["continue"]
