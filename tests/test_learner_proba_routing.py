"""Every estimator that takes a user classifier reads P(class 1) the same way.

``sp.SuperLearner(task='classification').predict_proba`` returns the
``(n,)`` vector of P(Y = 1), not scikit-learn's ``(n, 2)`` array. The DML
family, the metalearners, the causal forest (both engines) and
``sp.conformal_counterfactual`` indexed ``predict_proba(X)[:, 1]`` and
raised IndexError on it; ``sp.dynamic_dml`` called ``predict`` on a
classifier and residualised the treatment on its hard 0/1 label. All of
them now go through ``statspai.core._learners``.

The identity used: a SuperLearner whose library is a single learner puts
weight exactly 1 on it and refits it on the same rows, so every estimate
must equal the one obtained with that learner (rtol 1e-9; the two runs
perform the same floating-point operations, so they agree to the last
bit in practice). The sp.aipw / sp.ipw / sp.ltmle / sp.lmtp cases are in
``test_learner_types_te.py``.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import statspai as sp
from statspai.core._learners import positive_class_proba, proba_or_predict
from statspai.exceptions import MethodIncompatibility


def _expit(z):
    return 1 / (1 + np.exp(-z))


@pytest.fixture(scope="module")
def cs():
    """Cross-section with a binary treatment and a binary instrument."""
    rng = np.random.default_rng(11)
    n = 500
    X = rng.normal(size=(n, 3))
    D = rng.binomial(1, _expit(0.5 * X[:, 0] - 0.4 * X[:, 1]))
    Z = rng.binomial(1, _expit(0.3 * X[:, 2]))
    D_iv = np.where(Z == 1, rng.binomial(1, _expit(0.8 + 0.3 * X[:, 0])), 0)
    Y = 1.0 * D + X[:, 0] + 0.5 * X[:, 2] + rng.normal(size=n)
    Y_iv = 1.0 * D_iv + X[:, 0] + rng.normal(size=n)
    return pd.DataFrame(
        dict(y=Y, d=D, z=Z, d_iv=D_iv, y_iv=Y_iv, x1=X[:, 0], x2=X[:, 1], x3=X[:, 2])
    )


@pytest.fixture(scope="module")
def panel():
    """Long panel, binary treatment in each of three periods."""
    rng = np.random.default_rng(12)
    n_units, T = 300, 3
    rows = []
    for i in range(n_units):
        x = rng.normal()
        d_prev = 0
        for t in range(T):
            x = 0.5 * x + 0.3 * d_prev + rng.normal()
            d = rng.binomial(1, _expit(0.6 * x + 0.4 * d_prev))
            rows.append(dict(i=i, t=t, x=x, d=d))
            d_prev = d
    df = pd.DataFrame(rows)
    last = df.groupby("i")["d"].transform("sum")
    df["y"] = 0.8 * last + df["x"] + rng.normal(size=len(df))
    return df


COV = ["x1", "x2", "x3"]


def _lr():
    return LogisticRegression(C=1e9, max_iter=2000)


def _sl1():
    return sp.SuperLearner(library=[_lr()], n_folds=2, task="classification")


def _cf(d, L, **kw):
    cf = sp.causal_forest(
        y="y", d="d", x=COV, data=d, model_t=L, n_estimators=200, random_state=0, **kw
    )
    return cf.effect(d[COV].to_numpy())


CROSS_SECTION = {
    "dml_irm": lambda d, L: sp.dml(
        d, "y", "d", COV, model="irm", model_d=L, model_y=LinearRegression()
    ),
    "dml_plr": lambda d, L: sp.dml(
        d, "y", "d", COV, model="plr", model_d=L, model_y=LinearRegression()
    ),
    "dml_iivm": lambda d, L: sp.dml(
        d,
        "y_iv",
        "d_iv",
        COV,
        model="iivm",
        instrument="z",
        model_d=L,
        ml_r=_sl1() if isinstance(L, sp.SuperLearner) else _lr(),
        model_y=LinearRegression(),
    ),
    "dml_did": lambda d, L: sp.dml_did(
        d, "y", "d", COV, ml_m=L, ml_g=LinearRegression()
    ),
    "dml_model_averaging": lambda d, L: sp.dml_model_averaging(
        d,
        "y",
        "d",
        COV,
        candidates=[(LinearRegression(), L, "lin+logit")],
        weight_rule="equal",
    ),
    "metalearner_x": lambda d, L: sp.metalearner(
        d, "y", "d", COV, learner="x", propensity_model=L, n_bootstrap=5
    ),
    "metalearner_r": lambda d, L: sp.metalearner(
        d, "y", "d", COV, learner="r", propensity_model=L, n_bootstrap=5
    ),
    "metalearner_dr": lambda d, L: sp.metalearner(
        d, "y", "d", COV, learner="dr", propensity_model=L, n_bootstrap=5
    ),
}


@pytest.mark.parametrize("name", sorted(CROSS_SECTION))
def test_single_learner_superlearner_equals_the_learner(cs, name):
    want = CROSS_SECTION[name](cs, _lr())
    got = CROSS_SECTION[name](cs, _sl1())
    np.testing.assert_allclose(got.estimate, want.estimate, rtol=1e-9, atol=0)
    np.testing.assert_allclose(got.se, want.se, rtol=1e-9, atol=0)
    if name.startswith("metalearner"):
        # the ATE is the shared AIPW score; the learner-specific use of the
        # propensity (X-learner weights, R-learner residual) is in the CATE
        np.testing.assert_allclose(
            got.model_info["cate"], want.model_info["cate"], rtol=1e-9, atol=0
        )


@pytest.mark.parametrize("split_rule", ["grf", "legacy"])
def test_causal_forest_model_t_superlearner(cs, split_rule):
    want = _cf(cs, _lr(), split_rule=split_rule)
    got = _cf(cs, _sl1(), split_rule=split_rule)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=0)


def test_auto_cate_propensity_superlearner(cs):
    kw = dict(learners=("x", "dr"), n_bootstrap=5)
    want = sp.auto_cate(cs, "y", "d", COV, propensity_model=_lr(), **kw)
    got = sp.auto_cate(cs, "y", "d", COV, propensity_model=_sl1(), **kw)
    num = want.leaderboard.select_dtypes("number").columns
    np.testing.assert_allclose(
        got.leaderboard[num].to_numpy(float),
        want.leaderboard[num].to_numpy(float),
        rtol=1e-9,
        atol=0,
    )


def test_conformal_counterfactual_propensity_superlearner(cs):
    want = sp.conformal_counterfactual(
        cs, "y", "d", COV, propensity_model=_lr(), random_state=3
    )
    got = sp.conformal_counterfactual(
        cs, "y", "d", COV, propensity_model=_sl1(), random_state=3
    )
    for attr in ("lower_Y0", "upper_Y0", "lower_Y1", "upper_Y1"):
        np.testing.assert_allclose(
            getattr(got, attr), getattr(want, attr), rtol=1e-9, atol=0
        )


def _dyn(d, L):
    return sp.dynamic_dml(
        d, "y", "d", id="i", time="t", covariates=["x"], model_t=L, n_folds=3
    )


def test_dynamic_dml_classifier_reads_probability_not_label(panel):
    """A classifier as model_t is the conditional mean P(T = 1 | state).

    The equality with SuperLearner (whose ``predict`` is already the
    probability) holds only if the classifier's probability is read; with
    the hard label of ``predict``, the old path, the two disagree.
    """
    want = _dyn(panel, _lr())
    got = _dyn(panel, _sl1())
    num = want.periods.select_dtypes("number").columns
    np.testing.assert_allclose(
        got.periods[num].to_numpy(float),
        want.periods[num].to_numpy(float),
        rtol=1e-9,
        atol=0,
    )
    np.testing.assert_allclose(got.estimate, want.estimate, rtol=1e-9, atol=0)


def test_dynamic_dml_refuses_classifier_on_non_binary_target(panel):
    p = panel.assign(d=panel["d"] * 2.0)  # coded 0 / 2
    with pytest.raises(MethodIncompatibility, match="classifier"):
        _dyn(p, _lr())


# --- the helper itself ----------------------------------------------------


class _Fixed:
    """A fitted 'classifier' returning a fixed predict_proba."""

    def __init__(self, proba, classes=None):
        self._p = np.asarray(proba, dtype=float)
        if classes is not None:
            self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        return self._p


def test_positive_class_proba_column_follows_classes():
    X = np.zeros((3, 1))
    p = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    # classes_ ordered [1, 0]: P(class 1) is the FIRST column
    np.testing.assert_array_equal(
        positive_class_proba(_Fixed(p, [1, 0]), X, context="t"), p[:, 0]
    )
    np.testing.assert_array_equal(
        positive_class_proba(_Fixed(p, [False, True]), X, context="t"), p[:, 1]
    )


def test_positive_class_proba_single_class_fold():
    X = np.zeros((4, 1))
    ones = np.ones((4, 1))
    np.testing.assert_array_equal(
        positive_class_proba(_Fixed(ones, [0]), X, context="t"), np.zeros(4)
    )
    np.testing.assert_array_equal(
        positive_class_proba(_Fixed(ones, [1]), X, context="t"), np.ones(4)
    )
    with pytest.raises(MethodIncompatibility):
        positive_class_proba(_Fixed(ones), X, context="t")  # no classes_


def test_proba_or_predict_regressor_uses_predict():
    X = np.arange(6.0).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0
    reg = LinearRegression().fit(X, y)
    np.testing.assert_allclose(proba_or_predict(reg, X, context="t"), y, rtol=1e-12)


def test_overlap_weighted_did_reads_superlearner_proba():
    """sp.overlap_weighted_did(ps_model=<user classifier>) goes through the
    shared helper: a one-learner SuperLearner equals that learner."""
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(size=n)
    d = rng.binomial(1, _expit(0.8 * x))
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n), 2),
            "post": np.tile([0, 1], n),
            "x": np.repeat(x, 2),
            "d": np.repeat(d, 2),
        }
    )
    df["y"] = df.x + df.post + 2.0 * df.d * df.post + rng.normal(size=2 * n)
    kw = dict(y="y", treat="d", time="post", covariates=["x"], id="id")
    a = sp.overlap_weighted_did(df, ps_model=_lr(), **kw)
    b = sp.overlap_weighted_did(df, ps_model=_sl1(), **kw)
    np.testing.assert_allclose(b.estimate, a.estimate, rtol=1e-9)
