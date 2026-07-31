"""External parity: ``sp.metalearner`` CATE functions vs ``econml``.

Pins StatsPAI's S-, T- and X-learner conditional-average-treatment-effect
functions against ``econml.metalearners``, the reference implementation
of the Kunzel-Sekhon-Bickel-Yu (2019) meta-learner family.

Why the reference is Python
---------------------------
The meta-learners have no canonical R package to align against -- ``grf``
implements causal forests rather than the S/T/X construction -- so the
cross-package anchor is ``econml``. Before this module ``sp.metalearner``
carried only an ``external-replication`` grade: it reproduced the DGP
truths quoted in the CausalML book's chapters, which certifies that the
estimator is *consistent*, not that it computes the same function as
anyone else.

What is compared, and why it is not the ATE
-------------------------------------------
``sp.metalearner`` reports ``result.estimate`` as a **doubly-robust AIPW
ATE**, deliberately independent of which CATE learner was requested --
``model_info['ate_method'] == 'aipw_dr_pseudo_outcome'`` says so. That is
a design choice, not an oversight, but it means the ATE is the wrong
quantity for this comparison: it does not move when ``learner=`` changes.
The learner-specific object is the CATE vector in
``model_info['cate']``, and that is what these tests pin -- elementwise,
not merely in the mean, since two different CATE functions can share a
mean.

Aligning both model stages
--------------------------
Each meta-learner has two fitting stages: the arm outcome models and (for
X) the imputed-effect models. ``econml``'s ``models=`` / ``cate_models=``
map onto StatsPAI's ``outcome_model=`` / ``cate_model=``. Passing only
``outcome_model`` leaves StatsPAI on its *default* CATE learner, which is
not linear, and the resulting mismatch is a specification difference
rather than an implementation one. Both stages are therefore pinned to
the same closed-form learner on both sides.

Skipped automatically when ``econml`` is not installed -- an optional
pin, not a runtime dependency.

References
----------
[@kunzel2019metalearners]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

econml_metalearners = pytest.importorskip("econml.metalearners")
from sklearn.linear_model import (  # noqa: E402
    LinearRegression,
    LogisticRegression,
)
from sklearn.model_selection import KFold  # noqa: E402

N = 1200
K = 3
COVARIATES = [f"x{i + 1}" for i in range(K)]


def _lin():
    return LinearRegression()


def _log():
    return LogisticRegression(penalty=None, max_iter=5000, tol=1e-10)


@pytest.fixture(scope="module")
def fixture():
    """Heterogeneous-effect DGP with propensities bounded in (0.25, 0.75)."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(N, K))
    e = 0.5 + 0.25 * np.tanh(X[:, 0])
    d = (rng.uniform(size=N) < e).astype(int)
    tau = 1.0 + 0.5 * X[:, 1]
    y = X[:, 0] + tau * d + rng.normal(scale=0.5, size=N)
    df = pd.DataFrame(X, columns=COVARIATES)
    df["d"], df["y"] = d, y
    return df, X, y, d


def _sp_cate(df, learner, **kw):
    res = sp.metalearner(
        df,
        y="y",
        treat="d",
        covariates=COVARIATES,
        learner=learner,
        n_bootstrap=0,
        **kw,
    )
    return np.asarray(res.model_info["cate"], dtype=float), res


def test_s_learner_cate_matches_econml_elementwise(fixture):
    df, X, y, d = fixture
    ours, _ = _sp_cate(df, "s", outcome_model=_lin())
    ref = econml_metalearners.SLearner(overall_model=_lin())
    ref.fit(y, d, X=X)
    np.testing.assert_allclose(
        ours, np.asarray(ref.effect(X), dtype=float).ravel(), rtol=0, atol=1e-12
    )


def test_t_learner_cate_matches_econml_elementwise(fixture):
    df, X, y, d = fixture
    ours, _ = _sp_cate(df, "t", outcome_model=_lin())
    ref = econml_metalearners.TLearner(models=_lin())
    ref.fit(y, d, X=X)
    np.testing.assert_allclose(
        ours, np.asarray(ref.effect(X), dtype=float).ravel(), rtol=0, atol=1e-12
    )


def test_x_learner_cate_matches_econml_elementwise(fixture):
    """X requires aligning the imputed-effect stage as well as the arms."""
    df, X, y, d = fixture
    ours, _ = _sp_cate(
        df,
        "x",
        outcome_model=_lin(),
        cate_model=_lin(),
        propensity_model=_log(),
    )
    ref = econml_metalearners.XLearner(
        models=_lin(), cate_models=_lin(), propensity_model=_log()
    )
    ref.fit(y, d, X=X)
    np.testing.assert_allclose(
        ours, np.asarray(ref.effect(X), dtype=float).ravel(), rtol=0, atol=1e-12
    )


def test_s_and_t_learners_are_genuinely_different_functions(fixture):
    """Guard against the pins passing because everything collapsed.

    With a linear overall model the S-learner's effect is constant while
    the T-learner's varies with X. If these ever coincided, the three
    tests above could pass while ``learner=`` had stopped doing anything.
    """
    df, _, _, _ = fixture
    s_cate, _ = _sp_cate(df, "s", outcome_model=_lin())
    t_cate, _ = _sp_cate(df, "t", outcome_model=_lin())
    assert np.std(s_cate) == pytest.approx(0.0, abs=1e-10)
    assert np.std(t_cate) > 0.1
    assert not np.allclose(s_cate, t_cate)


def test_reported_ate_is_the_aipw_estimand_not_the_cate_mean(fixture):
    """Document the design that makes ``estimate`` learner-independent.

    ``sp.metalearner`` reports a doubly-robust AIPW ATE regardless of the
    CATE learner. Pinning that as if it were the learner's output would
    be comparing the wrong thing, so the contract is asserted here.

    The independence holds *given the same nuisances*: the AIPW score is
    built from the outcome and propensity models, so changing either
    legitimately moves the ATE. All three fits below therefore pin both
    nuisance models, and only ``learner=`` varies.
    """
    df, _, _, _ = fixture
    estimates = {}
    for learner, kw in (
        ("s", dict(outcome_model=_lin(), propensity_model=_log())),
        ("t", dict(outcome_model=_lin(), propensity_model=_log())),
        ("x", dict(outcome_model=_lin(), cate_model=_lin(), propensity_model=_log())),
    ):
        cate, res = _sp_cate(df, learner, **kw)
        assert res.model_info["ate_method"] == "aipw_dr_pseudo_outcome"
        estimates[learner] = float(res.estimate)
        # The CATE mean is learner-specific and generally differs from it.
        assert res.model_info["cate_mean"] == pytest.approx(
            float(np.mean(cate)), rel=1e-12
        )
    assert estimates["s"] == pytest.approx(estimates["t"], rel=1e-12)
    assert estimates["s"] == pytest.approx(estimates["x"], rel=1e-12)


# ---------------------------------------------------------------------------
# DR-learner: why it is not pinned elementwise, and what is pinned instead
# ---------------------------------------------------------------------------


def test_dr_pseudo_outcome_is_the_aipw_score_of_its_own_nuisances():
    """The operator is exact even though the end-to-end fit is not.

    ``sp.metalearner(learner='dr')`` cannot be compared elementwise with
    ``econml``'s ``DRLearner`` because the two parameterise the outcome
    nuisance differently (see the next test). What *is* exactly checkable
    is the piece in between: given the cross-fitted ``mu1``, ``mu0`` and
    ``e`` that StatsPAI actually used, the pseudo-outcome must be the
    textbook AIPW score. Pinning that separates "our nuisance models
    differ from econml's" — a modelling choice — from "our doubly-robust
    score is wrong", which would be a defect.
    """
    rng = np.random.default_rng(17)
    n = 900
    X = rng.normal(size=(n, 3))
    e = 0.5 + 0.25 * np.tanh(X[:, 0])
    d = (rng.uniform(size=n) < e).astype(int)
    y = X[:, 0] + (1.0 + 0.5 * X[:, 1]) * d + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(X, columns=COVARIATES)
    df["d"], df["y"] = d, y

    res = sp.metalearner(
        df,
        y="y",
        treat="d",
        covariates=COVARIATES,
        learner="dr",
        outcome_model=_lin(),
        propensity_model=_log(),
        cate_model=_lin(),
        n_folds=5,
        n_bootstrap=0,
    )
    est = res.model_info["_estimator"]
    diag = est._pseudo_diag
    mu1 = np.asarray(diag["mu1_hat"], dtype=float)
    mu0 = np.asarray(diag["mu0_hat"], dtype=float)
    e_hat = np.asarray(diag["e_hat"], dtype=float)

    expected = mu1 - mu0 + d * (y - mu1) / e_hat - (1 - d) * (y - mu0) / (1 - e_hat)
    np.testing.assert_allclose(
        np.asarray(est._pseudo_outcomes, dtype=float),
        expected,
        rtol=0,
        atol=1e-12,
        err_msg="the DR pseudo-outcome is not the AIPW score of its own "
        "cross-fitted nuisances",
    )


def test_dr_gap_to_econml_is_the_outcome_model_parameterisation():
    """Locate the DR difference rather than leaving it unexplained.

    StatsPAI fits the outcome nuisance **per arm** (a model on the treated
    rows and another on the controls); ``econml``'s ``DRLearner`` fits one
    joint regression on ``[X, T]``. With a linear learner the joint model
    can only express a constant treatment effect, so the two agree when
    the effect really is constant and separate when it is not.

    Making that the assertion turns an unexplained numerical gap into a
    identified, testable mechanism: if the gap ever stopped shrinking
    under a constant effect, the explanation recorded in the parity index
    would be wrong.
    """

    def _gap(tau_of_x) -> float:
        rng = np.random.default_rng(3)
        n = 1200
        X = rng.normal(size=(n, 3))
        e = 0.5 + 0.25 * np.tanh(X[:, 0])
        d = (rng.uniform(size=n) < e).astype(int)
        y = X[:, 0] + tau_of_x(X) * d + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame(X, columns=COVARIATES)
        df["d"], df["y"] = d, y

        ours = np.asarray(
            sp.metalearner(
                df,
                y="y",
                treat="d",
                covariates=COVARIATES,
                learner="dr",
                outcome_model=_lin(),
                propensity_model=_log(),
                cate_model=_lin(),
                n_folds=5,
                n_bootstrap=0,
            ).model_info["cate"],
            dtype=float,
        )
        # StatsPAI cross-fits with KFold(5, shuffle=True, random_state=42);
        # handing econml the identical partition removes the split as a
        # source of difference.
        from econml.dr import DRLearner

        ref = DRLearner(
            model_propensity=_log(),
            model_regression=_lin(),
            model_final=_lin(),
            cv=list(KFold(n_splits=5, shuffle=True, random_state=42).split(X)),
            random_state=42,
        )
        ref.fit(y, d, X=X)
        return float(
            np.abs(ours - np.asarray(ref.effect(X), dtype=float).ravel()).max()
        )

    heterogeneous = _gap(lambda X: 1.0 + 0.5 * X[:, 1])
    constant = _gap(lambda X: np.full(len(X), 1.0))
    assert constant < heterogeneous / 5.0, (
        f"gap under a constant effect ({constant:.3e}) is not much smaller "
        f"than under a heterogeneous one ({heterogeneous:.3e}); the "
        "per-arm-vs-joint outcome-model explanation for the DR difference "
        "no longer holds and the parity index note needs revisiting."
    )
