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
