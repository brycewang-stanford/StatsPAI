"""``sp.tmle``'s ``Q`` / ``g1W`` injection and fluctuation conventions.

These parameters exist so the targeting step can be isolated from the
initial fit -- the design that makes Track A module 72 a like-for-like
comparison against ``tmle::tmle`` rather than a comparison of two
different Super Learners.

The default ``fluctuation='single'`` is unchanged and remains the
documented behaviour; ``'per_arm'`` reproduces the R package's
two-covariate submodel.

References
----------
[@vanderlaan2006targeted], [@gruber2012tmle]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

N = 800


@pytest.fixture(scope="module")
def fixture():
    """Binary-outcome DGP with well-separated propensities."""
    rng = np.random.default_rng(11)
    W = rng.normal(size=(N, 2))
    g = 0.5 + 0.25 * np.tanh(W[:, 0])
    A = (rng.uniform(size=N) < g).astype(int)
    p_y = 1.0 / (1.0 + np.exp(-(-0.2 + 0.7 * A + 0.5 * W[:, 0])))
    Y = (rng.uniform(size=N) < p_y).astype(int)
    df = pd.DataFrame(W, columns=["w1", "w2"])
    df["A"], df["Y"] = A, Y

    from sklearn.linear_model import LogisticRegression

    q = LogisticRegression(penalty=None, max_iter=5000, tol=1e-10)
    q.fit(np.column_stack([A, W]), Y)
    Q = np.column_stack(
        [
            q.predict_proba(np.column_stack([np.zeros(N), W]))[:, 1],
            q.predict_proba(np.column_stack([np.ones(N), W]))[:, 1],
        ]
    )
    gm = LogisticRegression(penalty=None, max_iter=5000, tol=1e-10).fit(W, A)
    g1W = gm.predict_proba(W)[:, 1]
    return df, Q, g1W


def _fit(df, **kw):
    return sp.tmle(
        data=df,
        y="Y",
        treat="A",
        covariates=["w1", "w2"],
        propensity_bounds=(1e-8, 1 - 1e-8),
        **kw,
    )


def test_supplied_nuisances_bypass_the_super_learner(fixture):
    """Supplying Q / g1W must skip the SL stage and say so."""
    df, Q, g1W = fixture
    res = _fit(df, Q=Q, g1W=g1W)
    info = res.model_info
    assert info["nuisance_source"] == {"Q": "supplied", "g1W": "supplied"}
    # No Super Learner ran, so there are no ensemble weights to report;
    # reporting stale or fabricated weights would be worse than None.
    assert info["sl_outcome_weights"] is None
    assert info["sl_propensity_weights"] is None


def test_supplied_nuisances_make_the_fit_deterministic(fixture):
    """With both nuisances supplied, random_state cannot matter."""
    df, Q, g1W = fixture
    a = _fit(df, Q=Q, g1W=g1W, random_state=1)
    b = _fit(df, Q=Q, g1W=g1W, random_state=98765)
    assert float(a.estimate) == float(b.estimate)
    assert float(a.se) == float(b.se)


def test_default_fluctuation_is_single_and_reports_a_scalar_epsilon(fixture):
    """The historical scalar ``epsilon`` field keeps its type."""
    df, Q, g1W = fixture
    res = _fit(df, Q=Q, g1W=g1W)
    info = res.model_info
    assert info["fluctuation"] == "single"
    assert isinstance(info["epsilon"], float)
    assert len(info["epsilon_vec"]) == 1


def test_per_arm_fluctuation_reports_a_two_vector_and_no_scalar(fixture):
    """Under per-arm targeting there is no scalar fluctuation parameter."""
    df, Q, g1W = fixture
    res = _fit(df, Q=Q, g1W=g1W, fluctuation="per_arm")
    info = res.model_info
    assert info["fluctuation"] == "per_arm"
    assert info["epsilon"] is None
    assert len(info["epsilon_vec"]) == 2


def test_the_two_fluctuations_differ_but_agree_closely(fixture):
    """Both are valid TMLEs: close, but not the same estimator.

    If these ever became bit-identical, ``fluctuation='per_arm'`` would
    have silently stopped doing anything and module 72's agreement with
    ``tmle::tmle`` would be accidental.
    """
    df, Q, g1W = fixture
    single = _fit(df, Q=Q, g1W=g1W, fluctuation="single")
    per_arm = _fit(df, Q=Q, g1W=g1W, fluctuation="per_arm")
    gap = abs(float(single.estimate) - float(per_arm.estimate))
    assert gap > 0.0, "per_arm must not collapse onto the single-covariate fit"
    assert gap < 0.05, f"asymptotically equivalent estimators drifted: {gap:.4g}"


def test_per_arm_solves_both_arm_score_equations(fixture):
    """The defining property of the per-arm submodel.

    A correct two-dimensional targeting step drives *each* arm's score to
    zero, not just their difference. Checking the property directly means
    the test does not depend on R being installed.
    """
    df, Q, g1W = fixture
    res = _fit(df, Q=Q, g1W=g1W, fluctuation="per_arm")
    A = df["A"].to_numpy(dtype=float)
    Y = df["Y"].to_numpy(dtype=float)
    e = np.asarray(res.model_info["epsilon_vec"], dtype=float)

    from scipy.special import expit, logit

    q_bar = np.where(A == 1, Q[:, 1], Q[:, 0])
    h1, h0 = A / g1W, -(1 - A) / (1 - g1W)
    q_star = expit(logit(q_bar) + e[0] * h1 + e[1] * h0)
    assert abs(float(np.sum(h1 * (Y - q_star)))) < 1e-6
    assert abs(float(np.sum(h0 * (Y - q_star)))) < 1e-6


@pytest.mark.parametrize("bad", ["двух", "", "PER_ARM", "two"])
def test_unknown_fluctuation_is_rejected(fixture, bad):
    df, Q, g1W = fixture
    with pytest.raises(ValueError, match="fluctuation"):
        _fit(df, Q=Q, g1W=g1W, fluctuation=bad)


def test_wrong_shaped_Q_fails_loudly(fixture):
    df, Q, g1W = fixture
    with pytest.raises(ValueError, match=r"Q must have shape"):
        _fit(df, Q=Q[:, :1], g1W=g1W)


def test_wrong_length_g1W_fails_loudly(fixture):
    df, Q, g1W = fixture
    with pytest.raises(ValueError, match=r"g1W must have length"):
        _fit(df, Q=Q, g1W=g1W[:-1])
