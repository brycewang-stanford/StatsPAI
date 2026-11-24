"""GitHub issues #53 and #54 (Ian Gow, 2026-11-19).

#53: a binomial ``sp.glm`` reported fitted probabilities floored at 1e-15
from ``fitted_values()`` / ``predict()`` while ``predict(data)`` on the
same rows returned the exact inverse link; the example also claimed the
MLE does not exist although both outcomes occur at every x in {-1, 0, 1}.

#54: ``sp.glmnet(family="binomial")`` at a single small penalty overflowed
from the cold start and returned probabilities of 0, while a decreasing
path ending at the same penalty converged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import statspai as sp
from statspai.exceptions import ConvergenceWarning, StatsPAIWarning
from statspai.regression.logit_probit import _separating_direction

# --------------------------------------------------------------------- #53


@pytest.fixture
def boundary_frame() -> pd.DataFrame:
    x = np.r_[np.repeat([-1.0, 0.0, 1.0], 10), -25.0, -35.0]
    y = np.r_[np.zeros(9), 1, np.zeros(5), np.ones(5), 0, np.ones(9), 0, 0]
    return pd.DataFrame({"x": x, "y": y})


def _fit(df, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sp.glm("y ~ x", data=df, family="binomial", **kw)
    return fit, caught


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
def test_in_sample_and_explicit_predictions_agree(boundary_frame, link):
    fit, _ = _fit(boundary_frame, link=link)
    explicit = fit.predict(boundary_frame)
    # same convention on both paths; the two design builds differ in the
    # last bits of the linear index only
    np.testing.assert_allclose(fit.predict(), explicit, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(fit.fitted_values(), fit.predict())
    eta = fit.predict(boundary_frame, what="link")
    np.testing.assert_allclose(
        fit.predict(), fit._inverse_link()(eta), rtol=1e-12, atol=0
    )


def test_extreme_probabilities_are_not_floored(boundary_frame):
    fit, _ = _fit(boundary_frame, link="logit")
    eta = fit.predict(boundary_frame, what="link")
    # 1.39e-24 and 4.0e-34, not 1e-15 (issue #53)
    np.testing.assert_allclose(fit.fitted_values()[-2:], expit(eta[-2:]), rtol=1e-12)
    assert fit.fitted_values()[-1] < 1e-30
    # responses residuals are y - fitted on the same scale
    np.testing.assert_allclose(
        fit.residuals(), boundary_frame["y"].to_numpy() - fit.fitted_values()
    )


def test_closed_form_estimates_are_unchanged(boundary_frame):
    # P(y=1 | x) = 0.1 / 0.5 / 0.9 at x = -1 / 0 / 1: intercept 0,
    # slope log 9; the two far points add nothing to the information,
    # so SE = 1/sqrt(10 * 0.43) and 1/sqrt(10 * 0.18).
    fit, _ = _fit(boundary_frame, link="logit")
    np.testing.assert_allclose(fit.params.to_numpy(), [0.0, np.log(9.0)], atol=1e-8)
    np.testing.assert_allclose(
        fit.std_errors.to_numpy(), [1 / np.sqrt(4.3), 1 / np.sqrt(1.8)], rtol=1e-6
    )
    assert np.isfinite(fit.model_info.get("ll", fit.diagnostics.get("ll", 0.0)))


def test_boundary_probabilities_without_separation_do_not_claim_nonexistence(
    boundary_frame,
):
    _, caught = _fit(boundary_frame, link="logit")
    assert not [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    notes = [w for w in caught if "numerically 0 or 1" in str(w.message)]
    assert len(notes) == 1
    assert issubclass(notes[0].category, StatsPAIWarning)
    msg = str(notes[0].message)
    assert "not separated" in msg and "do not exist" not in msg


def test_quasi_complete_separation_is_still_claimed():
    # the 30 rows with x = 2 all have y = 1; elsewhere both outcomes occur
    df = pd.DataFrame(
        {
            "x": np.r_[np.zeros(10), np.ones(10), np.full(30, 2.0)],
            "y": np.r_[np.tile([0, 1], 10), np.ones(30)],
        }
    )
    df["d2"] = (df["x"] == 2).astype(float)
    with pytest.warns(ConvergenceWarning, match="Quasi-complete separation detected"):
        sp.glm("y ~ d2", data=df, family="binomial")


def test_complete_separation_is_still_claimed():
    df = pd.DataFrame({"x": np.arange(20.0), "y": (np.arange(20) >= 10).astype(float)})
    with pytest.warns(ConvergenceWarning, match="Complete separation"):
        sp.glm("y ~ x", data=df, family="binomial")


def test_separating_direction_lp():
    X = np.column_stack([np.ones(6), [0.0, 0, 0, 0, 1, 1]])
    # both outcomes where the dummy is 0, only ones where it is 1
    assert _separating_direction(np.array([0, 1, 0, 1, 1, 1.0]), X) is True
    # both outcomes at every x: no separation
    X2 = np.column_stack([np.ones(6), [0.0, 0, 1, 1, 2, 2]])
    assert _separating_direction(np.array([0, 1, 0, 1, 0, 1.0]), X2) is False
    # a zero-weight row does not enter the likelihood
    w = np.array([1, 1, 1, 1, 1, 0.0])
    assert _separating_direction(np.array([0, 1, 0, 1, 1, 0.0]), X, w) is True


# --------------------------------------------------------------------- #54


@pytest.fixture
def glmnet_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0] * 1000 + [1.0, 1.0, -1.0, -1.0, 4.0],
            "y": [0] * 1000 + [1, 1, 0, 0, 0],
        }
    )


def _glmnet(df, lam):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sp.glmnet(
            df, y="y", x=["x"], family="binomial", alpha=0, lambda_=lam, cv=False
        )
    return fit, caught


def test_scalar_penalty_converges_and_matches_the_path(glmnet_frame):
    target = 1 / len(glmnet_frame)
    scalar, caught = _glmnet(glmnet_frame, target)
    assert not caught, [str(w.message) for w in caught]
    assert scalar.model_info["converged"] is True
    path, _ = _glmnet(glmnet_frame, np.geomspace(0.1, target, 25))
    p_scalar = scalar.predict(glmnet_frame)[-1]
    np.testing.assert_allclose(p_scalar, path.predict(glmnet_frame)[-1], atol=1e-9)
    np.testing.assert_allclose(p_scalar, 0.48838872, atol=1e-6)
    np.testing.assert_allclose(
        scalar.params.to_numpy(), path.params.to_numpy(), atol=1e-8
    )


def test_scalar_penalty_agrees_with_sklearn(glmnet_frame):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    target = 1 / len(glmnet_frame)
    scalar, _ = _glmnet(glmnet_frame, target)
    x = glmnet_frame[["x"]].to_numpy()
    # glmnet: mean loss + lam/2 * (b * sd)^2; sklearn: C * sum loss + b^2 / 2
    C = 1.0 / (len(glmnet_frame) * target * x.std() ** 2)
    ref = LogisticRegression(C=C, tol=1e-14, max_iter=100_000).fit(x, glmnet_frame["y"])
    np.testing.assert_allclose(scalar.params.to_numpy(), ref.coef_[0], atol=1e-7)
    np.testing.assert_allclose(scalar.intercept, ref.intercept_[0], atol=1e-7)


def test_converged_fits_are_unchanged_by_the_retry():
    # a well-conditioned binomial lasso: the direct fit converges, so the
    # retry never runs and the scalar and path answers agree at the tolerance
    rng = np.random.default_rng(54)
    X = rng.normal(size=(300, 4))
    df = pd.DataFrame(X, columns=list("abcd"))
    df["y"] = (rng.uniform(size=300) < expit(X[:, 0] - 0.5 * X[:, 1])).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        one = sp.glmnet(
            df, "y", list("abcd"), family="binomial", lambda_=0.01, cv=False
        )
    full = sp.glmnet(
        df,
        "y",
        list("abcd"),
        family="binomial",
        lambda_=np.geomspace(0.2, 0.01, 30),
        cv=False,
    )
    np.testing.assert_allclose(one.params.to_numpy(), full.params.to_numpy(), atol=1e-7)


# ------------------------------------------------- correlated predictors
# iangow/era_statpai#20 (2026-11-24): a ridge logit on the 28 normalised
# financial-statement items of the book's Chapter 26 took about 4 s and
# warned that it had not converged on 8 of 12 samples, while its
# probabilities were within 6e-5 of a tightly converged scikit-learn fit.


def _correlated_frame(n: int, n_pos: int, p: int = 28) -> pd.DataFrame:
    """Rare outcome on unit-length rows of items and near-sums of items."""
    rng = np.random.default_rng(0)
    size = rng.lognormal(5, 2.5, size=(n, 1))
    base = np.abs(rng.standard_t(2.5, size=(n, 12))) * size
    mix = rng.uniform(0, 1, size=(12, p - 12))
    agg = base @ mix * (1 + 1e-2 * rng.standard_normal((n, p - 12)))
    X = np.hstack([base, agg])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    score = np.exp(3 * X[:, 0] - 2 * X[:, 5] + 1.5 * X[:, 9])
    y = np.zeros(n)
    y[rng.choice(n, n_pos, replace=False, p=score / score.sum())] = 1.0
    df = pd.DataFrame((X - X.mean(0)) / X.std(0), columns=[f"x{j}" for j in range(p)])
    df["y"] = y
    return df


def _penalised_gradient(df, fit, lam, alpha):
    """Largest violation of the first-order conditions, standardised units.

    Under glmnet's guard a fitted probability below 1e-5 counts as 0.
    """
    xs = [c for c in df.columns if c != "y"]
    X = df[xs].to_numpy()
    sd = X.std(axis=0)
    beta = fit.params.to_numpy() * sd
    prob = expit(fit.intercept + X @ fit.params.to_numpy())
    resid = df["y"].to_numpy() - np.where(prob < 1e-5, 0.0, prob)
    score = (X / sd).T @ resid / len(df) - lam * (1 - alpha) * beta
    active = beta != 0
    worst = np.abs(score[active] - lam * alpha * np.sign(beta[active])).max()
    if (~active).any():
        worst = max(worst, (np.abs(score[~active]) - lam * alpha).max())
    return max(worst, abs(resid.mean()))


# the first three warned on 1.46.3; the fourth converged there and stays put
@pytest.mark.parametrize(
    "alpha, n, n_pos",
    [(0.0, 8000, 12), (1.0, 8000, 12), (1.0, 12000, 20), (0.5, 8000, 12)],
)
def test_correlated_predictors_converge_to_the_optimum(alpha, n, n_pos):
    df = _correlated_frame(n, n_pos)
    xs = [c for c in df.columns if c != "y"]
    lam = 1 / len(df)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sp.glmnet(
            df, y="y", x=xs, family="binomial", alpha=alpha, lambda_=lam, cv=False
        )
    assert not caught, [str(w.message) for w in caught]
    assert fit.model_info["converged"] is True
    # The optimum is where the penalised score vanishes (lam is about 1e-4).
    # An observation whose probability sits at the guard can be on either
    # side of it, which moves the score by at most 1e-5 * |x| / n.
    assert _penalised_gradient(df, fit, lam, alpha) < 1e-7
