"""``sp.glmnet`` against R ``glmnet`` / ``cv.glmnet`` 4.1-10.

Five simulated designs (more rows than columns, more columns than rows, a
binary outcome, and two low-signal designs), three mixing weights each,
plus penalty factors with an unpenalised predictor, user-supplied
penalties, no standardisation, and the adaptive lasso. Reference:
``_fixtures/glmnet_R.json`` from ``_fixtures/_generate_glmnet.R``, run with
``thresh = 1e-14``; both sides read the same CSV bytes.

What is pinned, and how tightly:

* the penalty path, including where it stops: 1e-10, and equal length;
* coefficients along it: 1e-6 absolute with more rows than columns
  (observed 2e-7), 5e-5 with more columns than rows, where the minimiser
  is flat and ``glmnet``'s own iterate is the limit;
* deviance explained: 1e-7;
* cross-validated error and its standard error at every penalty: 1e-4
  relative (observed 1e-8 to 3e-5);
* ``lambda.min`` and ``lambda.1se``: the same grid point.

``glmnet`` is GPL; it was run as a black box. The four conventions that its
documentation does not state were recovered from its output and each has
its own test here: a Gaussian outcome is scaled before a ridge penalty is
applied, the first point of a computed path is the fit at an infinite
penalty, the path stops on a relative gain for a Gaussian outcome and an
absolute one for a binomial, and cross-validation lets every training set
build its own path and interpolates.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
CASES = ["tall", "wide", "binomial", "low", "low_binomial"]
ALPHAS = ["alpha1", "alpha0.5", "alpha0"]

pytestmark = pytest.mark.skipif(
    not (FIX / "glmnet_R.json").exists(), reason="glmnet fixture is not materialized"
)


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "glmnet_R.json").read_text(encoding="utf-8"))


def _data(case):
    df = pd.read_csv(FIX / f"glmnet_{case}.csv")
    return df, [c for c in df.columns if c.startswith("X")]


def _coef_tol(case):
    return 5e-5 if case == "wide" else 1e-6


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("key", ALPHAS)
def test_path_coefficients_and_deviance(ref, case, key):
    r = ref[case][key]
    df, xs = _data(case)
    fit = sp.glmnet(df, "y", xs, alpha=r["alpha"], family=ref[case]["family"], cv=False)
    lam = fit.path["lambda"].to_numpy()
    assert len(lam) == len(r["lambda"])  # stops where glmnet stops
    assert lam == pytest.approx(np.array(r["lambda"]), rel=1e-10)
    assert fit.path["dev_ratio"].to_numpy() == pytest.approx(
        np.array(r["dev"]), abs=1e-7
    )
    assert fit.path["df"].tolist() == r["df"]
    rows = [i - 1 for i in r["idx"]]
    got = fit.coefficients.iloc[rows].to_numpy()
    assert got[:, 1:].T == pytest.approx(np.array(r["beta"]), abs=_coef_tol(case))
    assert got[:, 0] == pytest.approx(np.array(r["a0"]), abs=_coef_tol(case))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("key", ALPHAS)
def test_cross_validation_selects_the_same_penalties(ref, case, key):
    r = ref[case][key]
    df, xs = _data(case)
    fit = sp.glmnet(
        df, "y", xs, alpha=r["alpha"], family=ref[case]["family"], foldid="foldid"
    )
    assert fit.cv["cvm"].to_numpy() == pytest.approx(np.array(r["cvm"]), rel=1e-4)
    assert fit.cv["cvsd"].to_numpy() == pytest.approx(np.array(r["cvsd"]), rel=1e-4)
    assert fit.lambda_min == pytest.approx(r["lambda_min"], rel=1e-10)
    assert fit.lambda_1se == pytest.approx(r["lambda_1se"], rel=1e-10)
    assert fit.penalty == fit.lambda_min
    assert fit.coef("lambda.1se").to_numpy() == pytest.approx(
        np.array(r["coef_1se"]), abs=_coef_tol(case)
    )
    one_se = sp.glmnet(
        df,
        "y",
        xs,
        alpha=r["alpha"],
        family=ref[case]["family"],
        foldid="foldid",
        rule="1se",
    )
    assert one_se.penalty == fit.lambda_1se
    assert one_se.params.to_numpy() == pytest.approx(
        fit.coef("lambda.1se").to_numpy()[1:]
    )


@pytest.mark.parametrize("case", CASES)
def test_penalty_factors_with_an_unpenalised_predictor(ref, case):
    r = ref[case]["penalty_factor"]
    df, xs = _data(case)
    fit = sp.glmnet(
        df,
        "y",
        xs,
        alpha=0.7,
        family=ref[case]["family"],
        penalty_factor=r["pf"],
        cv=False,
    )
    assert fit.path["lambda"].to_numpy() == pytest.approx(
        np.array(r["lambda"]), rel=1e-10
    )
    rows = [i - 1 for i in r["idx"]]
    got = fit.coefficients.iloc[rows].to_numpy()
    assert got[:, 1:].T == pytest.approx(np.array(r["beta"]), abs=_coef_tol(case))
    # at the top of the path only the unpenalised predictor is in the model
    first = fit.coefficients.iloc[0, 1:]
    assert first.iloc[0] != 0 and (first.iloc[1:] == 0).all()


@pytest.mark.parametrize("case", CASES)
def test_supplied_penalties_with_and_without_standardising(ref, case):
    r = ref[case]["user"]
    df, xs = _data(case)
    fam = ref[case]["family"]
    tol = 5e-5 if case == "wide" else 1e-6
    fit = sp.glmnet(df, "y", xs, family=fam, lambda_=r["lambda"], cv=False)
    raw = sp.glmnet(
        df, "y", xs, family=fam, lambda_=r["lambda"], cv=False, standardize=False
    )
    # With 40 rows and 60 predictors the lasso at the smallest penalty
    # (0.001) nearly interpolates and coordinate descent crawls: glmnet at
    # thresh = 1e-14 is still 1e-3 from the minimiser there. That penalty
    # is compared loosely, and the fit is checked against the subgradient
    # conditions instead (test_wide_design_small_penalty_is_the_minimiser).
    keep = slice(0, 2) if case == "wide" else slice(None)
    got, want = fit.coefficients.iloc[:, 1:].to_numpy().T, np.array(r["beta"])
    assert got[:, keep] == pytest.approx(want[:, keep], abs=tol)
    assert got == pytest.approx(want, abs=5e-3)
    assert fit.coefficients["intercept"].to_numpy()[keep] == pytest.approx(
        np.array(r["a0"])[keep], abs=tol
    )
    got_raw, want_raw = (
        raw.coefficients.iloc[:, 1:].to_numpy().T,
        np.array(r["beta_nostd"]),
    )
    assert got_raw[:, keep] == pytest.approx(want_raw[:, keep], abs=tol)
    # with penalties supplied, every fold is fitted at exactly those
    cvfit = sp.glmnet(
        df, "y", xs, alpha=0.5, family=fam, lambda_=r["cv_lambda"], foldid="foldid"
    )
    # With 32 training rows and 60 predictors the fit at a small penalty is
    # close to interpolating and its held-out error is not determined to
    # more than two or three digits by either program.
    cv_tol = 1e-2 if case == "wide" else 1e-5
    assert cvfit.cv["cvm"].to_numpy() == pytest.approx(np.array(r["cvm"]), rel=cv_tol)
    assert cvfit.lambda_min == pytest.approx(r["lambda_min"], rel=1e-12)
    assert cvfit.lambda_1se == pytest.approx(r["lambda_1se"], rel=1e-12)


def test_adaptive_lasso(ref):
    """Ridge first, then a lasso with weights 1 / |b|."""
    r = ref["adaptive"]
    df, xs = _data("tall")
    ridge = sp.glmnet(df, "y", xs, alpha=0.0, lambda_=0.1, cv=False)
    assert ridge.params.to_numpy() == pytest.approx(np.array(r["ridge"]), abs=1e-7)
    weights = 1.0 / np.abs(ridge.params.to_numpy())
    fit = sp.glmnet(df, "y", xs, penalty_factor=weights, foldid="foldid")
    assert fit.lambda_min == pytest.approx(r["lambda_min"], rel=1e-8)
    assert fit.lambda_1se == pytest.approx(r["lambda_1se"], rel=1e-8)
    assert fit.coef("lambda.min").to_numpy() == pytest.approx(
        np.array(r["coef_min"]), abs=1e-5
    )
    assert fit.coef("lambda.1se").to_numpy() == pytest.approx(
        np.array(r["coef_1se"]), abs=1e-5
    )


# --- the conventions, one at a time ---------------------------------------------


def test_gaussian_ridge_penalty_is_on_the_standardised_outcome():
    """In original units the ridge term is lambda / sd(y): the closed form."""
    df, xs = _data("tall")
    n = len(df)
    lam = 0.37
    fit = sp.glmnet(df, "y", xs, alpha=0.0, lambda_=lam, cv=False)
    X = df[xs].to_numpy()
    y = df["y"].to_numpy()
    sd = np.sqrt(np.mean((X - X.mean(0)) ** 2, axis=0))
    Z = (X - X.mean(0)) / sd
    sdy = np.sqrt(np.mean((y - y.mean()) ** 2))
    b = np.linalg.solve(
        Z.T @ Z / n + lam / sdy * np.eye(len(xs)), Z.T @ (y - y.mean()) / n
    )
    assert fit.params.to_numpy() == pytest.approx(b / sd, rel=1e-7)
    # the lasso is equivariant: rescaling y rescales lambda and the fit
    a = sp.glmnet(df, "y", xs, lambda_=0.2, cv=False)
    c = sp.glmnet(df.assign(y=df["y"] * 10), "y", xs, lambda_=2.0, cv=False)
    assert c.params.to_numpy() == pytest.approx(10 * a.params.to_numpy(), rel=1e-8)


def test_first_point_of_a_computed_ridge_path_is_the_null_model(ref):
    df, xs = _data("tall")
    fit = sp.glmnet(df, "y", xs, alpha=0.0, cv=False)
    assert np.abs(fit.coefficients.iloc[0, 1:]).max() < 1e-30
    assert fit.path["df"].iloc[0] == len(xs)  # tiny, not zero, as in glmnet
    assert fit.coefficients.iloc[0, 0] == pytest.approx(df["y"].mean())
    # the same penalty, supplied, gives the ridge fit there
    lam1 = float(fit.path["lambda"].iloc[0])
    given = sp.glmnet(df, "y", xs, alpha=0.0, lambda_=lam1, cv=False)
    assert np.abs(given.params.to_numpy()).max() > 1e-4


def test_stopping_rule_is_relative_for_gaussian_and_absolute_for_binomial(ref):
    """The low-signal designs explain 11% and 4% of the deviance; the two
    rules then stop at different places and the lengths tell them apart."""
    for case in ("low", "low_binomial"):
        df, xs = _data(case)
        fit = sp.glmnet(df, "y", xs, family=ref[case]["family"], cv=False)
        dev = fit.path["dev_ratio"].to_numpy()
        gain = dev[-1] - dev[-2]
        assert len(dev) == len(ref[case]["alpha1"]["lambda"]) < 100
        if case == "low":
            assert (
                gain < 1e-5 * dev[-1] and (np.diff(dev)[4:-1] >= 1e-5 * dev[5:-1]).all()
            )
        else:
            assert gain < 1e-5 and (np.diff(dev)[4:-1] >= 1e-5).all()


def test_minimiser_satisfies_the_subgradient_conditions():
    """No reference: the KKT conditions of the stated objective."""
    df, xs = _data("tall")
    n = len(df)
    lam, alpha = 0.4, 0.6
    fit = sp.glmnet(df, "y", xs, alpha=alpha, lambda_=lam, cv=False)
    X = df[xs].to_numpy()
    y = df["y"].to_numpy()
    sd = np.sqrt(np.mean((X - X.mean(0)) ** 2, axis=0))
    Z = (X - X.mean(0)) / sd
    sdy = np.sqrt(np.mean((y - y.mean()) ** 2))
    b = fit.params.to_numpy() * sd  # coefficients of the standardised predictors
    grad = Z.T @ (y - y.mean() - Z @ b) / n - lam / sdy * (1 - alpha) * b
    active = b != 0
    assert active.any() and (~active).any()
    assert grad[active] == pytest.approx(lam * alpha * np.sign(b[active]), abs=1e-7)
    assert (np.abs(grad[~active]) <= lam * alpha + 1e-9).all()


def test_wide_design_small_penalty_is_the_minimiser():
    """Where glmnet's iterate is not converged, ours satisfies the
    subgradient conditions of the lasso to 1e-7."""
    df, xs = _data("wide")
    n = len(df)
    lam = 0.001
    fit = sp.glmnet(df, "y", xs, lambda_=lam, cv=False)
    X = df[xs].to_numpy()
    y = df["y"].to_numpy()
    sd = np.sqrt(np.mean((X - X.mean(0)) ** 2, axis=0))
    Z = (X - X.mean(0)) / sd
    b = fit.params.to_numpy() * sd
    grad = Z.T @ (y - y.mean() - Z @ b) / n
    active = b != 0
    assert grad[active] == pytest.approx(lam * np.sign(b[active]), abs=1e-7)
    assert (np.abs(grad[~active]) <= lam + 1e-9).all()


def test_binomial_predictions_and_inputs(ref):
    from statspai.exceptions import MethodIncompatibility

    df, xs = _data("binomial")
    fit = sp.glmnet(df, "y", xs, family="binomial", foldid="foldid", rule="1se")
    prob = fit.predict(df)
    link = fit.predict(df, type="link")
    assert ((prob > 0) & (prob < 1)).all()
    assert prob == pytest.approx(1 / (1 + np.exp(-link)))
    assert fit.predict(df, s="lambda.min").shape == (len(df),)
    with pytest.raises(MethodIncompatibility, match="not a penalty on the fitted path"):
        fit.coef(0.123456)
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.glmnet(df.assign(y=df["y"] + 1), "y", xs, family="binomial")
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.glmnet(df, "y", xs, alpha=1.5)
    with pytest.raises(MethodIncompatibility, match="penalty_factor"):
        sp.glmnet(df, "y", xs, penalty_factor=[1.0, 2.0])
    with pytest.raises(MethodIncompatibility, match="needs cross-validation"):
        sp.glmnet(df, "y", xs, family="binomial", cv=False).coef("lambda.min")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        seeded = sp.glmnet(df, "y", xs, family="binomial", seed=3)
        again = sp.glmnet(df, "y", xs, family="binomial", seed=3)
    assert seeded.lambda_min == again.lambda_min
