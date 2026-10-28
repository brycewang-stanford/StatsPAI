"""``sp.simex(variance='asymptotic')`` and MC-SIMEX (``misclassification=``).

Known-truth recovery, identities and refusals. Deterministic parity with R
``simex`` / ``mcsimex`` on shared draws is in
``tests/reference_parity/test_simex_r_parity.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility
from statspai.measurement_error import _simex_asymptotic, _simex_misclass
from statspai.measurement_error._simex_extrapolate import _extrapolate

P_X = [[0.85, 0.10], [0.15, 0.90]]  # columns: true 0, true 1


def _normal_error(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + 0.5 * z + rng.normal(size=n),
            "w": x + rng.normal(scale=0.5, size=n),
            "z": z,
        }
    )


def _misclassified(n, seed, P=P_X):
    rng = np.random.default_rng(seed)
    xb = (rng.uniform(size=n) < 0.4).astype(float)
    z = rng.normal(size=n)
    xs = _simex_misclass.misclassify(xb, np.asarray(P), rng)
    db = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-0.3 + 1.2 * xb + 0.5 * z)))).astype(
        float
    )
    dm = _simex_misclass.misclassify(db, np.asarray([[0.9, 0.2], [0.1, 0.8]]), rng)
    return pd.DataFrame(
        {
            "yb": 1.0 + 1.5 * xb + 0.5 * z + rng.normal(size=n),
            "xs": xs,
            "z": z,
            "db": db,
            "dm": dm,
        }
    )


# --------------------------------------------------------------------
# Asymptotic variance
# --------------------------------------------------------------------


def test_asymptotic_se_matches_the_sampling_sd():
    """Mean asymptotic SE against the Monte Carlo SD of the SIMEX estimate
    over 200 samples (n = 400). The SD of a 200-draw SD is about 5%; the
    bound is 15%."""
    est, se = [], []
    for r in range(200):
        fit = sp.simex(
            "y ~ w + z",
            _normal_error(400, 1000 + r),
            error_sd={"w": 0.5},
            B=10,
            seed=r,
            variance="asymptotic",
        )
        est.append(fit.params["w"])
        se.append(fit.std_errors["w"])
    sd = float(np.std(est, ddof=1))
    assert float(np.mean(se)) == pytest.approx(sd, rel=0.15)


def test_point_estimate_does_not_depend_on_the_variance_choice():
    df = _normal_error(500, 3)
    jk = sp.simex("y ~ w + z", df, error_sd={"w": 0.5}, B=10, seed=7)
    asy = sp.simex(
        "y ~ w + z", df, error_sd={"w": 0.5}, B=10, seed=7, variance="asymptotic"
    )
    np.testing.assert_array_equal(jk.params.values, asy.params.values)
    np.testing.assert_array_equal(
        np.diag(asy.model_info["jackknife_cov"]) ** 0.5, jk.std_errors.values
    )
    assert "jackknife_cov" not in jk.model_info
    assert jk.model_info["variance"] == "jackknife"
    assert "asymptotic" in asy.model_info["robust"]


@pytest.mark.parametrize("family", ["gaussian", "logit", "probit", "poisson"])
def test_score_parts_are_the_score_and_its_jacobian(family):
    """At the naive MLE the scores sum to zero; the Jacobian is the
    derivative of the summed score (central differences)."""
    from statspai.measurement_error.simex import _fit

    rng = np.random.default_rng(5)
    n = 400
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    eta = X @ np.array([0.2, 0.6, -0.4])
    w = rng.uniform(0.5, 2.0, size=n)
    if family == "gaussian":
        y = eta + rng.normal(size=n)
    elif family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    beta, _ = _fit(X, y, w, family, "ols")
    psi, jac = _simex_asymptotic.score_parts(X, y, w, family, beta)
    assert np.abs(psi.sum(axis=0)).max() <= 1e-7 * np.abs(psi).sum()
    h = 1e-6
    num = np.empty((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        up = _simex_asymptotic.score_parts(X, y, w, family, beta + e)[0].sum(axis=0)
        dn = _simex_asymptotic.score_parts(X, y, w, family, beta - e)[0].sum(axis=0)
        num[:, j] = (up - dn) / (2 * h)
    if family == "probit":
        # Expected (Fisher) information, as in the IRLS bread: equals the
        # observed one only in expectation for a non-canonical link.
        np.testing.assert_allclose(jac, num, atol=0.05 * np.abs(num).max())
    else:
        np.testing.assert_allclose(jac, num, rtol=1e-6)


def test_asymptotic_needs_one_error_sd():
    df = _normal_error(200, 1)
    df["sd"] = np.where(df["z"] > 0, 0.4, 0.6)
    with pytest.raises(MethodIncompatibility, match="per-row"):
        sp.simex("y ~ w + z", df, error_sd={"w": "sd"}, B=5, variance="asymptotic")
    df["sd"] = 0.5  # a constant column is homoskedastic
    a = sp.simex("y ~ w + z", df, error_sd={"w": "sd"}, B=5, variance="asymptotic")
    b = sp.simex("y ~ w + z", df, error_sd={"w": 0.5}, B=5, variance="asymptotic")
    np.testing.assert_array_equal(a.std_errors.values, b.std_errors.values)


# --------------------------------------------------------------------
# MC-SIMEX
# --------------------------------------------------------------------


def test_mc_power_is_a_matrix_power():
    P = np.asarray(P_X)
    np.testing.assert_allclose(_simex_misclass.mc_power(P, 1.0), P, atol=1e-7)
    np.testing.assert_allclose(_simex_misclass.mc_power(P, 0.0), np.eye(2), atol=1e-7)
    np.testing.assert_allclose(_simex_misclass.mc_power(P, 2.0), P @ P, atol=1e-7)
    half = _simex_misclass.mc_power(P, 0.5)
    np.testing.assert_allclose(half @ half, P, atol=1e-6)
    np.testing.assert_allclose(half.sum(axis=0), 1.0, atol=1e-7)


def test_misclassify_draws_from_the_column():
    rng = np.random.default_rng(0)
    P = np.asarray(P_X)
    zeros = _simex_misclass.misclassify(np.zeros(200000), P, rng)
    ones = _simex_misclass.misclassify(np.ones(200000), P, rng)
    assert zeros.mean() == pytest.approx(P[1, 0], abs=0.003)
    assert ones.mean() == pytest.approx(P[1, 1], abs=0.003)


def test_mcsimex_reduces_the_bias_of_a_misclassified_regressor():
    """Linear model, true slope 1.5. The naive slope is attenuated to about
    1.5 * 0.70 (sensitivity 0.9, specificity 0.85, prevalence 0.4)."""
    df = _misclassified(20000, 11)
    for ex in ("quadratic", "loglinear"):
        fit = sp.simex(
            "yb ~ xs + z", df, misclassification={"xs": P_X}, B=30, extrapolant=ex
        )
        naive = fit.model_info["naive_params"]["xs"]
        assert naive < 1.2
        assert abs(fit.params["xs"] - 1.5) < 0.5 * abs(naive - 1.5)
    assert fit.model_info["model_type"] == "MC-SIMEX (gaussian)"
    assert fit.model_info["misclassification"] == {"xs": P_X}


def test_mcsimex_misclassified_outcome():
    """Logit outcome misclassified with sensitivity 0.8, specificity 0.9:
    the naive slope of xb (true 1.2) is attenuated; MC-SIMEX moves it back."""
    df = _misclassified(20000, 12)
    rng = np.random.default_rng(12)
    xb = (rng.uniform(size=20000) < 0.4).astype(float)
    z = df["z"].to_numpy()
    db = (
        rng.uniform(size=20000) < 1 / (1 + np.exp(-(-0.3 + 1.2 * xb + 0.5 * z)))
    ).astype(float)
    P_Y = [[0.9, 0.2], [0.1, 0.8]]
    df["xb"] = xb
    df["dm"] = _simex_misclass.misclassify(db, np.asarray(P_Y), rng)
    fit = sp.simex(
        "dm ~ xb + z", df, misclassification={"dm": P_Y}, family="logit", B=30
    )
    naive = fit.model_info["naive_params"]["xb"]
    assert naive < 0.9
    assert abs(fit.params["xb"] - 1.2) < 0.5 * abs(naive - 1.2)


def test_mcsimex_asymptotic_se_matches_the_sampling_sd():
    """100 samples of n = 600; bound 20% (SD of a 100-draw SD is ~7%)."""
    est, se = [], []
    for r in range(100):
        fit = sp.simex(
            "yb ~ xs + z",
            _misclassified(600, 2000 + r),
            misclassification={"xs": P_X},
            B=10,
            seed=r,
            variance="asymptotic",
        )
        est.append(fit.params["xs"])
        se.append(fit.std_errors["xs"])
    assert float(np.mean(se)) == pytest.approx(float(np.std(est, ddof=1)), rel=0.2)


def test_identity_matrix_means_no_error():
    df = _misclassified(300, 4)
    fit = sp.simex("yb ~ xs + z", df, misclassification={"xs": np.eye(2)}, B=5)
    np.testing.assert_array_equal(
        fit.params.values, fit.model_info["naive_params"].values
    )
    assert fit.diagnostics["no_measurement_error"]


def test_matrix_as_labelled_dataframe():
    df = _misclassified(300, 5)
    lab = pd.DataFrame(P_X, index=[0, 1], columns=[0, 1]).loc[[1, 0], [1, 0]]
    a = sp.simex("yb ~ xs + z", df, misclassification={"xs": P_X}, B=5)
    b = sp.simex("yb ~ xs + z", df, misclassification={"xs": lab}, B=5)
    np.testing.assert_array_equal(a.params.values, b.params.values)


def test_loglinear_extrapolant_is_exact_on_an_exponential_path():
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    pos = 2.0 * np.exp(-0.3 * grid)
    neg = pos - 3.0  # minimum -1.9 -> shift 2.9
    out, failed = _extrapolate(grid, np.column_stack([pos, pos]), "loglinear")
    np.testing.assert_allclose(out, 2.0 * np.exp(0.3), rtol=1e-12)
    assert failed == []
    # With a shift the curve is fitted to log(path + 2.9), not exact.
    out, _ = _extrapolate(grid, neg[:, None], "loglinear")
    shift = abs(neg.min()) + 1
    a, b = np.polyfit(grid, np.log(neg + shift), 1)[::-1]
    assert out[0] == pytest.approx(np.exp(a - b) - shift, rel=1e-12)


def test_mc_refusals():
    df = _misclassified(200, 6)
    df["w"] = np.random.default_rng(6).normal(size=len(df))
    bad = [
        (dict(error_sd={"w": 0.5}, misclassification={"xs": P_X}), "not both"),
        (dict(misclassification={"xs": [[0.9, 0.1]]}), "2 x 2"),
        (dict(misclassification={"xs": [[0.9, 0.2], [0.2, 0.8]]}), "sum to one"),
        (dict(misclassification={"xs": [[0.4, 0.6], [0.6, 0.4]]}), "<= 1"),
        (dict(misclassification={"z": P_X}), "0/1"),
        (dict(misclassification={"q": P_X}), "not a term"),
        (dict(misclassification={"xs": P_X}, variance="bootstrap"), "variance"),
        (dict(misclassification={"xs": P_X}, extrapolant="cubic"), "extrapolant"),
    ]
    for kwargs, msg in bad:
        with pytest.raises(MethodIncompatibility, match=msg):
            sp.simex("yb ~ xs + z + w", df, **kwargs)
    with pytest.raises(MethodIncompatibility, match="names no"):
        sp.simex("yb ~ xs + z", df)
