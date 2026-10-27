"""Tests for ``sp.semiparametric_bayes`` (Mueller and Norets 2026).

Known-truth checks follow the paper's Monte Carlo designs: a Student-t
regression under local skewness (Section 5.2, here with the skewness
proportional to ``x`` so that it biases the slope; with ``h(x) = x^2`` and
symmetric ``x`` only the intercept is biased, since ``E[h(x) x] = 0``) and a
correctly specified Weibull regression (Section 5.1). The sampler itself is
checked against an exact grid posterior in
``tests/reference_parity/test_semiparametric_bayes_exact_posterior.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    ConvergenceWarning,
    MethodIncompatibility,
)


def skewed_t(lam: np.ndarray, eta: float, rng) -> np.ndarray:
    """Hansen (1994) skewed t: mean 0, variance 1, skewness ``lam``."""
    c = np.exp(special.gammaln((eta + 1) / 2) - special.gammaln(eta / 2)) / np.sqrt(
        np.pi * (eta - 2)
    )
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(1 + 3 * lam**2 - a**2)
    t = np.abs(rng.standard_t(eta, size=lam.size)) * np.sqrt((eta - 2) / eta)
    neg = rng.random(lam.size) < (1 - lam) / 2
    w = np.where(neg, -(1 - lam) * t, (1 + lam) * t)
    return (w - a) / b


def test_skewed_t_generator_is_standardised():
    rng = np.random.default_rng(0)
    e = skewed_t(np.full(400_000, 0.5), 6.0, rng)
    # mean 0 and variance 1 by Hansen's construction; MC error ~ 0.002
    assert abs(e.mean()) < 0.01
    assert abs(e.var() - 1) < 0.03
    assert np.mean(e > 0) < 0.5  # right skew puts the median below the mean


@pytest.fixture(scope="module")
def local_fit():
    n = 1000
    rng = np.random.default_rng(100)
    x = rng.normal(size=n)
    lam = np.clip(8 * x / np.sqrt(n), -0.95, 0.95)
    df = pd.DataFrame({"x": x, "y": 1 + x + skewed_t(lam, 4.0, rng)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.semiparametric_bayes("y ~ x", df, draws=3000, burnin=1000, seed=0)
    return fit, df


def test_local_misspecification_baseline_biased_augmented_not(local_fit):
    """Paper Section 3.2 / Theorem 1: under local skewness the Student-t
    posterior of the slope is biased; the augmented one is centred near the
    efficient estimator (OLS) and the truth (1)."""
    fit, _ = local_fit
    c = fit.diagnostics_info["comparison"].loc["x"]
    se = c["efficient_se"]
    # baseline posterior sits more than 2 SE from the truth and more than one
    # SE from OLS (it was 2.4 SE on average over six datasets of this design)
    assert c["baseline_mean"] - 1 < -2 * se
    assert abs(c["baseline_mean"] - c["efficient"]) > se
    # the augmented posterior is within one SE of the truth, and closer to
    # OLS than the baseline (the N(0, 0.2^2) prior on delta leaves some pull
    # toward the baseline in finite samples, as in the paper's Table 2)
    assert abs(c["augmented_mean"] - 1) < se
    assert abs(c["augmented_mean"] - c["efficient"]) < abs(
        c["baseline_mean"] - c["efficient"]
    )


def test_local_misspecification_posterior_sd_at_efficiency_bound(local_fit):
    fit, _ = local_fit
    c = fit.diagnostics_info["comparison"].loc["x"]
    # asymptotically sd(beta | Y) = efficient SE; MC error in the sd is a few
    # percent at ESS ~ 200, and the SE itself is estimated
    assert 0.85 < c["augmented_sd"] / c["efficient_se"] < 1.25
    # the t likelihood claims more precision than the efficient estimator has
    assert c["baseline_sd"] < c["augmented_sd"]


def test_delta_flags_the_misspecification(local_fit):
    fit, _ = local_fit
    assert bool(fit.diagnostics_info["delta_excludes_zero"]["x"])
    assert any("delta excludes zero" in w for w in fit.diagnostics_info["warnings"])


def test_efficient_estimate_is_ols_with_classical_se(local_fit):
    fit, df = local_fit
    X = np.column_stack([np.ones(len(df)), df["x"]])
    b, *_ = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)
    r = df["y"].to_numpy() - X @ b
    V = (r @ r) / (len(df) - 2) * np.linalg.inv(X.T @ X)
    comp = fit.diagnostics_info["comparison"]
    np.testing.assert_allclose(comp["efficient"], b, rtol=1e-10)
    np.testing.assert_allclose(comp["efficient_se"], np.sqrt(np.diag(V)), rtol=1e-10)


def test_result_layout(local_fit):
    fit, _ = local_fit
    assert isinstance(fit, sp.mcmc.semiparametric.SemiparametricBayesResult)
    assert list(fit.params.index) == [
        "Intercept",
        "x",
        "sigma",
        "nu",
        "delta:Intercept",
        "delta:x",
    ]
    assert fit.draws.shape == (3000, 6)
    assert (fit.draws["nu"] > 2).all() and (fit.draws["sigma"] > 0).all()
    assert fit.cite().split("\n") == ["mueller2026locally", "rao2016data"]
    assert "Locally robust" in fit.method
    assert isinstance(fit.summary(), str)


def test_weibull_correct_specification():
    """Paper Section 5.1 design (alpha = 0.5, beta = (1, 1)): both posteriors
    are centred near the efficient estimator and the augmented sd is near
    its standard error."""
    n, alpha = 500, 0.5
    rng = np.random.default_rng(200)
    x = rng.normal(size=n)
    lam = np.exp(1 + x) / special.gamma(1 + 1 / alpha)
    df = pd.DataFrame({"x": x, "y": lam * rng.weibull(alpha, size=n)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.semiparametric_bayes(
            "y ~ x", df, model="weibull", draws=3000, burnin=1000, seed=0
        )
    comp = fit.diagnostics_info["comparison"]
    se = comp["efficient_se"]
    # within half an SE of the efficient estimator (0.33 SE at most over four
    # datasets of this design) and of the truth within 2 SE
    assert (np.abs(comp["augmented_mean"] - comp["efficient"]) < 0.5 * se).all()
    assert (np.abs(comp["augmented_mean"] - 1.0) < 2 * se).all()
    assert ((comp["augmented_sd"] / se).between(0.75, 1.35)).all()
    assert abs(fit.params["alpha"] - alpha) < 4 * fit.std_errors["alpha"]
    assert not fit.diagnostics_info["delta_excludes_zero"].any()


def test_tiny_delta_prior_gives_back_the_baseline_posterior():
    """Q(theta, 0) = P(theta): with delta pinned at zero the augmented
    sampler (rejected proposals and all) must reproduce the baseline
    posterior."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"x": rng.normal(size=300)})
    df["y"] = 1 + 0.5 * df["x"] + rng.standard_t(4, size=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.semiparametric_bayes(
            "y ~ x", df, delta_prior_sd=1e-4, draws=4000, burnin=1500, seed=2
        )
    base = fit.diagnostics_info["baseline_posterior"]
    for nm in ("Intercept", "x", "sigma"):
        row = fit.table.loc[nm]
        # the baseline chain (independence Metropolis) has ESS of thousands,
        # so the augmented mcse dominates the difference
        assert abs(row["mean"] - base.loc[nm, "mean"]) < 4 * row["mcse"] + 1e-3
        assert abs(row["sd"] / base.loc[nm, "sd"] - 1) < 0.12


def test_interest_subset_reports_original_coefficients():
    """Nuisance regressors are partialled out while sampling; the reported
    nuisance coefficients must be mapped back (x and z correlate 0.8, so a
    missing back-transform would move z by about 0.8)."""
    rng = np.random.default_rng(11)
    n = 600
    z = rng.normal(size=n)
    x = 0.8 * z + 0.6 * rng.normal(size=n)
    df = pd.DataFrame({"x": x, "z": z})
    df["y"] = 1 + 1.0 * x + 0.5 * z + rng.standard_t(5, size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.semiparametric_bayes(
            "y ~ x + z", df, interest=["x"], draws=2500, burnin=1000, seed=3
        )
    assert list(fit.diagnostics_info["comparison"].index) == ["x"]
    assert [c for c in fit.params.index if c.startswith("delta")] == ["delta:x"]
    for nm, truth in (("Intercept", 1.0), ("x", 1.0), ("z", 0.5)):
        assert abs(fit.params[nm] - truth) < 4 * fit.std_errors[nm]
    # the same index: fitted values from the reported coefficients match OLS
    X = np.column_stack([np.ones(n), x, z])
    b, *_ = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)
    fitted = X @ fit.params[["Intercept", "x", "z"]].to_numpy()
    assert np.corrcoef(fitted, X @ b)[0, 1] > 0.999


# --------------------------------------------------------------------------
# custom model
# --------------------------------------------------------------------------

NU = 5.0


def _custom_parts():
    def loglik(th, y, X):
        e = y - th[0]
        return (
            special.gammaln((NU + 1) / 2)
            - special.gammaln(NU / 2)
            - 0.5 * np.log(NU * np.pi)
            - (NU + 1) / 2 * np.log1p(e * e / NU)
        )

    def score(th, y, X):
        e = y - th[0]
        return ((NU + 1) * e / (NU + e * e))[:, None]

    def efficient_score(th, y, X):
        # sample mean is efficient; var of t_5 is 5/3
        return ((y - th[0]) * (NU - 2) / NU)[:, None]

    def simulate(th, X, rng):
        return th[0] + rng.standard_t(NU, size=X.shape[0])

    def log_prior(th):
        return -0.5 * (th[0] / 10.0) ** 2

    return dict(
        loglik=loglik,
        score=score,
        efficient_score=efficient_score,
        simulate=simulate,
        log_prior=log_prior,
    )


def test_custom_location_model_one_step_is_the_sample_mean():
    rng = np.random.default_rng(5)
    y = 0.3 + rng.standard_t(NU, size=200)
    df = pd.DataFrame({"y": y})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.semiparametric_bayes(
            "y ~ 1",
            df,
            model="custom",
            theta_start=[0.0],
            param_names=["mu"],
            draws=2000,
            burnin=800,
            seed=1,
            **_custom_parts(),
        )
    comp = fit.diagnostics_info["comparison"].loc["mu"]
    gbar = fit.diagnostics_info["baseline_posterior"].loc["mu", "mean"]
    # the efficient score is linear in mu, so one Newton step lands exactly
    # on the sample mean; the sandwich variance at gbar is sum (y-gbar)^2/n^2
    assert comp["efficient"] == pytest.approx(y.mean(), abs=1e-7)
    assert comp["efficient_se"] == pytest.approx(
        np.sqrt(np.sum((y - gbar) ** 2)) / y.size, rel=1e-6
    )
    assert abs(fit.params["mu"] - y.mean()) < 0.5 * comp["efficient_se"]
    assert list(fit.params.index) == ["mu", "delta:mu"]


def test_custom_wrong_score_shape_is_refused():
    df = pd.DataFrame({"y": np.random.default_rng(0).normal(size=50)})
    parts = _custom_parts()
    parts["score"] = lambda th, y, X: np.zeros(y.size + 1)
    with pytest.raises(MethodIncompatibility, match="score returned shape"):
        sp.semiparametric_bayes("y ~ 1", df, model="custom", theta_start=[0.0], **parts)
    parts = _custom_parts()
    parts["efficient_score"] = lambda th, y, X: np.zeros((y.size, 2))
    with pytest.raises(MethodIncompatibility, match="efficient_score"):
        sp.semiparametric_bayes("y ~ 1", df, model="custom", theta_start=[0.0], **parts)


def test_custom_needs_all_callables():
    df = pd.DataFrame({"y": np.random.default_rng(0).normal(size=50)})
    with pytest.raises(MethodIncompatibility, match="needs the callables") as err:
        sp.semiparametric_bayes("y ~ 1", df, model="custom", theta_start=[0.0])
    assert err.value.recovery_hint


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------


def _small_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = 1 + df["x"] + rng.standard_t(4, size=n)
    return df


def test_short_chain_warns_and_records():
    with pytest.warns(ConvergenceWarning):
        fit = sp.semiparametric_bayes(
            "y ~ x", _small_df(), draws=30, burnin=0, seed=0, baseline=False
        )
    assert any("mixes slowly" in w for w in fit.diagnostics_info["warnings"])
    assert np.isnan(fit.diagnostics_info["comparison"]["baseline_mean"]).all()


def test_weibull_needs_positive_durations():
    df = _small_df()
    with pytest.raises(MethodIncompatibility, match="positive outcome"):
        sp.semiparametric_bayes("y ~ x", df.assign(y=df["y"] - 10), model="weibull")


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"model": "probit"}, "model must be"),
        ({"interest": ["w"]}, "not a coefficient"),
        ({"delta_prior_sd": 0.0}, "delta_prior_sd must be positive"),
        ({"level": 1.5}, "level must be"),
        ({"draws": 0}, "draws must be"),
    ],
)
def test_bad_arguments_are_refused(kwargs, msg):
    with pytest.raises(MethodIncompatibility, match=msg):
        sp.semiparametric_bayes("y ~ x", _small_df(), **kwargs)


def test_assumption_warning_class_used_for_delta_signal(local_fit):
    fit, df = local_fit
    with pytest.warns(AssumptionWarning, match="delta excludes zero"):
        sp.semiparametric_bayes(
            "y ~ x", df, draws=600, burnin=300, seed=0, baseline=False
        )
