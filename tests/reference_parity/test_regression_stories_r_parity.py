"""The regression workflow of Gelman, Hill and Vehtari against R.

Reference values come from ``_fixtures/_generate_regression_stories_R.R``
(``loo`` 2.9.0, ``rstanarm`` 2.32.2, ``arm`` 1.15-3, ``retrodesign``
0.2.2, base ``glm``). Everything compared here is a deterministic
function of its inputs, so the tolerances are numerical ones:

* PSIS-LOO, WAIC and their comparison, given a log-likelihood matrix:
  1e-9. The Monte Carlo standard error of ``elpd`` is derived here by
  the delta method and in ``loo`` by a different approximation; the two
  agree to a few parts in a thousand and are held to 2 percent.
* GLM fits (robit link, quasi families, grouped binomial, logical
  outcome, aliased regressors): 1e-6 on coefficients and standard
  errors, the cross-language gate, with ``information='expected'`` where
  the link is not canonical because that is what R reports.
* ``arm`` helpers and the closed-form design analysis: 1e-9.
* Prior scales and Bayesian R-squared: the operator applied to
  ``rstanarm``'s own draws, 1e-9.

The samplers themselves are checked against exact posteriors in
``test_bayes_workflow_exact_posterior.py``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.mcmc._workflow import weakly_informative_prior
from statspai.regression.glm import RobitLink

FIXTURE = Path(__file__).parent / "_fixtures" / "regression_stories_R.json"


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data(ref) -> pd.DataFrame:
    return pd.DataFrame(ref["data"])


def loglik(n: int, n_draws: int, outliers: bool):
    """The recipe of the R generator, with no random numbers."""
    i = np.arange(1, n + 1)
    x = stats.norm.ppf((i - 0.5) / n)
    y = 1 + 2 * x + 0.8 * np.sin(3 * i)
    if outliers:
        y[2] += 8
        y[n - 3] -= 6
    s = np.arange(1, n_draws + 1)

    def u(m):
        return ((s * m) % n_draws + 0.5) / n_draws

    a = 1 + 0.2 * stats.norm.ppf(u(7919))
    b = 2 + 0.2 * stats.norm.ppf(u(104729))
    sig = 0.8 * np.exp(0.15 * stats.norm.ppf(u(1299709)))
    ll = stats.norm.logpdf(
        y[None, :], a[:, None] + b[:, None] * x[None, :], sig[:, None]
    )
    ll2 = stats.norm.logpdf(y[None, :], a[:, None], 2.2 * sig[:, None])
    return ll, ll2


@pytest.mark.parametrize("case", ["clean", "outlier", "long"])
def test_psis_loo_matches_r_loo(ref, case):
    r = ref["loo"][case]
    ll, ll2 = loglik(r["n"], r["S"], r["outliers"])
    # the inputs really are the ones R used
    assert ll.sum() == pytest.approx(r["ll_checksum"][0], rel=1e-12)
    assert ll2.sum() == pytest.approx(r["ll_checksum"][1], rel=1e-12)
    assert ll[6, 2] == pytest.approx(r["ll_checksum"][2], rel=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.exceptions.StatsPAIWarning)
        for matrix, key in ((ll, "a"), (ll2, "b")):
            out = sp.loo(matrix, r_eff=r["r_eff"])
            want = r[key]
            np.testing.assert_allclose(out.pareto_k, want["pareto_k"], atol=1e-9)
            np.testing.assert_allclose(out.pointwise["elpd"], want["elpd"], atol=1e-9)
            np.testing.assert_allclose(out.pointwise["p"], want["p"], atol=1e-9)
            np.testing.assert_allclose(
                out.estimates.to_numpy(), np.array(want["estimates"]), atol=1e-8
            )
            np.testing.assert_allclose(
                out.log_weights[:, 0], want["log_weights_first"], atol=1e-9
            )
            # loo reports NA for the effective size of an unusable weight set
            n_eff = np.array(want["n_eff"], dtype=float)
            ok = n_eff > 0
            np.testing.assert_allclose(
                out.pointwise["n_eff"].to_numpy()[ok], n_eff[ok], rtol=1e-9
            )
            # delta method here, another approximation in loo; compared
            # where the weights are usable, since past the threshold the
            # Monte Carlo error is not estimable by either
            good = out.pareto_k <= out.k_threshold
            np.testing.assert_allclose(
                out.pointwise["mcse_elpd"].to_numpy()[good],
                np.array(want["mcse"])[good],
                rtol=2e-2,
            )
            if want["mcse_total"] > 0:
                assert out.mcse_elpd == pytest.approx(want["mcse_total"], rel=2e-2)
            else:
                assert np.isnan(out.mcse_elpd)


@pytest.mark.parametrize("case", ["clean", "outlier", "long"])
def test_waic_and_comparison_match_r_loo(ref, case):
    r = ref["loo"][case]
    ll, ll2 = loglik(r["n"], r["S"], r["outliers"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.exceptions.StatsPAIWarning)
        w = sp.waic(ll)
        cmp = sp.loo_compare(
            sp.loo(ll, r_eff=r["r_eff"]),
            sp.loo(ll2, r_eff=r["r_eff"]),
            names=["model1", "model2"],
        )
    np.testing.assert_allclose(w.estimates.to_numpy(), r["waic_estimates"], atol=1e-8)
    np.testing.assert_allclose(w.pointwise["elpd"], r["waic_elpd"], atol=1e-9)
    np.testing.assert_allclose(w.pointwise["p"], r["waic_p"], atol=1e-9)
    assert list(cmp.index) == r["compare_rows"]
    np.testing.assert_allclose(
        cmp[["elpd_diff", "se_diff"]].to_numpy(), r["compare"], atol=1e-8
    )


def test_k_threshold_depends_on_the_number_of_draws(ref):
    # loo prints 0.62 for 400 draws and 0.7 from 2,200 on
    assert sp.loo(loglik(20, 400, False)[0]).k_threshold == pytest.approx(
        1 - 1 / np.log10(400)
    )
    assert sp.loo(loglik(6, 2300, False)[0]).k_threshold == pytest.approx(0.7)


# ---------------------------------------------------------------------
# GLM pieces
# ---------------------------------------------------------------------


def _fit_matches(result, want, rtol=1e-6):
    np.testing.assert_allclose(
        np.asarray(result.params), want["est"], rtol=rtol, atol=1e-8
    )
    np.testing.assert_allclose(np.asarray(result.std_errors), want["se"], rtol=rtol)


@pytest.mark.parametrize(
    "link, key",
    [("robit(4)", "robit4"), ("robit(7)", "robit7"), ("robit(4,unit)", "robit4_unit")],
)
def test_robit_link_matches_r_glm(ref, data, link, key):
    fit = sp.glm(
        "y ~ x + z",
        data,
        family="binomial",
        link=link,
        information="expected",
        tol=1e-13,
        maxiter=500,
    )
    _fit_matches(fit, ref[key])


def test_robit_limits_and_scale():
    eta = np.linspace(-4, 4, 41)
    # one degree of freedom is the Cauchy link, many the probit link
    np.testing.assert_allclose(RobitLink(1.0).inverse(eta), stats.cauchy.cdf(eta))
    np.testing.assert_allclose(
        RobitLink(1e7).inverse(eta), stats.norm.cdf(eta), atol=1e-7
    )
    unit = RobitLink(5.0, unit_variance=True)
    np.testing.assert_allclose(unit.link(unit.inverse(eta)), eta, atol=1e-9)
    mu = np.linspace(0.02, 0.98, 25)
    step = 1e-6
    numeric = (unit.link(mu + step) - unit.link(mu - step)) / (2 * step)
    np.testing.assert_allclose(unit.deriv(mu), numeric, rtol=1e-6)


def test_quasi_families_match_r_glm(ref, data):
    qp = sp.glm("cnt ~ x + z", data, family="quasipoisson", tol=1e-13)
    _fit_matches(qp, ref["quasipoisson"])
    qb = sp.glm("y ~ x + z", data, family="quasibinomial", tol=1e-13)
    _fit_matches(qb, ref["quasibinomial"])
    # same coefficients as the base family, standard errors scaled by the
    # square root of the Pearson dispersion
    base = sp.glm(
        "cnt ~ x + z", data, family="poisson", information="expected", tol=1e-13
    )
    np.testing.assert_allclose(qp.params, base.params, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(qp.std_errors) / np.asarray(base.std_errors),
        np.sqrt(ref["quasipoisson"]["dispersion"]),
        rtol=1e-6,
    )


def test_grouped_binomial_matches_r_glm(ref, data):
    fit = sp.glm(
        "cbind(succ, trials - succ) ~ x + z", data, family="binomial", tol=1e-13
    )
    _fit_matches(fit, ref["cbind"])
    assert fit.diagnostics["Deviance"] == pytest.approx(
        ref["cbind"]["deviance"], rel=1e-8
    )


def test_logical_outcome_matches_r_glm(ref, data):
    want = ref["logical"]
    for fit in (
        sp.logit("(yc > 1) ~ x + z", data, tol=1e-12),
        sp.logit("I(yc > 1) ~ x + z", data, tol=1e-12),
        sp.glm("(yc > 1) ~ x + z", data, family="binomial", tol=1e-13),
    ):
        _fit_matches(fit, want)


def _kept(want):
    return [n for n, a in zip(want["names"], want["aliased"]) if not a]


def test_aliased_regressors_are_omitted_as_in_r(ref, data):
    """R reports ``NA`` for the later member of a dependent set; the same
    member is omitted here and the others agree."""
    want = ref["alias_logit"]
    with pytest.warns(UserWarning, match="dc omitted because of collinearity"):
        fit = sp.logit("y ~ x + da + db + dc", data, tol=1e-12)
    assert list(fit.params.index)[1:] == _kept(want)[1:]
    _fit_matches(fit, want)
    assert fit.model_info["omitted"][0]["variable"] == "dc"

    want = ref["alias_poisson"]
    with pytest.warns(UserWarning, match="omitted because of collinearity"):
        fit = sp.poisson("cnt ~ x + z + I(x + 2 * z)", data)
    assert len(fit.params) == len(_kept(want))
    _fit_matches(fit, want, rtol=1e-5)

    want = ref["alias_lm_dot"]
    with pytest.warns(UserWarning, match="dc omitted"):
        fit = sp.regress("yc ~ . - g - y - cnt - trials - succ", data)
    assert list(fit.params.index)[1:] == _kept(want)[1:]
    _fit_matches(fit, want, rtol=1e-9)


# ---------------------------------------------------------------------
# arm helpers
# ---------------------------------------------------------------------

_BINNED = ["xbar", "ybar", "n", "x_lo", "x_hi", "two_se"]


def test_binned_residuals_match_arm(ref, data):
    fit = sp.logit("y ~ x + z", data, tol=1e-12)
    want = ref["binned"]
    np.testing.assert_allclose(
        sp.binned_residuals(fit)[_BINNED].to_numpy(), want["default"], atol=1e-8
    )
    np.testing.assert_allclose(
        sp.binned_residuals(fit, n_bins=12)[_BINNED].to_numpy(), want["n12"], atol=1e-8
    )
    np.testing.assert_allclose(
        sp.binned_residuals(fit, n_bins=10, by=data["x"])[_BINNED].to_numpy(),
        want["by_x"],
        atol=1e-8,
    )
    # tied values of the binning variable stay in one bin
    tied = sp.binned_residuals(want["ties_x"], want["ties_r"], n_bins=7)
    np.testing.assert_allclose(tied[_BINNED].to_numpy(), want["ties"], atol=1e-12)


def test_standardize_matches_arm_rescale(ref, data):
    want = ref["rescale"]
    np.testing.assert_allclose(sp.standardize(data["x"]), want["x"], atol=1e-12)
    np.testing.assert_allclose(sp.standardize(data["y"]), want["y_center"], atol=1e-12)
    np.testing.assert_allclose(
        sp.standardize(data["y"], binary="full"), want["y_full"], atol=1e-12
    )
    np.testing.assert_allclose(
        sp.standardize(data["y"] + 3, binary="0/1"), want["y_01"], atol=1e-12
    )
    np.testing.assert_allclose(
        sp.standardize(data["y"], binary="-0.5/0.5"), want["y_half"], atol=1e-12
    )


def test_refit_on_standardized_inputs_matches_arm_standardize(ref, data):
    formula = "yc ~ x + z + y + g + x:y"
    fit = sp.regress(formula, sp.standardize(data, formula=formula))
    want = dict(zip(ref["standardize"]["names"], ref["standardize"]["est"]))
    mine = fit.params
    pairs = {
        "Intercept": "(Intercept)",
        "x": "z.x",
        "z": "z.z",
        "y": "c.y",
        "g[T.b]": "gb",
        "g[T.c]": "gc",
        "x:y": "z.x:c.y",
    }
    for here, there in pairs.items():
        assert mine[here] == pytest.approx(want[there], rel=1e-9, abs=1e-10)


# ---------------------------------------------------------------------
# design analysis
# ---------------------------------------------------------------------


def test_retrodesign_matches_the_closed_form_of_the_r_package(ref):
    want = ref["retro"]
    out = sp.retrodesign(want["effect"], want["se"]).table
    np.testing.assert_allclose(
        out[["power", "type_s", "type_m"]].to_numpy(), want["closed"], rtol=1e-9
    )


def test_retrodesign_with_degrees_of_freedom(ref):
    want = ref["retro"]
    exact = sp.retrodesign(0.5, 1.0, dof=20)
    # power and sign error of a t-test are noncentral-t probabilities,
    # which is what the R package computes
    assert exact.power == pytest.approx(want["df20_power"], rel=1e-9)
    assert exact.type_s == pytest.approx(want["df20_type_s"], rel=1e-9)
    # the published Gelman-Carlin function shifts a central t instead
    shifted = sp.retrodesign(0.5, 1.0, dof=20, method="shifted")
    z = stats.t.ppf(0.975, 20)
    assert shifted.power == pytest.approx(
        stats.t.sf(z - 0.5, 20) + stats.t.cdf(-z - 0.5, 20), rel=1e-12
    )
    # many degrees of freedom: both reduce to the normal case
    normal = sp.retrodesign(0.5, 1.0)
    for method in ("exact", "shifted"):
        big = sp.retrodesign(0.5, 1.0, dof=1e7, method=method)
        assert big.type_m == pytest.approx(normal.type_m, rel=1e-5)
        assert big.power == pytest.approx(normal.power, rel=1e-5)


def test_exact_exaggeration_ratio_is_what_a_simulated_t_test_gives():
    """The estimate is normal, its standard error is estimated with 5
    degrees of freedom, significance is |estimate / se_hat| > t critical.
    One million simulated studies; tolerance four Monte Carlo standard
    errors of each quantity."""
    rng = np.random.default_rng(20261007)
    effect, se, dof, n = 0.8, 1.0, 5, 1_000_000
    est = effect + se * rng.standard_normal(n)
    se_hat = se * np.sqrt(rng.chisquare(dof, n) / dof)
    sig = np.abs(est) > stats.t.ppf(0.975, dof) * se_hat
    out = sp.retrodesign(effect, se, dof=dof)
    power = sig.mean()
    assert out.power == pytest.approx(power, abs=4 * np.sqrt(power * (1 - power) / n))
    wrong = (est[sig] < 0).mean()
    assert out.type_s == pytest.approx(
        wrong, abs=4 * np.sqrt(wrong * (1 - wrong) / sig.sum())
    )
    ratio = np.abs(est[sig]) / effect
    assert out.type_m == pytest.approx(
        ratio.mean(), abs=4 * ratio.std() / np.sqrt(sig.sum())
    )
    # and the shifted-t approximation is measurably different here
    assert (
        abs(sp.retrodesign(effect, se, dof=dof, method="shifted").type_m - ratio.mean())
        > 0.1
    )


# ---------------------------------------------------------------------
# rstanarm conventions
# ---------------------------------------------------------------------


def test_weakly_informative_prior_scales_match_rstanarm(ref, data):
    from statspai.core.utils import create_design_matrices

    sub = data.iloc[:120]
    y_df, X_df = create_design_matrices("yc ~ x + z + da + x:z", sub)
    names = [str(c) for c in X_df.columns]
    X = np.asarray(X_df, dtype=float)
    y = np.asarray(y_df, dtype=float).ravel()
    mean, cov, info = weakly_informative_prior("normal", y, X, names)
    want = ref["prior_gaussian"]
    got = [info["scale"][n] for n in want["names"]]
    np.testing.assert_allclose(got, want["coef_scale"], rtol=1e-9)
    # the intercept prior refers to centred regressors: undo the centring
    xbar = X.mean(axis=0)
    centred_var = xbar @ cov @ xbar
    assert np.sqrt(centred_var) == pytest.approx(want["intercept_scale"], rel=1e-9)
    assert xbar @ mean == pytest.approx(want["intercept_location"][0], rel=1e-9)
    assert 1.0 / y.std(ddof=1) == pytest.approx(want["sigma_rate"], rel=1e-9)

    y_df, X_df = create_design_matrices("y ~ x + z + da", sub)
    X = np.asarray(X_df, dtype=float)
    names = [str(c) for c in X_df.columns]
    mean, cov, info = weakly_informative_prior(
        "logit", sub["y"].to_numpy(float), X, names
    )
    np.testing.assert_allclose(
        [info["scale"][n] for n in ("x", "z", "da")],
        ref["prior_binomial"]["coef_scale"],
        rtol=1e-9,
    )
    xbar = X.mean(axis=0)
    assert np.sqrt(xbar @ cov @ xbar) == pytest.approx(
        ref["prior_binomial"]["intercept_scale"], rel=1e-9
    )
    assert np.allclose(mean, 0.0)


def test_bayes_r2_operator_matches_rstanarm(ref):
    """``Var(mu) / (Var(mu) + residual variance)`` applied to rstanarm's
    own draws reproduces ``bayes_R2``: sigma^2 for the Gaussian model, the
    mean of ``mu (1 - mu)`` for the binomial one."""
    from statspai.mcmc import _workflow as W

    class _Stub:
        def __init__(self, name):
            self.name = name

    g = ref["r2_gaussian"]
    mu = np.array(g["mu"])
    draws = np.column_stack([np.zeros(len(mu)), np.array(g["sigma"]) ** 2])
    var_fit = mu.var(axis=1, ddof=1)
    var_res = W.residual_variance(_Stub("normal"), draws, mu)
    np.testing.assert_allclose(var_fit / (var_fit + var_res), g["r2"], rtol=1e-9)

    b = ref["r2_binomial"]
    mu = np.array(b["mu"])
    var_fit = mu.var(axis=1, ddof=1)
    var_res = W.residual_variance(_Stub("logit"), np.zeros((len(mu), 1)), mu)
    np.testing.assert_allclose(var_fit / (var_fit + var_res), b["r2"], rtol=1e-9)
