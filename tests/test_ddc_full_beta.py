"""sp.ddc: Rust's third stage (stage='full') and the discount factor (T1).

stage='full' maximises the full likelihood of choices and state increments
jointly over the utility parameters and the increment probabilities.
Checks: the analytic gradient against finite differences, a zero gradient
at the estimate, block-diagonality at beta = 0, the Schur-complement
ordering of the variances, the partial MLE at the full increment estimate
reproducing the full utility estimate, and recovery of the truth.

beta as a list or 'estimate' profiles the likelihood over beta; every
profile point is an ordinary fit at that beta, and beta='estimate'
recovers the discount factor where replacement is frequent enough to
identify it.
"""

import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning, MethodIncompatibility
from statspai.structural.ddc import _data, _full

TRUTH = np.array([10.0, 2.5])
P_TRUE = np.array([0.35, 0.60, 0.05])


@pytest.fixture(scope="module")
def big_panel():
    return sp.dgp_ddc(n_units=2000, n_periods=200, seed=11)


@pytest.fixture(scope="module")
def small_panel():
    return sp.dgp_ddc(n_units=200, n_periods=100, seed=0)


def _fit(df, **kw):
    kw.setdefault("beta", 0.9999)
    kw.setdefault("n_states", 90)
    return sp.ddc(df, "replace", "mileage", id="bus", time="t", **kw)


def _model(df, n_states=90):
    enc = _data.encode_panel(df, "replace", "mileage", "bus", "t", None, n_states)
    phi, _ = _data.build_features("rust", n_states, 2, "linear", 0.001, None)
    inc = _full.row_increments(enc, [1])
    return enc, phi, _full.IncrementModel(inc, n_states, 2, [1]), inc


# ----------------------------------------------------------------------
# stage='full'
# ----------------------------------------------------------------------


def test_full_gradient_equals_finite_differences(small_panel):
    enc, phi, model, _ = _model(small_panel)
    z = np.r_[9.0, 3.0, 0.33, 0.62]  # away from the optimum
    _, g = _full.full_loglik(phi, model, 0.99, z, enc["counts"])
    num = np.empty_like(z)
    for k in range(z.size):
        h = 1e-6 * max(1.0, abs(z[k]))
        e = np.zeros_like(z)
        e[k] = h
        up = _full.full_loglik(phi, model, 0.99, z + e, enc["counts"])[0]
        dn = _full.full_loglik(phi, model, 0.99, z - e, enc["counts"])[0]
        num[k] = (up - dn) / (2 * h)
    # central differences of a ~1e4 log-likelihood: error ~ 1e-6 relative
    np.testing.assert_allclose(g, num, rtol=1e-5)


def test_full_gradient_is_zero_at_the_estimate(small_panel):
    fit = _fit(small_panel, stage="full")
    assert fit.converged and fit.diagnostics["stage"] == "full"
    enc, phi, model, _ = _model(small_panel)
    q = np.asarray(fit.transition_params["increment_probs"])[model.free]
    _, g = _full.full_loglik(
        phi, model, 0.9999, np.r_[fit.params.to_numpy(), q], enc["counts"]
    )
    # 20,000 observations; a converged Newton ascent leaves rounding noise
    assert np.max(np.abs(g)) < 1e-5


def test_partial_mle_at_full_increments_is_the_full_estimate(small_panel):
    """The utility score of the full likelihood is the choice score."""
    full = _fit(small_panel, stage="full")
    p = np.asarray(full.transition_params["increment_probs"])
    part = _fit(small_panel, transition=p)
    np.testing.assert_allclose(part.params, full.params, rtol=1e-7)
    # Schur complement: estimating the increments can only add variance
    assert np.all(full.se.to_numpy() >= part.se.to_numpy())
    assert np.all(full.se.to_numpy() > part.se.to_numpy() * (1 + 1e-6))


def test_beta_zero_is_block_diagonal(small_panel):
    """At beta = 0 choices do not depend on the increments."""
    full = _fit(small_panel, beta=0.0, stage="full")
    two = _fit(small_panel, beta=0.0)
    np.testing.assert_allclose(full.params, two.params, rtol=1e-8)
    np.testing.assert_allclose(full.se, two.se, rtol=1e-6)
    _, _, model, _ = _model(small_panel)
    freq = model.n / model.n.sum()
    np.testing.assert_allclose(
        full.transition_params["increment_probs"], freq, rtol=0, atol=1e-10
    )
    V = np.asarray(full.diagnostics["vcov_full"]["matrix"])
    np.testing.assert_allclose(V[:2, 2:], 0.0, atol=1e-8 * np.abs(V).max())


def test_full_increment_se_is_multinomial_at_beta_zero(small_panel):
    full = _fit(small_panel, beta=0.0, stage="full", vce="hessian")
    p = np.asarray(full.transition_params["increment_probs"])
    n = full.transition_params["n_transitions"]
    np.testing.assert_allclose(
        full.transition_params["increment_se"], np.sqrt(p * (1 - p) / n), rtol=1e-5
    )


def test_full_recovers_the_truth(big_panel):
    fit = _fit(big_panel, stage="full")
    assert fit.converged
    z = np.abs(fit.params.to_numpy() - TRUTH) / fit.se.to_numpy()
    assert np.all(z < 4), z
    p = np.asarray(fit.transition_params["increment_probs"])
    se = np.asarray(fit.transition_params["increment_se"])
    assert np.all(np.abs(p - P_TRUE) < 4 * se)
    # the full log-likelihood is the choice part plus the increment part
    d = fit.diagnostics
    assert abs(d["loglik_full"] - (fit.loglik + d["loglik_transition"])) < 1e-8
    assert set(d["two_step_params"]) == {"RC", "theta_1"}


@pytest.mark.parametrize("vce", ["hessian", "opg", "sandwich"])
def test_full_variance_types_agree_roughly(small_panel, vce):
    """Information-matrix equality: the three variances are close."""
    ref = _fit(small_panel, stage="full", vce="hessian").se.to_numpy()
    se = _fit(small_panel, stage="full", vce=vce).se.to_numpy()
    np.testing.assert_allclose(se, ref, rtol=0.25)


def test_increment_column_equals_supplied_frequencies(small_panel):
    df = small_panel.sort_values(["bus", "t"]).reset_index(drop=True)
    nxt = df.groupby("bus")["mileage"].shift(-1)
    base = np.where(df["replace"] == 1, 0, df["mileage"])
    df["inc"] = nxt - base
    a = _fit(df, increment="inc")
    inc = df["inc"].dropna().astype(int)
    b = _fit(df, transition=np.bincount(inc) / inc.size)
    np.testing.assert_array_equal(a.params.to_numpy(), b.params.to_numpy())
    assert a.transition_params["mode"] == "increments_column"
    # outgoing increments from the panel: the same full-likelihood estimate
    full_col = _fit(df, increment="inc", stage="full")
    full_pan = _fit(df, stage="full")
    np.testing.assert_allclose(full_col.params, full_pan.params, rtol=1e-9)


def test_partial_default_has_no_stage_keys(small_panel):
    fit = _fit(small_panel)
    assert "stage" not in fit.diagnostics and fit.beta_profile is None
    assert "conditional on the first-stage" in fit.summary()
    assert "transitions estimated jointly" in _fit(small_panel, stage="full").summary()


# ----------------------------------------------------------------------
# beta
# ----------------------------------------------------------------------


def test_beta_grid_points_are_ordinary_fits(small_panel):
    grid = [0.0, 0.9, 0.99, 0.9999]
    best = _fit(small_panel, beta=grid)
    prof = best.beta_profile
    assert list(prof.index) == grid
    one = _fit(small_panel, beta=0.99)
    np.testing.assert_array_equal(
        prof.loc[0.99, ["RC", "theta_1"]].to_numpy(dtype=float), one.params.to_numpy()
    )
    assert prof.loc[0.99, "loglik"] == one.loglik
    assert best.beta == grid[int(np.argmax(prof["loglik"]))]
    assert best.diagnostics["beta_search"] == "grid"


def test_beta_grid_with_full_stage_profiles_the_full_likelihood(small_panel):
    best = _fit(small_panel, beta=[0.9, 0.9999], stage="full")
    one = _fit(small_panel, beta=0.9, stage="full")
    assert best.beta_profile.loc[0.9, "loglik"] == one.diagnostics["loglik_full"]


def test_beta_estimate_recovers_the_discount_factor():
    # frequent replacement (3 per cent of months) identifies beta
    df = sp.dgp_ddc(n_units=2000, n_periods=200, seed=3, beta=0.9, RC=6.0, theta_1=20.0)
    fit = _fit(df, beta="estimate")
    d = fit.diagnostics
    lo, hi = d["beta_ci"]
    assert lo < 0.9 < hi and hi - lo < 0.1
    assert abs(fit.beta - 0.9) < 0.03 and not d["beta_at_bound"]
    assert fit.beta == d["beta_hat"]
    # the estimate maximises the profile: no evaluated point is higher
    assert d["beta_loglik"] >= max(fit.beta_profile["loglik"]) - 1e-9
    # parameters at beta-hat recover the truth within 4 SE
    z = np.abs(fit.params.to_numpy() - [6.0, 20.0]) / fit.se.to_numpy()
    assert np.all(z < 4)


def test_beta_estimate_at_the_boundary_warns(small_panel):
    """With beta = .9999 in the DGP the profile rises to the upper end."""
    with pytest.warns(AssumptionWarning, match="boundary"):
        fit = _fit(small_panel, beta="estimate")
    assert fit.diagnostics["beta_at_bound"]
    assert fit.beta > 0.9999


# ----------------------------------------------------------------------
# argument errors
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(stage="full", method="ccp"), "nested fixed point"),
        (dict(stage="full", transition="estimate"), "not increments"),
        (dict(stage="full", transition=np.array([0.4, 0.6])), "not increments"),
        (dict(stage="three"), "not supported"),
        (dict(beta=[0.5, 1.0]), "outside"),
        (dict(beta=[]), "empty"),
        (dict(beta="max"), "not supported"),
    ],
)
def test_argument_errors(small_panel, kw, match):
    with pytest.raises(MethodIncompatibility, match=match):
        _fit(small_panel, **kw)


def test_increment_column_conflicts_with_supplied_transitions(small_panel):
    df = small_panel.assign(inc=1.0)
    with pytest.raises(MethodIncompatibility, match="cannot be combined"):
        _fit(df, increment="inc", transition=np.array([0.4, 0.6]))
    with pytest.raises(MethodIncompatibility, match="non-negative integers"):
        _fit(small_panel.assign(inc=-1.0), increment="inc", stage="full")
    with pytest.raises(MethodIncompatibility, match="only one increment"):
        _fit(df, increment="inc", stage="full")


def test_full_stage_without_panel_or_column_is_refused(small_panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(MethodIncompatibility, match="panel structure"):
            sp.ddc(
                small_panel,
                "replace",
                "mileage",
                beta=0.9,
                n_states=90,
                transition="increments",
                stage="full",
            )
