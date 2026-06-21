"""Coverage for previously-untested transportability and shift-share helpers:
``sp.pate``, ``sp.transport_generalize``, ``sp.ssaggregate``,
``sp.shift_share_se`` and a structural smoke test for ``sp.fci``.

The transport tests are built so that ground truth is known: the experimental
CATE is ``1 + 0.5*x`` and the target population is shifted to a higher mean of
``x``. A correct transport must therefore report an effect *above* the source
SATE (~1.0), since the target oversamples high-CATE units.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def experiment_and_target():
    rng = np.random.RandomState(0)
    n = 1200
    x = rng.randn(n)
    t = rng.binomial(1, 0.5, n)
    # CATE(x) = 1 + 0.5*x ; SATE over x~N(0,1) is ~1.0
    y = 1.0 * t + 0.5 * t * x + x + rng.randn(n)
    exp = pd.DataFrame({"y": y, "t": t, "x": x})
    target = pd.DataFrame({"x": rng.randn(2000) + 1.0})  # mean shifted to +1
    return exp, target


# --------------------------------------------------------------------------
# pate — population average treatment effect via reweighting
# --------------------------------------------------------------------------
def test_pate_transports_above_source_sate(experiment_and_target):
    exp, target = experiment_and_target
    res = sp.pate(
        data_experiment=exp,
        data_target=target,
        y="y",
        treatment="t",
        covariates=["x"],
        seed=0,
        n_boot=200,
    )
    # Target oversamples high-CATE units (x shifted up by 1) so the PATE must
    # exceed the homogeneous SATE benchmark of ~1.0; truth is ~1.5.
    assert 1.2 < res.estimate < 2.2
    lo, hi = res.ci
    assert lo < hi
    assert lo <= res.estimate <= hi


@pytest.mark.parametrize("method", ["ipw", "aipw", "calibration"])
def test_pate_recovers_constant_treatment_effect(method):
    x = np.repeat(np.array([-1.0, 0.0, 1.0, 2.0]), 12)
    t = np.tile(np.array([0.0, 1.0]), len(x) // 2)
    y = 10.0 + 0.25 * x + 1.75 * t
    exp = pd.DataFrame({"y": y, "t": t, "x": x})
    target = pd.DataFrame({"x": np.linspace(-2.0, 3.0, 51)})

    res = sp.pate(
        data_experiment=exp,
        data_target=target,
        y="y",
        treatment="t",
        covariates=["x"],
        method=method,
        n_boot=0,
    )

    np.testing.assert_allclose(res.estimate, 1.75, atol=1e-10)


def test_transport_generalize_direction_and_weights(experiment_and_target):
    exp, target = experiment_and_target
    tr = sp.transport_generalize(
        rct=exp,
        target_population=target,
        features=["x"],
        treatment="t",
        outcome="y",
    )
    # Transported effect exceeds the source effect for the shifted target.
    assert tr.effect_transported > tr.effect_source
    # Reweighting can only shrink the effective sample size.
    assert 0 < tr.ess <= len(exp)
    w = np.asarray(tr.weights)
    assert len(w) == len(exp)
    assert np.all(w >= 0)
    assert tr.max_weight >= 1.0


# --------------------------------------------------------------------------
# shift-share / Bartik
# --------------------------------------------------------------------------
@pytest.fixture
def bartik():
    return sp.dgp_bartik(n_regions=80, n_industries=10, effect=1.5, seed=0)


def test_ssaggregate_recovers_effect(bartik):
    res = sp.ssaggregate(
        data=bartik["data"],
        y="y",
        x="bartik",
        shares=bartik["shares"],
        shocks=bartik["shocks"],
    )
    assert float(res.params["bartik"]) == pytest.approx(1.5, abs=0.5)
    assert float(res.std_errors["bartik"]) > 0


def test_shift_share_se_preserves_point_estimate(bartik):
    base = sp.ssaggregate(
        data=bartik["data"],
        y="y",
        x="bartik",
        shares=bartik["shares"],
        shocks=bartik["shocks"],
    )
    akm = sp.shift_share_se(base, shares=bartik["shares"])
    # The AKM correction only re-estimates the variance — point estimates
    # are untouched, and the corrected SE remains a positive number.
    np.testing.assert_allclose(akm.params.values, base.params.values)
    assert float(akm.std_errors["bartik"]) > 0


# --------------------------------------------------------------------------
# fci — Fast Causal Inference (structural smoke test)
# --------------------------------------------------------------------------
def test_fci_returns_structured_pag():
    rng = np.random.RandomState(1)
    n = 400
    a = rng.randn(n)
    b = rng.randn(n)
    df = pd.DataFrame({"a": a, "b": b, "c": a + b + 0.3 * rng.randn(n)})
    res = sp.fci(df, alpha=0.05)
    assert res.n_obs == n
    # A PAG is reported as an edge list / adjacency we can enumerate.
    assert isinstance(res.edges, (list, tuple, np.ndarray, pd.DataFrame))


def test_shift_share_se_no_divide_by_zero_on_strong_first_stage(bartik):
    # Regression: a (near-)perfect first stage made the AKM first-stage F
    # divide by a ~zero residual sum of squares, raising a RuntimeWarning and
    # producing a non-finite statistic. The corrected code reports F = inf.
    import warnings

    base = sp.ssaggregate(
        data=bartik["data"],
        y="y",
        x="bartik",
        shares=bartik["shares"],
        shocks=bartik["shocks"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        akm = sp.shift_share_se(base, shares=bartik["shares"])
    assert np.all(np.isfinite(akm.std_errors.values))
