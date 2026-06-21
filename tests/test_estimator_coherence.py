"""End-to-end numerical-coherence tests for three more flagship estimators:
``sp.synth``, ``sp.dml`` and ``sp.psm`` / ``sp.match``.

Each anchors to a DGP with a known treatment effect and a *biased* naive
benchmark, then asserts the estimator recovers the truth, corrects the bias,
and reports a CI bracketing it (plus the synth simplex-weight invariant).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# --------------------------------------------------------------------------
# Synthetic control
# --------------------------------------------------------------------------
def test_synth_recovers_effect_with_simplex_weights():
    df = sp.dgp_synth(
        n_units=20,
        n_periods=30,
        treated_unit=0,
        treatment_time=20,
        effect=5.0,
        seed=2,
    )
    r = sp.synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=0,
        treatment_time=20,
    )
    assert float(r.estimate) == pytest.approx(5.0, abs=1.0)
    lo, hi = r.ci
    assert lo <= 5.0 <= hi
    # Donor weights form a convex combination (simplex).
    w = r.model_info["weights"]["weight"].to_numpy()
    assert w.min() >= -1e-9
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Double machine learning (partially linear model)
# --------------------------------------------------------------------------
def test_dml_corrects_confounding_bias():
    rng = np.random.RandomState(0)
    n = 2000
    X = rng.randn(n, 5)
    g = X[:, 0] + np.sin(X[:, 1])  # nonlinear confounding
    d = g + rng.randn(n)  # treatment depends on X
    y = 1.0 * d + g + rng.randn(n)  # true effect 1.0
    cols = [f"x{i}" for i in range(5)]
    df = pd.DataFrame(X, columns=cols)
    df["d"] = d
    df["y"] = y

    naive = float(sp.regress("y ~ d", df).params["d"])
    r = sp.dml(df, y="y", d="d", X=cols, model="plr")
    est = float(r.estimate)

    assert abs(naive - 1.0) > 0.3  # OLS is confounded
    assert est == pytest.approx(1.0, abs=0.2)  # DML recovers the truth
    assert abs(est - 1.0) < abs(naive - 1.0)  # and is less biased
    lo, hi = r.ci
    assert lo <= 1.0 <= hi


# --------------------------------------------------------------------------
# Propensity-score matching (selection on observables)
# --------------------------------------------------------------------------
@pytest.fixture
def matching_data():
    rng = np.random.RandomState(1)
    n = 2000
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    ps = 1.0 / (1.0 + np.exp(-(0.8 * x1 + 0.5 * x2)))
    t = (rng.uniform(size=n) < ps).astype(int)
    y = 2.0 * t + 1.5 * x1 + 1.0 * x2 + rng.randn(n)  # true ATT = 2
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


def test_psm_recovers_att_and_corrects_bias(matching_data):
    naive = (
        matching_data[matching_data["t"] == 1]["y"].mean()
        - matching_data[matching_data["t"] == 0]["y"].mean()
    )
    r = sp.psm(matching_data, y="y", d="t", X=["x1", "x2"], method="nn")
    est = float(r.estimate)

    assert naive > 2.5  # naive contrast is upward-biased
    assert est == pytest.approx(2.0, abs=0.4)  # matching recovers the ATT
    assert abs(est - 2.0) < abs(naive - 2.0)


def test_match_agrees_with_psm(matching_data):
    psm_res = sp.psm(matching_data, y="y", d="t", X=["x1", "x2"], method="nn")
    psm = float(psm_res.estimate)
    m = float(
        sp.match(data=matching_data, y="y", treat="t", covariates=["x1", "x2"]).estimate
    )
    # Both nearest-neighbour matchers should land on the same ATT.
    assert m == pytest.approx(psm, abs=0.5)
