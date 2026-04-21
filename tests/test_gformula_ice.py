"""Sprint-6 tests: parametric g-formula via ICE."""
import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_longitudinal_dgp(n=1500, seed=0):
    """DGP:
        L0 ~ N(0, 1)
        A0 ~ Bern(sigmoid(0.5 L0))
        L1 = L0 + 0.5 A0 + N(0, 0.5)
        A1 ~ Bern(sigmoid(0.3 L1 + 0.5 A0))
        Y  = 2 A0 + 3 A1 + L0 + N(0, 0.5)
    Under always-treat (A0=1, A1=1), E[Y] = 2 + 3 + E[L0] = 5.
    Under never-treat, E[Y] = E[L0] = 0.
    """
    rng = np.random.default_rng(seed)
    L0 = rng.normal(0, 1, n)
    p_a0 = 1 / (1 + np.exp(-0.5 * L0))
    A0 = rng.binomial(1, p_a0, n)
    L1 = L0 + 0.5 * A0 + rng.normal(0, 0.5, n)
    p_a1 = 1 / (1 + np.exp(-(0.3 * L1 + 0.5 * A0)))
    A1 = rng.binomial(1, p_a1, n)
    Y = 2 * A0 + 3 * A1 + L0 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({
        "id": np.arange(n),
        "L0": L0, "A0": A0, "L1": L1, "A1": A1, "Y": Y,
    })
    return df


def test_ice_recovers_always_treat_value():
    df = _make_longitudinal_dgp(n=3000, seed=0)
    res = sp.gformula.ice(
        data=df,
        id_col="id", time_col=None,
        treatment_cols=["A0", "A1"],
        confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y",
        treatment_strategy=[1, 1],
    )
    assert isinstance(res, sp.ICEResult)
    # True value ≈ 5.0
    assert abs(res.value - 5.0) < 0.5
    assert res.ci[0] < res.value < res.ci[1]
    assert "strategy=[1, 1]" in res.summary()


def test_ice_recovers_never_treat_value():
    df = _make_longitudinal_dgp(n=3000, seed=1)
    res = sp.gformula.ice(
        data=df,
        id_col="id", time_col=None,
        treatment_cols=["A0", "A1"],
        confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y",
        treatment_strategy=[0, 0],
    )
    assert abs(res.value - 0.0) < 0.5


def test_ice_contrast_is_5_between_always_and_never():
    df = _make_longitudinal_dgp(n=5000, seed=2)
    always = sp.gformula.ice(
        data=df, id_col="id", time_col=None,
        treatment_cols=["A0", "A1"],
        confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y", treatment_strategy=[1, 1],
    )
    never = sp.gformula.ice(
        data=df, id_col="id", time_col=None,
        treatment_cols=["A0", "A1"],
        confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y", treatment_strategy=[0, 0],
    )
    contrast = always.value - never.value
    assert abs(contrast - 5.0) < 0.8


def test_ice_bootstrap_gives_reasonable_se():
    df = _make_longitudinal_dgp(n=1000, seed=3)
    res = sp.gformula.ice(
        data=df, id_col="id", time_col=None,
        treatment_cols=["A0"],
        confounder_cols=[["L0"]],
        outcome_col="Y",
        treatment_strategy=[1],
        bootstrap=50,
        seed=0,
    )
    assert res.se > 0
    assert np.isfinite(res.ci[0]) and np.isfinite(res.ci[1])


def test_ice_strategy_validation():
    df = _make_longitudinal_dgp(n=100, seed=4)
    with pytest.raises(ValueError, match="strategy length"):
        sp.gformula.ice(
            data=df, id_col="id", time_col=None,
            treatment_cols=["A0", "A1"],
            confounder_cols=[["L0"], ["L1"]],
            outcome_col="Y",
            treatment_strategy=[1],  # wrong length
        )


def test_ice_accepts_callable_strategy():
    df = _make_longitudinal_dgp(n=500, seed=5)
    res = sp.gformula.ice(
        data=df, id_col="id", time_col=None,
        treatment_cols=["A0", "A1"],
        confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y",
        treatment_strategy=lambda t: 1 if t == 0 else 0,  # treat, then stop
    )
    assert isinstance(res, sp.ICEResult)
