"""Coverage campaign — IV reachable error/parameter tail branches.

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). Closes the *reachable* tail branches
left after the eight main iv files: the weak-id ``summary()`` renderers, NPIV
basis variants (polynomial / bspline / auto-large-K / unknown→error), the MTE
probit propensity model and its too-small-sample validation error, and the
``sp.iv`` dispatcher's lasso-with-formula and quantile-IV routes plus the
missing-arguments error. (Genuinely-unreachable defensive lines — e.g.
``except np.linalg.LinAlgError`` pinv fallbacks — are marked ``# pragma: no
cover`` in source per the maintainer's tail policy.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(61)
    n = 500
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    d = 0.8 * z1 + 0.6 * z2 + 0.3 * x + 0.6 * u + 0.5 * rng.standard_normal(n)
    y = 1.0 + 2.0 * d + 0.4 * x + u + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2, "x": x})


@pytest.fixture(scope="module")
def binary_df():
    rng = np.random.default_rng(62)
    n = 800
    z = rng.uniform(-2, 2, n)
    x = rng.standard_normal(n)
    v = rng.standard_normal(n)
    d = ((0.9 * z + 0.3 * x - 0.4 * v) > 0).astype(float)
    y = 1.0 + 1.3 * d + 0.4 * x + 0.5 * v + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "d": d, "z": z, "x": x})


# ─── weak-id summary renderers ───────────────────────────────────────────


def test_weak_id_summaries(df):
    kp = sp.iv.kleibergen_paap_rk(
        endog=df[["d"]], instruments=df[["z1", "z2"]], exog=df[["x"]]
    )
    assert "Kleibergen" in kp.summary()
    sw = sp.iv.sanderson_windmeijer(
        endog=df[["d"]], instruments=df[["z1", "z2"]], exog=df[["x"]]
    )
    assert "Sanderson" in sw.summary()
    clr = sp.iv.conditional_lr_test(
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        data=df,
        n_simulations=2000,
        random_state=0,
    )
    assert "Moreira" in clr.summary() or "LR" in clr.summary()


# ─── NPIV basis variants ─────────────────────────────────────────────────


@pytest.mark.parametrize("basis", ["polynomial", "bspline", "auto"])
def test_npiv_basis_variants(df, basis):
    res = sp.iv.npiv(
        y="y",
        endog="d",
        instruments=df[["z1", "z2"]],
        data=df,
        basis=basis,
        k_d=6,
        k_z=6,
    )
    assert np.all(np.isfinite(np.asarray(res.h_values, dtype=float)))
    assert isinstance(res.summary(), str)


def test_npiv_unknown_basis_raises(df):
    with pytest.raises(ValueError, match="[Uu]nknown basis"):
        sp.iv.npiv(
            y="y", endog="d", instruments=df[["z1", "z2"]], data=df, basis="nope"
        )


# ─── MTE probit propensity + validation error ────────────────────────────


def test_mte_probit_propensity(binary_df):
    res = sp.iv.mte(
        y="y",
        treatment="d",
        instruments=["z"],
        exog=["x"],
        data=binary_df,
        propensity_model="probit",
    )
    assert res is not None


def test_mte_too_few_obs_per_arm_raises():
    rng = np.random.default_rng(7)
    n = 40  # tiny sample, high polynomial degree → not enough per arm
    z = rng.uniform(-2, 2, n)
    d = (z > 0).astype(float)
    y = 1.0 + d + rng.standard_normal(n)
    df = pd.DataFrame({"y": y, "d": d, "z": z})
    with pytest.raises(ValueError, match="[Nn]ot enough|arm"):
        sp.iv.mte(y="y", treatment="d", instruments=["z"], data=df, poly_degree=5)


# ─── dispatcher: lasso-with-formula, quantile IV, missing args ───────────


def test_dispatch_lasso_formula_matches_native(df):
    # Regression test for the dispatcher's lasso formula route (was a bug: the
    # dispatcher forwarded ``formula=`` into ``lasso_iv``, which takes native
    # ``x_endog``/``z`` lists, raising ``TypeError``). The fix parses the
    # Patsy-style formula into those names, so the formula path must now
    # return the *same* estimates as the native x_endog/z calling convention.
    res_formula = sp.iv(method="lasso", formula="y ~ (d ~ z1 + z2) + x", data=df)
    res_native = sp.iv(
        method="lasso", data=df, y="y", x_endog=["d"], z=["z1", "z2"], x_exog=["x"]
    )
    assert res_formula is not None and res_native is not None

    def _coef(r):
        c = getattr(r, "coefficients", None)
        if c is None:
            c = getattr(r, "params", None)
        return np.asarray(c, dtype=float)

    np.testing.assert_allclose(_coef(res_formula), _coef(res_native), rtol=0, atol=0)


def test_dispatch_lasso_alias_endog_instruments(df):
    # The canonical dispatcher aliases (endog/instruments/exog) must also route
    # into lasso_iv's native x_endog/z/x_exog names.
    res = sp.iv(
        method="lasso",
        data=df,
        y="y",
        endog=["d"],
        instruments=["z1", "z2"],
        exog=["x"],
    )
    assert res is not None


def test_dispatch_ivqreg_route(df):
    # quantile IV at the median; routes through regression.iv_quantile
    res = sp.iv(
        method="ivqreg",
        data=df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        tau=0.5,
    )
    assert res is not None


def test_dispatch_missing_arguments_raises(df):
    # a method that needs explicit y/endog/instruments, given none of them
    with pytest.raises((ValueError, TypeError)):
        sp.iv(method="jive_mw", data=df)
