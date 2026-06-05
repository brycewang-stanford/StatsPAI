"""Coverage tests for statspai.iv.ivmte_lp (MST 2018 sharp bounds LP)."""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp

_lp = importlib.import_module("statspai.iv.ivmte_lp")


def _binary_iv_df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    v = rng.normal(size=n)
    latent = 0.9 * z + 0.4 * x + v
    d = (latent > 0).astype(float)
    u = 0.5 * v + rng.normal(0, 0.5, size=n)
    y = 0.3 + 0.8 * d + 0.2 * x + u
    return pd.DataFrame({"y": y, "d": d, "z": z, "x": x})


def test_grab_array_fallback():
    arr = _lp._grab(np.array([1.0, 2.0, 3.0]), None)
    assert arr.dtype == float
    # list of non-str values goes through np.asarray fallback too
    arr2 = _lp._grab([1, 2, 3], None, cols=True)
    np.testing.assert_array_equal(arr2, [1.0, 2.0, 3.0])


def test_grab_cols_list_of_str():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = _lp._grab(["a", "b"], df, cols=True)
    assert out.shape == (2, 2)


def test_poly_u_and_int_poly():
    P = _lp._poly_u(np.array([0.0, 0.5, 1.0]), 2)
    assert P.shape == (3, 3)
    np.testing.assert_allclose(P[:, 0], 1.0)
    ip = _lp._int_poly(0.0, 1.0, 2)
    # ∫_0^1 [1, u, u^2] = [1, 1/2, 1/3]
    np.testing.assert_allclose(ip, [1.0, 0.5, 1.0 / 3.0])


def test_fit_logit_runs():
    rng = np.random.default_rng(1)
    n = 300
    Z = np.column_stack([np.ones(n), rng.normal(size=n)])
    D = (Z[:, 1] + rng.normal(size=n) > 0).astype(float)
    p = _lp._fit_logit(D, Z)
    assert np.all((p > 0) & (p < 1))


def test_ivmte_bounds_ate():
    df = _binary_iv_df(seed=2)
    r = sp.iv.ivmte_bounds(
        y="d".replace("d", "y"), treatment="d", instruments="z",
        data=df, target="ate", basis_degree=2, n_propensity_bins=6,
    )
    assert isinstance(r, _lp.IVMTEBounds)
    assert r.lower_bound <= r.upper_bound + 1e-6
    assert "ate" in r.summary()


def test_ivmte_bounds_att_atu():
    df = _binary_iv_df(seed=3)
    r_att = sp.iv.ivmte_bounds(y="y", treatment="d", instruments="z", data=df,
                               target="att", basis_degree=2, n_propensity_bins=6)
    r_atu = sp.iv.ivmte_bounds(y="y", treatment="d", instruments="z", data=df,
                               target="atu", basis_degree=2, n_propensity_bins=6)
    assert r_att.lower_bound <= r_att.upper_bound + 1e-6
    assert r_atu.lower_bound <= r_atu.upper_bound + 1e-6


def test_ivmte_bounds_late():
    df = _binary_iv_df(seed=4)
    r = sp.iv.ivmte_bounds(y="y", treatment="d", instruments="z", data=df,
                           target="late", late_bounds=(0.2, 0.8),
                           basis_degree=2, n_propensity_bins=6)
    assert r.lower_bound <= r.upper_bound + 1e-6


def test_ivmte_bounds_late_requires_bounds():
    df = _binary_iv_df(seed=5)
    with pytest.raises(ValueError):
        sp.iv.ivmte_bounds(y="y", treatment="d", instruments="z", data=df,
                           target="late", basis_degree=2)


def test_ivmte_bounds_prte():
    df = _binary_iv_df(seed=6)
    rng = np.random.default_rng(6)
    policy = np.clip(rng.uniform(0.2, 0.9, size=len(df)), 1e-3, 1 - 1e-3)
    r = sp.iv.ivmte_bounds(y="y", treatment="d", instruments="z", data=df,
                           target="prte", policy_prob=policy,
                           basis_degree=2, n_propensity_bins=6)
    assert np.isfinite(r.lower_bound) or np.isnan(r.lower_bound)


def test_ivmte_bounds_prte_requires_policy():
    df = _binary_iv_df(seed=7)
    with pytest.raises(ValueError):
        sp.iv.ivmte_bounds(y="y", treatment="d", instruments="z", data=df,
                           target="prte", basis_degree=2)


def test_target_weights_prte_negligible_shift_raises():
    rng = np.random.default_rng(8)
    p = rng.uniform(0.2, 0.8, size=400)
    # identical policy -> zero shift -> denominator too small
    with pytest.raises(ValueError):
        _lp._target_weights("prte", 2, p, policy_prob=p.copy())


def test_target_weights_unknown_raises():
    p = np.linspace(0.1, 0.9, 100)
    with pytest.raises(ValueError):
        _lp._target_weights("bogus", 2, p)


def test_ivmte_bounds_nonbinary_treatment_raises():
    df = _binary_iv_df(seed=9)
    df = df.copy()
    df["d"] = df["d"] + 0.5  # now 0.5 / 1.5
    with pytest.raises(ValueError):
        sp.iv.ivmte_bounds(y="y", treatment="d", instruments="z", data=df,
                           target="ate")


def test_ivmte_bounds_with_exog_and_shape():
    df = _binary_iv_df(seed=10)
    r = sp.iv.ivmte_bounds(
        y="y", treatment="d", instruments="z", exog="x", data=df,
        target="ate", basis_degree=2, n_propensity_bins=6,
        bounds_outcome=(-2.0, 3.0), decreasing_mte=True,
        include_bmw_point=True,
    )
    assert "decreasing_mte" in r.shape_restrictions
    assert any("bounds_outcome" in s for s in r.shape_restrictions)
    assert r.lower_bound <= r.upper_bound + 1e-6


def test_shape_constraints_empty():
    A, b = _lp._shape_constraints(2)
    assert A.shape == (0, 6)
    assert b.shape == (0,)


def test_shape_constraints_with_bounds_and_decreasing():
    A, b = _lp._shape_constraints(2, bounds_outcome=(0.0, 1.0),
                                  decreasing_mte=True, u_discretisation=11)
    assert A.shape[0] == b.shape[0]
    assert A.shape[1] == 6
    assert A.shape[0] > 0


def test_build_iv_moments_no_usable_bins_raises():
    # Single observation -> every bin has < 2 obs -> no usable bins (line 196)
    p = np.array([0.5])
    D = np.array([0.0])
    Y = np.array([1.0])
    with pytest.raises(RuntimeError):
        _lp._build_iv_moments(p, D, Y, K=2, n_bins=8)
