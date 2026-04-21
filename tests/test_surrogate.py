"""
Tests for sp.surrogate — long-term effects via surrogate indices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# -------------------------------------------------------------------------
# Data generators
# -------------------------------------------------------------------------


def _make_surrogate_data(
    *, n_exp: int = 500, n_obs: int = 1500, true_ate: float = 1.0,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DGP where Y is driven by S, S is driven by T — classical surrogacy."""
    rng = np.random.default_rng(seed)
    # Experimental sample
    T = rng.binomial(1, 0.5, size=n_exp).astype(float)
    # Short-term surrogate: treatment shifts S by true_ate (on the S scale)
    S_e = 0.5 + true_ate * T + rng.normal(0, 1, size=n_exp)
    exp = pd.DataFrame({"T": T, "S": S_e})

    # Observational sample: draw S from the mixture, then Y = alpha + beta*S.
    S_o = rng.normal(0.5 + 0.5 * true_ate, np.sqrt(1 + 0.25 * true_ate**2), size=n_obs)
    Y_o = 2.0 + 1.5 * S_o + rng.normal(0, 0.5, size=n_obs)
    obs = pd.DataFrame({"S": S_o, "Y": Y_o})
    return exp, obs


# -------------------------------------------------------------------------
# surrogate_index
# -------------------------------------------------------------------------


def test_surrogate_index_recovers_true_ate():
    exp, obs = _make_surrogate_data(true_ate=1.0, seed=7)
    res = sp.surrogate_index(
        exp, obs, treatment="T", surrogates=["S"], long_term_outcome="Y",
    )
    # Long-term ATE = beta * true_ate_on_S = 1.5 * 1.0 = 1.5.
    # Require coverage within 3 SE (asymptotic).
    assert abs(res.estimate - 1.5) < 3 * res.se, (res.estimate, res.se)
    assert res.method == "surrogate_index"
    assert res.estimand == "ATE"
    assert res.se > 0
    lo, hi = res.ci
    assert lo < 1.5 < hi


def test_surrogate_index_zero_effect():
    exp, obs = _make_surrogate_data(true_ate=0.0, seed=11)
    res = sp.surrogate_index(
        exp, obs, treatment="T", surrogates=["S"], long_term_outcome="Y",
    )
    # Should be ~0, well within 2 SE
    assert abs(res.estimate) < 2 * res.se + 0.2


def test_surrogate_index_bootstrap_matches_delta():
    exp, obs = _make_surrogate_data(true_ate=1.0, seed=13)
    res_delta = sp.surrogate_index(
        exp, obs, treatment="T", surrogates=["S"], long_term_outcome="Y",
    )
    res_boot = sp.surrogate_index(
        exp, obs, treatment="T", surrogates=["S"], long_term_outcome="Y",
        n_boot=200, random_state=13,
    )
    # Point estimates should be *identical* (same data, same f-hat).
    assert abs(res_delta.estimate - res_boot.estimate) < 1e-10
    # SEs should be in the same order of magnitude.
    assert 0.3 * res_delta.se < res_boot.se < 3.0 * res_delta.se


def test_surrogate_index_missing_columns_errors():
    exp, obs = _make_surrogate_data()
    with pytest.raises(ValueError, match="experimental missing"):
        sp.surrogate_index(
            exp, obs, treatment="Tbogus",
            surrogates=["S"], long_term_outcome="Y",
        )
    with pytest.raises(ValueError, match="observational missing"):
        sp.surrogate_index(
            exp, obs, treatment="T",
            surrogates=["S"], long_term_outcome="Ybogus",
        )


def test_surrogate_index_non_binary_treatment_errors():
    exp, obs = _make_surrogate_data()
    exp = exp.copy()
    exp["T"] = exp["T"] * 3.0  # 0 and 3
    with pytest.raises(ValueError, match="must be binary"):
        sp.surrogate_index(
            exp, obs, treatment="T",
            surrogates=["S"], long_term_outcome="Y",
        )


# -------------------------------------------------------------------------
# long_term_from_short
# -------------------------------------------------------------------------


def test_long_term_from_short_single_wave_matches_classical():
    """Single wave should reduce to the classical surrogate index."""
    exp, obs = _make_surrogate_data(true_ate=1.0, seed=19)
    res_multi = sp.long_term_from_short(
        exp, obs, treatment="T",
        surrogates_waves=[["S"]],
        long_term_outcome="Y",
        n_boot=100, random_state=19,
    )
    res_classic = sp.surrogate_index(
        exp, obs, treatment="T", surrogates=["S"], long_term_outcome="Y",
    )
    # Same DGP → same point estimate up to numerical noise.
    assert abs(res_multi.estimate - res_classic.estimate) < 1e-8


def test_long_term_from_short_two_waves():
    rng = np.random.default_rng(23)
    n_exp, n_obs = 400, 1200
    T = rng.binomial(1, 0.5, size=n_exp).astype(float)
    S1_e = 0.3 + 0.8 * T + rng.normal(0, 0.5, size=n_exp)
    S2_e = 0.2 + 0.6 * S1_e + rng.normal(0, 0.5, size=n_exp)
    exp = pd.DataFrame({"T": T, "S1": S1_e, "S2": S2_e})

    S1_o = rng.normal(0.7, 1.0, size=n_obs)
    S2_o = 0.2 + 0.6 * S1_o + rng.normal(0, 0.5, size=n_obs)
    Y_o = 1.0 + 1.5 * S2_o + rng.normal(0, 0.5, size=n_obs)
    obs = pd.DataFrame({"S1": S1_o, "S2": S2_o, "Y": Y_o})

    res = sp.long_term_from_short(
        exp, obs, treatment="T",
        surrogates_waves=[["S1"], ["S2"]],
        long_term_outcome="Y",
        n_boot=100, random_state=23,
    )
    # Expected: ATE(S1 on T) * 0.6 * 1.5 ≈ 0.8 * 0.9 = 0.72
    assert 0.45 < res.estimate < 1.0, res.estimate
    assert res.model_info["n_waves"] == 2


# -------------------------------------------------------------------------
# proximal_surrogate_index
# -------------------------------------------------------------------------


def test_proximal_surrogate_index_runs_with_valid_proxy():
    rng = np.random.default_rng(29)
    n_exp, n_obs = 400, 1200
    # Experimental: T → S
    T = rng.binomial(1, 0.5, size=n_exp).astype(float)
    S_e = 0.5 + 1.0 * T + rng.normal(0, 1, size=n_exp)
    exp = pd.DataFrame({"T": T, "S": S_e})

    # Observational: U confounds S and Y; W is a proxy for U.
    U = rng.normal(0, 1, size=n_obs)
    S_o = 0.5 * U + rng.normal(0, 1, size=n_obs)
    W_o = 0.8 * U + rng.normal(0, 0.5, size=n_obs)
    Y_o = 2.0 + 1.5 * S_o + 0.7 * U + rng.normal(0, 0.5, size=n_obs)
    obs = pd.DataFrame({"S": S_o, "W": W_o, "Y": Y_o})

    res = sp.proximal_surrogate_index(
        exp, obs, treatment="T", surrogates=["S"], proxies=["W"],
        long_term_outcome="Y", n_boot=100, random_state=29,
    )
    assert np.isfinite(res.estimate)
    assert res.se > 0
    assert "proxies" in res.model_info


def test_proximal_surrogate_no_proxy_errors():
    exp, obs = _make_surrogate_data()
    with pytest.raises(ValueError, match="at least one proxy"):
        sp.proximal_surrogate_index(
            exp, obs, treatment="T", surrogates=["S"], proxies=[],
            long_term_outcome="Y",
        )


# -------------------------------------------------------------------------
# Registry integration
# -------------------------------------------------------------------------


def test_surrogate_in_registry():
    fns = set(sp.list_functions())
    assert "surrogate_index" in fns
    assert "long_term_from_short" in fns
    assert "proximal_surrogate_index" in fns
    spec = sp.describe_function("surrogate_index")
    assert spec["category"] == "surrogate"
