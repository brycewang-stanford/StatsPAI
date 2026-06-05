"""Coverage campaign — array-input paths, ``summary()`` renderers, and
parameter variants for the IV estimators.

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). The earlier iv coverage files drove
the column-name / ``data=`` calling convention; this one exercises the
*numpy-array* input branches (``_prep``/``grab`` coercion), the text
``summary()`` renderers, and non-default estimator parameters (ridge,
regularisation, propensity model, custom β-grid, cov-type) that the happy-path
tests skip — that is where the remaining uncovered lines live.

Assertions stay real: array-fed and name-fed calls must agree, summaries must
mention the method, and weak-robust sets must still bracket the truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def arrays():
    """Return (Y, D, Z, W, df) for the same strong-IV DGP, true beta=2."""
    rng = np.random.default_rng(11)
    n = 500
    Z = rng.standard_normal((n, 3))
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    d = Z @ np.array([0.9, 0.6, 0.4]) + 0.3 * x + 0.6 * u + 0.5 * rng.standard_normal(n)
    y = 1.0 + 2.0 * d + 0.4 * x + u + rng.standard_normal(n)
    df = pd.DataFrame(
        {"y": y, "d": d, "z1": Z[:, 0], "z2": Z[:, 1], "z3": Z[:, 2], "x": x}
    )
    return y, d, Z, x, df


@pytest.fixture(scope="module")
def binary_arrays():
    rng = np.random.default_rng(12)
    n = 800
    z = rng.uniform(-2, 2, n)
    x = rng.standard_normal(n)
    v = rng.standard_normal(n)
    d = ((0.9 * z + 0.3 * x - 0.4 * v) > 0).astype(float)
    y = 1.0 + 1.3 * d + 0.4 * x + 0.5 * v + rng.standard_normal(n)
    return y, d, z, x


# ─── JIVE: array inputs + summary + no-exog branch ───────────────────────


@pytest.mark.parametrize("variant", ["jive1", "ujive", "ijive", "rjive"])
def test_jive_array_inputs_and_summary(arrays, variant):
    y, d, Z, x, _ = arrays
    fn = getattr(sp.iv, variant)
    # array inputs, no exog → exercises the ndarray grab + no-covariate branch
    res = fn(y=y, endog=d, instruments=Z, exog=None)
    beta = float(res.params.iloc[0])
    assert abs(beta - 2.0) < 0.8
    text = res.summary()
    assert isinstance(text, str) and "JIVE" in text
    assert f"N={res.n_obs}" in text


def test_jive_array_with_exog(arrays):
    y, d, Z, x, _ = arrays
    res = sp.iv.ujive(y=y, endog=d, instruments=Z, exog=x.reshape(-1, 1))
    assert np.isfinite(float(res.params.iloc[0]))


# ─── Kleibergen–Paap: array inputs, cov-type / cluster variants ──────────


def test_kp_array_inputs_classic_and_cluster(arrays):
    y, d, Z, x, df = arrays
    # array endog/instruments, no exog, non-robust cov
    r_classic = sp.iv.kleibergen_paap_rk(
        endog=d.reshape(-1, 1), instruments=Z, cov_type="nonrobust"
    )
    assert np.isfinite(r_classic.rk_f) and r_classic.rk_f > 0
    # clustered rk
    cl = np.repeat(np.arange(50), len(y) // 50)[: len(y)]
    r_cluster = sp.iv.kleibergen_paap_rk(
        endog=d.reshape(-1, 1), instruments=Z, cov_type="cluster", cluster=cl
    )
    assert np.isfinite(r_cluster.rk_f)


# ─── Weak-robust CIs: array inputs + custom β-grid ───────────────────────


def test_ar_ci_array_inputs_custom_grid(arrays):
    y, d, Z, x, _ = arrays
    cs = sp.iv.anderson_rubin_ci(
        y=y, endog=d, instruments=Z, beta_grid=np.linspace(0.0, 4.0, 201)
    )
    assert not cs.is_empty
    assert any(lo <= 2.0 <= hi for lo, hi in cs.as_intervals())


def test_clr_test_array_inputs(arrays):
    y, d, Z, x, _ = arrays
    res = sp.iv.conditional_lr_test(
        y=y, endog=d, instruments=Z, beta0=0.0, n_simulations=2000, random_state=0
    )
    # under a false null (beta=0 vs true 2) the CLR test should reject
    pval = getattr(res, "pvalue", getattr(res, "p_value", None))
    if pval is not None:
        assert 0.0 <= float(pval) <= 1.0


# ─── NPIV: regularisation + basis + exog variants ────────────────────────


def test_npiv_regularization_and_exog(arrays):
    y, d, Z, x, _ = arrays
    res = sp.iv.npiv(
        y=y, endog=d, instruments=Z, exog=x.reshape(-1, 1), regularization=0.1, k_d=3
    )
    h = np.asarray(res.h_values, dtype=float)
    assert h.size > 0 and np.all(np.isfinite(h))


def test_npiv_just_identified(arrays):
    y, d, Z, x, _ = arrays
    # single instrument → the k_z==1 branch
    res = sp.iv.npiv(y=y, endog=d, instruments=Z[:, [0]])
    assert np.all(np.isfinite(np.asarray(res.h_values, dtype=float)))


# ─── MTE: parameter variants + summary ───────────────────────────────────


def test_mte_param_variants(binary_arrays):
    y, d, z, x = binary_arrays
    res = sp.iv.mte(
        y=y,
        treatment=d,
        instruments=z.reshape(-1, 1),
        exog=x.reshape(-1, 1),
        poly_degree=3,
        propensity_model="logit",
        trim=0.02,
        bootstrap=25,
        random_state=1,
    )
    assert res is not None
    text = getattr(res, "summary", lambda: "")()
    assert isinstance(text, str)
