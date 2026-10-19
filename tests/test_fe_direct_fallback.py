"""Unconverged fixed-effect projections are finished by a direct solve.

On a chain of firms linked by one mover each, plus groups of one,
alternating projections crawl: after the native default of 1000 sweeps the
slope was 0.51932 against the dummy-variable regression's 0.51983 (R
``lm(y ~ x + factor(id) + factor(firm))`` gives 0.519831177253658, SE
0.0365711824320098, on this design with ``seed=1``; R ``fixest`` 0.14
returns 0.51981 because it stops short too), and ``sp.feols`` /
``sp.hdfe_ols`` reported it without a word. The unconverged columns are now
residualised by a sparse direct solve of the normal equations
(``fast/_fe_direct.py``), so the fit equals the dummy-variable regression.

Truth here is the dense least-squares fit with every level as a dummy.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import NumericalInstability
from statspai.fast._fe_direct import fe_design, residualise, residualise_codes


def _chain(L: int = 800, S: int = 500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    i = np.repeat(np.arange(L), 3)
    f = np.repeat(np.arange(L), 3) + np.tile([0, 0, 1], L)
    i = np.concatenate([i, L + np.arange(S)])
    f = np.concatenate([f, rng.integers(0, L, S)])
    n = len(i)
    x = rng.normal(size=n) + 0.002 * f
    y = 0.5 * x + rng.normal(size=n) + 0.01 * f
    w = np.random.default_rng(0).uniform(0.5, 2.0, n)
    return pd.DataFrame(dict(y=y, x=x, id=i, firm=f, w=w))


def _dummy_ols(df: pd.DataFrame, weights: bool = False):
    """Slope and iid SE of x from the dense dummy-variable (W)LS fit."""
    D = np.column_stack(
        [
            df["x"].to_numpy(),
            pd.get_dummies(df["id"]).to_numpy(float),
            pd.get_dummies(df["firm"]).to_numpy(float)[:, 1:],
        ]
    )
    y = df["y"].to_numpy()
    sw = np.sqrt(df["w"].to_numpy()) if weights else np.ones(len(df))
    Dw, yw = D * sw[:, None], y * sw
    beta, _, rank, _ = np.linalg.lstsq(Dw, yw, rcond=None)
    u = yw - Dw @ beta
    sigma2 = float(u @ u) / (len(df) - rank)
    # x's variance from FWL: residualise x on the dummies
    Z = Dw[:, 1:]
    xr = Dw[:, 0] - Z @ np.linalg.lstsq(Z, Dw[:, 0], rcond=None)[0]
    return beta[0], np.sqrt(sigma2 / float(xr @ xr))


@pytest.fixture(scope="module")
def chain():
    return _chain()


@pytest.fixture(scope="module")
def truth(chain):
    return {w: _dummy_ols(chain, weights=w) for w in (False, True)}


@pytest.mark.parametrize("weighted", [False, True])
def test_native_feols_equals_dummy_regression(chain, truth, weighted):
    b, se = truth[weighted]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.feols("y ~ x | id + firm", chain, weights="w" if weighted else None)
    assert "sparse direct" in r.model_info["fe_solver"]
    np.testing.assert_allclose(r.params["x"], b, rtol=1e-10)
    np.testing.assert_allclose(r.std_errors["x"], se, rtol=1e-9)
    if not weighted:  # R lm on the same bytes (seed=1)
        np.testing.assert_allclose(r.params["x"], 0.519831177253658, rtol=1e-12)
        np.testing.assert_allclose(r.std_errors["x"], 0.0365711824320098, rtol=1e-12)


@pytest.mark.parametrize("weighted", [False, True])
def test_hdfe_ols_equals_dummy_regression(chain, truth, weighted):
    b, se = truth[weighted]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.hdfe_ols(
            "y ~ x | id + firm",
            chain,
            drop_singletons=False,
            weights="w" if weighted else None,
        )
    assert r.converged
    assert r.absorber.fe_solver == "sparse direct"
    np.testing.assert_allclose(r.params["x"], b, rtol=1e-10)
    np.testing.assert_allclose(r.std_errors["x"], se, rtol=1e-9)


def test_well_connected_design_keeps_the_projections():
    rng = np.random.default_rng(3)
    n = 2000
    df = pd.DataFrame(
        dict(
            x=rng.normal(size=n),
            a=rng.integers(0, 40, n),
            b=rng.integers(0, 30, n),
        )
    )
    df["y"] = df["x"] + rng.normal(size=n)
    r = sp.feols("y ~ x | a + b", df)
    assert r.model_info["fe_solver"] == "alternating projections"


def test_pyfixest_demeaning_failure_is_typed(chain):
    pytest.importorskip("pyfixest")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NumericalInstability) as info:
            sp.feols("y ~ x | id + firm", chain, backend="pyfixest")
    assert "native engine" in info.value.recovery_hint


@pytest.mark.parametrize("weighted", [False, True])
def test_residualise_is_the_exact_projection(weighted):
    """Three fixed effects, two of them nested in nothing: the residual is
    orthogonal to every dummy and equals the dense least-squares one."""
    rng = np.random.default_rng(7)
    n = 600
    codes = [rng.integers(0, g, n) for g in (90, 25, 6)]
    codes = [pd.factorize(c)[0].astype(np.int64) for c in codes]
    X = rng.normal(size=(n, 2))
    w = rng.uniform(0.2, 3.0, n) if weighted else None
    D = fe_design(codes, n).toarray()
    sw = np.sqrt(w) if weighted else np.ones(n)
    ref = X - D @ np.linalg.lstsq(D * sw[:, None], X * sw[:, None], rcond=None)[0]
    out, ok = residualise_codes(X.copy(), codes, w)
    assert ok
    np.testing.assert_allclose(out, ref, atol=1e-11)
    # same answer without eliminating a dimension first
    out2, ok2 = residualise(X.copy(), fe_design(codes, n), w)
    assert ok2
    np.testing.assert_allclose(out2, ref, atol=1e-11)
