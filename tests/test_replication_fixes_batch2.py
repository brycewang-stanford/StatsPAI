"""Regressions from the top-5 replication recheck (batch 2).

* ``sp.oprobit`` reported ``converged=False`` for correct estimates: the flag
  came from the BFGS start, not from the Newton-polished optimum (Princelings,
  QJE 2019). It now uses Stata's ``g' (-H)^{-1} g < 1e-5`` at the optimum.
* ``sp.rlasso_iv`` returned a number (SE ~1e14) when no instrument was
  selected (AI-tocracy, QJE 2023); the coefficient is not identified -> NaN.
* ``sp.regress(...).pvalues`` was a bare ndarray while ``params`` /
  ``std_errors`` are labelled; results now also expose ``.nobs`` and ``.r2``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def test_oprobit_converged_flag_with_unscaled_regressors():
    rng = np.random.default_rng(0)
    n = 600
    x1 = rng.normal(size=n) * 1000.0  # badly scaled: BFGS stops early
    x2 = rng.normal(size=n)
    latent = 0.001 * x1 + 0.5 * x2 + rng.normal(size=n)
    y = np.digitize(latent, [-0.5, 0.5])
    d = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.oprobit(data=d, y="y", x=["x1", "x2"])
    assert r.model_info["converged"] is True


def test_newton_converged_rejects_a_large_gradient():
    from statspai.regression._optim_helpers import newton_converged

    H = -np.eye(2)
    assert newton_converged(np.array([[1e-4, 0.0]]), H)
    assert not newton_converged(np.array([[1.0, 0.0]]), H)


def test_rlasso_iv_no_instrument_selected_is_nan():
    rng = np.random.default_rng(5)
    n = 500
    z = rng.normal(size=(n, 20))
    d = rng.normal(size=n)
    y = d + rng.normal(size=n)
    with pytest.warns(UserWarning, match="not identified"):
        f = sp.rlasso_iv(y=y, d=d, z=z, select_Z=True, select_X=False)
    assert np.isnan(np.ravel(f.coef)[0]) and np.isnan(np.ravel(f.se)[0])


def test_rlasso_iv_with_a_strong_instrument_is_unchanged():
    rng = np.random.default_rng(5)
    n = 500
    z = rng.normal(size=(n, 20))
    d = rng.normal(size=n)
    z[:, 0] = d + 0.3 * rng.normal(size=n)
    y = d + rng.normal(size=n)
    f = sp.rlasso_iv(y=y, d=d, z=z, select_Z=True, select_X=False)
    assert np.isfinite(np.ravel(f.se)[0])
    assert np.ravel(f.coef)[0] == pytest.approx(1.0, abs=0.2)


def test_regress_pvalues_are_labelled_and_nobs_r2_exposed():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.normal(size=80)})
    d["y"] = d.x + rng.normal(size=80)
    r = sp.regress("y ~ x", data=d)
    assert isinstance(r.pvalues, pd.Series)
    assert list(r.pvalues.index) == list(r.params.index)
    assert r.pvalues["x"] < 1e-6
    assert r.nobs == 80
    assert 0 < r.r2 < 1
