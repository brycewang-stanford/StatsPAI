"""Tests for the unified ``sp.iv`` dispatcher.

These tests exercise the callable-module pattern installed in
``statspai.iv.__init__`` and the ``method=`` table — they do *not*
re-test numerical correctness of the underlying estimators, which is
covered by the per-method test files (``test_iv.py`` etc.) and the
reference-parity suite.

The point: prevent regression of the v1.10 ``sp.iv()`` unification —
prior to v1.10 ``sp.iv("y ~ (d ~ z)", data=df)`` silently raised
``TypeError: 'module' object is not callable`` despite being advertised
in the registry, agent summaries, and `question/question.py`.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def iv_data():
    """A small over-identified IV DGP with one endogenous regressor."""
    rng = np.random.default_rng(0)
    n = 400
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    d = 0.6 * z1 + 0.5 * z2 + 0.3 * x1 + 0.5 * rng.standard_normal(n)
    y = 1.0 + 2.0 * d + 0.5 * x1 + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2, "x1": x1})


# ─── Callable-module surface ────────────────────────────────────────────


def test_sp_iv_is_callable():
    """The headline regression: ``sp.iv`` must be callable, not a bare module."""
    assert callable(sp.iv), (
        "sp.iv must be callable.  If this fails, the callable-module "
        "shim in statspai/iv/__init__.py was clobbered."
    )


def test_sp_iv_keeps_submodule_access():
    """Callable trick must not break ``sp.iv.kernel_iv``-style lookup."""
    assert callable(sp.iv.fit)
    assert callable(sp.iv.kernel_iv)
    assert callable(sp.iv.bayesian_iv)
    assert callable(sp.iv.npiv)
    assert callable(sp.iv.ivdml)
    assert callable(sp.iv.mte)
    # diagnostics are re-exports, also reachable
    assert callable(sp.iv.anderson_rubin_test)
    assert callable(sp.iv.kleibergen_paap_rk)


def test_function_first_api_survives_iv_bootstrap():
    """``sp.bartik`` / ``sp.deepiv`` must stay callable, not module objects."""
    assert callable(sp.bartik)
    assert callable(sp.deepiv)
    assert callable(sp.iv.bartik)
    assert callable(sp.iv.deepiv)
    assert sp.bartik.__module__.startswith("statspai.bartik.")
    assert sp.deepiv.__module__.startswith("statspai.deepiv.")


def test_sp_iv_default_is_2sls(iv_data):
    """Calling without ``method=`` should be 2SLS."""
    r = sp.iv("y ~ (d ~ z1 + z2) + x1", data=iv_data)
    # coefficient should be near the true value of 2.0
    assert abs(r.params["d"] - 2.0) < 0.2


def test_fit_alias_matches_dispatcher(iv_data):
    """``sp.iv.fit(...)`` must equal ``sp.iv(...)`` for the same args."""
    r1 = sp.iv("y ~ (d ~ z1 + z2) + x1", data=iv_data, method="liml")
    r2 = sp.iv.fit("y ~ (d ~ z1 + z2) + x1", data=iv_data, method="liml")
    assert r1.params["d"] == pytest.approx(r2.params["d"])


# ─── Method coverage ────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["2sls", "liml", "fuller", "gmm", "jive"])
def test_kclass_methods(iv_data, method):
    r = sp.iv("y ~ (d ~ z1 + z2) + x1", data=iv_data, method=method)
    assert "d" in r.params
    assert abs(r.params["d"] - 2.0) < 0.5


@pytest.mark.parametrize("method", ["jive1", "ujive", "ijive", "rjive"])
def test_modern_jive(iv_data, method):
    r = sp.iv(
        method=method, y="y", endog="d",
        instruments=["z1", "z2"], exog=["x1"], data=iv_data,
    )
    # JIVEResult exposes ``coef`` (not params) — just check it runs.
    assert r is not None


def test_kernel_iv(iv_data):
    r = sp.iv(method="kernel", y="y", endog="d",
              instruments=["z1"], data=iv_data, n_boot=20)
    assert r is not None


def test_npiv(iv_data):
    r = sp.iv(method="npiv", y="y", endog="d",
              instruments=["z1", "z2"], exog=["x1"], data=iv_data)
    assert r is not None


def test_ivdml(iv_data):
    r = sp.iv(method="ivdml", y="y", endog="d",
              instruments=["z1", "z2"], exog=["x1"], data=iv_data, n_folds=2)
    assert r is not None


def test_continuous_late(iv_data):
    r = sp.iv(method="continuous_late", y="y", endog="d",
              instruments=["z1"], data=iv_data, n_boot=20)
    assert r is not None


def test_bayesian_iv(iv_data):
    r = sp.iv(method="bayes", y="y", endog="d",
              instruments=["z1", "z2"], exog=["x1"], data=iv_data,
              n_draws=200, n_warmup=100)
    assert r is not None


def test_plausibly_exog(iv_data):
    r = sp.iv(method="plausibly_exog", y="y", endog="d",
              instruments=["z1", "z2"], exog=["x1"], data=iv_data,
              gamma_mean=0.0, gamma_var=0.01)
    assert r is not None


# ─── Alias / normalization ──────────────────────────────────────────────


@pytest.mark.parametrize("alias", ["2sls", "tsls", "iv", "IV", "TSLS"])
def test_2sls_aliases_equivalent(iv_data, alias):
    r = sp.iv("y ~ (d ~ z1 + z2) + x1", data=iv_data, method=alias)
    assert "d" in r.params


def test_endog_alias_normalizes_to_treat(iv_data):
    """For kernel_iv (which expects ``treat``), passing ``endog`` should work."""
    r1 = sp.iv(method="kernel", y="y", endog="d",
               instruments=["z1"], data=iv_data, n_boot=10, seed=0)
    r2 = sp.iv(method="kernel", y="y", treat="d",
               instruments=["z1"], data=iv_data, n_boot=10, seed=0)
    # Same DGP and seed → same result.
    assert type(r1).__name__ == type(r2).__name__


def test_singleton_instrument_list_unwrapped(iv_data):
    """``instruments=['z1']`` should work for singular-instrument methods."""
    r = sp.iv(method="kernel", y="y", endog="d",
              instruments=["z1"], data=iv_data, n_boot=10)
    assert r is not None


# ─── Error paths ────────────────────────────────────────────────────────


def test_unknown_method_raises_value_error(iv_data):
    with pytest.raises(ValueError, match="Unknown method"):
        sp.iv("y ~ (d ~ z1) + x1", data=iv_data, method="not_a_method")


def test_non_string_method_raises_type_error(iv_data):
    with pytest.raises(TypeError, match="method must be a string"):
        sp.iv("y ~ (d ~ z1) + x1", data=iv_data, method=42)


def test_ambiguous_alias_raises(iv_data):
    """Passing both ``endog`` and ``treat`` to a kernel-style method = error."""
    with pytest.raises(TypeError, match="Got both"):
        sp.iv(method="kernel", y="y", endog="d", treat="d",
              instruments=["z1"], data=iv_data)


def test_kernel_with_multiple_instruments_raises(iv_data):
    """Singular-instrument methods reject multi-instrument lists clearly."""
    with pytest.raises(ValueError, match="single 'instrument' column"):
        sp.iv(method="kernel", y="y", endog="d",
              instruments=["z1", "z2"], data=iv_data)


def test_formula_methods_require_formula_and_data(iv_data):
    with pytest.raises(ValueError, match="requires `formula` and `data`"):
        sp.iv(method="liml")


# ─── Backward-compat shims ──────────────────────────────────────────────


def test_ivreg_top_level_still_works(iv_data):
    """``sp.ivreg`` is the legacy alias and must keep working."""
    r = sp.ivreg("y ~ (d ~ z1 + z2) + x1", data=iv_data)
    assert "d" in r.params


def test_regression_iv_iv_function_still_importable(iv_data):
    """Direct import path should keep working — used by tests and provenance."""
    from statspai.regression.iv import iv as _iv_fn
    r = _iv_fn("y ~ (d ~ z1 + z2) + x1", data=iv_data)
    assert "d" in r.params
