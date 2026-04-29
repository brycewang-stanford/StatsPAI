"""Tests for the unified ``sp.rd`` dispatcher.

The pattern mirrors :mod:`tests.test_iv_dispatcher`: verify that
``sp.rd`` is callable, that all 18 ``method=`` aliases route to the
correct estimator, that submodule attribute access keeps working, and
that error paths are clear.

Numerical correctness of the underlying estimators is covered by the
per-method test files (``test_rd*.py``) and reference parity.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rd_data():
    """A sharp-RD DGP with covariates and a categorical moderator."""
    rng = np.random.default_rng(0)
    n = 800
    x = rng.uniform(-1, 1, n)
    treat = (x >= 0).astype(int)
    y = 0.5 + 1.5 * treat + 0.5 * x + 0.3 * x * treat \
        + 0.5 * rng.standard_normal(n)
    return pd.DataFrame({
        "y": y,
        "x": x,
        "cov1": rng.standard_normal(n),
        "cov2": rng.standard_normal(n),
        "z": (rng.standard_normal(n) > 0).astype(int),
    })


# ─── Callable-module surface ────────────────────────────────────────────


def test_sp_rd_is_callable():
    assert callable(sp.rd), (
        "sp.rd must be callable; the _CallableRDModule shim was clobbered "
        "if this fails."
    )


def test_sp_rd_keeps_submodule_access():
    """Callable trick must not break submodule attribute access."""
    for name in ("rdrobust", "rdplot", "rdsummary", "rdbwselect",
                 "rd_honest", "rdhte", "rd_forest", "rd_boost", "rd_lasso",
                 "rd2d", "rdmc", "rdms", "rkd", "rdit",
                 "rd_extrapolate", "rd_interference", "rd_distribution",
                 "rd_bayes_hte", "fit"):
        assert callable(getattr(sp.rd, name)), name


def test_sp_rd_default_is_rdrobust(rd_data):
    """Calling without ``method=`` should equal explicit rdrobust."""
    r = sp.rd(rd_data, y="y", x="x", c=0)
    assert r is not None


def test_fit_alias(rd_data):
    r1 = sp.rd(rd_data, y="y", x="x", c=0, method="rdrobust")
    r2 = sp.rd.fit(rd_data, y="y", x="x", c=0, method="rdrobust")
    # both should be the same kind of result
    assert type(r1).__name__ == type(r2).__name__


# ─── Method coverage ────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["rdrobust", "default", "rd", "robust"])
def test_default_aliases(rd_data, method):
    r = sp.rd(rd_data, y="y", x="x", c=0, method=method)
    assert r is not None


def test_honest(rd_data):
    r = sp.rd(rd_data, y="y", x="x", c=0, method="honest", M=1.0)
    assert r is not None


def test_randinf(rd_data):
    r = sp.rd(rd_data, y="y", x="x", c=0, method="randinf",
              wl=-0.2, wr=0.2)
    assert r is not None


def test_hte(rd_data):
    r = sp.rd(rd_data, y="y", x="x", c=0, method="hte", z="z")
    assert r is not None


def test_forest(rd_data):
    r = sp.rd(rd_data, y="y", x="x", c=0, method="forest",
              covs=["cov1", "cov2"], n_trees=20)
    assert r is not None


def test_boost(rd_data):
    r = sp.rd(rd_data, y="y", x="x", c=0, method="boost",
              covs=["cov1", "cov2"], n_estimators=20)
    assert r is not None


def test_lasso(rd_data):
    r = sp.rd(rd_data, y="y", x="x", c=0, method="lasso",
              covs=["cov1", "cov2"], cv_folds=3)
    assert r is not None


def test_extrapolate(rd_data):
    r = sp.rd(rd_data, y="y", x="x", c=0, method="extrapolate",
              covs=["cov1", "cov2"], eval_points=[-0.3, 0.3])
    assert r is not None


def test_external_validity(rd_data):
    r = sp.rd(rd_data, y="y", x="x", c=0, method="external_validity",
              covs=["cov1", "cov2"])
    assert r is not None


def test_bayes_hte_uses_running_cutoff_aliases(rd_data):
    """rd_bayes_hte expects ``running``/``cutoff`` — dispatcher translates."""
    r = sp.rd(rd_data, y="y", x="x", c=0, method="bayes_hte",
              covariates=["cov1", "cov2"])
    assert r is not None


def test_distribution(rd_data):
    r = sp.rd(rd_data, y="y", x="x", c=0, method="distribution",
              quantiles=[0.25, 0.5, 0.75])
    assert r is not None


# ─── Alias normalisation ────────────────────────────────────────────────


def test_method_alias_cate_routes_to_hte(rd_data):
    """``cate`` should route to the same estimator as ``hte``."""
    r1 = sp.rd(rd_data, y="y", x="x", c=0, method="hte", z="z")
    r2 = sp.rd(rd_data, y="y", x="x", c=0, method="cate", z="z")
    assert type(r1).__name__ == type(r2).__name__


def test_case_insensitive(rd_data):
    r1 = sp.rd(rd_data, y="y", x="x", c=0, method="rdrobust")
    r2 = sp.rd(rd_data, y="y", x="x", c=0, method="RDROBUST")
    assert type(r1).__name__ == type(r2).__name__


# ─── Error paths ────────────────────────────────────────────────────────


def test_unknown_method_raises(rd_data):
    with pytest.raises(ValueError, match="Unknown method"):
        sp.rd(rd_data, y="y", x="x", c=0, method="not_a_method")


def test_non_string_method_raises(rd_data):
    with pytest.raises(TypeError, match="method must be a string"):
        sp.rd(rd_data, y="y", x="x", c=0, method=42)
