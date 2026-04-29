"""Tests for the unified ``sp.match`` dispatcher (v1.10).

The classical ``sp.match`` function (nearest / stratify / cem /
psm / mahalanobis) was already a dispatcher.  v1.10 expands it
to also route to advanced matching/weighting estimators
(ebalance / cbps / sbw / overlap / genmatch / optimal /
cardinality), so callers can write ``sp.match(..., method='X')``
for any of the 12 estimator families this package implements
without remembering which submodule X lives in.

Numerical correctness of each estimator is covered by per-method
test files (``test_ebalance.py``, ``test_overlap_and_cbps.py``,
``test_matching.py``, ``test_matching_optimal.py``).  These tests
guard the dispatcher contract: alias coverage, kwarg forwarding,
back-compat of the standalone APIs, and clear error paths.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def match_data():
    """A treatment-effect DGP with selection-on-observables."""
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    ps = 1 / (1 + np.exp(-0.5 * x1 - 0.3 * x2))
    treat = (rng.uniform(size=n) < ps).astype(int)
    y = 1.0 + 2.0 * treat + 0.5 * x1 + 0.3 * x2 + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "treat": treat, "x1": x1, "x2": x2})


# ─── Classical methods (back-compat) ────────────────────────────────────


@pytest.mark.parametrize("method", ["nearest", "stratify", "cem",
                                     "psm", "mahalanobis"])
def test_classical_methods(match_data, method):
    r = sp.match(match_data, y="y", treat="treat",
                 covariates=["x1", "x2"], method=method)
    assert r is not None


# ─── Advanced methods (new in v1.10) ────────────────────────────────────


def test_ebalance(match_data):
    r = sp.match(match_data, y="y", treat="treat",
                 covariates=["x1", "x2"], method="ebalance")
    assert r is not None


def test_cbps(match_data):
    r = sp.match(match_data, y="y", treat="treat",
                 covariates=["x1", "x2"], method="cbps", n_bootstrap=20)
    assert r is not None


def test_sbw(match_data):
    r = sp.match(match_data, y="y", treat="treat",
                 covariates=["x1", "x2"], method="sbw")
    assert r is not None


def test_overlap_weights(match_data):
    r = sp.match(match_data, y="y", treat="treat",
                 covariates=["x1", "x2"], method="overlap",
                 n_bootstrap=20)
    assert r is not None


def test_genmatch(match_data):
    r = sp.match(match_data, y="y", treat="treat",
                 covariates=["x1", "x2"], method="genmatch",
                 population_size=20, generations=2)
    assert r is not None


def test_optimal(match_data):
    """optimal_match's signature uses ``treatment``/``outcome`` —
    dispatcher must translate from canonical ``treat``/``y``."""
    r = sp.match(match_data, y="y", treat="treat",
                 covariates=["x1", "x2"], method="optimal")
    assert r is not None


def test_cardinality(match_data):
    r = sp.match(match_data, y="y", treat="treat",
                 covariates=["x1", "x2"], method="cardinality",
                 smd_tolerance=0.2)
    assert r is not None


# ─── Aliases ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("alias,canon", [
    ("entropy", "ebalance"),
    ("entropy_balancing", "ebalance"),
    ("subclass", "stratify"),
    ("subclassification", "stratify"),
    ("coarsened_exact", "cem"),
    ("ow", "overlap"),
    ("overlap_weights", "overlap"),
    ("stable_balancing", "sbw"),
    ("genetic", "genmatch"),
    ("optimal_match", "optimal"),
    ("cardinality_match", "cardinality"),
])
def test_aliases_route_correctly(match_data, alias, canon):
    """``method=`` aliases should route to the same estimator type."""
    if canon == "overlap":
        r1 = sp.match(match_data, y="y", treat="treat",
                      covariates=["x1", "x2"], method=alias, n_bootstrap=10)
        r2 = sp.match(match_data, y="y", treat="treat",
                      covariates=["x1", "x2"], method=canon, n_bootstrap=10)
    elif canon == "cbps":
        r1 = sp.match(match_data, y="y", treat="treat",
                      covariates=["x1", "x2"], method=alias, n_bootstrap=10)
        r2 = sp.match(match_data, y="y", treat="treat",
                      covariates=["x1", "x2"], method=canon, n_bootstrap=10)
    elif canon == "genmatch":
        r1 = sp.match(match_data, y="y", treat="treat",
                      covariates=["x1", "x2"], method=alias,
                      population_size=10, generations=1)
        r2 = sp.match(match_data, y="y", treat="treat",
                      covariates=["x1", "x2"], method=canon,
                      population_size=10, generations=1)
    elif canon == "cardinality":
        r1 = sp.match(match_data, y="y", treat="treat",
                      covariates=["x1", "x2"], method=alias,
                      smd_tolerance=0.2)
        r2 = sp.match(match_data, y="y", treat="treat",
                      covariates=["x1", "x2"], method=canon,
                      smd_tolerance=0.2)
    else:
        r1 = sp.match(match_data, y="y", treat="treat",
                      covariates=["x1", "x2"], method=alias)
        r2 = sp.match(match_data, y="y", treat="treat",
                      covariates=["x1", "x2"], method=canon)
    assert type(r1).__name__ == type(r2).__name__


def test_case_insensitive(match_data):
    r1 = sp.match(match_data, y="y", treat="treat",
                  covariates=["x1", "x2"], method="ebalance")
    r2 = sp.match(match_data, y="y", treat="treat",
                  covariates=["x1", "x2"], method="EBALANCE")
    assert type(r1).__name__ == type(r2).__name__


# ─── Back-compat: standalone names ──────────────────────────────────────


def test_standalone_ebalance_still_works(match_data):
    r = sp.ebalance(match_data, y="y", treat="treat",
                    covariates=["x1", "x2"])
    assert r is not None


def test_standalone_cbps_still_works(match_data):
    r = sp.cbps(match_data, y="y", treat="treat",
                covariates=["x1", "x2"], n_bootstrap=20)
    assert r is not None


def test_standalone_genmatch_still_works(match_data):
    r = sp.genmatch(match_data, y="y", treat="treat",
                    covariates=["x1", "x2"],
                    population_size=20, generations=2)
    assert r is not None


def test_standalone_optimal_match_still_works(match_data):
    """optimal_match used directly takes ``treatment``/``outcome``."""
    r = sp.optimal_match(match_data, treatment="treat", outcome="y",
                         covariates=["x1", "x2"])
    assert r is not None


# ─── Error paths ────────────────────────────────────────────────────────


def test_unknown_method_raises(match_data):
    with pytest.raises(ValueError, match="method must be"):
        sp.match(match_data, y="y", treat="treat",
                 covariates=["x1", "x2"], method="not_a_method")


def test_non_string_method_raises(match_data):
    with pytest.raises(TypeError, match="method must be a string"):
        sp.match(match_data, y="y", treat="treat",
                 covariates=["x1", "x2"], method=42)


def test_classical_kwarg_blocked_on_advanced(match_data):
    """``caliper=`` is meaningful only for nearest-neighbour-style
    methods.  Passing it to ``method='ebalance'`` should error
    rather than silently ignore — the agent contract is "fail
    loudly."""
    with pytest.raises(TypeError,
                       match="does not accept these classical-matching kwargs"):
        sp.match(match_data, y="y", treat="treat",
                 covariates=["x1", "x2"], method="ebalance",
                 caliper=0.1)
