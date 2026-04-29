"""Tests for the unified ``sp.panel`` dispatcher (v1.10).

The classical ``sp.panel`` already had ``method=`` for fe / re / be /
fd / pooled / twoway / mundlak / chamberlain / ab / system.  v1.10:

  • Adds case-insensitive aliases (``Fixed`` → ``fe``, ``random`` →
    ``re``, ``between`` → ``be``, ``gmm`` → ``ab``,
    ``blundell_bond`` → ``system``, etc.).
  • Adds ``method='hdfe'`` (a.k.a. ``feols`` / ``reghdfe``) routing
    to ``feols.hdfe_ols`` for high-dimensional FE absorption — the
    Stata ``reghdfe`` / R ``fixest::feols`` slot.

Numerical correctness of each estimator is covered by the existing
panel test files (``test_panel.py``, ``test_panel_dynamic.py``,
``test_feols.py``).  These tests guard the dispatcher contract.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def panel_data():
    """A balanced panel with entity + time FE structure."""
    rng = np.random.default_rng(0)
    n_id, T = 50, 8
    ids = np.repeat(np.arange(n_id), T)
    years = np.tile(np.arange(T), n_id)
    n = len(ids)
    alpha_i = rng.standard_normal(n_id)[ids]
    mu_t = rng.standard_normal(T)[years]
    exp = rng.standard_normal(n) + 0.3 * alpha_i
    edu = rng.standard_normal(n)
    y = (0.5 * exp + 0.3 * edu + alpha_i + mu_t
         + 0.5 * rng.standard_normal(n))
    return pd.DataFrame({
        "id": ids, "year": years,
        "wage": y, "exp": exp, "edu": edu,
    })


# ─── Classical methods (back-compat) ────────────────────────────────────


@pytest.mark.parametrize("method", [
    "fe", "re", "be", "fd", "pooled", "twoway",
    "mundlak", "chamberlain",
])
def test_classical_methods(panel_data, method):
    r = sp.panel(panel_data, "wage ~ exp + edu",
                 entity="id", time="year", method=method)
    assert isinstance(r, sp.PanelResults)


# ─── Aliases ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("alias,canon", [
    ("Fixed", "fe"),
    ("fixed_effects", "fe"),
    ("within", "fe"),
    ("random", "re"),
    ("random_effects", "re"),
    ("between", "be"),
    ("between_effects", "be"),
    ("first_difference", "fd"),
    ("first_diff", "fd"),
    ("pols", "pooled"),
    ("ols", "pooled"),
    ("pooled_ols", "pooled"),
    ("two_way", "twoway"),
    ("2way", "twoway"),
    ("two_way_fe", "twoway"),
    ("mundlak_cre", "mundlak"),
    ("chamberlain_cre", "chamberlain"),
])
def test_aliases_route_correctly(panel_data, alias, canon):
    r1 = sp.panel(panel_data, "wage ~ exp + edu",
                  entity="id", time="year", method=alias)
    r2 = sp.panel(panel_data, "wage ~ exp + edu",
                  entity="id", time="year", method=canon)
    assert type(r1).__name__ == type(r2).__name__


def test_case_insensitive(panel_data):
    r1 = sp.panel(panel_data, "wage ~ exp + edu",
                  entity="id", time="year", method="fe")
    r2 = sp.panel(panel_data, "wage ~ exp + edu",
                  entity="id", time="year", method="FE")
    r3 = sp.panel(panel_data, "wage ~ exp + edu",
                  entity="id", time="year", method="Fe")
    assert type(r1).__name__ == type(r2).__name__ == type(r3).__name__


# ─── HDFE absorption (new in v1.10) ─────────────────────────────────────


def test_hdfe_with_auto_formula(panel_data):
    """When formula has no `|`, dispatcher bolts entity+time on as FE."""
    r = sp.panel(panel_data, "wage ~ exp + edu",
                 entity="id", time="year", method="hdfe")
    # HDFE returns FEOLSResult, not PanelResults
    assert type(r).__name__ == "FEOLSResult"


def test_hdfe_with_explicit_fe(panel_data):
    """Caller-supplied `| fe1 + fe2` formula must be respected."""
    r = sp.panel(panel_data, "wage ~ exp + edu | id + year",
                 method="hdfe")
    assert type(r).__name__ == "FEOLSResult"


@pytest.mark.parametrize("alias", ["hdfe", "feols", "reghdfe", "absorbed_ols"])
def test_hdfe_aliases(panel_data, alias):
    r = sp.panel(panel_data, "wage ~ exp + edu",
                 entity="id", time="year", method=alias)
    assert type(r).__name__ == "FEOLSResult"


def test_hdfe_requires_formula(panel_data):
    with pytest.raises(ValueError, match="hdfe.*requires a formula"):
        sp.panel(panel_data, formula=None, entity="id", time="year",
                 method="hdfe")


# ─── Back-compat: standalone names ──────────────────────────────────────


def test_standalone_hdfe_ols_still_works(panel_data):
    r = sp.hdfe_ols("wage ~ exp + edu | id + year", data=panel_data)
    assert type(r).__name__ == "FEOLSResult"


def test_panel_logit_still_top_level():
    """panel_logit is intentionally NOT in the method= table; it should
    still be reachable as ``sp.panel_logit`` for callers that need it."""
    assert callable(sp.panel_logit)
    assert callable(sp.panel_probit)


# ─── Error paths ────────────────────────────────────────────────────────


def test_unknown_method_raises(panel_data):
    with pytest.raises(ValueError, match="method must be"):
        sp.panel(panel_data, "wage ~ exp", entity="id", time="year",
                 method="not_a_method")


def test_non_string_method_raises(panel_data):
    with pytest.raises(TypeError, match="method must be a string"):
        sp.panel(panel_data, "wage ~ exp", entity="id", time="year",
                 method=42)
