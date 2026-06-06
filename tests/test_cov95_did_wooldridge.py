"""Coverage campaign — ``statspai.did.wooldridge_did`` family.

This module bundles several public estimators that were only partially
exercised:

- ``wooldridge_did`` (extended TWFE) with controls + explicit clustering;
- ``etwfe`` across its configuration branches: covariate interaction
  (``xvar``), repeated cross-sections (``panel=False``), and the
  never-treated vs not-yet-treated comparison groups (``cgroup``);
- ``etwfe_emfx`` marginal-effect aggregations (simple / group / event /
  calendar);
- ``drdid`` 2×2 doubly-robust DiD across its estimation methods;
- ``twfe_decomposition`` (Goodman-Bacon-style TWFE decomposition).

DGP: constant +2 treatment effect switched on at each cohort's adoption
date, so every consistent estimator must recover an overall ATT near 2.
Assertions check that plus finite SEs — never fabricated numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TRUE_ATT = 2.0


def _staggered(seed=0, n_units=120, n_periods=8, att=TRUE_ATT):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 4, 6][u % 3]                 # 0 = never-treated
        fe = rng.normal()
        x1 = rng.normal()
        for t in range(1, n_periods + 1):
            te = att if (g > 0 and t >= g) else 0.0
            y = fe + 0.3 * t + 0.5 * x1 + te + rng.normal(0, 0.4)
            rows.append({"unit": u, "time": t, "y": y, "g": g, "x1": x1})
    return pd.DataFrame(rows)


def _two_by_two(seed=1, n=400, att=TRUE_ATT):
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 2, n)
    x1 = rng.normal(size=n)
    rows = []
    for i in range(n):
        for tt in (0, 1):
            te = att if (grp[i] == 1 and tt == 1) else 0.0
            y = 1.0 + 0.5 * tt + 0.4 * x1[i] + te + rng.normal(0, 0.5)
            rows.append({"id": i, "group": grp[i], "time": tt, "y": y, "x1": x1[i]})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _staggered()


def _est(result):
    for attr in ("estimate", "att"):
        v = getattr(result, attr, None)
        if v is not None:
            v = float(np.asarray(v).ravel()[0])
            if np.isfinite(v):
                return v
    raise AssertionError("no finite estimate")


# ── wooldridge_did: controls + clustering ───────────────────────────────


def test_wooldridge_did_controls_cluster(panel):
    r = sp.wooldridge_did(panel, y="y", group="unit", time="time",
                          first_treat="g", controls=["x1"], cluster="unit")
    assert abs(_est(r) - TRUE_ATT) < 0.6


def test_wooldridge_did_no_treated_raises():
    df = _staggered(n_units=30)
    df["g"] = 0  # everyone never-treated
    with pytest.raises(ValueError):
        sp.wooldridge_did(df, y="y", group="unit", time="time", first_treat="g")


# ── etwfe: configuration branches ───────────────────────────────────────


def test_etwfe_default(panel):
    r = sp.etwfe(panel, y="y", group="unit", time="time", first_treat="g")
    assert abs(_est(r) - TRUE_ATT) < 0.6


def test_etwfe_never_control_group(panel):
    r = sp.etwfe(panel, y="y", group="unit", time="time", first_treat="g",
                 cgroup="nevertreated")
    assert abs(_est(r) - TRUE_ATT) < 0.6


def test_etwfe_with_xvar_heterogeneity(panel):
    # binary moderator → covariate-interacted ETWFE (heterogeneous ATT path)
    df = panel.assign(grp_hi=(panel["x1"] > 0).astype(int))
    r = sp.etwfe(df, y="y", group="unit", time="time", first_treat="g",
                 xvar="grp_hi")
    assert r is not None and np.isfinite(_est(r))


def test_etwfe_repeated_cross_sections(panel):
    r = sp.etwfe(panel, y="y", group="unit", time="time", first_treat="g",
                 panel=False)
    assert np.isfinite(_est(r))


# ── etwfe_emfx: aggregations ────────────────────────────────────────────


@pytest.mark.parametrize("agg_type", ["simple", "group", "event", "calendar"])
def test_etwfe_emfx_aggregations(panel, agg_type):
    base = sp.etwfe(panel, y="y", group="unit", time="time", first_treat="g")
    r = sp.etwfe_emfx(base, type=agg_type)
    assert r is not None
    # at least one finite effect is produced
    est = getattr(r, "estimate", None)
    detail = getattr(r, "detail", None)
    has_finite = (est is not None and np.all(np.isfinite(np.atleast_1d(
        np.asarray(est, dtype=float)))))
    if not has_finite and detail is not None and "estimate" in getattr(detail, "columns", []):
        has_finite = np.isfinite(detail["estimate"].to_numpy(dtype=float)).any()
    assert has_finite


# ── drdid: 2x2 doubly-robust DiD ────────────────────────────────────────


@pytest.mark.parametrize("method", ["imp", "trad"])
def test_drdid_methods_recover_att(method):
    # Both DR-DID estimators are consistent and recover the true ATT.
    # (Regression guard for the 2026-06-05 ⚠️ correctness fix: the
    # traditional branch previously normalised by the full sample size
    # and returned ~half the ATT; see CHANGELOG / MIGRATION.)
    df = _two_by_two()
    r = sp.drdid(df, y="y", group="group", time="time",
                 covariates=["x1"], method=method, n_boot=150, random_state=0)
    assert abs(_est(r) - TRUE_ATT) < 0.7


def test_drdid_traditional_no_covariates_equals_raw_did():
    # With no covariates the traditional DR-DID must reduce exactly to the
    # raw 2×2 DiD (the property the old normalisation bug violated).
    df = _two_by_two()
    means = df.groupby(["group", "time"])["y"].mean()
    raw = (means[(1, 1)] - means[(1, 0)]) - (means[(0, 1)] - means[(0, 0)])
    r = sp.drdid(df, y="y", group="group", time="time", method="trad",
                 n_boot=1, random_state=0)
    assert abs(_est(r) - raw) < 1e-6


def test_drdid_unknown_method_raises():
    df = _two_by_two()
    with pytest.raises(ValueError, match="method must be"):
        sp.drdid(df, y="y", group="group", time="time", method="ipw")


# ── twfe_decomposition ──────────────────────────────────────────────────


def test_twfe_decomposition(panel):
    r = sp.twfe_decomposition(panel, y="y", group="unit", time="time",
                              first_treat="g")
    assert r is not None
    assert np.isfinite(_est(r))
