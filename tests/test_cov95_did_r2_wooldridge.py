"""Round-2 coverage for statspai.did.wooldridge_did:
wooldridge_did (controls / no-cohort error), etwfe dispatcher branches
(xvar, repeated cross-section, never-only), drdid (covariates, trad method,
no-covariate path), twfe_decomposition, etwfe_emfx aggregations.

Uses real synthetic staggered panels; asserts shapes / signs / valid p-values,
never fabricated numbers."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _stag(seed=0, n_units=120, n_periods=8):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 4, 6][u % 3]
        fe = rng.normal()
        for t in range(1, n_periods + 1):
            te = 1.5 * (t - g + 1) if (g > 0 and t >= g) else 0.0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": fe + 0.4 * t + te + rng.normal(0, 0.4),
                    "first_treat": float(g) if g > 0 else np.nan,
                    "x1": rng.normal(),
                    "x2": rng.normal(),
                    "cl": u % 10,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def stag():
    return _stag()


# ---------------------------------------------------------------- wooldridge_did
def test_wooldridge_with_controls(stag):
    r = sp.wooldridge_did(
        stag,
        y="y",
        group="unit",
        time="time",
        first_treat="first_treat",
        controls=["x1", "x2"],
    )
    assert r.estimate > 0
    assert r.se > 0
    assert 0 <= r.pvalue <= 1


def test_wooldridge_explicit_cluster(stag):
    r = sp.wooldridge_did(
        stag, y="y", group="unit", time="time", first_treat="first_treat", cluster="cl"
    )
    assert r.se > 0


def test_wooldridge_no_cohort_raises(stag):
    bad = stag.copy()
    bad["first_treat"] = np.nan
    with pytest.raises(ValueError):
        sp.wooldridge_did(
            bad, y="y", group="unit", time="time", first_treat="first_treat"
        )


# ---------------------------------------------------------------- etwfe branches
def test_etwfe_basic(stag):
    r = sp.etwfe(stag, y="y", group="unit", time="time", first_treat="first_treat")
    assert r.se > 0
    assert isinstance(r.model_info, dict)
    assert "cohorts" in r.model_info


def test_etwfe_with_xvar(stag):
    r = sp.etwfe(
        stag, y="y", group="unit", time="time", first_treat="first_treat", xvar=["x1"]
    )
    assert r.se > 0


def test_etwfe_with_xvar_string(stag):
    r = sp.etwfe(
        stag,
        y="y",
        group="unit",
        time="time",
        first_treat="first_treat",
        xvar="x1",
        controls=["x2"],
    )
    assert r.se > 0


def test_etwfe_repeated_cross_section(stag):
    r = sp.etwfe(
        stag, y="y", group="unit", time="time", first_treat="first_treat", panel=False
    )
    assert r.se > 0


def test_etwfe_never_only_cgroup(stag):
    r = sp.etwfe(
        stag,
        y="y",
        group="unit",
        time="time",
        first_treat="first_treat",
        cgroup="nevertreated",
    )
    assert r.se > 0


# ---------------------------------------------------------------- drdid
def test_drdid_with_covariates():
    rng = np.random.default_rng(3)
    n = 600
    G = rng.integers(0, 2, n)
    T = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n)
    y = 1 + 0.5 * x + 2 * G + 3 * T + 4 * G * T + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "treated": G, "post": T, "x": x})
    r = sp.drdid(
        df,
        y="y",
        group="treated",
        time="post",
        covariates=["x"],
        n_boot=80,
        random_state=0,
    )
    assert abs(r.estimate - 4.0) < 1.5
    assert r.se > 0
    assert 0 <= r.pvalue <= 1


def test_drdid_traditional_method():
    rng = np.random.default_rng(4)
    n = 500
    G = rng.integers(0, 2, n)
    T = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n)
    y = 1 + 0.5 * x + 2 * G + 3 * T + 4 * G * T + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "treated": G, "post": T, "x": x})
    r = sp.drdid(
        df,
        y="y",
        group="treated",
        time="post",
        covariates=["x"],
        method="trad",
        n_boot=60,
        random_state=1,
    )
    assert r.se > 0


def test_drdid_no_covariates():
    rng = np.random.default_rng(5)
    n = 500
    G = rng.integers(0, 2, n)
    T = rng.integers(0, 2, n)
    y = 1 + 2 * G + 3 * T + 4 * G * T + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "treated": G, "post": T})
    r = sp.drdid(df, y="y", group="treated", time="post", n_boot=60, random_state=2)
    assert r.se >= 0


# ---------------------------------------------------------------- twfe_decomposition
def test_twfe_decomposition(stag):
    r = sp.twfe_decomposition(
        stag, y="y", group="unit", time="time", first_treat="first_treat"
    )
    assert r.detail is not None
    assert len(r.detail) > 0
    assert {"estimate", "weight"}.issubset(set(r.detail.columns))


# ---------------------------------------------------------------- etwfe_emfx
@pytest.fixture(scope="module")
def fit(stag):
    return sp.etwfe(stag, y="y", group="unit", time="time", first_treat="first_treat")


def test_emfx_simple(fit):
    r = sp.etwfe_emfx(fit, type="simple")
    assert r.se > 0


def test_emfx_group(fit):
    r = sp.etwfe_emfx(fit, type="group")
    assert len(r.detail) >= 1


def test_emfx_event(fit):
    r = sp.etwfe_emfx(fit, type="event", include_leads=True)
    assert len(r.detail) >= 1


def test_emfx_calendar(fit):
    r = sp.etwfe_emfx(fit, type="calendar")
    assert len(r.detail) >= 1


def test_emfx_treated_weighting(fit):
    r = sp.etwfe_emfx(fit, type="simple", weighting="treated")
    assert r.se >= 0


def test_emfx_bad_type_raises(fit):
    with pytest.raises(ValueError):
        sp.etwfe_emfx(fit, type="nonsense")


def test_emfx_bad_weighting_raises(fit):
    with pytest.raises(ValueError):
        sp.etwfe_emfx(fit, weighting="nonsense")
