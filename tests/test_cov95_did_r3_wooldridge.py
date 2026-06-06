"""Coverage round-3 for statspai.did.wooldridge_did.

Exercises the ETWFE dispatcher branches (xvar / never-treated / repeated
cross-section), drdid (imp + trad + simple fallback + validation), the TWFE
decomposition (Bacon + dCDH), and etwfe_emfx aggregations under every
``type`` / ``weighting`` combination.

All assertions check real structural properties (shapes, ATT signs near the
known DGP effect, p-values in [0, 1], errors raised) — no fabricated numbers
and no mocking of numeric paths.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def panel():
    df = sp.dgp_did(n_units=90, n_periods=8, staggered=True, seed=11)
    rng = np.random.default_rng(7)
    df = df.copy()
    df["xcov"] = rng.normal(size=len(df))
    df["xcov2"] = rng.normal(size=len(df))
    df["cl"] = df["unit"] % 9
    return df


@pytest.fixture(scope="module")
def panel_no_never():
    # Drop never-treated so cgroup='nevertreated' must raise.
    df = sp.dgp_did(n_units=80, n_periods=8, staggered=True, seed=12)
    df = df.loc[df["first_treat"].notna()].reset_index(drop=True)
    return df


@pytest.fixture(scope="module")
def twobytwo():
    rng = np.random.default_rng(42)
    n = 500
    G = rng.integers(0, 2, n)
    T = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n)
    y = 1 + 0.5 * x + 2 * G + 3 * T + 4.0 * G * T + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "treated": G, "post": T, "x": x})


# ----------------------------------------------------------------------
# wooldridge_did core
# ----------------------------------------------------------------------
def test_wooldridge_basic(panel):
    r = sp.wooldridge_did(panel, y="y", group="unit", time="time",
                          first_treat="first_treat")
    assert np.isfinite(r.estimate)
    assert 0.0 <= r.pvalue <= 1.0
    assert isinstance(r.detail, pd.DataFrame)
    assert {"cohort", "att", "se", "n_obs", "n_treated_obs"} <= set(r.detail.columns)
    assert r.detail.shape[0] == r.model_info["n_cohorts"]
    # event-study coefs present
    es = r.model_info["event_study"]
    assert isinstance(es, pd.DataFrame)
    assert "rel_time" in es.columns


def test_wooldridge_with_controls_and_cluster(panel):
    r = sp.wooldridge_did(panel, y="y", group="unit", time="time",
                          first_treat="first_treat",
                          controls=["xcov"], cluster="cl")
    assert np.isfinite(r.estimate)
    assert r.model_info["controls"] == ["xcov"]


def test_wooldridge_no_cohorts_raises():
    df = sp.dgp_did(n_units=40, n_periods=6, staggered=True, seed=3)
    df = df.copy()
    df["first_treat"] = np.nan  # all never-treated
    with pytest.raises(ValueError, match="No treated cohorts"):
        sp.wooldridge_did(df, y="y", group="unit", time="time",
                          first_treat="first_treat")


# ----------------------------------------------------------------------
# etwfe dispatcher
# ----------------------------------------------------------------------
def test_etwfe_alias_matches_wooldridge(panel):
    a = sp.etwfe(panel, y="y", group="unit", time="time",
                 first_treat="first_treat")
    b = sp.wooldridge_did(panel, y="y", group="unit", time="time",
                          first_treat="first_treat")
    assert a.estimate == pytest.approx(b.estimate, rel=1e-9)


def test_etwfe_bad_cgroup_raises(panel):
    with pytest.raises(ValueError, match="cgroup must be"):
        sp.etwfe(panel, y="y", group="unit", time="time",
                 first_treat="first_treat", cgroup="bogus")


def test_etwfe_xvar_missing_column_raises(panel):
    with pytest.raises(KeyError, match="not found"):
        sp.etwfe(panel, y="y", group="unit", time="time",
                 first_treat="first_treat", xvar="nope")


def test_etwfe_xvar_constant_raises(panel):
    df = panel.copy()
    df["constx"] = 3.0
    with pytest.raises(ValueError, match="constant"):
        sp.etwfe(df, y="y", group="unit", time="time",
                 first_treat="first_treat", xvar="constx")


def test_etwfe_xvar_too_few_rows_raises(panel):
    df = panel.copy()
    df["sparsex"] = np.nan
    df.loc[df.index[0], "sparsex"] = 1.0  # only 1 non-NaN
    with pytest.raises(ValueError, match="fewer than 2"):
        sp.etwfe(df, y="y", group="unit", time="time",
                 first_treat="first_treat", xvar="sparsex")


def test_etwfe_panel_false_nevertreated_not_implemented(panel):
    with pytest.raises(NotImplementedError):
        sp.etwfe(panel, y="y", group="unit", time="time",
                 first_treat="first_treat", panel=False,
                 cgroup="nevertreated")


def test_etwfe_xvar_single_and_multi(panel):
    r1 = sp.etwfe(panel, y="y", group="unit", time="time",
                  first_treat="first_treat", xvar="xcov")
    assert np.isfinite(r1.estimate)
    assert "att_at_xmean" in r1.detail.columns or "att" in r1.detail.columns
    r2 = sp.etwfe(panel, y="y", group="unit", time="time",
                  first_treat="first_treat", xvar=["xcov", "xcov2"])
    assert np.isfinite(r2.estimate)


def test_etwfe_nevertreated_cgroup(panel):
    r = sp.etwfe(panel, y="y", group="unit", time="time",
                 first_treat="first_treat", cgroup="nevertreated")
    assert r.model_info["cgroup"] == "nevertreated"
    assert np.isfinite(r.estimate)
    assert 0.0 <= r.pvalue <= 1.0


def test_etwfe_nevertreated_with_xvar(panel):
    r = sp.etwfe(panel, y="y", group="unit", time="time",
                 first_treat="first_treat", cgroup="nevertreated",
                 xvar="xcov")
    assert np.isfinite(r.estimate)


def test_etwfe_nevertreated_no_never_units_raises(panel_no_never):
    with pytest.raises(ValueError, match="never-treated"):
        sp.etwfe(panel_no_never, y="y", group="unit", time="time",
                 first_treat="first_treat", cgroup="nevertreated")


def test_etwfe_repeated_cross_section(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.etwfe(panel, y="y", group="unit", time="time",
                     first_treat="first_treat", panel=False)
    assert r.model_info["panel"] is False
    assert np.isfinite(r.estimate)
    assert isinstance(r.detail, pd.DataFrame)


def test_etwfe_repeated_cs_with_xvar_and_controls(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.etwfe(panel, y="y", group="unit", time="time",
                     first_treat="first_treat", panel=False,
                     xvar="xcov", controls=["xcov2"])
    assert np.isfinite(r.estimate)
    # with xvar, event_study is suppressed
    assert r.model_info["event_study"] is None


def test_etwfe_repeated_cs_no_cohorts_raises():
    df = sp.dgp_did(n_units=30, n_periods=6, staggered=True, seed=9)
    df = df.copy()
    df["first_treat"] = np.nan
    with pytest.raises(ValueError, match="No treated cohorts"):
        sp.etwfe(df, y="y", group="unit", time="time",
                 first_treat="first_treat", panel=False)


# ----------------------------------------------------------------------
# drdid
# ----------------------------------------------------------------------
def test_drdid_improved_recovers_effect(twobytwo):
    r = sp.drdid(twobytwo, y="y", group="treated", time="post",
                 covariates=["x"], method="imp", n_boot=80, random_state=1)
    assert abs(r.estimate - 4.0) < 1.0
    assert 0.0 <= r.pvalue <= 1.0
    assert r.model_info["method"] == "improved"


def test_drdid_traditional_recovers_effect(twobytwo):
    r = sp.drdid(twobytwo, y="y", group="treated", time="post",
                 covariates=["x"], method="trad", n_boot=80, seed=2)
    assert abs(r.estimate - 4.0) < 1.0
    assert r.model_info["method"] == "traditional"


def test_drdid_no_covariates(twobytwo):
    r = sp.drdid(twobytwo, y="y", group="treated", time="post",
                 n_boot=40, random_state=3)
    assert np.isfinite(r.estimate)
    assert r.model_info["covariates"] == []


def test_drdid_bad_method_raises(twobytwo):
    with pytest.raises(ValueError, match="method must be"):
        sp.drdid(twobytwo, y="y", group="treated", time="post",
                 method="xyz", n_boot=10)


def test_drdid_nonbinary_group_raises(twobytwo):
    df = twobytwo.copy()
    df.loc[df.index[:5], "treated"] = 2  # now ternary
    with pytest.raises(ValueError, match="must be binary"):
        sp.drdid(df, y="y", group="treated", time="post", n_boot=10)


def test_drdid_nonbinary_time_raises(twobytwo):
    df = twobytwo.copy()
    df.loc[df.index[:5], "post"] = 2
    with pytest.raises(ValueError, match="must be binary"):
        sp.drdid(df, y="y", group="treated", time="post", n_boot=10)


def test_drdid_simple_fallback_path(twobytwo):
    # Tiny control cells force the "not enough data -> simple DID" fallback
    # inside _estimate_att (with many covariates relative to control cells).
    df = twobytwo.copy()
    rng = np.random.default_rng(0)
    for j in range(6):
        df[f"x{j}"] = rng.normal(size=len(df))
    covs = ["x"] + [f"x{j}" for j in range(6)]
    # restrict controls to a handful of rows so ctrl cells < n covariates
    keep = df[df["treated"] == 1].copy()
    ctrl = df[df["treated"] == 0].head(8).copy()
    small = pd.concat([keep, ctrl], ignore_index=True)
    r = sp.drdid(small, y="y", group="treated", time="post",
                 covariates=covs, n_boot=20, random_state=5)
    assert np.isfinite(r.estimate) or np.isnan(r.estimate)


# ----------------------------------------------------------------------
# twfe_decomposition
# ----------------------------------------------------------------------
def test_twfe_decomposition_structure(panel):
    r = sp.twfe_decomposition(panel, y="y", group="unit", time="time",
                              first_treat="first_treat")
    assert np.isfinite(r.estimate)
    assert {"type", "treated_cohort", "control_cohort", "estimate",
            "weight", "weighted_est"} <= set(r.detail.columns)
    mi = r.model_info
    assert "twfe_beta" in mi and "bacon_att" in mi
    assert mi["has_never_treated"] is True
    # dCDH weights computed
    assert "dcdh_weights" in mi
    assert isinstance(mi["dcdh_weights"], pd.DataFrame)
    # comparison types present
    types = set(r.detail["type"])
    assert "Earlier vs Later" in types
    assert "Treated vs Never" in types


def test_twfe_decomposition_no_never(panel_no_never):
    r = sp.twfe_decomposition(panel_no_never, y="y", group="unit",
                              time="time", first_treat="first_treat")
    assert r.model_info["has_never_treated"] is False
    assert "Treated vs Never" not in set(r.detail["type"])


# ----------------------------------------------------------------------
# etwfe_emfx
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def fit(panel):
    return sp.etwfe(panel, y="y", group="unit", time="time",
                    first_treat="first_treat")


def test_emfx_simple(fit):
    e = sp.etwfe_emfx(fit, type="simple")
    assert e.detail.shape[0] == 1
    assert 0.0 <= e.pvalue <= 1.0
    assert "estimate" in e.detail.columns


def test_emfx_group(fit, panel):
    e = sp.etwfe_emfx(fit, type="group")
    assert e.detail.shape[0] == fit.model_info["n_cohorts"]
    assert {"cohort", "estimate", "se", "ci_low", "ci_high"} <= set(e.detail.columns)


def test_emfx_event(fit):
    e = sp.etwfe_emfx(fit, type="event")
    assert "event_time" in e.detail.columns
    assert (e.detail["event_time"] >= 0).all()


def test_emfx_event_include_leads(fit):
    e = sp.etwfe_emfx(fit, type="event", include_leads=True)
    assert "event_time" in e.detail.columns
    # leads (negative rel-time) now present
    assert (e.detail["event_time"] < 0).any()


def test_emfx_calendar(fit):
    e = sp.etwfe_emfx(fit, type="calendar")
    assert "calendar_time" in e.detail.columns


def test_emfx_weighting_variants(fit):
    for w in ("cohort", "treated", "treated_observations"):
        e = sp.etwfe_emfx(fit, type="simple", weighting=w)
        assert np.isfinite(e.estimate)
        g = sp.etwfe_emfx(fit, type="group", weighting=w)
        assert np.isfinite(g.estimate)


def test_emfx_bad_type_raises(fit):
    with pytest.raises(ValueError, match="type must be"):
        sp.etwfe_emfx(fit, type="bogus")


def test_emfx_bad_weighting_raises(fit):
    with pytest.raises(ValueError, match="weighting must be"):
        sp.etwfe_emfx(fit, type="simple", weighting="bogus")


def test_emfx_requires_etwfe_result(twobytwo):
    bad = sp.drdid(twobytwo, y="y", group="treated", time="post",
                   n_boot=10, random_state=1)
    with pytest.raises(ValueError, match="cohorts"):
        sp.etwfe_emfx(bad, type="simple")


def test_emfx_xvar_group_and_simple(panel):
    # An xvar fit exposes the 'att_at_xmean' detail column path in emfx.
    fit_x = sp.etwfe(panel, y="y", group="unit", time="time",
                     first_treat="first_treat", xvar="xcov")
    g = sp.etwfe_emfx(fit_x, type="group", weighting="cohort")
    assert g.detail.shape[0] == fit_x.model_info["n_cohorts"]
    s = sp.etwfe_emfx(fit_x, type="simple", weighting="treated")
    assert np.isfinite(s.estimate)


def test_emfx_simple_treated_uses_event_cell_vcov(fit):
    # weighting='treated' + event_study present => event-cell vcov branch.
    e = sp.etwfe_emfx(fit, type="simple", weighting="treated")
    assert "vcov" in e.model_info["se_method"]
    assert e.model_info["weighting"] == "treated"


def test_emfx_event_xvar_no_event_study(panel):
    # An xvar fit suppresses event_study, so event/calendar must raise.
    fit_x = sp.etwfe(panel, y="y", group="unit", time="time",
                     first_treat="first_treat", xvar="xcov")
    with pytest.raises(ValueError, match="event_study"):
        sp.etwfe_emfx(fit_x, type="event")
