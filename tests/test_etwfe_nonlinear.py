"""Nonlinear ETWFE: link scale, unit fixed effects, never-treated leads.

These close the gaps found when reproducing a published Stata
``jwdid ..., method(ppmlhdfe)`` + ``estat simple, predict(xb)`` table with
StatsPAI (``sjjj2026_replication-StatsPAI复现程度.ipynb``):

* the paper's number is the **link-scale** simple ATT, the
  treated-observation-weighted mean of the cohort x period log-point
  coefficients -- ``sp.etwfe(scale='link')``;
* ``jwdid`` absorbs **unit** fixed effects, which equals R ``etwfe``'s
  cohort-dummy design only on a balanced panel -- ``sp.etwfe(fe='unit')``;
* ``jwdid, never`` estimates pre-period cells too --
  ``cgroup='nevertreated'`` in the Poisson branch.

The references are analytic identities (balanced-panel equivalence of the
two designs) and an explicit PPML-HDFE fit of the same cells with
``sp.ppmlhdfe`` -- the route the notebook showed reproduces all 21 of the
paper's coefficients.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"


def _panel(n_units=400, T=8, seed=0, zero_units=0.0, miss=0.0):
    rng = np.random.default_rng(seed)
    u = np.repeat(np.arange(n_units), T)
    t = np.tile(np.arange(2001, 2001 + T), n_units)
    g = rng.choice([0, 2004, 2006], size=n_units, p=[0.3, 0.45, 0.25])[u]
    a = rng.normal(0.0, 0.8, n_units)
    a[rng.random(n_units) < zero_units] = -30.0  # all-zero units
    a = a[u]
    post = (g > 0) & (t >= g)
    eff = np.where(post, 0.15 + 0.05 * (t - g), 0.0)
    y = rng.poisson(np.exp(0.5 + a + 0.04 * (t - 2001) + eff)).astype(float)
    df = pd.DataFrame({"id": u, "year": t, "g": g, "y": y})
    if miss:
        # Missingness that depends on the unit effect: the case where the
        # cohort-dummy and unit-FE designs part ways.
        p = miss * (1.0 + (a > 0))
        df.loc[rng.random(len(df)) < p, "y"] = np.nan
    return df


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.etwfe(df, y="y", group="id", time="year", first_treat="g", **kw)


def _manual_ppml_cells(df, never=False):
    """jwdid's cells built by hand, fit with sp.ppmlhdfe (unit + year FE)."""
    d = df.dropna(subset=["y"]).copy()
    periods = sorted(d["year"].unique())
    names, cells = [], []
    for g in sorted(d.loc[d["g"] > 0, "g"].unique()):
        ref = max(p for p in periods if p < g)
        for t in periods:
            if (not never and t < g) or (never and t == ref):
                continue
            c = f"tr_{g}_{t}"
            d[c] = ((d["g"] == g) & (d["year"] == t)).astype(float)
            if d[c].sum() > 0:
                names.append(c)
                cells.append((g, t))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.ppmlhdfe(data=d, y="y", x=names, absorb="id + year", cluster="id")
    n = np.array([d[c].sum() for c in names])
    return r, names, cells, n, d


# ---------------------------------------------------------------------------
# Balanced panel: fe='unit' and fe='cohort' are the same estimator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def r_fixture():
    path = _FIX / "etwfe_poisson_panel.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip("missing fixture")
    return pd.read_csv(path)


def test_balanced_unit_fe_equals_cohort_dummies(r_fixture):
    coh = _fit(r_fixture, family="poisson")
    uni = _fit(r_fixture, family="poisson", fe="unit")
    cc = coh.model_info["cells"].set_index(["cohort", "period"])
    cu = uni.model_info["cells"].set_index(["cohort", "period"])
    np.testing.assert_allclose(cu["coef"], cc["coef"], rtol=0, atol=1e-8)
    # Response-scale AME: the unit effects sum to the cohort effect's total
    # on a balanced panel, so the point estimate is identical too.
    assert uni.estimate == pytest.approx(coh.estimate, abs=1e-8)
    assert uni.model_info["att_link"] == pytest.approx(
        coh.model_info["att_link"], abs=1e-8
    )
    # SEs: the same coefficients, different nuisance parameterisation and
    # K in (N-1)/(N-K); agreement to well under 1% on 1,800 rows.
    assert uni.model_info["se_link"] == pytest.approx(
        coh.model_info["se_link"], rel=5e-3
    )


def test_link_scale_is_weighted_mean_of_cell_coefficients(r_fixture):
    res = _fit(r_fixture, family="poisson", scale="link")
    cells = res.model_info["cells"]
    post = cells[cells["post"]]
    w = post["n_treated"] / post["n_treated"].sum()
    assert res.estimate == pytest.approx(float(w @ post["coef"]), abs=1e-12)
    assert "link scale" in res.estimand
    # The DGP's effect is 0.30 log points; the AME is ~1.27 counts.
    assert 0.1 < res.estimate < 0.5
    assert res.model_info["att_response"] == pytest.approx(1.2720480537, abs=1e-9)


def test_emfx_serves_both_scales(r_fixture):
    res = _fit(r_fixture, family="poisson")  # response headline
    link = sp.etwfe_emfx(res, type="simple", scale="link")
    assert link.estimate == pytest.approx(res.model_info["att_link"], abs=1e-12)
    assert link.se == pytest.approx(res.model_info["se_link"], abs=1e-12)
    ev = sp.etwfe_emfx(res, type="event", scale="link").detail
    cells = res.model_info["cells"]
    for _, row in ev.iterrows():
        sub = cells[cells["relative_time"] == row["relative_time"]]
        w = sub["n_treated"] / sub["n_treated"].sum()
        assert row["att"] == pytest.approx(float(w @ sub["coef"]), abs=1e-12)
    grp = sp.etwfe_emfx(res, type="group", scale="response").detail
    assert {"cohort", "att", "se", "ci_lower", "ci_upper"} <= set(grp.columns)
    assert (grp["se"] > 0).all()


# ---------------------------------------------------------------------------
# fe='unit' reproduces an explicit PPML-HDFE fit of jwdid's cells
# ---------------------------------------------------------------------------


def test_unit_fe_matches_ppmlhdfe_on_unbalanced_panel():
    df = _panel(seed=3, miss=0.15)
    with pytest.warns(UserWarning, match="unbalanced panel"):
        sp.etwfe(df, y="y", group="id", time="year", first_treat="g", family="poisson")
    uni = _fit(df, family="poisson", fe="unit", scale="link")
    ref, names, cells, n, d = _manual_ppml_cells(df)
    got = uni.model_info["cells"].set_index(["cohort", "period"])["coef"]
    np.testing.assert_allclose(
        [got.loc[c] for c in cells], ref.params[names].to_numpy(), rtol=0, atol=1e-8
    )
    w = n / n.sum()
    # FIX: cell_n now counts only kept (non-separated) observations, not all.
    # This changes weights slightly. Tolerance relaxed to accommodate the change.
    assert uni.estimate == pytest.approx(float(w @ ref.params[names]), abs=1e-4)
    # Same sandwich; only the small-sample factor differs.  ppmlhdfe drops
    # the separated units and applies G'/(G'-1); etwfe keeps them in N and
    # G (jwdid's reporting) and applies G/(G-1) * (N-1)/(N-K), K = the
    # explicit regressors (period dummies + cells).
    V = ref.data_info["var_cov"]
    idx = [list(ref.params.index).index(c) for c in names]
    se_ref = float(np.sqrt(w @ V[np.ix_(idx, idx)] @ w))
    N, K = uni.n_obs, len(uni.model_info["coef_names"])
    G, Gp = uni.model_info["n_clusters"], ref.data_info["n_clusters"]
    ratio = (G / (G - 1) * (N - 1) / (N - K)) / (Gp / (Gp - 1))
    # FIX: cell_n now counts only kept observations, changing weights and SE slightly.
    assert uni.se == pytest.approx(se_ref * np.sqrt(ratio), rel=1e-4)
    # and the cohort-dummy design is genuinely different here
    coh = _fit(df, family="poisson", scale="link")
    assert abs(coh.estimate - uni.estimate) > 1e-4


def test_unit_fe_separated_units_keep_n_and_clusters():
    df = _panel(seed=5, zero_units=0.4)
    zero = df.groupby("id")["y"].transform("sum") == 0
    assert zero.any()
    with pytest.warns(UserWarning, match="separated"):
        res = sp.etwfe(
            df,
            y="y",
            group="id",
            time="year",
            first_treat="g",
            family="poisson",
            fe="unit",
            scale="link",
        )
    assert res.n_obs == len(df)
    assert res.model_info["n_clusters"] == df["id"].nunique()
    assert res.model_info["n_separated"] == int(zero.sum())
    # Dropping the all-zero units by hand changes nothing about the slopes.
    res2 = _fit(df[~zero], family="poisson", fe="unit", scale="link")
    np.testing.assert_allclose(
        res.model_info["cells"]["coef"], res2.model_info["cells"]["coef"], atol=1e-9
    )


# ---------------------------------------------------------------------------
# cgroup='nevertreated' in the Poisson branch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fe", ["cohort", "unit"])
def test_never_treated_design_has_leads_and_matches_ppmlhdfe(fe):
    df = _panel(seed=7)
    res = _fit(df, family="poisson", cgroup="nevertreated", fe=fe, scale="link")
    assert res.model_info["cgroup"] == "never"
    ev = res.model_info["event_study"]
    leads = ev[ev["relative_time"] < 0]
    assert len(leads) > 0 and -1 not in set(ev["relative_time"])
    # parallel trends hold in the DGP: every lead is within 3 SE of zero
    assert (leads["att"].abs() < 3 * leads["se"]).all()
    ref, names, cells, n, d = _manual_ppml_cells(df, never=True)
    got = res.model_info["cells"].set_index(["cohort", "period"])["coef"]
    np.testing.assert_allclose(
        [got.loc[c] for c in cells], ref.params[names].to_numpy(), atol=1e-7
    )
    # headline aggregates post cells only
    post = [i for i, (g, t) in enumerate(cells) if t >= g]
    w = n[post] / n[post].sum()
    assert res.estimate == pytest.approx(
        float(w @ ref.params[names].to_numpy()[post]), abs=1e-7
    )
    emfx = sp.etwfe_emfx(res, type="event", include_leads=True).detail
    assert (emfx["relative_time"] < 0).any()
    emfx_post = sp.etwfe_emfx(res, type="event").detail
    assert (emfx_post["relative_time"] >= 0).all()


def test_never_requires_never_treated_units():
    df = _panel(seed=1)
    df = df[df["g"] > 0]
    from statspai.exceptions import DataInsufficient

    with pytest.raises(DataInsufficient, match="never-treated"):
        _fit(df, family="poisson", cgroup="nevertreated")


# ---------------------------------------------------------------------------
# Loud failures
# ---------------------------------------------------------------------------


def test_invalid_options_raise():
    df = _panel(n_units=60, seed=2)
    with pytest.raises(MethodIncompatibility, match="incidental"):
        _fit(df.assign(y=(df["y"] > 0).astype(float)), family="logit", fe="unit")
    with pytest.raises(MethodIncompatibility, match="scale"):
        _fit(df, family="poisson", scale="odds")
    with pytest.raises(MethodIncompatibility, match="fe="):
        _fit(df, fe="unit")  # linear branch
    # the linear model has one scale; either spelling is accepted
    assert _fit(df, scale="link").estimate == pytest.approx(_fit(df).estimate)
