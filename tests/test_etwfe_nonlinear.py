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
    # small-sample factor (statsmodels' cluster default vs ppmlhdfe's
    # G/(G-1)); agreement to well under 1% on 1,800 rows.
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
    assert uni.estimate == pytest.approx(float(w @ ref.params[names]), abs=1e-8)
    # Same sandwich; only the cluster count differs.  ppmlhdfe drops the
    # separated units and applies G'/(G'-1); etwfe keeps them in G (jwdid's
    # reporting) and applies G/(G-1) -- ppmlhdfe's clustered convention,
    # with no (N-1)/(N-K) term.
    V = ref.data_info["var_cov"]
    idx = [list(ref.params.index).index(c) for c in names]
    se_ref = float(np.sqrt(w @ V[np.ix_(idx, idx)] @ w))
    G, Gp = uni.model_info["n_clusters"], ref.data_info["n_clusters"]
    ratio = (G / (G - 1)) / (Gp / (Gp - 1))
    assert uni.se == pytest.approx(se_ref * np.sqrt(ratio), rel=1e-7)
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
        _fit(df, fe="within")
    with pytest.raises(MethodIncompatibility, match="fe='cohort'"):
        _fit(df, fe="cohort", hettype="event")  # linear hettype absorbs units
    # the linear model has one scale; either spelling is accepted
    assert _fit(df, scale="link").estimate == pytest.approx(_fit(df).estimate)


# ---------------------------------------------------------------------------
# Aggregation weights, joint event-study covariance and its consumers
#
# Regressions caught by redoing the jwdid replication after 027813fb: the
# aggregation weights had been switched to non-separated rows only (the
# 21 published coefficients stopped matching; the response ATT inflated
# ~6x), the event-study covariance was weighted by 1/(n1 n2) (honest_did
# returned NaN / zero-width sets), and pretrends_test still fell back to
# the diagonal (chi2 29.9 against jwdid's 52.65).
# ---------------------------------------------------------------------------


def _treated_counts(df):
    d = df.dropna(subset=["y"])
    return d[d["g"] > 0].groupby(["g", "year"]).size()


def test_separated_rows_stay_in_aggregation_weights():
    df = _panel(seed=5, zero_units=0.4)
    zero = df.groupby("id")["y"].transform("sum") == 0
    res = _fit(df, family="poisson", fe="unit", scale="link")
    cells = res.model_info["cells"].set_index(["cohort", "period"])
    counts = _treated_counts(df)
    # every treated row counts, all-zero ones included (jwdid's weights)
    for key, n in cells["n_treated"].items():
        assert n == counts[key]
    post = cells[cells["post"]]
    w = post["n_treated"] / post["n_treated"].sum()
    assert res.estimate == pytest.approx(float(w @ post["coef"]), abs=1e-12)
    # A separated row's fitted mean and marginal effect are exactly zero:
    # it enters the response-scale average through the denominator only.
    kept = _fit(df[~zero], family="poisson", fe="unit", scale="link")
    n_all = res.model_info["aggregations"]["response"]["simple"]["n_treated"]
    n_kept = kept.model_info["aggregations"]["response"]["simple"]["n_treated"]
    assert n_all > n_kept
    assert res.model_info["att_response"] * n_all == pytest.approx(
        kept.model_info["att_response"] * n_kept, rel=1e-8
    )


@pytest.mark.parametrize("scale", ["link", "response"])
def test_event_study_vcov_is_joint_and_matches_the_table(scale):
    df = _panel(seed=7, zero_units=0.2)
    res = _fit(df, family="poisson", fe="unit", cgroup="nevertreated", scale=scale)
    es = res.model_info["event_study"]
    V = res.model_info["event_study_vcov"]
    assert list(V.index) == [int(t) for t in es["relative_time"]]
    assert not V.attrs.get("block_diagonal", False)
    np.testing.assert_allclose(V.to_numpy(), V.to_numpy().T, atol=1e-15)
    assert np.linalg.eigvalsh(V.to_numpy()).min() > -1e-12
    np.testing.assert_allclose(
        np.sqrt(np.diag(V.to_numpy())), es["se"].to_numpy(), rtol=1e-10
    )
    if scale == "link":
        # link scale: W V W' with W the within-event-time treated shares
        mi = res.model_info
        names = list(mi["coef_names"])
        Vc = pd.DataFrame(mi["vcov"], index=names, columns=names)
        cells = mi["cells"]
        W = []
        for tau in V.index:
            c = cells[cells["relative_time"] == tau]
            row = pd.Series(0.0, index=names)
            row[[f"treat[{g},{t}]" for g, t in zip(c["cohort"], c["period"])]] = (
                c["n_treated"] / c["n_treated"].sum()
            ).to_numpy()
            W.append(row)
        W = pd.DataFrame(W)
        np.testing.assert_allclose(V.to_numpy(), (W @ Vc @ W.T).to_numpy(), atol=1e-14)
    evc = sp.event_study_vcov(res, allow_diagonal=False)
    assert evc.joint and evc.source == "model_info['event_study_vcov']"


def test_emfx_switches_table_and_covariance_together():
    df = _panel(seed=7)
    res = _fit(df, family="poisson", fe="unit", cgroup="nevertreated", scale="link")
    out = sp.etwfe_emfx(res, type="event", scale="response", include_leads=True)
    V = out.model_info["event_study_vcov"]
    es = out.model_info["event_study"]
    resp = res.model_info["aggregations"]["response"]["event"]
    np.testing.assert_allclose(es["att"].to_numpy(), resp["att"].to_numpy())
    np.testing.assert_allclose(
        np.sqrt(np.diag(V.to_numpy())), es["se"].to_numpy(), rtol=1e-10
    )


def test_pretrends_test_uses_the_joint_covariance_and_window():
    df = _panel(seed=7, zero_units=0.2)
    res = _fit(df, family="poisson", fe="unit", cgroup="nevertreated", scale="link")
    V = res.model_info["event_study_vcov"]
    es = res.model_info["event_study"].set_index("relative_time")
    leads = [t for t in V.index if t < 0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no diagonal fallback
        out = sp.pretrends_test(res)
    b = es.loc[leads, "att"].to_numpy()
    S = V.loc[leads, leads].to_numpy()
    assert out["statistic"] == pytest.approx(
        float(b @ np.linalg.solve(S, b)), rel=1e-10
    )
    assert out["df"] == len(leads) and out["pre_periods"] == leads
    diag = float(np.sum(b**2 / np.diag(S)))
    assert abs(out["statistic"] - diag) > 1e-3  # really not the diagonal
    win = [t for t in leads if -3 <= t <= -2]
    sub = sp.pretrends_test(res, window=(-3, -2))
    bw = es.loc[win, "att"].to_numpy()
    Sw = V.loc[win, win].to_numpy()
    assert sub["statistic"] == pytest.approx(
        float(bw @ np.linalg.solve(Sw, bw)), rel=1e-10
    )
    assert sub["pre_periods"] == win and sub["df"] == len(win)


def test_pretrends_test_window_validation():
    df = _panel(seed=7)
    res = _fit(df, family="poisson", fe="unit", cgroup="nevertreated", scale="link")
    from statspai.exceptions import DataInsufficient

    with pytest.raises(MethodIncompatibility, match="lo <= hi"):
        sp.pretrends_test(res, window=(-2, -5))
    with pytest.raises(MethodIncompatibility, match="pair"):
        sp.pretrends_test(res, window=-3)
    with pytest.raises(DataInsufficient, match="window"):
        sp.pretrends_test(res, window=(-99, -90))


def test_honest_did_reads_the_poisson_etwfe_covariance():
    df = _panel(seed=7, zero_units=0.2)
    res = _fit(df, family="poisson", fe="unit", cgroup="nevertreated", scale="link")
    es = res.model_info["event_study"].set_index("relative_time")
    est, se = float(es.loc[0, "att"]), float(es.loc[0, "se"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no worst-case fallback
        sd = sp.honest_did(res, e=0, method="smoothness", m_grid=[0.0, 0.05])
    assert sd.attrs.get("interval") == "flci"
    rm = sp.honest_did(res, e=0, method="relative_magnitude", m_grid=[0.0])
    lo, hi = float(rm["ci_lower"].iloc[0]), float(rm["ci_upper"].iloc[0])
    # Mbar = 0 imposes exact parallel trends after treatment: about the
    # conventional interval, never a NaN or a sliver.
    assert np.isfinite(lo) and np.isfinite(hi) and lo < est < hi
    assert 0.8 < (hi - lo) / (2 * 1.959964 * se) < 1.2
    assert (sd["ci_upper"] - sd["ci_lower"]).min() > 0.5 * 2 * 1.959964 * se


# ── response_se: profiled unit effect vs. Stata margins ─────────────────


def _fit_unit(df, **kw):
    return _fit(df, family="poisson", fe="unit", **kw)


def test_response_se_changes_only_response_scale_ses():
    """Point estimates, the link scale and the coefficient covariance do
    not depend on the convention; the response-scale SEs do."""
    df = _panel(n_units=300, seed=3)
    p = _fit_unit(df).model_info
    m = _fit_unit(df, response_se="margins").model_info
    assert (p["response_se"], m["response_se"]) == ("profile", "margins")
    np.testing.assert_array_equal(p["coefficients"], m["coefficients"])
    np.testing.assert_array_equal(p["vcov"], m["vcov"])
    for tab in ("event", "group", "calendar"):
        lp, lm = p["aggregations"]["link"][tab], m["aggregations"]["link"][tab]
        np.testing.assert_allclose(lp["se"], lm["se"], rtol=1e-12)
        rp, rm = p["aggregations"]["response"][tab], m["aggregations"]["response"][tab]
        np.testing.assert_allclose(rp["att"], rm["att"], rtol=1e-12)
        assert not np.allclose(rp["se"], rm["se"], rtol=1e-3)
    assert p["se_response"] != pytest.approx(m["se_response"], rel=1e-3)


def test_response_se_margins_event_vcov_is_the_table_covariance():
    df = _panel(n_units=300, seed=4)
    r = _fit_unit(df, response_se="margins", cgroup="nevertreated")
    V = r.model_info["event_study_vcov"]
    ev = r.model_info["event_study"].set_index("relative_time")
    np.testing.assert_allclose(np.sqrt(np.diag(V)), ev.loc[V.index, "se"], rtol=1e-12)


def test_response_se_conventions_coincide_without_absorbed_effects():
    """With cohort dummies nothing is absorbed, so margins' gradient is
    the profiled one."""
    df = _panel(n_units=300, seed=5)
    a = _fit(df, family="poisson", fe="cohort")
    b = _fit(df, family="poisson", fe="cohort", response_se="margins")
    assert a.se == pytest.approx(b.se, rel=1e-12)
    lin = _fit(df, fe="unit", response_se="margins")
    assert lin.se == pytest.approx(_fit(df, fe="unit").se, rel=1e-12)


def test_response_se_margins_constant_block_zero_iff_clusters_nest_units():
    """ppmlhdfe's constant has score y - mu, which the Poisson FOC sums to
    zero within every unit (and, the periods being regressors, within every
    period): its covariance with the slopes vanishes when the clusters nest
    either, and not otherwise."""
    from statspai.did._etwfe_glm_fit import _fit_poisson_unit_fe

    df = _panel(n_units=200, seed=6)
    y = df["y"].to_numpy(float)
    units = pd.factorize(df["id"])[0].astype(np.intp)
    years = pd.factorize(df["year"])[0].astype(np.intp)
    mixed = ((df["id"] + df["year"]) % 7).to_numpy().astype(np.intp)
    X = pd.get_dummies(df["year"], drop_first=True).to_numpy(float)
    X = np.column_stack([X, ((df["g"] > 0) & (df["year"] >= df["g"])).to_numpy(float)])
    for cl, nested in ((units, True), (years, True), (mixed, False)):
        res = _fit_poisson_unit_fe(y, X, units, cl, int(cl.max()) + 1, len(y))
        Vc = res["vcov_cons"]
        np.testing.assert_allclose(Vc[:-1, :-1], res["vcov"], rtol=1e-12)
        if nested:
            assert np.abs(Vc[-1]).max() < 1e-10 * np.abs(Vc).max()
        else:
            assert np.abs(Vc[-1]).max() > 1e-4 * np.abs(Vc).max()


def test_response_se_invalid_value_raises():
    df = _panel(n_units=120, seed=7)
    with pytest.raises(MethodIncompatibility, match="response_se"):
        _fit_unit(df, response_se="stata")
    with pytest.raises(MethodIncompatibility, match="response_se"):
        _fit(df, response_se="unconditional")
