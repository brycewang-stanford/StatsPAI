"""Reference parity for three DiD variants: continuous dose, time-varying
covariates, and the distributional DiD.

What is pinned, and against what
--------------------------------
``sp.continuous_did(method="twfe")``
    ``fixest::feols(y ~ dp | id + time)`` with ``dp = dose * post`` (0.14.0),
    point estimate and SE, on a balanced and an unbalanced panel, with and
    without a control, under ``vcov = "iid"``, ``~id`` and ``~region``.
    Conventions the SE depends on (fixest defaults): iid ``sigma^2 = RSS /
    (n - K)`` with ``K`` = slopes + every absorbed fixed-effect parameter
    (``N + T - 1`` on a connected panel); clustered ``G/(G-1) * (n-1)/(n-K)``
    with ``K`` = slopes + fixed effects **not nested** in the cluster
    (``ssc(fixef.K = "nested")``; unit effects are nested in both ``id`` and
    ``region`` here, so ``K = k + T``: all ``T`` period levels count, because
    the unit effect that would absorb their collinearity is the nested one
    being dropped -- ``did._core.fe_dof_not_nested``). p-values / CIs are
    normal in StatsPAI and t in fixest; only estimates and SEs are compared.
``sp.continuous_did(method="att_gt" | "dose_response")``
    Heuristic modes with no canonical reference. Pinned by reference-free
    identities instead (analytic 2x2 SE, exact recovery of a linear dose
    response).
``sp.did_timevarying_covariates``
    ``ptetools::pte_default(d_outcome = TRUE, est_method = "reg", xformula =
    ~x1 + x2, control_group = "nevertreated")`` (1.0.1) -- every post-period
    ATT(g, t) and the overall (group-aggregated) ATT -- and the same cells from
    ``did::att_gt(xformla = ~x1 + x2, est_method = "reg")`` (2.3.0) with
    ``aggte(type = "simple" | "group")``. The two R packages agree with each
    other to ~1e-14, which is the independent check that the target is the
    ``X_{g-1}`` outcome-regression estimator. SEs are a unit bootstrap here and
    a multiplier bootstrap there (T3), so they are not compared.
``sp.distributional_did``
    ``didFF::distDD`` (0.1.0) on ``did::mpdta`` for the configurations the
    didFF extended fixture does not reach: covariates under dr / reg / ipw,
    not-yet-treated comparisons, dynamic / calendar aggregation, the dynamic
    event window, binpoints, and weights + covariates + not-yet-treated. The
    ``balance_e = 1`` case crashes ``distDD`` itself (a degenerate bin makes
    its output table ragged), so that case is pinned against distDD's own
    recipe re-run step by step in R (``distdd_manual``); the same recipe
    reproduces ``distDD``'s ``dynamic`` output, checked here too.

Tolerance
---------
All comparisons are deterministic. ``rtol = 1e-9`` on estimates and SEs
(``atol = 1e-11`` for bin masses that are exactly or nearly zero). Observed
agreement is 1e-15 to 1e-12 except the propensity-score (ipw / dr with
covariates) distDD cases, which sit at ~2e-10 relative on SEs because both
sides iterate a logit to a convergence tolerance.

Regenerate
----------
``python tests/reference_parity/_generate_did_synth_didvar_data.py`` (panels)
then ``Rscript tests/reference_parity/_generate_did_synth_didvar_R.R``.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).resolve().parent / "_fixtures"
_REF_PATH = _FIX / "did_synth_didvar_R.json"

_RTOL = 1e-9
_ATOL = 1e-11


def _load_ref() -> dict:
    if not _REF_PATH.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing reference fixture: {_REF_PATH}")
    return json.loads(_REF_PATH.read_text(encoding="utf-8"))


_REF = _load_ref()


def _read(name: str) -> pd.DataFrame:
    path = _FIX / name
    if not path.exists():  # pragma: no cover
        pytest.skip(f"missing panel fixture: {path}")
    return pd.read_csv(path)


def _num(v) -> np.ndarray:
    return np.array([np.nan if (x is None or x == "NA") else x for x in v], dtype=float)


def test_fixture_meta():
    meta = _REF["meta"]
    assert meta["fixest_version"] == "0.14.0"
    assert meta["ptetools_version"] == "1.0.1"
    assert meta["did_version"] == "2.3.0"
    assert meta["didFF_version"] == "0.1.0"


# --------------------------------------------------------------------------
# continuous_did(method="twfe") vs fixest
# --------------------------------------------------------------------------

_TWFE_CASES = {
    "iid": {},
    "cl_id": {"cluster": "id"},
    "cl_region": {"cluster": "region"},
    "ctrl_iid": {"controls": ["x"]},
    "ctrl_cl_id": {"controls": ["x"], "cluster": "id"},
}


@pytest.mark.parametrize("panel", ["balanced", "unbalanced"])
@pytest.mark.parametrize("case", sorted(_TWFE_CASES))
def test_continuous_did_twfe_matches_fixest(panel, case):
    fname = (
        "did_synth_didvar_contdose.csv"
        if panel == "balanced"
        else "did_synth_didvar_contdose_unbal.csv"
    )
    df = _read(fname)
    ref = _REF["twfe"][panel][case]
    res = sp.continuous_did(
        df,
        "y",
        "dose",
        "time",
        "id",
        t_pre=3,
        t_post=4,
        method="twfe",
        **_TWFE_CASES[case],
    )
    assert res.n_obs == ref["nobs"]
    np.testing.assert_allclose(res.estimate, ref["coef"], rtol=_RTOL)
    np.testing.assert_allclose(res.se, ref["se"], rtol=_RTOL)


def test_continuous_did_twfe_unbalanced_is_not_one_pass_demeaning():
    """On the unbalanced panel the slope differs from the one-pass transform.

    ``y - ybar_i - ybar_t + ybar`` is the exact two-way within transform only
    on balanced panels; the reference slope is recovered only by the
    iterated projection. Guard that the fixture really is unbalanced enough to
    tell the two apart.
    """
    df = _read("did_synth_didvar_contdose_unbal.csv")
    df["dp"] = df["dose"] * (df["time"] >= 4)
    one_pass = {}
    for c in ("y", "dp"):
        one_pass[c] = (
            df[c]
            - df.groupby("id")[c].transform("mean")
            - df.groupby("time")[c].transform("mean")
            + df[c].mean()
        )
    naive = float(one_pass["dp"] @ one_pass["y"] / (one_pass["dp"] @ one_pass["dp"]))
    ref = _REF["twfe"]["unbalanced"]["iid"]["coef"]
    assert abs(naive / ref - 1) > 1e-4


# --------------------------------------------------------------------------
# continuous_did(method="att_gt") -- heuristic, reference-free checks
# --------------------------------------------------------------------------


def _two_period_panel(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dose = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(1, 3, n))
    a = rng.normal(0, 1, n)
    rows = []
    for i in range(n):
        for t in (0, 1):
            rows.append(
                {
                    "id": i,
                    "time": t,
                    "dose": dose[i],
                    "y": a[i] + 0.7 * dose[i] * t + rng.normal(0, 1 + dose[i] * t),
                }
            )
    return pd.DataFrame(rows)


def test_att_gt_single_bin_matches_analytic_2x2():
    """One dose bin, two periods: the point estimate is the 2x2 DID of means
    and its bootstrap SE must match the analytic ``sqrt(s_t^2/n_t + s_c^2/n_c)``
    of the long differences up to Monte-Carlo error.

    A bootstrap that collapses repeated draws of the same unit (sampling
    ~63% of the units without multiplicity) understates this SE badly; this
    is the check that caught it.
    """
    df = _two_period_panel()
    res = sp.continuous_did(
        df,
        "y",
        "dose",
        "time",
        "id",
        post=None,
        t_pre=0,
        t_post=1,
        method="att_gt",
        n_quantiles=1,
        n_boot=4000,
        seed=3,
    )
    wide = df.pivot(index="id", columns="time", values="y")
    dy = wide[1] - wide[0]
    dose = df.groupby("id")["dose"].first()
    tr, co = dy[dose > 0], dy[dose == 0]
    att = tr.mean() - co.mean()
    se = np.sqrt(tr.var(ddof=0) / len(tr) + co.var(ddof=0) / len(co))
    np.testing.assert_allclose(res.estimate, att, rtol=1e-12)
    np.testing.assert_allclose(res.detail["att"].iloc[0], att, rtol=1e-12)
    # 4000 draws: MC sd of a bootstrap SE ~ 1/sqrt(2B) ~ 1.1%; 5% is ~4.5 sd.
    assert abs(res.se / se - 1) < 0.05


def test_att_gt_pooled_se_reflects_shared_control():
    """The pooled SE is bootstrapped jointly, not ``sqrt(sum w^2 se^2)``.

    Every bin is compared with the same zero-dose units, so the bin DIDs are
    positively correlated and the independence formula understates the
    pooled SE.
    """
    df = _read("did_synth_didvar_contdose.csv")
    res = sp.continuous_did(
        df,
        "y",
        "dose",
        "time",
        "id",
        t_pre=3,
        t_post=4,
        method="att_gt",
        n_boot=1000,
        seed=0,
    )
    w = res.detail["n_treated"] / res.detail["n_treated"].sum()
    indep = float(np.sqrt(np.sum(w**2 * res.detail["se"] ** 2)))
    assert res.se > 1.2 * indep


def test_att_gt_warns_that_controls_are_ignored():
    df = _read("did_synth_didvar_contdose.csv")
    with pytest.warns(UserWarning, match="controls"):
        sp.continuous_did(
            df,
            "y",
            "dose",
            "time",
            "id",
            t_pre=3,
            t_post=4,
            method="att_gt",
            controls=["x"],
            n_boot=10,
            seed=0,
        )


def test_att_gt_fallback_control_has_finite_se():
    """With no zero-dose units the lowest bin is the comparison arm; the
    bootstrap must resample that arm too (it used to resample the empty
    zero-dose set and return NaN)."""
    df = _read("did_synth_didvar_contdose.csv")
    df = df[df["dose"] > 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.continuous_did(
            df,
            "y",
            "dose",
            "time",
            "id",
            t_pre=3,
            t_post=4,
            method="att_gt",
            n_quantiles=4,
            n_boot=200,
            seed=0,
        )
    assert res.model_info["control_arm"] == "lowest_dose_bin"
    assert np.isfinite(res.se) and res.se > 0


# --------------------------------------------------------------------------
# continuous_did(method="dose_response") -- reference-free checks
# --------------------------------------------------------------------------


def test_dose_response_recovers_exact_linear_slope():
    """A local-linear fit reproduces a linear ``dY`` exactly, so the average
    derivative is the slope and every bootstrap replicate equals it."""
    n = 60
    dose = np.linspace(0, 4, n)
    rows = []
    for i in range(n):
        rows.append({"id": i, "time": 0, "dose": dose[i], "y": float(i % 5)})
        rows.append(
            {"id": i, "time": 1, "dose": dose[i], "y": float(i % 5) + 2.5 * dose[i]}
        )
    df = pd.DataFrame(rows)
    res = sp.continuous_did(
        df,
        "y",
        "dose",
        "time",
        "id",
        t_pre=0,
        t_post=1,
        method="dose_response",
        n_boot=50,
        seed=0,
    )
    np.testing.assert_allclose(res.estimate, 2.5, rtol=1e-10)
    assert res.se < 1e-8
    assert res.model_info["se_method"].startswith("unit bootstrap")


# --------------------------------------------------------------------------
# did_timevarying_covariates vs ptetools / did
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tvc() -> pd.DataFrame:
    return _read("did_synth_didvar_tvc.csv")


def _cells(res) -> pd.DataFrame:
    return res.detail.set_index(["cohort", "time"])["att_gt"]


def _ref_cells(block: dict) -> pd.Series:
    idx = pd.MultiIndex.from_arrays(
        [np.asarray(block["group"], int), np.asarray(block["time"], int)],
        names=["cohort", "time"],
    )
    return pd.Series(np.asarray(block["att"], float), index=idx)


def test_timevarying_cells_match_ptetools_and_did(tvc):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.did_timevarying_covariates(
            tvc,
            "y",
            unit="id",
            time="time",
            cohort="g",
            covariates=["x1", "x2"],
            n_boot=5,
            seed=0,
        )
    got = _cells(res)
    for key in ("tvc_pte", "tvc_did"):
        ref = _ref_cells(_REF[key])
        assert set(ref.index) == set(got.index), key
        np.testing.assert_allclose(
            got.loc[ref.index].to_numpy(), ref.to_numpy(), rtol=_RTOL, err_msg=key
        )


def test_timevarying_overall_matches_references(tvc):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grp = sp.did_timevarying_covariates(
            tvc,
            "y",
            unit="id",
            time="time",
            cohort="g",
            covariates=["x1", "x2"],
            n_boot=5,
            seed=0,
        )
        smp = sp.did_timevarying_covariates(
            tvc,
            "y",
            unit="id",
            time="time",
            cohort="g",
            covariates=["x1", "x2"],
            aggregation="simple",
            n_boot=5,
            seed=0,
        )
    np.testing.assert_allclose(grp.estimate, _REF["tvc_pte"]["overall_att"], rtol=_RTOL)
    np.testing.assert_allclose(grp.estimate, _REF["tvc_did"]["group_att"], rtol=_RTOL)
    np.testing.assert_allclose(smp.estimate, _REF["tvc_did"]["simple_att"], rtol=_RTOL)
    assert grp.model_info["att_simple"] == pytest.approx(smp.estimate, rel=1e-14)


def test_timevarying_single_covariate_matches_did(tvc):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.did_timevarying_covariates(
            tvc,
            "y",
            unit="id",
            time="time",
            cohort="g",
            covariates=["x1"],
            n_boot=5,
            seed=0,
        )
    ref = _REF["tvc_did_x1"]
    got = _cells(res)
    r = _ref_cells(ref)
    np.testing.assert_allclose(got.loc[r.index].to_numpy(), r.to_numpy(), rtol=_RTOL)
    np.testing.assert_allclose(res.estimate, ref["group_att"], rtol=_RTOL)
    np.testing.assert_allclose(
        res.model_info["att_simple"], ref["simple_att"], rtol=_RTOL
    )


def test_timevarying_comparison_units_use_the_cohort_baseline(tvc):
    """Identity: comparison units' covariates enter cohort g only at g - 1.

    Perturb every never-treated unit's covariate in period 4. That is cohort
    5's baseline, so cohort-5 cells may move; it is *not* the baseline of
    cohorts 4 (period 3) or 6 (period 5), so their cells must not move. The
    earlier implementation froze never-treated covariates at each unit's
    median period (4 here) for every cohort, and failed this.
    """
    kw = dict(
        unit="id", time="time", cohort="g", covariates=["x1", "x2"], n_boot=2, seed=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = _cells(sp.did_timevarying_covariates(tvc, "y", **kw))
        bumped = tvc.copy()
        mask = (bumped["g"] == 0) & (bumped["time"] == 4)
        bumped.loc[mask, "x1"] += 100.0 * np.sin(bumped.loc[mask, "id"])
        moved = _cells(sp.did_timevarying_covariates(bumped, "y", **kw))
    for g in (4, 6):
        np.testing.assert_allclose(
            moved.loc[g].to_numpy(), base.loc[g].to_numpy(), rtol=1e-12
        )
    assert np.max(np.abs(moved.loc[5].to_numpy() - base.loc[5].to_numpy())) > 1e-6


def test_timevarying_rejects_unknown_aggregation(tvc):
    with pytest.raises(ValueError, match="aggregation"):
        sp.did_timevarying_covariates(
            tvc,
            "y",
            unit="id",
            time="time",
            cohort="g",
            covariates=["x1"],
            aggregation="dynamic",
            n_boot=2,
        )


# --------------------------------------------------------------------------
# distributional_did vs didFF::distDD
# --------------------------------------------------------------------------

_DIST_KW = dict(y="lemp", g="first_treat", t="year", i="countyreal")
_DIST_CASES = {
    "x_dr": dict(x=["lpop"]),
    "x_reg": dict(x=["lpop"], estimator="reg"),
    "x_ipw": dict(x=["lpop"], estimator="ipw"),
    "notyet": dict(control_group="notyettreated"),
    "dynamic": dict(aggregation="dynamic"),
    "dynamic_window": dict(aggregation="dynamic", min_e=-2, max_e=2),
    "calendar": dict(aggregation="calendar"),
    "weighted_x_notyet": dict(x=["lpop"], weights="w", control_group="notyettreated"),
    "binpoints": dict(binpoints=[4, 5, 6, 7, 8, 9], n_bins=None),
}


@pytest.fixture(scope="module")
def mpdta() -> pd.DataFrame:
    return _read("did_synth_didvar_mpdta.csv")


def _dist(mpdta, **kw):
    kw.setdefault("n_bins", 6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.distributional_did(mpdta, **_DIST_KW, **kw)


@pytest.mark.parametrize("name", sorted(_DIST_CASES))
def test_distributional_did_matches_distdd(mpdta, name):
    ref = _REF["distdd"][name]
    assert "error" not in ref, ref
    res = _dist(mpdta, **_DIST_CASES[name])
    np.testing.assert_allclose(
        res.table["estimate"].to_numpy(float),
        _num(ref["estimates"]),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{name}: estimates",
    )
    np.testing.assert_allclose(
        res.table["se"].to_numpy(float),
        _num(ref["se"]),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{name}: se",
    )
    assert res.diagnostics["effect_sum"] == pytest.approx(0.0, abs=1e-10)


def test_distdd_crashes_on_balance_e_and_manual_recipe_is_distdd(mpdta):
    """``distDD`` errors on ``balance_e = 1``; the manual recipe stands in.

    The manual recipe is first shown to *be* distDD by reproducing its
    ``dynamic`` output, then used as the reference for ``balance_e = 1``.
    """
    assert "differing number of rows" in _REF["distdd"]["dynamic_balance"]["error"]
    man_dyn = _REF["distdd_manual"]["dynamic"]
    np.testing.assert_allclose(
        _num(man_dyn["estimates"]),
        _num(_REF["distdd"]["dynamic"]["estimates"]),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        _num(man_dyn["se"]), _num(_REF["distdd"]["dynamic"]["se"]), rtol=1e-12
    )
    man = _REF["distdd_manual"]["dynamic_balance"]
    res = _dist(mpdta, aggregation="dynamic", balance_e=1)
    np.testing.assert_allclose(
        res.table["estimate"].to_numpy(float),
        _num(man["estimates"]),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        res.table["se"].to_numpy(float), _num(man["se"]), rtol=_RTOL, atol=_ATOL
    )
    assert res.table["used"].tolist() == list(man["used"])
