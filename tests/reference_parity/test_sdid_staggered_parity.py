"""sp.sdid(treat=...) against Stata ``sdid`` 2.0.2 (T2).

Fixtures: ``_fixtures/sdid_staggered/`` holds the two author-distributed
datasets of Clarke, Pailanir, Athey & Imbens (2024) -- Proposition 99 (block
design, one treated unit) and the gender-quota panel (seven adoption
cohorts) -- and ``sdid_stata.json`` from ``_generate_sdid_stata.do`` run on
them with Stata 18.

Precision: ``sdid.ado`` stores ``e(ATT)`` / ``e(se)`` via ``strofreal()``
(about 7 significant digits), so those are checked at rel 1e-6. ``e(tau)``
is full double precision: per-cohort effects and jackknife SEs, and the ATT
rebuilt from them with the eq. (7) weights, are checked at rel 1e-10 (the
observed agreement is 1e-13 or better; the data are stored as float and
read identically on both sides).

Bootstrap and placebo draws cannot match Stata's; they are only screened
(S): finite, seeded-reproducible, and of the order Stata reports on the same
sub-sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures" / "sdid_staggered"
REF = json.loads((FIX / "sdid_stata.json").read_text(encoding="utf-8"))
RTOL_FULL = 1e-10
RTOL_7DIGIT = 1e-6
SINGLE_COHORTS = ["Algeria", "Kenya", "Samoa", "Swaziland", "Tanzania"]


def _prop99():
    df = pd.read_stata(FIX / "prop99_example.dta")
    df["packspercapita"] = df["packspercapita"].astype(float)
    return df


def _quota():
    df = pd.read_stata(FIX / "quota_example.dta")
    for c in ("womparl", "lngdp"):
        df[c] = df[c].astype(float)
    return df


def _att(tau, sizes):
    w = {a: sizes[a][0] * sizes[a][1] for a in sizes}
    return sum(tau[a][0] * w[a] for a in tau) / sum(w.values())


def _fit(df, dataset, **kw):
    y, unit, w = (
        ("packspercapita", "state", "treated")
        if dataset == "prop99"
        else ("womparl", "country", "quota")
    )
    return sp.sdid(df, outcome=y, unit=unit, time="year", treat=w, **kw)


def _cohorts(res):
    return res.model_info["tau_by_cohort"].set_index("adoption")


@pytest.mark.parametrize("method", ["sdid", "did", "sc"])
def test_block_design_matches_stata(method):
    res = _fit(_prop99(), "prop99", method=method, se_method="noinference")
    np.testing.assert_allclose(
        res.estimate, REF[f"prop99_{method}"]["1989"][0], rtol=RTOL_FULL
    )
    np.testing.assert_allclose(
        res.estimate, REF[f"prop99_{method}_eATT"], rtol=RTOL_7DIGIT
    )
    assert res.model_info["design"] == "block"


def test_block_design_equals_the_treated_unit_interface():
    df = _prop99()
    a = _fit(df, "prop99", se_method="noinference")
    b = sp.sdid(
        df,
        outcome="packspercapita",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        se_method="placebo",
        n_reps=2,
        seed=0,
    )
    np.testing.assert_allclose(a.estimate, b.estimate, rtol=1e-12)


def test_staggered_att_and_cohorts_match_stata():
    res = _fit(_quota(), "quota", se_method="noinference")
    tab = _cohorts(res)
    for year, (tau,) in REF["quota_tau"].items():
        np.testing.assert_allclose(tab.loc[int(year), "tau"], tau, rtol=RTOL_FULL)
    for year, (n_tr, t_post) in REF["quota_cohort_size"].items():
        assert tab.loc[int(year), "n_treated"] == n_tr
        assert tab.loc[int(year), "T_post"] == t_post
    np.testing.assert_allclose(
        res.estimate, _att(REF["quota_tau"], REF["quota_cohort_size"]), rtol=RTOL_FULL
    )
    np.testing.assert_allclose(res.estimate, REF["quota_eATT"], rtol=RTOL_7DIGIT)
    assert res.model_info["design"] == "staggered"
    assert np.isnan(res.se)


def test_projected_covariates_match_stata():
    df = _quota().dropna(subset=["lngdp"])
    res = _fit(
        df,
        "quota",
        covariates=["lngdp"],
        covariate_method="projected",
        se_method="noinference",
    )
    np.testing.assert_allclose(
        res.model_info["covariate_beta"]["lngdp"], REF["cov_beta"], rtol=RTOL_FULL
    )
    tab = _cohorts(res)
    for year, (tau,) in REF["cov_tau"].items():
        np.testing.assert_allclose(tab.loc[int(year), "tau"], tau, rtol=RTOL_FULL)
    np.testing.assert_allclose(
        res.estimate, _att(REF["cov_tau"], REF["cov_cohort_size"]), rtol=RTOL_FULL
    )
    np.testing.assert_allclose(res.estimate, REF["cov_eATT"], rtol=RTOL_7DIGIT)


@pytest.mark.parametrize("method", ["sdid", "did", "sc"])
def test_jackknife_matches_stata(method):
    df = _quota()
    df = df[~df["country"].isin(SINGLE_COHORTS)]
    res = _fit(df, "quota", method=method, se_method="jackknife")
    tab = _cohorts(res)
    for year, (tau, se) in REF[f"jk_{method}_tau"].items():
        np.testing.assert_allclose(tab.loc[int(year), "tau"], tau, rtol=RTOL_FULL)
        np.testing.assert_allclose(tab.loc[int(year), "se"], se, rtol=RTOL_FULL)
    np.testing.assert_allclose(
        res.estimate,
        _att(REF[f"jk_{method}_tau"], REF["jk_cohort_size"]),
        rtol=RTOL_FULL,
    )
    np.testing.assert_allclose(res.se, REF[f"jk_{method}_ese"], rtol=RTOL_7DIGIT)


def test_jackknife_reprojects_covariates_on_every_sample():
    df = _quota()
    df = df[~df["country"].isin(SINGLE_COHORTS)].dropna(subset=["lngdp"])
    res = _fit(
        df,
        "quota",
        covariates=["lngdp"],
        covariate_method="projected",
        se_method="jackknife",
    )
    tab = _cohorts(res)
    for year, (tau, se) in REF["jk_cov_tau"].items():
        np.testing.assert_allclose(tab.loc[int(year), "tau"], tau, rtol=RTOL_FULL)
        np.testing.assert_allclose(tab.loc[int(year), "se"], se, rtol=RTOL_FULL)
    np.testing.assert_allclose(res.se, REF["jk_cov_ese"], rtol=RTOL_7DIGIT)


@pytest.mark.parametrize(
    "dataset,se_method",
    [
        ("prop99", "placebo"),
        ("prop99", "bootstrap"),
        ("prop99", "jackknife"),
        ("quota", "placebo"),
        ("quota", "bootstrap"),
        ("quota", "jackknife"),
    ],
)
def test_refusals_follow_stata(dataset, se_method):
    df = _prop99() if dataset == "prop99" else _quota()
    rc = REF[f"{dataset}_rc_{se_method}"]
    if rc == 451:
        with pytest.raises(MethodIncompatibility, match="r\\(451\\)"):
            _fit(df, dataset, se_method=se_method, n_reps=3, seed=0)
    else:
        assert rc == 0
        res = _fit(df, dataset, se_method=se_method, n_reps=3, seed=0)
        assert np.isfinite(res.se)


@pytest.mark.parametrize(
    "se_method,stata_se", [("bootstrap", 4.729109), ("placebo", 2.3404)]
)
def test_resampling_se_screen(se_method, stata_se):
    # S (stochastic screen), not parity: Stata's numbers are one seed(1213)
    # reps(50) run on this sub-sample (recorded with the original fixture).
    df = _quota()
    df = df[~df["country"].isin(SINGLE_COHORTS)]
    a = _fit(df, "quota", se_method=se_method, n_reps=100, seed=1)
    b = _fit(df, "quota", se_method=se_method, n_reps=100, seed=1)
    assert a.se == b.se and np.isfinite(a.se)
    assert 0.5 < a.se / stata_se < 2.0
    assert a.model_info["tau_by_cohort"]["se"].notna().all()


class TestInputs:
    def test_covariates_need_an_explicit_method(self):
        with pytest.raises(MethodIncompatibility, match="covariate_method="):
            _fit(_quota().dropna(subset=["lngdp"]), "quota", covariates=["lngdp"])

    def test_optimized_covariates_refuse_staggered_adoption(self):
        with pytest.raises(MethodIncompatibility, match="single adoption date"):
            _fit(
                _quota().dropna(subset=["lngdp"]),
                "quota",
                covariates=["lngdp"],
                covariate_method="optimized",
                se_method="noinference",
            )

    def test_missing_covariate_is_refused(self):
        with pytest.raises(DataInsufficient, match="lngdp"):
            _fit(_quota(), "quota", covariates=["lngdp"], covariate_method="projected")

    def test_treatment_and_treated_unit_are_exclusive(self):
        with pytest.raises(MethodIncompatibility, match="not both"):
            sp.sdid(
                _prop99(),
                outcome="packspercapita",
                unit="state",
                time="year",
                treat="treated",
                treated_unit="California",
                treatment_time=1989,
            )

    def test_noinference_needs_treatment(self):
        with pytest.raises(MethodIncompatibility, match="treat= only"):
            sp.sdid(
                _prop99(),
                outcome="packspercapita",
                unit="state",
                time="year",
                treated_unit="California",
                treatment_time=1989,
                se_method="noinference",
            )

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (lambda d: d.assign(treated=d["treated"] * 2), "other than 0 and 1"),
            (lambda d: d.iloc[1:], "unbalanced"),
            (
                lambda d: d.assign(
                    treated=np.where(
                        (d["state"] == "California") & (d["year"] == 2000),
                        0,
                        d["treated"],
                    )
                ),
                "absorbing",
            ),
            (
                lambda d: d.assign(
                    treated=np.where(d["state"] == "Texas", 1, d["treated"])
                ),
                "first period",
            ),
            (lambda d: d.assign(treated=0), "all units are controls"),
        ],
    )
    def test_panel_checks(self, mutate, match):
        with pytest.raises((MethodIncompatibility, DataInsufficient), match=match):
            _fit(mutate(_prop99()), "prop99", se_method="noinference")

    def test_no_never_treated_unit(self):
        df = _prop99()
        df["treated"] = np.where(
            (df["state"] != "California") & (df["year"] >= 1995), 1, df["treated"]
        )
        with pytest.raises(MethodIncompatibility, match="never-treated"):
            _fit(df, "prop99", se_method="noinference")
