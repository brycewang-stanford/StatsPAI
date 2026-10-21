"""DiD paper gallery: five published applications, reproduced.

Each class reruns one paper with ``import statspai as sp`` and compares the
result with (a) the numbers the paper (or the authors' replication output)
prints, at the decimals printed, and (b) Stata 18 MP on the same data
bytes, at full precision. The reference numbers live in
``data/did_paper_gallery.json`` with their provenance; nothing here reads
outside the repository. ``docs/guides/did_paper_gallery.md`` shows the
same recipes.

Data
----
* castle-doctrine panel: shipped (``sp.datasets.castle_doctrine``), so
  those tests always run.
* the other four are fetched on demand with a pinned SHA-256
  (``sp.datasets.jel_medicaid / bacon_divorce / lalonde_panel`` and
  ``mpdta(simulated=False)``). Their tests run when the file is already in
  the cache (``$STATSPAI_DATA_DIR``) or when ``STATSPAI_FETCH_DATASETS=1``
  allows the download; otherwise they skip and say why. The nightly CI
  job sets ``STATSPAI_FETCH_DATASETS=1``.

Tolerances
----------
* Paper numbers: half a unit in the last printed decimal (the paper rounds).
* Stata on the same bytes: relative 1e-6 (the Track A default) unless a
  test states a looser bound and why (float storage in a ``.dta`` output,
  a logit stopping rule).

All calls pass every convention explicitly (base period, estimator,
control group, share variance), so a later change of a default cannot
move these numbers.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.datasets import _did_gallery, _fetch

FIXTURE = json.loads(
    (Path(__file__).parent / "data" / "did_paper_gallery.json").read_text(
        encoding="utf-8"
    )
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _load(loader_name, spec, **kwargs):
    """Load a fetched dataset or skip with the reason."""
    if not _fetch.is_cached(spec) and not _fetch.fetch_allowed_in_tests():
        pytest.skip(
            f"{spec.filename} is not cached in {_fetch.cache_dir()} and "
            f"{_fetch.ENV_FETCH}=1 is not set (network fetch disabled)"
        )
    if loader_name == "mpdta":
        return sp.datasets.mpdta(simulated=False)
    return getattr(sp.datasets, loader_name)(**kwargs)


def _paper_ok(value, printed, decimals):
    """``value`` rounds to the printed number (half a unit in the last place)."""
    return abs(value - printed) <= 0.5 * 10.0 ** (-decimals) + 1e-12


def _rel_close(a, b, rtol=1e-6, atol=0.0):
    return abs(a - b) <= atol + rtol * abs(b)


#: The JEL Stata run wrote its numbers with 6 decimals ("%f").
STATA6 = 5e-7 + 1e-12


# ====================================================================== #
#  Sant'Anna & Zhao (2020): DR-DiD on LaLonde, Table 3
# ====================================================================== #

X_LIN = ["age", "educ", "black", "married", "nodegree", "hisp", "re74"]
X_DW = X_LIN + ["u74", "age2", "age3", "educ2", "educ_re74"]
X_ADW = X_DW + ["mar_re74", "mar_u74"]
SPECS = {"lin": X_LIN, "dw": X_DW, "adw": X_ADW}

#: Table 3's six columns -> sp.drdid arguments.
LALONDE_ESTIMATORS = {
    "dr_trad": dict(method="trad"),
    "dr_imp": dict(method="imp"),
    "or_did": dict(est_method="reg"),
    "ipw_abadie": dict(est_method="ipw", normalized=False),
    "ipw_did": dict(est_method="ipw", normalized=True),
    "twfe_cov": dict(est_method="twfe"),
}


@pytest.fixture(scope="module")
def lalonde():
    df = _load("lalonde_panel", _did_gallery.LALONDE_PANEL).copy()
    df["post"] = (df["year"] == 1978).astype(int)
    df["u74"] = (df["re74"] == 0).astype(float)
    df["age2"] = df["age"] ** 2
    df["age3"] = df["age"] ** 3 / 1000
    df["educ2"] = df["educ"] ** 2
    df["educ_re74"] = df["educ"] * df["re74"]
    df["mar_re74"] = df["married"] * df["re74"]
    df["mar_u74"] = df["married"] * df["u74"]
    return df


def _lalonde_sample(df, sample):
    nsw = df["sample"] == 1
    return {
        "lalonde": nsw,
        "dw": nsw & (df["dwincl"] == 1),
        "era": nsw & (df["early_ra"] == 1),
    }[sample]


def _lalonde_fit(df, sample, design, spec, estimator):
    m = _lalonde_sample(df, sample)
    if design == "experiment":
        e = df[m & (df["year"] == 1978)]
        r = sp.regress("re ~ treated", data=e, robust="hc1")
        return float(r.params["treated"]), float(r.std_errors["treated"])
    arm = 0 if design == "evalbias" else 1
    d = df[(m & (df["treated"] == arm)) | (df["sample"] == 2)]
    if estimator == "did_nocov":
        r = sp.drdid(d, y="re", group="experimental", time="post", id="id",
                     covariates=None)
    else:
        r = sp.drdid(d, y="re", group="experimental", time="post", id="id",
                     covariates=SPECS[spec], **LALONDE_ESTIMATORS[estimator])
    return float(r.estimate), float(r.se)


class TestSantAnnaZhao2020:
    """All 54 cells of Table 3, plus the 81 Stata ``drdid`` estimates."""

    ref = FIXTURE["santanna_zhao_2020"]

    @pytest.mark.parametrize("sample", ["lalonde", "dw", "era"])
    def test_table3_cells_round_to_the_paper(self, lalonde, sample):
        cells = self.ref["paper_table3"][sample]
        n = 0
        for spec, by_est in cells.items():
            for est, printed in by_est.items():
                att, se = _lalonde_fit(lalonde, sample, "evalbias", spec, est)
                assert _paper_ok(att, printed["att"], 0), (sample, spec, est, att)
                assert _paper_ok(se, printed["se"], 0), (sample, spec, est, se)
                n += 1
        assert n == 18  # 3 specifications x 6 estimators per sample

    def test_experimental_benchmarks(self, lalonde):
        for sample, printed in self.ref["paper_experimental_benchmark"].items():
            att, se = _lalonde_fit(lalonde, sample, "experiment", "none", "exp_benchmark")
            assert _paper_ok(att, printed["att"], 0) and _paper_ok(se, printed["se"], 0)

    def test_every_stata_drdid_estimate(self, lalonde):
        """81 estimates; Stata wrote them with 4 decimals.

        Bound: $0.05 on the ATT, $0.01 on the SE. The 4-decimal CSV accounts
        for 5e-5; the rest is Stata's logit stopping rule in the IPW and DR
        propensity scores (largest gap observed 0.016 on an ATT of about
        $1,000, i.e. 1.6e-5 relative).
        """
        rows = self.ref["stata"]
        assert len(rows) == 81
        for r in rows:
            att, se = _lalonde_fit(lalonde, r["sample"], r["design"], r["spec"], r["estimator"])
            assert abs(att - r["att"]) <= 0.05, (r, att)
            assert abs(se - r["se"]) <= 0.01, (r, se)


# ====================================================================== #
#  Goodman-Bacon (2021): divorce reform, the 156 2x2s
# ====================================================================== #


@pytest.fixture(scope="module")
def divorce():
    return _load("bacon_divorce", _did_gallery.BACON_DIVORCE)


class TestGoodmanBacon2021:
    ref = FIXTURE["goodman_bacon_2021"]

    def test_twfe_and_its_three_standard_errors(self, divorce):
        st = self.ref["stata_twfe"]
        for vce, kw in [
            ("cluster", {"vcov": {"CRV1": "stfips"}}),
            ("robust", {"vcov": "HC1"}),
            ("unadjusted", {"vcov": "iid"}),
        ]:
            m = sp.feols("asmrs ~ post | stfips + year", divorce, **kw)
            att, se = float(m.params["post"]), float(m.std_errors["post"])
            assert _rel_close(att, st[vce]["att"]) and _rel_close(se, st[vce]["se"]), vce
        assert _paper_ok(att, self.ref["paper"]["twfe"], 2)
        robust = sp.feols("asmrs ~ post | stfips + year", divorce, vcov="HC1")
        assert _paper_ok(float(robust.std_errors["post"]),
                         self.ref["paper"]["twfe_se_robust_fig5"], 2)

    def test_all_156_two_by_twos_match_bacondecomp(self, divorce):
        """Stata stores bacondecomp's 2x2 estimates as float: 1e-4 absolute."""
        bd = sp.bacon_decomposition(divorce, y="asmrs", treat="post", time="year",
                                    id="stfips")
        dec = bd["decomposition"].copy()
        dec["treated"] = dec["treated"].astype(int)
        dec["control_key"] = [
            "Never" if t == "Treated vs Untreated"
            else ("Always" if t == "Later vs Always Treated" else str(int(c)))
            for t, c in zip(dec["type"], dec["control"])
        ]
        stata = pd.DataFrame(self.ref["stata_bacon_2x2"])
        stata["control_key"] = stata["control"].astype(str)
        m = dec.merge(stata, on=["treated", "control_key"], suffixes=("", "_st"))
        assert len(dec) == len(stata) == len(m) == self.ref["paper"]["n_2x2"] == 156
        assert (m["type"] == m["type_st"]).all()
        np.testing.assert_allclose(m["estimate"], m["estimate_st"], atol=1e-4, rtol=0)
        np.testing.assert_allclose(m["weight"], m["weight_st"], atol=1e-6, rtol=0)
        # The weighted sum of the 2x2s is the TWFE coefficient (the theorem).
        assert abs(float((dec["weight"] * dec["estimate"]).sum()) - bd["beta_twfe"]) < 1e-8

    def test_summary_by_comparison_type(self, divorce):
        bd = sp.bacon_decomposition(divorce, y="asmrs", treat="post", time="year",
                                    id="stfips")
        summ = bd["summary"].set_index("type")
        for typ, st in self.ref["stata_bacon_summary"].items():
            assert abs(summ.loc[typ, "weight"] - st["weight"]) < 1e-6, typ
            assert abs(summ.loc[typ, "estimate"] - st["estimate"]) < 1e-5, typ
            assert int(summ.loc[typ, "n_comparisons"]) == st["n_comparisons"], typ
            printed = self.ref["paper"]["summary"][typ]
            assert _paper_ok(summ.loc[typ, "weight"], printed["weight"], 2), typ
            assert _paper_ok(summ.loc[typ, "estimate"], printed["estimate"], 2), typ

    def test_callaway_santanna_without_the_always_treated(self, divorce):
        d41 = divorce[divorce["first_treat"] != 1964]
        tw = sp.feols("asmrs ~ post | stfips + year", d41, vcov={"CRV1": "stfips"})
        st = self.ref["stata_twfe_41_states"]
        assert _rel_close(float(tw.params["post"]), st["att"])
        assert _rel_close(float(tw.std_errors["post"]), st["se"])
        cs = sp.callaway_santanna(d41, y="asmrs", g="first_treat", t="year", i="stfips",
                                  estimator="dr", control_group="nevertreated",
                                  base_period="universal")
        simple = sp.aggte(cs, type="simple")
        st = self.ref["stata_cs_simple_41_states"]
        assert _rel_close(float(simple.estimate), st["att"])
        assert _rel_close(float(simple.se), st["se"])


# ====================================================================== #
#  Callaway & Sant'Anna (2021): mpdta
# ====================================================================== #


@pytest.fixture(scope="module")
def mpdta_cs():
    df = _load("mpdta", _did_gallery.MPDTA_ORIGINAL)
    return sp.callaway_santanna(df, y="lemp", g="first_treat", t="year", i="countyreal",
                                estimator="dr", control_group="nevertreated",
                                base_period="varying")


class TestCallawaySantAnna2021:
    ref = FIXTURE["callaway_santanna_2021"]

    def test_twelve_group_time_effects(self, mpdta_cs):
        gt = mpdta_cs.detail.set_index(["group", "time"])
        for r in self.ref["r_attgt"]:
            att = float(gt.loc[(r["group"], r["time"]), "att"])
            assert _paper_ok(att, r["att"], 4), r
        for r in self.ref["stata_attgt"]:
            row = gt.loc[(r["group"], r["time"])]
            assert _rel_close(float(row["att"]), r["att"], atol=1e-12), r
            assert _rel_close(float(row["se"]), r["se"]), r

    @pytest.mark.parametrize("agg", ["simple", "group", "calendar", "dynamic"])
    def test_aggregations(self, mpdta_cs, agg):
        a = sp.aggte(mpdta_cs, type=agg)
        printed = self.ref["r_aggregations"][agg]
        decimals = self.ref["r_simple_printed_decimals"] if agg == "simple" else 4
        if agg == "group":
            decimals = 3  # ?aggte prints the group average as -0.031
        assert _paper_ok(float(a.estimate), printed, decimals), (agg, a.estimate)
        st = self.ref["stata_aggregations"][agg]
        assert _rel_close(float(a.estimate), st["att"]), agg
        if agg == "group":
            # csdid's GAverage holds the cohort shares fixed; R did (and the
            # StatsPAI default) adds their sampling variance.
            fixed = sp.aggte(mpdta_cs, type="group", share_variance=False)
            assert _rel_close(float(fixed.se), st["se"])
            assert float(a.se) > float(fixed.se)
        else:
            assert _rel_close(float(a.se), st["se"]), agg

    @pytest.mark.parametrize("agg", ["simple", "group", "calendar", "dynamic"])
    def test_aggregations_equal_r_did_analytic(self, mpdta_cs, agg):
        """R did 2.5.1 with bstrap=FALSE, printed to 10 significant digits.

        The group SE includes the cohort-share variance on both sides
        (share_variance=True is the StatsPAI default, as in R).
        """
        a = sp.aggte(mpdta_cs, type=agg)
        r = self.ref["r_analytic"][agg]
        assert _rel_close(float(a.estimate), r["att"], rtol=1e-8), agg
        assert _rel_close(float(a.se), r["se"], rtol=1e-8), agg

    def test_aggregation_elements(self, mpdta_cs):
        for agg, key, col in [("group", "group_elements", "group"),
                              ("calendar", "calendar_elements", "time"),
                              ("dynamic", "dynamic_elements", "relative_time")]:
            det = sp.aggte(mpdta_cs, type=agg).detail.set_index(col)
            for k, printed in self.ref["r_aggregations"][key].items():
                assert _paper_ok(float(det.loc[int(k), "att"]), printed, 4), (agg, k)

    def test_pretest_wald_and_chi2_p(self, mpdta_cs):
        pt = mpdta_cs.model_info["pretrend_test"]
        st = self.ref["stata_pretest"]
        assert int(pt["df"]) == st["df"]
        assert _rel_close(float(pt["statistic"]), st["chi2"])
        p = float(stats.chi2.sf(pt["statistic"], pt["df"]))
        assert _rel_close(p, st["p"], rtol=1e-6)
        assert _paper_ok(p, self.ref["r_pretest_p"], 5)


# ====================================================================== #
#  Cheng & Hoekstra (2013): castle doctrine (shipped data)
# ====================================================================== #


@pytest.fixture(scope="module")
def castle():
    d = sp.datasets.castle_doctrine().copy()
    d["region"] = (d.northeast + 2 * d.midwest + 3 * d.south + 4 * d.west).astype(int)
    d["gvar"] = d["effyear"].fillna(0).astype(int)
    d["D"] = ((d["year"] >= d["effyear"]) & d["effyear"].notna()).astype(int)
    return d


class TestChengHoekstra2013:
    ref = FIXTURE["cheng_hoekstra_2013"]

    def test_twfe_specifications(self, castle):
        """Six weighted TWFE specifications.

        Bound 1e-6 relative: the shipped CSV holds ``popwt`` in float64
        decimals while the Stata run read the Mixtape ``castle.dta`` float32
        values (largest relative difference 5.7e-8), which moves the
        coefficients by about 1e-8 relative.
        """
        for row in self.ref["stata_twfe"]:
            m = sp.feols(f"l_homicide ~ {row['x']} | {row['fe']}", castle,
                         vcov={"CRV1": "sid"}, weights="popwt")
            assert _rel_close(float(m.params[row["x"]]), row["att"]), row
            assert _rel_close(float(m.std_errors[row["x"]]), row["se"]), row

    def test_table5_columns_1_and_2(self, castle):
        for key, fe in [("col1_state_year_fe", "sid + year"),
                        ("col2_region_by_year_fe", "sid + region^year")]:
            m = sp.feols(f"l_homicide ~ cdl | {fe}", castle, vcov={"CRV1": "sid"},
                         weights="popwt")
            printed = self.ref["paper"][key]
            assert _paper_ok(float(m.params["cdl"]), printed["att"], 4), key
            assert _paper_ok(float(m.std_errors["cdl"]), printed["se"], 4), key

    def test_callaway_santanna_overall(self, castle):
        cs = sp.callaway_santanna(castle, y="l_homicide", g="gvar", t="year", i="sid",
                                  weights="popwt", estimator="reg",
                                  control_group="nevertreated", base_period="universal")
        simple = sp.aggte(cs, type="simple")
        st = self.ref["stata_cs_simple"]
        assert _rel_close(float(simple.estimate), st["att"], rtol=1e-6)
        assert _rel_close(float(simple.se), st["se"], rtol=1e-6)
        post = sp.aggte(cs, type="dynamic", min_e=0, max_e=4)
        st = self.ref["stata_cs_post_average"]
        assert _rel_close(float(post.estimate), st["att"], rtol=1e-6)
        assert _rel_close(float(post.se), st["se"], rtol=1e-6)


# ====================================================================== #
#  Baker, Callaway, Cunningham, Goodman-Bacon & Sant'Anna (JEL)
# ====================================================================== #

JEL_COVS = list(_did_gallery.JEL_COVARIATES)


@pytest.fixture(scope="module")
def jel_2x2():
    d = _load("jel_medicaid", _did_gallery.JEL_MEDICAID, sample="2x2")
    return d[d["year"].isin([2013, 2014])].copy()


@pytest.fixture(scope="module")
def jel_staggered():
    d = _load("jel_medicaid", _did_gallery.JEL_MEDICAID, sample="staggered").copy()
    d["D"] = ((d["treat_year"] > 0) & (d["year"] >= d["treat_year"])).astype(int)
    return d


def _baseline_covariates(s):
    """Covariates frozen at their 2013 values (JEL Section 4.2)."""
    s = s.copy()
    x13 = s[s["year"] == 2013].set_index("county_code")[JEL_COVS]
    for c in JEL_COVS:
        s[c] = s["county_code"].map(x13[c])
    return s


class TestBakerEtAlJEL:
    ref = FIXTURE["baker_et_al_jel"]

    def test_two_by_two(self, jel_2x2):
        """``y ~ Treat * Post`` clustered by county, as in the paper's Table 5.

        The pooled interaction regression, not county fixed effects: the two
        give the same coefficient but different small-sample factors in the
        clustered SE (3.7484 against 3.7480).
        """
        s = jel_2x2.assign(
            Treat=(jel_2x2.treat_year == 2014).astype(int),
            Post=(jel_2x2.year == 2014).astype(int),
        )
        s["D"] = s["Treat"] * s["Post"]
        for key, w, col in [("did_2x2_unweighted", None, "col1_unweighted_no_covs"),
                            ("did_2x2_weighted", "set_wt", "col4_weighted_no_covs")]:
            m = sp.feols("crude_rate_20_64 ~ D + Treat + Post", s,
                         vcov={"CRV1": "county_code"}, weights=w)
            att, se = float(m.params["D"]), float(m.std_errors["D"])
            assert abs(att - self.ref["stata"][key]["att"]) <= STATA6, key
            assert abs(se - self.ref["stata"][key]["se"]) <= STATA6, key
            printed = self.ref["paper"]["table5"][col]
            assert _paper_ok(att, printed["att"], 2), key
            assert _paper_ok(se, printed["se"], 2), key
            table2 = self.ref["paper"]["table2_did"]["weighted" if w else "unweighted"]
            assert _paper_ok(att, table2, 1), key

    @pytest.mark.parametrize("est", ["reg", "ipw", "dr"])
    def test_table7_with_baseline_covariates(self, jel_2x2, est):
        s = _baseline_covariates(jel_2x2)
        common = dict(y="crude_rate_20_64", g="treat_year", t="year", i="county_code",
                      x=JEL_COVS, estimator=est, control_group="nevertreated",
                      base_period="varying")
        w = sp.callaway_santanna(s, weights="set_wt", **common)
        st = self.ref["stata"][est]
        # Bound 1e-5: the do-file builds the covariates with ``gen``, which
        # stores them as float32, so Stata fits on covariates rounded at about
        # 1e-7 relative (see test_twfe_with_time_varying_covariates).
        assert abs(float(w.estimate) - st["att"]) < 1e-5, est
        assert abs(float(w.se) - st["se"]) < 1e-5, est
        assert _paper_ok(float(w.estimate), self.ref["paper"]["table7"]["weighted"][est], 2)
        u = sp.callaway_santanna(s, **common)
        assert _paper_ok(float(u.estimate), self.ref["paper"]["table7"]["unweighted"][est], 2)

    def test_twfe_with_time_varying_covariates(self, jel_2x2):
        s = jel_2x2.assign(D=((jel_2x2.treat_year == 2014) & (jel_2x2.year == 2014)).astype(int))
        f = "crude_rate_20_64 ~ D + " + " + ".join(JEL_COVS) + " | county_code + year"
        w = sp.feols(f, s, weights="set_wt", vcov={"CRV1": "county_code"})
        st = self.ref["stata"]["twfe_xt_weighted"]
        # Bound 1e-5 relative, not 1e-6: Stata's ``gen`` stored the six
        # covariates as float32, which moves this coefficient by 3e-6 relative
        # (-1.369975 in Stata, -1.369979 on the float64 covariates).
        assert _rel_close(float(w.params["D"]), st["att"], rtol=1e-5)
        assert _rel_close(float(w.std_errors["D"]), st["se"], rtol=1e-5)
        # Point estimates only against the paper: its Table 5 SE for column 6
        # (1.62) uses R's own small-sample factor; Stata reghdfe and StatsPAI
        # both give 1.6149.
        printed = self.ref["paper"]["table5"]
        assert _paper_ok(float(w.params["D"]), printed["col6_weighted_xt"]["att"], 2)
        u = sp.feols(f, s, vcov={"CRV1": "county_code"})
        assert _paper_ok(float(u.params["D"]), printed["col3_unweighted_xt"]["att"], 2)

    def test_staggered_twfe_and_bacon(self, jel_staggered):
        d = jel_staggered
        st = self.ref["stata"]
        for key, w in [("staggered_twfe_weighted", "set_wt"),
                       ("staggered_twfe_unweighted", None)]:
            m = sp.feols("crude_rate_20_64 ~ D | county_code + year", d, weights=w,
                         vcov={"CRV1": "county_code"})
            assert abs(float(m.params["D"]) - st[key]["att"]) <= STATA6, key
            assert abs(float(m.std_errors["D"]) - st[key]["se"]) <= STATA6, key
        bd = sp.bacon_decomposition(d, y="crude_rate_20_64", treat="D", time="year",
                                    id="county_code")
        summ = bd["summary"].set_index("type")
        for typ, ref in st["bacon"].items():
            assert abs(summ.loc[typ, "weight"] - ref["weight"]) <= STATA6, typ
            assert abs(summ.loc[typ, "estimate"] - ref["estimate"]) <= STATA6, typ

    def test_staggered_callaway_santanna_unweighted(self, jel_staggered):
        cs = sp.callaway_santanna(jel_staggered, y="crude_rate_20_64", g="treat_year",
                                  t="year", i="county_code", estimator="reg",
                                  control_group="notyettreated", base_period="universal")
        simple = sp.aggte(cs, type="simple")
        st = self.ref["stata"]["staggered_cs_simple_unweighted"]
        assert abs(float(simple.estimate) - st["att"]) <= STATA6
        assert abs(float(simple.se) - st["se"]) <= STATA6


def test_fixture_provenance_is_recorded():
    for key in ("santanna_zhao_2020", "goodman_bacon_2021", "callaway_santanna_2021",
                "cheng_hoekstra_2013", "baker_et_al_jel"):
        block = FIXTURE[key]
        assert block.get("stata_source"), key
        assert block.get("paper_source") or block.get("r_source"), key


def test_castle_is_the_shipped_real_extract():
    """The castle panel used here is the shipped real extract."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = sp.datasets.castle_doctrine()
    assert d.attrs["data_source"] == "real" and len(d) == 550
