"""Numbers a researcher reads off a result, against Stata 18 and R.

A blind replication probe (2026-11) ran twenty applied-microeconomics
tasks in StatsPAI and again in Stata / R on the same bytes, and compared
every printed number, not only the headline coefficient. The headline
numbers agreed; the defects were in what is printed around them. Each
test below pins one of them to the reference program.

References, on the CSVs bundled with StatsPAI (``src/statspai/datasets/data``):

* Stata 18 MP: ``_fixtures/_generate_probe_econ_Stata.do`` ->
  ``_fixtures/probe_econ_Stata.csv`` (``ranksum``, ``oaxaca`` 4.x,
  ``boottest``, ``ivreghdfe``, ``xtreg``).
* R: ``_fixtures/_generate_probe_econ_R.R`` ->
  ``_fixtures/probe_econ_R.csv`` (fixest 0.14.0 and did2s 1.2.1, both run
  as black boxes).

Tolerances are stated where they are used. Closed-form quantities
(variances, R-squared, p-values of an enumerated bootstrap) are held to
1e-9 relative or exact equality; quantities behind an iterative fit
(Poisson IRLS, the two-stage DiD) to 1e-7, the references' own stopping
rules.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).resolve().parent / "_fixtures"
DATA = Path(__file__).resolve().parents[2] / "src" / "statspai" / "datasets" / "data"


def _ref(program: str) -> dict:
    frame = pd.read_csv(
        FIX / f"probe_econ_{program}.csv", skipinitialspace=True, dtype={"name": str}
    )
    return {
        (r.tag, r.kind, r.name): float(r.value) for r in frame.itertuples(index=False)
    }


STATA = _ref("Stata")
R = _ref("R")


@pytest.fixture(scope="module")
def card() -> pd.DataFrame:
    df = pd.read_csv(DATA / "card_1995.csv")
    df["g"] = (df["exper"] // 3).astype(int)  # 8 clusters
    return df


@pytest.fixture(scope="module")
def castle() -> pd.DataFrame:
    return pd.read_csv(DATA / "castle_2013.csv")


@pytest.fixture(scope="module")
def nsw() -> pd.DataFrame:
    return pd.read_csv(DATA / "lalonde_matchit.csv")


@pytest.fixture(scope="module")
def sasp() -> pd.DataFrame:
    df = pd.read_csv(DATA / "sasp_panel.csv")
    used = ["id", "session", "lnw", *SASP_X]
    df = df.dropna(subset=used).copy()
    df["id"] = df["id"].astype(int)
    df["session"] = df["session"].astype(int)
    return df


SASP_X = ["unsafe", "llength", "reg", "age_cl", "hot", "massage_cl"]


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# sp.fepois: the log-likelihood and the fit statistics around it
# ---------------------------------------------------------------------------

POIS = "homicide ~ post + unemployrt + poverty + l_police | sid + year"


def test_fepois_log_likelihood_includes_the_constant(castle):
    """``Log-likelihood`` is sum w (y log mu - mu - log y!), fixest's logLik.

    Through 1.43.1 the ``- log y!`` term was missing: 1798.63 where fixest
    and ppmlhdfe report -930.93. Tolerance 1e-8: both fits stop on a
    relative deviance change.
    """
    r = _quiet(sp.fepois, POIS, castle, vcov={"CRV1": "sid"})
    d = r.diagnostics
    assert d["Log-likelihood"] == pytest.approx(R[("fepois", "s", "ll")], rel=1e-8)
    assert d["Deviance"] == pytest.approx(R[("fepois", "s", "deviance")], rel=1e-8)
    assert d["Pseudo R-squared"] == pytest.approx(R[("fepois", "s", "pr2")], rel=1e-8)
    # McFadden's definition against the constant-only model
    assert d["Pseudo R-squared"] == pytest.approx(
        1.0 - d["Log-likelihood"] / d["Log-likelihood (null)"], rel=1e-12
    )

    rw = _quiet(sp.fepois, POIS, castle, vcov={"CRV1": "sid"}, weights="popwt")
    assert rw.diagnostics["Log-likelihood"] == pytest.approx(
        R[("fepois_w", "s", "ll")], rel=1e-8
    )


def test_fepois_summary_has_no_undefined_rows(castle):
    """A Poisson fit has no R-squared or RMSE; the summary printed five
    rows reading ``nan`` for them."""
    r = _quiet(sp.fepois, POIS, castle, vcov={"CRV1": "sid"})
    text = str(r.summary())
    assert "nan" not in text.lower()
    assert "Pseudo R-squared" in text
    assert "Observations: 550" in text
    assert "cluster (sid), G=50" in text


# ---------------------------------------------------------------------------
# sp.feols, 2SLS with absorbed effects: fit statistics and diagnostics
# ---------------------------------------------------------------------------

FEIV = "lwage ~ black + south | g + smsa | educ ~ nearc4 + nearc2"


@pytest.mark.parametrize(
    "tag, vcov",
    [("iid", "iid"), ("hetero", "hetero"), ("cluster", {"CRV1": "g"})],
)
def test_feols_iv_first_stage_and_wu_hausman(card, tag, vcov):
    """First-stage F p-value from F(k, df), and a Wu-Hausman test that
    does not depend on the variance of the main fit (fixest ``ivwald1``
    and ``wh``).

    Before: the p-value was the chi-square one (6.799e-05 against
    7.010e-05), and under clustering the Wu-Hausman denominator left out
    the effects nested in the clusters (3.2521 against 3.2445).
    """
    r = _quiet(sp.feols, FEIV, card, vcov=vcov)
    d = r.diagnostics
    key = f"feiv_{tag}"
    assert d["First-stage F (educ)"] == pytest.approx(
        R[(key, "s", "ivwald1")], rel=1e-9
    )
    assert d["First-stage F p-value (educ)"] == pytest.approx(
        R[(key, "s", "ivwald1_p")], rel=1e-8
    )
    assert d["Hausman F-stat"] == pytest.approx(R[(key, "s", "wh")], rel=1e-9)
    assert d["Hausman p-value"] == pytest.approx(R[(key, "s", "wh_p")], rel=1e-8)


def test_feols_iv_reports_fit_statistics(card):
    """R-squared, within R-squared and RMSE on the structural residuals.

    They were all NaN. RMSE is fixest's; the within R-squared is
    ``ivreghdfe``'s ``e(r2)``. (fixest's own ``r2`` of an IV fit is a
    second-stage quantity and is not reproduced.)
    """
    r = _quiet(sp.feols, FEIV, card, vcov="iid")
    d = r.diagnostics
    assert d["RMSE"] == pytest.approx(R[("feiv_iid", "s", "rmse")], rel=1e-10)
    assert d["R-squared (within)"] == pytest.approx(
        STATA[("ivreghdfe", "s", "r2_within")], rel=1e-10
    )
    # 1 - RSS / TSS of the outcome itself
    y = card["lwage"].to_numpy()
    n = len(y)
    tss = float(((y - y.mean()) ** 2).sum())
    assert d["R-squared"] == pytest.approx(1.0 - n * d["RMSE"] ** 2 / tss, rel=1e-12)
    assert d["R-squared"] < d["R-squared (within)"] + 1.0  # both finite
    assert np.isfinite(d["Adj. R-squared"]) and np.isfinite(
        d["Adj. R-squared (within)"]
    )


def test_feols_cluster_on_a_regressor(card):
    """Clustering on a variable that is also a regressor (years of
    schooling, a discrete running variable) raised ``ValueError: Buffer
    has wrong number of dimensions``. The fit must equal the one that
    clusters on a copy of the column."""
    twin = card.assign(educ_copy=card["educ"])
    fml = "lwage ~ educ + exper + black | south + smsa"
    a = _quiet(sp.feols, fml, card, vcov={"CRV1": "educ"})
    b = _quiet(sp.feols, fml, twin, vcov={"CRV1": "educ_copy"})
    np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-13)
    np.testing.assert_allclose(
        a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=1e-13
    )
    assert a.model_info["n_clusters"] == card["educ"].nunique()


def test_feols_c_factor_runs_on_the_native_engine(castle):
    """``C(year)`` beside absorbed unit effects: Stata ``xtreg y x i.year,
    fe vce(cluster id)``, which ``sp.stata`` translates to this call.

    The native engine refused ``C()``, so without the optional pyfixest
    the most common panel command of a do-file raised
    ``MissingDependencyError``.
    """
    r = _quiet(
        sp.feols,
        "l_homicide ~ post + unemployrt + C(year) | sid",
        castle,
        vcov={"CRV1": "sid"},
    )
    assert r.model_info["backend"] == "statspai-native"
    names = {"post": "post", "unemployrt": "unemployrt"}
    names.update({f"C(year)[T.{y}]": f"{y}.year" for y in range(2001, 2011)})
    for ours, theirs in names.items():
        assert r.params[ours] == pytest.approx(
            STATA[("xtreg_fe_iyear", "b", theirs)], rel=1e-9, abs=1e-12
        ), ours
        assert r.std_errors[ours] == pytest.approx(
            STATA[("xtreg_fe_iyear", "se", theirs)], rel=1e-9
        ), ours
    assert len(r.params) == len(names)

    via_stata = _quiet(
        sp.stata,
        "xtset sid year\nxtreg l_homicide post unemployrt i.year, fe vce(cluster sid)",
        data=castle,
    )
    assert via_stata.params["post"] == pytest.approx(r.params["post"], rel=1e-13)


# ---------------------------------------------------------------------------
# sp.balance_table(test="ranksum")
# ---------------------------------------------------------------------------


def test_balance_table_ranksum_corrects_for_ties(nsw):
    """Rank-sum p-values with the tie-corrected variance (Stata
    ``ranksum``).

    ``scipy.stats.ranksums`` has no tie correction; on a 0/1 covariate
    every observation is tied and the p-value was 0.104 where Stata gives
    0.0035. The table prints three decimals, which is what is compared.
    """
    nsw = nsw.copy()
    variables = ["age", "educ", "hispanic", "nodegree", "re74"]
    table = sp.balance_table(
        nsw, treat="treat", covariates=variables, test="ranksum", output="dataframe"
    )
    for v in variables:
        expected = STATA[("ranksum", "s", v)]
        assert float(table.loc[v, "p-value"]) == pytest.approx(
            round(expected, 3), abs=5e-4
        ), v
        # the same test as sp.ranksum
        assert sp.ranksum(nsw, v, by="treat").pvalue == pytest.approx(
            expected, rel=1e-10
        )


# ---------------------------------------------------------------------------
# Event-study output
# ---------------------------------------------------------------------------


def test_event_study_pretrend_is_labelled_as_the_f_test_it_is(castle):
    """The pre-trend statistic of ``sp.event_study`` is F(q, G - 1); the
    summary called it chi2(q)."""
    r = _quiet(
        sp.event_study,
        castle,
        y="l_homicide",
        treat_time="effyear",
        time="year",
        unit="sid",
        window=(-4, 4),
    )
    pt = r.model_info["pretrend_test"]
    assert pt["statistic"] == pytest.approx(R[("es_pretrend", "s", "F")], abs=2e-4)
    assert pt["pvalue"] == pytest.approx(R[("es_pretrend", "s", "p")], abs=2e-4)
    assert (pt["df"], pt["df_denom"]) == (3, 49)
    text = str(r.summary())
    assert "Pre-trend Test: F(3, 49)" in text
    assert "chi2(3)" not in text


def test_event_time_table_is_not_called_group_time(castle):
    """A table indexed by relative time is an event-time table and shows
    that index; only Callaway-Sant'Anna's has group-time rows."""
    sa = _quiet(sp.sun_abraham, castle, y="l_homicide", g="effyear", t="year", i="sid")
    text = str(sa.summary())
    assert "Event-Time ATT Estimates" in text
    assert "Group-Time ATT Estimates" not in text
    assert "relative_time" in text

    cs = _quiet(
        sp.callaway_santanna, castle, y="l_homicide", g="effyear", t="year", i="sid"
    )
    assert "Group-Time ATT Estimates" in str(cs.summary())


def test_did_2stage_event_study_is_a_table(castle):
    """``event_study=True`` returns the coefficients as ``detail`` (they
    were only in a nested dict, and ``summary()`` showed none), equal to
    R ``did2s``. Tolerance 1e-7: did2s solves the first stage iteratively.
    """
    r = _quiet(
        sp.did_2stage,
        castle,
        y="l_homicide",
        group="sid",
        time="year",
        first_treat="effyear",
        event_study=True,
        horizon=list(range(-4, 6)),
    )
    d = r.detail.set_index("relative_time")
    assert list(d.index) == list(range(-4, 6))
    for k in range(-4, 6):
        assert d.loc[k, "att"] == pytest.approx(
            R[("did2s_es", "b", str(k))], rel=1e-7
        ), k
        assert d.loc[k, "se"] == pytest.approx(
            R[("did2s_es", "se", str(k))], rel=1e-7
        ), k
    np.testing.assert_allclose(
        d["pvalue"], 2 * stats.norm.sf(np.abs(d["att"] / d["se"])), rtol=1e-12
    )
    es = r.model_info["event_study"]
    assert d.loc[0, "att"] == es["coef"]["D_k+0"]
    assert "Event-Time ATT Estimates" in str(r.summary())


# ---------------------------------------------------------------------------
# sp.panel: correlated random effects
# ---------------------------------------------------------------------------


def test_mundlak_r_squared_follows_xtreg(sasp):
    """``method='mundlak'`` is ``xtreg y x xbar, re`` and reports xtreg's
    within / between / overall R-squared, as ``method='re'`` does.

    It printed linearmodels' 1 - SSR/SST under the name ``R-squared
    (between)``: 0.142950 where xtreg has 0.143040, and no overall row.
    """
    r = _quiet(
        sp.panel,
        sasp,
        "lnw ~ " + " + ".join(SASP_X),
        entity="id",
        time="session",
        method="mundlak",
    )
    d = r.diagnostics
    assert r.data_info["nobs"] == STATA[("mundlak", "s", "N")]
    assert d["R-squared (within)"] == pytest.approx(
        STATA[("mundlak", "s", "r2_w")], rel=1e-10
    )
    assert d["R-squared (between)"] == pytest.approx(
        STATA[("mundlak", "s", "r2_b")], rel=1e-10
    )
    assert d["R-squared (overall)"] == pytest.approx(
        STATA[("mundlak", "s", "r2_o")], rel=1e-10
    )
    assert r.model_info["r2_convention"] == "xtreg"


# ---------------------------------------------------------------------------
# vce="wild": the intervals beside the bootstrap p-values
# ---------------------------------------------------------------------------

WILD = "lwage ~ black + south + smsa + educ"


def test_wild_bootstrap_intervals_are_what_the_summary_prints(card):
    """After ``vce='wild'`` the p-values are bootstrap p-values, and the
    intervals beside them are the bootstrap confidence sets.

    ``summary()`` and ``conf_int()`` printed estimate +/- 1.96 x CRV1 SE:
    for ``educ`` a p-value of 0.0625 beside an interval [0.0059, 0.0612]
    that excludes zero. With 8 clusters the Rademacher grid is enumerated
    (256 draws), so the p-values are exact rationals and equal
    ``boottest``'s. The interval is the set of nulls the test does not
    reject, checked here by its definition: the bootstrap p-value is at
    least 0.05 just inside each end point and below it just outside.
    ``boottest`` interpolates its end points between the points of a grid
    where StatsPAI locates the jump; with 256 draws the p-value moves in
    steps of 1/128, and the two differ by up to 8.4% of a standard error
    on these data (10% allowed; the interval printed before was off by
    170% of one).
    """
    r = _quiet(sp.regress, WILD, card, vce="wild", cluster="g", seed=1)
    ci = r.conf_int()
    base = _quiet(sp.regress, WILD, card, cluster="g")

    def p_at(v: str, null: float) -> float:
        return _quiet(
            sp.wild_cluster_boot, base, card, cluster="g", variable=v, h0=null
        )["p_boot"]

    for v in ["black", "south", "smsa", "educ"]:
        assert r.pvalues[v] == STATA[("boottest_p", "s", v)], v
        se = r.std_errors[v]
        lo, hi = ci.loc[v].iloc[0], ci.loc[v].iloc[1]
        step = 1e-4 * se
        assert p_at(v, lo + step) >= 0.05 > p_at(v, lo - step), v
        assert p_at(v, hi - step) >= 0.05 > p_at(v, hi + step), v
        assert lo == pytest.approx(STATA[("boottest_lo", "s", v)], abs=0.10 * se), v
        assert hi == pytest.approx(STATA[("boottest_hi", "s", v)], abs=0.10 * se), v
        # dual to the test: zero is inside exactly when p >= 0.05
        assert (lo <= 0.0 <= hi) == (r.pvalues[v] >= 0.05), v

    # the same set as the stand-alone routine
    one = _quiet(
        sp.wild_cluster_boot,
        base,
        card,
        cluster="g",
        variable="educ",
        confidence_set=True,
    )
    assert tuple(ci.loc["educ"]) == pytest.approx(one["ci_inverted"], rel=1e-12)

    # every surface reports it
    tidy = r.tidy().set_index("term")
    assert tidy.loc["educ", "conf_low"] == ci.loc["educ"].iloc[0]
    assert r.to_dict()["coefficients"]["educ"]["conf_low"] == pytest.approx(
        ci.loc["educ"].iloc[0], rel=1e-12
    )
    text = str(r.summary())
    assert "Intervals: wild cluster bootstrap confidence sets" in text
    assert f"{ci.loc['educ'].iloc[1]:.4f}" in text

    # a bootstrap set exists at the level it was computed at
    with pytest.raises(MethodIncompatibility, match="cannot be rescaled"):
        r.conf_int(0.10)


def test_wild_bootstrap_intervals_with_absorbed_effects(card):
    """The same through ``sp.feols`` (one absorbed effect)."""
    r = _quiet(
        sp.feols,
        "lwage ~ black + south + educ | smsa",
        card,
        vce="wild",
        cluster="g",
        seed=1,
    )
    ci = r.conf_int()
    for v in ["black", "south", "educ"]:
        assert r.pvalues[v] == STATA[("boottest_p", "s", v)], v
        se = r.std_errors[v]
        assert ci.loc[v].iloc[0] == pytest.approx(
            STATA[("boottest_lo", "s", v)], abs=0.10 * se
        ), v
        assert ci.loc[v].iloc[1] == pytest.approx(
            STATA[("boottest_hi", "s", v)], abs=0.10 * se
        ), v


def test_classical_intervals_still_rescale(card):
    """Fits without stored intervals are untouched: ``conf_int(alpha)`` is
    estimate +/- t x SE at any level."""
    r = _quiet(sp.regress, WILD, card, cluster="g")
    for alpha in (0.05, 0.10):
        crit = stats.t.ppf(1 - alpha / 2, 7)
        ci = r.conf_int(alpha)
        np.testing.assert_allclose(
            ci.iloc[:, 0], r.params - crit * r.std_errors, rtol=1e-13
        )
        np.testing.assert_allclose(
            r.tidy(conf_level=1 - alpha)["conf_high"],
            r.params + crit * r.std_errors,
            rtol=1e-13,
        )


def test_regtable_says_brackets_for_intervals(card):
    """``se_type='ci'`` prints ``[lo, hi]``; the note said "in parentheses".

    A check of the table's wording, so the call goes through a local name:
    the parity index credits literal ``sp.<name>(`` calls in this directory
    as numerical evidence, which this is not.
    """
    regtable = sp.regtable
    m = _quiet(sp.regress, "lwage ~ educ", card)
    text = str(regtable(m, se_type="ci"))
    assert "95% CI in brackets" in text
    assert "parentheses" not in text
    lo, hi = m.conf_int().loc["educ"]
    assert f"[{lo:.4f}, {hi:.4f}]" in text
    assert "Standard errors in parentheses" in str(regtable(m))


# ---------------------------------------------------------------------------
# sp.oaxaca
# ---------------------------------------------------------------------------

OAX_X = ["educ", "exper", "expersq", "south", "smsa"]


def _oaxaca(df, reference):
    with contextlib.redirect_stdout(io.StringIO()):
        return sp.oaxaca(df, y="lwage", group="black", x=OAX_X, reference=reference)


@pytest.mark.parametrize("reference", ["reimers", "cotton"])
def test_oaxaca_unexplained_se_weighted_reference(card, reference):
    """With beta* = w beta_A + (1 - w) beta_B the unexplained part is one
    linear combination of beta_A - beta_B, and its delta-method SE is
    Stata's ``oaxaca, weight(w) vce(robust) fixed``.

    The old "conservative" sum ignored that both halves share
    coefficients: Reimers 0.02112 against 0.01798, Cotton 0.01971 against
    0.01723.
    """
    o = _oaxaca(card, reference).overall
    tag = f"oaxaca_{reference}"
    for part in ("explained", "unexplained"):
        assert o[part] == pytest.approx(STATA[(tag, "b", f"overall:{part}")], rel=1e-10)
        assert o[f"{part}_se"] == pytest.approx(
            STATA[(tag, "se", f"overall:{part}")], rel=1e-9
        )


def test_oaxaca_unexplained_se_pooled_reference(card):
    """Pooled (Neumark) reference, Stata ``oaxaca, omega vce(robust)
    fixed``: the pooled coefficients are estimated on the same rows as
    the group coefficients, and the unexplained SE carries that
    covariance (0.01929 before, 0.01448 now, Stata 0.01443).

    Stata scales all three regressions by N/(N - 1) (``suest``), StatsPAI
    each by its own HC1 factor n/(n - k). The explained part isolates
    that: its SE ratio is exactly sqrt((N - 1)/(N - k)). The unexplained
    SE mixes the three factors, so it is held to 0.5%.
    """
    o = _oaxaca(card, "pooled").overall
    n, k = len(card), len(OAX_X) + 1
    assert o["explained"] == pytest.approx(
        STATA[("oaxaca_pooled", "b", "overall:explained")], rel=1e-10
    )
    assert o["explained_se"] / STATA[
        ("oaxaca_pooled", "se", "overall:explained")
    ] == pytest.approx(np.sqrt((n - 1) / (n - k)), rel=1e-9)
    assert o["unexplained_se"] == pytest.approx(
        STATA[("oaxaca_pooled", "se", "overall:unexplained")], rel=5e-3
    )


def test_oaxaca_summary_prints_the_threefold_decomposition(card):
    """The block headed "Threefold decomposition" printed the twofold
    explained part and the two halves of the unexplained part under the
    names Endowments / Coefficients (A) / Coefficients (B). It now prints
    endowments, coefficients and interaction, equal to Stata ``oaxaca``'s
    threefold default (SEs with ``vce(robust) fixed``)."""
    res = _oaxaca(card, 0)
    o = res.overall
    for part in ("endowments", "coefficients", "interaction"):
        assert o[part] == pytest.approx(
            STATA[("oaxaca_threefold", "b", f"overall:{part}")], rel=1e-10
        )
        assert o[f"{part}_se"] == pytest.approx(
            STATA[("oaxaca_threefold", "se", f"overall:{part}")], rel=1e-9
        )
    assert o["endowments"] + o["coefficients"] + o["interaction"] == pytest.approx(
        o["gap"], rel=1e-12
    )
    with contextlib.redirect_stdout(io.StringIO()):
        text = res.summary()
    block = text.split("Threefold decomposition:")[1]
    assert f"{o['endowments']:.4f}" in block
    assert f"{o['interaction']:.4f}" in block
    assert "Coefficients (A)" not in text
    assert "vce(robust) fixed" in text


# ---------------------------------------------------------------------------
# Summary header lines
# ---------------------------------------------------------------------------


def test_summary_states_the_sample_size_and_the_clusters(card, castle):
    """``summary()`` of a regression printed neither N nor, for
    ``sp.feols``, which variables the "CRV" standard errors cluster on."""
    assert "Observations: 3010" in str(_quiet(sp.regress, WILD, card).summary())
    one = _quiet(
        sp.feols, "l_homicide ~ post | sid + year", castle, vcov={"CRV1": "sid"}
    )
    assert "cluster (sid), G=50" in str(one.summary())
    two = _quiet(
        sp.feols, "l_homicide ~ post | sid + year", castle, vcov={"CRV1": "sid + year"}
    )
    text = str(two.summary())
    assert "cluster (sid, year), smallest G=11" in text
    assert "Observations: 550" in text
