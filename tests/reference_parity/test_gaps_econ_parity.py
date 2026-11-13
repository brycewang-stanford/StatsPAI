"""Reference parity for four additions an applied economist asked for.

The 2026-11 econometrics probe listed features a referee expects and did
not find. These are the ones with a number behind them:

* ``sp.oaxaca(means='stochastic')``: delta-method standard errors that
  count the sampling variance of the covariate means (Jann 2008), the
  default of Stata ``oaxaca``. The default ``means='fixed'`` is unchanged.
* ``sp.regtable(stats=['rmse', 'aic', 'bic'])`` for ``sp.feols`` /
  ``sp.fepois`` columns: fixest's ``fitstat(~ rmse + ll + aic + bic)``.
* ``sp.ivreg`` attaches the Kleibergen-Paap rk statistics and the
  Montiel Olea-Pflueger effective F that ``sp.iv`` reports: Stata
  ``ivreg2`` and ``weakivtest``.
* ``sp.avg_predictions(data=, variables=)``: predictions on new data and
  on counterfactual copies, ``marginaleffects::avg_predictions``.

Frozen references, on the data files shipped with the package
(``card_1995.csv``, ``castle_2013.csv``):

* Stata 18 MP, ``_fixtures/_generate_gaps_econ_Stata.do`` ->
  ``_fixtures/gaps_econ_Stata.csv`` (``oaxaca`` 4.1.1, ``ivreg2`` 4.1.12,
  ``ranktest`` 2.0.04, ``weakivtest``; all run as black boxes).
* R 4.5.2, ``_fixtures/_generate_gaps_econ_R.R`` ->
  ``_fixtures/gaps_econ_R.csv`` (fixest 0.14.0, marginaleffects 0.32.0).

Tolerances are stated where they are used. Closed-form quantities are
held to 1e-9 relative; quantities behind an iterative fit to 1e-7 or 1e-6
with the reason next to the assertion.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).resolve().parent / "_fixtures"
DATA = Path(__file__).resolve().parents[2] / "src" / "statspai" / "datasets" / "data"


def _load(name):
    frame = pd.read_csv(FIX / name)
    return {
        (r.tag, r.kind, r.name): float(r.value) for r in frame.itertuples(index=False)
    }


STATA = _load("gaps_econ_Stata.csv")
R = _load("gaps_econ_R.csv")


@pytest.fixture(scope="module")
def card():
    return pd.read_csv(DATA / "card_1995.csv")


@pytest.fixture(scope="module")
def castle():
    return pd.read_csv(DATA / "castle_2013.csv")


# ---------------------------------------------------------------------------
# sp.oaxaca(means='stochastic')
# ---------------------------------------------------------------------------

OAX_X = ["educ", "exper", "expersq", "south", "smsa"]


def _oaxaca(df, reference, means="stochastic"):
    with contextlib.redirect_stdout(io.StringIO()):
        return sp.oaxaca(
            df, y="lwage", group="black", x=OAX_X, reference=reference, means=means
        )


@pytest.mark.parametrize(
    "reference, tag",
    [(0, "ref0"), (1, "ref1"), ("reimers", "reimers"), ("cotton", "cotton")],
)
def test_oaxaca_stochastic_means_twofold(card, reference, tag):
    """Stata ``oaxaca ..., weight(w) vce(robust)``: the variance of each
    part adds ``g_x' V(xbar) g_x`` to the fixed-means delta-method
    variance, with ``V(xbar)`` the within-group covariance over n.

    All four are closed form given the group regressions; 1e-9 relative.
    """
    o = _oaxaca(card, reference).overall
    tag = f"oaxaca_{tag}"
    for part in ("explained", "unexplained"):
        assert o[part] == pytest.approx(STATA[(tag, "b", f"overall:{part}")], rel=1e-10)
        assert o[f"{part}_se"] == pytest.approx(
            STATA[(tag, "se", f"overall:{part}")], rel=1e-9
        )
    assert o["gap"] == pytest.approx(STATA[(tag, "b", "overall:difference")], rel=1e-10)
    assert o["gap_se"] == pytest.approx(
        STATA[(tag, "se", "overall:difference")], rel=1e-9
    )


@pytest.mark.parametrize(
    "reference, tag",
    [(0, "ref0"), (1, "ref1"), ("reimers", "reimers"), ("cotton", "cotton")],
)
def test_oaxaca_stochastic_means_detail(card, reference, tag):
    """Variable-level explained contributions, Stata ``oaxaca, detail``."""
    res = _oaxaca(card, reference)
    det = res.detailed.set_index("variable")
    for v in OAX_X:
        key = f"explained:{v}"
        assert det.loc[v, "contribution"] == pytest.approx(
            STATA[(f"oaxaca_{tag}", "b", key)], rel=1e-9
        )
        assert det.loc[v, "se"] == pytest.approx(
            STATA[(f"oaxaca_{tag}", "se", key)], rel=1e-9
        )


def test_oaxaca_stochastic_means_threefold(card):
    """Stata ``oaxaca ..., vce(robust)`` (threefold, from Group B)."""
    o = _oaxaca(card, 0).overall
    for part in ("endowments", "coefficients", "interaction"):
        assert o[part] == pytest.approx(
            STATA[("oaxaca_threefold", "b", f"overall:{part}")], rel=1e-10
        )
        assert o[f"{part}_se"] == pytest.approx(
            STATA[("oaxaca_threefold", "se", f"overall:{part}")], rel=1e-9
        )


def test_oaxaca_stochastic_means_pooled_adds_the_same_term_as_stata(card):
    """Pooled reference, Stata ``oaxaca, omega``.

    Stata scales the three regressions by N/(N - 1) (``suest``) and
    StatsPAI each by its own HC1 factor, so the fixed-means variances
    differ (documented in ``docs/dev/modules/decomposition.md``). The
    term that ``means='stochastic'`` adds does not involve the
    coefficient covariance, so it is the same in both programs:
    Var(stochastic) - Var(fixed) agrees to 1e-7 (a difference of two
    numbers printed to 17 digits, each ten times larger).
    """
    sto = _oaxaca(card, "pooled").overall
    fix = _oaxaca(card, "pooled", means="fixed").overall
    for part in ("explained", "unexplained"):
        added = sto[f"{part}_se"] ** 2 - fix[f"{part}_se"] ** 2
        ref = (
            STATA[("oaxaca_pooled", "se", f"overall:{part}")] ** 2
            - STATA[("oaxaca_pooled_fixed", "se", f"overall:{part}")] ** 2
        )
        assert added == pytest.approx(ref, rel=1e-7)


def test_oaxaca_default_is_the_fixed_means_variance(card):
    """``means='fixed'`` is the default and is Stata's ``fixed`` option;
    the new ``gap_se`` follows the same rule."""
    default = sp.oaxaca(card, y="lwage", group="black", x=OAX_X, reference=0)
    fixed = _oaxaca(card, 0, means="fixed")
    assert default.means == "fixed"
    assert default.overall == fixed.overall
    assert fixed.overall["gap_se"] == pytest.approx(
        STATA[("oaxaca_ref0_fixed", "se", "overall:difference")], rel=1e-9
    )
    assert fixed.overall["explained_se"] == pytest.approx(
        STATA[("oaxaca_ref0_fixed", "se", "overall:explained")], rel=1e-9
    )


def test_oaxaca_means_option_is_validated_and_labelled(card):
    with pytest.raises(MethodIncompatibility, match="means="):
        sp.oaxaca(card, y="lwage", group="black", x=OAX_X, means="random")
    with contextlib.redirect_stdout(io.StringIO()):
        text = _oaxaca(card, 0).summary()
    assert "treated as estimated" in text
    with contextlib.redirect_stdout(io.StringIO()):
        text = _oaxaca(card, 0, means="fixed").summary()
    assert "held fixed" in text


# ---------------------------------------------------------------------------
# sp.regtable: rmse, AIC and BIC for feols / fepois columns
# ---------------------------------------------------------------------------

FE = "post + unemployrt | sid + year"
NOFE = "post + unemployrt"


def _fit(castle, tag):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if tag == "feols_fe":
            return sp.feols(f"l_homicide ~ {FE}", castle, vcov={"CRV1": "sid"})
        if tag == "feols_nofe":
            return sp.feols(f"l_homicide ~ {NOFE}", castle)
        if tag == "fepois_fe":
            return sp.fepois(f"homicide ~ {FE}", castle, vcov={"CRV1": "sid"})
        if tag == "fepois_nofe":
            return sp.fepois(f"homicide ~ {NOFE}", castle)
        if tag == "fepois_w":
            return sp.fepois(f"homicide ~ {FE}", castle, weights="popwt")
        return sp.feols(f"l_homicide ~ {FE}", castle, weights="popwt")


@pytest.mark.parametrize(
    "tag", ["feols_fe", "feols_nofe", "fepois_fe", "fepois_nofe", "fepois_w"]
)
def test_fit_statistics_match_fixest_fitstat(castle, tag):
    """``fitstat(m, ~ rmse + ll + aic + bic)``. AIC and BIC count the
    fixed effects as parameters (62 here with state and year effects),
    under a clustered variance too.

    OLS quantities are closed form (1e-9). The Poisson log-likelihood is
    behind an IRLS fit that both programs stop at 1e-8 on the deviance;
    1e-7 on the statistics.
    """
    d = _fit(castle, tag).diagnostics
    rel = 1e-9 if tag.startswith("feols") else 1e-7
    assert d["RMSE"] == pytest.approx(R[(tag, "s", "rmse")], rel=rel)
    assert d["Log-likelihood"] == pytest.approx(R[(tag, "s", "ll")], rel=rel)
    assert d["AIC"] == pytest.approx(R[(tag, "s", "aic")], rel=rel)
    assert d["BIC"] == pytest.approx(R[(tag, "s", "bic")], rel=rel)


def test_weighted_feols_reports_rmse_but_no_information_criteria(castle):
    """fixest's log-likelihood of a weighted ``feols`` ignores the weights
    (it is the Gaussian likelihood of the unweighted residuals), R ``lm``
    and Stata each use another weighted likelihood. StatsPAI reports the
    weighted RMSE (= fixest) and no AIC / BIC for a weighted linear fit.
    """
    fit = _fit(castle, "feols_w")
    d = fit.diagnostics
    assert d["RMSE"] == pytest.approx(R[("feols_w", "s", "rmse")], rel=1e-9)
    assert "AIC" not in d and "BIC" not in d
    # The reference's number is the unweighted-residual likelihood.
    u = np.asarray(fit.data_info["residuals"], dtype=float)
    n = u.size
    ll_unweighted = -0.5 * n * (np.log(2 * np.pi * float(u @ u) / n) + 1.0)
    assert ll_unweighted == pytest.approx(R[("feols_w", "s", "ll")], rel=1e-9)


def test_regtable_prints_rmse_aic_bic_rows(castle):
    ols = _fit(castle, "feols_fe")
    pois = _fit(castle, "fepois_fe")
    table = sp.regtable(
        ols, pois, stats=["N", "rmse", "aic", "bic"], stats_fmt="%.4f"
    ).to_text()
    rows = {
        line.split()[0]: line.split()[1:]
        for line in table.splitlines()
        if line.split() and line.split()[0] in ("RMSE", "AIC", "BIC")
    }
    for label, key in (("RMSE", "rmse"), ("AIC", "aic"), ("BIC", "bic")):
        cells = [float(c.replace(",", "")) for c in rows[label]]
        assert cells[0] == pytest.approx(R[("feols_fe", "s", key)], abs=5.1e-5)
        assert cells[1] == pytest.approx(R[("fepois_fe", "s", key)], abs=5.1e-5)


# ---------------------------------------------------------------------------
# sp.ivreg: Kleibergen-Paap rk statistics and the effective F
# ---------------------------------------------------------------------------

IV = "lwage ~ (educ ~ nearc4 + nearc2) + exper + expersq + black + south + smsa"


def _ivreg(card, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.ivreg(IV, card, **kw)


def test_ivreg_rank_statistics_iid_match_ivreg2(card):
    """``ivreg2``: with i.i.d. errors the underidentification LM is the
    Anderson statistic and the weak-identification F is Cragg-Donald's;
    the effective F then equals it."""
    d = _ivreg(card).diagnostics
    assert d["KP rk LM"] == pytest.approx(STATA[("iv_iid", "s", "idstat")], rel=1e-9)
    assert d["KP rk LM p-value"] == pytest.approx(
        STATA[("iv_iid", "s", "idp")], rel=1e-8
    )
    assert d["KP rk Wald F"] == pytest.approx(STATA[("iv_iid", "s", "cdf")], rel=1e-9)
    assert d["Olea-Pflueger effective F"] == pytest.approx(
        STATA[("iv_iid", "s", "cdf")], rel=1e-9
    )


@pytest.mark.parametrize(
    "tag, kwargs",
    [("iv_robust", {"robust": "hc1"}), ("iv_cluster", {"cluster": "exper"})],
)
def test_ivreg_rank_statistics_and_effective_f_match_stata(card, tag, kwargs):
    """``ivreg2, robust`` / ``cluster()``: ``e(idstat)``, ``e(idp)``,
    ``e(rkf)``; ``weakivtest``: ``r(F_eff)``. Closed form, 1e-9."""
    d = _ivreg(card, **kwargs).diagnostics
    assert d["KP rk LM"] == pytest.approx(STATA[(tag, "s", "idstat")], rel=1e-9)
    assert d["KP rk LM p-value"] == pytest.approx(STATA[(tag, "s", "idp")], rel=1e-8)
    assert d["KP rk Wald F"] == pytest.approx(STATA[(tag, "s", "rkf")], rel=1e-9)
    assert d["Olea-Pflueger effective F"] == pytest.approx(
        STATA[(tag, "s", "F_eff")], rel=1e-9
    )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"robust": "hc1"}, {"cluster": "exper"}, {"vce": "cr2", "cluster": "exper"}],
)
def test_ivreg_fit_is_unchanged_and_agrees_with_sp_iv(card, kwargs):
    """The statistics are read off the fitted model. Coefficients,
    standard errors and every other diagnostic are those of the call
    without them, and the new entries are the ones ``sp.iv`` attaches."""
    with_diag = _ivreg(card, **kwargs)
    without = _ivreg(card, augmented_diagnostics=False, **kwargs)
    assert with_diag.params.equals(without.params)
    assert with_diag.std_errors.equals(without.std_errors)
    for key, value in without.diagnostics.items():
        assert with_diag.diagnostics[key] == value or (
            value != value and with_diag.diagnostics[key] != with_diag.diagnostics[key]
        )
    assert "KP rk Wald F" not in without.diagnostics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sp.iv(IV, card, **kwargs).diagnostics
    for key in ("KP rk LM", "KP rk Wald F", "Olea-Pflueger effective F"):
        assert with_diag.diagnostics[key] == ref[key]


def test_ivreg_two_endogenous_attaches_sanderson_windmeijer(card):
    """With two endogenous regressors the conditional first-stage F of
    each is attached, as in ``sp.iv`` (whose values are pinned to
    ``ivreg2`` in ``test_iv_sw_cluster_parity``-style tests)."""
    fml = "lwage ~ (educ + smsa ~ nearc4 + nearc2 + south) + exper + expersq + black"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = sp.ivreg(fml, card, robust="hc1").diagnostics
        ref = sp.iv(fml, card, robust="hc1").diagnostics
    keys = [k for k in ref if k.startswith(("SW conditional F", "KP rk"))]
    assert "SW conditional F (educ)" in keys and "SW conditional F (smsa)" in keys
    for key in keys:
        assert d[key] == ref[key]
    assert "Olea-Pflueger effective F" not in d


# ---------------------------------------------------------------------------
# sp.avg_predictions(data=, variables=)
# ---------------------------------------------------------------------------

# marginaleffects differentiates numerically. The reference is generated
# with numderiv = "richardson" (its default forward difference leaves the
# standard errors about 2e-6 from the analytic delta method used here) and
# a logit fitted with epsilon = 1e-14. Estimates are held to 1e-8, standard
# errors to 1e-7.
AP_EST, AP_SE = 1e-8, 1e-7


@pytest.fixture(scope="module")
def logit_fit(card):
    d = card.copy()
    d["hi"] = (d["lwage"] > d["lwage"].median()).astype(int)
    return d, sp.logit("hi ~ educ + exper + black + south + smsa", data=d)


def _check(table, tag, keys=()):
    assert len(table) == sum(1 for k in R if k[0] == tag and k[1] == "b")
    for _, row in table.iterrows():
        name = ";".join(f"{k}={int(row[k])}" for k in keys) if keys else "all"
        assert row["estimate"] == pytest.approx(R[(tag, "b", name)], rel=AP_EST)
        assert row["std_error"] == pytest.approx(R[(tag, "se", name)], rel=AP_SE)


def test_avg_predictions_on_new_data(logit_fit):
    d, fit = logit_fit
    _check(sp.avg_predictions(fit, newdata=d[d["black"] == 1]), "ap_subset")
    _check(sp.avg_predictions(fit, data=d[d["black"] == 1]), "ap_subset")
    _check(sp.avg_predictions(fit, data=d.assign(black=1)), "ap_setto1")


def test_avg_predictions_counterfactual_variables(logit_fit):
    """``avg_predictions(m, variables = list(black = c(0, 1)))``: every
    row predicted under both values, one average each (the standardised
    means of g-computation)."""
    d, fit = logit_fit
    _check(sp.avg_predictions(fit, variables={"black": [0, 1]}), "ap_cf", ["black"])
    _check(sp.avg_predictions(fit, variables="black"), "ap_cf_name", ["black"])
    _check(
        sp.avg_predictions(fit, variables={"black": [0, 1]}, type="link"),
        "ap_cf_link",
        ["black"],
    )
    _check(
        sp.avg_predictions(fit, variables={"black": [0, 1], "educ": [12, 16]}),
        "ap_cf_grid",
        ["black", "educ"],
    )


def test_avg_predictions_counterfactual_with_by(logit_fit):
    """An explicit ``by=`` replaces the grouping by the manipulated
    variable, as in marginaleffects."""
    _, fit = logit_fit
    _check(
        sp.avg_predictions(fit, variables={"black": [0, 1]}, by="south"),
        "ap_cf_by",
        ["south"],
    )


def test_avg_predictions_counterfactual_ols_with_interaction(card):
    fit = sp.regress("lwage ~ educ + exper + black * south + smsa", data=card)
    table = sp.avg_predictions(fit, variables={"black": [0, 1]})
    _check(table, "ap_ols_cf", ["black"])


def test_predictions_counterfactual_rows_and_identity(logit_fit):
    """Unit-level counterfactual predictions: one row per unit and value,
    and their group means are the averages above."""
    d, fit = logit_fit
    unit = sp.predictions(fit, variables={"black": [0, 1]})
    assert len(unit) == int(R[("ap_cf", "s", "n_unit_rows")]) == 2 * len(d)
    avg = sp.avg_predictions(fit, variables={"black": [0, 1]})
    means = unit.groupby("black")["estimate"].mean()
    np.testing.assert_allclose(means.to_numpy(), avg["estimate"].to_numpy(), rtol=1e-12)
    # setting the variable by hand is the same average
    by_hand = sp.avg_predictions(fit, data=d.assign(black=1))
    assert by_hand["estimate"].iloc[0] == pytest.approx(
        avg["estimate"].iloc[1], rel=1e-12
    )
    assert by_hand["std_error"].iloc[0] == pytest.approx(
        avg["std_error"].iloc[1], rel=1e-12
    )


def test_avg_predictions_argument_errors(logit_fit):
    d, fit = logit_fit
    with pytest.raises(MethodIncompatibility, match="not both"):
        sp.avg_predictions(fit, newdata=d, data=d)
    with pytest.raises(MethodIncompatibility, match="not a regressor"):
        sp.avg_predictions(fit, variables={"lwage": [0, 1]})
    with pytest.raises(MethodIncompatibility, match="needs the values"):
        sp.avg_predictions(fit, variables="educ")
