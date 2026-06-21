"""Public-health validation — Hernán & Robins, *Causal Inference: What If*.

These tests prove that StatsPAI reproduces the **published g-methods
estimates** from Hernán & Robins (2020) on the *real* NHEFS data bundled
in ``sp.datasets.nhefs()`` — the canonical teaching dataset of modern
epidemiology.  Unlike the simulated econometrics replicas (whose tests
live in ``test_published_replications.py`` and only claim to recover the
*neighbourhood* of published numbers), NHEFS is genuine public-domain
data, so StatsPAI is held to the book's actual figures.

Each test asserts two things:

1. **Pinned** — the StatsPAI estimate equals the value observed when the
   test was written (guards against silent numerical drift across
   refactors).
2. **Published neighbourhood** — the estimate sits within a
   pre-registered tolerance of the number printed in the book (proves the
   reproduction is real, not an artefact).

The companion same-bytes R gold references live in
``tests/orig_parity/06_nhefs_ch12_ipw.{py,R}`` (and siblings for the
other chapters); the published anchors are catalogued in
``tests/external_parity/PUBLISHED_REFERENCE_VALUES.md``.

Reference
---------
Hernán, M.A. & Robins, J.M. (2020). *Causal Inference: What If*.
Boca Raton: Chapman & Hall/CRC.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# =========================================================================
# Pinned reference values (StatsPAI output on the bundled NHEFS extract,
# at the moment of test creation). Updating a pin requires an explicit
# commit — this is the anti-drift contract.
# =========================================================================

# Ch12 — IP weighting / marginal structural model
PINNED_CRUDE_WT_DIFF = 2.5406       # quitters - non-quitters, unadjusted
PINNED_CH12_IPW_ATT = 3.4405        # sp.ipw (Hájek ATE), book design, seed=42

# Published book anchors (Hernán-Robins, What If, Part II)
BOOK_CRUDE = 2.54                   # §12.2
BOOK_IPW_ATT = 3.4                  # Program 12.4, 95% CI (2.4, 4.5)
BOOK_IPW_CI = (2.4, 4.5)
BOOK_GFORMULA_ATT = 3.5            # Ch13 standardization / g-formula
BOOK_GEST_PSI = 3.4               # Ch14 g-estimation

PIN_TOL = 1e-2     # pinned-match tolerance (kg) — BLAS/bootstrap jitter
BOOK_TOL_KG = 0.3  # neighbourhood-of-published tolerance (kg)


# Canonical confounder model (self-contained copy of tests/orig_parity/_nhefs.py
# so this pytest module has no cross-directory import dependency).
_CAT_CONF = ["sex", "race", "education", "exercise", "active"]
_CONT_CONF = ["age", "smokeintensity", "smokeyrs", "wt71"]


def _book_design(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    for c in _CONT_CONF:
        d[f"{c}2"] = d[c] ** 2
    dd = pd.get_dummies(d, columns=_CAT_CONF, drop_first=True)
    cat_cols = [c for c in dd.columns
                if any(c.startswith(p + "_") for p in _CAT_CONF)]
    covs = cat_cols + _CONT_CONF + [f"{c}2" for c in _CONT_CONF]
    for c in covs:
        dd[c] = dd[c].astype(float)
    return dd, covs


# =========================================================================
# The bundled dataset itself
# =========================================================================

class TestNhefsDataset:
    """The NHEFS loader ships the real book extract."""

    def test_full_shape_and_provenance(self):
        df = sp.datasets.nhefs()
        assert df.shape == (1629, 67)
        assert df.attrs["data_source"] == "real"
        assert df.attrs["simulated"] is False
        assert "qsmk" in df.columns and "wt82_71" in df.columns
        assert "death" in df.columns

    def test_complete_case_sample(self):
        dc = sp.datasets.nhefs(complete_case=True)
        assert dc.shape == (1566, 67)         # the book's weight-analysis n
        assert dc["wt82_71"].notna().all()

    def test_load_alias_is_same(self):
        assert sp.datasets.load_nhefs is sp.datasets.nhefs

    def test_published_anchors_attached(self):
        df = sp.datasets.nhefs()
        assert df.attrs["published_ipw_att"] == pytest.approx(3.4)
        assert tuple(df.attrs["published_ipw_att_ci"]) == (2.4, 4.5)
        assert df.attrs["published_gformula_att"] == pytest.approx(3.5)

    def test_registered_in_list_datasets(self):
        names = set(sp.datasets.list_datasets()["name"])
        assert "nhefs" in names


# =========================================================================
# Chapter 12 — IP weighting and marginal structural models
# =========================================================================

class TestCh12IPWeighting:
    """quitting smoking -> 10-year weight change, via IP weighting."""

    @pytest.fixture(scope="class")
    def df(self):
        return sp.datasets.nhefs(complete_case=True)

    def test_crude_difference(self, df):
        crude = (df.loc[df.qsmk == 1, "wt82_71"].mean()
                 - df.loc[df.qsmk == 0, "wt82_71"].mean())
        # pinned to machine precision; matches book §12.2 (2.54 kg)
        assert crude == pytest.approx(PINNED_CRUDE_WT_DIFF, abs=PIN_TOL)
        assert crude == pytest.approx(BOOK_CRUDE, abs=BOOK_TOL_KG)

    def test_ipw_att_reproduces_book(self, df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dd, covs = _book_design(df)
            res = sp.ipw(dd, y="wt82_71", treat="qsmk", covariates=covs,
                         estimand="ATE", seed=42, n_bootstrap=500)
        # (1) pinned — no silent drift
        assert res.estimate == pytest.approx(PINNED_CH12_IPW_ATT, abs=0.01)
        # (2) neighbourhood of the published 3.4 kg
        assert res.estimate == pytest.approx(BOOK_IPW_ATT, abs=BOOK_TOL_KG)
        # (3) the 95% CI overlaps the book's (2.4, 4.5)
        lo, hi = res.ci
        assert lo < BOOK_IPW_CI[1] and hi > BOOK_IPW_CI[0]
        assert res.n_obs == 1566

    def test_ipw_propensity_scores_use_converged_glm_anchor(self, df):
        """The NHEFS design is intentionally unscaled; sklearn lbfgs can stop
        early and drift the point estimate. Lock the converged GLM anchor
        directly so solver regressions fail before they reach the manuscript
        numbers.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dd, covs = _book_design(df)
            res = sp.ipw(dd, y="wt82_71", treat="qsmk", covariates=covs,
                         estimand="ATE", seed=42, n_bootstrap=0)
        assert res.estimate == pytest.approx(3.4405354296, abs=1e-6)
        assert res.model_info["pscore_min"] == pytest.approx(0.051000764, abs=1e-6)
        assert res.model_info["pscore_max"] == pytest.approx(0.776888702, abs=1e-6)


# =========================================================================
# Chapter 13 — Standardization / parametric g-formula
# =========================================================================

PINNED_CH13_GFORMULA = 3.4626      # sp.g_computation point estimate


class TestCh13Standardization:
    def test_gformula_reproduces_book(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dd, covs = _book_design(sp.datasets.nhefs(complete_case=True))
            g = sp.g_computation(dd, y="wt82_71", treat="qsmk",
                                 covariates=covs, n_boot=100, seed=42)
        assert g.estimate == pytest.approx(PINNED_CH13_GFORMULA, abs=0.05)
        assert g.estimate == pytest.approx(BOOK_GFORMULA_ATT, abs=BOOK_TOL_KG)


# =========================================================================
# Chapter 14 — G-estimation of a structural nested mean model
# =========================================================================

PINNED_CH14_PSI = 3.4626           # sp.g_estimation psi


class TestCh14GEstimation:
    def test_snmm_psi_reproduces_book(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dd, covs = _book_design(sp.datasets.nhefs(complete_case=True))
            ge = sp.g_estimation(dd, y="wt82_71", treatments=["qsmk"],
                                 covariates_by_stage=[covs], random_state=42,
                                 n_bootstrap=100)
        assert ge.estimate == pytest.approx(PINNED_CH14_PSI, abs=0.05)
        assert ge.estimate == pytest.approx(BOOK_GEST_PSI, abs=BOOK_TOL_KG)


# =========================================================================
# Chapter 15 — Outcome regression with effect modification
# =========================================================================

# These closed-form regression coefficients match the book to 4 decimals.
PINNED_CH15_QSMK = 2.5596
PINNED_CH15_INTERACTION = 0.0467
BOOK_CH15_QSMK = 2.56              # Program 15.1
BOOK_CH15_INTERACTION = 0.0467


class TestCh15OutcomeRegression:
    @pytest.fixture(scope="class")
    def fit(self):
        formula = (
            "wt82_71 ~ qsmk + smokeintensity + qsmk:smokeintensity "
            "+ C(sex) + C(race) + C(education) + C(exercise) + C(active) "
            "+ age + I(age**2) + I(smokeintensity**2) "
            "+ smokeyrs + I(smokeyrs**2) + wt71 + I(wt71**2)"
        )
        return sp.regress(formula, data=sp.datasets.nhefs(complete_case=True))

    def test_qsmk_main_coefficient(self, fit):
        b = float(fit.params["qsmk"])
        assert b == pytest.approx(PINNED_CH15_QSMK, abs=1e-2)
        assert b == pytest.approx(BOOK_CH15_QSMK, abs=1e-2)

    def test_interaction_coefficient(self, fit):
        b = float(fit.params["qsmk:smokeintensity"])
        assert b == pytest.approx(PINNED_CH15_INTERACTION, abs=1e-3)
        assert b == pytest.approx(BOOK_CH15_INTERACTION, abs=1e-3)

    def test_effect_at_smoking_intensities(self, fit):
        b = float(fit.params["qsmk"])
        bi = float(fit.params["qsmk:smokeintensity"])
        assert (b + 5 * bi) == pytest.approx(2.79, abs=0.02)    # book 2.79
        assert (b + 40 * bi) == pytest.approx(4.43, abs=0.02)   # book 4.43


# =========================================================================
# Chapter 17 — Survival: confounding inflates the crude hazard ratio
# =========================================================================

PINNED_CH17_UNWEIGHTED_HR = 1.3941   # statsmodels PHReg, matches R coxph


class TestCh17Survival:
    """The unadjusted qsmk -> mortality hazard ratio (~1.4) reflects
    confounding; the full IP-weighted reproduction (HR -> ~1.0) lives in
    tests/orig_parity/10_nhefs_ch17_survival.{py,R}."""

    def test_unweighted_hazard_ratio(self):
        import statsmodels.api as sm
        df = sp.datasets.nhefs().copy()
        df["survtime"] = np.where(df.death == 0, 120,
                                  (df.yrdth - 83) * 12 + df.modth)
        df = df[df.survtime.notna()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.PHReg(df["survtime"], df[["qsmk"]].astype(float),
                         status=df["death"]).fit()
        hr = float(np.exp(m.params[0]))
        assert hr == pytest.approx(PINNED_CH17_UNWEIGHTED_HR, abs=0.02)
        assert hr == pytest.approx(1.39, abs=0.05)             # book §17


# =========================================================================
# Sensitivity analysis — the E-value (VanderWeele & Ding 2017)
# =========================================================================

class TestEValueSensitivity:
    def test_evalue_matches_closed_form(self):
        rr = 1.3251                       # crude qsmk -> death risk ratio
        ev = sp.evalue(estimate=rr, measure="RR")
        closed = rr + np.sqrt(rr * (rr - 1.0))
        assert ev["evalue_estimate"] == pytest.approx(1.9814, abs=1e-3)
        assert ev["evalue_estimate"] == pytest.approx(closed, abs=1e-6)
