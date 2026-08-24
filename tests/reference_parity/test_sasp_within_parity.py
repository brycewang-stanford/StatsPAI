"""Stata parity for the SASP within transformation (Mixtape Ch. 8).

Reference values from **Stata 18 MP** on the same bundled CSV, following
``Do/sasp.do``: drop rows missing any modelling variable, keep providers
with exactly four sessions (N=1028, 257 providers), then::

    global X age asq bmi hispanic black other asian schooling cohab      ///
             married divorced separated age_cl unsafe llength reg asq_cl ///
             appearance_cl provider_second asian_cl black_cl hispanic_cl ///
             othrace_cl hot massage_cl

    reg   lnw $X, robust                     // pooled
    xtreg lnw $X, fe i(id) robust            // within
    reg   w_lnw w_age ..., robust cluster(id) // manual demeaning

The third specification is the point of the chapter: demeaning by hand
and running OLS reproduces ``xtreg, fe`` *exactly*, standard error
included.  These tests pin that identity so a refactor of either
``sp.feols`` or ``sp.demean`` cannot quietly break it.
"""

from __future__ import annotations

import pandas as pd
import pytest

import statspai as sp
from statspai.datasets import SASP_COVARIATES, SASP_TIME_INVARIANT

# Stata 18 MP, coefficient on `unsafe` and its standard error.
STATA_POOLED = (0.013407389, 0.028300101)
STATA_WITHIN = (0.051033874, 0.028283095)
STATA_DEMEANED = (0.051033874, 0.028283095)

ATOL = 1e-6

X = list(SASP_COVARIATES)


@pytest.fixture(scope="module")
def sasp():
    return sp.datasets.sasp_panel(analytic_sample=True)


@pytest.fixture(scope="module")
def demeaned(sasp):
    """Provider-demeaned copy of the modelling columns."""
    cols = ["lnw"] + X
    values, _ = sp.demean(sasp[cols].to_numpy(float), sasp[["id"]])
    w = pd.DataFrame(values, columns=["w_" + c for c in cols], index=sasp.index)
    w["id"] = sasp["id"].values
    return w


# ---------------------------------------------------------------------------
# Analytic sample
# ---------------------------------------------------------------------------


def test_analytic_sample_matches_the_book_recipe(sasp):
    """Complete cases, then providers with exactly four sessions."""
    assert len(sasp) == 1028
    assert sasp["id"].nunique() == 257
    assert (sasp.groupby("id")["session"].size() == 4).all()
    assert not sasp[["lnw", *X]].isna().any().any()


def test_full_panel_is_larger_than_the_analytic_sample():
    full = sp.datasets.sasp_panel()
    assert len(full) == 1787
    assert full.attrs["analytic_sample"] is False
    assert sp.datasets.sasp_panel(analytic_sample=True).attrs["analytic_sample"]


# ---------------------------------------------------------------------------
# The three routes
# ---------------------------------------------------------------------------


def test_pooled_ols_matches_stata(sasp):
    r = sp.regress("lnw ~ " + " + ".join(X), data=sasp, robust="hc1")
    assert float(r.params["unsafe"]) == pytest.approx(STATA_POOLED[0], abs=ATOL)
    assert float(r.std_errors["unsafe"]) == pytest.approx(STATA_POOLED[1], abs=ATOL)


def test_within_estimator_matches_stata(sasp):
    """`xtreg, fe robust` is cluster-robust on the panel id, not plain HC1."""
    pytest.importorskip(
        "pyfixest", reason="sp.feols is backed by the optional [fixest] extra"
    )
    r = sp.feols("lnw ~ " + " + ".join(X) + " | id", data=sasp, vcov={"CRV1": "id"})
    assert float(r.params["unsafe"]) == pytest.approx(STATA_WITHIN[0], abs=ATOL)
    assert float(r.std_errors["unsafe"]) == pytest.approx(STATA_WITHIN[1], abs=ATOL)


def test_manual_demeaning_reproduces_the_within_estimator(sasp, demeaned):
    """The identity Chapter 8 exists to teach — exact, not approximate."""
    pytest.importorskip(
        "pyfixest", reason="sp.feols is backed by the optional [fixest] extra"
    )
    alive = [c for c in X if c not in SASP_TIME_INVARIANT]
    by_hand = sp.regress(
        "w_lnw ~ " + " + ".join("w_" + c for c in alive),
        data=demeaned,
        vce="cluster",
        cluster="id",
    )
    fe = sp.feols("lnw ~ " + " + ".join(X) + " | id", data=sasp, vcov={"CRV1": "id"})

    assert float(by_hand.params["w_unsafe"]) == pytest.approx(
        STATA_DEMEANED[0], abs=ATOL
    )
    assert float(by_hand.std_errors["w_unsafe"]) == pytest.approx(
        STATA_DEMEANED[1], abs=ATOL
    )
    # ...and identical to sp.feols, not merely close to Stata.
    assert float(by_hand.params["w_unsafe"]) == pytest.approx(
        float(fe.params["unsafe"]), abs=1e-9
    )
    assert float(by_hand.std_errors["w_unsafe"]) == pytest.approx(
        float(fe.std_errors["unsafe"]), abs=1e-9
    )


# ---------------------------------------------------------------------------
# What the within transformation costs you
# ---------------------------------------------------------------------------


def test_provider_level_controls_are_annihilated(sasp, demeaned):
    """Exactly the twelve documented columns lose all variation."""
    dead = [c for c in X if demeaned["w_" + c].std() < 1e-12]
    assert set(dead) == set(SASP_TIME_INVARIANT)
    assert len(dead) == 12


def test_regress_refuses_a_constant_regressor_instead_of_dropping_it(demeaned):
    """StatsPAI names the unidentified regressor; Stata omits it silently.

    Guards the "fail loudly" contract: handing a demeaned time-invariant
    control to sp.regress must raise, not silently return a zero.
    """
    from statspai.exceptions import NumericalInstability

    with pytest.raises(NumericalInstability, match="w_age"):
        sp.regress("w_lnw ~ w_age + w_unsafe", data=demeaned, robust="hc1")


def test_pooled_and_within_disagree_substantively(sasp):
    """Not a precision story — pooling points somewhere else entirely."""
    pytest.importorskip(
        "pyfixest", reason="sp.feols is backed by the optional [fixest] extra"
    )
    pooled = sp.regress("lnw ~ " + " + ".join(X), data=sasp, robust="hc1")
    fe = sp.feols("lnw ~ " + " + ".join(X) + " | id", data=sasp, vcov={"CRV1": "id"})
    b_pooled = float(pooled.params["unsafe"])
    b_within = float(fe.params["unsafe"])
    assert b_within > 3 * b_pooled
    # Pooled cannot reject zero; the within estimate is on the boundary.
    assert abs(b_pooled) < 2 * float(pooled.std_errors["unsafe"])


# ---------------------------------------------------------------------------
# Replication-guide wiring
# ---------------------------------------------------------------------------


def test_replication_entry_registered_and_renders():
    table = sp.list_replications()
    row = table.loc[table["key"] == "sasp_within"]
    assert len(row) == 1
    assert bool(row["has_real_data"].iloc[0])

    _data, guide = sp.replicate("sasp_within")
    assert "within" in guide.lower()
    assert "annihilated" in guide.lower()
