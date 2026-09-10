"""Reference parity for the v0.10 DiD frontier:

- ``sp.did_imputation``     ↔  R didimputation::did_imputation (BJS 2024)
- ``sp.gardner_did``        ↔  R did2s::did2s (Gardner 2021)
- ``sp.wooldridge_did``     ↔  R etwfe::etwfe + emfx (Wooldridge 2021)

All three estimators target the same heterogeneous-effects-robust
ATT under staggered adoption.  We re-use the staggered DGP from
``cs_data.csv`` (true simple ATT ≈ 2.75) so all three references
share fixtures.

Tolerance: 10% relative on the ATT coefficient — these estimators
target the same population parameter and should land in the same
neighbourhood.  Each estimator has its own SE construction
(analytic vs cluster-robust vs sandwich), so SE comparisons use
order-of-magnitude (2×) bounds rather than tight relative bands.

Each test skips gracefully when the corresponding R reference is
unavailable (R package missing or API mismatch).

References
----------
- Borusyak, K., Jaravel, X. and Spiess, J. (2024).
  "Revisiting Event Study Designs: Robust and Efficient Estimation."
  *Review of Economic Studies*. [@borusyak2024revisiting]
- Gardner, J. (2022). "Two-stage differences in differences."
  arXiv:2207.05943. [@gardner2022twostage]
- Wooldridge, J.M. (2021). "Two-way fixed effects, the two-way
  Mundlak regression, and difference-in-differences estimators."
  SSRN. [@wooldridge2021two]
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

import statspai as sp

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def cs_data():
    return pd.read_csv(_FIXTURE_DIR / "cs_data.csv")


@pytest.fixture(scope="module")
def r_reference():
    with open(_FIXTURE_DIR / "did_variants_R.json") as f:
        return json.load(f)


# ─── BJS did_imputation ─────────────────────────────────────────────────


# KNOWN DIVERGENCE — flagged for v1.11 follow-up.
#
# On the shared cs_data.csv DGP (true ATT = 2.75), R didimputation
# returns 2.7468, but sp.did_imputation returns 3.4039 — a 24%
# upward bias.  Both packages claim BJS 2024.  Possible sources:
#   (a) the imputation step uses ALL pre-treatment periods in R but
#       only the immediate pre-period in sp — re-check pre_periods;
#   (b) horizon weighting differs (R averages over post-periods
#       weighted by cohort size; sp may weight by N(g, t) cells);
#   (c) never-treated treatment of g=0 / 9999 sentinel is handled
#       differently between the two stacks.
#
# The BJS divergence flagged in v1.11 was fixed in commit 3607a2f
# (``fix(did): repair BJS imputation support``); both tests below now
# match the R reference within 10%. The previous ``@pytest.mark.xfail``
# markers were left stale and have been removed — a future regression
# will fail the suite loudly rather than be hidden as XPASS.


def test_bjs_did_imputation_matches_R(cs_data, r_reference):
    bjs_meta = r_reference["bjs"]["meta"]
    if not bjs_meta.get("available", False):
        pytest.skip(f"R didimputation unavailable: {bjs_meta.get('error', '?')}")
    res = sp.did_imputation(
        data=cs_data,
        y="y",
        group="id",
        time="year",
        first_treat="first_treat",
    )
    py_att = float(res.estimate)
    r_att = r_reference["bjs"]["estimate"]
    rel = abs(py_att - r_att) / abs(r_att)
    assert rel < 0.10, (
        f"sp.did_imputation drifted from R didimputation by {rel:.1%} "
        f"(Python={py_att:.4f}, R={r_att:.4f})"
    )


def test_bjs_did_imputation_close_to_truth(cs_data, r_reference):
    if not r_reference["bjs"]["meta"].get("available", False):
        pytest.skip("BJS R reference unavailable")
    res = sp.did_imputation(
        data=cs_data,
        y="y",
        group="id",
        time="year",
        first_treat="first_treat",
    )
    assert abs(float(res.estimate) - 2.75) < 0.5


# ─── Gardner two-stage ─────────────────────────────────────────────────


def test_gardner_did_matches_R(cs_data, r_reference):
    g_meta = r_reference["gardner"]["meta"]
    if not g_meta.get("available", False):
        pytest.skip(f"R did2s unavailable: {g_meta.get('error', '?')}")
    res = sp.gardner_did(
        data=cs_data,
        y="y",
        group="id",
        time="year",
        first_treat="first_treat",
    )
    py_att = float(res.estimate)
    r_att = r_reference["gardner"]["estimate"]
    rel = abs(py_att - r_att) / abs(r_att)
    assert rel < 0.10, (
        f"sp.gardner_did drifted from R did2s by {rel:.1%} "
        f"(Python={py_att:.4f}, R={r_att:.4f})"
    )


# ─── Wooldridge etwfe ──────────────────────────────────────────────────


def test_wooldridge_did_matches_R(cs_data, r_reference):
    """sp.wooldridge_did against R etwfe on the shared staggered fixture.

    This assertion was an ``xfail(strict=False)`` from v1.11 to v1.26,
    recording that sp.wooldridge_did returned 2.15 against R's 2.75 -- a
    ~22% downward bias -- with the note "likely a cohort-weighting
    difference in the Mundlak transform". The note was wrong (reweighting
    the reported cohort ATTs only reaches 2.21) and the non-strict xfail
    meant the defect could neither fail nor announce a fix.

    The cause was that the headline and ``detail`` were read off a
    *separate* cohort x post regression carrying one post dummy per
    cohort. That design is not saturated in cohort x period, so under
    dynamic effects the already-treated cohorts enter the period fixed
    effects and contaminate every coefficient. Fixed in 1.27.0; the test
    is strict from here on.
    """
    e_meta = r_reference["etwfe"]["meta"]
    if not e_meta.get("available", False):
        pytest.skip(f"R etwfe unavailable: {e_meta.get('error', '?')}")
    res = sp.wooldridge_did(
        data=cs_data,
        y="y",
        group="id",
        time="year",
        first_treat="first_treat",
    )
    py_att = float(res.estimate)
    r_att = r_reference["etwfe"]["estimate"]
    rel = abs(py_att - r_att) / abs(r_att)
    assert rel < 0.10, (
        f"sp.wooldridge_did drifted from R etwfe by {rel:.1%} "
        f"(Python={py_att:.4f}, R={r_att:.4f})"
    )


def test_wooldridge_did_recovers_known_cohort_atts(cs_data):
    """Cohort ATTs must recover the DGP's true per-cohort effects.

    The fixture's ``tau`` column carries the true effect, so the target is
    known without any external software: cohort 2 averages 3.0 over its
    five post periods, cohort 3 averages 2.5 over its four. The headline
    is the cohort-size-weighted average of the two, i.e. 2.75.

    Through 1.26.0 this returned 2.67 and 1.63 (headline 2.15) -- errors of
    11% and 35% against a *known* truth -- while the registry graded the
    function 'validated' on the strength of this very file. A known-truth
    tier with no passing known-truth assertion behind it is the failure
    mode this test exists to close.
    """
    res = sp.wooldridge_did(
        data=cs_data, y="y", group="id", time="year", first_treat="first_treat"
    )
    truth = (
        cs_data.loc[
            (cs_data["first_treat"] > 0) & (cs_data["year"] >= cs_data["first_treat"])
        ]
        .groupby("first_treat")["tau"]
        .mean()
    )
    detail = res.detail.set_index("cohort")["att"]
    for cohort, true_att in truth.items():
        got = float(detail.loc[int(cohort)])
        assert abs(got - true_att) < 0.15, (
            f"cohort {int(cohort)}: sp.wooldridge_did returned {got:.4f} "
            f"against a true ATT of {true_att:.4f}. The saturated "
            "cohort x period design recovers this DGP; a miss here means "
            "the design regressed to an unsaturated one (see the 1.27.0 "
            "CHANGELOG entry)."
        )
    assert abs(float(res.estimate) - 2.75) < 0.15


def test_gardner_close_to_truth(cs_data, r_reference):
    """Gardner converges; this is the 'green' member of the trio."""
    if not r_reference["gardner"]["meta"].get("available", False):
        pytest.skip("Gardner R reference unavailable")
    res = sp.gardner_did(
        data=cs_data, y="y", group="id", time="year", first_treat="first_treat"
    )
    assert abs(float(res.estimate) - 2.75) < 0.5


def test_fixture_metadata(r_reference):
    assert "bjs" in r_reference
    assert "gardner" in r_reference
    assert "etwfe" in r_reference
