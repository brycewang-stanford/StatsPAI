"""Prove every Track A alias claim on the module's own committed bytes.

``statspai._parity_taxonomy.TRACK_A_ALIASES`` lets a standalone public
function inherit the parity grade of a Track A module that never calls it —
``sp.oaxaca`` inherits module ``30_oaxaca``, which is written in terms of
``sp.decompose('oaxaca')``. That inheritance is only legitimate if the two
entry points really do reach the same estimator core.

Before this file existed the claim was circular: ``build_parity_index.py``
credited the alias because "the registry marks it certified", and the
registry marked it certified because of its own hand-written alias table.
This suite replaces the circle with a measurement — each alias is run
against its canonical entry point on the *same committed CSV bytes* the
R and Stata goldens were computed from.

The audit that produced this file also refuted one standing alias:
``sp.wooldridge_did`` was credited with module ``17_etwfe`` while nothing
in that module ever called it (see ``_parity_taxonomy.REFUTED_ALIASES``).
It is not a *different estimator* — the 1.26.0 note saying so was wrong,
and the design it described as saturated was the defect fixed in 1.27.0.
It is the same extended-TWFE fit under a different documented headline
aggregation, which is why it cannot inherit ``sp.etwfe``'s number. The
refutation is asserted here too, so the claim cannot quietly come back;
module ``17_etwfe`` now calls ``sp.wooldridge_did`` directly instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._parity_taxonomy import REFUTED_ALIASES, TRACK_A_ALIASES

DATA = Path(__file__).resolve().parents[2] / "tests" / "r_parity" / "data"


def _max_rel(actual, expected) -> float:
    a = np.asarray(actual, dtype=float).ravel()
    b = np.asarray(expected, dtype=float).ravel()
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def _check(alias_key: str, leg: str, actual, expected) -> None:
    """Assert one leg of an alias reproduces the canonical numbers."""
    proof = TRACK_A_ALIASES[alias_key]
    budget, recorded = proof.legs[leg]
    observed = _max_rel(actual, expected)
    assert observed <= budget, (
        f"alias sp.{proof.alias} ({leg}) no longer reproduces {proof.call} on "
        f"the committed {proof.module} bytes: max relative deviation "
        f"{observed:.3e} exceeds the registered budget {budget:g}. Either the "
        f"alias diverged or it was never an alias — do not widen the budget to "
        f"make this pass (CLAUDE.md §5.1)."
    )
    # The recorded value is a floor on the claim's strength, not just
    # documentation: a silent regression that still fits inside the budget
    # would otherwise go unnoticed. Allow an order of magnitude of platform
    # noise on top of what was measured.
    floor = max(recorded * 10.0, 1e-14)
    assert observed <= floor, (
        f"alias sp.{proof.alias} ({leg}) still passes its budget but degraded "
        f"from the recorded {recorded:g} to {observed:.3e}; update "
        f"_parity_taxonomy.TRACK_A_ALIASES only with a stated reason."
    )


# --------------------------------------------------------------------------- #
#  02_iv — sp.iv  ==  sp.ivreg
# --------------------------------------------------------------------------- #
def test_iv_alias_of_ivreg() -> None:
    df = pd.read_csv(DATA / "02_iv.csv")
    formula = "lwage ~ exper + expersq + black + south + smsa + (educ ~ nearc4)"
    canonical = sp.ivreg(formula, data=df, robust="hc1")
    alias = sp.iv(formula, data=df, robust="hc1")
    _check("iv", "coef", alias.params.values, canonical.params.values)
    _check("iv", "se", alias.std_errors.values, canonical.std_errors.values)


# --------------------------------------------------------------------------- #
#  03_hdfe — sp.hdfe_ols  ==  sp.fast.feols(vcov='iid', ssc='fixest')
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", ["03_hdfe", "15_hdfe_cluster"])
def test_hdfe_ols_alias_of_feols(module: str) -> None:
    df = pd.read_csv(DATA / f"{module}.csv")
    formula = "y ~ x1 + x2 | firm + year"
    if module == "03_hdfe":
        canonical = sp.fast.feols(formula, data=df, vcov="iid", ssc="fixest")
        alias = sp.hdfe_ols(formula, data=df)
    else:
        canonical = sp.fast.feols(
            formula, data=df, vcov="cr1", cluster="firm", ssc="fixest"
        )
        alias = sp.hdfe_ols(formula, data=df, cluster="firm")
    names = ["x1", "x2"]
    _check(
        "hdfe_ols",
        "coef",
        [float(alias.params[n]) for n in names],
        [float(canonical.coef()[n]) for n in names],
    )
    # The clustered leg carries its own, looser budget: see the mechanism
    # note in _parity_taxonomy.TRACK_A_ALIASES["hdfe_ols"].
    _check(
        "hdfe_ols",
        "se_iid" if module == "03_hdfe" else "se_cluster",
        [float(alias.std_errors[n]) for n in names],
        [float(canonical.se()[n]) for n in names],
    )


# --------------------------------------------------------------------------- #
#  30_oaxaca — sp.oaxaca  ==  sp.decompose('oaxaca')
# --------------------------------------------------------------------------- #
def test_oaxaca_alias_of_decompose() -> None:
    df = pd.read_csv(DATA / "30_oaxaca.csv")
    kw = dict(data=df, y="log_wage", group="female", x=["educ", "exper"])
    canonical = sp.decompose("oaxaca", **kw)
    alias = sp.oaxaca(**kw)
    keys = ["gap", "explained", "unexplained", "explained_se", "unexplained_se"]
    _check(
        "oaxaca",
        "components",
        [float(alias.overall[k]) for k in keys],
        [float(canonical.overall[k]) for k in keys],
    )


# --------------------------------------------------------------------------- #
#  31_dfl — sp.dfl_decompose  ==  sp.decompose('dfl')
# --------------------------------------------------------------------------- #
def test_dfl_alias_of_decompose() -> None:
    df = pd.read_csv(DATA / "31_dfl.csv")
    kw = dict(data=df, y="log_wage", group="female", x=["educ", "exper"], reference=1)
    canonical = sp.decompose("dfl", **kw)
    alias = sp.dfl_decompose(**kw)
    keys = ["gap", "composition", "structure", "stat_a", "stat_b", "stat_cf"]
    _check(
        "dfl_decompose",
        "components",
        [float(getattr(alias, k)) for k in keys],
        [float(getattr(canonical, k)) for k in keys],
    )


# --------------------------------------------------------------------------- #
#  36_mediation — sp.mediate  ==  sp.mediation
# --------------------------------------------------------------------------- #
def test_mediate_alias_of_mediation() -> None:
    df = pd.read_csv(DATA / "36_mediation.csv")
    canonical = sp.mediation(df, y="y", d="treat", m="m")
    alias = sp.mediate(df, y="y", treat="treat", mediator="m")
    keys = ["acme", "ade", "total_effect", "prop_mediated", "se_acme"]
    _check(
        "mediate",
        "effects",
        [float(alias.model_info[k]) for k in keys],
        [float(canonical.model_info[k]) for k in keys],
    )


# --------------------------------------------------------------------------- #
#  Refuted alias — sp.wooldridge_did is NOT sp.etwfe
# --------------------------------------------------------------------------- #
def test_wooldridge_did_is_not_an_etwfe_alias() -> None:
    """Pin the measurement that withdrew the 17_etwfe alias claim.

    Asserting the *disagreement* keeps the refutation from decaying into a
    comment: if a future refactor really did merge the two headlines, this
    test fails and forces the alias question to be reopened deliberately.

    The gap is an aggregation difference, not a bug in either function:
    ``sp.wooldridge_did`` reports the cohort-size-weighted average of
    ``ATT(g)`` under a never-treated comparison group, while ``sp.etwfe``
    reports the treated-observation-weighted simple ATT that R
    ``emfx(type='simple')`` and Stata ``jwdid, estat simple`` report,
    under the not-yet-treated group by default. Both are checked against R
    by module ``17_etwfe``; neither is an alias of the other.
    """
    assert "wooldridge_did" in REFUTED_ALIASES
    assert "wooldridge_did" not in TRACK_A_ALIASES

    df = pd.read_csv(DATA / "17_etwfe.csv")
    kw = dict(
        data=df,
        y="lemp",
        group="countyreal",
        time="year",
        first_treat="first_treat",
    )
    saturated = sp.wooldridge_did(**kw)
    # Measured 15.9% (notyet) and 10.5% (nevertreated) at 1.27.0; the
    # floors sit well below so that ordinary numerical drift does not trip
    # the test, but a genuine merge of the two headlines does.
    for cgroup, floor in (("notyet", 0.05), ("nevertreated", 0.05)):
        etwfe_fit = sp.etwfe(**kw, cgroup=cgroup)
        simple = sp.etwfe_emfx(etwfe_fit, type="simple", weighting="treated")
        gap = abs(saturated.estimate - simple.estimate) / abs(simple.estimate)
        assert gap > floor, (
            f"sp.wooldridge_did now agrees with sp.etwfe(cgroup={cgroup!r}) to "
            f"{gap:.3%}; the refuted-alias record in _parity_taxonomy must be "
            "revisited rather than left stale."
        )


def test_wooldridge_did_reproduces_r_etwfe_group_aggregation() -> None:
    """The positive half: sp.wooldridge_did *is* R etwfe(cgroup='never').

    The 1.26.0 audit withdrew the alias and left sp.wooldridge_did with no
    cross-language evidence at all. It has some: on the committed 17_etwfe
    bytes its per-cohort ATTs reproduce R
    ``etwfe(cgroup='never', ivar=countyreal)`` + ``emfx(type='group')``.
    Those R numbers are committed as the ``att_group_never_*`` rows of
    module 17_etwfe, so this assertion and the Track A table cannot drift
    apart. Values inlined here are the R side, read from that golden.
    """
    import json

    golden = json.loads(
        (DATA.parent / "results" / "17_etwfe_R.json").read_text(encoding="utf-8")
    )
    expected = {
        int(row["statistic"].rsplit("_", 1)[1]): row
        for row in golden["rows"]
        if row["statistic"].startswith("att_group_never_")
    }
    assert expected, "17_etwfe_R.json carries no att_group_never_* rows"

    df = pd.read_csv(DATA / "17_etwfe.csv")
    res = sp.wooldridge_did(
        data=df,
        y="lemp",
        group="countyreal",
        time="year",
        first_treat="first_treat",
        cluster="countyreal",
    )
    got = res.detail.set_index("cohort")
    assert set(got.index.astype(int)) == set(expected)
    for cohort, row in expected.items():
        rel_est = abs(float(got.loc[cohort, "att"]) - row["estimate"]) / abs(
            row["estimate"]
        )
        rel_se = abs(float(got.loc[cohort, "se"]) - row["se"]) / row["se"]
        # Registered 17_etwfe budget: rel_est 1e-6, rel_se 1e-3.
        assert rel_est < 1e-6, f"cohort {cohort}: rel_est {rel_est:.2e}"
        assert rel_se < 1e-3, f"cohort {cohort}: rel_se {rel_se:.2e}"


def test_every_registered_alias_has_a_proof_here() -> None:
    """No alias may enter the taxonomy without a test in this file."""
    proven = {
        "iv",
        "hdfe_ols",
        "oaxaca",
        "dfl_decompose",
        "mediate",
    }
    assert set(TRACK_A_ALIASES) == proven, (
        "TRACK_A_ALIASES changed but tests/reference_parity/"
        "test_track_a_alias_equivalence.py was not updated. An alias without a "
        "proof here is an assertion, which is what this file exists to forbid."
    )


def test_wooldridge_did_headline_is_the_cohort_size_weighted_mean() -> None:
    """Close the evidence chain from the pinned cohort ATTs to the headline.

    No reference command prints ``sp.wooldridge_did``'s headline. R
    ``etwfe::emfx(type='group')`` and Stata ``jwdid, estat group`` both stop
    at the per-cohort vector, which is what module ``17_etwfe`` pins. The
    headline is therefore only as trustworthy as the *aggregation* applied
    to that pinned vector, and that aggregation has to be checked here
    rather than against an external number.

    Both fixtures used elsewhere in this suite are blind to it: on the
    ``17_etwfe`` bytes all three mpdta cohorts contribute 625 rows, and the
    staggered DGP's two cohorts are equal-sized too, so cohort-size weights
    and a plain unweighted mean coincide on both. This fixture makes the
    cohorts deliberately unequal (60 / 30 / 10 units), where the two
    averages differ by a wide margin, and asserts three things:

    * the weights are the cohort row counts, normalised;
    * the headline is exactly ``w @ ATT(g)`` -- an algebraic identity, so
      atol is numerical, not statistical;
    * the unweighted mean is far away, i.e. the test has power.
    """
    rng = np.random.default_rng(20260910)
    cohort_units = {2: 60, 3: 30, 4: 10}
    cohort_effect = {2: 1.0, 3: 4.0, 4: 9.0}
    n_periods = 6
    rows = []
    uid = 0
    for cohort, n_units in cohort_units.items():
        for _ in range(n_units):
            uid += 1
            alpha = rng.normal()
            for t in range(1, n_periods + 1):
                rows.append(
                    {
                        "id": uid,
                        "year": t,
                        "first_treat": cohort,
                        "y": alpha
                        + 0.3 * t
                        + (cohort_effect[cohort] if t >= cohort else 0.0)
                        + 0.01 * rng.normal(),
                    }
                )
    for _ in range(40):  # never-treated comparison group
        uid += 1
        alpha = rng.normal()
        for t in range(1, n_periods + 1):
            rows.append(
                {
                    "id": uid,
                    "year": t,
                    "first_treat": 0,
                    "y": alpha + 0.3 * t + 0.01 * rng.normal(),
                }
            )
    df = pd.DataFrame(rows)

    res = sp.wooldridge_did(
        data=df, y="y", group="id", time="year", first_treat="first_treat"
    )
    detail = res.detail.set_index("cohort")

    expected_rows = {g: n * n_periods for g, n in cohort_units.items()}
    for g, n_rows in expected_rows.items():
        assert int(detail.loc[g, "n_obs"]) == n_rows, (
            f"cohort {g}: n_obs {int(detail.loc[g, 'n_obs'])} is not the "
            f"cohort's row count {n_rows}; the headline weights are built "
            "from this column."
        )

    total = float(sum(expected_rows.values()))
    weights = np.array([expected_rows[g] / total for g in detail.index], dtype=float)
    atts = detail["att"].to_numpy(dtype=float)
    weighted = float(weights @ atts)
    unweighted = float(atts.mean())

    assert res.model_info["cohort_weighting"] == "cohort"
    assert res.model_info["cohort_weights"] == pytest.approx(
        {int(g): expected_rows[int(g)] / total for g in detail.index}, rel=1e-12
    )
    assert float(res.estimate) == pytest.approx(weighted, rel=1e-12, abs=1e-12), (
        f"headline {float(res.estimate):.10f} is not the cohort-size-weighted "
        f"mean {weighted:.10f} of the pinned ATT(g) vector"
    )
    # Power check: on this fixture the two averages are far apart, so the
    # identity above would fail loudly if the weighting silently changed.
    assert abs(weighted - unweighted) / abs(weighted) > 0.25, (
        "fixture lost its unbalanced cohorts -- weighted and unweighted "
        f"means are {weighted:.4f} vs {unweighted:.4f}, so this test can no "
        "longer tell them apart"
    )
