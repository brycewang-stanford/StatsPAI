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
``sp.wooldridge_did`` was credited with module ``17_etwfe`` and is a
different estimator (see ``_parity_taxonomy.REFUTED_ALIASES``). The
refutation is asserted here too, so the claim cannot quietly come back.
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
    comment: if a future refactor really did merge the two estimators, this
    test fails and forces the alias question to be reopened deliberately.
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
    for cgroup, floor in (("notyet", 0.05), ("nevertreated", 0.10)):
        etwfe_fit = sp.etwfe(**kw, cgroup=cgroup)
        simple = sp.etwfe_emfx(etwfe_fit, type="simple", weighting="treated")
        gap = abs(saturated.estimate - simple.estimate) / abs(simple.estimate)
        assert gap > floor, (
            f"sp.wooldridge_did now agrees with sp.etwfe(cgroup={cgroup!r}) to "
            f"{gap:.3%}; the refuted-alias record in _parity_taxonomy must be "
            "revisited rather than left stale."
        )


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
