"""Single source of truth for the parity systems' shared vocabulary.

Two independent subsystems used to keep private copies of the same two
tables and drifted apart:

* :mod:`statspai.registry` derived ``validation_status`` from a hand-written
  Track A alias table plus a scan of ``tests/reference_parity``.
* ``scripts/build_parity_index.py`` derived the parity grade from the
  committed Track A / reference-parity / external-parity artifacts, with a
  *second*, differently-populated alias table and its own non-estimator
  exclusion set.

The result was a reconciliation gap in both directions (the registry
under-stated 85 functions that had cross-language evidence and over-stated
3 that did not) and — worse — a circular citation: the index credited an
alias because "the registry marks it certified", while the registry marked
it certified because of its own hand-written table. Neither side held an
artifact. CLAUDE.md §10 forbids exactly that.

This module holds both tables once. Every alias entry names the pytest that
*proves* the equivalence on the committed Track A bytes, so "alias of a
certified module" is an auditable claim rather than an assertion.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

#: pytest that runs every entry of :data:`TRACK_A_ALIASES` against its
#: canonical entry point on the committed Track A CSV bytes.
ALIAS_PROOF_TEST = "tests/reference_parity/test_track_a_alias_equivalence.py"


class AliasProof:
    """One proven alias: same estimator core, same numbers, same bytes.

    ``legs`` records, per measured leg, the maximum relative deviation
    :data:`ALIAS_PROOF_TEST` actually observed when the entry was added, and
    the budget that leg must stay inside. Legs are separated because an
    alias can be exact on the point estimate and merely machine-precise on a
    clustered standard error: collapsing them into one number would hide
    which half of the claim is strong. A reader sees the strength of each
    half without running the suite, and a regression that stays inside the
    budget but degrades the observed value is still caught.
    """

    __slots__ = ("alias", "canonical", "module", "call", "legs", "note")

    def __init__(
        self,
        alias: str,
        canonical: str,
        module: str,
        call: str,
        legs: Dict[str, Tuple[float, float]],
        note: str = "",
    ) -> None:
        self.alias = alias
        self.canonical = canonical
        self.module = module
        self.call = call
        #: leg name -> (registered budget, observed maximum relative deviation)
        self.legs = legs
        self.note = note

    @property
    def rtol(self) -> float:
        """Loosest registered budget across this alias's legs."""
        return max(budget for budget, _ in self.legs.values())

    @property
    def observed(self) -> float:
        """Largest deviation actually measured across this alias's legs."""
        return max(seen for _, seen in self.legs.values())

    def evidence_note(self) -> str:
        return (
            f"Alias of {self.call} — proven equivalent on the committed "
            f"{self.module} bytes to rtol {self.rtol:g} (observed "
            f"{self.observed:g}) by {ALIAS_PROOF_TEST}."
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"AliasProof({self.alias!r} -> {self.canonical!r} @ {self.module})"


#: Standalone public functions that reach the *same* estimator core as a
#: Track A module's canonical entry point, and therefore inherit that
#: module's parity grade.
#:
#: Admission rule: an entry may only exist here if
#: :data:`ALIAS_PROOF_TEST` asserts the two entry points agree on the
#: module's committed CSV bytes. An alias that cannot be proven is not
#: demoted quietly — it is deleted from this table and the function falls
#: back to whatever its own evidence supports.
TRACK_A_ALIASES: Dict[str, AliasProof] = {
    "iv": AliasProof(
        alias="iv",
        canonical="ivreg",
        module="02_iv",
        call="sp.ivreg",
        legs={"coef": (1e-12, 0.0), "se": (1e-12, 0.0)},
        note=(
            "sp.ivreg is a thin keyword-normalising wrapper that returns "
            "iv(formula=..., data=..., robust=..., cluster=..., **kwargs) "
            "(src/statspai/regression/iv.py), so the agreement is exact by "
            "construction as well as by measurement."
        ),
    ),
    "hdfe_ols": AliasProof(
        alias="hdfe_ols",
        canonical="feols",
        module="03_hdfe + 15_hdfe_cluster",
        call="sp.fast.feols(ssc='fixest')",
        legs={
            "coef": (1e-12, 3.6e-15),
            "se_iid": (1e-12, 2.6e-15),
            "se_cluster": (1e-10, 1.9e-12),
        },
        note=(
            "Two different absorption implementations of the same within "
            "estimator, so agreement is at floating-point noise rather than "
            "exactly zero. The legs are budgeted separately because the "
            "mechanism differs: the coefficient and the iid standard error "
            "agree to ~3e-15 on both modules, while the CR1 clustered "
            "standard error reaches 1.9e-12 -- residual differences of order "
            "1e-15 are summed within cluster and squared in the sandwich "
            "meat matrix, which amplifies them by about three orders. This "
            "is floating-point accumulation in a shared estimator, not a "
            "convention difference, and it is four orders inside the loosest "
            "parity tier (1e-6)."
        ),
    ),
    "oaxaca": AliasProof(
        alias="oaxaca",
        canonical="decompose",
        module="30_oaxaca",
        call="sp.decompose('oaxaca')",
        legs={"components": (1e-12, 0.0)},
    ),
    "dfl_decompose": AliasProof(
        alias="dfl_decompose",
        canonical="decompose",
        module="31_dfl",
        call="sp.decompose('dfl')",
        legs={"components": (1e-12, 0.0)},
    ),
    "mediate": AliasProof(
        alias="mediate",
        canonical="mediation",
        module="36_mediation",
        call="sp.mediation",
        legs={"effects": (1e-12, 0.0)},
        note=(
            "The bootstrap leg is seeded, so the alias reproduces the "
            "canonical ACME/ADE and their standard errors exactly, not "
            "merely within Monte Carlo error."
        ),
    ),
}


#: Aliases that a previous release asserted and that measurement refuted.
#: Kept as a data record so the removal is auditable and cannot be
#: reintroduced by someone re-reading the old docstrings.
REFUTED_ALIASES: Dict[str, str] = {
    "wooldridge_did": (
        "Was credited with Track A module 17_etwfe as an alias of sp.etwfe. "
        "Measured on the committed 17_etwfe bytes with identical arguments, "
        "sp.wooldridge_did returns ATT -0.0378480795 (se 0.0058045845) while "
        "sp.etwfe returns -0.0351082766 (se 0.0069250918) under its default "
        "not-yet-treated control group and -0.0329765138 (se 0.0077660899) "
        "under cgroup='nevertreated' — 6.1% and 12.1% apart, and no setting "
        "reconciles them. They are different estimators (saturated cohort x "
        "post TWFE versus ETWFE with a not-yet-treated control group), so the "
        "alias claim was false and the certified grade it carried was "
        "withdrawn. sp.wooldridge_did keeps the grade its own evidence "
        "supports."
    ),
}


#: Leaves that a naive ``sp.<name>(`` scan of a parity test picks up but that
#: are not estimators — dataset loaders, DGP helpers, and calibration
#: fixtures. They appear in parity tests because they *build* the fixture,
#: not because anything about them was compared against R or Stata, so they
#: must never receive a parity grade or a validation tier.
NON_ESTIMATOR_LEAVES: FrozenSet[str] = frozenset(
    {
        # DGP helpers
        "dgp_bartik",
        "dgp_bunching",
        "dgp_cluster_rct",
        "dgp_did",
        "dgp_iv",
        "dgp_observational",
        "dgp_panel",
        "dgp_rct",
        "dgp_rd",
        "dgp_rd_2d",
        "dgp_rd_hte",
        "dgp_rd_kink",
        "dgp_rd_multi",
        "dgp_rdit",
        "dgp_synth",
        "dag_simulate",
        # Dataset loaders / calibrated replicas
        "list_datasets",
        "mpdta",
        "nsw_dw",
        "nsw_lalonde",
        "card_1995",
        "nhefs",
        "angrist_krueger_1991",
        "lee_2008_senate",
        "basque_terrorism",
        "california_prop99",
        "california_tobacco",
        "german_reunification",
        "chilean_households",
        "cps_wage",
        "mincer_wage_panel",
        # Introspection and rendering. A parity test calls these to check
        # metadata or resolve a citation, never to compare a number, so a
        # bare call-site scan credited them with the grade of whatever
        # estimator the test was actually about. `sp.bibtex` was recorded as
        # `external-replication` -- a citation resolver graded as if it had
        # reproduced published estimates.
        "describe_function",
        "function_schema",
        "list_functions",
        "search_functions",
        "bibtex",
        "bib_for",
        "citation",
        "replicate",
        "list_replications",
        "parity_status",
        "parity_matrix",
        "parity_summary",
    }
)


#: Registry categories whose members render, orchestrate or introspect
#: rather than estimate. Excluded from the estimator denominator.
#:
#: ``other`` and ``experimental`` are deliberately absent even though an
#: earlier hand-maintained inventory listed them: they are catch-alls
#: holding real estimators (``sp.ancova``, ``sp.geolift``, ``sp.negd``,
#: ``sp.attrition_test``, ``sp.optimal_design``), and excluding them would
#: have quietly shrunk the denominator that coverage is measured against.
INFRASTRUCTURE_CATEGORIES: FrozenSet[str] = frozenset(
    {
        "agent",
        "core",
        "datasets",
        "output",
        "plots",
        "smart",
        "utils",
        "validation",
        "workflow",
    }
)


#: Parity grades that mean "compared against a named external reference
#: implementation on identical inputs" (evidence tier T2), as opposed to
#: recovering a known truth with no second implementation in the loop.
CROSS_LANGUAGE_STATUSES: Tuple[str, ...] = ("bit-exact", "aligned")

#: The two T2 rows whose canonical reference implementation is Python, not
#: R or Stata.
#:
#: CLAUDE.md §5.1 says to follow the implementation the method's own authors
#: maintain, and for these two that implementation is a Python package --
#: comparing against an R port instead would be comparing against a bridge.
#: The evidence is therefore cross-*package* but not cross-*language*, and
#: any prose claiming "every certified symbol links to an R or Stata module"
#: has to name them. Pinned so the set cannot grow without the claim being
#: revisited (``tests/test_parity_index.py``).
PYTHON_REFERENCE_ROWS: Dict[str, str] = {
    "metalearner": "econml.metalearners (SLearner / TLearner)",
    "dml_sensitivity": "doubleml (Python) DoubleML.sensitivity_analysis",
}

#: Parity grades backed by an artifact that is *not* an external software
#: reference: a deterministic DGP whose population parameter is known (T1),
#: or published paper numbers reproduced on a calibrated replica.
INTERNAL_EVIDENCE_STATUSES: Tuple[str, ...] = (
    "analytical-only",
    "external-replication",
)


def validation_tier_for(status: str) -> str:
    """Map a parity grade onto the registry's ``validation_status`` tier.

    This is the *only* mapping between the two vocabularies. Both the
    registry and ``scripts/build_parity_index.py`` call it, so the two
    systems cannot disagree about a function by construction.
    """
    if status in CROSS_LANGUAGE_STATUSES:
        return "certified"
    if status in INTERNAL_EVIDENCE_STATUSES:
        return "validated"
    return "api_stable"
