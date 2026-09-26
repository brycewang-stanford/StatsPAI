#!/usr/bin/env python3
"""Build the StatsPAI parity index from committed parity artifacts.

This script is the *single producer* of ``src/statspai/_parity_index.json``,
the frozen, queryable snapshot that backs :func:`statspai.parity_status` and
:func:`statspai.parity_matrix`.

Design contract (zero-hallucination, §10 of CLAUDE.md):
  Every field in every record traces to a committed artifact in this
  checkout — the 3-way Track A harness (``tests/r_parity/`` +
  ``tests/stata_parity/``), the pinned R environment (``renv.lock`` +
  per-run ``provenance``), and the pre-registered tolerance budget
  (``tests/r_parity/compare.py::TOLERANCES``).  Nothing is asserted from
  model memory; if an artifact is absent the function is honestly marked
  ``unverified``.

Status taxonomy (the user-facing parity grade):
  * ``bit-exact``           — matches a named R/Stata reference within the
                              strict tolerance tier (rel <= 1e-6); a
                              tolerance grade, not bitwise equality.
  * ``aligned``             — matches a named R/Stata reference within a
                              documented looser tolerance (iterative /
                              moderate / methodological tier).
  * ``analytical-only``     — recovers a known population parameter on a
                              deterministic DGP (no cross-package ref).
                              [populated in a later pass from
                              tests/reference_parity/]
  * ``external-replication``— reproduces published paper numbers.
                              [populated in a later pass from
                              tests/external_parity/]
  * ``unverified``          — registered, no qualifying numerical evidence.

Usage
-----
    python scripts/build_parity_index.py            # regenerate snapshot
    python scripts/build_parity_index.py --check     # CI drift check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Windows consoles default to cp1252, which cannot encode the status glyphs
# this script prints -- so it used to die with UnicodeEncodeError while
# reporting its own verdict, and a gate that exits non-zero for its output
# encoding is indistinguishable from a gate that found a real problem.
# Inlined rather than shared: ``scripts/`` is only on sys.path when a script is
# run directly, and the tests import some of these as ``scripts.<name>``.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statspai._parity_taxonomy import (  # noqa: E402
    ALIAS_PROOF_TEST,
    CROSS_LANGUAGE_STATUSES,
    INFRASTRUCTURE_CATEGORIES,
    INTERNAL_EVIDENCE_STATUSES,
    NON_ESTIMATOR_LEAVES,
    TRACK_A_ALIASES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
R_PARITY = REPO_ROOT / "tests" / "r_parity"
STATA_PARITY = REPO_ROOT / "tests" / "stata_parity"
REFERENCE_PARITY = REPO_ROOT / "tests" / "reference_parity"
EXTERNAL_PARITY = REPO_ROOT / "tests" / "external_parity"
SNAPSHOT = REPO_ROOT / "src" / "statspai" / "_parity_index.json"
DOC = REPO_ROOT / "docs" / "parity.md"

# Leaf function name from any ``sp.<a>.<b>.name`` mention (paren optional, so
# README cells like ``sp.fast.feols`` match).
_SP_LEAF_RE = re.compile(r"sp\.((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)")
# Call sites only (``sp.ipw(...)``) — excludes module refs (``sp.datasets``)
# and result-attribute access (``res.params``) when scanning test files.
_SP_CALL_RE = re.compile(r"sp\.((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)\s*\(")

# Family dispatchers: a Track A module exercises ONE method/variant, so the
# grade is variant-specific. We say so explicitly rather than let
# "synth: bit-exact" read as "every synth method is bit-exact".
_DISPATCHERS = {"synth", "decompose", "dml", "panel"}

# Dataset loaders / DGP helpers that the test scan picks up but that are not
# estimators — they must not receive a parity grade.
# The non-estimator exclusion set and the Track A alias table are shared with
# the registry via ``statspai._parity_taxonomy`` so the two systems cannot
# drift apart (they did: see the module docstring there).
_NON_ESTIMATOR_LEAVES = NON_ESTIMATOR_LEAVES


# Curated factor-level notes. Some estimators factor into a closed-form
# operator (exactly pinnable) and a stochastic component (not pinnable
# across implementations). The module's headline tolerance necessarily
# covers the looser factor, so the record would otherwise understate the
# evidence. Every string below is copied verbatim from the asserting test
# — no model-memory facts (CLAUDE.md §10).
_FACTOR_NOTES: Dict[str, Tuple[str, ...]] = {
    "causal_forest": (
        "Factored evidence: the AIPW operator -- the closed-form map from "
        "(Y, W, tau.hat, Y.hat, W.hat) to the score vector, point estimate "
        "and standard error -- is pinned exactly. Fed grf's own forest "
        "outputs, StatsPAI reproduces grf::get_scores elementwise to "
        "2.3e-14 and grf's reported ATE and ATT (estimate and std.err) to "
        "1e-15 (tests/reference_parity/test_grf_aipw_operator_parity.py). "
        "The module tolerance below covers the forest itself, which is not "
        "pinnable across implementations; its calibration is evidenced by "
        "the Track B coverage sweep.",
    ),
}

# Curated frozen-reference promotions: functions pinned to *exact* base-R or
# closed-form numbers in tests/reference_parity but NOT covered by the Track A
# harness. Every field is copied verbatim from
# tests/reference_parity/REFERENCES.md (the "Frozen R-value fixtures" table)
# and the asserting test — no model-memory facts (CLAUDE.md §10).
_FROZEN_PROMOTIONS: Dict[str, Dict[str, Any]] = {
    # ---- RD local randomization / power / multi-cutoff / honest CIs ----
    #
    # These six were graded `analytical-only` -- the grade whose published
    # definition is "no cross-package reference" -- while frozen fixtures
    # generated from the Cattaneo group's own R packages sat in
    # `_fixtures/` and tight equality assertions ran against them on every
    # test run. The evidence existed; only the bookkeeping was missing,
    # because `tests/reference_parity/REFERENCES.md` never listed these
    # three fixtures and this table is populated from that list. Under-
    # stating real evidence is the mirror image of over-claiming, and the
    # ledger has to be right in both directions.
    "rdrandinf": {
        "status": "bit-exact",
        "reference": "rdlocrand::rdrandinf 2.0 (Cattaneo, Titiunik & Vazquez-Bare)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "rdlocrand": "2.0",
        },
        "tolerance": "observed statistic & asymptotic p-value 1e-8 rel (observed 2.3e-15)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rdlocrand_parity.py",
            "tests/reference_parity/_fixtures/rdlocrand_R.json",
        ],
        "note": (
            "Frozen-R fixture across three windows x three statistics. The "
            "randomization p-value is deliberately NOT pinned -- it is a draw "
            "from R's Mersenne stream, which Python cannot reproduce, so "
            "equality there would be a test passing for the wrong reason; it "
            "is checked by its sampling behaviour instead. The deterministic "
            "quantities are what carry this grade. Regenerate via "
            "_generate_rdlocrand_R.R."
        ),
    },
    "rdwinselect": {
        "status": "bit-exact",
        "reference": "rdlocrand::rdwinselect 2.0 (Cattaneo, Titiunik & Vazquez-Bare)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "rdlocrand": "2.0",
        },
        "tolerance": (
            "window grid 1e-12 rel (observed 0); per-window counts Nl / Nr "
            "asserted as exact integer equality"
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rdlocrand_parity.py",
            "tests/reference_parity/_fixtures/rdlocrand_R.json",
        ],
        "note": (
            "The counts assertion is what makes this a data-dependent claim: "
            "the window grid alone is fixed by wmin/wstep and would agree "
            "with any sample. Adding it exposed that the function had no "
            "missing-data handling and was running on up to 20% more rows "
            "per window than rdlocrand; all twelve counts match exactly now. "
            "Balance p-values are randomization-based and excluded."
        ),
    },
    "rdpower": {
        "status": "bit-exact",
        "reference": "rdpower::rdpower 3.0 (Cattaneo, Titiunik & Vazquez-Bare)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "rdpower": "3.0",
        },
        "tolerance": "robust bias-corrected SE & power 1e-8 rel (observed 4.2e-14)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rdlocrand_parity.py",
            "tests/reference_parity/_fixtures/rdlocrand_R.json",
        ],
        "note": (
            "Data mode: R's rdpower(data=) is rdrobust's robust SE plus the "
            "power formula, so this pins the whole chain rather than the "
            "closed form on top of it. Three effect sizes."
        ),
    },
    "rdsampsi": {
        "status": "bit-exact",
        "reference": "rdpower::rdsampsi 3.0 (Cattaneo, Titiunik & Vazquez-Bare)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "rdpower": "3.0",
        },
        "tolerance": (
            "required sample sizes n_left / n_right / n_total asserted as "
            "exact integer equality (no tolerance)"
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rdlocrand_parity.py",
            "tests/reference_parity/_fixtures/rdlocrand_R.json",
        ],
        "note": (
            "Data mode, added in 1.27.0 so the reference call has a "
            "counterpart at all. Two details decide the answer and neither "
            "survives a tolerance band: the sample size is ceilinged inside "
            "the Newton-Raphson solve rather than at the end, and the sides "
            "are allocated by sqrt(variance) rather than by observed counts "
            "-- allocating by counts reproduces the total to ~1% while "
            "splitting the sides wrong."
        ),
    },
    "rd_honest": {
        "status": "bit-exact",
        "reference": "RDHonest::RDHonest 1.0.1.9000 (Armstrong & Kolesar)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "RDHonest": "1.0.1.9000",
        },
        "tolerance": (
            "estimate / std.error / maximum.bias / conf.low / conf.high "
            "1e-9 rel at fixed bandwidth; 1e-6 rel when the bandwidth and M "
            "are selected"
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rdhonest_parity.py",
            "tests/reference_parity/_fixtures/rdhonest_R.json",
        ],
        "note": (
            "The two tiers are the honest split: the fixed-bandwidth "
            "interval is a deterministic function of the design, while the "
            "selected bandwidth and curvature bound come from an "
            "optimisation whose convergence path differs across "
            "implementations."
        ),
    },
    "rdmc": {
        "status": "bit-exact",
        "reference": "rdmulti::rdmc 2.0.0 (Cattaneo, Titiunik, Vazquez-Bare & Keele)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "rdmulti": "2.0.0",
        },
        "tolerance": (
            "per-cutoff coefficients, robust coefficients, robust SEs and "
            "the pooled weighted estimate 1e-9 rel; selected bandwidths "
            "1e-5 rel"
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rdmulti_parity.py",
            "tests/reference_parity/_fixtures/rdmulti_R.json",
        ],
        "note": (
            "Pins the per-unit-cutoff estimator R's rdmc actually "
            "implements, in which cutoff c is identified only from the "
            "units assigned to c. Bandwidths carry the looser tier because "
            "they are selected rather than closed-form."
        ),
    },
    "panel_qtet": {
        "status": "bit-exact",
        "reference": "qte::panel.qtet 1.3.1 (Callaway & Li 2019)",
        "tolerance": (
            "all 19 quantiles: abs < 1e-8 (observed 6.8e-12); ATT abs < 1e-6. "
            "panel.qtet composes ordinary ecdf evaluations and type-7 "
            "quantiles, both of which have exact numpy equivalents, so this "
            "is machine-precision agreement rather than a tolerance band."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_panel_qtet_parity.py",
            "tests/reference_parity/_fixtures/qte_panel_qtet_nocov_R.json",
        ],
        "note": (
            "Frozen-R fixture on qte::lalonde.psid.panel. Was reported as "
            "analytical-only because the promotion table did not list it, "
            "not because the evidence was missing."
        ),
    },
    "qdid": {
        "status": "aligned",
        "reference": "qte::QDiD 1.3.1",
        "tolerance": (
            "max deviation / scale < 0.08, sign agreement on the large "
            "effects and correlation > 0.999. R's quantiles come from "
            "BMisc::weighted_quantile (stats::optimize on a piecewise-linear "
            "check function, which has plateaus where every point is a "
            "minimiser) while sp.qdid interpolates the empirical inverse "
            "CDF; on this fixture the gap is at most ~152 currency units "
            "against effects running to ~8900."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_qdid_parity.py",
            "tests/reference_parity/_fixtures/qte_lalonde_panel.csv",
        ],
        "note": (
            "Frozen-R fixture on qte::lalonde.psid.panel, post 1978 / pre "
            "1975. Documented optimiser-plateau convention gap, not a "
            "numerical failure."
        ),
    },
    "qte": {
        "status": "aligned",
        "reference": "qte::ci.qte / qte::ci.qtet 1.3.1 (Firpo 2007)",
        "tolerance": (
            "max relative deviation < 0.01 on lalonde.exp / lalonde.psid. "
            "Both sides minimise the same weighted check function; R's "
            "BMisc::weighted_quantile uses a golden-section search whose "
            "answer on a plateau is an optimiser artifact, so point-value "
            "equality is not asserted."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_firpo_qte_parity.py",
            "tests/reference_parity/_fixtures/qte_firpo_R.json",
        ],
        "note": (
            "Frozen-R fixture on the qte package's own lalonde.exp and "
            "lalonde.psid samples."
        ),
    },
    "cbps": {
        "status": "aligned",
        "reference": "CBPS::CBPS 0.24 (Imai & Ratkovic 2014)",
        "tolerance": (
            "ATE over/exact and ATT exact: rel <= 5e-3 (R's optimiser slack). "
            "ATT over is NOT pinned to R -- CBPS's ATT gradient mis-scales the "
            "balance block by n/n_1 and stops off-stationarity; StatsPAI is "
            "asserted to attain strictly better covariate balance instead."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_matching_r_parity.py",
            "tests/reference_parity/_fixtures/matching_R.json",
        ],
        "note": (
            "Frozen-R fixture on MatchIt::lalonde. Just-identified CBPS "
            "additionally balances to <1e-6 |SMD| where CBPS::CBPS leaves "
            "~1e-3. See tests/reference_parity/REFERENCES.md."
        ),
    },
    "ebalance": {
        "status": "bit-exact",
        "reference": "ebal::ebalance 0.2.1 (Hainmueller 2012)",
        "tolerance": "ATT rel <= 1e-5 (observed 3.2e-7); moment gap <= 1e-10",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_matching_r_parity.py",
            "tests/reference_parity/_fixtures/matching_R.json",
        ],
        "note": (
            "Frozen-R fixture on MatchIt::lalonde. StatsPAI solves the "
            "entropy-balancing dual to a true stationary point (relative "
            "moment gap ~1e-15) where ebal stops around 1e-7."
        ),
    },
    "cardinality_match": {
        "status": "analytical-only",
        "reference": (
            "Exact optimum of the documented integer program, solved "
            "independently in the test (scipy.optimize.milp / HiGHS)"
        ),
        "tolerance": (
            "Feasibility exact (max |SMD| <= smd_tolerance in 12 of 12 "
            "seed x tolerance cells) and optimality exact (selected count "
            "equals the independent optimum in all 12)."
        ),
        "sides": ["py"],
        "test": ["tests/reference_parity/test_cardinality_match_parity.py"],
        "note": (
            "Deliberately not a cross-package grade: designmatch::cardmatch "
            "solves a different program (it selects both arms; this one "
            "keeps every treated unit and chooses controls), so agreement "
            "with it would not mean anything. Promoted from 'unverified' "
            "on the strength of an exact-optimum oracle, which also caught "
            "the LP-relaxation defect fixed in v1.22 -- the previous "
            "implementation breached its own tolerance in 9 of 12 cells."
        ),
    },
    "overlap_weights": {
        "status": "bit-exact",
        "reference": "WeightIt::weightit 1.7.0 (method='glm'), R 4.5.2",
        "reference_versions": {"R": "4.5.2", "WeightIt": "1.7.0"},
        "tolerance": (
            "All four estimands of the shared-propensity family (Li, Li & "
            "Li 2019 Table 1) relative to WeightIt: ATO 2.5e-14, ATE "
            "4.1e-14, ATT 2.3e-14, ATC 4.1e-14. The propensity score itself "
            "matches R glm(family=binomial) to 2.6e-14 absolute."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_overlap_weights_r_parity.py",
            "tests/reference_parity/_fixtures/overlap_weights_R.json",
        ],
        "note": (
            "Frozen-R fixture. Promoting this required fixing the estimator: "
            "overlap_weights fitted a *penalised* logit "
            "(sklearn LogisticRegression C=1e6) while the rest of the "
            "matching module used the unpenalised MLE, so one package gave "
            "two propensity scores for the same specification. The "
            "overlap-weight exact-balance property is derived at the "
            "unpenalised score equations, so this was a theory mismatch, not "
            "only a parity gap."
        ),
    },
    "sqreg": {
        "status": "bit-exact",
        "reference": "R quantreg::rq (Barrodale-Roberts), Koenker 2005",
        "reference_versions": {"quantreg": "see sqreg_R.json provenance"},
        "tolerance": (
            "Coefficients 3.5e-14 against quantreg::rq at tau = 0.25 / 0.50 "
            "/ 0.75 -- both sides minimise the same pinball loss with the "
            "same simplex. Standard errors differ from R's se='iid' by ONE "
            "SCALAR PER QUANTILE, constant across coefficients to 6e-16: "
            "the sandwich is identical and only the sparsity estimate "
            "1/f(0) differs (Powell kernel here, Koenker-Bassett with a "
            "Siddiqui/Hall-Sheather bandwidth there). The test asserts the "
            "ratio's constancy rather than a numerical band, which a "
            "structural difference could not satisfy. R's default se='nid' "
            "(Hendricks-Koenker, also Stata qreg's) is a third convention "
            "and is recorded as one."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_sqreg_parity.py"],
        "note": (
            "Promoted in 1.27.0 by removing a defect rather than by "
            "loosening anything: sp.sqreg rounded its returned coefficients "
            "to four decimals -- in the value, not for display -- which "
            "capped agreement at 7.8e-03 on a coefficient of order 1e-3. "
            "The previous test recorded that ceiling as the estimator's "
            "accuracy and stayed at the analytical tier because of it."
        ),
    },
    "centrality": {
        "status": "aligned",
        "reference": "R igraph degree / betweenness / closeness / page_rank / eigen_centrality",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "degree, betweenness (normalised and raw), closeness and PageRank at 1e-10 on Zachary's karate club. The eigenvector column is L2-normalised (networkx) where igraph max-scales it: the ratio is constant across nodes to 1e-14, so it is the same vector under a documented normalisation."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "closeness_centrality": {
        "status": "bit-exact",
        "reference": "Wasserman-Faust closeness from R igraph::distances",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "Exact (0.0) on a disconnected graph with three blocks and three isolates -- the case the correction exists for; the connected-graph values also match igraph::closeness(normalized = TRUE) to 1e-10."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "pagerank": {
        "status": "bit-exact",
        "reference": "R igraph::page_rank (damping 0.85)",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": ("5e-12 undirected, 1e-12 directed."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "transitivity": {
        "status": "bit-exact",
        "reference": "R igraph::transitivity(type = 'global')",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": ("Exact on karate."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "assortativity": {
        "status": "bit-exact",
        "reference": "R igraph::assortativity_degree",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": ("1.2e-16 on karate."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "reciprocity": {
        "status": "bit-exact",
        "reference": "R igraph::reciprocity",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": ("Exact on a 40-node directed graph."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "network_modularity": {
        "status": "bit-exact",
        "reference": "R igraph::modularity",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "Exact for a fixed split and for igraph's own fast-greedy partition."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "network_components": {
        "status": "bit-exact",
        "reference": "R igraph::components",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "Counts and sizes exact on a disconnected graph; weak and strong counts on the directed graph."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "network_summary": {
        "status": "bit-exact",
        "reference": "R igraph edge_density / diameter / mean_distance / transitivity",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "Density, diameter, mean path length, transitivity and assortativity exact; average clustering matches igraph::transitivity(type = 'average', isolates = 'zero'), the convention used here."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "bonacich_power": {
        "status": "bit-exact",
        "reference": "R igraph::power_centrality and sna::bonpow",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": ("3.9e-16 against igraph and 5.8e-16 against sna at beta = 0.1."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "katz_centrality": {
        "status": "bit-exact",
        "reference": "R igraph::alpha_centrality",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "7.1e-16 with normalized = False (normalized = True L2-scales the same vector)."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "netlm": {
        "status": "bit-exact",
        "reference": "R sna::netlm",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "Coefficients 2e-15 directed and undirected. QAP p-values are permutation draws and are not compared."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "netlogit": {
        "status": "bit-exact",
        "reference": "R sna::netlogit",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "Coefficients 1.4e-9 (IRLS on both sides). QAP p-values are permutation draws and are not compared."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "ergm": {
        "status": "bit-exact",
        "reference": "R ergm::ergm(estimate = 'MPLE')",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "Coefficients 2e-16 to 7e-13 for edges + triangle + nodematch + nodecov + absdiff (undirected) and edges + mutual (directed); standard errors 2e-8 (directed) and <= 3.2e-7 (undirected), inside the 1e-6 budget."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "dyadic_regression": {
        "status": "bit-exact",
        "reference": "R dyadRobust (Aronow-Samii-Assenova dyadic-robust variance)",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "Coefficients 7e-16 and standard errors 2e-15 on undirected and directed dyads, after the 1.28.0 fix to the shared-member weighting; also asserted against a brute-force construction of the definition."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "karate_club": {
        "status": "bit-exact",
        "reference": "R igraph::make_graph('Zachary')",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": ("Adjacency matrix identical."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "florentine_families": {
        "status": "bit-exact",
        "reference": "R ergm flomarriage",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "The 20 marriage ties are identical edge for edge; the Pucci isolate is omitted (15 nodes against 16), which is documented."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "hits": {
        "status": "aligned",
        "reference": "R igraph::hits_scores",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "Hub and authority vectors are igraph's up to normalisation: L1 here (documented), max = 1 in igraph; the ratio is constant across nodes to 1e-11."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "community_detection": {
        "status": "aligned",
        "reference": "R igraph::cluster_louvain (T3: randomised on both sides)",
        "reference_versions": {
            "igraph": "2.3.3",
            "sna": "2.8",
            "ergm": "4.12.0",
            "dyadRobust": "0.0.1.0001",
        },
        "tolerance": (
            "T3, not T2: 200 seeded runs on karate have mean modularity within four combined standard errors of igraph's 200 runs (0.4157 vs 0.4145), and both reach the same maximum, 0.41979, the known optimum for this graph."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_network_parity.py"],
        "note": (
            "Added in 1.28.0 by the network sweep. One defect: sp.dyadic_regression weighted each pair of dyads by the NUMBER of members they share instead of whether they share one, double-counting (i, j) with (j, i) on directed data (1.8% on the SEs). One reference bug: dyadRobust recodes ego / alter inside a single dplyr::mutate(), whose sequential evaluation builds the alter codes from the already-recoded ego column; the fixture passes ids for which that recode is the identity and asserts the precondition. Louvain is compared as a 200-seed distribution -- a single-seed comparison looked like a 1% shortfall and was not one."
        ),
    },
    "moran": {
        "status": "bit-exact",
        "reference": "R spdep::moran.test (randomisation null)",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": (
            "I 1.9e-15, expectation, variance and z all at 1e-15 on the row-standardised lattice."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "moran_local": {
        "status": "bit-exact",
        "reference": "R spdep::localmoran",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": ("Every Ii at 8.1e-15."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "geary": {
        "status": "bit-exact",
        "reference": "R spdep::geary.test",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": (
            "C 2.0e-15. The closed-form variance and z are new in 1.27.0 (they were NaN whenever permutations=0) and match both spdep nulls at 1.5e-14: randomisation (with the m4/m2^2 term) and normality."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "getis_ord_g": {
        "status": "bit-exact",
        "reference": "R spdep::globalG.test (binary weights)",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": (
            "G 5.4e-16. Binary weights, which is what spdep recommends for this statistic."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "getis_ord_local": {
        "status": "bit-exact",
        "reference": "R spdep::localG",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": (
            "Gi* 1.7e-14; Gi 4.3e-13 after the star=False branch stopped borrowing Gi*'s standardisation."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "join_counts": {
        "status": "bit-exact",
        "reference": "R spdep::joincount.multi (binary weights)",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": (
            "BB, WW and BW all exact. A reference-free guard also asserts BB + WW + BW = S0/2, the identity the BW defect violated (70.75 against 50)."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "lm_tests": {
        "status": "bit-exact",
        "reference": "R spdep::lm.RStests",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": (
            "All five statistics and their p-values at 1e-9. Before the fix: LM_err 39.47 against 19.58, and Robust_LM_err 20.49 (p=6e-6) against 0.0397 (p=0.84) -- the Anselin lag-vs-error decision rule, reversed."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "moran_residuals": {
        "status": "bit-exact",
        "reference": "R spdep::lm.morantest",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": (
            "Statistic 5e-16; the p-value at 1e-7 once X is supplied so the Cliff-Ord regression-residual null can be formed. Both spdep alternatives are recorded because lm.morantest defaults to one-sided."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "slx": {
        "status": "bit-exact",
        "reference": "R spatialreg::lmSLX",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": ("Every coefficient at 1e-10."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "sac": {
        "status": "bit-exact",
        "reference": "R spatialreg::sacsarlm",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": (
            "rho and lambda at 1e-5, slope coefficients at 1e-6 -- a bounded two-parameter ML line search on both sides."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "impacts": {
        "status": "bit-exact",
        "reference": "R spatialreg::impacts on a lagsarlm fit",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": (
            "Direct, indirect and total at 1e-6, inheriting the SAR rho's own agreement."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "knn_weights": {
        "status": "bit-exact",
        "reference": "R spdep::knearneigh + knn2nb",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": (
            "Neighbour sets identical for all 120 points, k=4, on a random point set chosen so no distance ties make the answer non-unique."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "distance_band": {
        "status": "bit-exact",
        "reference": "R spdep::dnearneigh",
        "reference_versions": {"spdep": "1.4.2", "spatialreg": "1.4.3"},
        "tolerance": ("Neighbour sets identical for all 120 points at a 0.25 radius."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_spdep_parity.py"],
        "note": (
            "Added in 1.27.0 by the spdep sweep, which found four correctness defects on the way: sp.lm_tests computed the wrong trace term AND built the lag statistic from W y instead of W X beta (both times the comment above the line named the right formula); sp.join_counts halved the double sum for BB and WW but not BW, breaking BB + WW + BW = S0/2; sp.getis_ord_local(star=False) standardised Gi with Gi*'s whole-sample moments; and sp.moran_residuals used the raw-variable null for OLS residuals. Weight style is not incidental here -- sp.W is binary until w.transform = 'R' while spdep's nb2listw defaults to row-standardised, and the fixture emits both so each statistic is compared under the style its reference uses."
        ),
    },
    "anderson_rubin_test": {
        "status": "bit-exact",
        "reference": "R ivmodel::AR.test",
        "reference_versions": {"ivmodel": "1.9.1", "car": "3.1.5", "metafor": "5.0.1"},
        "tolerance": (
            "Statistic 1.1e-15, p-value 1.2e-13, degrees of freedom exact, and the analytic AR confidence set 1.2e-14."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_weakiv_meta_parity.py"],
        "note": (
            "Added in 1.27.0 by the weak-IV / diagnostics sweep, which found two defects: sp.vif returned VIF rounded to two decimals and 1/VIF to four (in the frame, not a display -- the conventional threshold of 10 was being decided in the fourth significant digit), and the grid-inversion confidence sets reported the extreme grid point still inside the acceptance region as the endpoint, biasing every interval inward by up to one grid step. sp.anderson_rubin_test computed the same AR interval analytically all along, so the package disagreed with itself about one quantity by 8e-3."
        ),
    },
    "anderson_rubin_ci": {
        "status": "bit-exact",
        "reference": "R ivmodel::AR.test confidence set",
        "reference_versions": {"ivmodel": "1.9.1", "car": "3.1.5", "metafor": "5.0.1"},
        "tolerance": (
            "Both endpoints at 5e-15 after the boundary bisection replaced the grid-point endpoints (previously 8.1e-3 / 4.9e-3). A reference-free test also asserts the two AR entry points agree with each other and that neither endpoint lands exactly on a grid node."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_weakiv_meta_parity.py"],
        "note": (
            "Added in 1.27.0 by the weak-IV / diagnostics sweep, which found two defects: sp.vif returned VIF rounded to two decimals and 1/VIF to four (in the frame, not a display -- the conventional threshold of 10 was being decided in the fourth significant digit), and the grid-inversion confidence sets reported the extreme grid point still inside the acceptance region as the endpoint, biasing every interval inward by up to one grid step. sp.anderson_rubin_test computed the same AR interval analytically all along, so the package disagreed with itself about one quantity by 8e-3."
        ),
    },
    "vif": {
        "status": "bit-exact",
        "reference": "R car::vif",
        "reference_versions": {"ivmodel": "1.9.1", "car": "3.1.5", "metafor": "5.0.1"},
        "tolerance": (
            "2.0e-16 on every variance inflation factor, once the returned values stopped being rounded to two decimals."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_weakiv_meta_parity.py"],
        "note": (
            "Added in 1.27.0 by the weak-IV / diagnostics sweep, which found two defects: sp.vif returned VIF rounded to two decimals and 1/VIF to four (in the frame, not a display -- the conventional threshold of 10 was being decided in the fourth significant digit), and the grid-inversion confidence sets reported the extreme grid point still inside the acceptance region as the endpoint, biasing every interval inward by up to one grid step. sp.anderson_rubin_test computed the same AR interval analytically all along, so the package disagreed with itself about one quantity by 8e-3."
        ),
    },
    "meta_analysis": {
        "status": "bit-exact",
        "reference": "R metafor::rma (method='FE' and 'DL')",
        "reference_versions": {"ivmodel": "1.9.1", "car": "3.1.5", "metafor": "5.0.1"},
        "tolerance": (
            "6.2e-16 across all nine reported quantities: fixed and random pooled effect and standard error, tau^2, Cochran Q and its p-value, I^2 and H^2. REML is not implemented here, which is a capability gap rather than a disagreement."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_weakiv_meta_parity.py"],
        "note": (
            "Added in 1.27.0 by the weak-IV / diagnostics sweep, which found two defects: sp.vif returned VIF rounded to two decimals and 1/VIF to four (in the frame, not a display -- the conventional threshold of 10 was being decided in the fourth significant digit), and the grid-inversion confidence sets reported the extreme grid point still inside the acceptance region as the endpoint, biasing every interval inward by up to one grid step. sp.anderson_rubin_test computed the same AR interval analytically all along, so the package disagreed with itself about one quantity by 8e-3."
        ),
    },
    "conditional_lr_ci": {
        "status": "aligned",
        "reference": "R ivmodel::CLR",
        "reference_versions": {"ivmodel": "1.9.1"},
        "tolerance": (
            "Monte Carlo by construction and graded as such: ivmodel "
            "integrates Moreira's conditional distribution while StatsPAI "
            "simulates it, so the two cannot agree deterministically. The "
            "test asserts the error SHRINKS with n_sim rather than pinning "
            "a number -- 3.8e-3 at n_sim=5,000 and 1.7e-4 at 200,000 -- "
            "which is the only honest statement about a simulated critical "
            "value. The endpoints themselves are bisected off-grid, so the "
            "residual is the critical value and not the grid."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_weakiv_meta_parity.py"],
        "note": (
            "Added in 1.27.0 by the weak-IV / diagnostics sweep, which found two defects: sp.vif returned VIF rounded to two decimals and 1/VIF to four (in the frame, not a display -- the conventional threshold of 10 was being decided in the fourth significant digit), and the grid-inversion confidence sets reported the extreme grid point still inside the acceptance region as the endpoint, biasing every interval inward by up to one grid step. sp.anderson_rubin_test computed the same AR interval analytically all along, so the package disagreed with itself about one quantity by 8e-3."
        ),
    },
    "panel_fgls": {
        "status": "bit-exact",
        "reference": "Stata 18 xtgls, panels(hetero)",
        "reference_versions": {"Stata": "18 MP"},
        "tolerance": (
            "6.3e-16 on every coefficient and 7.0e-16 on every standard error against the two-step default, and 4.7e-08 against `xtgls, igls` for the iterated variant. A reference-free test also asserts the two are distinct estimators, so a silent return to iterating fails."
        ),
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_panel_stata_parity.py"],
        "note": (
            "Added in 1.27.0 by the panel sweep, which found two defects. sp.panel_fgls iterated its variance estimates unconditionally -- that is Stata's `igls` option, not the default xtgls -- while its docstring claimed equivalence to the plain command (2.8% on the slope); the default is now two-step and `igls=True` keeps the old estimator under its own name. sp.panel_logit / sp.panel_probit with method='re' built their design from the regressor list alone, which is right for conditional FE logit and wrong for RE, so both were fitted with NO intercept and every slope was biased. What identified it was the gap's stubbornness -- 0.39% at 12 and at 30 quadrature points -- rather than its size."
        ),
    },
    "panel_logit": {
        "status": "bit-exact",
        "reference": "Stata 18 xtlogit, re",
        "reference_versions": {"Stata": "18 MP"},
        "tolerance": (
            "Graded by CONVERGENCE rather than a fixed tolerance: Stata integrates adaptively and StatsPAI does not, so the honest claim is that agreement improves as the Gauss-Hermite rule is refined. Observed 3.8e-04 at 12 points and 2.4e-07 at 60, with the log-likelihood at 2.1e-08 -- the sharpest single check, since it is the same objective evaluated at the same optimum. sigma_u and rho at 1e-4."
        ),
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_panel_stata_parity.py"],
        "note": (
            "Added in 1.27.0 by the panel sweep, which found two defects. sp.panel_fgls iterated its variance estimates unconditionally -- that is Stata's `igls` option, not the default xtgls -- while its docstring claimed equivalence to the plain command (2.8% on the slope); the default is now two-step and `igls=True` keeps the old estimator under its own name. sp.panel_logit / sp.panel_probit with method='re' built their design from the regressor list alone, which is right for conditional FE logit and wrong for RE, so both were fitted with NO intercept and every slope was biased. What identified it was the gap's stubbornness -- 0.39% at 12 and at 30 quadrature points -- rather than its size."
        ),
    },
    "panel_probit": {
        "status": "bit-exact",
        "reference": "Stata 18 xtprobit, re",
        "reference_versions": {"Stata": "18 MP"},
        "tolerance": (
            "Same convergence grading: 8.0e-05 at 12 quadrature points and 4.0e-08 at 60, log-likelihood at 1.7e-09, sigma_u and rho at 1e-4."
        ),
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_panel_stata_parity.py"],
        "note": (
            "Added in 1.27.0 by the panel sweep, which found two defects. sp.panel_fgls iterated its variance estimates unconditionally -- that is Stata's `igls` option, not the default xtgls -- while its docstring claimed equivalence to the plain command (2.8% on the slope); the default is now two-step and `igls=True` keeps the old estimator under its own name. sp.panel_logit / sp.panel_probit with method='re' built their design from the regressor list alone, which is right for conditional FE logit and wrong for RE, so both were fitted with NO intercept and every slope was biased. What identified it was the gap's stubbornness -- 0.39% at 12 and at 30 quadrature points -- rather than its size."
        ),
    },
    "etregress": {
        "status": "bit-exact",
        "reference": "Stata 18 MP official `etregress` (Maddala 1983 model)",
        "reference_versions": {"Stata": "18 MP"},
        "tolerance": (
            "Two-step: 5e-9 on every coefficient and every standard error, "
            "including the Heckman correction for the estimated first "
            "stage. ML: the likelihood, score and observed information are "
            "pinned at 9e-11 -- our Hessian reproduces Stata's reported "
            "standard errors when evaluated at Stata's own parameter "
            "vector, which is independent of either optimiser. At our own "
            "optimum the parameters sit within 2e-5 of Stata's; that gap "
            "is the two optimisers' stopping points, not a formula "
            "difference, and StatsPAI's stops at the HIGHER log-likelihood "
            "with a gradient ~300x smaller (asserted, so a regression that "
            "makes our optimum worse fails even though the 1e-4 parity "
            "assertions would still pass). vce(robust) carries Stata's "
            "N/(N-1) meat factor and vce(cluster) its g/(g-1)."
        ),
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_etregress_stata_parity.py"],
        "note": (
            "Added in 1.27.0 alongside three correctness fixes this "
            "comparison found: `method='mle'` ran a verbatim copy of the "
            "two-step branch (10.7% on the treatment effect), the two-step "
            "reported uncorrected OLS standard errors (11.2% too small), "
            "and `robust=` / `cluster=` were accepted and never used. The "
            "pre-existing analytical test could not see any of it -- a "
            "two-step also recovers delta on a known DGP."
        ),
    },
    "psmatch2": {
        "status": "bit-exact",
        "reference": (
            "Stata 18 MP + psmatch2 4.0.12 / pstest 4.2.2 (Leuven & Sianesi 2003)"
        ),
        "reference_versions": {"Stata": "18 MP", "psmatch2": "4.0.12"},
        "tolerance": (
            "Observed py<->Stata relative gaps on the committed fixtures: "
            "nearest-neighbour ATT 1.2e-16 and its analytic SE exactly 0 "
            "(as is _weight per row); radius ATT 2.8e-16; Abadie-Imbens "
            "ai(1)/ai(2) SE 3.0e-15/1.7e-15; PSM-DID across all five weight "
            "regimes 2.0e-14; pstest per-covariate rows 1.3e-14; "
            "Mahalanobis ATT 1.2e-13; llr ATT 2.1e-10 (worst of four "
            "kernels); kernel ATT 9.1e-10; pstest summary block 1.3e-9. "
            "The two loosest rows are bounded by fixture precision, not by "
            "the estimator: pstest accumulates MeanBias/MedBias in a Stata "
            "float (2.6e-9) and r(seatt) for the llr reroute was captured "
            "at 8 significant digits (9.9e-9)."
        ),
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_psmatch2_parity.py",
            "tests/reference_parity/test_psmatch2_llr_parity.py",
            "tests/reference_parity/test_pstest_parity.py",
            "tests/reference_parity/test_psmdid_weight_parity.py",
        ],
        "note": (
            "Five frozen Stata fixtures, each regenerable from a committed "
            ".do file, covering nearest-neighbour / kernel / radius / local "
            "linear regression / Mahalanobis matching, the psmatch2 and "
            "Abadie-Imbens standard errors, the full pstest table, and the "
            "PSM-DID aweight / fweight / unweighted regimes. Reported as "
            "analytical-only before v1.22 because the promotion table did "
            "not list it, not because the evidence was missing. The "
            "fixtures also pin three psmatch2 behaviours that differ from "
            "what its option names imply -- see "
            "tests/reference_parity/REFERENCES.md."
        ),
    },
    "match": {
        "status": "bit-exact",
        "reference": "MatchIt::matchit 4.7.2 (nearest, glm/logit distance)",
        "tolerance": (
            "1:1 and 2:1 PS matching without replacement: rel <= 1e-9. "
            "Mahalanobis metric pinned against MatchIt:::mahalanobis_dist "
            "(rel <= 1e-10); greedy m_order='data'/'closest' rel <= 1e-9."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_matching_r_parity.py",
            "tests/reference_parity/_fixtures/matching_R.json",
        ],
        "note": (
            "Frozen-R fixture on MatchIt::lalonde. With replacement "
            "(ties='all', tie_tolerance=1e-5) the ATT and the Abadie-Imbens "
            "population SE (se_method='abadie_imbens_pop') also match "
            "Matching::Match 4.10-15 at M = 1 and M = 3. m_order='farthest' "
            "and the bias-correction regression convention remain "
            "documented parity boundaries (see the sp.match registry "
            "limitations), not pinned rows."
        ),
    },
    "sbw": {
        "status": "bit-exact",
        "reference": "sbw::sbw 1.2 (Zubizarreta 2015), quadprog solver",
        "tolerance": (
            "ATT rel <= 1e-8 (observed <= 4e-10) under both standardisation "
            "conventions: tolerance_scale='target' == bal_std='target' and "
            "'group' == bal_std='group', at bal_tol 0.05 and 0.02."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_matching_r_parity.py",
            "tests/reference_parity/_fixtures/matching_R.json",
        ],
        "note": (
            "Frozen-R fixture on MatchIt::lalonde. The balance tolerance is "
            "only reproducible together with the standard deviation it is "
            "quoted in; both sbw::sbw conventions are exposed."
        ),
    },
    "genmatch": {
        "status": "aligned",
        "reference": "Matching::Match 4.10-15 (Weight = 3, Weight.matrix)",
        "tolerance": (
            "Deterministic kernel only: given the same diagonal W, the 1-NN "
            "assignment agrees with Matching::Match on all 163 uniquely "
            "matched treated units on MatchIt::lalonde."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_matching_r_parity.py",
            "tests/reference_parity/_fixtures/matching_R.json",
        ],
        "note": (
            "The genetic search is stochastic and is not reproducible "
            "across languages, so what is pinned is the generalised "
            "distance + assignment kernel it optimises over."
        ),
    },
    "optimal_match": {
        "status": "aligned",
        "reference": "optmatch::pairmatch 0.10.8 on a logit propensity score",
        "tolerance": (
            "Total matched distance <= optmatch's (1 + 1e-6). The matched "
            "pairs are not pinned: the assignment problem is degenerate on "
            "this data, so equally optimal solutions report different ATTs."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_matching_r_parity.py",
            "tests/reference_parity/_fixtures/matching_R.json",
        ],
        "note": (
            "Frozen-R fixture on MatchIt::lalonde. StatsPAI solves the "
            "assignment problem exactly (Hungarian); optmatch discretises "
            "distances for its network-flow solver."
        ),
    },
    "metalearner": {
        "status": "bit-exact",
        "reference": "econml.metalearners SLearner / TLearner / XLearner",
        "tolerance": (
            "S / T / X conditional-average-treatment-effect vectors match "
            "econml elementwise to 1e-12 absolute (observed <= 1.1e-15) when "
            "both fitting stages use the same base learner"
        ),
        "sides": ["py"],
        "test": ["tests/external_parity/test_metalearner_econml_parity.py"],
        "note": (
            "Scope: the grade covers the CATE functions, which is what "
            "learner= selects. It does NOT cover result.estimate, which "
            "sp.metalearner reports as a doubly-robust AIPW ATE "
            "(model_info['ate_method'] == 'aipw_dr_pseudo_outcome') and "
            "which is deliberately invariant to learner=; that quantity's "
            "evidence is the CausalML-book replication in "
            "tests/external_parity/test_causalml_book.py. The DR-learner is "
            "not pinned elementwise and cannot be: StatsPAI fits the outcome "
            "nuisance per arm while econml's DRLearner fits one joint "
            "regression on [X, T], so with a linear learner the two agree "
            "only when the treatment effect is constant (max elementwise gap "
            "5.2e-2 under a heterogeneous effect, 4.7e-3 under a constant "
            "one, on a shared fold partition). What is pinned for DR is the "
            "operator: given the cross-fitted nuisances StatsPAI actually "
            "used, the pseudo-outcome is asserted to equal the AIPW score to "
            "1e-12, and the per-arm-vs-joint mechanism is itself asserted. "
            "The R-learner is not pinned."
        ),
    },
    "dml_sensitivity": {
        "status": "bit-exact",
        "reference": "doubleml (Python) DoubleML.sensitivity_analysis",
        "tolerance": (
            "bias_bound and adjusted theta bounds 1e-12 (observed 2.5e-15); "
            "RV 1e-6 (observed 9.2e-8); RVa is a documented convention gap "
            "(<5e-3, observed 1.4e-3) because StatsPAI exhausts |theta|-z*se "
            "with the unadjusted SE while doubleml lets the SE move with the "
            "confounding scenario"
        ),
        "sides": ["py"],
        "test": ["tests/external_parity/test_dml_sensitivity_parity.py"],
        "note": (
            "Cross-package pin against doubleml-for-py, not R: DoubleML 1.0.2 "
            "exposes no sensitivity method on DoubleMLPLR or its base class, "
            "so the DML omitted-variable-bias analysis has no R counterpart. "
            "Both engines share an explicit fold partition and the test "
            "asserts the underlying PLR fits are identical before comparing "
            "any sensitivity quantity."
        ),
    },
    "ipw": {
        "status": "bit-exact",
        "reference": "base R stats::glm(binomial) + hand-rolled Hajek weighted means",
        "tolerance": "Hajek ATE/ATT estimate 1e-9 (observed <= 2e-15; SE not pinned)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_ipw_parity.py",
            "tests/reference_parity/_fixtures/ipw_R.json",
        ],
        "note": (
            "Frozen-R fixture: sp.ipw's propensity is the unpenalized logit MLE, "
            "so Hajek ATE/ATT reduce to base-R glm + weighted means. See "
            "tests/reference_parity/REFERENCES.md."
        ),
    },
    "g_computation": {
        "status": "bit-exact",
        "reference": "base R stats::lm g-formula standardization (Robins 1986)",
        "tolerance": "psi 1e-8 (observed <= 7e-16; bootstrap SE pinned loosely +/-25%)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_gformula_parity.py",
            "tests/reference_parity/_fixtures/gformula_R.json",
        ],
        "note": (
            "Frozen-R fixture: single additive OLS Q makes the g-formula contrast "
            "collapse to the base-R lm standardization. See "
            "tests/reference_parity/REFERENCES.md."
        ),
    },
    "tmle": {
        "status": "bit-exact",
        "reference": "base R stats::glm TMLE (van der Laan & Rubin 2006)",
        "tolerance": "psi 1e-9 (observed 5.6e-12), EIF SE 1e-9, epsilon 1e-8",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_tmle_parity.py",
            "tests/reference_parity/_fixtures/tmle_R.json",
        ],
        "note": (
            "Frozen-R fixture: single-learner LogisticRegression(penalty=None) "
            "fits the identical unpenalised MLEs and solves the same 1-D "
            "fluctuation score. See tests/reference_parity/REFERENCES.md."
        ),
    },
    "kaplan_meier": {
        "status": "bit-exact",
        "reference": "survival::survfit",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "survival": "3.8.3",
        },
        "tolerance": "S(t) at every event time 1e-12 (observed ~3e-17); median exact",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_survival_km_parity.py",
            "tests/reference_parity/_fixtures/survival_km_R.json",
        ],
        "note": (
            "Frozen-R fixture: the Kaplan-Meier survival curve and median match "
            "R survival::survfit to machine precision on a committed two-group "
            "dataset. Regenerate via _generate_survival_km.R."
        ),
    },
    "logrank_test": {
        "status": "bit-exact",
        "reference": "survival::survdiff",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "survival": "3.8.3",
        },
        "tolerance": "chi-square 1e-10 rel (observed ~8e-16); p-value 1e-10 abs",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_survival_km_parity.py",
            "tests/reference_parity/_fixtures/survival_km_R.json",
        ],
        "note": (
            "Frozen-R fixture: the log-rank chi-square, p-value, and per-group "
            "observed/expected events match R survival::survdiff to machine "
            "precision. Regenerate via _generate_survival_km.R."
        ),
    },
    "bonferroni": {
        "status": "bit-exact",
        "reference": "base R stats::p.adjust(method='bonferroni')",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "exact (atol 1e-15; observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_mht_parity.py",
            "tests/reference_parity/_fixtures/mht_R.json",
        ],
        "note": (
            "Frozen-R fixture: identical multiple-testing correction; matches "
            "base R stats::p.adjust exactly. Regenerate via _generate_mht_R.R."
        ),
    },
    "holm": {
        "status": "bit-exact",
        "reference": "base R stats::p.adjust(method='holm')",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "exact (atol 1e-15; observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_mht_parity.py",
            "tests/reference_parity/_fixtures/mht_R.json",
        ],
        "note": (
            "Frozen-R fixture: identical step-down procedure; matches base R "
            "stats::p.adjust exactly. Regenerate via _generate_mht_R.R."
        ),
    },
    "benjamini_hochberg": {
        "status": "bit-exact",
        "reference": "base R stats::p.adjust(method='BH')",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "exact (atol 1e-15; observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_mht_parity.py",
            "tests/reference_parity/_fixtures/mht_R.json",
        ],
        "note": (
            "Frozen-R fixture: identical FDR step-up procedure; matches base R "
            "stats::p.adjust exactly. Regenerate via _generate_mht_R.R."
        ),
    },
    "adjust_pvalues": {
        "status": "bit-exact",
        "reference": "base R stats::p.adjust (bonferroni/holm/BH)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "exact (atol 1e-15; observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_mht_parity.py",
            "tests/reference_parity/_fixtures/mht_R.json",
        ],
        "note": (
            "Frozen-R fixture: dispatcher matches base R stats::p.adjust across "
            "bonferroni/holm/BH. Regenerate via _generate_mht_R.R."
        ),
    },
    "het_test": {
        "status": "bit-exact",
        "reference": "lmtest::bptest (studentized Breusch-Pagan)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "lmtest": "0.9.40",
        },
        "tolerance": "statistic & p-value 1e-10 rel (observed ~1e-13)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_diagnostics_parity.py",
            "tests/reference_parity/_fixtures/diagnostics_R.json",
        ],
        "note": (
            "Frozen-R fixture: studentized Breusch-Pagan matches lmtest::bptest "
            "to machine precision. Regenerate via _generate_diagnostics_R.R."
        ),
    },
    "reset_test": {
        "status": "bit-exact",
        "reference": "lmtest::resettest(power=2:3, type='fitted')",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "lmtest": "0.9.40",
        },
        "tolerance": "F-statistic & p-value 1e-10 rel (observed ~1e-13)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_diagnostics_parity.py",
            "tests/reference_parity/_fixtures/diagnostics_R.json",
        ],
        "note": (
            "Frozen-R fixture: Ramsey RESET matches lmtest::resettest to machine "
            "precision. Regenerate via _generate_diagnostics_R.R."
        ),
    },
    "survreg": {
        "status": "aligned",
        "reference": "survival::survreg (Weibull AFT)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "survival": "3.8.3",
        },
        "tolerance": "coefficients & log-scale 5e-5 abs (observed ~1e-5)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_aft_parity.py",
            "tests/reference_parity/_fixtures/aft_R.json",
        ],
        "note": (
            "Frozen-R fixture: Weibull AFT log-time coefficients + log-scale "
            "match survival::survreg to iterative-MLE convergence tolerance "
            "(graded aligned, not bit-exact). Regenerate via _generate_aft_R.R."
        ),
    },
    "aft": {
        "status": "aligned",
        "reference": "survival::survreg (Weibull AFT)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "survival": "3.8.3",
        },
        "tolerance": "coefficients & log-scale 5e-5 abs (observed ~1e-5)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_aft_parity.py",
            "tests/reference_parity/_fixtures/aft_R.json",
        ],
        "note": (
            "Frozen-R fixture: Weibull AFT (formula API) matches survival::survreg "
            "to iterative-MLE convergence tolerance (graded aligned). Regenerate "
            "via _generate_aft_R.R."
        ),
    },
    "fracreg": {
        "status": "bit-exact",
        "reference": "stats::glm(quasibinomial('logit')) [fractional response]",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "coefficients 1e-10 abs (observed ~8e-15)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_glm_ext_parity.py",
            "tests/reference_parity/_fixtures/glm_ext_R.json",
        ],
        "note": (
            "Frozen-R fixture: fractional-response GLM matches base R "
            "glm(quasibinomial) to machine precision. Regenerate via "
            "_generate_glm_ext_R.R."
        ),
    },
    "hurdle": {
        "status": "bit-exact",
        "reference": "pscl::hurdle(dist='poisson', zero.dist='binomial')",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "pscl": "1.5.9"},
        "tolerance": "count + zero coefficients 1e-6 abs (observed ~2e-8)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_glm_ext_parity.py",
            "tests/reference_parity/_fixtures/glm_ext_R.json",
        ],
        "note": (
            "Frozen-R fixture: Poisson-logit hurdle count + zero coefficients "
            "match pscl::hurdle to machine tolerance. Regenerate via "
            "_generate_glm_ext_R.R."
        ),
    },
    "cloglog": {
        "status": "aligned",
        "reference": "stats::glm(binomial('cloglog'))",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "coefficients 5e-5 abs (observed ~1e-5; IRLS convergence)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_glm_ext_parity.py",
            "tests/reference_parity/_fixtures/glm_ext_R.json",
        ],
        "note": (
            "Frozen-R fixture: complementary-log-log binary GLM matches base R "
            "glm(binomial('cloglog')) to IRLS convergence tolerance (graded "
            "aligned). Regenerate via _generate_glm_ext_R.R."
        ),
    },
    "odds_ratio": {
        "status": "bit-exact",
        "reference": "base-R closed form (Woolf logit; = epiR::epi.2by2)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "estimate, se_log, CI 1e-12 abs (observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_epi_parity.py",
            "tests/reference_parity/_fixtures/epi_R.json",
        ],
        "note": (
            "Frozen-R fixture: Woolf-logit odds ratio + CI match the canonical "
            "base-R closed form exactly. Regenerate via _generate_epi_R.R."
        ),
    },
    "relative_risk": {
        "status": "bit-exact",
        "reference": "base-R closed form (Katz-log; = epiR::epi.2by2 / Stata epitab)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "estimate, se_log, CI 1e-12 abs (observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_epi_parity.py",
            "tests/reference_parity/_fixtures/epi_R.json",
        ],
        "note": (
            "Frozen-R fixture: Katz-log relative risk + CI match the canonical "
            "base-R closed form exactly. Regenerate via _generate_epi_R.R."
        ),
    },
    "risk_difference": {
        "status": "bit-exact",
        "reference": "base-R closed form (Wald; = epiR::epi.2by2 / Stata epitab)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "estimate, se, CI 1e-12 abs (observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_epi_parity.py",
            "tests/reference_parity/_fixtures/epi_R.json",
        ],
        "note": (
            "Frozen-R fixture: Wald risk difference + CI match the canonical "
            "base-R closed form exactly. Regenerate via _generate_epi_R.R."
        ),
    },
    "mantel_haenszel": {
        "status": "bit-exact",
        "reference": "base-R closed form (Robins-Breslow-Greenland MH; = epiR)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "estimate, se_log, CI 1e-12 abs (observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_epi_parity.py",
            "tests/reference_parity/_fixtures/epi_R.json",
        ],
        "note": (
            "Frozen-R fixture: Mantel-Haenszel OR + Robins-Breslow-Greenland CI "
            "match the canonical base-R closed form exactly. Regenerate via "
            "_generate_epi_R.R."
        ),
    },
    "prevalence_ratio": {
        "status": "bit-exact",
        "reference": "base-R closed form (Katz-log; = epiR::epi.2by2)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "estimate, se_log, CI 1e-12 abs (observed ~2e-16)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_epi_parity.py",
            "tests/reference_parity/_fixtures/epi_R.json",
        ],
        "note": (
            "Frozen-R fixture: Katz-log prevalence ratio + CI match the canonical "
            "base-R closed form exactly. Regenerate via _generate_epi_R.R."
        ),
    },
    "number_needed_to_treat": {
        "status": "bit-exact",
        "reference": "base-R closed form (NNT = 1/risk difference)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "estimate 1e-12 abs (observed 0); CI not pinned",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_epi_parity.py",
            "tests/reference_parity/_fixtures/epi_R.json",
        ],
        "note": (
            "Frozen-R fixture: NNT point estimate = 1/RD matches base R exactly. "
            "CI is not pinned (convention differs when the RD CI crosses zero). "
            "Regenerate via _generate_epi_R.R."
        ),
    },
    "incidence_rate_ratio": {
        "status": "bit-exact",
        "reference": "base-R closed form (rate ratio + conditional-binomial exact CI)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "estimate 1e-12; exact CI 1e-10 abs (observed ~3e-15)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_epi_parity.py",
            "tests/reference_parity/_fixtures/epi_R.json",
        ],
        "note": (
            "Frozen-R fixture: incidence rate ratio + conditional-binomial exact "
            "CI match the canonical base-R closed form (binom.test). Regenerate "
            "via _generate_epi_R.R."
        ),
    },
    "inequality_index": {
        "status": "bit-exact",
        "reference": "base-R closed form (Gini/Theil-T/Theil-L/Atkinson; = ineq)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "all indices 1e-12 abs (observed ~2e-16)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_inequality_parity.py",
            "tests/reference_parity/_fixtures/inequality_R.json",
        ],
        "note": (
            "Frozen-R fixture: bias-corrected Gini, Theil-T, Theil-L (MLD) and "
            "Atkinson (epsilon=1) match the canonical base-R closed form exactly. "
            "Regenerate via _generate_inequality_R.R."
        ),
    },
    "cohen_kappa": {
        "status": "bit-exact",
        "reference": "base-R closed form (Cohen's kappa point estimate)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "kappa + agreements 1e-12 abs (observed ~1e-16); SE not pinned",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_epi_extra_parity.py",
            "tests/reference_parity/_fixtures/epi_extra_R.json",
        ],
        "note": (
            "Frozen-R fixture: unweighted Cohen's kappa + observed/expected "
            "agreement match base R exactly (SE convention-specific, not pinned). "
            "Regenerate via _generate_epi_extra_R.R."
        ),
    },
    "attributable_risk": {
        "status": "bit-exact",
        "reference": "base-R closed form (attributable fraction exposed + PAF)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "AFE + PAF point estimates 1e-12 abs (observed 0); CI not pinned",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_epi_extra_parity.py",
            "tests/reference_parity/_fixtures/epi_extra_R.json",
        ],
        "note": (
            "Frozen-R fixture: attributable fraction exposed (RR-1)/RR and "
            "population attributable fraction match base R exactly (CI method "
            "not pinned). Regenerate via _generate_epi_extra_R.R."
        ),
    },
    "power_rct": {
        "status": "bit-exact",
        "reference": "base-R closed form (two-sample pooled-sigma z-approx power)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "power 1e-12 abs (observed ~2e-16)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_power_parity.py",
            "tests/reference_parity/_fixtures/power_R.json",
        ],
        "note": (
            "Frozen-R fixture: two-sample RCT power matches the canonical "
            "large-sample z-approximation exactly. Regenerate via "
            "_generate_power_R.R."
        ),
    },
    "power_two_proportions": {
        "status": "bit-exact",
        "reference": "base-R closed form (unpooled Wald two-proportion z-approx)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "power 1e-12 abs (observed ~2e-16)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_power_parity.py",
            "tests/reference_parity/_fixtures/power_R.json",
        ],
        "note": (
            "Frozen-R fixture: two-proportion power (unpooled Wald z-approx) "
            "matches base R exactly. Regenerate via _generate_power_R.R."
        ),
    },
    "power_logrank": {
        "status": "bit-exact",
        "reference": "base-R closed form (Schoenfeld log-rank power)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "power 1e-12 abs (observed ~2e-16)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_power_parity.py",
            "tests/reference_parity/_fixtures/power_R.json",
        ],
        "note": (
            "Frozen-R fixture: Schoenfeld log-rank power matches base R exactly. "
            "Regenerate via _generate_power_R.R."
        ),
    },
    "mde": {
        "status": "bit-exact",
        "reference": "base-R closed form (RCT minimum detectable effect)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "effect size 1e-6 abs (output rounded to 6 dp; observed ~2e-8)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_power_extra_parity.py",
            "tests/reference_parity/_fixtures/power_extra_R.json",
        ],
        "note": (
            "Frozen-R fixture: MDE = (z_{1-a/2}+z_pow)/sqrt(n_g/2) matches the "
            "closed-form inverse of the two-sample power (sp rounds to 6 dp). "
            "Regenerate via _generate_power_extra_R.R."
        ),
    },
    "power_cluster_rct": {
        "status": "bit-exact",
        "reference": "base-R closed form (design-effect-inflated z-approx power)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "power 1e-12 abs (observed ~2e-16)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_power_extra_parity.py",
            "tests/reference_parity/_fixtures/power_extra_R.json",
        ],
        "note": (
            "Frozen-R fixture: cluster-RCT power with design effect "
            "1+(m-1)*icc matches base R exactly. Regenerate via "
            "_generate_power_extra_R.R."
        ),
    },
    "evalue_rr": {
        "status": "bit-exact",
        "reference": "R EValue::evalues.RR",
        "reference_versions": {"EValue": "4.1.4"},
        "tolerance": (
            "Point and CI E-values at 1e-12 across ten cases, including RR < 1 and CIs crossing the null."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_evalue_rr_parity.py"],
        "note": (
            "Until 1.28.0 this entry was graded cross-language on a closed form typed into the test, with a docstring asserting that R's EValue implements the same formula; nothing consulted R. The fixture is now EValue's own output."
        ),
    },
    "svymean": {
        "status": "bit-exact",
        "reference": "survey::svymean (Horvitz-Thompson/Hajek + Taylor SE)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "estimate + SE 1e-10 abs (observed ~5e-15 / 8e-17)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_survey_parity.py",
            "tests/reference_parity/_fixtures/survey_R.json",
        ],
        "note": (
            "Frozen-R fixture: weights-only design survey mean + "
            "Taylor-linearization SE match R survey::svymean to machine "
            "precision. Regenerate via _generate_survey_R.R."
        ),
    },
    "svytotal": {
        "status": "bit-exact",
        "reference": "survey::svytotal (Horvitz-Thompson + Taylor SE)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "estimate 1e-12 rel; SE 1e-10 rel (observed ~2e-12 / 1e-14)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_survey_parity.py",
            "tests/reference_parity/_fixtures/survey_R.json",
        ],
        "note": (
            "Frozen-R fixture: weights-only design survey total + "
            "Taylor-linearization SE match R survey::svytotal to machine "
            "precision. Regenerate via _generate_survey_R.R."
        ),
    },
    "svyglm": {
        "status": "bit-exact",
        "reference": "survey::svyglm (design-based GLM + linearization SE)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "coefficients + SE 1e-10 abs (observed ~2e-15 / 6e-15)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_survey_parity.py",
            "tests/reference_parity/_fixtures/survey_R.json",
        ],
        "note": (
            "Frozen-R fixture: survey-weighted GLM coefficients and "
            "design-based (linearization) standard errors match R "
            "survey::svyglm to machine precision. Regenerate via "
            "_generate_survey_R.R."
        ),
    },
    "degree_centrality": {
        "status": "bit-exact",
        "reference": "R igraph::degree",
        "reference_versions": {"igraph": "2.3.3"},
        "tolerance": (
            "Normalised on karate and raw in / out / all modes on a 40-node directed graph, exact."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_network_parity.py",
            "tests/reference_parity/test_network_centrality_parity.py",
        ],
        "note": (
            "Until 1.28.0 this entry was graded cross-language on a test of closed forms (star, triangle, path) that never consulted R; the reference_versions field named an R version that nothing had run. The comparison below is real."
        ),
    },
    "betweenness_centrality": {
        "status": "bit-exact",
        "reference": "R igraph::betweenness",
        "reference_versions": {"igraph": "2.3.3"},
        "tolerance": (
            "Normalised and raw on karate, raw on the directed graph, all at 1e-10."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_network_parity.py",
            "tests/reference_parity/test_network_centrality_parity.py",
        ],
        "note": (
            "Until 1.28.0 this entry was graded cross-language on a test of closed forms (star, triangle, path) that never consulted R; the reference_versions field named an R version that nothing had run. The comparison below is real."
        ),
    },
    "clustering": {
        "status": "bit-exact",
        "reference": "R igraph::transitivity(type = 'local', isolates = 'zero')",
        "reference_versions": {"igraph": "2.3.3"},
        "tolerance": (
            "Exact on karate and on a disconnected graph whose isolates and degree-1 nodes score 0 on both sides."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_network_parity.py",
            "tests/reference_parity/test_network_centrality_parity.py",
        ],
        "note": (
            "Until 1.28.0 this entry was graded cross-language on a test of closed forms (star, triangle, path) that never consulted R; the reference_versions field named an R version that nothing had run. The comparison below is real."
        ),
    },
    "eigenvector_centrality": {
        "status": "bit-exact",
        "reference": "R sna::evcent (unit L2 norm, as here)",
        "reference_versions": {"sna": "2.8", "igraph": "2.3.3"},
        "tolerance": (
            "5e-11 undirected, 8e-11 directed -- power-iteration tolerance on both sides. igraph::eigen_centrality max-scales instead (and 2.x ignores scale = FALSE), so it agrees only up to one scalar; the earlier note claiming the igraph convention was wrong."
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_network_parity.py",
            "tests/reference_parity/test_network_centrality_parity.py",
        ],
        "note": (
            "Until 1.28.0 this entry was graded cross-language on a test of closed forms (star, triangle, path) that never consulted R; the reference_versions field named an R version that nothing had run. The comparison below is real."
        ),
    },
    "glm": {
        "status": "bit-exact",
        "reference": "base R stats::glm (binomial logit + Poisson log)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)"},
        "tolerance": "coef / logLik / AIC 1e-8 abs (observed <= 5e-13); SE ~1e-3 rel",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_glm_parity.py",
            "tests/reference_parity/_fixtures/glm_R.json",
        ],
        "note": (
            "Frozen-R fixture: IRLS converges to the identical unpenalized MLE, "
            "so coefficients, maximized log-likelihood, and AIC match base R "
            "stats::glm to machine precision on a committed dataset; model-based "
            "SEs align to ~1e-3 relative. Regenerate via _generate_glm_R.R."
        ),
    },
    "three_sls": {
        "status": "bit-exact",
        "reference": "R systemfit::systemfit(method='3SLS')",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "systemfit": "1.1.30",
        },
        "tolerance": "coef 1e-9 abs (observed <= 1e-15); SE ~5e-3 rel",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_threesls_parity.py",
            "tests/reference_parity/_fixtures/threesls_R.json",
        ],
        "note": (
            "Frozen-R fixture: 3SLS coefficients on a 2-equation simultaneous "
            "system match R systemfit to machine precision on a committed "
            "dataset; SEs align to ~5e-3 relative (small-sample residual-"
            "covariance d.o.f. convention). Regenerate via _generate_threesls_R.R."
        ),
    },
    "biprobit": {
        "status": "bit-exact",
        "reference": "R VGAM::vglm(binom2.rho) bivariate probit",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "VGAM": "1.1.14",
        },
        "tolerance": "coef / rho 1e-6 abs (observed <= 2e-7); logLik 1e-6 rel",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_biprobit_parity.py",
            "tests/reference_parity/_fixtures/biprobit_R.json",
        ],
        "note": (
            "Frozen-R fixture: both maximize the same joint bivariate-normal "
            "likelihood, so the two equations' coefficients and the error "
            "correlation rho match R VGAM to the shared optimizer tolerance "
            "on a committed dataset. Regenerate via _generate_biprobit_R.R."
        ),
    },
    "did_2x2": {
        "status": "bit-exact",
        "reference": "Stata 18 MP regress [aw=w], robust (aweight HC1)",
        "provenance": (
            "Stata 18 MP output captured live on 2026-07-23 from the seed-20260723 "
            "dataset rebuilt by _make_data() in the test; values embedded as constants."
        ),
        "reference_versions": {"Stata": "18 MP"},
        "tolerance": "b / se 1e-12 abs (observed <= 3e-16)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_did2x2_ddd_weighted_robust_parity.py",
        ],
        "note": (
            "Frozen-Stata anchor (captured live 2026-07-23, asdouble import): "
            "weighted HC1-robust 2x2 DiD matches regress y treat post tp "
            "[aw=w], robust to machine precision on a deterministic seed-"
            "20260723 dataset. Pins the 2026-07 w->w^2 sandwich-meat "
            "correctness fix."
        ),
    },
    "ddd": {
        "status": "bit-exact",
        "reference": "Stata 18 MP regress [aw=w], robust (aweight HC1)",
        "provenance": (
            "Stata 18 MP output captured live on 2026-07-23 from the seed-20260723 "
            "dataset rebuilt by _make_data() in the test; values embedded as constants."
        ),
        "reference_versions": {"Stata": "18 MP"},
        "tolerance": "b / se 1e-12 abs (observed <= 3e-15)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_did2x2_ddd_weighted_robust_parity.py",
        ],
        "note": (
            "Frozen-Stata anchor (captured live 2026-07-23, asdouble import): "
            "weighted HC1-robust triple-difference matches the saturated "
            "regress ... tps [aw=w], robust coefficient to machine precision "
            "on the same seed-20260723 dataset as did_2x2. Pins the 2026-07 "
            "w->w^2 sandwich-meat correctness fix."
        ),
    },
    "mr": {
        "status": "bit-exact",
        "reference": "R MendelianRandomization::mr_ivw through the sp.mr dispatcher",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": ("IVW estimate and default random-effects SE, 1e-10."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_ivw": {
        "status": "bit-exact",
        "reference": "R MendelianRandomization::mr_ivw (default / fixed / random), TwoSampleMR::mr_ivw",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": (
            "Estimate, SE under all three models, RSE and Cochran's Q at 1e-10 (observed <= 1e-15)."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_egger": {
        "status": "bit-exact",
        "reference": "R MendelianRandomization::mr_egger, TwoSampleMR::mr_egger_regression",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": ("Slope, intercept and both SEs at 1e-10."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_median": {
        "status": "bit-exact",
        "reference": "R MendelianRandomization::mr_median (weighted / simple / penalized)",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": (
            "Point estimates for all three weightings at 1e-10. The bootstrap SE is Monte Carlo on both sides and is not compared (T3)."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_mode": {
        "status": "bit-exact",
        "reference": "R MendelianRandomization::mr_mbe (weighted / unweighted, stderror = simple)",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": (
            "Point estimates for both weightings at 1e-10 -- the same point of the same 512-point density grid. The bootstrap SE is Monte Carlo on both sides and is not compared (T3)."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_cml": {
        "status": "bit-exact",
        "reference": "R MendelianRandomization::mr_cML (DP = FALSE, n = 17723)",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": (
            "Estimate and SE for every K = 0..6, the BIC-selected fit with its invalid set {12, 14}, and the MA-BIC average, all at 1e-9 (both sides iterate to |d theta| <= 1e-7)."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_raps": {
        "status": "aligned",
        "reference": "R mr.raps 0.4.3 (simple / overdispersed / overdispersed.robust)",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": (
            "Simple and L2-overdispersed fits (beta, SE, tau2) at 1e-8. Robust Huber / Tukey: the sandwich reproduces R's SEs at 1e-10 when evaluated at R's own (beta, tau2) and integrate() moments; the fitted values agree to 5e-5 (beta), 5e-4 (SE), 1.5e-3 (tau2) because R stops uniroot at its default tolerance and integrate() at 1.2e-4 -- a test shows StatsPAI's root satisfies the estimating equation more tightly than R's."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_steiger": {
        "status": "bit-exact",
        "reference": "R TwoSampleMR::mr_steiger with r from get_r_from_bsen",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": (
            "R^2 on both traits and the direction at 1e-10; the p-value (1.8e-73) at 1e-12."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_radial": {
        "status": "bit-exact",
        "reference": "R RadialMR::ivw_radial (alpha = 0.05, no Bonferroni)",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": (
            "Square-root weights, per-variant Q contributions and total Q at 1e-10; the outlier set is identical with bonferroni=False (StatsPAI's default applies Bonferroni)."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_presso": {
        "status": "bit-exact",
        "reference": "R MRPRESSO::mr_presso (NbDistribution = 2000)",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": (
            "Raw estimate and SE, observed RSS, and the outlier-corrected estimate and SE at 1e-10; outlier set {12, 14} identical. The simulated p-values are Monte Carlo on both sides (T3) and follow the reference's k / B convention."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_leave_one_out": {
        "status": "bit-exact",
        "reference": "R MendelianRandomization::mr_ivw on each leave-one-out subset",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": ("All 28 leave-one-out estimates and default-model SEs at 1e-10."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_pleiotropy_egger": {
        "status": "bit-exact",
        "reference": "R TwoSampleMR::mr_egger_regression (intercept test)",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": ("Intercept, SE and t(n - 2) p-value at 1e-10."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_heterogeneity": {
        "status": "bit-exact",
        "reference": "R TwoSampleMR::mr_ivw / mr_egger_regression (Q, Q_df, Q_pval)",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": (
            "IVW and Egger (Ruecker) Q, degrees of freedom and p-values at 1e-10."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_f_statistic": {
        "status": "bit-exact",
        "reference": "R MendelianRandomization::mr_ivw @Fstat",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": ("Mean F statistic at 1e-10."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "mr_multivariable": {
        "status": "bit-exact",
        "reference": "R MendelianRandomization::mr_mvivw (default random effects)",
        "reference_versions": {
            "MendelianRandomization": "0.10.0",
            "TwoSampleMR": "0.7.9",
            "RadialMR": "1.2.4",
            "MRPRESSO": "1.0",
            "mr.raps": "0.4.3",
        },
        "tolerance": ("Three direct effects and SEs at 1e-10."),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_mr_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the Mendelian-randomisation sweep, on the 28-variant LDL-C / CHD data shipped with MendelianRandomization. It found: IVW reporting the fixed-effect SE regardless of heterogeneity (half the reference's here); Egger not orienting variants (slope 14%, intercept 38% off); a step weighted median, a lower-tail penalty and weights redrawn in the bootstrap; a mode-based bandwidth and grid unlike Hartwig et al.'s; cML penalising BIC by the number of variants instead of the sample size (six invalid variants selected instead of two) with a non-profile SE; mr_raps a different estimator under the name; Steiger one-sided with p = 0 from 1 - Phi; PRESSO without the Bonferroni step; and 1 - cdf p-values losing 1e-7 relative at p = 3e-10."
        ),
    },
    "das_gupta": {
        "status": "bit-exact",
        "reference": "R DasGuptR::dgnpop (product rate function, summed over strata)",
        "reference_versions": {
            "DasGuptR": "2.2.0",
            "ddecompose": "1.0.0",
            "cdgd": "1.0.1",
        },
        "tolerance": (
            "Das Gupta's Table 2.1 (two factors) and Table 6.5 (four factors x six age groups), every factor effect and both crude rates at 1e-10."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_decomp_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the decomposition sweep. It found: sp.das_gupta decomposing the product of factor MEANS instead of the sum over strata of factor products its docstring stated (factor shares 0% / +333% where DasGuptR gives 37% / -52% on Das Gupta's Table 6.5); sp.gap_closing reweighting by the reciprocal of the density ratio in its IPW and AIPW paths (IPW counterfactual gap twice the observed gap on a DGP whose true value is zero); and sp.yu_elwert_decompose(method='efficient') computing selection as a covariance of DR scores, so its components did not add up to the disparity."
        ),
    },
    "kitagawa_decompose": {
        "status": "bit-exact",
        "reference": "R DasGuptR::dgnpop with ratefunction sum(size*rate)/sum(size)",
        "reference_versions": {
            "DasGuptR": "2.2.0",
            "ddecompose": "1.0.0",
            "cdgd": "1.0.1",
        },
        "tolerance": (
            "Rate and composition effects on Das Gupta's Table 5.1 at 1e-10; interaction exactly 0."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_decomp_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the decomposition sweep. It found: sp.das_gupta decomposing the product of factor MEANS instead of the sum over strata of factor products its docstring stated (factor shares 0% / +333% where DasGuptR gives 37% / -52% on Das Gupta's Table 6.5); sp.gap_closing reweighting by the reciprocal of the density ratio in its IPW and AIPW paths (IPW counterfactual gap twice the observed gap on a DGP whose true value is zero); and sp.yu_elwert_decompose(method='efficient') computing selection as a covariance of DR scores, so its components did not add up to the disparity."
        ),
    },
    "gap_closing": {
        "status": "bit-exact",
        "reference": "R ddecompose::dfl_decompose (method='ipw') and ob_decompose (method='regression')",
        "reference_versions": {
            "DasGuptR": "2.2.0",
            "ddecompose": "1.0.0",
            "cdgd": "1.0.1",
        },
        "tolerance": (
            "Observed, counterfactual and closed gaps at 1e-9 for IPW in both directions (logit MLE in the path) and 1e-10 for regression. method='aipw' has no reference and is checked for double robustness on a known-truth DGP (T1)."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_decomp_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the decomposition sweep. It found: sp.das_gupta decomposing the product of factor MEANS instead of the sum over strata of factor products its docstring stated (factor shares 0% / +333% where DasGuptR gives 37% / -52% on Das Gupta's Table 6.5); sp.gap_closing reweighting by the reciprocal of the density ratio in its IPW and AIPW paths (IPW counterfactual gap twice the observed gap on a DGP whose true value is zero); and sp.yu_elwert_decompose(method='efficient') computing selection as a covariance of DR scores, so its components did not add up to the disparity."
        ),
    },
    "yu_elwert_decompose": {
        "status": "bit-exact",
        "reference": "R cdgd::cdgd0_manual on independently fitted within-cell lm / within-group glm nuisances",
        "reference_versions": {
            "DasGuptR": "2.2.0",
            "ddecompose": "1.0.0",
            "cdgd": "1.0.1",
        },
        "tolerance": (
            "method='efficient': disparity, baseline, prevalence, effect, selection and their EIF standard errors at 1e-9. method='plugin' has no reference implementation and is covered by its exact additivity identity."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_decomp_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the decomposition sweep. It found: sp.das_gupta decomposing the product of factor MEANS instead of the sum over strata of factor products its docstring stated (factor shares 0% / +333% where DasGuptR gives 37% / -52% on Das Gupta's Table 6.5); sp.gap_closing reweighting by the reciprocal of the density ratio in its IPW and AIPW paths (IPW counterfactual gap twice the observed gap on a DGP whose true value is zero); and sp.yu_elwert_decompose(method='efficient') computing selection as a covariance of DR scores, so its components did not add up to the disparity."
        ),
    },
    "rifreg": {
        "status": "bit-exact",
        "reference": "R rifreg::rifreg (variance, quantiles) and dineq::rif + lm (Gini)",
        "reference_versions": {
            "rifreg": "1.1.0",
            "dineq": "0.1.0",
            "ddecompose": "1.0.0",
        },
        "tolerance": (
            "Coefficients at 1e-10 for the variance and the 10th/50th/90th percentiles (quantile_convention='rifreg') and for the Gini against dineq's exact RIF; the stock rifreg Gini, which integrates the Lorenz curve numerically, agrees to 1e-4."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_decomp_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the decomposition sweep. It found the Gini RIF built from a midpoint ECDF and the n/(n-1)-corrected Gini, averaging to neither Gini (RIF-regression coefficients up to 1% off), and sp.ffl_decompose storing the specification and reweighting errors under each other's names with a sign error for reference=1, so its components did not add up."
        ),
    },
    "ffl_decompose": {
        "status": "bit-exact",
        "reference": "R ddecompose::ob_decompose(reweighting = TRUE)",
        "reference_versions": {
            "rifreg": "1.1.0",
            "dineq": "0.1.0",
            "ddecompose": "1.0.0",
        },
        "tolerance": (
            "Observed difference, composition, structure, specification and reweighting errors at 1e-9 (1e-12 absolute floor; logit MLE in the path), both reference directions, for the variance, Gini (exact RIF supplied as custom_rif_function) and 10th/50th/90th percentiles."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_decomp_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the decomposition sweep. It found the Gini RIF built from a midpoint ECDF and the n/(n-1)-corrected Gini, averaging to neither Gini (RIF-regression coefficients up to 1% off), and sp.ffl_decompose storing the specification and reweighting errors under each other's names with a sign error for reference=1, so its components did not add up."
        ),
    },
    "gelbach": {
        "status": "bit-exact",
        "reference": "Stata b1x2 (Gelbach's own command), robust and homoskedastic",
        "reference_versions": {"Stata": "18 MP", "b1x2": "SSC"},
        "tolerance": (
            "Per-variable contributions at 1e-10; their full covariance, SEs and the total-change variance at 1e-9, robust and homoskedastic."
        ),
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_decomp_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the decomposition sweep against Stata 18 MP and SSC b1x2 (Gelbach), ineqdeco (Jenkins) and descogini (Lopez-Feldman), run by tests/reference_parity/_fixtures/_generate_decomp_stata.do. It found sp.gelbach's SEs omitting the covariance between the auxiliary and long regressions."
        ),
    },
    "subgroup_decompose": {
        "status": "bit-exact",
        "reference": "Stata ineqdeco, bygroup()",
        "reference_versions": {"Stata": "18 MP", "ineqdeco": "SSC"},
        "tolerance": (
            "GE(0), GE(1), GE(2) totals and within components at 1e-12, between components at 1e-11. The Gini path (Dagum) has no ineqdeco counterpart and keeps its analytical checks."
        ),
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_decomp_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the decomposition sweep against Stata 18 MP and SSC b1x2 (Gelbach), ineqdeco (Jenkins) and descogini (Lopez-Feldman), run by tests/reference_parity/_fixtures/_generate_decomp_stata.do. It found sp.gelbach's SEs omitting the covariance between the auxiliary and long regressions."
        ),
    },
    "source_decompose": {
        "status": "bit-exact",
        "reference": "Stata descogini (Lerman-Yitzhaki)",
        "reference_versions": {"Stata": "18 MP", "descogini": "SSC"},
        "tolerance": (
            "With gini='population': total Gini, and each source's S_k, G_k, R_k and share of the total at 1e-12. The default n/(n-1)-corrected Gini differs by exactly that factor; S_k, R_k and shares are identical."
        ),
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_decomp_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the decomposition sweep against Stata 18 MP and SSC b1x2 (Gelbach), ineqdeco (Jenkins) and descogini (Lopez-Feldman), run by tests/reference_parity/_fixtures/_generate_decomp_stata.do. It found sp.gelbach's SEs omitting the covariance between the auxiliary and long regressions."
        ),
    },
    "bauer_sinning": {
        "status": "bit-exact",
        "reference": "Stata mvdcmp (Powers, Yoshioka & Yun), logit",
        "reference_versions": {"Stata": "18 MP", "mvdcmp": "SSC"},
        "tolerance": (
            "Logit: explained, unexplained, gap, Yun-weighted detailed explained and unexplained terms, their 6x6 delta-method covariance and the aggregate SEs at 1e-9. Probit aligned at 2e-6 (reference probit not fully converged; see note)."
        ),
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_decomp_R_parity.py"],
        "note": (
            "Added in 1.28.0 by the decomposition sweep against Stata 18 MP and SSC mvdcmp (Powers, Yoshioka & Yun), run by tests/reference_parity/_fixtures/_generate_decomp_stata.do. bauer_sinning had no inference and no detailed unexplained component; both are now ports of mvdcmp. Probit is graded within logit's entry: mvdcmp's probit stops at Stata's default tolerance, so it agrees to 2e-6, and the fixture shows Stata's tightly converged probit equals StatsPAI's to 1e-12."
        ),
    },
    "xtdpdsys": {
        "status": "bit-exact",
        "reference": "Stata 18 xtdpdsys (built-in); xtabond2 iv(x, eq(diff)) h(2)",
        "reference_versions": {"Stata": "18 MP", "xtabond2": "SSC 03.07.00"},
        "tolerance": (
            "abdata, n on L.n (and w k): coefficients and SEs at 1e-9 (observed <= 6e-12) for one-step robust, two-step Windmeijer and classical one-step, instrument count equal; xtabond2 with iv(w k, eq(diff)) h(2) reproduces the same numbers."
        ),
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_dynpanel_abdata_parity.py"],
        "note": (
            "Promoted in 1.28.0. sp.xtdpdsys had run xtabond2's default moment set (exogenous regressors in both equations, h(3)), so it did not reproduce the Stata command it is named after: L.n 0.686 against xtdpdsys's 0.542 on abdata. The Stata fixture already carried the xtdpdsys covariate spec, but no test read it. Now defaults to xtdpdsys's convention; iv_equation='both', h=3 gives xtabond2's."
        ),
    },
    "xtlsdvc": {
        "status": "bit-exact",
        "reference": "Stata xtlsdvc V1.0.4 (Bruno 2005), SSC",
        "reference_versions": {"Stata": "18 MP", "xtlsdvc": "1.0.4"},
        "tolerance": (
            "abdata: bias-corrected coefficients for initial(ab/ah/bb) x bias(1/2/3) and the AR(1)-only model at rtol 1e-7 (observed 4e-14 to 1.6e-9, the looser end through the Anderson-Hsiao initialiser). Standard errors are excluded by design: xtlsdvc reports the uncorrected LSDV ones, which sp.xtlsdvc reproduces and warns about."
        ),
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_lsdvc_parity.py"],
        "note": (
            "Promoted in 1.28.0. The comparison against Stata's xtlsdvc (fixture specs H1-H6) had been in the suite since the dynamic-panel campaign, but no promotion was ever registered, so the index kept the function at analytical-only."
        ),
    },
    "olley_pakes": {
        "status": "aligned",
        "reference": (
            "Stata prodest, method(op) valueadded (Rovigatti & Mollisi); R "
            "prodest::prodestOP"
        ),
        "reference_versions": {
            "Stata": "18 MP",
            "prodest": "SSC",
            "R prodest": "1.0.2",
        },
        "tolerance": (
            "Simulated unbalanced panel with 31 calendar gaps. Free-input coefficient "
            "(stage-1 OLS) at 1e-12 against both, polynomial degree 2 and 3. State "
            "coefficient: StatsPAI solves the first-order condition and has a sum of "
            "squares no larger than at either reference estimate; the references stop "
            "their optimisers early (R BFGS 5.9e-6 away, Stata Nelder-Mead at tolerance"
            " 1e-5 up to 4.3e-3 away)."
        ),
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_prodest_parity.py"],
        "note": (
            "Added in 1.28.0 by the production-function sweep, run by "
            "tests/reference_parity/_fixtures/_generate_prodest_R.R and "
            "_generate_prodest_stata.do. sp.olley_pakes had estimated the free-input "
            "coefficient in a GMM second stage with contemporaneous labour as its own "
            "instrument, not from the stage-1 regression, and its lag operator shifted "
            "rows, pairing years across gaps. Aligned rather than bit-exact because the"
            " state coefficient is compared as a minimiser, not a matched number."
        ),
    },
    "levinsohn_petrin": {
        "status": "aligned",
        "reference": (
            "Stata prodest, method(lp) valueadded (Rovigatti & Mollisi); R "
            "prodest::prodestLP"
        ),
        "reference_versions": {
            "Stata": "18 MP",
            "prodest": "SSC",
            "R prodest": "1.0.2",
        },
        "tolerance": (
            "As olley_pakes with materials as the proxy: free-input coefficient at "
            "1e-12 against both; state coefficient at the minimiser, R 1.3e-6 and Stata"
            " up to 1.8e-3 away where their optimisers stop."
        ),
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_prodest_parity.py"],
        "note": (
            "Added in 1.28.0 by the production-function sweep; same defects and fixes "
            "as olley_pakes."
        ),
    },
    "ackerberg_caves_frazer": {
        "status": "aligned",
        "reference": "Stata prodest, method(lp) acf valueadded; R prodest::prodestACF",
        "reference_versions": {
            "Stata": "18 MP",
            "prodest": "SSC",
            "R prodest": "1.0.2",
        },
        "tolerance": (
            "StatsPAI returns an exact root of the just-identified moment conditions "
            "(criterion < 1e-25), the one nearest the stage-1 coefficients; R stops "
            "2.4e-6 (relative) from that root with criterion 9e-16. Stata's Nelder-Mead"
            " stops at non-roots (criterion 4e-6 / 7e-6), which the test records. The "
            "parity panel has three roots, all reported in diagnostics['acf_roots']."
        ),
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_prodest_parity.py"],
        "note": (
            "Added in 1.28.0 by the production-function sweep. "
            "sp.ackerberg_caves_frazer used a positional lag and a linear Markov "
            "process by default, and returned whichever local optimum Nelder-Mead "
            "reached from five fixed starts, with no indication that the moment "
            "conditions had other roots."
        ),
    },
    "wooldridge_prod": {
        "status": "analytical-only",
        "reference": (
            "Known truth on the simulated parity panel; convention='prodest' against "
            "Stata prodest, method(wrdg) and R prodestWRDG"
        ),
        "tolerance": (
            "Default GMM (free Markov polynomial) recovers labour 0.60, capital 0.30 "
            "and rho 0.70 on the parity panel within 0.05 / 0.10 / 0.05. "
            "convention='prodest' matches Stata's stacked 2SLS coefficients and "
            "unadjusted variance at 1e-8 (polynomial degree 2 and 3), and R prodestWRDG"
            " is rebuilt from the same design at 1e-12."
        ),
        "sides": ["py"],
        "test": ["tests/reference_parity/test_prodest_parity.py"],
        "note": (
            "Graded on its default, which has no reference implementation: both prodest"
            " implementations impose a unit-slope Markov process, which on the parity "
            "panel (rho = 0.7) puts capital at -0.64 (Stata) and -0.79 (R) against a "
            "true 0.30. The previous sp.wooldridge_prod was a stacked NLS that treated "
            "labour as exogenous in both equations."
        ),
    },
    # ======================================================================
    # Cross-language campaign, phase 3 (2026-09). One section per family; the
    # per-function tables, defects found, before/after numbers and unresolved
    # items are in docs/dev/campaign_phase3/<family>*.md. Every field below is
    # copied from the asserting test / fixture of that family.
    # ======================================================================
    # ---- phase 3: survival / epidemiology / nonparametric (13) ----
    "auc": {
        "status": "bit-exact",
        "reference": "R pROC::auc; Stata roctab",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "pROC": "1.19.0.1",
            "Stata": "18 MP",
        },
        "tolerance": "1e-10 rel (observed 2e-16)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "Ties counted one half; also asserted against the mid-rank "
        "Mann-Whitney identity.",
    },
    "breslow_day_test": {
        "status": "bit-exact",
        "reference": "R DescTools::BreslowDayTest (correct=FALSE/TRUE); Stata cc, "
        "by() bd tarone",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "DescTools": "0.99.60",
            "Stata": "18 MP",
        },
        "tolerance": "1e-10 rel (observed 1.3e-14)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "Four strata, one with small cells; Mantel-Haenszel common OR on all "
        "three sides.",
    },
    "cox_frailty": {
        "status": "aligned",
        "reference": "R survival::coxph(... + frailty(id, theta=, sparse=FALSE)); "
        "Stata stcox, shared()",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "survival": "3.8.3",
            "Stata": "18 MP",
        },
        "tolerance": "fixed theta: beta and SE 1e-9, integrated log likelihood 1e-10 "
        "(observed 4e-14); theta maximiser 1e-6 vs R optimize (observed "
        "5e-8) and 5e-5 vs Stata e(theta) (observed 7.5e-6)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "Bit-exact at a fixed theta, including at Stata's own theta. Aligned "
        "rather than bit-exact because theta is compared as the maximiser of "
        "a flat integrated likelihood; ours attains a log likelihood at least "
        "as high as Stata's. R's default sparse=TRUE / method='em' fit is not "
        "the comparison target.",
    },
    "cuminc": {
        "status": "bit-exact",
        "reference": "R cmprsk::cuminc (estimate, var, Tests); Stata stcompet (ci, "
        "se, hi, lo)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "cmprsk": "2.2.12",
            "Stata": "18 MP",
            "stcompet": "1.0.7 (06nov2012)",
        },
        "tolerance": "1e-10 rel (observed: CIF 6e-16, Gray variance 3e-13, delta SE "
        "2e-16, Gray test 9e-15)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "variance='gray' is cmprsk's asymptotic variance; variance='delta' "
        "(default) is the Marubini-Valsecchi delta method stcompet reports, "
        "conf_type='log-log' its bounds. Gray's test is cmprsk's, "
        "unstratified, rho 0 and 1, two and three groups. Tied event and "
        "censoring times in the fixture. Regenerate via "
        "_generate_survival_epi_R.R / _generate_survival_epi_stata.do.",
    },
    "diagnostic_test": {
        "status": "bit-exact",
        "reference": "R epiR::epi.tests; Stata diagti",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "epiR": "2.0.94",
            "Stata": "18 MP",
            "diagt": "2.032 (diagti 2.053)",
        },
        "tolerance": "1e-12 rel (observed 2e-15)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "Alias of sensitivity_specificity; the Stata block calls "
        "sp.diagnostic_test.",
    },
    "direct_standardize": {
        "status": "bit-exact",
        "reference": "R epitools::ageadjust.direct; Stata dstdize",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "epitools": "0.5.10.1",
            "Stata": "18 MP",
        },
        "tolerance": "1e-10 rel (observed 3e-13)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "ci_method='gamma' (Poisson variance) is epitools; ci_method='normal' "
        "with variance='binomial' is dstdize. The default lognormal interval "
        "has no package reference; the rate itself is pinned to both.",
    },
    "finegray": {
        "status": "bit-exact",
        "reference": "R cmprsk::crr (coef, var, invinf, loglik); Stata stcrreg",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "cmprsk": "2.2.12",
            "Stata": "18 MP",
        },
        "tolerance": "R 1e-10 rel (observed 6e-15); Stata coefficients 1e-7 and SEs "
        "1e-8 (Stata's ml stops 5e-9 away), log likelihood 1e-10",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "Censoring KM at left limits, as crr and stcrreg. vce='robust' "
        "(default) is crr's var; small_sample=True is stcrreg's N/(N-1) "
        "scaling; vce='model' is crr's invinf. Both causes. Breslow ties.",
    },
    "indirect_standardize": {
        "status": "bit-exact",
        "reference": "Stata istdize (exact CI); R epitools::ageadjust.indirect "
        "(log-normal CI)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "epitools": "0.5.10.1",
            "Stata": "18 MP",
        },
        "tolerance": "1e-10 rel (observed 6e-16)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "ci_method='exact' (default) is istdize; 'lognormal' is epitools. "
        "p_value not referenced.",
    },
    "kdensity": {
        "status": "bit-exact",
        "reference": "Stata kdensity (at(), bwidth(), kernel()); R bw.nrd0 / bw.SJ",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "Stata": "18 MP"},
        "tolerance": "density and default widths 1e-10 rel (observed 9e-16); "
        "Sheather-Jones 1e-6 vs bw.SJ(nb=1e7, tol=1e-14) (observed 2e-7, "
        "R's pair-count binning)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "Kernels epanechnikov (= Stata epan2), biweight, gaussian, uniform (= "
        "rectangle), triangular (= triangle). bw_method='silverman' is "
        "bw.nrd0; 'stata' is kdensity's default width. The cosine kernel is "
        "not Stata's and is not compared.",
    },
    "lpoly": {
        "status": "bit-exact",
        "reference": "Stata lpoly (at(), bwidth(), degree(), kernel(), se(), "
        "pwidth())",
        "reference_versions": {"Stata": "18 MP"},
        "tolerance": "1e-10 rel (observed 1.6e-12)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "Degrees 0/1/2, kernels epan2 / gaussian / biweight, fits and "
        "se_method='stata' SEs with pwidth. Stata's rule-of-thumb bandwidth "
        "is not implemented; the default robust SE is not Stata's.",
    },
    "power_case_control": {
        "status": "bit-exact",
        "reference": "Stata power twoproportions; R "
        "stats::power.prop.test(strict=TRUE)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "Stata": "18 MP"},
        "tolerance": "1e-11 rel vs Stata (observed 4e-13, Stata's normal CDF); 1e-10 "
        "vs R (observed 6e-16)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "test='chi2' (pooled null, both tails), 1:1 to 1:3 allocation, one- "
        "and two-sided. The default test='wald' has no package reference.",
    },
    "roc_curve": {
        "status": "bit-exact",
        "reference": "R pROC::roc/auc/var/ci.auc (DeLong); Stata roctab (default and "
        "hanley)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "pROC": "1.19.0.1",
            "Stata": "18 MP",
        },
        "tolerance": "1e-10 rel (observed 6e-15)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "Continuous and heavily tied scores. se_method='delong' is pROC and "
        "roctab's default; 'hanley-empirical' is roctab, hanley. The default "
        "'hanley' (exponential Q1/Q2 approximation) has no package reference.",
    },
    "sensitivity_specificity": {
        "status": "bit-exact",
        "reference": "R epiR::epi.tests (method wilson / exact); Stata diagti",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "epiR": "2.0.94",
            "Stata": "18 MP",
            "diagt": "2.032 (diagti 2.053)",
        },
        "tolerance": "1e-12 rel on intervals, 1e-10 on ratios (observed 2e-15)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survival_epi_R_parity.py",
            "tests/reference_parity/_fixtures/survival_epi_R.json",
            "tests/reference_parity/_fixtures/survival_epi_stata.json",
        ],
        "note": "Three tables, one with a zero cell. ci_method='wilson' (default) or "
        "'exact'.",
    },
    # ---- phase 3: time series and panel unit roots (9) ----
    "cusum_test": {
        "status": "bit-exact",
        "reference": "strucchange::efp(type = 'Rec-CUSUM') + sctest 1.5.4; Stata 18 "
        "estat sbcusum",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "strucchange": "1.5.4",
            "Stata": "18",
        },
        "tolerance": "process, statistic and p-value 1e-10 rel (observed 7.1e-14); "
        "Stata boundary constants 3e-6",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_timeseries_R_parity.py",
            "tests/reference_parity/_fixtures/timeseries_R.json",
            "tests/reference_parity/_fixtures/timeseries_Stata.json",
        ],
        "note": "Stata's printed boundary coefficients differ from the root of "
        "strucchange's closed-form crossing probability by <= 2.5e-6 "
        "relative; mechanism not verified. The statistic agrees at 1e-10.",
    },
    "engle_granger": {
        "status": "bit-exact",
        "reference": "egranger 1.0.6 (Stata SSC); urca::ur.df on lm residuals; "
        "aTSA::coint.test",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "urca": "1.3.4",
            "aTSA": "3.1.2.1",
            "Stata": "18",
            "egranger": "1.0.6",
        },
        "tolerance": "Z(t) and step-1 coefficients 1e-10 rel (observed 2.1e-13); "
        "MacKinnon (2010) critical values 1e-12 vs egranger",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_timeseries_R_parity.py",
            "tests/reference_parity/_fixtures/timeseries_R.json",
            "tests/reference_parity/_fixtures/timeseries_Stata.json",
        ],
        "note": "Six cases: 2 and 3 series, lags 0/1/2, trend c/ct/ctt (egranger "
        "trend/qtrend). Residual ADF without deterministic terms; critical "
        "values at T = n - 1.",
    },
    "garch": {
        "status": "aligned",
        "reference": "Stata 18 arch; rugarch::ugarchfit 1.5.6",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "rugarch": "1.5.6",
            "Stata": "18",
        },
        "tolerance": "log-likelihood at the reference optimum 1e-12 (observed "
        "8.5e-16); vs Stata: b 1e-5, SE 2e-5 (observed 8.2e-6 / 1e-5); "
        "vs rugarch: params 5e-4, SE at their parameters 1e-2",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_timeseries_R_parity.py",
            "tests/reference_parity/_fixtures/timeseries_R.json",
            "tests/reference_parity/_fixtures/timeseries_Stata.json",
        ],
        "note": "Same objective on both sides (our log-likelihood at their parameters "
        "equals theirs). Parameter gaps are their optimiser's: both "
        "references stop below our optimum (Stata by 2.6e-10). Stata "
        "vce(robust) carries N/(N-1). rugarch SEs come from a second "
        "difference of the log-likelihood (.hessian2sided, step eps^(1/3)|x|) "
        "and are about 0.5% noisy; ours agree with Stata's to 4e-6 at the "
        "same parameters.",
    },
    "granger_causality": {
        "status": "bit-exact",
        "reference": "Stata 18 vargranger; vars::causality 1.6.1",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "vars": "1.6.1",
            "Stata": "18",
        },
        "tolerance": "chi2 and F 1e-10 rel (observed 4.1e-15), p-values 1e-8",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_timeseries_R_parity.py",
            "tests/reference_parity/_fixtures/timeseries_R.json",
            "tests/reference_parity/_fixtures/timeseries_Stata.json",
        ],
        "note": "All nine vargranger rows (incl. ALL) after var, var small, var small "
        "dfk. vars::causality F equal; its df2 is the system K(T - m), "
        "reproduced.",
    },
    "irf": {
        "status": "bit-exact",
        "reference": "vars::irf 1.6.1; Stata 18 irf create",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "vars": "1.6.1",
            "Stata": "18",
        },
        "tolerance": "1e-10 rel vs vars, 1e-9 vs Stata irf file (observed 2.6e-14)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_timeseries_R_parity.py",
            "tests/reference_parity/_fixtures/timeseries_R.json",
            "tests/reference_parity/_fixtures/timeseries_Stata.json",
        ],
        "note": "Orthogonalised, simple and cumulative responses. Residual covariance "
        "divisor T (Stata var) or T - m (vars::Psi, Stata var, dfk) via "
        "sigma_df.",
    },
    "its": {
        "status": "bit-exact",
        "reference": "lm + sandwich::NeweyWest 3.1.1; Stata 18 newey; itsa 1.0.0 "
        "(SSC)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "sandwich": "3.1.1",
            "Stata": "18",
            "itsa": "1.0.0",
        },
        "tolerance": "coefficients and Newey-West SE 1e-10 rel (observed 7.9e-14); "
        "itsa 1e-6 (glm2 IRLS)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_timeseries_R_parity.py",
            "tests/reference_parity/_fixtures/timeseries_R.json",
            "tests/reference_parity/_fixtures/timeseries_Stata.json",
        ],
        "note": "Bartlett L = 4, no prewhitening. Default = NeweyWest(adjust = "
        "FALSE); hac_small_sample=True = NeweyWest(adjust = TRUE) = Stata "
        "newey.",
    },
    "johansen": {
        "status": "bit-exact",
        "reference": "urca::ca.jo 1.3.4; Stata 18 vecrank",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "urca": "1.3.4",
            "Stata": "18",
        },
        "tolerance": "eigenvalues, trace & max-eigenvalue statistics 1e-11 rel vs "
        "ca.jo, 1e-10 vs vecrank (observed 8.2e-14); Osterwald-Lenum "
        "table equal to Stata _vecgetcv cell by cell",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_timeseries_R_parity.py",
            "tests/reference_parity/_fixtures/timeseries_R.json",
            "tests/reference_parity/_fixtures/timeseries_Stata.json",
        ],
        "note": "lags = ca.jo K - 1 = vecrank lags() - 1. trend 'c'/'rc'/'rt' vs "
        "ca.jo ecdet none/const/trend (K = 2, 3); all five vecrank trend() "
        "cases vs Stata. ca.jo ships a different critical-value table (not "
        "asserted); ours is Stata's.",
    },
    "panel_unitroot": {
        "status": "bit-exact",
        "reference": "plm::purtest 2.6.7; Stata 18 xtunitroot",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "plm": "2.6.7",
            "Stata": "18",
        },
        "tolerance": "all statistics 1e-10 rel (observed 9.1e-15)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_timeseries_R_parity.py",
            "tests/reference_parity/_fixtures/timeseries_R.json",
            "tests/reference_parity/_fixtures/timeseries_Stata.json",
        ],
        "note": "LLC, IPS W-t-bar, Fisher (P, Z, L*, Pm), Hadri (both variants); "
        "convention='plm' vs purtest (dfcor both ways), convention='stata' vs "
        "xtunitroot. Stata LLC with trend uses sigma* = .971 at T = 40 (plm "
        ".871); rebuilt from our quantities. Stata L* uses 5N+3 in its scale "
        "constant, plm 5N+4.",
    },
    # ---- phase 3: inference and sensitivity (20) ----
    "bias_factor": {
        "status": "bit-exact",
        "reference": "R EValue::multi_bound(confounding(), RRAUc, RRUcY)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "EValue": "4.1.4"},
        "tolerance": "1e-14 rel (observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
        ],
        "note": "Plus the identity B(e, e) = RR at the E-value e of RR.",
    },
    "cluster_robust_se": {
        "status": "bit-exact",
        "reference": "R sandwich::vcovCL (HC1, cadjust; HC0; two-way multi0=FALSE); "
        "Stata regress, vce(cluster)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "sandwich": "3.1.1",
            "Stata": "18",
        },
        "tolerance": "SE 1e-10 rel (observed 2.1e-15 R, 9.8e-16 Stata)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "One-way CR1 = vcovCL(type='HC1', cadjust=TRUE) = Stata vce(cluster); "
        "CR0 with df_adjust=False; two-way Cameron-Gelbach-Miller with each "
        "component's own G. The two-way matrix is positive definite on the "
        "fixture, so the default PSD projection (sandwich fix=TRUE) is "
        "inactive; both fix settings are in the fixture.",
    },
    "conley": {
        "status": "bit-exact",
        "reference": "Stata acreg (Colella, Lalive, Sakalli & Thoenig)",
        "provenance": "Stata 18 MP acreg output embedded as constants in "
        "test_conley_acreg_spacetime_parity.py (commands recorded next "
        "to each oracle matrix, synthetic geo-panel regenerated from "
        "default_rng(7)); the IV case in test_iv_hdfe_stata_parity.py "
        "records acreg 1.1.0 on _fixtures/iv_hdfe_panel.csv.",
        "reference_versions": {"Stata": "18 MP", "acreg": "1.1.0"},
        "tolerance": "SE 1e-9 rel (observed ~5e-15); absorbed-IV spatial 1e-10",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_conley_acreg_spacetime_parity.py",
            "tests/reference_parity/test_iv_hdfe_stata_parity.py",
        ],
        "note": "Seven spatial / spatio-temporal kernel configurations; the full "
        "acreg e(V) is reproduced entrywise including Mata _makesymmetric. "
        "Off-diagonals of acreg depend on regressor order (asserted); "
        "StatsPAI reports the symmetric part. The IV space-time case is 1e-3 "
        "(acreg carries a numerically zero constant column).",
    },
    "cr3_jackknife_vcov": {
        "status": "bit-exact",
        "reference": "R sandwich::vcovJK(center='estimate'), summclust (CV3); Stata "
        "regress, vce(jackknife, cluster() mse double)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "sandwich": "3.1.1",
            "summclust": "0.7.0",
            "Stata": "18",
        },
        "tolerance": "vcov 1e-10 rel (observed 6.5e-14 R, SE 4.3e-15 Stata)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "(G-1)/G * sum (b_(g) - b)(b_(g) - b)', centred at the full-sample "
        "estimate. clubSandwich's CR3 is the same matrix without the (G-1)/G "
        "factor, asserted as an identity. Stata's jackknife prefix stores "
        "replicates in float unless `double` (SEs move ~7e-8); the fixture "
        "uses double and records the float default.",
    },
    "evalue_from_result": {
        "status": "bit-exact",
        "reference": "R EValue::twoXtwoRR -> evalues.RR",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "EValue": "4.1.4"},
        "tolerance": "1e-13 rel (observed 3.3e-16)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
        ],
        "note": "End-to-end chain sp.relative_risk -> "
        "sp.evalue_from_result(measure='RR') on three 2x2 tables (harmful, "
        "protective, CI crossing the null). The SMD path is covered only "
        "through sp.evalue (Track A 23_evalue).",
    },
    "evalue_rd": {
        "status": "bit-exact",
        "reference": "R EValue::evalues.RD",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "EValue": "4.1.4"},
        "tolerance": "1e-12 rel (observed 5.8e-14: grid built by seq vs np.arange)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
        ],
        "note": "Five tables incl. non-null true, alpha = 0.1, grid = 1e-3 and a CI "
        "crossing the null (E-value 1).",
    },
    "fisher_exact": {
        "status": "bit-exact",
        "reference": "R ri2::conduct_ri (randomizr full enumeration); Stata ritest "
        "over the full assignment set",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "ri2": "0.5.0",
            "Stata": "18",
            "ritest": "1.1.7",
        },
        "tolerance": "p exact (k / N_assignments); statistic 1e-12 rel vs R, 2^-23 vs "
        "Stata",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "Complete (924 assignments), cluster (70) and stratified (4900) "
        "designs; ATE, KS and rank-sum statistics. Exact enumeration when the "
        "design has <= n_perm assignments (ri2's rule) was added in this "
        "sweep. ritest stores T(obs) in single precision. The Hodges-Lehmann "
        "interval is not pinned.",
    },
    "jackknife_se": {
        "status": "bit-exact",
        "reference": "R sandwich::vcovJK(center='mean'); Stata regress, "
        "vce(jackknife, cluster() double)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "sandwich": "3.1.1",
            "Stata": "18",
        },
        "tolerance": "SE / CI 1e-10 rel, p 1e-9 (observed SE 2.0e-15, p 1.4e-14, CI "
        "9.2e-12)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "Replicates centred at their mean; t(G-1) p-values and intervals as "
        "Stata reports them.",
    },
    "lincom": {
        "status": "aligned",
        "reference": "Stata 18 lincom (after regress / ivregress / logit / poisson)",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-6 rel (observed <= 2.3e-15 on linear fits, <= 2.3e-8 "
        "overall)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_postestimation_stata_parity.py",
            "tests/reference_parity/_fixtures/postestimation_stata.json",
        ],
        "note": "Seven model blocks, sum and mixed contrasts with the constant. The "
        "ML-block gap is Stata's default ML convergence tolerance (see the "
        "sp.test record).",
    },
    "margins": {
        "status": "aligned",
        "reference": "Stata 18 margins, dydx(*)",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-6 rel (observed 2.7e-7 AME / 7.5e-8 SE on probit and "
        "logit-with-interaction; <= 1e-10 otherwise)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_postestimation_stata_parity.py",
            "tests/reference_parity/_fixtures/postestimation_stata.json",
        ],
        "note": "Average marginal effects with delta-method SEs after logit, probit, "
        "poisson, regress-with-interaction, logit-with-interaction. The 1e-7 "
        "gap is Stata's default ML convergence: with tolerances 1e-14 Stata's "
        "AMEs match StatsPAI to 2e-12 (probit) and 1e-13 (logit interaction). "
        "The fixture's glm_cloglog_margins block is not asserted by the test.",
    },
    "oster_bounds": {
        "status": "bit-exact",
        "reference": "Stata psacalc (Oster); R robomit::o_delta / o_beta",
        "reference_versions": {
            "Stata": "18",
            "psacalc": "2.1",
            "R": "R version 4.5.2 (2025-10-31)",
            "robomit": "1.0.7",
        },
        "tolerance": "1e-12 rel vs psacalc (observed 4.4e-14); 5e-7 abs vs robomit "
        "(it rounds to 6 dp)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "Data path: Oster's exact solution (quadratic at delta = 1, cubic "
        "otherwise, psacalc's root selection). Until 1.28 the function used "
        "her first-order approximation with the two R-squared gains swapped "
        "(delta* 1.205 vs psacalc 0.904). The summary-statistics path remains "
        "the (corrected) approximation and is analytical.",
    },
    "oster_delta": {
        "status": "bit-exact",
        "reference": "Stata psacalc (Oster); R robomit::o_delta / o_beta",
        "reference_versions": {
            "Stata": "18",
            "psacalc": "2.1",
            "R": "R version 4.5.2 (2025-10-31)",
            "robomit": "1.0.7",
        },
        "tolerance": "1e-12 rel vs psacalc (observed 4.4e-14); 5e-7 abs vs robomit",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "delta_star and beta_star_delta1 only (bootstrap SEs of the bounds "
        "are not pinned). The default r_max = 1.3 now means min(1, 1.3 "
        "R_full); before it was used as an R-squared of 1.3. Extra x_base "
        "entries = psacalc mcontrol() (verified).",
    },
    "ri_test": {
        "status": "bit-exact",
        "reference": "R ri2::conduct_ri (randomizr full enumeration); Stata ritest "
        "over the full assignment set",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "ri2": "0.5.0",
            "Stata": "18",
            "ritest": "1.1.7",
        },
        "tolerance": "p exact (k / N_assignments); statistic 1e-12 rel vs R, 2^-23 vs "
        "Stata",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "Difference in means, Welch t and KS; complete and cluster "
        "randomization.",
    },
    "rosenbaum_bounds": {
        "status": "bit-exact",
        "reference": "R DOS2::senWilcox (Rosenbaum); Stata rbounds; R "
        "stats::binom.test (sign test); R rbounds::psens "
        "(zero_method='wilcox', 4-dp)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "DOS2": "0.5.2",
            "rbounds": "2.2",
            "Stata": "18",
            "rbounds (Stata)": "1.1.6",
        },
        "tolerance": "bounding p-values 1e-12 rel (observed 4.6e-15); Stata sig- "
        "1e-15 abs; psens at its own 4-dp rounding",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "Fixed in this sweep: a continuity correction no reference applies, "
        "zeros dropped before ranking (DOS2 and Stata rank them with weight "
        "0; psens convention kept as zero_method='wilcox'), and a two-sided "
        "bound that evaluated the wrong tail for negative effects (p = 0 at "
        "Gamma = 3 where DOS2 gives 0.919).",
    },
    "rosenbaum_gamma": {
        "status": "bit-exact",
        "reference": "R DOS2::senWilcox (Rosenbaum); Stata rbounds",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "DOS2": "0.5.2",
            "Stata": "18",
            "rbounds (Stata)": "1.1.6",
        },
        "tolerance": "bounding p-values 1e-12 rel (observed 4.6e-15)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
        ],
        "note": "Alias of sp.rosenbaum_bounds (same object); called directly in "
        "test_rosenbaum_gamma_alias_and_long_format_agree.",
    },
    "subcluster_wild_bootstrap": {
        "status": "bit-exact",
        "reference": "Stata boottest, bootcluster() (CRVE clustered at g6, signs "
        "flipped at s12)",
        "reference_versions": {"Stata": "18", "boottest": "4.5.3"},
        "tolerance": "p exact (multiple of 1/4096); t 1e-10 rel (observed 1.8e-14)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "fwildclusterboot 0.14.3 refuses a bootcluster that is neither a "
        "clustering variable nor a regressor, so there is no R side. "
        "Rademacher only; the default Webb weights are sampled (not "
        "enumerable) and are not pinned.",
    },
    "test": {
        "status": "aligned",
        "reference": "Stata 18 test (after regress / ivregress / logit / poisson)",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-6 rel (observed <= 2.3e-15 on linear fits, <= 8e-10 on ML "
        "fits, far-tail p <= 4.7e-8)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_postestimation_stata_parity.py",
            "tests/reference_parity/_fixtures/postestimation_stata.json",
        ],
        "note": "Seven model blocks. The ML-block gap is Stata's default ML "
        "convergence tolerance: refitting with "
        "nrtolerance/tolerance/ltolerance 1e-14 closes the probit AME gap "
        "from 1e-7 to 2e-12 (scratch check, 2026-09-18).",
    },
    "wild_cluster_boot": {
        "status": "bit-exact",
        "reference": "R fwildclusterboot::boottest; Stata boottest (WCR, Rademacher, "
        "full enumeration)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "fwildclusterboot": "0.14.3",
            "Stata": "18",
            "boottest": "4.5.3",
        },
        "tolerance": "p exact (multiple of 1/4096); t 1e-10 rel (observed 1.9e-14)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "Result-object entry point (sp.regress fit) to the same WCR engine as "
        "sp.wild_cluster_bootstrap.",
    },
    "wild_cluster_bootstrap": {
        "status": "bit-exact",
        "reference": "R fwildclusterboot::boottest; Stata boottest (WCR, Rademacher, "
        "full enumeration)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "fwildclusterboot": "0.14.3",
            "Stata": "18",
            "boottest": "4.5.3",
        },
        "tolerance": "p exact (multiple of 1/4096); t 1e-10 rel (observed 1.9e-14)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "G = 12 and B >= 2^12, so both references and StatsPAI enumerate all "
        "4096 sign vectors (boottest's rule, adopted in this sweep). p = "
        "#{|t*| > |t|}/B, strict; before the sweep StatsPAI sampled and "
        "counted ties. Nonzero null (h0 = 0.2) included.",
    },
    "wild_cluster_ci_inv": {
        "status": "bit-exact",
        "reference": "R fwildclusterboot::boottest confidence interval (uniroot tol "
        "1e-13)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "fwildclusterboot": "0.14.3",
        },
        "tolerance": "CI endpoints 1e-9 rel (observed 4.8e-12)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
        ],
        "note": "Endpoints are the jumps of the enumerated p(h0) step function, found "
        "by bisection (was: linear interpolation on a 41-point grid, 0.7% "
        "off). Stata boottest's CI is T4: its Chandrupatla search returns "
        "early on step functions, and both reported endpoints are values its "
        "own test rejects (p = 204/4096, 202/4096); asserted in the Stata "
        "test.",
    },
    # ---- phase 3: panel / GLMM (12) ----
    "absorb_ols": {
        "status": "bit-exact",
        "reference": "fixest::feols 0.14.0 and Stata reghdfe (Track A 03_hdfe / "
        "15_hdfe_cluster goldens); Stata reghdfe with aweights, "
        "singleton dropping, two-way clustering",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "fixest": "0.14.0",
            "Stata": "18 MP",
        },
        "tolerance": "coefficients rtol 1e-12 (observed 2e-15), iid SEs 1e-12 "
        "(observed 8e-15), clustered SEs 1e-10 (observed 5.6e-11)",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_absorb_ols_parity.py"],
        "note": "Called directly on the committed Track A bytes (the modules go "
        "through sp.fast.feols / sp.hdfe_ols). Two-way clustering: "
        "cluster_df='min' reproduces reghdfe's G_min/(G_min-1) on every "
        "inclusion-exclusion term; the default per-term factor "
        "(sandwich::vcovCL) differs by 1.8e-4 / 1.2e-2 in variance here.",
    },
    "gmm": {
        "status": "bit-exact",
        "reference": "Stata 18 gmm (linear and exponential-mean IV; twostep, igmm, "
        "onestep); R gmm::gmm",
        "reference_versions": {
            "Stata": "18 MP",
            "R": "R version 4.5.2 (2025-10-31)",
            "gmm": "1.9.1",
        },
        "tolerance": "vs Stata estimates rtol 1e-10 (observed 1.2e-12), SEs 1e-9 "
        "linear / 1e-6 nonlinear (Stata's numerical Jacobian; observed "
        "2.5e-7), J rtol 1e-10; vs R rtol 1e-6 (observed 1.8e-7)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_panel_gmm_stata_parity.py",
            "tests/reference_parity/test_general_gmm_parity.py",
        ],
        "note": "Stata's two-step robust VCE keeps the estimation weight: "
        "sandwich_weight='estimation'; the default re-estimates S^-1 at the "
        "final estimate (R gmm), a documented 1/n-order difference (2.6e-6 "
        "linear, 4.4e-5 nonlinear here). Onestep J differs by construction "
        "(Stata recomputes an unadjusted weight). Nonlinear fits now end with "
        "Gauss-Newton steps.",
    },
    "icc": {
        "status": "bit-exact",
        "reference": "Stata 18 estat icc after mixed (ML, REML) and melogit; "
        "performance::icc; psych::ICC (balanced ANOVA identity)",
        "reference_versions": {
            "Stata": "18 MP",
            "R": "R version 4.5.2 (2025-10-31)",
            "performance": "0.16.0",
            "psych": "2.6.5",
        },
        "tolerance": "estimate / SE rtol 1e-6, logit-scale CI rtol 2e-6 (observed <= "
        "2.4e-7 / 7e-7 / 1.2e-6); balanced REML ICC = ANOVA ICC(1) rtol "
        "1e-9",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_icc_lrtest_parity.py"],
        "note": "Before 1.28.x the SE was a heuristic (var(log s2_u) ~ 2/n_groups, no "
        "covariance) and every GLMM returned NaN silently. Now the delta "
        "method on the observed information of the variance parameters and a "
        "logit-scale Wald CI, latent residual variance pi^2/3 after melogit / "
        "meologit.",
    },
    "interactive_fe": {
        "status": "bit-exact",
        "reference": "Stata regife (SSC, Gomez) ..., noconstant; R "
        "phtt::Eup(additive.effects = 'none')",
        "reference_versions": {
            "Stata": "18 MP",
            "regife": "2026-03-30 SSC",
            "R": "R version 4.5.2 (2025-10-31)",
            "phtt": "3.1.2",
        },
        "tolerance": "slopes rtol 1e-9 vs both (observed <= 2e-11); SEs rtol 1e-9 vs "
        "regife with dof='regife' (homoskedastic and cluster), phtt SE "
        "reconstructed rtol 1e-9",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_ife_parity.py"],
        "note": "SEs are Bai's D0 with Z = M_Lambda X M_F since 1.28.x (was M_F X "
        "only). Default dof counts r(N+T-r) absorbed parameters; "
        "regife/reghdfe count r(N+T). phtt's sig2.hat demeans residuals by "
        "unit and is not copied.",
    },
    "lrtest": {
        "status": "bit-exact",
        "reference": "Stata 18 lrtest; R anova() on lme4 ML fits",
        "reference_versions": {
            "Stata": "18 MP",
            "R": "R version 4.5.2 (2025-10-31)",
            "lme4": "2.0.1",
        },
        "tolerance": "chi2 rtol 1e-6, df exact, p rtol 1e-5 (observed chi2 <= 1e-9)",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_icc_lrtest_parity.py"],
        "note": "boundary=False reproduces Stata / R (naive chi2(df)); the default "
        "applies the chibar2 mixture, exact (Stram-Lee) for one added random "
        "effect under an unstructured covariance. df = difference in e(k); "
        "MixedResult.n_params no longer double-counts the residual variance.",
    },
    "megamma": {
        "status": "bit-exact",
        "reference": "glmmTMB 1.1.14 Gamma(link = 'log') (Laplace, AD Hessian); Stata "
        "18 meglm, family(gamma) link(log) intmethod(laplace)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "glmmTMB": "1.1.14",
            "TMB": "1.9.25",
            "Stata": "18 MP",
        },
        "tolerance": "vs glmmTMB estimates / SEs rtol 1e-6 (observed 6.9e-11 / "
        "3.8e-8); vs Stata Laplace estimates rtol 1e-6 (observed "
        "2.2e-7); objective identity rtol 1e-11",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_glmm_parity.py"],
        "note": "Before 1.28.x the Laplace used the Fisher curvature 1/phi (lme4's "
        "convention), a different approximation: _cons 0.6196 vs 0.6372. "
        "Stata's gamma SEs are not the Hessian of its own objective (5.5e-3 "
        "Laplace) and are not asserted.",
    },
    "meglm": {
        "status": "bit-exact",
        "reference": "Stata 18 meglm (gaussian; binomial with binomial()); "
        "lme4::lmer(REML = FALSE); lme4::glmer",
        "reference_versions": {
            "Stata": "18 MP",
            "R": "R version 4.5.2 (2025-10-31)",
            "lme4": "2.0.1",
        },
        "tolerance": "gaussian vs Stata estimates / SEs rtol 1e-6 (observed 4.2e-9 / "
        "2.8e-7), vs lmer ML estimates 6.1e-11; binomial-trials vs Stata "
        "2.4e-9 / 6.6e-8",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_glmm_parity.py"],
        "note": "Before 1.28.x family='gaussian' held the residual variance at 1 (x1 "
        "SE 0.0428 vs 0.0335); it is now estimated and the fit equals "
        "sp.mixed(method='ml'). SEs are the full OIM (Stata meglm); lmer's "
        "vcov is theta-conditional (1.2e-4 away) like sp.mixed / Stata mixed.",
    },
    "menbreg": {
        "status": "bit-exact",
        "reference": "glmmTMB 1.1.14 nbinom2 (Laplace, AD Hessian); Stata 18 menbreg "
        "intmethod(laplace) / mcaghermite(7)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "glmmTMB": "1.1.14",
            "TMB": "1.9.25",
            "lme4": "2.0.1",
            "Stata": "18 MP",
        },
        "tolerance": "vs glmmTMB estimates / SEs rtol 1e-6 (observed 1.7e-11 / "
        "2.3e-7); vs Stata Laplace estimates rtol 1e-6 (observed "
        "7.5e-8); objective identity at Stata's and glmmTMB's estimates "
        "rtol 1e-11",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_glmm_parity.py"],
        "note": "Laplace with the observed curvature (curvature='observed', the "
        "default since 1.28.x; Stata, glmmTMB). curvature='expected' is "
        "lme4's PIRLS Laplace and is pinned against glmer.nb (objective "
        "1.6e-10, estimates 7.3e-7). Stata's Laplace SEs are not the Hessian "
        "of its own objective (4.5e-4) and are not asserted; glmmTMB's AD "
        "Hessian agrees with ours to 2.3e-7.",
    },
    "meologit": {
        "status": "bit-exact",
        "reference": "Stata 18 meologit intmethod(mcaghermite) intpoints(7) / "
        "intmethod(laplace); ordinal::clmm",
        "reference_versions": {
            "Stata": "18 MP",
            "R": "R version 4.5.2 (2025-10-31)",
            "ordinal": "2025.12.29",
        },
        "tolerance": "AGHQ-7 vs Stata estimates rtol 1e-6 (observed 3.8e-7), SEs "
        "incl. cutpoints rtol 2e-6 (observed 1.3e-6; 5.7e-7 at Stata's "
        "estimates); objective identity vs Stata and clmm rtol 1e-11 "
        "(observed <= 1.4e-11)",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_glmm_parity.py"],
        "note": "Observed-curvature Laplace / AGHQ since 1.28.x (was Fisher); SEs are "
        "now the full OIM with delta-method threshold SEs (was the "
        "conditional information, no cutpoint SEs). clmm stops with gradient "
        "2e-3; Stata's Laplace-ologit SEs are not the Hessian of its own "
        "objective (5e-5); neither is asserted.",
    },
    "mepoisson": {
        "status": "bit-exact",
        "reference": "Stata 18 mepoisson, intmethod(laplace) / intmethod(mcaghermite) "
        "intpoints(7); lme4::glmer(nAGQ = 1 / 7)",
        "reference_versions": {
            "Stata": "18 MP",
            "R": "R version 4.5.2 (2025-10-31)",
            "lme4": "2.0.1",
        },
        "tolerance": "objective identity at Stata's estimates rtol 1e-11 (observed "
        "6.6e-12); Laplace estimates / SEs vs Stata rtol 1e-6 (observed "
        "9.0e-11 / 1.2e-7); AGHQ-7 vs lme4 rtol 1e-6 (observed 4.4e-8 / "
        "9.3e-8)",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_glmm_parity.py"],
        "note": "Laplace = Stata intmethod(laplace); nAGQ = k = "
        "intmethod(mcaghermite) intpoints(k). Stata's default mvaghermite is "
        "a different rule and is not compared. Stata's mcaghermite estimate "
        "stops 2.9e-5 short of the optimum of its own objective (same "
        "function to 6.6e-12, ours strictly larger); lme4 needs tolPwrss = "
        "1e-13 (its default 1e-7 leaves the Laplace fit 2.7e-4 away). The "
        "1.28.x fix finishing the optimum with Newton steps moved the Laplace "
        "estimate by 1.3e-4.",
    },
    "mixlogit": {
        "status": "bit-exact",
        "reference": "Stata mixlogit 1.4.0 (SSC, Hole), nrep(50) burn(15), on "
        "identical Halton draws",
        "reference_versions": {"Stata": "18 MP", "mixlogit": "1.4.0"},
        "tolerance": "means / SDs / Sigma / SEs rtol 1e-6 (observed <= 2.2e-7), "
        "log-likelihood rtol 1e-10 (observed 5e-13)",
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_panel_mixlogit_parity.py"],
        "note": "Deterministic, not Monte-Carlo: n_draws=50, halton_burn=15, "
        "halton_shift=False builds Stata's draw matrix invnormal(halton(50, "
        "k, 1 + 15 + 50(n-1))), so both maximise the same simulated "
        "likelihood. oim (robust=False), robust (x N/(N-1)), lognormal ln(1) "
        "and corr (compared on Sigma) all covered. With the default shifted "
        "draws the two are different simulators.",
    },
    "xtnbreg": {
        "status": "bit-exact",
        "reference": "Stata 18 xtnbreg, fe / re (Hausman-Hall-Griliches); "
        "pglm::pglm(family = negbin, model = 'within' / 'random')",
        "reference_versions": {
            "Stata": "18 MP",
            "R": "R version 4.5.2 (2025-10-31)",
            "pglm": "0.2.4",
        },
        "tolerance": "coefficients / SEs rtol 1e-7 (observed <= 2.7e-8: Stata fe's "
        "score at its own estimate is 3e-7, pglm's gradtol), "
        "log-likelihood rtol 1e-12",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_xtnbreg_parity.py"],
        "note": "Before 1.28.x model='fe' fitted an unconditional dummy-variable NB-2 "
        "labelled 'xtnbreg, fe' and model='re' the normal random-intercept "
        "NB-2 GLMM; both remain as model='ufe' / 'normal_re'. Conditional FE "
        "drops all-zero and singleton panels as Stata does (N = 228 of 240).",
    },
    # ---- phase 3: treatment effects (16) ----
    "aipw": {
        "status": "bit-exact",
        "reference": "Stata teffects aipw (ATE, POmeans); R AIPW::AIPW 0.6.9.3 "
        "stratified_fit(k_split = 1)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "AIPW": "0.6.9.3",
            "SuperLearner": "2.0.40",
            "Stata": "18",
        },
        "tolerance": "1e-10 rel on ATE, potential-outcome means and SEs (observed <= "
        "1.4e-14)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "Full-sample nuisances (cross_fit=False). Stata's robust SE is the "
        "stacked M-estimation sandwich (se_method='sandwich'); R AIPW reports "
        "sd(EIF)/sqrt(n) (se_method='influence'). AIPW's stratified_fit() "
        "omits Q.model = FALSE for the propensity, so the R side forces "
        "SL.glm to binomial. R AIPW's ATT divides its control term by P(A = "
        "0); the test rebuilds that number from the same nuisances, and the "
        "DR ATT is cross-checked against DoubleML's ATTE score.",
    },
    "dose_response": {
        "status": "bit-exact",
        "reference": "Stata doseresponse / gpscore (Hirano-Imbens normal GPS, "
        "quadratic T and GPS with interaction)",
        "reference_versions": {"Stata": "18", "doseresponse": "SSC"},
        "tolerance": "1e-9 rel on the dose-response function at 5 doses (observed "
        "5.1e-10)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "treatment_model=LinearRegression(), "
        "outcome_model=PolynomialFeatures(2)+LinearRegression(); the default "
        "GBM path is not pinned. R causaldrf::hi_est uses the n-p sigma "
        "(convention gap).",
    },
    "four_way_decomposition": {
        "status": "bit-exact",
        "reference": "Stata med4way (yreg/mreg linear); CMAverse::cmest 0.1.0 (rb, "
        "paramfunc, delta)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "CMAverse": "0.1.0",
            "Stata": "18",
        },
        "tolerance": "components 1e-12 rel; delta SEs 1e-12 rel vs CMAverse "
        "(vcov='ols'); variances 5e-9 rel vs med4way (vcov='ml')",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "med4way fits the linear models by ML (ml maximize), so its residual "
        "variances stop at ml's tolerance (observed 6.6e-10).",
    },
    "g_estimation": {
        "status": "bit-exact",
        "reference": "DTRreg::DTRreg 2.4 (method = 'gest', treat.type = 'bin', weight "
        "= 'none')",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "DTRreg": "2.4"},
        "tolerance": "1e-10 rel on each stage psi (observed 4.6e-15)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "Constant blips, logit propensity (default propensity_model); with "
        "and without separate propensity_covariates.",
    },
    "horowitz_manski": {
        "status": "bit-exact",
        "reference": "Stata tebounds 1.8 worst-case bounds (identical estimand for "
        "ATE worst-case bounds)",
        "reference_versions": {"Stata": "18", "tebounds": "1.8 (SJ15-2 st0386)"},
        "tolerance": "1e-12 rel / 1e-15 abs on both bounds",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "Averaging stratum worst-case bounds is the unconditional bound "
        "exactly; also asserted with a one-arm stratum.",
    },
    "ipcw": {
        "status": "bit-exact",
        "reference": "survival::coxph(ties = 'breslow') + basehaz(centered = FALSE)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "survival": "3.8.3",
        },
        "tolerance": "1e-9 rel on every weight (observed 2.0e-11)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "method='cox_ph' only. The default method='pooled_logistic' is a "
        "complete-case IPW of the observed indicator and has no reference.",
    },
    "lee_bounds": {
        "status": "bit-exact",
        "reference": "Stata leebounds 1.5 (Tauchmann), vce(analytic)",
        "reference_versions": {
            "Stata": "18",
            "leebounds": "1.5 (2013-07-17, Tauchmann)",
        },
        "tolerance": "1e-10 rel on bounds and analytic variances",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "trimming='quantile' matches the shipped upper bound; "
        "trimming='exact' matches both bounds once the thresholds are held "
        "exactly (%21x). As shipped, leebounds keeps its threshold in a "
        "15-digit local macro, so its tie branch never runs and its lower "
        "bound drops the quantile observation. The test rebuilds that number "
        "from the rounded threshold (reference defect, T4 for that one "
        "number).",
    },
    "ltmle": {
        "status": "bit-exact",
        "reference": "ltmle::ltmle 1.3-0 (glm, gbounds c(0.01, 1), variance.method = "
        "'ic')",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "ltmle": "1.3.0"},
        "tolerance": "psi1 / psi0 / ATE / SE 1e-9 rel without censoring (observed "
        "5e-13); 1e-8 with censoring nodes (observed 5e-10)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "Binary and continuous outcomes; the censored rows are limited by R "
        "glm's default deviance tolerance.",
    },
    "manski_bounds": {
        "status": "bit-exact",
        "reference": "Stata tebounds 1.8 (SJ15-2 st0386), erates(0)",
        "reference_versions": {"Stata": "18", "tebounds": "1.8 (SJ15-2 st0386)"},
        "tolerance": "1e-12 rel / 1e-15 abs on both bounds",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "Worst case, MTS (positive selection), MTS+MTR; binary outcome in [0, "
        "1].",
    },
    "mediate_interventional": {
        "status": "bit-exact",
        "reference": "CMAverse::cmest 0.1.0 (gformula with postc: rpnde / rpnie / te; "
        "rb paramfunc without)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "CMAverse": "0.1.0",
        },
        "tolerance": "IIE / IDE / total 1e-9 rel (observed 2.3e-15)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "Point estimates only; SEs are bootstrap on both sides.",
    },
    "msm": {
        "status": "bit-exact",
        "reference": "ipw::ipwtm + lm / glm(quasibinomial) + sandwich::vcovCL(type = "
        "'HC1')",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "ipw": "1.3.0",
            "sandwich": "3.1.1",
        },
        "tolerance": "coefficients and cluster SEs 1e-9 rel (observed 1.5e-13); Stata "
        "regress / logit [pw] 1e-7",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "Cumulative, ever and current exposures; gaussian and binomial. "
        "Stata's logit cluster SE omits (N-1)/(N-k), which the test applies "
        "explicitly; the Stata side is at 1e-7 because its logits stop at "
        "Stata's default tolerance.",
    },
    "multi_treatment": {
        "status": "bit-exact",
        "reference": "Stata teffects aipw with a multivalued treatment (mlogit "
        "propensity)",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-10 rel on both contrasts, potential-outcome means and "
        "sandwich SEs",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "outcome_model='linear', se_method='sandwich'. The default GBM "
        "outcome model is not pinned.",
    },
    "principal_strat": {
        "status": "bit-exact",
        "reference": "AER::ivreg 1.2.16 (Wald LATE); SACE bounds via Stata leebounds",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "AER": "1.2.16"},
        "tolerance": "1e-10 rel on the monotonicity complier LATE",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "method='monotonicity'. The principal-score path is not pinned.",
    },
    "stabilized_weights": {
        "status": "bit-exact",
        "reference": "ipw::ipwtm 1.3.0 (type = 'all'; binomial logit; gaussian via "
        "geepack::geeglm)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "ipw": "1.3.0",
            "geepack": "1.3.13",
        },
        "tolerance": "1e-10 rel on every weight (observed 1.5e-12)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "Gaussian treatment: density_sd='ml' (geeglm's dispersion is RSS / N; "
        "the default divides by N - k).",
    },
    "survivor_average_causal_effect": {
        "status": "bit-exact",
        "reference": "Stata leebounds 1.5 (Zhang-Rubin SACE bounds = Lee bounds under "
        "monotonicity)",
        "reference_versions": {
            "Stata": "18",
            "leebounds": "1.5 (2013-07-17, Tauchmann)",
        },
        "tolerance": "1e-10 rel (upper bound; both bounds equal sp.lee_bounds)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
            "tests/reference_parity/_fixtures/teffects_Stata.json",
        ],
        "note": "Outcome missing for non-survivors, as under truncation by death.",
    },
    # ---- phase 3: spatial / survey / frontier / structural (16) ----
    "block_weights": {
        "status": "bit-exact",
        "reference": "spdep::nb2blocknb(NULL, ID) + nb2listw(style = 'W')",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "spdep": "1.4.2"},
        "tolerance": "neighbour sets exact; weights 1e-12 rel (observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_spatial_survey_R_parity.py",
            "tests/reference_parity/_fixtures/spatial_survey_R.json",
        ],
        "note": "Four regimes (POLYID %% 4) on Columbus. Regenerate via "
        "_generate_spatial_survey_R.R.",
    },
    "blp": {
        "status": "aligned",
        "reference": "pyblp 1.2.0 (Conlon & Gortmaker), identical Halton nodes via "
        "agent_data",
        "reference_versions": {"pyblp": "1.2.0", "numpy": "2.2.6", "python": "3.10.20"},
        "tolerance": "beta, sigma, SEs, objective/N, own elasticities 1e-6 rel "
        "(observed <= 1.6e-8)",
        "sides": ["py"],
        "test": [
            "tests/reference_parity/test_blp_pyblp_parity.py",
            "tests/reference_parity/_fixtures/blp_pyblp.json",
        ],
        "note": "Python cross-package reference, not R or Stata. Two-step GMM with "
        "center_moments=False. sp gmm_objective = N * pyblp objective. "
        "Random-price-coefficient elasticities are pinned at fixed parameters "
        "to 1e-8.",
    },
    "gwr": {
        "status": "bit-exact",
        "reference": "GWmodel::gwr.basic 2.4.1",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "GWmodel": "2.4.1"},
        "tolerance": "local betas and SEs 1e-9 rel (observed 1.3e-11 / 3.5e-14); RSS, "
        "AIC, AICc, BIC, enp, edf 1e-10 rel",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_spatial_survey_R_parity.py",
            "tests/reference_parity/_fixtures/spatial_survey_R.json",
        ],
        "note": "Georgia, bisquare / gaussian / exponential x adaptive / fixed, "
        "fractional bw and bw > n. Fixed on the way: the exponential kernel "
        "was truncated, the adaptive Gaussian / exponential were "
        "k-NN-truncated, and fractional bw was rounded up. Local R^2 is a "
        "documented difference: StatsPAI follows mgwr (local weighted mean). "
        "GWmodel uses the global mean and, under an adaptive kernel, the "
        "transposed weight matrix; both are reconstructed and asserted. "
        "Regenerate via _generate_spatial_survey_R.R.",
    },
    "gwr_bandwidth": {
        "status": "bit-exact",
        "reference": "GWmodel::bw.gwr 2.4.1 (golden section gold(); gwr.aic / gwr.cv)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "GWmodel": "2.4.1"},
        "tolerance": "selected bandwidth 1e-10 rel (observed 0 on 8 configurations); "
        "criterion values 1e-11 rel (observed 7.0e-15)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_spatial_survey_R_parity.py",
            "tests/reference_parity/_fixtures/spatial_survey_R.json",
        ],
        "note": "AICc and CV x bisquare and gaussian x adaptive and fixed. The CV "
        "criterion used to be in-sample RSS (always the smallest bandwidth). "
        "The search now transcribes GWmodel's gold() because the criterion is "
        "not unimodal on the neighbour lattice. Regenerate via "
        "_generate_spatial_survey_R.R.",
    },
    "kernel_weights": {
        "status": "bit-exact",
        "reference": "spdep::nb2listwdist(type = 'dpd', alpha = 2) on dnearneigh(0, "
        "h); GWmodel::gw.weight",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "spdep": "1.4.2",
            "GWmodel": "2.4.1",
        },
        "tolerance": "weights 1e-12 rel / 1e-15 abs (observed 6.9e-16 rel, 3.3e-16 "
        "abs)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_spatial_survey_R_parity.py",
            "tests/reference_parity/_fixtures/spatial_survey_R.json",
        ],
        "note": "Fixed bisquare = dpd alpha 2, raw and row-standardised. The W "
        "comparison found W.transform rebuilding the weights from 1.0, which "
        "silently turned kernel weights binary. Fixed Gaussian, fixed "
        "bisquare and adaptive bisquare (k = gw.weight bw k + 1) are compared "
        "against gw.weight. The adaptive Gaussian keeps only the k nearest "
        "neighbours (a documented convention; GWmodel weights all points), "
        "asserted as such. Regenerate via _generate_spatial_survey_R.R.",
    },
    "lcsf": {
        "status": "aligned",
        "reference": "sfaR::sfalcmcross 1.0.1 (2 classes, half-normal)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "sfaR": "1.0.1"},
        "tolerance": "estimates 1e-6 rel (observed 1.1e-8); OIM SEs 1e-6 rel "
        "(observed 8.6e-8)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_frontier_struct_R_parity.py",
            "tests/reference_parity/_fixtures/frontier_struct_R.json",
        ],
        "note": "sfaR reports log variances; compared as ln_sigma = Zu/2. Classes "
        "matched by ascending sigma_u. Class-1 sigma_u is weakly identified, "
        "so both sides are optimiser-limited; the fixture keeps the best of "
        "four sfaR optimisers by max |gradient|, and StatsPAI's "
        "log-likelihood is at least as high. Production-with-z and "
        "cost-without-z cases.",
    },
    "linear_calibration": {
        "status": "bit-exact",
        "reference": "survey::calibrate(calfun='linear', unbounded); Stata svycal "
        "regress",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "survey": "4.5",
            "Stata": "18 MP",
        },
        "tolerance": "calibrated weights 1e-12 rel (observed 5.5e-15)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survey_calib_R_parity.py",
            "tests/reference_parity/_fixtures/survey_calib_R.json",
            "tests/reference_parity/_fixtures/survey_calib_stata.json",
        ],
        "note": "No intercept added: ~0 + income + age, and ~sex + income via one / "
        "male columns. Closed-form chi-squared projection, also asserted "
        "reference-free. Regenerate via _generate_survey_calib_R.R and "
        "_fixtures/_generate_survey_calib_stata.do.",
    },
    "markup": {
        "status": "bit-exact",
        "reference": "Stata markupest 1.0.1 (Rovigatti), method(dlw) pmethod(lp)",
        "reference_versions": {"Stata": "18", "markupest": "1.0.1 10May2020"},
        "tolerance": "1e-10 rel (observed 8.1e-13 corrected, 3.4e-15 uncorrected)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_frontier_struct_R_parity.py",
            "tests/reference_parity/_fixtures/frontier_struct_stata.json",
        ],
        "note": "Markup of the free input l on the prodest parity panel; the "
        "eta-corrected and uncorrected shares are both pinned. Compared on "
        "the 2084 firm-years in StatsPAI's production sample; Stata also "
        "returns the 281 first-year rows.",
    },
    "queen_weights": {
        "status": "bit-exact",
        "reference": "spdep::poly2nb(queen = TRUE) + nb2listw(style = 'W' / 'S' / "
        "'U')",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "spdep": "1.4.2",
            "sf": "1.1.1",
            "spData": "2.3.5",
        },
        "tolerance": "neighbour sets exact; weights 1e-12 rel (observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_spatial_survey_R_parity.py",
            "tests/reference_parity/_fixtures/spatial_survey_R.json",
        ],
        "note": "Columbus polygons (spData) as WKT at 17 digits, read by both sides. "
        "transform 'R' / 'V' / 'D' = nb2listw styles W / S / U. The 'V' "
        "comparison exposed a missing n / Q rescale. Regenerate via "
        "_generate_spatial_survey_R.R.",
    },
    "rake": {
        "status": "bit-exact",
        "reference": "survey::rake (to its fixed point) and "
        "survey::calibrate(calfun='raking'); Stata svycal rake",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "survey": "4.5",
            "Stata": "18 MP",
        },
        "tolerance": "calibrated weight shares 1e-12 rel at tol=1e-14 (observed "
        "1.5e-15); 1e-8 at the default tol=1e-10 (observed 1.0e-10)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survey_calib_R_parity.py",
            "tests/reference_parity/_fixtures/survey_calib_R.json",
            "tests/reference_parity/_fixtures/survey_calib_stata.json",
        ],
        "note": "sp.rake returns weights summing to 1; references divided by their "
        "sum (N = 10000). IPF and Newton raking share one fixed point (R's "
        "two routes agree to 2e-15). Fixed here: the convergence test was an "
        "absolute change on O(1/n) weights (4.3e-4 margin error at n = "
        "100000). Weights only: fed to sp.svydesign they give the "
        "fixed-weights SE (matches R/Stata), not the calibration-adjusted SE. "
        "Regenerate via _generate_survey_calib_R.R and "
        "_fixtures/_generate_survey_calib_stata.do.",
    },
    "rook_weights": {
        "status": "bit-exact",
        "reference": "spdep::poly2nb(queen = FALSE) + nb2listw(style = 'W')",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "spdep": "1.4.2",
            "sf": "1.1.1",
            "spData": "2.3.5",
        },
        "tolerance": "neighbour sets exact; weights 1e-12 rel (observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_spatial_survey_R_parity.py",
            "tests/reference_parity/_fixtures/spatial_survey_R.json",
        ],
        "note": "Columbus: 236 queen links vs 200 rook links, so the fixture "
        "discriminates the two criteria. Regenerate via "
        "_generate_spatial_survey_R.R.",
    },
    "sarar_gmm": {
        "status": "aligned",
        "reference": "spatialreg::gstsls 1.4.3 (Kelejian-Prucha GS2SLS)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "spatialreg": "1.4.3",
            "spdep": "1.4.2",
        },
        "tolerance": "beta, rho and SEs 1e-9 rel (observed 3.8e-11 / 3.0e-11); lambda "
        "1e-7 rel (observed 6.5e-9)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_spatial_survey_R_parity.py",
            "tests/reference_parity/_fixtures/spatial_survey_R.json",
        ],
        "note": "Default, robust = TRUE (HC0) and sig2n_k = TRUE. lambda is the exact "
        "admissible minimiser of the KP (1999) moment objective; gstsls's "
        "nlminb stops 6.5e-9 short on this flat objective (mpmath check). "
        "Rewritten in this sweep: the old path filtered the instruments and "
        "reported stage-1 SEs next to stage-3 estimates. sphet::spreg(model = "
        "'sarar') is the KP (2010) weighted estimator, a different estimator. "
        "Regenerate via _generate_spatial_survey_R.R.",
    },
    "spatial_iv": {
        "status": "bit-exact",
        "reference": "sphet::spreg(model = 'lag', het = TRUE) 2.1.1",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "sphet": "2.1.1",
            "spdep": "1.4.2",
        },
        "tolerance": "coefficients and HC0 SEs 1e-10 rel (observed 9.5e-14 / 6.8e-14)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_spatial_survey_R_parity.py",
            "tests/reference_parity/_fixtures/spatial_survey_R.json",
        ],
        "note": "Columbus: CRIME on INC with HOVAL endogenous and DISCBD excluded "
        "(not lagged), row-standardised queen W. The docstring used to claim "
        "Conley HAC SEs; the code, and sphet, report White / HC0. Regenerate "
        "via _generate_spatial_survey_R.R.",
    },
    "spatial_panel": {
        "status": "aligned",
        "reference": "splm::spml(model = 'within') 1.6.5; Stata xsmle 1.4.5 fe "
        "type(ind) vce(oim)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "splm": "1.6.5",
            "plm": "2.6.7",
            "Stata": "18",
            "xsmle": "version 1.4.5 5jun2017",
        },
        "tolerance": "vs splm: estimates and SEs 1e-7 rel (observed 7.2e-8 / 1.3e-8, "
        "the splm optimize floor); vs xsmle (tightened ml tolerances): "
        "estimates and beta SEs 1e-9 (observed 2.5e-13 / 8.5e-11), "
        "spatial-parameter SE 1e-8 (observed 3.0e-9)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_spatial_survey_R_parity.py",
            "tests/reference_parity/_fixtures/spatial_survey_R.json",
            "tests/reference_parity/_fixtures/spatial_survey_stata.json",
        ],
        "note": "Produc / usaww. SAR, SEM and SDM x individual and two-way effects "
        "against splm (SDM = SAR with per-period W-lagged raw X). Stata "
        "covers entity effects only: for two-way SAR / SDM, xsmle lags the "
        "within transform of W y and reports non-convergence, a documented "
        "reference disagreement. vce = 'information' (splm, default) or 'oim' "
        "(xsmle). Regenerate via _generate_spatial_survey_R.R and "
        "_fixtures/_generate_spatial_survey_stata.do.",
    },
    "svydesign": {
        "status": "bit-exact",
        "reference": "survey::svydesign + svymean/svytotal/svyglm/degf (strata, "
        "nested PSUs, fpc, survey.lonely.psu); Stata svyset + svy: "
        "mean/total/regress/logit/poisson + estat effects",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "survey": "4.5",
            "Stata": "18 MP",
        },
        "tolerance": "estimates, SEs, DEFF, CI bounds 1e-10 rel, p-values 1e-9 rel "
        "(observed <= 4e-14; p 6e-13)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_survey_design_R_parity.py",
            "tests/reference_parity/_fixtures/survey_design_R.json",
            "tests/reference_parity/_fixtures/survey_design_stata.json",
        ],
        "note": "Stratified clustered design with PSU ids repeating across strata, "
        "unequal weights, fpc as PSU counts / fractions / element counts, "
        "cluster-only and element designs, four lonely-PSU rules. GLM df: "
        "dof='design' = Stata e(df_r), dof='residual' = R summary.svyglm. "
        "DEFF: deff='wor' = R deff=TRUE, 'replace' = Stata without fpc. R "
        "svyglm references refitted from the converged coefficients (one-pass "
        "glm.fit weights are one step stale, ~2e-7 in the logit SE). Fixed "
        "here: fpc counts divided by elements not PSUs (NaN SE), df on "
        "non-nested ids, logit/poisson bread, scale-dependent DEFF. "
        "Regenerate via _generate_survey_design_R.R and "
        "_fixtures/_generate_survey_design_stata.do.",
    },
    # ---- phase 3: RD / IV / rlasso (24) ----
    "effective_f_test": {
        "status": "bit-exact",
        "reference": "Stata weakivtest (Montiel Olea & Pflueger) after ivreg2; R "
        "ivDiag::eff_F 1.0.6",
        "reference_versions": {
            "Stata": "18 MP",
            "weakivtest": "10/28/2020",
            "ivreg2": "4.1.12",
            "R": "R version 4.5.2 (2025-10-31)",
            "ivDiag": "1.0.6",
        },
        "tolerance": "F_eff rel 1e-9 on 6 designs (observed 3.7e-14 vs Stata, 1.3e-12 "
        "vs ivDiag for k = 1)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_rd_iv_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_Stata.json",
            "tests/reference_parity/_fixtures/rd_iv_R.json",
        ],
        "note": "k = 1 and 3 instruments, HC1 and clustered, on ivDiag::rueda and a "
        "seeded weak-IV design. ivDiag's k > 1 effective F uses the "
        "un-partialled Z'Z and differs by 13-16% from weakivtest and StatsPAI "
        "(asserted as a reference disagreement).",
    },
    "iv_diag": {
        "status": "bit-exact",
        "reference": "R ivDiag::ivDiag 1.0.6 (analytic block)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "ivDiag": "1.0.6",
            "lfe": "3.1.1",
        },
        "tolerance": "2SLS / OLS coefficients and SEs, classical first-stage F, "
        "effective F, tF critical value and interval: rel 1e-9 on 6 "
        "designs (observed <= 6e-12)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_iv_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_R.json",
        ],
        "note": "Bootstrap, CLR / K sets and LTZ are not part of this record. Fixed "
        "on the way: tF indexed by the homoskedastic F and reported for k > "
        "1; se_ols ignored vcov and cluster.",
    },
    "jive": {
        "status": "bit-exact",
        "reference": "Stata jive 1.0.2 (Stata Journal st0108) ujive1 / ujive2",
        "reference_versions": {"Stata": "18 MP", "jive": "1.0.2"},
        "tolerance": "coefficients and SEs (default and robust) rel 1e-9 (observed "
        "3.0e-13 / 4.4e-13)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_rd_iv_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_Stata.json",
        ],
        "note": "variant='jive1' = ujive1, 'jive2' = ujive2. Through 1.28.0 jive2 was "
        "fitted/(1-h) (not a jackknife instrument) and the default SE omitted "
        "the IV sandwich (half of Stata's). A brute-force leave-one-out "
        "reconstruction is asserted alongside.",
    },
    "mccrary_test": {
        "status": "bit-exact",
        "reference": "R rdd::DCdensity 0.57 (CRAN archive)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "rdd": "0.57"},
        "tolerance": "theta, se, z rel 1e-9; p rel 1e-8; bin width 1e-12 (observed "
        "6.5e-12)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_iv_rd_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_rd_R.json",
        ],
        "note": "Default and fixed bin/bandwidth, and the Senate data. Through 1.28.0 "
        "a different estimator with silent fallbacks.",
    },
    "multi_cutoff_rd": {
        "status": "bit-exact",
        "reference": "rdmulti::rdmc 2.0.0 (Cattaneo, Titiunik, Vazquez-Bare & Keele)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "rdmulti": "2.0.0"},
        "tolerance": "identical to sp.rdmc on the fixture (exact); per-cutoff and "
        "pooled estimates vs R 1e-9 rel",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rdmulti_parity.py",
            "tests/reference_parity/_fixtures/rdmulti_R.json",
        ],
        "note": "Alias of sp.rdmc (returns rdmc(*args, **kwargs)); asserted equal to "
        "rdmc and to R in test_multi_cutoff_rd_is_rdmc_on_the_r_fixture.",
    },
    "rd_bias_aware_fuzzy": {
        "status": "aligned",
        "reference": "R RDHonest 1.0.1.9000, RDHonest(y | d ~ x)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "RDHonest": "1.0.1.9000",
        },
        "tolerance": "estimate, std.error, maximum.bias, conf.low/high, M, first "
        "stage: rel 1e-9 at fixed h and M (observed 1.1e-14), 1e-6 with "
        "selected h (observed 3.5e-8, optimiser)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_iv_rd_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_rd_R.json",
        ],
        "note": "The record covers every ingredient and RDHonest's linearised "
        "interval (model_info['bias_aware']['rdhonest']). The headline "
        "Anderson-Rubin-type set has no reference implementation; it is "
        "pinned by its defining identity |T(t)| = cv(b(t)) at both endpoints. "
        "Through 1.28.0 the bias bound was h^2 M / 12 (below the Holder worst "
        "case).",
    },
    "rdbwhte": {
        "status": "bit-exact",
        "reference": "R rdhte::rdbwhte 0.2.0",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "rdhte": "0.2.0",
            "rdrobust": "4.0.0",
        },
        "tolerance": "bandwidths rel 1e-8 (continuous and per-subgroup)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_iv_rd_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_rd_R.json",
        ],
        "note": "rdrobust::rdbwselect on x (per subgroup for a 0/1 moderator).",
    },
    "rdhte": {
        "status": "bit-exact",
        "reference": "R rdhte::rdhte 0.2.0 (sandwich 3.1.1)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "rdhte": "0.2.0",
            "sandwich": "3.1.1",
            "rdrobust": "4.0.0",
        },
        "tolerance": "coef, coef.bc, se.rb, vcov rel 1e-9 (observed 2.2e-12); "
        "bandwidths 1e-8",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_iv_rd_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_rd_R.json",
        ],
        "note": "11 cells: continuous and 0/1-subgroup moderators, HC0-HC3, CR1, p = "
        "2, three kernels, selected and fixed bandwidths. Through 1.28.0 "
        "inference was conventional, h a rule of thumb and binary z not "
        "subgroups.",
    },
    "rdhte_lincom": {
        "status": "bit-exact",
        "reference": "R rdhte::rdhte_lincom 0.2.0",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "rdhte": "0.2.0"},
        "tolerance": "estimate, z, CI, joint chi-square rel 1e-9; p 1e-8",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_iv_rd_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_rd_R.json",
        ],
        "note": "Subgroup difference and continuous CATE at z = 1 (linfct=).",
    },
    "rdplot": {
        "status": "bit-exact",
        "reference": "R rdrobust::rdplot 4.0.0",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "rdrobust": "4.0.0",
        },
        "tolerance": "J / J_IMSE / J_MV and bin counts exact; bin means, SEs, t "
        "intervals and polynomial values rel 1e-9 (observed 4.0e-12); p "
        "= 4 global polynomial coefficients rel 1e-8 (observed 1.6e-10, "
        "raw-scale conditioning)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_iv_rd_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_rd_R.json",
        ],
        "note": "All eight binselect rules, manual bins with a triangular kernel, "
        "covariates, and the Senate data (missing outcomes, mass points). "
        "Numbers returned on fig.rdplot_data. Through 1.28.0 the bin count "
        "was a rule of thumb (esmv 17/16 bins vs R's 38/41) and kernel was "
        "ignored.",
    },
    "rdplotdensity": {
        "status": "bit-exact",
        "reference": "R rddensity::rdplotdensity 2.6 (lpdensity 2.5)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "rddensity": "2.6",
            "lpdensity": "2.5",
        },
        "tolerance": "f_p, f_q, se_p, se_q at every grid point rel 1e-8 (observed "
        "8.0e-12); nh exact",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_iv_rd_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_rd_R.json",
        ],
        "note": "Mass-point data and the Senate margin. Through 1.28.0 each side used "
        "its own ECDF (density at the cutoff 0.60 vs R's 0.23 on the "
        "fixture).",
    },
    "rdrbounds": {
        "status": "aligned",
        "reference": "R rdlocrand::rdrbounds 2.0",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "rdlocrand": "2.0"},
        "tolerance": "upper and lower bounds within 4 pooled binomial SEs at 4000 "
        "draws (Monte Carlo on both sides, not parity)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_iv_rd_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_rd_R.json",
        ],
        "note": "T3. Through 1.28.0 only the median threshold was used: upper bounds "
        "0.012 / 0.056 / 0.159 vs R 0.021 / 0.094 / 0.346 "
        "(anti-conservative).",
    },
    "rdsensitivity": {
        "status": "aligned",
        "reference": "R rdlocrand::rdsensitivity / rdrandinf 2.0",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "rdlocrand": "2.0"},
        "tolerance": "per-window estimates rel 1e-9 (rdrandinf observed statistic); "
        "randomization p-values within 4 pooled binomial SEs at 4000 "
        "draws (Monte Carlo on both sides, not parity)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_iv_rd_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_rd_R.json",
        ],
        "note": "T3 for the p-values. R's rdsensitivity reports a p-value surface "
        "over tau; StatsPAI's tau = 0 column is compared.",
    },
    "rkd": {
        "status": "bit-exact",
        "reference": "R rdrobust::rdrobust(deriv = 1, vce = 'hc1') 4.0.0; Stata "
        "rdrobust, deriv(1) 11.1.0",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "rdrobust": "4.0.0",
            "Stata": "18 MP",
            "rdrobust (Stata)": "11.1.0",
        },
        "tolerance": "conventional and robust estimate / SE, bandwidth: rel 1e-9 "
        "(observed 5.2e-12 vs R, 6.0e-11 vs Stata)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_rd_iv_rd_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_rd_R.json",
            "tests/reference_parity/_fixtures/rd_iv_Stata.json",
        ],
        "note": "sp.rkd is rdrobust(deriv=1, vce='hc1') with the conventional row as "
        "headline. Through 1.28.0 its default bandwidth was a rule of thumb "
        "and the fuzzy SE dropped the kink covariance (3.6% on the fixture). "
        "Sharp, fuzzy, clustered, fixed and selected bandwidth.",
    },
    "rlasso": {
        "status": "bit-exact",
        "reference": "R hdm::rlasso 0.3.2",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "hdm": "0.3.2"},
        "tolerance": "support exact; beta / sigma / loadings / residuals atol 1e-6, "
        "lambda0 rtol 1e-8 (observed rel <= 1.8e-12)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rlasso_parity.py",
            "tests/reference_parity/_fixtures/rlasso_R.json",
        ],
        "note": "Evidence existed since the hdm port; the tests called "
        "statspai.rlasso.* rather than sp.*, so no record was built. Four "
        "specifications (post / intercept / homoscedastic).",
    },
    "rlasso_effect": {
        "status": "bit-exact",
        "reference": "R hdm::rlassoEffect 0.3.2",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "hdm": "0.3.2"},
        "tolerance": "alpha / se atol 1e-6 (observed rel <= 3.3e-15), incl. hdm's "
        "GrowthData vignette",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rlasso_parity.py",
            "tests/reference_parity/test_rlasso_vignette_parity.py",
            "tests/reference_parity/_fixtures/rlasso_R.json",
        ],
        "note": "Partialling out and double selection.",
    },
    "rlasso_effects": {
        "status": "bit-exact",
        "reference": "R hdm::rlassoEffects 0.3.2",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "hdm": "0.3.2"},
        "tolerance": "alpha / se rtol 1e-9 on cps2012 (observed 2.8e-12); atol 1e-6 "
        "on the synthetic fixture",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rlasso_parity.py",
            "tests/reference_parity/test_rlasso_vignette_parity.py",
            "tests/reference_parity/_fixtures/rlasso_vignette_R.json",
        ],
        "note": "15 identified cps2012 targets pinned; the 16th (female:hsd08) is "
        "unidentified on both sides and now warns.",
    },
    "rlasso_iv": {
        "status": "bit-exact",
        "reference": "R hdm::rlassoIV 0.3.2",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "hdm": "0.3.2"},
        "tolerance": "coef / se atol 1e-6 (observed <= 3e-15); EminentDomain atol "
        "1e-4 (observed 6.1e-9: pseudo-inverse of a rank-deficient "
        "control block)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rlasso_parity.py",
            "tests/reference_parity/test_rlasso_vignette_parity.py",
            "tests/reference_parity/_fixtures/rlasso_R.json",
        ],
        "note": "All four select_Z / select_X paths, BCCH EminentDomain and the AJR "
        "vignette.",
    },
    "rlassologit": {
        "status": "aligned",
        "reference": "R hdm::rlassologit 0.3.2 (glmnet 4.1.10 engine)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "hdm": "0.3.2",
            "glmnet": "4.1.10",
        },
        "tolerance": "support exact; post-Lasso atol 1e-5 (observed rel 1.7e-10); "
        "post=False atol 1e-4 (observed 1.3e-6: glmnet "
        "coordinate-descent convergence)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rlassologit_parity.py",
            "tests/reference_parity/_fixtures/rlassologit_R.json",
        ],
        "note": "The non-post fit inherits glmnet's stopping rule; the post-Lasso fit "
        "is an unpenalised logit and agrees to 1e-10.",
    },
    "rlassologit_effect": {
        "status": "bit-exact",
        "reference": "R hdm::rlassologitEffect 0.3.2",
        "reference_versions": {"R": "4.5.2", "hdm": "0.3.2"},
        "tolerance": "alpha / se atol 1e-6 (observed rel 1.1e-15 / 5.3e-14 post; se "
        "3.2e-7 with post=False)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rlassologit_effect_parity.py",
            "tests/reference_parity/_fixtures/rlassologit_effect_R.json",
        ],
        "note": "",
    },
    "rlassologit_effects": {
        "status": "bit-exact",
        "reference": "R hdm::rlassologitEffects 0.3.2",
        "reference_versions": {"R": "4.5.2", "hdm": "0.3.2"},
        "tolerance": "coef / se atol 1e-6 (observed rel 8.6e-16 / 2.5e-14)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rlassologit_effect_parity.py",
            "tests/reference_parity/_fixtures/rlassologit_effect_R.json",
        ],
        "note": "",
    },
    "tF_adjustment": {
        "status": "bit-exact",
        "reference": "R ivDiag::tF 1.0.6 (LMMP 2022 tF table)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "ivDiag": "1.0.6"},
        "tolerance": "rel 1e-12 on the same 26 F values (observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_iv_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_R.json",
        ],
        "note": "Thin alias of tF_critical_value, asserted directly in the same test.",
    },
    "tF_critical_value": {
        "status": "bit-exact",
        "reference": "R ivDiag::tF 1.0.6 (LMMP 2022 tF table)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "ivDiag": "1.0.6"},
        "tolerance": "critical value at 26 F values from 4 to 1e4: rel 1e-12 "
        "(observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_iv_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_R.json",
        ],
        "note": "Through 1.28.0 the table was not LMMP's (c(10) 3.16 vs 3.4353; 1.96 "
        "from F = 75), every error anti-conservative. Now ivDiag's 84-point "
        "sqrt(F) table and interpolation. Below F = 4 StatsPAI returns inf "
        "where ivDiag clamps to 18.66.",
    },
    "weakrobust": {
        "status": "aligned",
        "reference": "R ivmodel::CLR 1.9.1; Stata weakiv 2.4.07 (md small)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "ivmodel": "1.9.1",
            "Stata": "18 MP",
            "weakiv": "2.4.07",
        },
        "tolerance": "CLR statistic and p-value vs ivmodel rel 1e-9 (observed "
        "1.2e-11); CLR set vs ivmodel 5e-5 (observed 1.2e-5, ivmodel's "
        "uniroot default tolerance); CLR / K / AR vs Stata weakiv 1e-6 "
        "(observed 1.1e-7 / 1.1e-7 / 2.1e-8, not bisected further)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_rd_iv_R_parity.py",
            "tests/reference_parity/_fixtures/rd_iv_R.json",
            "tests/reference_parity/_fixtures/rd_iv_Stata.json",
        ],
        "note": "Three fixes: the orthonormalised instruments were not orthonormal "
        "for k >= 2 (CLR 4.1% high); K 'at h0' was read off the nearest grid "
        "point; the CLR critical value was simulated (now integrated exactly, "
        "clr_method='simulate' keeps the old path). The set endpoints also "
        "satisfy p_CLR(endpoint) = alpha to 1e-9 (reference-free identity).",
    },
    # ---- phase 3: DiD / synthetic control / shift-share (22) ----
    "bartik": {
        "status": "bit-exact",
        "reference": "2SLS: R AER::ivreg + sandwich (HC1 / classical), Stata "
        "ivregress 2sls, vce(robust) small / small; Rotemberg weights: R "
        "bartik.weight::bw and Stata bartik_weight (Goldsmith-Pinkham, "
        "Sorkin & Swift)",
        "reference_versions": {
            "R": "4.5.2",
            "AER": "1.2.16",
            "sandwich": "3.1.1",
            "bartik.weight": "0.1.0 (GitHub "
            "paulgp/bartik-weight@722ceb85484d6a2bf77985edf2403515eacd1770, "
            "R-code/pkg)",
            "Stata": "18 MP",
            "bartik_weight": "GitHub "
            "paulgp/bartik-weight@722ceb85484d6a2bf77985edf2403515eacd1770 "
            "code/bartik_weight.ado",
        },
        "tolerance": "1e-9 rel on all coefficients and SEs (observed <= 5.4e-15) and "
        "on Rotemberg alpha_k / beta_k (observed 2.5e-12)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_did_synth_shiftshare_parity.py",
            "tests/reference_parity/_fixtures/did_synth_shiftshare_R.json",
            "tests/reference_parity/_fixtures/did_synth_shiftshare_stata.json",
        ],
        "note": "leave_one_out=False (the leave-one-out instrument has no reference). "
        "Rotemberg weights are exposed in model_info['rotemberg_weights'] "
        "with the per-industry just-identified beta_k; robust= now rejects "
        "values it does not implement instead of silently using HC1.",
    },
    "breakdown_m": {
        "status": "aligned",
        "reference": "HonestDiD::findOptimalFLCI 0.2.8 (Rambachan & Roth), breakdown "
        "by uniroot on the bound facing zero",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "HonestDiD": "0.2.8",
            "CVXR": "1.8.2",
        },
        "tolerance": "vs HonestDiD with its Monte-Carlo folded-normal quantile "
        "replaced by the exact one: 1e-9 (observed 6.1e-11); vs "
        "HonestDiD as shipped: 1e-3 (observed 4.3e-4), the simulation "
        "error of .qfoldednormal (1e6 draws, seed 0; 1.96224 vs exact "
        "1.95996 at mu = 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_R_parity.py",
            "tests/reference_parity/_fixtures/did_synth_honest_R.json",
        ],
        "note": "Grade applies to method='smoothness' with a recoverable event-study "
        "covariance (Callaway-Sant'Anna fits). method='relative_magnitude' "
        "and the covariance-free fallback invert honest_did's native "
        "approximate intervals (closed forms, T1, warned). Fixed in the "
        "did_synth sweep: breakdown_m ignored `method` and returned (|theta| "
        "- z SE)/(e+1) everywhere; the native FLCI's SLSQP stopped at its "
        "start point and froze the worst-case bias. Regenerate via "
        "_generate_did_synth_honest_R.R.",
    },
    "continuous_did": {
        "status": "bit-exact",
        "reference": "fixest::feols(y ~ dose:post | id + time) (method='twfe')",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "fixest": "0.14.0"},
        "tolerance": "slope & SE 1e-9 rel (observed 2.2e-15), iid / ~id / ~region, "
        "balanced + unbalanced",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_didvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_didvar_R.json",
        ],
        "note": "Grade covers method='twfe' only. iid SE divides by n-K with K "
        "counting every absorbed unit/period parameter; clustered SE uses "
        "G/(G-1)(n-1)/(n-K) with fixest's nested-FE rule. The default "
        "method='att_gt' (dose-bin 2x2 rollup) and 'dose_response' are "
        "heuristics with no package reference: pinned by an analytic 2x2 SE "
        "and exact linear-slope recovery instead. Not the CGS ATT(d|g,t) "
        "estimator -- that is sp.cgs_continuous_did (contdid). Regenerate via "
        "_generate_did_synth_didvar_R.R.",
    },
    "demeaned_synth": {
        "status": "aligned",
        "reference": "augsynth::augsynth(progfunc = 'None', fixedeff = TRUE) 0.2.0 "
        "(de-meaned SCM)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "augsynth": "0.2.0",
            "osqp": "1.0.0",
        },
        "tolerance": "gap path and ATT 1e-7 rel (observed 1.2e-9); weights 1e-8 abs",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_synthvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_synthvar_R.json",
        ],
        "note": "variant='demeaned' only. Limited by augsynth's synth_qp, which "
        "hard-codes OSQP at eps_abs = eps_rel = 1e-8 (R's zero weights come "
        "back as +-1e-9). variant='detrended' has no reference and is covered "
        "by an exact-fit identity.",
    },
    "did_estimate": {
        "status": "bit-exact",
        "reference": "synthdid::did_estimate 0.0.9",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "synthdid": "0.0.9",
        },
        "tolerance": "estimate and jackknife SE at 1e-9 on the Prop. 99 replica and a "
        "five-treated panel (observed est 3.1e-15, jackknife SE "
        "5.7e-16); R's 40 placebo and 40 bootstrap replications replayed "
        "at 1e-9 (observed 1.9e-13)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_R_parity.py",
            "tests/reference_parity/_fixtures/did_synth_R.json",
        ],
        "note": "Uniform weights reduce to the 2x2 difference in means, asserted "
        "reference-free at 1e-12. Placebo/bootstrap end-to-end SE is T3 (see "
        "sc_estimate). Regenerate via _generate_did_synth_R.R.",
    },
    "did_timevarying_covariates": {
        "status": "bit-exact",
        "reference": "ptetools::pte_default(d_outcome=TRUE, est_method='reg') and "
        "did::att_gt(est_method='reg')",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "ptetools": "1.0.1",
            "did": "2.3.0",
            "DRDID": "1.2.3",
        },
        "tolerance": "ATT(g,t) and overall ATT 1e-9 rel (observed 3.8e-15)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_didvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_didvar_R.json",
        ],
        "note": "X_{g-1} outcome-regression estimator, never-treated comparison: "
        "every post-period ATT(g,t), the group-aggregated overall ATT "
        "(ptetools overall / did aggte type='group') and the simple aggregate "
        "(did aggte type='simple'), with two covariates and one. ptetools and "
        "did agree with each other to ~1e-14. Point estimates only: the SE is "
        "a unit bootstrap here and a multiplier bootstrap in ptetools (T3), "
        "not compared.",
    },
    "discos": {
        "status": "bit-exact",
        "reference": "DiSCos::DiSCo 0.1.4 (Gunsilius distributional synthetic "
        "controls), mixture = FALSE; mixture = TRUE vs GLPK on DiSCo's "
        "LP",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "DiSCos": "0.1.4",
            "pracma": "2.4.6",
            "quadprog": "1.5.8",
            "CVXR": "1.8.2",
            "Rglpk": "0.6.5.1",
        },
        "tolerance": "quantile weights 1e-11 abs (observed 4.9e-13), counterfactual "
        "quantile functions 1e-10 rel, quantile effects 1e-10 abs "
        "(observed 1.9e-12); mixture LP weights vs GLPK 1e-12 abs, vs "
        "DiSCo's SCS solution 5e-6 abs (observed 7.1e-7)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_synthvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_synthvar_R.json",
        ],
        "note": "Individual-level data only (the aggregate-panel fallback is a "
        "StatsPAI heuristic and warns). R's random quantile nodes / CDF grids "
        "are replayed from its L'Ecuyer stream and passed via q_nodes / "
        "cdf_grid. Permutation test: T4 -- DiSCos 0.1.4's DiSCo_per_iter "
        "leaves the original treated unit's quantile column at zero; StatsPAI "
        "matches the one-line-patched function (8.3e-12) and reproduces the "
        "unpatched numbers when it zeroes that column (1e-9).",
    },
    "distributional_did": {
        "status": "bit-exact",
        "reference": "didFF::distDD 0.1.0 (Roth & Sant'Anna)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "didFF": "0.1.0",
            "did": "2.3.0",
        },
        "tolerance": "per-bin effect & SE 1e-9 (observed est 2.6e-12 abs, SE 2.5e-10 "
        "rel)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_didvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_didvar_R.json",
            "tests/reference_parity/test_functional_form_extended_parity.py",
            "tests/reference_parity/_fixtures/didff_extended_reference.json",
        ],
        "note": "Fourteen configurations on did::mpdta: nbins / binpoints / discrete "
        "binning, weights, covariates under dr / reg / ipw, not-yet-treated "
        "comparisons, simple / group / dynamic / calendar aggregation and the "
        "dynamic event window. distDD itself crashes when a bin's influence "
        "function is degenerate (balance_e=1 here); that case is pinned "
        "against distDD's recipe re-run step by step in R, which first "
        "reproduces distDD's own dynamic output to 1e-12. Largest gaps are "
        "the propensity-score cases (logit iterated to tolerance on both "
        "sides).",
    },
    "harvest_did": {
        "status": "bit-exact",
        "reference": "R did::att_gt(control_group='notyettreated', "
        "base_period='universal') for every (cohort, horizon) cell; "
        "did::aggte(type='dynamic') for the event study under "
        "weighting='n_treated'",
        "reference_versions": {"did": "2.3.0", "DRDID": "1.2.3"},
        "tolerance": "Every 2x2 cell (ATT and SE) and every event-study horizon (ATT "
        "and SE) at 1e-9 relative; observed agreement 1.7e-14. The "
        "inverse-variance aggregate over horizons (the default headline) "
        "has no reference and is checked by identity only.",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_did_synth_misc_parity.py"],
        "note": "The estimator had been attributed to Abadie, Angrist, Frandsen & "
        "Pischke (NBER WP 34550, 2025), a survey chapter that defines no such "
        "estimator; its building blocks are Callaway-Sant'Anna cells. "
        "Promoted by removing defects: pre-period placebo cells counted the "
        "treated cohort among its own controls, and every aggregation (event "
        "study, headline aggregate, pre-trend Wald test) treated cells as "
        "independent although they share units -- headline SE 0.062 before, "
        "0.101 after, Monte Carlo sd 0.114.",
    },
    "mc_panel": {
        "status": "bit-exact",
        "reference": "MCPanel::mcnnm_fit (Athey, Bayati, Doudchenko, Imbens & "
        "Khosravi; github.com/susanathey/MCPanel) and fect::fect(method "
        '= "mc")',
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "MCPanel": "0.0 @ " "6b2706fd7c35f3266048ceb22a7e9a61ae1774da",
            "fect": "2.4.1",
            "gsynth": "1.4.0",
        },
        "tolerance": "ATT and fitted untreated matrix 1e-9 rel at fixed lambda "
        "(observed <= 2.6e-13), four fixed-effect modes x two lambdas",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_mc_parity.py",
            "tests/reference_parity/_fixtures/did_synth_mc_R.json",
        ],
        "note": "Fixed lambda: StatsPAI theta = MCPanel lambda_L * |O| / 2 = fect "
        "lambda * N * T (same minimiser). fixed_effects "
        "two-way/unit/time/none = MCPanel (to_estimate_u, to_estimate_v). "
        "References run past their default stopping rules (MCPanel rel_tol = "
        "0, 3000 sweeps; fect tol 1e-15). The bootstrap SE and the heuristic "
        "default lambda have no reference and are not compared. Regenerate "
        "via _generate_did_synth_mc_R.R.",
    },
    "mc_synth": {
        "status": "bit-exact",
        "reference": "MCPanel::mcnnm_fit (Athey, Bayati, Doudchenko, Imbens & "
        "Khosravi; github.com/susanathey/MCPanel) and fect::fect(method "
        '= "mc")',
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "MCPanel": "0.0 @ " "6b2706fd7c35f3266048ceb22a7e9a61ae1774da",
            "fect": "2.4.1",
        },
        "tolerance": "ATT and fitted untreated matrix 1e-9 rel at fixed lambda "
        "(observed <= 6.0e-13), two-way and no-FE x two lambdas",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_mc_parity.py",
            "tests/reference_parity/_fixtures/did_synth_mc_R.json",
        ],
        "note": "Single treated unit (unit 40 from period 21) masked in the same 40 x "
        "25 panel; same shared solver as mc_panel. The placebo-spread SE and "
        "the default K-fold CV lambda (own random folds) have no reference "
        "and are not compared. Regenerate via _generate_did_synth_mc_R.R.",
    },
    "robust_synth": {
        "status": "bit-exact",
        "reference": "scpi::scest(w.constr = list(name = 'ols')) with scdata(constant "
        "= TRUE) and stats::lm (unconstrained SC with intercept); glmnet "
        "(ridge / lasso / elastic net, unpenalised intercept)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "scpi": "4.0.1",
            "glmnet": "4.1.10",
        },
        "tolerance": "OLS weights / intercept / fitted path 1e-10 rel (observed "
        "8.4e-13); penalised paths 1e-8 rel, weights atol 1e-10 "
        "(observed 3.4e-11 abs, glmnet's coordinate-descent stop)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_synthvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_synthvar_R.json",
        ],
        "note": "Covers variant='unconstrained' (l2 = 0 and the default 0.01) and "
        "'elastic_net'. glmnet is compared after scaling y to unit (1/n) SD "
        "with n*lambda*(1-alpha) = l2 and 2*n*lambda*alpha = l1/sd(y). "
        "variant='penalized' (simplex + ridge) and the placebo SE have no "
        "reference. No canonical package implements Doudchenko & Imbens' "
        "CV-tuned estimator end to end; the penalty is user-set here.",
    },
    "sc_estimate": {
        "status": "bit-exact",
        "reference": "synthdid::sc_estimate 0.0.9 (Arkhangelsky, Athey, Hirshberg, "
        "Imbens & Wager)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "synthdid": "0.0.9",
        },
        "tolerance": "estimate, unit/time weights and jackknife SE at 1e-9 on the "
        "Prop. 99 replica and a five-treated panel (observed est "
        "8.1e-15, jackknife SE 4.8e-15); each of R's 40 placebo and 40 "
        "bootstrap replications replayed at 1e-9 (observed 2.1e-11)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_R_parity.py",
            "tests/reference_parity/_fixtures/did_synth_R.json",
        ],
        "note": "Placebo and bootstrap SEs are Monte-Carlo draws: the replication map "
        "is pinned draw for draw against R's recorded index vectors, the "
        "end-to-end seeded SE only within pooled Monte-Carlo error (T3). "
        "Fixed in the did_synth sweep: all three SE methods now follow "
        "synthdid::vcov (warm start, frozen regularisation, fixed-weight "
        "jackknife over all units). Regenerate via _generate_did_synth_R.R.",
    },
    "scdata": {
        "status": "bit-exact",
        "reference": "R scpi::scdata (features = outcome, no cov.adj, constant = "
        "FALSE)",
        "reference_versions": {
            "scpi": "4.0.1",
            "CVXR": "1.9.2",
            "ECOSolveR": "0.6.1",
            "Qtools": "1.6.0",
            "quantreg": "6.1",
        },
        "tolerance": "A, B, P matrices identical (0 difference) on scpi_germany; "
        "donor order = R sort(B.names).",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_did_synth_scpi_parity.py"],
        "note": "Added by the phase-3 did_synth scpi sweep, which found sp.scest's "
        "lasso / ridge were penalised estimators on standardised data rather "
        "than R scpi's norm-constrained weights, and sp.scpi was not the "
        "Cattaneo-Feng-Titiunik procedure (subsampling variance + Gaussian "
        "PI; intervals 2-4x too narrow on scpi_germany). Both were rewritten "
        "as a port of R scpi.",
    },
    "scest": {
        "status": "aligned",
        "reference": "R scpi::scest (w.constr simplex / lasso / ridge / ols / L1-L2, "
        "V = 'separate')",
        "reference_versions": {
            "scpi": "4.0.1",
            "CVXR": "1.9.2",
            "clarabel": "0.11.2",
            "osqp": "1.0.0",
        },
        "tolerance": "ols and lasso weights at 1e-9 abs; ridge Q / lambda and L1-L2 "
        "Q2 at 1e-10. simplex / ridge / L1-L2 weights are bounded by the "
        "conic solver: R's objective exceeds StatsPAI's exact optimum by "
        "<= 1e-7 relative (CLARABEL gap 1e-8), R's point is feasible, "
        "and ||w_R - w_py|| <= sqrt(gap / lambda_min(B'B)) (observed "
        "2.0e-6 / 1.4e-5 / 2.9e-7).",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_did_synth_scpi_parity.py"],
        "note": "Correctness fix in the phase-3 sweep: lasso / ridge were "
        "penalised coordinate descent / ridge regression with lambda = 1 on "
        "standardised data (weights off by up to 0.23 from R); now ||w||_1 <= "
        "1 and ||w||_2 <= Q with R's shrinkage.EST radius. lasso_lambda / "
        "ridge_lambda deprecated (ignored with DeprecationWarning); L1-L2 "
        "added.",
    },
    "scpi": {
        "status": "aligned",
        "reference": "R scpi::scpi (effect = 'unit-time', u.missp, u.sigma = HC1, "
        "u.order = e.order = 1, rho = type-2, e.method = all)",
        "reference_versions": {
            "scpi": "4.0.1",
            "CVXR": "1.9.2",
            "ECOSolveR": "0.6.1",
            "Qtools": "1.6.0",
            "quantreg": "6.1",
        },
        "tolerance": "On R's weights: rho, Q.star, u.mean, Omega, Sigma, e.mean at "
        "1e-9; out-of-sample e.var and gaussian / ls / qreg bounds at "
        "1e-9 against R with rrq(method = 'br') (exact LP) and 5e-4 abs "
        "against the default Frisch-Newton rrq. In-sample simulation fed "
        "R's draws: per-draw median <= 1e-6 and max <= 2e-4 vs ECOS at "
        "1e-12, quantile bounds at 1e-5; vs default ECOS (1e-8) bounds "
        "within 2e-3 abs. Average-effect CI = scdataMulti(effect = "
        "'unit') within 5e-4. J > T0 (California, 38 donors) covered.",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_did_synth_scpi_parity.py"],
        "note": "Correctness fix in the phase-3 sweep: sp.scpi was a different "
        "procedure under the scpi name (subsampling in-sample variance, "
        "residual-variance out-of-sample term, Gaussian PI, invented SE / "
        "p-value); now a port of R scpi solved exactly (active-set QCQP per "
        "draw, exact LP quantile regressions). Simulated bounds with "
        "StatsPAI's own RNG are Monte Carlo relative to R (T3); pass draws= "
        "to reproduce R. Default simplex end-to-end rho / df differ from R "
        "because CLARABEL leaves one donor at 2.0e-6 >= scpi's 1e-6 active "
        "threshold (T4; R's own L1-L2 fit of the same optimum reproduces "
        "StatsPAI's rho).",
    },
    "shift_share_se": {
        "status": "bit-exact",
        "reference": "R ShiftShareSE::ivreg_ss (Adao, Kolesar & Morales), AKM row; "
        "Stata SSC ivreg_ss",
        "reference_versions": {
            "R": "4.5.2",
            "ShiftShareSE": "1.1.0",
            "Stata": "18 MP",
            "ivreg_ss": "SSC 20241116",
        },
        "tolerance": "1e-9 rel on beta and the AKM / AKM0 / EHW / Homoscedastic SEs; "
        "observed <= 6e-14",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_did_synth_shiftshare_parity.py",
            "tests/reference_parity/_fixtures/did_synth_shiftshare_R.json",
            "tests/reference_parity/_fixtures/did_synth_shiftshare_stata.json",
        ],
        "note": "Applied to an sp.bartik 2SLS fit with and without controls. Until "
        "this sweep it used the second-stage fitted values as the instrument "
        "and returned 0.0176 against AKM's 0.2904; it now raises on results "
        "that do not record the shift-share inputs.",
    },
    "spillover_did": {
        "status": "bit-exact",
        "reference": "R did::att_gt(control_group='nevertreated') + "
        "did::aggte(type='simple') per group (direct / ring r, ring "
        "cohort = exposure onset); single cohort also fixest::feols(dbar "
        "~ treat + ring1 + ring2, vcov='hetero', ssc(adj=FALSE))",
        "reference_versions": {"did": "2.3.0", "DRDID": "1.2.3", "fixest": "0.14.0"},
        "tolerance": "Direct and ring effects, their SEs and every (group, onset "
        "cohort, period) cell at 1e-9 relative on a single-cohort and a "
        "staggered spatial panel; observed agreement 1e-14. The ring "
        "construction is recomputed independently in R from the "
        "coordinates.",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_did_synth_misc_parity.py"],
        "note": "No package implements Butts's ring estimator; the regression step is "
        "pinned to the fixest form in Butts's own replication code and every "
        "design to did on the constructed groups. Promoted by removing two "
        "defects: under staggered adoption every ring unit entered every "
        "cohort's cell regardless of when it was exposed (ring effects 0.79 / "
        "0.25 against did's 1.24 / 0.40), and the cohort-share weight term "
        "was missing from the standard errors (direct SE 0.078 vs 0.091). "
        "Single-cohort output is unchanged.",
    },
    "ssaggregate": {
        "status": "bit-exact",
        "reference": "R ShiftShareSE::ivreg_ss / reg_ss (Adao, Kolesar & Morales) and "
        "ssaggregate (Borusyak, Hull & Jaravel; R kylebutts/ssaggregate "
        "+ AER::ivreg/sandwich HC0); Stata SSC ivreg_ss / reg_ss and "
        "ssaggregate + ivreg2, robust",
        "reference_versions": {
            "R": "4.5.2",
            "ShiftShareSE": "1.1.0",
            "ssaggregate (R)": "0.0.0.9000 (GitHub "
            "kylebutts/ssaggregate@22df93980250891a0cc247f6020136cd33c65ba2)",
            "AER": "1.2.16",
            "sandwich": "3.1.1",
            "Stata": "18 MP",
            "reg_ss / ivreg_ss": "SSC 20241116",
            "ssaggregate (Stata)": "SSC 1.2.2 (20200826)",
        },
        "tolerance": "1e-9 rel on beta, every SE row (Homoscedastic, EHW, Reg. "
        "cluster, AKM, AKM0), AKM/AKM0 CIs and the shock-level frame; "
        "p-values also atol 1e-15 (references use 2*(1-Phi)); observed "
        "<= 6e-14 (frame 2.9e-13)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_did_synth_shiftshare_parity.py",
            "tests/reference_parity/_fixtures/did_synth_shiftshare_R.json",
            "tests/reference_parity/_fixtures/did_synth_shiftshare_stata.json",
        ],
        "note": "Incomplete shares (row sums 0.55-0.95), IV and OLS (reduced-form) "
        "modes, with and without controls, region clustering, AKM0 at alpha "
        "0.05 and 0.10. Until this sweep the AKM SE used u_k = sum_i s_ik Z_i "
        "e_i instead of hX_k * s_k'e and was 4.4x too small on this data "
        "(0.0665 vs 0.2904); the BHJ shock-level aggregation did not exist. "
        "Regenerate via _generate_did_synth_shiftshare_R.R and "
        "_fixtures/_generate_did_synth_shiftshare_stata.do.",
    },
    "staggered_cs": {
        "status": "bit-exact",
        "reference": "staggered::staggered_cs 1.2.2 (Roth & Sant'Anna)",
        "reference_versions": {"R": "4.5.2", "staggered": "1.2.2"},
        "tolerance": "estimate, Neyman SE and adjusted SE at abs 1e-9 on mpdta, a "
        "randomised rollout and a null panel x simple/cohort/calendar "
        "(observed rel 6.6e-14)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_staggered_extended_parity.py",
            "tests/reference_parity/_fixtures/staggered_extended_reference.json",
        ],
        "note": "Also Track A module 82_staggered row cs_simple (R and Stata). "
        "Plug-in weights (beta = 1), every not-yet-treated cohort as control, "
        "units treated in the first period dropped -- as R staggered_cs. "
        "Regenerate via _generate_staggered_extended_R.R.",
    },
    "staggered_sa": {
        "status": "bit-exact",
        "reference": "staggered::staggered_sa 1.2.2 (Roth & Sant'Anna)",
        "reference_versions": {"R": "4.5.2", "staggered": "1.2.2"},
        "tolerance": "estimate, Neyman SE and adjusted SE at abs 1e-9 on mpdta, a "
        "randomised rollout and a null panel x simple/cohort/calendar "
        "(observed rel 6.6e-14)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_staggered_extended_parity.py",
            "tests/reference_parity/_fixtures/staggered_extended_reference.json",
        ],
        "note": "Also Track A module 82_staggered row sa_simple (R and Stata). "
        "Plug-in weights with only the last-treated cohort as control -- as R "
        "staggered_sa. Regenerate via _generate_staggered_extended_R.R.",
    },
    "staggered_synth": {
        "status": "bit-exact",
        "reference": "augsynth::multisynth 0.2.0 (Ben-Michael, Feller & Rothstein; "
        "partially pooled SCM for staggered adoption)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "augsynth": "0.2.0",
            "osqp": "1.0.0",
        },
        "tolerance": "ATT, per-unit / per-cohort ATT, event-time ATT and jackknife SE "
        "1e-8 rel (observed <= 2.6e-11); weights 1e-8 abs (observed "
        "2.1e-10); nu 1e-8, imbalance norms 1e-6 rel",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_synthvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_synthvar_R.json",
        ],
        "note": "Six multisynth configurations on one staggered panel: nu = 0 / 0.5 / "
        "auto, fixedeff on and off, time_cohort = TRUE, n_lags = 5 with "
        "lambda = 0.1, and multisynth's own defaults; jackknife SE via "
        "summary(inf_type = 'jackknife'). multisynth run with OSQP at eps "
        "1e-12; StatsPAI solves the same QP exactly by active set. The "
        "placebo SE (StatsPAI's default se_method) has no reference. "
        "Regenerate via _generate_did_synth_synthvar_R.R.",
    },
    # ======================================================================
    # Cross-language campaign, phase 3 round 2 (2026-09). Deliverables:
    # docs/dev/campaign_phase3/r2_<family>.md. structural_break, mgwr and
    # gformula_ice_fn supersede their round-1 records (same tests plus more).
    # ======================================================================
    # ---- round 2: round-1 open items (9) ----
    "contrast": {
        "status": "bit-exact",
        "reference": "Stata 18 margins r.g / rb3.g / ar.g / gw.g",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-10 rel (observed <= 7.9e-15 est / 4.9e-15 SE)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_r2_postest_parity.py",
            "tests/reference_parity/_fixtures/r2_postest_stata.json",
        ],
        "note": "Contrasts of predictive margins (asobserved). Stata's contrast "
        "command is asbalanced and holds interacting continuous covariates at "
        "0; both mechanisms asserted as identities.",
    },
    "gformula_ice_fn": {
        "status": "bit-exact",
        "reference": "ltmle::ltmle 1.3.0 (gcomp = TRUE, SL.library = list(Q = "
        "'SL.lm')) point estimate; base-R lm() ICE with geex::m_estimate "
        "1.1.1 sandwich SE",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "ltmle": "1.3.0",
            "SuperLearner": "2.0.40",
            "geex": "1.1.1",
        },
        "tolerance": "point 1e-10 rel vs ltmle (observed 2.1e-11), 1e-12 vs hand lm "
        "(observed 2.4e-15); sandwich SE 1e-9 rel (observed 7.1e-12)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_r2_teffects_parity.py",
            "tests/reference_parity/_fixtures/r2_teffects_R.json",
            "tests/reference_parity/test_teffects_R_parity.py",
            "tests/reference_parity/_fixtures/teffects_R.json",
        ],
        "note": "Pooled sequential regression, 2- and 4-period regimes; ltmle sets "
        "past A to the regime, sp keeps them observed (identical under OLS on "
        "nested histories). SL.lm clips to [0, 1]: the binary-Yb always-treat "
        "block, where the clip binds, is a documented convention gap.",
    },
    "malmquist": {
        "status": "aligned",
        "reference": "sfaR::sfacross 1.0.1 per period + sfaR::efficiencies (teBC / "
        "teJLMS); TC from sfaR betas",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "sfaR": "1.0.1",
            "metafrontier": "0.3.1",
        },
        "tolerance": "EC / TC / M 1e-6 rel (observed 2.7e-7); "
        "metafrontier::malmquist_meta(method='sfa') EC_group and "
        "1/TC_group 1e-4 (observed 1.8e-5, reference optim BFGS with "
        "finite-difference gradients)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_r2_frontier_parity.py",
            "tests/reference_parity/_fixtures/r2_frontier_R.json",
        ],
        "note": "Default efficiency='bc': EC = TE_{t+1}/TE_t (Battese-Coelli, each "
        "period's own frontier), TC = exp(0.5 (x_t + x_{t+1})'(b_{t+1} - "
        "b_t)), M = EC * TC. metafrontier's TC_group uses Farrell distances "
        "and equals 1 / TC. efficiency='residual' (pre-1.29.0 default, "
        "composed-residual EC) has no package reference. Regenerate via "
        "_generate_r2_frontier_R.R.",
    },
    "margins_at": {
        "status": "bit-exact",
        "reference": "Stata 18 margins, at(...); R marginaleffects::avg_predictions",
        "reference_versions": {
            "Stata": "18",
            "R": "R version 4.5.2 (2025-10-31)",
            "marginaleffects": "0.32.0",
        },
        "tolerance": "1e-10 rel (observed <= 3.5e-15 est / SE; marginaleffects SE "
        "1.8e-10, numerical Jacobian)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_r2_postest_parity.py",
            "tests/reference_parity/_fixtures/r2_postest_stata.json",
            "tests/reference_parity/_fixtures/r2_postest_R.json",
        ],
        "note": "Single and multiple at-variables, at() on a factor, g##c.x and "
        "g##i.h, OLS / vce(robust) / vce(cluster). Predictive margins average "
        "over the observed covariates (asobserved).",
    },
    "mgwr": {
        "status": "bit-exact",
        "reference": "PySAL mgwr 2.2.1 Sel_BW(multi=True).search() + MGWR.fit() "
        "(authors' implementation); fixed-bandwidth back-fitting also "
        "GWmodel::gwr.multiscale 2.4.1",
        "reference_versions": {
            "python": "3.13.9",
            "mgwr": "2.2.1",
            "numpy": "2.2.6",
            "R": "R version 4.5.2 (2025-10-31)",
            "GWmodel": "2.4.1",
        },
        "tolerance": "bandwidths, initial bandwidth, bandwidth history and iteration "
        "count exact; betas 1e-9 rel (observed 3.0e-12); SEs, ENP_j, "
        "tr(S), sigma2, AICc/AIC/BIC 1e-10 rel (observed <= 1e-15); SOC "
        "path 1e-9 (observed 8e-13)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_r2_spatial_parity.py",
            "tests/reference_parity/_fixtures/r2_spatial_mgwr.json",
            "tests/reference_parity/test_spatial_survey_R_parity.py",
            "tests/reference_parity/_fixtures/spatial_survey_R.json",
        ],
        "note": "Georgia, y = PctBach, X = (PctFB, PctBlack, PctRural) standardised "
        "(ddof 0) as in mgwr's docs; adaptive bisquare AICc, fixed Gaussian "
        "AICc, adaptive bisquare CV (200 sweeps, not converged on either "
        "side), adaptive exponential BIC with multi_bw_min 20. The search is "
        "mgwr's golden_section (delta 0.38197, integer probes, memo, "
        "2-decimal rounding), kernel eps 1.0000001, SOC stop and 5-sweep "
        "bandwidth freeze; SEs from replaying the bandwidth history. "
        "kernel_eps = 1 reproduces GWmodel's fixed-bandwidth fixed point. "
        "GWmodel's own bandwidth search is a different algorithm and is not "
        "reproduced. Regenerate via _generate_r2_spatial_mgwr.py (mgwr on "
        "PYTHONPATH) and _generate_spatial_survey_R.R. Sides: the R leg is "
        "GWmodel::gwr.multiscale at fixed bandwidths; the bandwidth search is "
        "pinned to PySAL mgwr 2.2.1 (Python, the authors' implementation).",
    },
    "policy_value": {
        "status": "bit-exact",
        "reference": "grf::average_treatment_effect(subset = policy == 1) x "
        "mean(policy) on grf get_scores; "
        "policytree::double_robust_scores reward contrast",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "grf": "2.6.1",
            "policytree": "1.2.4",
        },
        "tolerance": "1e-12 rel (observed 6.7e-16)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_r2_teffects_parity.py",
            "tests/reference_parity/_fixtures/r2_teffects_R.json",
        ],
        "note": "Value gain over treat-none, mean(Gamma * pi), on grf's own AIPW "
        "scores (causal_forest seed 7); the scores are an input, so forest "
        "randomness does not enter. No SE.",
    },
    "pwcompare": {
        "status": "bit-exact",
        "reference": "Stata 18 margins g, pwcompare(effects) "
        "mcompare(noadjust|bonferroni|sidak); R emmeans pairs(adjust=)",
        "reference_versions": {
            "Stata": "18",
            "R": "R version 4.5.2 (2025-10-31)",
            "emmeans": "2.0.3",
        },
        "tolerance": "1e-10 rel (observed <= 7.9e-15 diff / 4.9e-15 SE)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_r2_postest_parity.py",
            "tests/reference_parity/_fixtures/r2_postest_stata.json",
            "tests/reference_parity/_fixtures/r2_postest_R.json",
        ],
        "note": "Six pairwise comparisons, OLS / robust / cluster, additive, g##c.x "
        "and g##i.h. Holm intervals follow the emmeans convention "
        "(Bonferroni).",
    },
    "structural_break": {
        "status": "bit-exact",
        "reference": "strucchange::Fstats + sctest(type = 'supF') / breakpoints; "
        "mbreaks::dosequa (Bai-Perron sequential); Stata estat sbsingle",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "strucchange": "1.5.4",
            "mbreaks": "1.0.1",
            "Stata": "18",
        },
        "tolerance": "statistics 1e-10 rel (observed <= 1e-14); sup-F p-values 1e-10 "
        "rel for p > 1e-6, 1e-13 abs below (R uses 1 - pchisq); break "
        "dates exact",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_r2_ts_parity.py",
            "tests/reference_parity/_fixtures/r2_ts_R.json",
            "tests/reference_parity/_fixtures/r2_ts_Stata.json",
            "tests/reference_parity/test_timeseries_R_parity.py",
        ],
        "note": "sup-F p-value = Hansen (1997) pv_sup (table equal to strucchange "
        "sc.beta.sup and Stata pvalsup) at strucchange's lambda; "
        "method='bai-perron' = mbreaks::dosequa(prewhit=0, robust=0, "
        "hetdat=1, hetvar=0), 6 series x 4 levels, every step statistic "
        "matched to the traced pftest; 'global' = breakpoints (round 1). "
        "xtbreak's F(l+1|l) is the BP test with global breaks and a pooled "
        "sigma (convention, rebuilt). The default bai-perron path was a "
        "binary segmentation with a full-sample null before this round.",
    },
    "zisf": {
        "status": "aligned",
        "reference": "Stata chks 1.1 (estimation(zsf) eoption(ml)); R sfa::zsfm 1.2.0 "
        "(ZISF / ZISF_Z, likelihood at its optimum)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "sfa": "1.2.0",
            "numDeriv": "2016.8.1.1",
            "stata": "18",
            "chks": "1.1 (chks.pkg dated 20190320)",
        },
        "tolerance": "estimates and OIM SEs 1e-6 rel (observed chks 8.6e-8 / 9.4e-8; "
        "sfa likelihood at its optimum 1.3e-8 / 5.4e-8); sfa's reported "
        "L-BFGS-B point 5e-5 / 5e-4 (observed 1.6e-5 / 1.8e-4)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_r2_frontier_parity.py",
            "tests/reference_parity/_fixtures/r2_frontier_R.json",
            "tests/reference_parity/_fixtures/r2_frontier_stata.json",
        ],
        "note": "chks drops rows with y <= 0 (ln(depvar) check), so it is run on y + "
        "10 and _cons - 10 is compared. sfa ZISF parameterises P = "
        "exp(-|gamma|) and sigma as SDs (mapped by the delta method); ZISF_Z "
        "uses the same logit link; cost via inefdec=FALSE. sfa's reported "
        "point stops at score ~3e-4; the T2 comparison is its own "
        "log-likelihood Newton-polished to score <= 3.5e-8. Regenerate via "
        "_generate_r2_frontier_R.R and "
        "_fixtures/_generate_r2_frontier_stata.do.",
    },
    # ---- round 2: RD remaining (rd2d, rd_discrete) (3) ----
    "rd2d": {
        "status": "bit-exact",
        "reference": "R rd2d::rd2d / rd2d.distance 1.0.0 (Cattaneo, Titiunik & Yu)",
        "reference_versions": {"R": "4.5", "rd2d": "1.0.0", "sandwich": "3.1.1"},
        "tolerance": "estimate.p/q, std.err.p/q, t, CI, cross-point covariance rel "
        "1e-9 (observed 1.6e-11); bandwidths rel 1e-8 (observed "
        "1.2e-12); N exact",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_open_R_parity.py",
            "tests/reference_parity/_fixtures/rd_open_R.json",
        ],
        "note": "Location and distance approaches, sharp/fuzzy, clusters, HC0-3, "
        "joint/separate, 8 bwselect rules, kinks, mass points, WBATE "
        "headline. Radial-kernel SEs deliberately differ: R scales by hx*hy "
        "while fitting at radius sqrt(hx^2+hy^2) (pinned to lm()+sandwich "
        "instead). Through 1.28.0 a different, pooled estimator.",
    },
    "rd2d_bw": {
        "status": "bit-exact",
        "reference": "R rd2d::rdbw2d / rdbw2d.distance 1.0.0",
        "reference_versions": {"R": "4.5", "rd2d": "1.0.0"},
        "tolerance": "per-point bandwidths rel 1e-8 (observed 5.0e-13)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_open_R_parity.py",
            "tests/reference_parity/_fixtures/rd_open_R.json",
        ],
        "note": "With the selectors' own defaults (bwcheck 20, scaleregul 1). Returns "
        "a DataFrame since 1.29.0.",
    },
    "rd_discrete": {
        "status": "aligned",
        "reference": "R RDHonest::RDHonest / RDHonestBME 1.0.1.9000 (Kolesar)",
        "reference_versions": {"R": "4.5", "RDHonest": "1.0.1.9000"},
        "tolerance": "estimate, std.error, maximum.bias, conf.low/high rel 1e-9 at "
        "fixed h and for BME; 1e-6 when RDHonest selects h and M "
        "(optimiser)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_rd_open_R_parity.py",
            "tests/reference_parity/_fixtures/rd_open_R.json",
        ],
        "note": "method='bsd' = RDHonest, 'bme' = RDHonestBME. RDHonestBME's p-value "
        "adds the bias in outcome units to a z statistic; reproduced as "
        "p_value_rdhonest, headline uses bias/SE. Through 1.28.0 SEs were "
        "understated by ~sqrt(bin size).",
    },
    # ---- round 2: decomposition / QTE (5) ----
    "cfm_decompose": {
        "status": "aligned",
        "reference": "Stata cdeco, method(logit) 1.0.2 with drprocess (Chernozhukov, "
        "Fernandez-Val & Melly)",
        "reference_versions": {
            "Stata": "18.0 MP",
            "cdeco": "1.0.2 01mar2023 (bmelly/Stata "
            "counterfactual, Distribution-Date 20220803)",
        },
        "tolerance": "CDFs at the thresholds 1e-8 abs (observed 2.4e-10: Stata's "
        "logit stops at its default tolerance); quantiles 1e-12 abs "
        "(observed 0)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_decomp_qte_parity.py",
            "tests/reference_parity/_fixtures/decomp_qte_Stata.json",
        ],
        "note": "Requires thresholds= (39 shared thresholds 1.5(0.05)3.4) and "
        "inversion='step' (cdeco's getquantile). The default "
        "inversion='interpolate' is a smoothed quantile, not cdeco's.",
    },
    "fairlie": {
        "status": "bit-exact",
        "reference": "Stata fairlie 1.0.7 (Jann, SSC)",
        "reference_versions": {"Stata": "18.0 MP", "fairlie": "1.0.7 16jun2008 (SSC)"},
        "tolerance": "tightly converged logit/probit: contributions and SEs 1e-12 rel "
        "(observed 4.5e-15 / 5.4e-14); at Stata's default logit "
        "tolerance contributions 1e-8 and SEs 1e-5 (observed 1.6e-10 / "
        "1.6e-6)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_decomp_qte_parity.py",
            "tests/reference_parity/_fixtures/decomp_qte_Stata.json",
        ],
        "note": "Equal group sizes (500/500), so fairlie draws no subsample and its "
        "rank-to-rank matching is deterministic. Logit ref 0/1 and probit.",
    },
    "mediation_decompose": {
        "status": "bit-exact",
        "reference": "Stata paramed (Liu & Emsley, SSC), yreg(linear) mreg(linear) "
        "with interaction",
        "reference_versions": {"Stata": "18.0 MP", "paramed": "SSC"},
        "tolerance": "CDE/NDE/NIE/total and delta-method SEs 1e-12 rel (observed "
        "9.5e-15)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_decomp_qte_parity.py",
            "tests/reference_parity/_fixtures/decomp_qte_Stata.json",
        ],
        "note": "With covariates, all numbers match. Without covariates paramed's NIE "
        "standard error uses theta3 where theta2 + theta3 belongs; the test "
        "reconstructs paramed's number from that gradient exactly and "
        "StatsPAI's from the correct one (T4 on that single SE).",
    },
    "melly_decompose": {
        "status": "bit-exact",
        "reference": "Stata cdeco, method(qr) 1.0.2 (Chernozhukov, Fernandez-Val & "
        "Melly; bmelly/Stata counterfactual)",
        "reference_versions": {
            "Stata": "18.0 MP",
            "cdeco": "1.0.2 01mar2023 (bmelly/Stata "
            "counterfactual, Distribution-Date 20220803)",
        },
        "tolerance": "fitted and counterfactual quantiles 1e-10 rel (observed "
        "6.2e-16); QR process coefficients 1e-12 abs (observed 7.5e-15)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_decomp_qte_parity.py",
            "tests/reference_parity/_fixtures/decomp_qte_Stata.json",
        ],
        "note": "100 exact quantile regressions at (j-0.5)/100 (Stata simplex), "
        "pooled predictions inverted with mm_quantile definition 2, both "
        "reference directions. Evaluated at offset quantiles (0.10003, ...) "
        "so that tau*N is never an integer. Regenerate via "
        "_fixtures/_generate_decomp_qte_stata.do.",
    },
    "shapley_inequality": {
        "status": "aligned",
        "reference": "Stata shapley2 1.5 (Chavez Juarez) over regress + ineqdeco "
        "(Jenkins)",
        "reference_versions": {"Stata": "18.0 MP", "shapley2": "1.5 10jun15 (SSC)"},
        "tolerance": "value function v(S) 1e-11 rel (observed 1.4e-13); Shapley "
        "values 1e-6 rel (observed 2.7e-7, shapley2 routes v(S) through "
        "float variables); totals 1e-12",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_decomp_qte_parity.py",
            "tests/reference_parity/_fixtures/decomp_qte_Stata.json",
        ],
        "note": "GE(0), GE(1), GE(2). For index='gini' StatsPAI reports the "
        "bias-corrected Gini; ineqdeco's plug-in Gini value function is "
        "matched to 2e-12 by gini_population.",
    },
    # ---- round 2: grf operators / CATE / OPE / DML (14) ----
    "average_treatment_effect": {
        "status": "bit-exact",
        "reference": "grf::average_treatment_effect 2.6.1 (target.sample all, "
        "treated, control, overlap), forest outputs held fixed",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "grf": "2.6.1"},
        "tolerance": "estimate and std.err 1e-10 rel (observed 2.0e-15)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_ml_causal_R_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_R.json",
        ],
        "note": "Factored evidence: grf grows one causal forest (2000 trees, seed 42) "
        "and its OOB tau.hat, Y.hat and W.hat are frozen in the fixture; "
        "StatsPAI's operator runs on exactly those vectors. The forest itself "
        "is not pinnable across implementations (see causal_forest). clip=0 "
        "on the comparison (grf does not clip); the default clip=0.01 is "
        "asserted inert on this fixture.",
    },
    "blp_test": {
        "status": "bit-exact",
        "reference": "GenericML::BLP 0.2.3 (vcovHC const and HC1; with and without "
        "the baseline proxy)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "GenericML": "0.2.3",
            "sandwich": "3.1.1",
        },
        "tolerance": "beta and SE 1e-10 rel (observed 2.3e-15); one-sided p 1e-8 rel",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_ml_causal_R_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_R.json",
        ],
        "note": "Grade covers the weighted regression given the CATE proxy, "
        "propensity and baseline proxy (identical inputs on both sides); the "
        "default out-of-fold proxy refit is StatsPAI's own.",
    },
    "calibration_test": {
        "status": "bit-exact",
        "reference": "grf::test_calibration 2.6.1 (vcov.type HC3 default and HC1), "
        "forest outputs held fixed",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "grf": "2.6.1",
            "sandwich": "3.1.1",
        },
        "tolerance": "coef, se, t 1e-10 rel (observed 1.1e-15); one-sided p-value "
        "1e-8 rel (observed 1.6e-13)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_ml_causal_R_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_R.json",
        ],
        "note": "Factored evidence: grf grows one causal forest (2000 trees, seed 42) "
        "and its OOB tau.hat, Y.hat and W.hat are frozen in the fixture; "
        "StatsPAI's operator runs on exactly those vectors. The forest itself "
        "is not pinnable across implementations (see causal_forest).",
    },
    "cate_eval": {
        "status": "aligned",
        "reference": "grf::rank_average_treatment_effect 2.6.1 (AUTOC, QINI, TOC), "
        "fed grf's nuisances; shares sp.rate's operator",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "grf": "2.6.1"},
        "tolerance": "Point estimate (AUTOC, QINI) and TOC curve on q = 0.1..1 exact "
        "at 1e-10 rel (observed 4.6e-15), including a 30-group "
        "tied-priority case against rank_average_treatment_effect.fit. "
        "SE is T3: the analytic rank-corrected influence-function SE is "
        "compared with grf's half-sample bootstrap (R = 2000) at 6% rel "
        "(observed 2.6% AUTOC, 1.7% QINI).",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_ml_causal_R_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_R.json",
        ],
        "note": "Given e_hat, m_hat, mu1_hat, mu0_hat from grf's forest (no internal "
        "cross-fit). Factored evidence: grf grows one causal forest (2000 "
        "trees, seed 42) and its OOB tau.hat, Y.hat and W.hat are frozen in "
        "the fixture; StatsPAI's operator runs on exactly those vectors. The "
        "forest itself is not pinnable across implementations (see "
        "causal_forest).",
    },
    "direct_method": {
        "status": "bit-exact",
        "reference": "Open Bandit Pipeline (obp) 0.5.7 DirectMethod (same "
        "reward-model matrix via q_hat=)",
        "reference_versions": {"obp": "0.5.7"},
        "tolerance": "value 1e-12 rel (observed 0.0); SE identity 1e-10",
        "sides": ["py"],
        "test": [
            "tests/reference_parity/test_ml_causal_obp_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_obp.json",
        ],
        "note": "Python cross-package reference, not R or Stata: no R/Stata "
        "implementation takes K-action logged-bandit data with given "
        "behaviour and evaluation policies. Behaviour propensities, "
        "evaluation policy and a fixed reward model (q_hat=) are columns of "
        "ml_causal_ope.csv, so the estimator is deterministic. obp reports "
        "bootstrap intervals only; the analytic SE is checked as sd(obp round "
        "rewards)/sqrt(n) at 1e-10. The default internal random-forest reward "
        "model is not graded.",
    },
    "dml_model_averaging": {
        "status": "aligned",
        "reference": "ddml::ddml_plm 0.3.1 (shortstack = TRUE, ensemble_type = "
        "'nnls1'), OLS candidates, shared folds",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "ddml": "0.3.1",
            "sandwich": "3.1.1",
        },
        "tolerance": "short-stacking weights 1e-10 (observed 4.4e-14); ddml's final "
        "lm(y_r ~ d_r) with intercept and HC1 SE rebuilt from StatsPAI's "
        "stacked residuals at 1e-10 (observed 4.8e-16)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_ml_causal_dml_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_panel_R.json",
            "tests/reference_parity/_fixtures/ml_causal_doubleml.json",
        ],
        "note": "Convention difference in the last step only: StatsPAI solves the "
        "no-intercept PLR moment with the HC0 sandwich (DoubleML convention); "
        "ddml fits OLS with an intercept and reports HC1. The raw point "
        "estimates differ by 1.2e-5 rel on the fixture for that reason.",
    },
    "dml_panel": {
        "status": "bit-exact",
        "reference": "fixest::demean 0.14.0 (unit and unit+time absorption, balanced "
        "and unbalanced) and ddml::ddml_plm 0.3.1 (OLS learner, unit "
        "clusters, shared folds)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "fixest": "0.14.0",
            "ddml": "0.3.1",
            "sandwich": "3.1.1",
            "doubleml": "0.11.3",
            "scikit-learn": "1.6.1",
        },
        "tolerance": "within transform 1e-10 abs (observed 2.6e-14); estimate 1e-10 "
        "rel vs ddml and DoubleML (observed 4.1e-16); SE 1e-10 rel vs "
        "DoubleML (observed 1.4e-15) and vs ddml after the CR1 factor",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_ml_causal_dml_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_panel_R.json",
            "tests/reference_parity/_fixtures/ml_causal_doubleml.json",
        ],
        "note": "OLS learners and shared unit-level folds (fold_indices=). R leg: "
        "fixest pins the FE absorption, ddml the PLR estimate; ddml's SE is "
        "CR1 = StatsPAI's times sqrt(G/(G-1)(n-1)/(n-2)) (tested to 1e-10). "
        "Python leg: DoubleML 0.11.3 DoubleMLPLR with one-way unit clustering "
        "on the fixest-demeaned data matches estimate and SE (R DoubleML "
        "1.0.2 refuses external splits with clustered data).",
    },
    "doubly_robust": {
        "status": "bit-exact",
        "reference": "Open Bandit Pipeline (obp) 0.5.7 DoublyRobust (lambda_ inf and "
        "2; same reward model via q_hat=)",
        "reference_versions": {"obp": "0.5.7"},
        "tolerance": "value 1e-12 rel (observed 1.8e-16); SE identity 1e-10",
        "sides": ["py"],
        "test": [
            "tests/reference_parity/test_ml_causal_obp_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_obp.json",
        ],
        "note": "Python cross-package reference, not R or Stata: no R/Stata "
        "implementation takes K-action logged-bandit data with given "
        "behaviour and evaluation policies. Behaviour propensities, "
        "evaluation policy and a fixed reward model (q_hat=) are columns of "
        "ml_causal_ope.csv, so the estimator is deterministic. obp reports "
        "bootstrap intervals only; the analytic SE is checked as sd(obp round "
        "rewards)/sqrt(n) at 1e-10. The default internal random-forest reward "
        "model is not graded.",
    },
    "gate_test": {
        "status": "bit-exact",
        "reference": "GenericML::GATES 0.2.3 (monotonize = FALSE) with "
        "GenericML::quantile_group membership",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "GenericML": "0.2.3",
            "sandwich": "3.1.1",
        },
        "tolerance": "gamma_k, SE and gamma_K - gamma_1 1e-10 rel (observed 1.9e-15); "
        "group membership identical",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_ml_causal_R_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_R.json",
        ],
        "note": "Only the y/treat/covariates (GATES regression) path is graded; the "
        "descriptive fallback without outcomes is not.",
    },
    "ips": {
        "status": "bit-exact",
        "reference": "Open Bandit Pipeline (obp) 0.5.7 InverseProbabilityWeighting "
        "(lambda_ inf and 2)",
        "reference_versions": {"obp": "0.5.7"},
        "tolerance": "value 1e-12 rel (observed 0.0); SE identity 1e-10",
        "sides": ["py"],
        "test": [
            "tests/reference_parity/test_ml_causal_obp_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_obp.json",
        ],
        "note": "Python cross-package reference, not R or Stata: no R/Stata "
        "implementation takes K-action logged-bandit data with given "
        "behaviour and evaluation policies. Behaviour propensities, "
        "evaluation policy and a fixed reward model (q_hat=) are columns of "
        "ml_causal_ope.csv, so the estimator is deterministic. obp reports "
        "bootstrap intervals only; the analytic SE is checked as sd(obp round "
        "rewards)/sqrt(n) at 1e-10.",
    },
    "model_averaging_dml": {
        "status": "aligned",
        "reference": "ddml::ddml_plm 0.3.1 (shortstack = TRUE, ensemble_type = "
        "'nnls1'), OLS candidates, shared folds",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "ddml": "0.3.1",
            "sandwich": "3.1.1",
        },
        "tolerance": "short-stacking weights 1e-10 (observed 4.4e-14); ddml's final "
        "lm(y_r ~ d_r) with intercept and HC1 SE rebuilt from StatsPAI's "
        "stacked residuals at 1e-10 (observed 4.8e-16)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_ml_causal_dml_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_panel_R.json",
            "tests/reference_parity/_fixtures/ml_causal_doubleml.json",
        ],
        "note": "Alias of dml_model_averaging. Convention difference in the last step "
        "only: StatsPAI solves the no-intercept PLR moment with the HC0 "
        "sandwich (DoubleML convention); ddml fits OLS with an intercept and "
        "reports HC1. The raw point estimates differ by 1.2e-5 rel on the "
        "fixture for that reason.",
    },
    "rate": {
        "status": "aligned",
        "reference": "grf::rank_average_treatment_effect and "
        "rank_average_treatment_effect.fit 2.6.1 (AUTOC, QINI, TOC), "
        "forest outputs held fixed",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "grf": "2.6.1"},
        "tolerance": "Point estimate (AUTOC, QINI) and TOC curve on q = 0.1..1 exact "
        "at 1e-10 rel (observed 4.6e-15), including a 30-group "
        "tied-priority case against rank_average_treatment_effect.fit. "
        "SE is T3: the analytic rank-corrected influence-function SE is "
        "compared with grf's half-sample bootstrap (R = 2000) at 6% rel "
        "(observed 2.6% AUTOC, 1.7% QINI).",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_ml_causal_R_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_R.json",
        ],
        "note": "Factored evidence: grf grows one causal forest (2000 trees, seed 42) "
        "and its OOB tau.hat, Y.hat and W.hat are frozen in the fixture; "
        "StatsPAI's operator runs on exactly those vectors. The forest itself "
        "is not pinnable across implementations (see causal_forest).",
    },
    "snips": {
        "status": "bit-exact",
        "reference": "Open Bandit Pipeline (obp) 0.5.7 "
        "SelfNormalizedInverseProbabilityWeighting",
        "reference_versions": {"obp": "0.5.7"},
        "tolerance": "value 1e-12 rel (observed 0.0); delta-method SE identity 1e-10 "
        "and equal to sp.ope.snips",
        "sides": ["py"],
        "test": [
            "tests/reference_parity/test_ml_causal_obp_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_obp.json",
        ],
        "note": "Python cross-package reference, not R or Stata: no R/Stata "
        "implementation takes K-action logged-bandit data with given "
        "behaviour and evaluation policies. Behaviour propensities, "
        "evaluation policy and a fixed reward model (q_hat=) are columns of "
        "ml_causal_ope.csv, so the estimator is deterministic. obp reports "
        "bootstrap intervals only; the analytic SE is checked as sd(obp round "
        "rewards)/sqrt(n) at 1e-10.",
    },
    "test_calibration": {
        "status": "bit-exact",
        "reference": "grf::test_calibration 2.6.1 (vcov.type HC3 default and HC1), "
        "forest outputs held fixed",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "grf": "2.6.1",
            "sandwich": "3.1.1",
        },
        "tolerance": "coef, se, t 1e-10 rel (observed 1.1e-15); one-sided p-value "
        "1e-8 rel (observed 1.6e-13)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_ml_causal_R_parity.py",
            "tests/reference_parity/_fixtures/ml_causal_R.json",
        ],
        "note": "Alias of calibration_test (asserted to return an identical frame). "
        "Factored evidence: grf grows one causal forest (2000 trees, seed 42) "
        "and its OOB tau.hat, Y.hat and W.hat are frozen in the fixture; "
        "StatsPAI's operator runs on exactly those vectors. The forest itself "
        "is not pinnable across implementations (see causal_forest).",
    },
    # ---- round 2: MR / sensitivity / transport / experimental (17) ----
    "ancova": {
        "status": "bit-exact",
        "reference": "Stata regress, vce(robust) / vce(cluster)",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-9 rel",
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_misc_sens_stata_parity.py"],
        "note": "HC1 and CR1 with t(N-k) / t(G-1).",
    },
    "attrition_bounds": {
        "status": "bit-exact",
        "reference": "Stata leebounds (thresholds held exactly)",
        "reference_versions": {"leebounds": "1.5", "Stata": "18"},
        "tolerance": "1e-12 rel",
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_misc_sens_stata_parity.py"],
        "note": "trimming='exact'; default quantile rule equals sp.lee_bounds.",
    },
    "attrition_test": {
        "status": "bit-exact",
        "reference": "Stata tabulate, chi2; regress",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-10 rel",
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_misc_sens_stata_parity.py"],
        "note": "correction=False (Pearson); default keeps Yates.",
    },
    "balance_check": {
        "status": "bit-exact",
        "reference": "Stata iebaltab (ietoolkit)",
        "reference_versions": {"ietoolkit": "7.5", "Stata": "18"},
        "tolerance": "1e-9 rel",
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_misc_sens_stata_parity.py"],
        "note": "equal_var=True; iebaltab's difference is control minus treatment.",
    },
    "calibrate_confounding_strength": {
        "status": "bit-exact",
        "reference": "R sensemakr::ovb_bounds (kd = ky = k)",
        "reference_versions": {"sensemakr": "0.1.6"},
        "tolerance": "1e-12 rel (CI 1e-10)",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_misc_sens_R_parity.py"],
        "note": "Benchmark covariate x2; dof required.",
    },
    "grapple": {
        "status": "aligned",
        "reference": "GRAPPLE::grappleRobustEst (GitHub jingshuw/GRAPPLE 317e837)",
        "reference_versions": {"GRAPPLE": "0.2.2"},
        "tolerance": "sandwich at GRAPPLE's estimates 1e-6 (l2 1e-12); fitted beta "
        "1e-3, tau2 / SE 1e-4",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_misc_sens_R_parity.py"],
        "note": "GRAPPLE stops its optim / uniroot early (score O(1e-3) at its "
        "estimate); StatsPAI solves the equations to machine precision.",
    },
    "heterogeneity_of_effect": {
        "status": "bit-exact",
        "reference": "R metafor::rma(method = 'DL')",
        "reference_versions": {"metafor": "5.0.1"},
        "tolerance": "1e-12 rel",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_misc_sens_R_parity.py"],
        "note": "tau2, Q, Q p-value, I2.",
    },
    "mediate_sensitivity": {
        "status": "bit-exact",
        "reference": "R mediation::medsens (lm/lm, rho.by = 0.1)",
        "reference_versions": {"mediation": "4.5.1"},
        "tolerance": "1e-10 rel",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_misc_sens_R_parity.py"],
        "note": "eps=sqrt(machine eps) reproduces medsens' stopping rule; default "
        "iterates to the fixed point (= medsens(eps=1e-26)).",
    },
    "mendelian_randomization": {
        "status": "bit-exact",
        "reference": "R MendelianRandomization::mr_allmethods(method = 'main')",
        "reference_versions": {"MendelianRandomization": "0.10.0"},
        "tolerance": "1e-12 rel",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_misc_sens_R_parity.py"],
        "note": "IVW, Egger (+intercept), weighted-median estimate; WM bootstrap SE "
        "is Monte Carlo.",
    },
    "mi_estimate": {
        "status": "bit-exact",
        "reference": "R mice::pool; Stata mi estimate: regress",
        "reference_versions": {"mice": "3.19.0", "Stata": "18"},
        "tolerance": "1e-9 rel (p, CI via t quantiles at fractional df); 1e-12 "
        "est/SE/df",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_misc_sens_R_parity.py",
            "tests/reference_parity/test_misc_sens_stata_parity.py",
        ],
        "note": "Pooling only, on R's five mice imputations; Barnard-Rubin df with "
        "dfcom = residual df. Stata's FMI is the Barnard-Rubin small-sample "
        "FMI (fmi_barnard_rubin).",
    },
    "mr_bma": {
        "status": "bit-exact",
        "reference": "Zuber et al. summary_mvMR_BF (GitHub verena-zuber/demo_AMD "
        "4981b5a)",
        "reference_versions": {"summary_mvMR_BF.R": "demo_AMD@4981b5a"},
        "tolerance": "1e-12 rel",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_misc_sens_R_parity.py"],
        "note": "IVW-scaled Bayes factor, prior sd 0.5, prior_prob 0.5 and 0.1; "
        "method='bic' keeps the pre-1.30 quantity.",
    },
    "mr_mediation": {
        "status": "bit-exact",
        "reference": "R MendelianRandomization::mr_ivw + mr_mvivw",
        "reference_versions": {"MendelianRandomization": "0.10.0"},
        "tolerance": "1e-12 rel",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_misc_sens_R_parity.py"],
        "note": "Total and direct effects; the indirect SE has no reference.",
    },
    "negd": {
        "status": "bit-exact",
        "reference": "Stata regress, vce(robust)",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-9 rel",
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_misc_sens_stata_parity.py"],
        "note": "ANCOVA and change-score forms.",
    },
    "subgroup_analysis": {
        "status": "bit-exact",
        "reference": "Stata regress + testparm",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-9 rel",
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_misc_sens_stata_parity.py"],
        "note": "HC1 and classical; interaction test as F (testparm) and chi2 = qF.",
    },
    "synthesise_evidence": {
        "status": "bit-exact",
        "reference": "R metafor::rma(method = 'FE')",
        "reference_versions": {"metafor": "5.0.1"},
        "tolerance": "1e-12 rel",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_misc_sens_R_parity.py"],
        "note": "weight_mode='inverse_variance' only.",
    },
    "transport_weights_fn": {
        "status": "bit-exact",
        "reference": "R glm + quantile(type = 7) + sandwich::vcovHC(HC0)",
        "reference_versions": {"sandwich": "3.1.1"},
        "tolerance": "1e-12 rel",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_misc_sens_R_parity.py"],
        "note": "Composition of base-R pieces; no canonical transport package exists.",
    },
    "unified_sensitivity": {
        "status": "bit-exact",
        "reference": "R EValue::evalues.OLS; sensemakr::sensemakr (rv_q, rv_qa)",
        "reference_versions": {"EValue": "4.1.4", "sensemakr": "0.1.6"},
        "tolerance": "1e-10 rel",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_misc_sens_R_parity.py"],
        "note": "Oster component equals sp.oster_delta (psacalc-graded).",
    },
    # ---- round 2: synthetic-control inference tools, shift-share political (8) ----
    "conformal_synth": {
        "status": "bit-exact",
        "reference": "scinference (Chernozhukov-Wuthrich-Zhu authors' package, GitHub "
        "kwuthrich/scinference 567c688): estimation_method='sc', "
        "permutation_method='mb'",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "scinference": "0.0.0.9000 " "567c6889ce0a1d269a62d415b88aa6baf723a3fe",
            "limSolve": "2.0.3",
            "quadprog": "1.5.8",
        },
        "tolerance": "p-values exact (rank statistics); ATT 1e-9 rel (observed "
        "9e-16); CI end points exact on the grid",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_synth_rest_R_parity.py",
            "tests/reference_parity/_fixtures/synth_rest_R.json",
        ],
        "note": "Joint moving-block p-values on a 45-point grid, pointwise p at 0, "
        "pointwise CIs at alpha 0.1. At two extreme grid values "
        "limSolve::lsei fails (IsError) and scinference uses infeasible "
        "weights (reference defect); there StatsPAI is held to scinference's "
        "code with quadprog as the solver.",
    },
    "multi_outcome_synth": {
        "status": "bit-exact",
        "reference": "augsynth::augsynth_multiout 0.2.0 (progfunc='None', scm=TRUE, "
        "combine_method 'concat' / 'avg'); synth_qp re-run at OSQP eps "
        "1e-12",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "augsynth": "0.2.0",
            "osqp": "1.0.0",
        },
        "tolerance": "weights 1e-10 abs (observed <= 9e-15); per-outcome ATT 1e-9 rel",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_synth_rest_R_parity.py",
            "tests/reference_parity/_fixtures/synth_rest_R.json",
        ],
        "note": "Shared weights and per-outcome ATTs only. The placebo SE / p-values "
        "and the Fisher joint p-value are StatsPAI's own (augsynth offers "
        "only conformal inference for multiple outcomes). Stock augsynth (eps "
        "1e-8) agrees to 1e-6.",
    },
    "shift_share_political": {
        "status": "bit-exact",
        "reference": "AER::ivreg + sandwich HC1; ShiftShareSE::ivreg_ss (EHW / AKM / "
        "AKM0); bartik.weight::bw; anova(lm) share balance",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "AER": "1.2.16",
            "sandwich": "3.1.1",
            "ShiftShareSE": "1.1.0",
            "bartik.weight": "0.1.0",
        },
        "tolerance": "estimate / SEs / Rotemberg / F 1e-9 rel (observed <= 5e-15); "
        "AKM p-value 1e-7 rel (ShiftShareSE uses 2*(1-pnorm), "
        "cancellation at p ~ 1e-9)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_synth_rest_R_parity.py",
            "tests/reference_parity/_fixtures/synth_rest_R.json",
        ],
        "note": "Long difference (period 5 minus 1) with national shock g = shocks[5] "
        "- shocks[1], leave_one_out=False. AKM / AKM0 in diagnostics; "
        "Rotemberg alpha_k, beta_k are GPSS.",
    },
    "shift_share_political_panel": {
        "status": "bit-exact",
        "reference": "fixest::feols 0.14.0 (ssc adj=FALSE, cluster.adj=FALSE; unit / "
        "time / two-way clusters; unit, time, two-way FE; unbalanced "
        "panel); ShiftShareSE::ivreg_ss with FE dummies (AKM); "
        "bartik.weight::bw with FE dummies; AER + HC0 per period",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "fixest": "0.14.0",
            "ShiftShareSE": "1.1.0",
            "bartik.weight": "0.1.0",
            "AER": "1.2.16",
            "sandwich": "3.1.1",
        },
        "tolerance": "estimate / SEs / Rotemberg / first-stage F 1e-9 rel (observed "
        "<= 1.1e-14)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_synth_rest_R_parity.py",
            "tests/reference_parity/_fixtures/synth_rest_R.json",
        ],
        "note": "Cluster SEs are CR0 (two-way = Cameron-Gelbach-Miller); AKM shock "
        "clusters are industry x period because the fixture's shocks vary "
        "over time. first_stage_F is the textbook partial F with FE df "
        "(computed from feols RSS); fixest's fitstat 'ivf1' is a different "
        "quantity and not compared.",
    },
    "synth_donor_sensitivity": {
        "status": "bit-exact",
        "reference": "Synth::synth 1.1.10 on the replayed numpy donor subsets "
        "(custom.v matched; quadprog exact)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "Synth": "1.1.10",
            "quadprog": "1.5.8",
        },
        "tolerance": "ATT and pre-RMSPE 1e-9 rel (observed <= 1e-13)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_synth_rest_R_parity.py",
            "tests/reference_parity/_fixtures/synth_rest_R.json",
        ],
        "note": "The subsets (k=6, n_samples=5, seed=7) are replayed in "
        "_generate_synth_rest_data.py and the test asserts the replay equals "
        "the function's draws.",
    },
    "synth_loo": {
        "status": "bit-exact",
        "reference": "Synth::synth 1.1.10 (per fit, custom.v matched; QP solved "
        "exactly by quadprog::solve.QP)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "Synth": "1.1.10",
            "quadprog": "1.5.8",
            "kernlab": "0.9.33",
        },
        "tolerance": "ATT and pre-RMSPE 1e-9 rel (observed <= 3e-14)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_synth_rest_R_parity.py",
            "tests/reference_parity/_fixtures/synth_rest_R.json",
        ],
        "note": "Each leave-one-out fit is Synth's QP with custom.v = var_k / "
        "range_k^2, which makes Synth's sd-scaled problem identical to "
        "StatsPAI's range-scaled equal-V problem; Synth's own ipop solve "
        "agrees at ipop precision (<= 1e-4). The se / pvalue columns are "
        "descriptive i.i.d.-gap quantities with no reference. Regenerate via "
        "_generate_synth_rest_R.R.",
    },
    "synth_rmspe_filter": {
        "status": "bit-exact",
        "reference": "SCtools::mspe.test 0.3.3.1 (discard.extreme, mspe.limit 2/5/20) "
        "on Synth fits",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "SCtools": "0.3.3.1",
            "Synth": "1.1.10",
            "quadprog": "1.5.8",
        },
        "tolerance": "placebo pre-RMSPE and ratios 1e-9 rel (observed <= 1e-13); "
        "p-values and kept counts exact",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_synth_rest_R_parity.py",
            "tests/reference_parity/_fixtures/synth_rest_R.json",
        ],
        "note": "SCtools conventions via metric='mspe', "
        "placebo_pool='exclude_treated'; the default (RMSPE scale, treated "
        "unit in placebo pools) is also held to the same fits with "
        "placebo_pool='include_treated'. mspe.test is run on a tdf object "
        "built as generate.placebos builds it.",
    },
    "synth_time_placebo": {
        "status": "bit-exact",
        "reference": "Synth::synth 1.1.10 (per placebo time, custom.v matched; "
        "quadprog exact)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "Synth": "1.1.10",
            "quadprog": "1.5.8",
        },
        "tolerance": "placebo ATT 1e-9 rel (observed <= 1e-13) on identified placebo "
        "times",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_synth_rest_R_parity.py",
            "tests/reference_parity/_fixtures/synth_rest_R.json",
        ],
        "note": "Post-treatment rows are discarded before the placebo fits. Placebo "
        "times with fewer pre-periods than donors have non-unique weights and "
        "are not compared (7 of 12 on the fixture).",
    },
}


# --------------------------------------------------------------------------- #
#  Source parsers
# --------------------------------------------------------------------------- #
def _load_compare_module() -> Any:
    """Import ``tests/r_parity/compare.py`` to reuse its registered tolerances."""
    compare = R_PARITY / "compare.py"
    spec = importlib.util.spec_from_file_location("statspai_parity_compare", compare)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {compare}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_readme_modules() -> Dict[str, Dict[str, str]]:
    """Parse the module table → ``module_number -> {label, py_api, reference}``.

    The README rows look like::

        | 03 | HDFE 2-way FE | `sp.fast.feols` | `fixest::feols` |
    """
    readme = R_PARITY / "README.md"
    out: Dict[str, Dict[str, str]] = {}
    for line in readme.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        number, label, py_api, reference = parts[:4]
        out[number.zfill(2)] = {
            "label": _strip_md(label),
            "py_api": _strip_md(py_api),
            "reference": _strip_md(reference),
        }
    return out


def _parse_renv_versions() -> Dict[str, str]:
    """Map R package name -> pinned version from ``renv.lock``."""
    lock = R_PARITY / "renv.lock"
    if not lock.exists():
        return {}
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    pkgs = payload.get("Packages", {})
    return {
        name: str(meta.get("Version", ""))
        for name, meta in pkgs.items()
        if isinstance(meta, dict)
    }


def _strip_md(text: str) -> str:
    return text.replace("`", "").replace("\\", "").strip()


def _leaf_functions(py_api: str) -> List[str]:
    """Extract every leaf ``sp.*`` function name referenced in a call string."""
    names: List[str] = []
    for match in _SP_LEAF_RE.findall(py_api):
        leaf = match.split(".")[-1]
        if leaf and leaf not in names:
            names.append(leaf)
    return names


def _reference_packages(reference: str) -> List[str]:
    """Pull the R package names (token before ``::``) out of a reference cell."""
    return re.findall(r"([A-Za-z][A-Za-z0-9.]*)::", reference)


# --------------------------------------------------------------------------- #
#  Per-module result joining
# --------------------------------------------------------------------------- #
def _max_attr(rows: List[Any], attr: str) -> Optional[float]:
    """Worst-case value of ``attr`` over ``rows`` (compare.RowDiff objects)."""
    vals = [
        v
        for d in rows
        if (v := getattr(d, attr, None)) is not None and math.isfinite(v)
    ]
    return max(vals) if vals else None


def _headline_attrs(metric: str) -> Tuple[str, str, str]:
    """Map a HEADLINE ``metric`` to (R-attr, Stata-attr, TOLERANCES key)."""
    if metric == "abs_est":
        return "abs_est", "abs_est_st", "abs_est"
    if metric == "rel_se":
        return "rel_se", "rel_se_st", "rel_se"
    return "rel_est", "rel_est_st", "rel_est"


def _module_provenance(module_id: str) -> Dict[str, Any]:
    """Read the R-side provenance block (versions, platform) for a module."""
    path = R_PARITY / "results" / f"{module_id}_R.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload.get("provenance", {}) if isinstance(payload, dict) else {}


# --------------------------------------------------------------------------- #
#  Index assembly
# --------------------------------------------------------------------------- #
def build_track_a_records(
    compare: Any,
    renv: Dict[str, str],
    warnings: List[str],
) -> List[Dict[str, Any]]:
    """Build one parity record per (function, Track A module).

    The headline statistic, metric, and pass verdict are taken from the
    project's own ``compare.HEADLINE`` / ``compare.collect`` /
    ``compare.TOLERANCES`` — the exact selection the JSS Appendix B
    reports — so the index never claims a tighter grade than the
    committed comparison actually supports.
    """
    modules = _parse_readme_modules()
    records: List[Dict[str, Any]] = []

    for module_num, meta in modules.items():
        py_json = sorted((R_PARITY / "results").glob(f"{module_num}_*_py.json"))
        if not py_json:
            continue
        module_id = py_json[0].stem[: -len("_py")]

        diffs = compare.collect(module_id)
        if not diffs:
            continue
        stata_present = any(getattr(d, "Stata_est", None) is not None for d in diffs)
        sides = ["py", "R"] + (["Stata"] if stata_present else [])

        hspec = compare.HEADLINE.get(module_id, {})
        metric = hspec.get("metric", "rel_est")
        filt = hspec.get("headline_filter")
        hrows = [d for d in diffs if filt(d)] if filt else list(diffs)
        if not hrows:
            hrows = list(diffs)

        est_attr, st_attr, tol_key = _headline_attrs(metric)
        tol = compare.TOLERANCES.get(module_id, {})
        tol_val = tol.get(tol_key)
        tier = compare.tolerance_tier(module_id)

        rel_vs_R = _max_attr(hrows, est_attr)
        rel_vs_St = _max_attr(hrows, st_attr)

        # Guard: the committed golden must satisfy its own registered
        # tolerance at the headline. If it does not, fail loud (CLAUDE.md
        # §7) rather than ship an over-claimed grade.
        def _ok(v: Optional[float]) -> bool:
            return v is None or tol_val is None or v <= tol_val * (1 + 1e-9)

        # A module may carry a *registered* Stata-side convention gap, where
        # the third language disagrees for a documented, located reason while
        # py<->R stays inside budget. compare.py::STATA_HEADLINE_GAP_EXCEPTIONS
        # is where that is declared, with the evidence. Such a module is not
        # over-claiming: it demotes to `aligned` (never `bit-exact`) and the
        # gap is recorded in its note rather than warned away, but it is not a
        # budget violation of the py<->R contract the tolerance governs.
        stata_gap_documented = module_id in compare.STATA_HEADLINE_GAP_EXCEPTIONS
        r_passes = _ok(rel_vs_R)
        stata_passes = _ok(rel_vs_St)
        passes = r_passes and stata_passes
        if not passes and not (r_passes and stata_gap_documented):
            worst = max(v for v in (rel_vs_R, rel_vs_St) if v is not None)
            warnings.append(
                f"{module_id}: headline {metric}={worst:.3g} "
                f"exceeds registered {tol_key}<={tol_val:g}"
            )

        status = "bit-exact" if (tier == "machine" and passes) else "aligned"

        provenance = _module_provenance(module_id)
        ref_pkgs = _reference_packages(meta["reference"])
        ref_versions: Dict[str, str] = {}
        r_version = provenance.get("r_version", "")
        if r_version:
            ref_versions["R"] = r_version
        for pkg in ref_pkgs:
            if pkg in renv:
                ref_versions[pkg] = renv[pkg]

        tests = [f"tests/r_parity/{module_id}.py", f"tests/r_parity/{module_id}.R"]
        if stata_present:
            tests.append(f"tests/stata_parity/{module_id}.do")

        tol_str = ", ".join(
            f"{k}<={v:g}" for k, v in tol.items() if isinstance(v, (int, float))
        )

        for fn in _leaf_functions(meta["py_api"]):
            records.append(
                {
                    "function": fn,
                    "status": status,
                    "source": "track_a",
                    "module_id": module_id,
                    "label": meta["label"],
                    "reference": meta["reference"],
                    "reference_versions": ref_versions,
                    "python_call": meta["py_api"],
                    # What the StatsPAI side of the module executes; checked
                    # against a call trace (test_parity_implementation_provenance).
                    "implementation": compare.implementation_kind(module_id),
                    # T2 same-byte parity, T3 seed-replicated stochastic
                    # equivalence, or T4 documented reference disagreement.
                    "evidence_grade": compare.evidence_grade(module_id),
                    "tolerance": tol_str,
                    "tier": tier,
                    "sides": sides,
                    "headline": {
                        "statistic": [d.statistic for d in hrows],
                        "metric": metric,
                        "tolerance": tol_val,
                        "rel_vs_R": rel_vs_R,
                        "rel_vs_Stata": rel_vs_St,
                    },
                    "point_estimate_rel": {
                        "R": _max_attr(hrows, "rel_est"),
                        "Stata": _max_attr(hrows, "rel_est_st"),
                    },
                    "se_rel": {
                        "R": _max_attr(hrows, "rel_se"),
                        "Stata": _max_attr(hrows, "rel_se_st"),
                    },
                    "test": tests,
                    "last_verified": r_version,
                    "notes": [],
                }
            )
    return records


def _registered_functions() -> set:
    """Public surface via ``sp.list_functions()``; empty set if unimportable."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        import statspai as sp

        return set(sp.list_functions())
    except Exception:  # pragma: no cover - keep generator usable off-tree
        return set()


def _scan_test_calls(directory: Path) -> Dict[str, List[str]]:
    """Map ``leaf_function -> sorted [test file relpaths]`` for a test dir."""
    out: Dict[str, List[str]] = {}
    if not directory.exists():
        return out
    for path in sorted(directory.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT).as_posix()
        leaves = {m.split(".")[-1] for m in _SP_CALL_RE.findall(text)}
        for leaf in leaves:
            if leaf and not leaf.startswith("_") and leaf not in _NON_ESTIMATOR_LEAVES:
                out.setdefault(leaf, []).append(rel)
    return out


def build_reference_parity_records() -> List[Dict[str, Any]]:
    """Deterministic-DGP / closed-form recovery tests -> analytical-only.

    A safe floor: these recover a known truth within a tolerance but carry no
    cross-package reference, so the honest grade is ``analytical-only``.
    Functions that ALSO have a frozen cross-package reference get promoted in
    :data:`_FROZEN_PROMOTIONS` (merged separately, strongest grade wins).
    """
    records: List[Dict[str, Any]] = []
    for fn, tests in _scan_test_calls(REFERENCE_PARITY).items():
        records.append(
            {
                "function": fn,
                "status": "analytical-only",
                "source": "reference_parity",
                "reference": "",
                "reference_versions": {},
                "tolerance": "",
                "sides": ["py"],
                "test": tests,
                "notes": [
                    "Recovers a known population parameter / closed-form identity "
                    "on a deterministic DGP within tolerance; no cross-package "
                    "reference. See tests/reference_parity/REFERENCES.md."
                ],
            }
        )
    return records


def build_external_parity_records() -> List[Dict[str, Any]]:
    """Published-paper-number replication tests -> external-replication."""
    records: List[Dict[str, Any]] = []
    for fn, tests in _scan_test_calls(EXTERNAL_PARITY).items():
        records.append(
            {
                "function": fn,
                "status": "external-replication",
                "source": "external_parity",
                "reference": (
                    "published reference values "
                    "(tests/external_parity/PUBLISHED_REFERENCE_VALUES.md)"
                ),
                "reference_versions": {},
                "tolerance": "",
                "sides": ["py"],
                "test": tests,
                "notes": [
                    "Reproduces published-paper numbers on a calibrated replica. "
                    "See tests/external_parity/PUBLISHED_REFERENCE_VALUES.md."
                ],
            }
        )
    return records


def build_dispatcher_alias_records(
    track_a: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Credit aliases that reach a Track A module's estimator core.

    The alias table lives in ``statspai._parity_taxonomy`` and every entry
    names the pytest that proves the equivalence on the module's committed
    bytes. This function used to justify the credit by pointing at the
    registry's ``validation_status`` while the registry justified that
    status with its own copy of this table -- a circular citation with no
    artifact behind it (CLAUDE.md §10). The credit is now transitive from a
    measurement instead.
    """
    by_key = {(r["function"], r["module_id"]): r for r in track_a}
    records: List[Dict[str, Any]] = []
    for alias, proof in TRACK_A_ALIASES.items():
        base = None
        for module in proof.module.split(" + "):
            base = by_key.get((proof.canonical, module))
            if base is not None:
                break
        if base is None:
            continue
        rec = dict(base)
        rec["function"] = alias
        rec["source"] = "track_a_alias"
        notes = list(base.get("notes", [])) + [proof.evidence_note()]
        if proof.note:
            notes.append(proof.note)
        rec["notes"] = notes
        tests = list(base.get("test", []) or [])
        if ALIAS_PROOF_TEST not in tests:
            tests.append(ALIAS_PROOF_TEST)
        rec["test"] = tests
        records.append(rec)
    return records


# A test counts as touching an external reference if it loads a committed
# fixture (written by an R / Stata generator) or calls R itself.
_EXTERNAL_EVIDENCE = re.compile(
    r"_fixtures|_FIX\b|Rscript|rpy2|_R\.json|_Stata\.json|_stata\.json"
    r"|backend\s*=\s*[\"']r[\"']"
)


def _check_external_evidence(fn: str, meta: Dict[str, Any]) -> None:
    """Refuse a cross-language grade that no test backs.

    Through 1.27.0, 32 promotions were graded bit-exact against R or Stata
    on tests of closed-form identities that never consulted either -- several
    with a ``reference_versions`` naming an R build nothing had run, one
    asserting in its docstring that an R package "implements the identical
    closed form" in place of comparing against it. Cross-language evidence
    has to be evidence: at least one listed test must load a reference
    fixture or call R, or the entry must say where its embedded constants
    came from in ``provenance`` (the live-captured Stata values behind
    ``did_2x2`` / ``ddd``). Anything else belongs in ``analytical-only``,
    which the reference_parity scan already assigns without an entry here.
    """
    if "sides" not in meta:
        raise ValueError(
            f"_FROZEN_PROMOTIONS[{fn!r}] has no 'sides'. It used to default to "
            "['py', 'R'], which let an entry claim an R comparison by omission."
        )
    if not set(meta["sides"]) & {"R", "Stata"} or meta.get("provenance"):
        return
    for rel in meta["test"]:
        path = REPO_ROOT / rel
        if path.exists() and _EXTERNAL_EVIDENCE.search(
            path.read_text(encoding="utf-8")
        ):
            return
    raise ValueError(
        f"_FROZEN_PROMOTIONS[{fn!r}] claims sides {meta['sides']} but none of "
        f"{meta['test']} loads a reference fixture or calls R, and it records no "
        "'provenance' for embedded constants. Build a real comparison or drop the "
        "entry (the function then keeps its analytical-only grade)."
    )


def build_frozen_promotion_records() -> List[Dict[str, Any]]:
    """Curated frozen-reference bit-exact promotions (see _FROZEN_PROMOTIONS)."""
    records: List[Dict[str, Any]] = []
    for fn, meta in _FROZEN_PROMOTIONS.items():
        _check_external_evidence(fn, meta)
        records.append(
            {
                "function": fn,
                "status": meta["status"],
                "source": "reference_parity_frozen",
                "reference": meta["reference"],
                "reference_versions": meta.get("reference_versions", {}),
                "tolerance": meta["tolerance"],
                "sides": meta["sides"],
                "test": meta["test"],
                "notes": [meta["note"]],
            }
        )
    return records


_GRADE_RANK = {
    "bit-exact": 0,
    "aligned": 1,
    "external-replication": 2,
    "analytical-only": 3,
    "unverified": 4,
}


def build_index() -> Tuple[Dict[str, Any], List[str]]:
    compare = _load_compare_module()
    renv = _parse_renv_versions()
    warnings: List[str] = []
    track_a = build_track_a_records(compare, renv, warnings)
    aliases = build_dispatcher_alias_records(track_a)
    reference = build_reference_parity_records()
    external = build_external_parity_records()
    frozen = build_frozen_promotion_records()

    grade_rank = _GRADE_RANK
    # Merge order matters only for tie context; grade rank decides the winner.
    all_records = track_a + aliases + frozen + external + reference
    # A parity test calls `sp.describe_function` to check metadata and
    # `sp.bibtex` to resolve a citation; neither compares a number, so the
    # call-site scan that builds `reference` would otherwise hand them the
    # grade of whatever estimator the test was really about -- `sp.bibtex`
    # was being published as `external-replication`. Filtering here rather
    # than only in the registry keeps docs/parity.md from printing a claim
    # the registry itself refuses to make.
    all_records = [
        rec for rec in all_records if rec["function"] not in _NON_ESTIMATOR_LEAVES
    ]

    by_fn: Dict[str, Dict[str, Any]] = {}
    extra_modules: Dict[str, List[str]] = {}
    # Every distinct ``sp.*`` call string a dispatcher was certified under,
    # so the variant-specificity note can enumerate them instead of naming
    # only whichever module happened to win the grade tie-break.
    extra_calls: Dict[str, List[str]] = {}
    sources: Dict[str, List[str]] = {}
    all_tests: Dict[str, List[str]] = {}
    for rec in all_records:
        fn = rec["function"]
        sources.setdefault(fn, [])
        if rec["source"] not in sources[fn]:
            sources[fn].append(rec["source"])
        all_tests.setdefault(fn, [])
        for t in rec.get("test", []):
            if t not in all_tests[fn]:
                all_tests[fn].append(t)
        if rec.get("module_id"):
            extra_modules.setdefault(fn, []).append(rec["module_id"])
            call = rec.get("python_call")
            if call:
                extra_calls.setdefault(fn, [])
                if call not in extra_calls[fn]:
                    extra_calls[fn].append(call)
        cur = by_fn.get(fn)
        if cur is None or grade_rank[rec["status"]] < grade_rank[cur["status"]]:
            by_fn[fn] = rec

    for fn, rec in by_fn.items():
        mods = sorted(set(extra_modules.get(fn, [])))
        if len(mods) > 1:
            rec["also_in_modules"] = [m for m in mods if m != rec.get("module_id")]
        other_sources = [s for s in sources.get(fn, []) if s != rec["source"]]
        if other_sources:
            rec["also_verified_by"] = other_sources
        # Record every test that touches this function, not just the winner's.
        winner_tests = set(rec.get("test", []))
        extra_tests = [t for t in all_tests.get(fn, []) if t not in winner_tests]
        if extra_tests:
            rec["additional_tests"] = extra_tests
        # Variant-specificity honesty note for family dispatchers.
        if fn in _DISPATCHERS and rec["status"] in {"bit-exact", "aligned"}:
            calls = extra_calls.get(fn) or [rec.get("python_call", "") or fn]
            certified = ", ".join(sorted(calls))
            note = (
                f"Grade is variant-specific: certified for the tested "
                f"{'calls' if len(calls) > 1 else 'call'} "
                f"({certified}); other {fn}() methods/variants may differ."
            )
            if note not in rec.get("notes", []):
                rec.setdefault("notes", []).append(note)
        # Curated factor-level notes: where an estimator splits into a
        # pinnable closed-form operator and an unpinnable stochastic
        # component, the headline grade alone under-describes the
        # evidence. Copied verbatim from the asserting test.
        for note in _FACTOR_NOTES.get(fn, ()):
            if note not in rec.get("notes", []):
                rec.setdefault("notes", []).append(note)

    # Keep only records for registered public functions. Anything dropped is
    # either scan noise (dataset loaders) or a tested-but-unregistered function
    # (a real registry gap worth flagging) — surface the latter as a warning.
    registered = _registered_functions()
    if registered:
        dropped = sorted(fn for fn in by_fn if fn not in registered)
        for fn in dropped:
            warnings.append(
                f"tested function not registered (invisible to sp.list_functions): {fn}"
            )
        by_fn = {fn: rec for fn, rec in by_fn.items() if fn in registered}

    index = {
        "schema_version": 1,
        "generator": "scripts/build_parity_index.py",
        "taxonomy": [
            "bit-exact",
            "aligned",
            "analytical-only",
            "external-replication",
            "unverified",
        ],
        "records": sorted(by_fn.values(), key=lambda r: r["function"]),
    }
    return index, warnings


# --------------------------------------------------------------------------- #
#  Public markdown matrix (docs/parity.md)
# --------------------------------------------------------------------------- #
def _fmt_rel(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if x == 0:
        return "0"
    return f"{x:.1e}"


def _fmt_versions(versions: Dict[str, str]) -> str:
    parts: List[str] = []
    for pkg, ver in versions.items():
        if pkg == "R":
            m = re.search(r"\d+\.\d+\.\d+", ver)
            parts.append(f"R {m.group(0)}" if m else "R")
        else:
            parts.append(f"{pkg} {ver}")
    return "; ".join(parts) if parts else "—"


def _primary_test_link(rec: Dict[str, Any]) -> str:
    tests = rec.get("test", [])
    if not tests:
        return "—"
    rel = tests[0]
    name = rel.split("/")[-1]
    extra = f" (+{len(tests) - 1})" if len(tests) > 1 else ""
    return f"[`{name}`](../{rel}){extra}"


def _denominators(index: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Split the registered surface into parity-applicable strata.

    A single ``verified / total`` fraction over every registered symbol is
    not a meaningful coverage metric: roughly a quarter of the surface is
    result and exception classes, and another sixth is infrastructure. Both
    are counted here so the honest fraction is visible without anyone having
    to recompute it, and so the paper and the docs quote the same split.
    """
    import inspect

    import statspai as sp
    from statspai import registry as R

    infra_categories = INFRASTRUCTURE_CATEGORIES
    statuses = {
        rec["function"]: rec.get("status", "unverified")
        for rec in index.get("records", [])
    }
    buckets = {
        k: {"cross": 0, "verified": 0, "total": 0}
        for k in ("estimator", "infra", "classes", "all")
    }
    for name in sp.list_functions():
        spec = R._REGISTRY.get(name)
        status = statuses.get(name, "unverified")
        obj = getattr(sp, name, None)
        if inspect.isclass(obj):
            key = "classes"
        elif spec is not None and spec.category in infra_categories:
            key = "infra"
        else:
            key = "estimator"
        for target in (key, "all"):
            buckets[target]["total"] += 1
            if status != "unverified":
                buckets[target]["verified"] += 1
            if status in CROSS_LANGUAGE_STATUSES:
                buckets[target]["cross"] += 1
    return buckets


def _family_coverage(
    index: Dict[str, Any],
) -> List[Tuple[str, Dict[str, int]]]:
    """Per-registry-category coverage over estimator callables only.

    Result classes and infrastructure are excluded for the same reason they
    are excluded from the honest denominators above: they cannot carry a
    parity grade, so counting them would make every family look worse than
    it is. Sorted by size so the largest gaps read first.
    """
    import inspect

    import statspai as sp
    from statspai import registry as R

    infra_categories = INFRASTRUCTURE_CATEGORIES
    # Iterate the *registered* surface, not the index records: the index
    # holds only functions that carry evidence, so keying off it would make
    # every family's denominator equal its numerator and report 100%
    # coverage everywhere.
    statuses = {
        rec["function"]: rec.get("status", "unverified")
        for rec in index.get("records", [])
    }
    families: Dict[str, Dict[str, int]] = {}
    for name in sp.list_functions():
        spec = R._REGISTRY.get(name)
        if spec is None or spec.category in infra_categories:
            continue
        if inspect.isclass(getattr(sp, name, None)):
            continue
        bucket = families.setdefault(
            spec.category, {"cross": 0, "verified": 0, "total": 0}
        )
        bucket["total"] += 1
        status = statuses.get(name, "unverified")
        if status != "unverified":
            bucket["verified"] += 1
        if status in CROSS_LANGUAGE_STATUSES:
            bucket["cross"] += 1
    return sorted(families.items(), key=lambda kv: -kv[1]["total"])


def render_parity_doc(index: Dict[str, Any], total_functions: int) -> str:
    records = index["records"]
    by_status: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_status.setdefault(rec["status"], []).append(rec)
    counts = {k: len(v) for k, v in by_status.items()}
    verified = sum(
        counts.get(g, 0)
        for g in ("bit-exact", "aligned", "analytical-only", "external-replication")
    )
    unverified = total_functions - verified

    out: List[str] = []
    w = out.append
    w("# Cross-language parity matrix")
    w("")
    w(
        "> **Auto-generated — do not hand-edit.** Regenerate with "
        "`python scripts/build_parity_index.py`. Every row traces to a "
        "committed test artifact; nothing here is asserted from memory."
    )
    w("")
    w(
        "StatsPAI's promise is that every number it reports is either "
        "*aligned with an external reference implementation*, *recovered "
        "against a known truth*, or *honestly marked as neither*. This page "
        "makes that promise auditable function-by-function, and keeps the "
        "three cases apart — a method with no Stata or R sibling can reach "
        "the second and never the first, and saying so is the point. Query "
        "any function programmatically:"
    )
    w("")
    w("```python")
    w("import statspai as sp")
    w('sp.parity_status("feols")     # one function')
    w("sp.parity_matrix()            # the whole matrix")
    w("sp.parity_summary()           # honest coverage counts")
    w("```")
    w("")
    w("## Taxonomy")
    w("")
    w("| grade | meaning |")
    w("| --- | --- |")
    w(
        "| `bit-exact` | matches a named R/Stata reference on identical input "
        "bytes within the strict pre-registered tolerance (headline estimate "
        "and SE relative error ≤ 1e-6; most modules land at 1e-9 to 1e-15). "
        "A numerical-tolerance grade, not IEEE bitwise equality, and a "
        "statement about the compared configuration only — see "
        "`sp.validation_scope` for which options and outputs were compared |"
    )
    w(
        "| `aligned` | matches a named reference within a documented, "
        "pre-registered looser tolerance (cross-fit / convention disagreement) |"
    )
    w(
        "| `analytical-only` | recovers a known population parameter on a "
        "deterministic DGP, or a closed-form identity (no cross-package reference) |"
    )
    w(
        "| `external-replication` | reproduces published-paper numbers on a "
        "calibrated replica |"
    )
    w(
        "| `unverified` | registered public API, no qualifying numerical-parity "
        "evidence attached yet — **the honest gap** |"
    )
    w("")
    cross = sum(counts.get(g, 0) for g in CROSS_LANGUAGE_STATUSES)
    internal = sum(counts.get(g, 0) for g in INTERNAL_EVIDENCE_STATUSES)
    denom = _denominators(index)

    w("## Coverage at a glance")
    w("")
    w(
        "Read the two evidence kinds separately. Only the first answers "
        '"does StatsPAI agree with Stata/R"; the second answers "does '
        'StatsPAI recover the right answer", which is a different — and for '
        "methods with no Stata/R sibling, the only available — question. "
        'Summing them into one "verified" figure would let the smaller '
        "claim borrow the authority of the larger one, so this page does not "
        "print that total."
    )
    w("")
    w("| evidence kind | grade | functions |")
    w("| --- | --- | ---: |")
    w(
        f"| **Compared against R/Stata** (T2) | bit-exact | "
        f"{counts.get('bit-exact', 0)} |"
    )
    w(f"| | aligned | {counts.get('aligned', 0)} |")
    w(f"| | **subtotal** | **{cross}** |")
    w(
        f"| **No external software reference** | analytical-only (T1) | "
        f"{counts.get('analytical-only', 0)} |"
    )
    w(
        f"| | external-replication (published numbers) | "
        f"{counts.get('external-replication', 0)} |"
    )
    w(f"| | **subtotal** | **{internal}** |")
    w(f"| No numerical evidence yet | unverified | {unverified} |")
    w("")
    w("### Honest denominators")
    w("")
    w(
        "The all-registered denominator understates coverage: it counts "
        "result and exception classes, which can never carry a parity grade, "
        "and infrastructure functions that render tables, draw plots, build "
        "agent schemas or load data. The estimator denominator is the number "
        "to drive release over release."
    )
    w("")
    w("| denominator | cross-language | any evidence | total | cross-lang share |")
    w("| --- | ---: | ---: | ---: | ---: |")
    for label, key in (
        ("estimator callables", "estimator"),
        ("infrastructure (parity N/A)", "infra"),
        ("result / exception classes", "classes"),
        ("**all registered**", "all"),
    ):
        d = denom[key]
        share = f"{d['cross'] / d['total'] * 100:.1f}%" if d["total"] else "—"
        w(f"| {label} | {d['cross']} | {d['verified']} | {d['total']} " f"| {share} |")
    w("")

    w("### Coverage by estimator family")
    w("")
    w(
        "Families with zero cross-language rows are the highest-leverage "
        "targets when a reference implementation exists, and the honest "
        "ceiling when one does not — a method with no Stata/R sibling can "
        "reach `analytical-only` and no further. This table is generated "
        "from the same records as the rest of the page, so it cannot drift "
        "from them."
    )
    w("")
    w("| family | cross-language | any evidence | estimator callables |")
    w("| --- | ---: | ---: | ---: |")
    for family, counts_ in _family_coverage(index):
        w(
            f"| {family} | {counts_['cross']} | {counts_['verified']} "
            f"| {counts_['total']} |"
        )
    w("")

    # Bit-exact + aligned: full cross-language detail.
    for grade, blurb in (
        ("bit-exact", "Machine-tolerance agreement with a named R/Stata reference."),
        ("aligned", "Agreement within a documented, pre-registered looser tolerance."),
    ):
        recs = sorted(by_status.get(grade, []), key=lambda r: r["function"])
        if not recs:
            continue
        w(f"## {grade} — {len(recs)} functions")
        w("")
        w(f"{blurb}")
        w("")
        w(
            "| function | reference | versions | tolerance "
            "| rel err (R / Stata) | test |"
        )
        w("| --- | --- | --- | --- | --- | --- |")
        for r in recs:
            head = r.get("headline", {}) or {}
            rel = (
                f"{_fmt_rel(head.get('rel_vs_R'))} / "
                f"{_fmt_rel(head.get('rel_vs_Stata'))}"
            )
            w(
                f"| `{r['function']}` | {r.get('reference', '') or '—'} "
                f"| {_fmt_versions(r.get('reference_versions', {}))} "
                f"| {r.get('tolerance', '') or '—'} | {rel} "
                f"| {_primary_test_link(r)} |"
            )
        w("")

    # External replication.
    recs = sorted(
        by_status.get("external-replication", []), key=lambda r: r["function"]
    )
    if recs:
        w(f"## external-replication — {len(recs)} functions")
        w("")
        w(
            "Reproduces published-paper numbers; sources in "
            "`tests/external_parity/PUBLISHED_REFERENCE_VALUES.md`."
        )
        w("")
        w("| function | test |")
        w("| --- | --- |")
        for r in recs:
            w(f"| `{r['function']}` | {_primary_test_link(r)} |")
        w("")

    # Analytical-only.
    recs = sorted(by_status.get("analytical-only", []), key=lambda r: r["function"])
    if recs:
        w(f"## analytical-only — {len(recs)} functions")
        w("")
        w(
            "Recovers a known DGP truth / closed-form identity within tolerance; "
            "no cross-package reference. See "
            "`tests/reference_parity/REFERENCES.md`."
        )
        w("")
        w("| function | test |")
        w("| --- | --- |")
        for r in recs:
            w(f"| `{r['function']}` | {_primary_test_link(r)} |")
        w("")

    w(f"## unverified — {unverified} functions")
    w("")
    w(
        "These are registered public functions with no cross-language or "
        "published-reference parity evidence attached **yet**. This is the "
        "honest coverage gap, not a claim of incorrectness — many are frontier "
        "methods with no Stata/R sibling to align against. Query any of them "
        "with `sp.parity_status(name)`; the closing roadmap lives in "
        "[`docs/dev/parity_status_roadmap.md`](dev/parity_status_roadmap.md)."
    )
    w("")
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed snapshot is stale (CI drift gate).",
    )
    args = parser.parse_args(argv)

    index, warnings = build_index()
    serialized = json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    total_functions = len(_registered_functions()) or len(index["records"])
    doc = render_parity_doc(index, total_functions)

    if warnings:
        print("PARITY GUARD WARNINGS (committed golden under-performs its budget):")
        for w in warnings:
            print(f"  ! {w}", file=sys.stderr)

    if args.check:
        stale = []
        cur_json = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else ""
        cur_doc = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
        if cur_json != serialized:
            stale.append("src/statspai/_parity_index.json")
        if cur_doc != doc:
            stale.append("docs/parity.md")
        if stale:
            print(
                "parity artifacts stale: " + ", ".join(stale) + "\n"
                "Run: python scripts/build_parity_index.py",
                file=sys.stderr,
            )
            return 1
        print(f"parity index up to date ({len(index['records'])} function records).")
        return 0

    SNAPSHOT.write_text(serialized, encoding="utf-8")
    DOC.write_text(doc, encoding="utf-8")
    counts: Dict[str, int] = {}
    for rec in index["records"]:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    print(f"wrote {SNAPSHOT.relative_to(REPO_ROOT)} ({len(index['records'])} records)")
    print(f"wrote {DOC.relative_to(REPO_ROOT)}")
    for status, n in sorted(counts.items()):
        print(f"  {status:22s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
