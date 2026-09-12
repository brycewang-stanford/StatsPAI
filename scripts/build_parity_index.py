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
  * ``bit-exact``           — matches a named R/Stata reference to the
                              machine tolerance tier (rel <= 1e-6).
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
        "| `bit-exact` | matches a named R/Stata reference to machine tolerance "
        "(headline relative error ≤ 1e-6) |"
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
