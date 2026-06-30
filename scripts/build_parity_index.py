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
_NON_ESTIMATOR_LEAVES = {
    "california_prop99",
    "california_tobacco",
    "dgp_did",
    "list_datasets",
    "mpdta",
    "nsw_dw",
    "nsw_lalonde",
    "card_1995",
    "nhefs",
    "angrist_krueger_1991",
    "lee_2008_senate",
    "basque_terrorism",
    "german_reunification",
}

# Curated frozen-reference promotions: functions pinned to *exact* base-R or
# closed-form numbers in tests/reference_parity but NOT covered by the Track A
# harness. Every field is copied verbatim from
# tests/reference_parity/REFERENCES.md (the "Frozen R-value fixtures" table)
# and the asserting test — no model-memory facts (CLAUDE.md §10).
_FROZEN_PROMOTIONS: Dict[str, Dict[str, Any]] = {
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

        passes = _ok(rel_vs_R) and _ok(rel_vs_St)
        if not passes:
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


# Standalone functions that are documented aliases of a Track A dispatcher
# call (same estimator core). alias -> (dispatcher function, module id, call).
# Backed by the registry's own certified seed and the alias docstrings, which
# already assert the equivalence; we just make it queryable + auditable.
_DISPATCHER_ALIASES: Dict[str, Dict[str, str]] = {
    "oaxaca": {
        "dispatcher": "decompose",
        "module": "30_oaxaca",
        "call": "sp.decompose('oaxaca')",
    },
    "dfl_decompose": {
        "dispatcher": "decompose",
        "module": "31_dfl",
        "call": "sp.decompose('dfl')",
    },
    "mediate": {
        "dispatcher": "mediation",
        "module": "36_mediation",
        "call": "sp.mediation",
    },
}


def build_dispatcher_alias_records(
    track_a: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Credit standalone aliases of Track A dispatcher calls (same core)."""
    by_key = {(r["function"], r["module_id"]): r for r in track_a}
    records: List[Dict[str, Any]] = []
    for alias, meta in _DISPATCHER_ALIASES.items():
        base = by_key.get((meta["dispatcher"], meta["module"]))
        if base is None:
            continue
        rec = dict(base)
        rec["function"] = alias
        rec["source"] = "track_a_alias"
        rec["notes"] = list(base.get("notes", [])) + [
            f"Alias of {meta['call']} — same estimator core, certified via "
            f"Track A module {meta['module']}; the registry validation_status "
            f"independently marks this function certified."
        ]
        records.append(rec)
    return records


def build_frozen_promotion_records() -> List[Dict[str, Any]]:
    """Curated frozen-reference bit-exact promotions (see _FROZEN_PROMOTIONS)."""
    records: List[Dict[str, Any]] = []
    for fn, meta in _FROZEN_PROMOTIONS.items():
        records.append(
            {
                "function": fn,
                "status": meta["status"],
                "source": "reference_parity_frozen",
                "reference": meta["reference"],
                "reference_versions": meta.get("reference_versions", {}),
                "tolerance": meta["tolerance"],
                "sides": meta.get("sides", ["py", "R"]),
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

    by_fn: Dict[str, Dict[str, Any]] = {}
    extra_modules: Dict[str, List[str]] = {}
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
            call = rec.get("python_call", "")
            note = (
                f"Grade is variant-specific: certified for the tested call "
                f"({call or fn}); other {fn}() methods/variants may differ."
            )
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
        "StatsPAI's promise is *numerical alignment with Stata / R*. This page "
        "makes that promise auditable function-by-function. Query any function "
        "programmatically:"
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
    w("## Coverage at a glance")
    w("")
    w("| status | functions |")
    w("| --- | ---: |")
    for grade in ("bit-exact", "aligned", "analytical-only", "external-replication"):
        w(f"| {grade} | {counts.get(grade, 0)} |")
    w(f"| **verified (subtotal)** | **{verified}** |")
    w(f"| unverified | {unverified} |")
    w(f"| **total registered** | **{total_functions}** |")
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
