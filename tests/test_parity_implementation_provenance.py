"""Every Track A row states which implementation its StatsPAI side ran.

A parity row supports a claim about a StatsPAI-native algorithm only if the
Python side executed one. ``tests/r_parity/compare.py::
IMPLEMENTATION_PROVENANCE`` records the exceptions (official ports,
third-party Python libraries, reference backends), and
``tests/r_parity/results/_implementation_trace.json`` -- written by
``scripts/trace_parity_provenance.py``, which *runs* each module under a
profiler -- records what actually happened. This test holds the two
together, so a module cannot quietly start delegating (or stop) without the
table, the appendix marker and the manuscript counts following.

The trace is bound to the implementation, not only to the entry script: it
records the hash of every StatsPAI file on each module's estimation path and
of the committed result, so an internal delegation change behind an
unchanged script makes it stale. A package outside the reviewed lists makes
a module ``unclassified`` rather than native.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "r_parity"
TRACE = PARITY / "results" / "_implementation_trace.json"

#: Packages whose use makes a row third-party even if StatsPAI wraps it.
THIRD_PARTY_ESTIMATORS = {
    "linearmodels",
    "statsmodels",
    "pyfixest",
    "econml",
    "doubleml",
    "lifelines",
    "causalml",
}
#: Python ports maintained by the method's own authors.
OFFICIAL_PORTS = {"rdrobust", "rddensity", "rdd", "rdmulti", "rdlocrand"}
#: statsmodels namespaces that are utilities, not estimators.
STATSMODELS_UTILITY_PREFIXES = ("statsmodels.tools.", "statsmodels.compat")
#: User-chosen nuisance learners: they fit E[Y|X] / E[D|X], not the estimator.
NUISANCE_LEARNERS = {"sklearn", "lightgbm", "xgboost"}


def _load_tracer():
    spec = importlib.util.spec_from_file_location(
        "_trace_prov", ROOT / "scripts" / "trace_parity_provenance.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_compare():
    spec = importlib.util.spec_from_file_location(
        "_compare_prov", PARITY / "compare.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trace() -> dict:
    return json.loads(TRACE.read_text(encoding="utf-8"))["modules"]


def _observed_kind(rec: dict, side_checks: frozenset = frozenset()) -> str:
    if rec["rscript_launches"] > 0:
        return "reference_backend"
    estimator_pkgs = set()
    for call in rec["boundary_calls"]:
        pkg = call["package"]
        if pkg in side_checks:
            continue
        if pkg == "statsmodels" and call["callee"].startswith(
            STATSMODELS_UTILITY_PREFIXES
        ):
            continue
        estimator_pkgs.add(pkg)
    if estimator_pkgs & THIRD_PARTY_ESTIMATORS:
        return "third_party_python"
    if estimator_pkgs & OFFICIAL_PORTS:
        return "official_python_port"
    if estimator_pkgs - NUISANCE_LEARNERS:
        # Never default an unreviewed package to native.
        return "unclassified"
    return "native"


def test_every_module_is_traced_and_the_trace_is_current():
    compare = _load_compare()
    tracer = _load_tracer()
    trace = _trace()
    assert set(trace) == set(compare.TOLERANCES)
    stale = {
        stem: reasons
        for stem, rec in trace.items()
        if (reasons := tracer.stale_reasons(stem, rec))
    }
    assert not stale, (
        "Track A traces no longer describe the tree (entry script, estimation-"
        "path source, or committed result changed); re-run "
        f"`python scripts/trace_parity_provenance.py {' '.join(s[:2] for s in stale)}`"
        f": {stale}"
    )
    errors = {stem: rec["error"] for stem, rec in trace.items() if rec.get("error")}
    assert not errors


def test_registered_provenance_matches_the_trace():
    compare = _load_compare()

    def observed(stem, rec):
        allowed = frozenset(compare.IMPLEMENTATION_SIDE_CHECKS.get(stem, {}))
        return _observed_kind(rec, allowed)

    mismatches = {
        stem: (compare.implementation_kind(stem), observed(stem, rec))
        for stem, rec in _trace().items()
        if compare.implementation_kind(stem) != observed(stem, rec)
    }
    # A side-check allowance must be real: the package is used, and only by a
    # module that is otherwise native.
    for stem, pkgs in compare.IMPLEMENTATION_SIDE_CHECKS.items():
        assert set(pkgs) <= set(_trace()[stem]["packages"]), stem
    assert not mismatches, mismatches


def test_no_parity_row_compares_a_reference_with_itself():
    compare = _load_compare()
    assert (
        compare.implementation_census(list(compare.TOLERANCES))["reference_backend"]
        == 0
    )
    for stem, rec in _trace().items():
        assert rec["rscript_launches"] == 0, stem


def test_honest_did_modules_are_native():
    trace = _trace()
    for stem in ("10_honest_did", "21_honest_relmags"):
        assert _observed_kind(trace[stem]) == "native"


@pytest.mark.parametrize("stem", ["06_rd"])
def test_official_port_is_a_recorded_side_check_not_the_headline(stem):
    """The RD headline is native; the port appears only as unjoined rows."""
    compare = _load_compare()
    assert compare.implementation_kind(stem) == "native"
    rows = {d.statistic for d in compare.collect(stem)}
    assert not any(r.startswith("cct_port_") for r in rows)


def test_internal_delegation_change_behind_an_unchanged_script_is_stale(tmp_path):
    """The failure mode the trace exists for: same entry script, new internals.

    Rebuild module 01's traced files in a scratch tree, then edit one file on
    its estimation path (not the entry script). The trace must go stale.
    """
    tracer = _load_tracer()
    stem = "01_ols"
    rec = _trace()[stem]
    files = [f"tests/r_parity/{stem}.py", f"tests/r_parity/results/{stem}_py.json"]
    files += list(rec["exercised_sources"])
    for rel in files:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    assert tracer.stale_reasons(stem, rec, root=tmp_path) == []
    target = next(r for r in rec["exercised_sources"] if r.endswith("core/results.py"))
    with open(tmp_path / target, "a", encoding="utf-8") as fh:
        fh.write("\n# delegation changed here\n")
    reasons = tracer.stale_reasons(stem, rec, root=tmp_path)
    assert reasons == [f"{target} changed"]
    assert (
        tracer._sha256(tmp_path / f"tests/r_parity/{stem}.py") == rec["source_sha256"]
    )


def test_trace_is_current_on_archive_transliterated_sources(tmp_path):
    """The JSS archive ships source files ASCII-transliterated.

    Hashing raw bytes made the freshness check fail on the extracted archive
    for every module whose estimation path touches a file with an em dash or
    a Greek letter, although no code had changed. Rebuild a module's traced
    files as the archive writes them and require the trace to stay current.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from ascii_source import ASCII_SOURCE_SUFFIXES, ascii_source_text

    tracer = _load_tracer()
    trace = _trace()

    def _is_ascii(rel):
        try:
            (ROOT / rel).read_bytes().decode("ascii")
            return True
        except UnicodeDecodeError:
            return False

    stem = next(
        (
            s
            for s, rec in sorted(trace.items())
            if any(not _is_ascii(r) for r in rec.get("exercised_sources", []))
        ),
        None,
    )
    if stem is None:
        pytest.skip("every traced source is already ASCII (e.g. inside the archive)")
    rec = trace[stem]
    files = [f"tests/r_parity/{stem}.py", f"tests/r_parity/results/{stem}_py.json"]
    files += list(rec["exercised_sources"])
    rewritten = 0
    for rel in files:
        src, dst = ROOT / rel, tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in ASCII_SOURCE_SUFFIXES and not _is_ascii(rel):
            text = ascii_source_text(src.read_text(encoding="utf-8"))
            dst.write_bytes(text.encode("ascii"))
            rewritten += 1
        else:
            shutil.copyfile(src, dst)
    assert rewritten > 0
    assert tracer.stale_reasons(stem, rec, root=tmp_path) == []


def test_unknown_package_is_unclassified_not_native():
    rec = {
        "rscript_launches": 0,
        "boundary_calls": [
            {"package": "somecausalpkg", "callee": "somecausalpkg.fit", "caller": "x"}
        ],
    }
    assert _observed_kind(rec) == "unclassified"
    rec["boundary_calls"][0]["package"] = "sklearn"
    assert _observed_kind(rec) == "native"


def test_trace_records_the_estimation_path_and_versions():
    for stem, rec in _trace().items():
        assert rec["exercised_sources"], stem
        assert all(p.startswith("src/statspai/") for p in rec["exercised_sources"])
        assert "numpy" in rec["dependency_versions"], stem
