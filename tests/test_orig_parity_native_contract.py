"""The original-data ledger states which implementation each row ran.

``tests/orig_parity/`` compares StatsPAI with the canonical R reference on
the *original* public extracts, and the manuscript prints a selection of
those rows (Table 7). Until 1.32.0 the tracing audit that classifies Track A
rows did not cover this ledger, and its RD row ran
``sp.rdrobust(bwselect="cct")`` -- the official rdrobust Python port -- so
the printed agreement was the authors' code compared with itself. This test
holds the ledger to the same standard as Track A: every module is traced,
the trace is current, no module calls an official port or the R reference,
and any other package use is registered in
``compare_orig.ORIG_IMPLEMENTATION_PROVENANCE``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "tests" / "orig_parity"
TRACE = ORIG / "results" / "_implementation_trace.json"

OFFICIAL_PORTS = {"rdrobust", "rddensity", "rdd", "rdmulti", "rdlocrand"}
#: statsmodels calls that fit a GLM/OLS/Logit nuisance model or are utilities.
NUISANCE_CALLEES = (
    "statsmodels.genmod.",
    "statsmodels.regression.linear_model.",
    "statsmodels.discrete.discrete_model.",
    "statsmodels.base.",
    "statsmodels.tools.",
)
#: Delegation switches that hand the computation to a port or a reference.
DELEGATION = re.compile(r"""bwselect\s*=\s*["']cct["']|backend\s*=""")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trace() -> dict:
    return json.loads(TRACE.read_text(encoding="utf-8"))["modules"]


def _modules() -> list[str]:
    return sorted(p.stem for p in ORIG.glob("[0-9][0-9]*_*.py"))


def _observed_kind(rec: dict) -> str:
    if rec["rscript_launches"] > 0:
        return "reference_backend"
    pkgs = {c["package"] for c in rec["boundary_calls"]}
    if pkgs & OFFICIAL_PORTS:
        return "official_python_port"
    if not pkgs:
        return "native"
    if pkgs == {"statsmodels"} and all(
        c["callee"].startswith(NUISANCE_CALLEES) for c in rec["boundary_calls"]
    ):
        return "statsmodels_nuisance"
    return "unclassified"


def test_no_module_source_delegates_to_a_port_or_reference():
    offenders = {
        stem: m.group(0)
        for stem in _modules()
        if (
            m := DELEGATION.search(
                # Docstrings may name the switch to explain why it is absent.
                re.sub(
                    r'"""[\s\S]*?"""',
                    "",
                    (ORIG / f"{stem}.py").read_text(encoding="utf-8"),
                )
            )
        )
    }
    assert not offenders, offenders


def test_every_module_is_traced_and_the_trace_is_current():
    tracer = _load("_trace_prov_orig", ROOT / "scripts" / "trace_parity_provenance.py")
    trace = _trace()
    assert set(trace) == set(_modules())
    stale = {
        stem: reasons
        for stem, rec in trace.items()
        if (reasons := tracer.stale_reasons(stem, rec, ledger="orig"))
    }
    assert not stale, (
        "original-data traces are stale; re-run "
        "`python scripts/trace_parity_provenance.py --ledger orig`: " + str(stale)
    )
    assert not {s: r["error"] for s, r in trace.items() if r.get("error")}


def test_registered_provenance_matches_the_trace():
    compare = _load("_compare_orig", ORIG / "compare_orig.py")
    mismatches = {
        stem: (compare.implementation_kind(stem), _observed_kind(rec))
        for stem, rec in _trace().items()
        if compare.implementation_kind(stem) != _observed_kind(rec)
    }
    assert not mismatches, mismatches


@pytest.mark.parametrize("kind", ["reference_backend", "official_python_port"])
def test_no_row_compares_a_reference_with_itself(kind):
    assert not [s for s, r in _trace().items() if _observed_kind(r) == kind]


def test_rd_row_runs_the_native_selector():
    rec = _trace()["05_lee_original"]
    assert _observed_kind(rec) == "native"
    assert any(p.endswith("rd/rdrobust.py") for p in rec["exercised_sources"])
    py = json.loads(
        (ORIG / "results" / "05_lee_original_py.json").read_text(encoding="utf-8")
    )
    assert py["extra"]["bwselect"] != "cct"


def test_a_port_call_would_be_caught():
    rec = {
        "rscript_launches": 0,
        "boundary_calls": [
            {
                "package": "rdrobust",
                "callee": "rdrobust.rdrobust.rdrobust",
                "caller": "rd/rdrobust.py",
            }
        ],
    }
    assert _observed_kind(rec) == "official_python_port"
    rec["boundary_calls"][0] = {
        "package": "statsmodels",
        "callee": "statsmodels.tsa.arima.model.ARIMA.fit",
        "caller": "x",
    }
    assert _observed_kind(rec) == "unclassified"
