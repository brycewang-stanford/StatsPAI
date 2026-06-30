"""Gate test: the declared failure-mode contract must stay consistent.

StatsPAI's registry declares, per estimator, a machine-readable failure-mode
contract — ``FailureMode(symptom, exception, remedy, alternative)`` and a ranked
``alternatives`` list — that agents and humans rely on for recovery. This test
runs :mod:`scripts.failure_mode_audit` and fails the build if that contract
drifts out of sync with reality:

* an ``alternative`` points at an ``sp.<name>`` that is not registered
  (a dangling pointer sends the caller to a function that does not exist);
* a ``FailureMode.exception`` names a class that cannot be imported
  (an agent's ``except`` clause would never fire).

Soft findings (module-only "imprecise" alternatives, taxonomy gaps) are surfaced
by the script but intentionally do **not** fail the build here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_AUDIT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "failure_mode_audit.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("_fm_audit", _AUDIT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the @dataclass in the module can resolve its
    # __module__ (Python 3.13 dataclass creation reads sys.modules).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def report():
    audit = _load_audit_module()
    registry = audit._load_registry()
    rep = audit.audit_registry(registry)
    return rep


def test_no_dangling_alternatives(report):
    """Every ``sp.<name>`` an estimator suggests on failure must resolve."""
    assert (
        report.dangling_alternatives == []
    ), "FailureMode/alternatives point at unregistered functions: " + ", ".join(
        f"{d['function']}→sp.{d['target']}" for d in report.dangling_alternatives
    )


def test_no_unknown_exception_names(report):
    """Every declared failure exception must name an importable class."""
    assert (
        report.unknown_exceptions == []
    ), "FailureMode.exception names unknown classes: " + ", ".join(
        f"{d['function']}:{d['exception']}" for d in report.unknown_exceptions
    )


def test_hard_error_count_is_zero(report):
    assert report.hard_error_count == 0
