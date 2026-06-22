"""CI gate for the StatsPAI full-data-analysis skill's API claims.

The skill at ``StatsPAI_full_data_analysis_skill/`` ships a runnable validation
gate, ``validate_api_claims.py`` (SkillOpt-style: a claim that no longer holds
turns the gate red). This test wires that gate into the pytest suite so that a
library change which silently breaks a documented API claim in ``SKILL.md`` —
e.g. a renamed argument or a dropped result-object attribute — fails CI instead
of only surfacing when an agent runs the skill.

Two layers:

* ``test_skill_api_claims_quick`` — the ``--quick`` gate (every ``sp.*``
  reference resolves, module members exist, documented argument names exist).
  Needs only base ``statspai`` (symbols resolve without importing torch / jax /
  pyfixest), so it runs in the default fast suite and the pre-commit hook.
* ``test_skill_api_claims_attributes`` — result-object attribute and plot
  return-shape claims via smoke fits. Marked ``slow`` (excluded from the default
  run) and guarded on ``matplotlib`` since it draws figures.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GATE_PATH = _REPO_ROOT / "StatsPAI_full_data_analysis_skill" / "validate_api_claims.py"


def _load_gate():
    """Load the skill's standalone validation gate as a module."""
    if not _GATE_PATH.exists():
        pytest.skip(f"skill validation gate not found at {_GATE_PATH}")
    spec = importlib.util.spec_from_file_location("statspai_skill_gate", _GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_api_claims_quick():
    """Existence + module members + documented argument names hold (base statspai)."""
    gate = _load_gate()
    failures: list[str] = []
    gate.check_references(failures)
    gate.check_modules(failures)
    gate.check_signatures(failures)
    assert not failures, (
        "SKILL.md API-claim drift — update SKILL.md and validate_api_claims.py "
        "together:\n  - " + "\n  - ".join(failures)
    )


@pytest.mark.slow
def test_skill_api_claims_attributes():
    """Result-object attribute / return-shape claims hold."""
    pytest.importorskip("matplotlib")
    gate = _load_gate()
    failures: list[str] = []
    gate.check_attributes(failures)
    assert not failures, (
        "SKILL.md attribute-claim drift — update SKILL.md and validate_api_claims.py "
        "together:\n  - " + "\n  - ".join(failures)
    )
