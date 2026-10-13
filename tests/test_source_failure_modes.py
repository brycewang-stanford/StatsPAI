"""Failure modes quoted from a function's own ``raise`` statements.

Functions with no hand-written or inherited failure mode get the typed errors
their own body raises, each with the ``recovery_hint`` written next to it
(``scripts/gen_source_failure_modes.py``). The table is generated, so the
tests are about the contract around it: it is current, it never displaces a
curated card, and what it says can be made to happen.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

import statspai as sp
from statspai import registry
from statspai._source_failure_modes import SOURCE_FAILURE_MODES

ROOT = Path(__file__).resolve().parent.parent


def _generator():
    path = ROOT / "scripts" / "gen_source_failure_modes.py"
    spec = importlib.util.spec_from_file_location("gen_source_failure_modes", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_table_is_current():
    gen = _generator()
    assert (
        gen.collect() == SOURCE_FAILURE_MODES
    ), "run python scripts/gen_source_failure_modes.py"


def test_every_entry_is_used_and_tagged_source():
    sp.list_functions()
    assert SOURCE_FAILURE_MODES
    for name, modes in SOURCE_FAILURE_MODES.items():
        card = sp.agent_card(name)
        assert card["provenance"]["failure_modes"] == "source", name
        assert [m["symptom"] for m in card["failure_modes"]] == [
            m["symptom"] for m in modes
        ], name
        for mode in modes:
            assert mode["exception"].startswith("statspai."), name
            assert mode["remedy"].strip() and mode["symptom"].strip(), name


def test_a_curated_card_is_never_displaced():
    # ``did`` raises typed errors with hints too; its card is hand-written.
    assert "did" not in registry._SOURCE_FAILURE_MODES_FILLED
    assert sp.agent_card("did")["provenance"]["failure_modes"] != "source"


def test_exception_names_resolve_to_the_taxonomy():
    from statspai import exceptions

    for modes in SOURCE_FAILURE_MODES.values():
        for mode in modes:
            cls = getattr(exceptions, mode["exception"].split(".", 1)[1])
            assert issubclass(cls, exceptions.StatsPAIError)


def test_a_quoted_failure_mode_really_happens():
    """The card's symptom and remedy are what the call raises."""
    mode = next(
        m
        for m in SOURCE_FAILURE_MODES["pscore"]
        if "must be a pandas DataFrame" in m["symptom"]
    )
    with pytest.raises(sp.MethodIncompatibility) as err:
        sp.pscore([1, 2, 3], treat="d", covariates=["x"])
    assert mode["symptom"] in str(err.value)
    assert err.value.recovery_hint == mode["remedy"]

    trend = next(
        m for m in SOURCE_FAILURE_MODES["ardl"] if "unknown trend" in m["symptom"]
    )
    frame = pd.DataFrame({"y": range(30), "x": range(30)})
    with pytest.raises(sp.MethodIncompatibility) as err:
        sp.ardl(frame, y="y", x=["x"], trend="quadratic")
    assert trend["symptom"].split("<")[0] in str(err.value)
    assert err.value.recovery_hint == trend["remedy"]
