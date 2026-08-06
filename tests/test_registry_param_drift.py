"""Hand-written registry entries must not hide parameters from agents.

``sp.function_schema()`` is what an LLM planner reads to decide how to
call StatsPAI. It is built from ``registry._REGISTRY[name].params``, and
that list is **hand-written** for the richer entries — the auto-builder
explicitly refuses to overwrite them ("Never overwrite a hand-written
entry — those carry richer metadata").

The consequence is silent: add a parameter to a function's signature and
its agent-facing schema simply does not gain it. Nothing errors, nothing
warns, and the capability becomes invisible to every agent caller while
looking perfectly healthy from Python. This was found when five new
``callaway_santanna`` options failed to appear in ``schemas/*.json``
while the auto-derived aliases ``bjs`` / ``borusyak_jaravel_spiess``
picked theirs up straight away.

A sweep at that moment found **124 other entries already drifting, hiding
501 parameters between them** — including ``regress`` (``vce``,
``weights``, the Conley options), ``ivreg``, ``rdrobust`` and ``did``.
Fixing all of those blind is not safe in one pass: some omissions may be
deliberate (deprecated or internal-only arguments), and each needs a
description written by someone who knows the argument.

So this is a **ratchet**, matching the pattern already used by
``scripts/signature_house_style_baseline.json`` and friends:

* the existing 124 are frozen in
  ``scripts/registry_param_drift_baseline.json``;
* a baseline entry may **shrink** or disappear — that is progress;
* a baseline entry may **not grow**;
* a function **not** in the baseline may not start drifting at all.

Regenerate after deliberately fixing entries::

    pytest tests/test_registry_param_drift.py --update-drift-baseline
"""

from __future__ import annotations

import inspect
import json
import pathlib

import pytest

import statspai as sp
from statspai import registry as R

_BASELINE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "registry_param_drift_baseline.json"
)


def _live_params(fn) -> set:
    """Named parameters a caller can actually pass."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return set()
    return {
        name
        for name, p in sig.parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        and not name.startswith("_")
    }


def _current_drift() -> dict:
    R._ensure_full_registry()
    out = {}
    for name, spec in sorted(R._REGISTRY.items()):
        fn = getattr(sp, name, None)
        if fn is None or not callable(fn):
            continue
        live = _live_params(fn)
        if not live:
            continue
        missing = sorted(live - {p.name for p in spec.params})
        if missing:
            out[name] = missing
    return out


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(_BASELINE.read_text(encoding="utf-8"))["entries"]


@pytest.fixture(scope="module")
def drift() -> dict:
    return _current_drift()


def test_no_function_starts_hiding_parameters(drift, baseline):
    """The load-bearing check: no NEW entry may drift."""
    fresh = sorted(set(drift) - set(baseline))
    assert not fresh, (
        "these registry entries now hide parameters from "
        "sp.function_schema(), so agents cannot see them:\n"
        + "\n".join(f"  {n}: {drift[n]}" for n in fresh)
        + "\n\nAdd a ParamSpec(...) for each in registry.py — with a real "
        "description, since that description IS the agent-facing "
        "documentation."
    )


def test_baseline_entries_do_not_grow(drift, baseline):
    """A drifting entry may be fixed, but must not get worse."""
    worse = {}
    for name, allowed in baseline.items():
        now = drift.get(name, [])
        added = sorted(set(now) - set(allowed))
        if added:
            worse[name] = added
    assert (
        not worse
    ), "these entries hide MORE parameters than the baseline allows:\n" + "\n".join(
        f"  {n}: newly hidden {v}" for n, v in worse.items()
    )


def test_baseline_has_no_stale_entries(drift, baseline):
    """Entries fixed since the freeze must be removed from the baseline.

    Without this the ratchet quietly loosens: a name left in the baseline
    could start drifting again and nothing would complain.
    """
    fixed = sorted(n for n in baseline if n not in drift)
    assert not fixed, (
        "these entries no longer drift and must be dropped from "
        f"{_BASELINE.name} so the ratchet stays tight:\n  " + "\n  ".join(fixed)
    )


class TestDidOptionDepthSurface:
    """The estimators touched by the DiD option-depth campaign expose
    every option they accept."""

    @pytest.mark.parametrize(
        "name",
        [
            "callaway_santanna",
            "sun_abraham",
            "did_imputation",
            "did_multiplegt_dyn",
        ],
    )
    def test_no_hidden_parameters(self, name, drift):
        assert (
            name not in drift
        ), f"{name} hides {drift.get(name)} from its agent schema"

    @pytest.mark.parametrize(
        "name,param",
        [
            ("callaway_santanna", "notyet_cutoff"),
            ("callaway_santanna", "pscore_trim"),
            ("callaway_santanna", "pretest"),
            ("callaway_santanna", "se_method"),
            ("sun_abraham", "control_cohort"),
            ("sun_abraham", "pretest"),
            ("did_imputation", "unit_covariates"),
            ("did_imputation", "time_covariates"),
            ("did_imputation", "fe"),
            ("did_imputation", "project"),
            ("did_multiplegt_dyn", "switchers"),
            ("did_multiplegt_dyn", "same_switchers"),
            ("did_multiplegt_dyn", "effects_equal"),
        ],
    )
    def test_option_reaches_the_agent_schema(self, name, param):
        props = sp.function_schema(name)["parameters"]["properties"]
        assert param in props, f"{name}.{param} missing from function_schema"
        desc = props[param].get("description", "")
        assert len(desc) > 30, (
            f"{name}.{param} has a stub description {desc!r}; the schema "
            "description is what an agent reads instead of the docstring"
        )
        assert not desc.startswith(f"{param} parameter"), (
            f"{name}.{param} fell back to the auto-generated description, "
            "which means no ParamSpec was written for it"
        )
