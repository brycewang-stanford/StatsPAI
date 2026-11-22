"""Relational-integrity contract for the agent-native surface.

The agent-card coverage ratchet (``tests/test_agent_card_coverage.py``)
guards *how much* metadata exists.  This file guards that the metadata
that exists is **internally consistent** — every pointer an agent might
follow resolves to something real:

* Every ``alternatives`` entry and every ``failure_modes.alternative``
  points to a registered StatsPAI function (a dead ``sp.xxx`` pointer
  sends an agent into an ``AttributeError`` mid-recovery).
* Every ``failure_modes.exception`` names a real exception class an
  agent could ``except`` on (a builtin or a ``statspai.exceptions``
  member).
* Every MCP tool the server advertises resolves to an executable target
  (a registered function or a workflow/pipeline tool) — no dangling
  tool that 500s on dispatch.
* Every ``_FOLLOWUP_BY_TOOL`` "next call" the server hands an agent is
  itself an advertised tool (no copy-paste dead end).
* Every ``_CITATIONS_BY_TOOL`` bib key exists in ``paper.bib`` — the
  citation-hallucination red line (CLAUDE.md §10), enforced at the
  enrichment layer rather than trusted.
* Every advertised tool's ``input_schema.required`` is a subset of its
  ``properties``, and parameterless tools (empty ``properties``) stay
  on a known allowlist so a *new* empty-schema tool trips this test and
  forces a decision rather than silently shipping.

These tests run with no R / Stata / network — they read the live
registry and the static enrichment / manifest tables only.
"""

from __future__ import annotations

import builtins
import re
from pathlib import Path

import pytest

import statspai as sp  # noqa: F401  (import side-effect builds the registry)
from statspai import exceptions as sp_exceptions
from statspai.registry import _REGISTRY, _ensure_full_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_BIB = REPO_ROOT / "paper.bib"


# --------------------------------------------------------------------------- #
#  Shared resolution helpers
# --------------------------------------------------------------------------- #


def _leading_token(target: str) -> str:
    """Extract the function token from an alternatives entry.

    The curated convention is either a bare name (``"recommend"``), a
    ``sp.``-prefixed name (``"sp.dml"``), or a name followed by a prose
    gloss (``"sp.dml: double ML with manual text features"`` /
    ``"sp.regress with raw LLM label"``).  We take the leading
    identifier-or-dotted-path and ignore the gloss.
    """
    t = (target or "").strip()
    for prefix in ("statspai.", "sp."):
        if t.startswith(prefix):
            t = t[len(prefix) :]
            break
    # Cut at the first gloss separator: ``:``, whitespace, or ``(``.
    t = re.split(r"[:\s(]", t, maxsplit=1)[0]
    return t.strip()


def _mcp_tool_names() -> set[str]:
    from statspai.agent.pipeline_tools import PIPELINE_TOOL_NAMES
    from statspai.agent.tools import tool_manifest
    from statspai.agent.workflow_tools import WORKFLOW_TOOL_NAMES

    return (
        {t["name"] for t in tool_manifest()}
        | set(WORKFLOW_TOOL_NAMES)
        | set(PIPELINE_TOOL_NAMES)
    )


def _dotted_attr_exists(root, dotted: str) -> bool:
    """Walk ``sp.fast.feols`` style dotted paths against an object."""
    obj = root
    for part in dotted.split("."):
        if not part or not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


def _resolves_to_function(target: str, *, mcp_names: set[str] | None = None) -> bool:
    """True when ``target`` is a real recovery pointer an agent can follow.

    Valid targets: a registered function, a (possibly dotted) ``sp.*``
    attribute, or an advertised MCP tool name.  Parameter-tweak hints
    (an entry containing ``=``, e.g. ``"variant='x'"``) are accepted as
    informational — they are a legitimate "try this next" that is not a
    function pointer.  An empty string is accepted (no alternative).
    """
    raw = (target or "").strip()
    if not raw:
        return True
    if "=" in raw:  # parameter-tweak recovery hint, not a function pointer
        return True
    name = _leading_token(raw)
    if not name:
        return False
    if name in _REGISTRY:
        return True
    if _dotted_attr_exists(sp, name):
        return True
    names = mcp_names if mcp_names is not None else _mcp_tool_names()
    return name in names


def _valid_exception_names() -> set[str]:
    """Names an agent could legitimately ``except`` on.

    Builtins + ``statspai.exceptions`` members + any top-level
    ``statspai`` attribute that is itself an exception class (e.g.
    ``statspai.IdentificationError``, which lives in
    ``statspai.smart.identification`` but is re-exported at the top
    level).
    """
    builtin_excs = {
        n
        for n in dir(builtins)
        if isinstance(getattr(builtins, n), type)
        and issubclass(getattr(builtins, n), BaseException)
    }
    sp_excs = {n for n in dir(sp_exceptions) if not n.startswith("_")}
    top_level_excs = {
        n
        for n in dir(sp)
        if isinstance(getattr(sp, n, None), type)
        and issubclass(getattr(sp, n), BaseException)
    }
    return builtin_excs | sp_excs | top_level_excs


def _exception_is_valid(exc: str) -> bool:
    """A failure-mode exception is valid if it names a real class or is the
    documented "no exception raised" sentinel (``"(none ...)"`` / empty).
    """
    raw = (exc or "").strip()
    if not raw or raw.lower().startswith("(none"):
        return True
    return raw.split(".")[-1].strip() in _valid_exception_names()


@pytest.fixture(scope="module", autouse=True)
def _full_registry():
    _ensure_full_registry()


# --------------------------------------------------------------------------- #
#  1. Agent-card pointer integrity
# --------------------------------------------------------------------------- #


def test_every_alternative_resolves_to_a_registered_function():
    """``alternatives`` and ``failure_modes.alternative`` must not dangle."""
    mcp_names = _mcp_tool_names()
    broken: list[str] = []
    for name, spec in _REGISTRY.items():
        for alt in spec.alternatives:
            if not _resolves_to_function(alt, mcp_names=mcp_names):
                broken.append(f"{name}.alternatives -> {alt!r}")
        for fm in spec.failure_modes:
            alt = getattr(fm, "alternative", "")
            if not _resolves_to_function(alt, mcp_names=mcp_names):
                broken.append(f"{name}.failure_modes.alternative -> {alt!r}")
    assert not broken, (
        "Agent-card alternatives point to functions that do not exist. An "
        "agent following these recovery hints hits AttributeError:\n  "
        + "\n  ".join(sorted(broken))
    )


def test_every_failure_mode_exception_is_a_real_class():
    """``failure_modes.exception`` must name something an agent can except on."""
    broken: list[str] = []
    for name, spec in _REGISTRY.items():
        for fm in spec.failure_modes:
            if not _exception_is_valid(getattr(fm, "exception", "")):
                broken.append(f"{name}: {fm.exception!r}")
    assert not broken, (
        "failure_modes.exception names a class that is neither a builtin, a "
        "statspai exception, nor the '(none ...)' sentinel:\n  "
        + "\n  ".join(sorted(broken))
    )


def test_inherited_alternatives_also_resolve():
    """The merged (post-inheritance) view an agent sees must also be clean."""
    mcp_names = _mcp_tool_names()
    broken: list[str] = []
    for name, spec in _REGISTRY.items():
        card = spec.agent_card(merge_inherited=True)
        for alt in card.get("alternatives", []):
            if not _resolves_to_function(alt, mcp_names=mcp_names):
                broken.append(f"{name} (merged) -> {alt!r}")
    assert not broken, "Merged agent-card alternatives dangle:\n  " + "\n  ".join(
        sorted(broken)
    )


# --------------------------------------------------------------------------- #
#  2. MCP manifest <-> registry / executor consistency
# --------------------------------------------------------------------------- #


def _manifest():
    from statspai.agent.tools import tool_manifest

    return tool_manifest()


def _executable_tool_name(name: str) -> bool:
    """Mirror the four dispatch paths in ``execute_tool``.

    A tool is executable iff it is a workflow tool, a pipeline tool, a
    curated ``TOOL_REGISTRY`` entry (these carry bespoke handlers /
    routers like ``sensitivity``), or resolves through the registry-
    driven auto dispatch (``_resolve_fn``).
    """
    from statspai.agent.pipeline_tools import PIPELINE_TOOL_NAMES
    from statspai.agent.tools import TOOL_REGISTRY
    from statspai.agent.tools._dispatch import _resolve_fn
    from statspai.agent.workflow_tools import WORKFLOW_TOOL_NAMES

    if name in WORKFLOW_TOOL_NAMES or name in PIPELINE_TOOL_NAMES:
        return True
    if any(t["name"] == name for t in TOOL_REGISTRY):
        return True
    try:
        return _resolve_fn(name) is not None
    except Exception:
        return False


def test_every_advertised_tool_is_executable():
    """No tool may be advertised that the executor cannot dispatch."""
    dangling = [t["name"] for t in _manifest() if not _executable_tool_name(t["name"])]
    assert not dangling, (
        "MCP advertises tools the executor cannot resolve (calling them "
        f"would 500):\n  {sorted(dangling)}"
    )


def test_tool_required_is_subset_of_properties():
    """A required parameter not present in properties is an invalid schema."""
    bad: list[str] = []
    for t in _manifest():
        schema = t.get("input_schema") or {}
        props = set((schema.get("properties") or {}).keys())
        required = set(schema.get("required") or [])
        if not required.issubset(props):
            bad.append(f"{t['name']}: required-not-in-props={sorted(required - props)}")
    assert (
        not bad
    ), "Tool schemas with required params absent from properties:\n  " + "\n  ".join(
        bad
    )


#: Tools that legitimately expose no estimator parameters: dataset
#: loaders (data is the return value) and zero-arg catalog/utility
#: tools.  A *new* empty-schema tool not on this list trips the test —
#: forcing a deliberate choice (add params, or add it here with reason).
_KNOWN_PARAMETERLESS_TOOLS = {
    # Built-in dataset loaders — return a DataFrame, take no estimator args.
    "basque_terrorism",
    "california_prop99",
    "california_tobacco",
    "german_reunification",
    "boundary_rd",
    "geographic_rd",
    "multi_cutoff_rd",
    "multi_score_rd",
    # Catalog / utility tools.
    "available_methods",
    # The nine steps of the DiD protocol as a table; it reads no data.
    "did_protocol",
    "list_replications",
    "etable",
    # Inference helpers whose only inputs are result handles / arrays
    # injected by the server envelope rather than the estimator schema.
    "anderson_rubin_ci",
    "conditional_lr_ci",
}


def test_empty_schema_tools_stay_on_the_known_allowlist():
    """New parameterless tools must be triaged, not silently shipped."""
    empty = {
        t["name"]
        for t in _manifest()
        if not (t.get("input_schema") or {}).get("properties")
    }
    surprise = empty - _KNOWN_PARAMETERLESS_TOOLS
    assert not surprise, (
        "New tool(s) advertise an empty input_schema. Either give them "
        "parameters or add them to _KNOWN_PARAMETERLESS_TOOLS with a "
        f"reason:\n  {sorted(surprise)}"
    )


# --------------------------------------------------------------------------- #
#  3. Enrichment-table closure (next_calls / citations)
# --------------------------------------------------------------------------- #


def test_followup_next_calls_are_all_advertised_tools():
    """Every 'next call' the server suggests must be a real tool."""
    from statspai.agent._enrichment import _FOLLOWUP_BY_TOOL

    tool_names = {t["name"] for t in _manifest()}
    broken: list[str] = []
    for tool, follows in _FOLLOWUP_BY_TOOL.items():
        for f in follows:
            target = f.get("tool")
            if target not in tool_names:
                broken.append(f"{tool} -> {target!r}")
    assert not broken, (
        "_FOLLOWUP_BY_TOOL points at tools the server does not advertise "
        "(agent copy-pastes a dead next-call):\n  " + "\n  ".join(sorted(broken))
    )


def test_enrichment_citation_keys_exist_in_paper_bib():
    """Every bib key the enrichment layer cites must be in paper.bib.

    Enforces the CLAUDE.md §10 single-source-of-truth rule at the
    enrichment layer: a citation an agent is handed must resolve to a
    verified BibTeX body, never a synthesised stub.
    """
    from statspai.agent._enrichment import _CITATIONS_BY_TOOL

    bib_text = PAPER_BIB.read_text(encoding="utf-8")
    keys_in_bib = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    broken: list[str] = []
    for tool, keys in _CITATIONS_BY_TOOL.items():
        for key in keys:
            if key not in keys_in_bib:
                broken.append(f"{tool} -> {key!r}")
    assert not broken, (
        "Enrichment cites bib keys absent from paper.bib (verify and add "
        "them per CLAUDE.md §10, never invent):\n  " + "\n  ".join(sorted(broken))
    )
