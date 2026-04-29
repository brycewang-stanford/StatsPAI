"""Auto-generate MCP / LLM tool manifest from the StatsPAI registry.

The hand-curated :data:`statspai.agent.tools.TOOL_REGISTRY` covers ~8
flagship estimators in depth (tuned schemas, bespoke serializers, top-
quality descriptions).  With 880+ registered functions this leaves most
of the package invisible to agents — so this module walks the registry
and synthesises MCP tool specs for every agent-safe estimator it finds.

Design
------

* **Registry first.** ``sp.function_schema(name)`` already emits OpenAI-
  compatible schemas for every registered function.  We start from those
  and turn them into MCP ``{name, description, inputSchema}`` tuples.
* **Category whitelist.**  We only expose categories that make sense as
  agent tools — causal, regression, panel, inference, diagnostics, …
  Plots, CLI helpers, datasets, and pure utilities are skipped so the
  manifest stays signal-dense.
* **Agent-card enrichment.**  If :func:`sp.agent_card` returns rich
  metadata (assumptions, failure modes, alternatives), we append a
  compact block to the tool description so agents see *when to use* and
  *what to try when it breaks* right in the tool-list payload.
* **Hand-curated wins on collision.**  The merged manifest lets every
  hand-curated spec override its auto-generated counterpart.
* **Deny list.**  Estimators that need optional heavy deps (torch /
  jax / pymc) and are known to fail eagerly on ``getattr(sp, name)`` are
  excluded by name.

The public entry points are :func:`auto_tool_manifest` and
:func:`merged_tool_manifest`.  ``tool_manifest()`` in ``tools.py``
delegates to the latter.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set


# Categories we want reachable by an agent by default.  See
# ``statspai.help.CATEGORY_DESCRIPTIONS`` for the full list.
DEFAULT_WHITELIST: Set[str] = {
    'causal',
    'regression',
    'panel',
    'inference',
    'diagnostics',
    'smart',
    'decomposition',
    'robustness',
    'postestimation',
    'survival',
    'bayesian',
    'timeseries',
}

# Names we explicitly keep out of the auto manifest.  Reasons vary —
# optional deep-learning dependencies, plotting helpers that slipped a
# non-plot category, obvious internals, etc.  Keep this list short; for
# broader exclusions, adjust the category whitelist instead.
DEFAULT_EXCLUDE: Set[str] = {
    # Deep-learning — heavy deps (torch / jax)
    'dragonnet',
    'deepiv',
    'tarnet',
    'cfrnet',
    # Neural-causal estimators
    'neural_causal_forest',
    'cevae',
    'ganite',
    # Plots that happen to land in a stats category
    'event_study_plot',
    'coefplot',
    'binscatter',
    # Misc utilities that slipped into exposed API
    'describe',
    'pwcorr',
    'winsor',
    'read_data',
}

# Max length for a tool description sent to the agent.  MCP / Anthropic
# tool-use enforces no hard limit, but large descriptions eat context
# for every tool call the agent does, so we cap to keep the manifest
# lean.
_MAX_DESCRIPTION_LEN = 1200


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _is_agent_safe(name: str, spec: Any) -> bool:
    """Return True iff ``name``/``spec`` should appear in auto manifest.

    Rejects:
    * leading underscore (private)
    * deny-listed names
    * anything flagged by spec metadata (tags contains 'internal')
    * classes (PascalCase names that resolve to a type, not a function);
      agents call functions, not constructors directly
    """
    if not name or name.startswith('_'):
        return False
    if name in DEFAULT_EXCLUDE:
        return False
    tags = getattr(spec, 'tags', []) or []
    if any(str(t).lower() == 'internal' for t in tags):
        return False
    # Heuristic: PascalCase → class, which shouldn't be a tool.  Verify
    # against statspai to avoid false positives on SCREAMING names.
    if name[:1].isupper():
        import inspect
        import statspai as _sp
        obj = getattr(_sp, name, None)
        if obj is not None and inspect.isclass(obj):
            return False
    return True


def _enrich_description(base_desc: str, card: Optional[Dict[str, Any]]) -> str:
    """Append compact agent-card blurbs to a tool description.

    The result is kept under :data:`_MAX_DESCRIPTION_LEN` characters.
    """
    desc = (base_desc or '').strip()
    if not card:
        return desc[:_MAX_DESCRIPTION_LEN]

    chunks: List[str] = [desc] if desc else []

    assumptions = card.get('assumptions') or []
    if assumptions:
        bullets = '; '.join(str(a).rstrip('.') for a in assumptions[:3])
        chunks.append(f"Assumptions: {bullets}.")

    pre = card.get('pre_conditions') or []
    if pre:
        bullets = '; '.join(str(p).rstrip('.') for p in pre[:3])
        chunks.append(f"Pre-conditions: {bullets}.")

    fm = card.get('failure_modes') or []
    if fm:
        pieces = []
        for f in fm[:3]:
            sym = str(f.get('symptom') or '').strip()
            rem = str(f.get('remedy') or '').strip()
            if sym and rem:
                pieces.append(f"{sym} → {rem}")
        if pieces:
            chunks.append(f"Failure modes: {'; '.join(pieces)}.")

    alts = card.get('alternatives') or []
    if alts:
        chunks.append(
            f"Alternatives: {', '.join(f'sp.{a}' for a in alts[:4])}."
        )

    nmin = card.get('typical_n_min')
    if nmin:
        chunks.append(f"Typical minimum N: {nmin}.")

    merged = ' '.join(chunks).strip()
    if len(merged) > _MAX_DESCRIPTION_LEN:
        merged = merged[:_MAX_DESCRIPTION_LEN - 1].rstrip() + '…'
    return merged


def _json_safe_value(v: Any) -> Any:
    """Coerce a schema default to a JSON-safe primitive.

    Some registry specs populate ``p.default`` with sentinels from
    ``dataclasses.field(default_factory=...)``.  Those leak into
    ``to_openai_schema()`` output and then break ``json.dumps``.  We
    drop anything that can't round-trip as a JSON scalar or list.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        clean = [_json_safe_value(x) for x in v]
        return [x for x in clean if x is not None
                or isinstance(x, (bool, int, float, str))]
    # Anything else — dataclass sentinels, callables, module objects —
    # is unsafe; drop the default entirely.
    return None


def _clean_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    """Strip unsafe default values from a properties map in-place safe form."""
    out: Dict[str, Any] = {}
    for k, v in (props or {}).items():
        if not isinstance(v, dict):
            continue
        prop = dict(v)
        if 'default' in prop:
            cleaned = _json_safe_value(prop['default'])
            if cleaned is None and not isinstance(prop['default'], bool):
                prop.pop('default', None)
            else:
                prop['default'] = cleaned
        out[str(k)] = prop
    return out


def _schema_to_mcp_tool(schema: Dict[str, Any],
                        card: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert an OpenAI-style schema dict to MCP tool-list form.

    The OpenAI schema has ``{'name', 'description', 'parameters'}``;
    MCP's ``tools/list`` expects ``{'name', 'description', 'inputSchema'}``.
    We also strip the ``data`` pseudo-param (MCP clients pass a
    ``data_path`` instead — added by the outer MCP wrapper) and enrich
    the description with agent-card content.
    """
    name = schema.get('name', '')
    raw_desc = schema.get('description', '')
    params = dict(schema.get('parameters') or {})
    props = _clean_properties(params.get('properties') or {})
    required = list(params.get('required') or [])
    # `data` is injected by the MCP server via data_path/CSV loading;
    # remove from the exposed tool so the agent doesn't try to pass it.
    props.pop('data', None)
    required = [r for r in required if r != 'data' and r in props]

    input_schema: Dict[str, Any] = {
        'type': params.get('type', 'object'),
        'properties': props,
        'required': sorted(set(required)),
    }

    return {
        'name': name,
        'description': _enrich_description(raw_desc, card),
        'input_schema': input_schema,
    }


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def auto_tool_manifest(
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    *,
    max_tools: int = 500,
    warn_on_truncate: bool = True,
) -> List[Dict[str, Any]]:
    """Return MCP tool specs for every agent-safe registered function.

    Parameters
    ----------
    categories : iterable of str, optional
        Override the default category whitelist.  Default whitelist is
        :data:`DEFAULT_WHITELIST`.
    exclude : iterable of str, optional
        Extra names to skip in addition to :data:`DEFAULT_EXCLUDE`.
    max_tools : int, default 500
        Cap on output size.  Keeps the manifest under Anthropic/OpenAI
        tool-list payload limits. A warning fires when more eligible
        tools exist than the cap admits — see ``warn_on_truncate``.
    warn_on_truncate : bool, default True
        Emit a ``RuntimeWarning`` when the eligible-tool count exceeds
        ``max_tools``. Set False in code paths that intentionally crop
        the manifest.

    Returns
    -------
    list of dict
        Each entry ``{'name', 'description', 'input_schema'}`` — the same
        shape as :func:`statspai.agent.tools.tool_manifest`.
    """
    from ..registry import _REGISTRY, _ensure_full_registry

    _ensure_full_registry()

    wl: Set[str] = (set(categories) if categories is not None
                    else set(DEFAULT_WHITELIST))
    ex: Set[str] = set(DEFAULT_EXCLUDE)
    if exclude:
        ex |= set(exclude)

    out: List[Dict[str, Any]] = []
    eligible_total = 0
    for name, spec in _REGISTRY.items():
        if name in ex:
            continue
        if spec.category not in wl:
            continue
        if not _is_agent_safe(name, spec):
            continue
        eligible_total += 1
        if len(out) >= max_tools:
            continue  # keep counting so the warning shows the gap
        try:
            schema = spec.to_openai_schema()
        except Exception:
            continue
        try:
            card = spec.agent_card()
        except Exception:
            card = None
        try:
            tool = _schema_to_mcp_tool(schema, card)
        except Exception:
            continue
        if not tool.get('name'):
            continue
        out.append(tool)

    if warn_on_truncate and eligible_total > len(out):
        # Loud failure per CLAUDE.md §3 #7: silently truncating the
        # auto manifest is exactly the kind of degradation that hides
        # real problems (registry growth past the cap, mis-categorised
        # entries flooding the count). Operators want to know.
        import warnings
        warnings.warn(
            f"auto_tool_manifest truncated to {len(out)} tools out of "
            f"{eligible_total} eligible (max_tools={max_tools}). "
            f"Raise the cap or narrow the category whitelist.",
            RuntimeWarning, stacklevel=2,
        )

    # Stable ordering so downstream diffs are readable.
    out.sort(key=lambda t: t['name'])
    return out


def merged_tool_manifest(
    hand_curated: List[Dict[str, Any]],
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Merge hand-curated tools with the auto-generated manifest.

    Hand-curated entries win on name collision (they carry richer
    descriptions / bespoke serializers).  Auto entries fill the rest.

    The returned list is deduped and sorted by name.
    """
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []

    for t in hand_curated or []:
        name = t.get('name')
        if not name or name in seen:
            continue
        out.append({
            'name': name,
            'description': t.get('description', ''),
            'input_schema': t.get('input_schema') or {},
        })
        seen.add(name)

    for t in auto_tool_manifest(categories=categories, exclude=exclude):
        if t['name'] in seen:
            continue
        out.append(t)
        seen.add(t['name'])

    out.sort(key=lambda t: t['name'])
    return out


__all__ = [
    'auto_tool_manifest',
    'merged_tool_manifest',
    'DEFAULT_WHITELIST',
    'DEFAULT_EXCLUDE',
]
