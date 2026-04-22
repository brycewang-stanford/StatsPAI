"""Render ``## For Agents`` markdown blocks from registry agent cards.

The block layout is:

    ## For Agents

    **Pre-conditions**
    - item
    - item

    **Identifying assumptions**
    - item

    **Failure modes → recovery**
    | Symptom | Exception | Remedy | Try next |
    | --- | --- | --- | --- |
    | ... | ... | ... | ... |

    **Alternatives (ranked)**
    - `sp.alt_1`
    - `sp.alt_2`

    **Typical minimum N**: 50

Agents parse this; humans read it in MkDocs. A single agent card is
the source of truth so the two never drift apart.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


_HEADER = "## For Agents"


def _bullet(items: Iterable[str]) -> List[str]:
    return [f"- {item}" for item in items]


def _render_failure_table(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    lines = [
        "| Symptom | Exception | Remedy | Try next |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        sym = (r.get("symptom") or "").replace("|", "\\|")
        exc = f"`{r.get('exception', '')}`" if r.get("exception") else ""
        rem = (r.get("remedy") or "").replace("|", "\\|")
        alt = r.get("alternative") or ""
        alt_cell = f"`{alt}`" if alt else ""
        lines.append(f"| {sym} | {exc} | {rem} | {alt_cell} |")
    return lines


def render_agent_block(name: str, *, header: bool = True) -> str:
    """Render the ``## For Agents`` markdown block for one function.

    Parameters
    ----------
    name : str
        Registered function name (``"did"``, ``"iv"``, ``"rd"``, …).
    header : bool, default True
        If False, omit the ``## For Agents`` header (useful when
        embedding inside another section).

    Returns
    -------
    str
        Markdown. Empty string if the function has no agent-native
        metadata populated — callers should check ``bool(block)`` to
        decide whether to render at all.

    Raises
    ------
    KeyError
        If ``name`` is not a registered function.
    """
    from .registry import agent_card as _card

    card = _card(name)

    sections: List[str] = []
    if header:
        sections.append(_HEADER)
        sections.append("")

    pre = card.get("pre_conditions") or []
    assumptions = card.get("assumptions") or []
    failures = card.get("failure_modes") or []
    alternatives = card.get("alternatives") or []
    n_min = card.get("typical_n_min")

    if not (pre or assumptions or failures or alternatives or n_min):
        return ""

    if pre:
        sections.append("**Pre-conditions**")
        sections.extend(_bullet(pre))
        sections.append("")

    if assumptions:
        sections.append("**Identifying assumptions**")
        sections.extend(_bullet(assumptions))
        sections.append("")

    if failures:
        sections.append("**Failure modes → recovery**")
        sections.append("")
        sections.extend(_render_failure_table(failures))
        sections.append("")

    if alternatives:
        sections.append("**Alternatives (ranked)**")
        sections.extend(f"- `sp.{alt}`" for alt in alternatives)
        sections.append("")

    if n_min is not None:
        sections.append(f"**Typical minimum N**: {n_min}")
        sections.append("")

    # Always end with a trailing newline so multiple blocks concatenate
    # cleanly without touching each other.
    while sections and sections[-1] == "":
        sections.pop()
    sections.append("")
    return "\n".join(sections)


def render_agent_blocks(
    names: Optional[Iterable[str]] = None,
    *,
    category: Optional[str] = None,
) -> str:
    """Render the combined ``## For Agents`` section for several functions.

    The output starts with a single ``## For Agents`` header and then
    renders a ``### <name>`` sub-heading per function (with no repeated
    top-level headers). Useful for a family guide that covers multiple
    estimators.

    Parameters
    ----------
    names : iterable of str, optional
        Explicit list of function names. If ``None`` and ``category``
        is set, pulls every function in that category that has
        agent-native metadata. If both are ``None``, returns ``""``.
    category : str, optional
        Restrict to one registry category.

    Returns
    -------
    str
        Markdown; empty if nothing to render.
    """
    from .registry import agent_cards as _cards

    cards = _cards(category=category)
    if names is not None:
        keep = set(names)
        cards = [c for c in cards if c["name"] in keep]

    if not cards:
        return ""

    out: List[str] = [_HEADER, ""]
    for card in cards:
        out.append(f"### `sp.{card['name']}`")
        out.append("")
        body = render_agent_block(card["name"], header=False)
        out.append(body.rstrip())
        out.append("")
    return "\n".join(out).rstrip() + "\n"


__all__ = ["render_agent_block", "render_agent_blocks"]
