"""Auto-spec type-resolution integrity.

The auto-registry derives a param's ``type`` string from its signature annotation
(falling back to the docstring type). The correctness invariant worth locking is
*not* "few ``Any`` types" — a param the author left unannotated, or annotated
``: Any``, is **honestly** ``Any`` and inventing a narrower type would mislead an
agent (telling it ``int`` when the function accepts ``str | int`` is worse than
``Any``). The invariant is that the stringifier never *loses* a concrete
annotation: if the signature says ``List[str]`` / ``Optional[int]`` / ``float``,
the spec must not collapse it to ``Any``.

A loose ceiling on the overall ``Any`` rate is included purely as a regression
backstop — if a refactor strips annotations from many functions at once, it trips.
"""

import inspect

import pytest

import statspai as sp
from statspai import registry as R


def _auto_specs():
    R._ensure_full_registry()
    return [(n, s) for n, s in R._REGISTRY.items() if getattr(s, "_auto", False)]


def _is_effectively_any(annotation) -> bool:
    """True when the annotation carries no information (empty or ``Any``)."""
    import typing

    if annotation is inspect.Parameter.empty:
        return True
    if annotation is typing.Any:
        return True
    # String annotations from ``from __future__ import annotations``.
    if isinstance(annotation, str) and annotation.strip() in ("Any", "typing.Any"):
        return True
    return False


def test_concrete_annotation_never_degrades_to_any():
    """If the signature annotation is concrete, the spec type must not be ``Any``."""
    offenders = []
    for name, spec in _auto_specs():
        obj = getattr(sp, name, None)
        if obj is None:
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        annots = {p.name: p.annotation for p in sig.parameters.values()}
        for ps in spec.params:
            ann = annots.get(ps.name, inspect.Parameter.empty)
            if ps.type == "Any" and not _is_effectively_any(ann):
                offenders.append((name, ps.name, repr(ann)[:60]))
    assert not offenders, (
        "Auto-spec dropped a concrete signature annotation to 'Any' "
        f"(function, param, annotation): {offenders[:30]}. "
        "Fix _stringify_annotation so the type survives."
    )


def test_any_rate_stays_within_regression_backstop():
    """Loose backstop: a mass annotation-stripping regression should trip this."""
    total = any_count = 0
    for _name, spec in _auto_specs():
        for ps in spec.params:
            total += 1
            if ps.type == "Any":
                any_count += 1
    assert total > 0
    rate = any_count / total
    assert rate <= 0.12, (
        f"Auto-spec 'Any' rate is {rate:.1%} (>{0.12:.0%}). Either many functions "
        f"lost their annotations, or the stringifier regressed — investigate before "
        f"raising this backstop."
    )
