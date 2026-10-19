"""What a translation left out, under one key for both translators.

``sp.from_stata`` lists what its call does not carry over in
``untranslated_options`` and ``unapplied_sample``; ``sp.from_r`` in
``untranslated_arguments``. Those keys stay (callers read them), and
every successful payload of either translator also carries

* ``dropped`` -- one list of everything the call leaves out: option or
  argument names, and the ``if`` / ``in`` qualifier as written;
* ``complete`` -- ``True`` only when ``dropped`` is empty, i.e. the call
  fits the model the source command fits.

``ok`` keeps meaning "a call was produced"; ``complete`` says whether it
is the whole command. ``sp.stata`` refuses to run an incomplete
translation.
"""

from __future__ import annotations

from typing import Any, Dict, List

#: Keys, in order, whose entries are pieces of the source command that the
#: emitted call does not carry.
DROPPED_SOURCES = ("untranslated_options", "untranslated_arguments")


def mark_completeness(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add ``dropped`` and ``complete`` to a successful payload (in place)."""
    if not payload.get("ok"):
        return payload
    dropped: List[str] = []
    for key in DROPPED_SOURCES:
        for item in payload.get(key) or []:
            if item not in dropped:
                dropped.append(str(item))
    sample = payload.get("unapplied_sample")
    if sample:
        dropped.append(str(sample))
    payload["dropped"] = dropped
    payload["complete"] = not dropped
    return payload


__all__ = ["mark_completeness", "DROPPED_SOURCES"]
