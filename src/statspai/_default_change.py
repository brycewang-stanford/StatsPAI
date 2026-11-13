"""Announce a default that a later release will change.

The rule (owner decisions of 2026-10-20 and 2026-11-13, CLAUDE.md
section 3.8): a default that should change is announced first. The
function keeps returning today's numbers, and a call that *relies on the
default* gets a ``FutureWarning`` naming the option. Passing the option
explicitly, with either value, is silent. The switch itself happens no
earlier than the next minor release and is listed in ``MIGRATION.md``.

Functions spell "not given" as ``None`` in the signature and resolve it
here, so the test for "the caller relied on the default" is one line.
"""

from __future__ import annotations

import os
import sys
import warnings
from types import FrameType
from typing import Optional

_STATSPAI_DIR = os.path.dirname(os.path.abspath(__file__))


def external_stacklevel() -> int:
    """``stacklevel`` that attributes a ``warnings.warn`` issued by this
    function's caller to the first frame outside ``statspai``.

    A fixed level lands on a decorator wrapper of the package, and
    Python's default filter then shows the warning once per package line,
    that is once per session, whatever the user's call site.
    """
    level = 2
    frame: Optional[FrameType] = sys._getframe(2)
    prefix = _STATSPAI_DIR + os.sep
    while frame is not None:
        filename = frame.f_code.co_filename
        if "importlib" in filename and "_bootstrap" in filename:
            # ``warnings`` skips these frames without counting a level.
            frame = frame.f_back
            continue
        if not os.path.abspath(filename).startswith(prefix):
            break
        level += 1
        frame = frame.f_back
    return level


def announce_default_change(
    function: str,
    option: str,
    *,
    today: str,
    later: str,
    why: str,
) -> None:
    """Emit the ``FutureWarning`` for a call that left ``option`` unset.

    Parameters
    ----------
    function : str
        Public name without the ``sp.`` prefix.
    option : str
        The keyword whose default will change.
    today, later : str
        ``repr``-style values of the current and the announced default.
    why : str
        One or two sentences: what differs and what stays the same.
    """
    warnings.warn(
        f"{function}: {option}= was not given, so this call uses today's "
        f"default {option}={today}. A later release (1.45 at the earliest) "
        f"changes the default to {option}={later}. {why} Pass "
        f"{option}={later} to adopt it now, or {option}={today} to keep "
        "these numbers; either one silences this warning. See MIGRATION.md.",
        FutureWarning,
        stacklevel=external_stacklevel(),
    )


__all__ = ["announce_default_change", "external_stacklevel"]
